import numpy as np
import tensorflow as tf

from Utilities import S, A, cdist_tf_batch, build_Rx2, loss_weight, ff_module

ONEHOTS_ELEMENTS = {
                        'H': np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                        'C': np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
                        'N': np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
                        'O': np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),                        
                        'F': np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
                        'S': np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
                        'CL': np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                        'Cl': np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                    }

NUM_KERNELS = 20
FREQUENCIES = np.pi * tf.range(1, NUM_KERNELS + 1, dtype=np.float32)[None]

class AMP(tf.keras.Model):
    def __init__(self, node_size=128, n_channels=16, cutoff=5.0, activation=tf.nn.swish, num_steps=3, order=2):
        super(AMP, self).__init__()
        self.num_steps = num_steps
        self.node_size = node_size
        self.cutoff = tf.constant(cutoff)
        self.order = order
        self.n_channels = n_channels

        self.embedding_nodes = tf.keras.layers.Dense(units=self.node_size, activation=None, use_bias=False)
        self.embedding_edges = tf.keras.layers.Dense(units=self.node_size // 4, activation=None, use_bias=False)
        self.in_message_layers = [ff_module(self.node_size, 2) for _ in range(self.num_steps)]
        self.in_update_layers = [ff_module(self.node_size, 2) for _ in range(self.num_steps)]
        self.eq_message_layers = [ff_module(self.node_size, 2, output_size=(self.order + 1) * self.n_channels, final_activation=None) for _ in range(self.num_steps)]
        self.edge_layers = [ff_module(self.node_size, 2, output_size=self.n_channels * 18, final_activation=None) for _ in range(self.num_steps)]
        self.V = ff_module(self.node_size, 2, output_size=1, final_activation=None)

    #@tf.function(experimental_relax_shapes=True)
    def __call__(self, coords, nodes, senders, receivers, n_segments):
        return self.V(self._pass_messages(coords, nodes, senders, receivers, n_segments))  
    
    def direct_call(self, coords_batch, senders_batch, receivers_batch, nodes_batch, mol_ids_batch, n_segments):
        with tf.GradientTape() as gradient_tape:
            gradient_tape.watch(coords_batch)
            potentials_predicted = self(coords_batch, nodes_batch, senders_batch, receivers_batch, n_segments)  
            potentials_predicted = tf.math.segment_sum(potentials_predicted, mol_ids_batch)
        gradients_predicted = gradient_tape.gradient(potentials_predicted, coords_batch)
        return potentials_predicted, gradients_predicted   
    
    @tf.function(experimental_relax_shapes=True)    
    def _embed(self, coords, nodes, senders, receivers):
        edges, envelope, Rx1, Rx2 = get_geometric_features(coords, senders, receivers, self.cutoff)
        features_edge = tf.concat((edges, tf.gather(nodes, receivers), tf.gather(nodes, senders)), axis=-1)
        edges = self.embedding_edges(features_edge)
        nodes = self.embedding_nodes(nodes)
        edge_features = tf.identity(edges)
        return edges, edge_features, envelope, nodes, Rx1, Rx2

    #@tf.function(experimental_relax_shapes=True) ###
    def _pass_messages(self, coords, nodes, senders, receivers, n_segments):
        edges, edge_features, envelope, nodes, Rx1, Rx2 = self._embed(coords, nodes, senders, receivers)
        for eq_message_layer, in_message_layer, in_update_layer, edge_layer in\
            zip(self.eq_message_layers, self.in_message_layers, self.in_update_layers, self.edge_layers):
            features_ij = tf.concat((tf.gather(nodes, receivers), tf.gather(nodes, senders)), axis=-1)
            edge_features = tf.concat((features_ij, edge_features), axis=-1)
            coefficients = eq_message_layer(edge_features) 
            aniso_feature, multi_feature = build_aniso_features(coefficients, Rx1, Rx2, senders, receivers, n_segments)
            edge_features = tf.concat((edge_layer(edge_features) * aniso_feature, edges), axis=-1)            
            mp_feature_aniso = tf.concat((features_ij, edge_features), axis=-1)
            messages = in_message_layer(mp_feature_aniso) * envelope
            messages = tf.math.unsorted_segment_sum(messages, receivers, num_segments=n_segments)
            nodes += in_update_layer(tf.concat((nodes, messages, multi_feature), axis=-1))
        return tf.concat((nodes, multi_feature), axis=-1)#, (monos, dipos, quads)

#@tf.function(experimental_relax_shapes=True) ###
def build_aniso_features(coefficients, Rx1, Rx2, senders, receivers, n_segments):
    monos, dipos, quads = build_poles(coefficients, Rx1, Rx2, receivers, n_segments)     
    multi_feature = build_multi_feature(monos, dipos, quads)
    aniso_feature = G_matrices(monos, dipos, quads, Rx1, Rx2, senders, receivers)
    aniso_feature = tf.reshape(aniso_feature, [aniso_feature.shape[0], aniso_feature.shape[1] * 18])
    multi_feature = tf.reshape(multi_feature, [multi_feature.shape[0], multi_feature.shape[1] * 3])
    return aniso_feature, multi_feature
    
@tf.function(experimental_relax_shapes=True)
def build_multi_feature(monos, dipos, quads):  
    d_norm = tf.linalg.norm(dipos, axis=-1, keepdims=True)
    q_norm = A(tf.linalg.norm(quads, axis=[-1, -2]))
    return tf.concat((monos, d_norm, q_norm), axis=-1)

@tf.function(experimental_relax_shapes=True)
def envelope(R1):
    p = 5 + 1
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    env_val = 1.0 / R1 + a * R1 ** (p - 1) + b * R1 ** p + c * R1 ** (p + 1)
    return tf.where(R1 < 1, env_val, 0)

@tf.function(experimental_relax_shapes=True)
def build_sin_kernel(R1, cutoff):
    d_scaled = R1 * (1 / cutoff)
    d_cutoff = envelope(d_scaled)
    return d_cutoff * tf.sin(FREQUENCIES * d_scaled), d_cutoff

@tf.function(experimental_relax_shapes=True)
def get_geometric_features(coordinates, senders, receivers, cutoff):
    coords_a = tf.gather(coordinates, senders) 
    coords_b = tf.gather(coordinates, receivers)
    Rx1 = coords_b - coords_a
    R1 = tf.linalg.norm(Rx1, axis=-1, keepdims=True)
    Rx1 /= R1
    Rx2 = build_Rx2(Rx1)
    edge_weights, envelope = build_sin_kernel(R1, cutoff=cutoff)
    envelope /= cutoff
    return edge_weights, envelope, A(Rx1, 1), A(Rx2, 1)

@tf.function(experimental_relax_shapes=True)
def build_poles(coefficients, Rx1, Rx2, receivers, n_segments):
    mono_coeff, dipo_coeff, quad_coeff = tf.split(coefficients, 3, axis=-1)
    monos = tf.math.unsorted_segment_sum(A(mono_coeff), receivers, num_segments=n_segments)
    dipos = tf.math.unsorted_segment_sum(A(dipo_coeff) * Rx1, receivers, num_segments=n_segments)
    quads = tf.math.unsorted_segment_sum(A(A(quad_coeff)) * Rx2, receivers, num_segments=n_segments)
    return monos, dipos, quads

@tf.function(experimental_relax_shapes=True)
def G_matrices(monos, dipos, quads, Rx1, Rx2, senders, receivers):
    monos_1, monos_2 = tf.gather(monos, senders), tf.gather(monos, receivers)
    dipos_1, dipos_2 = tf.gather(dipos, senders), tf.gather(dipos, receivers)
    quads_1, quads_2 = tf.gather(quads, senders), tf.gather(quads, receivers)
    D1_Rx1, D2_Rx1 = S(dipos_1, Rx1), S(dipos_2, Rx1)
    dipo_mono = D1_Rx1 * monos_2 
    mono_dipo = -D2_Rx1 * monos_1
    dipo_dipo = S(dipos_1, dipos_2) 
    dipo_R = -D1_Rx1 * D2_Rx1
    Q1_Rx1, Q2_Rx1 = tf.einsum('bmjk, buk -> bmj', quads_1, Rx1), tf.einsum('bmjk, buk -> bmj', quads_2, Rx1)
    Q1_Rx2, Q2_Rx2 = A(tf.einsum('bmjk, bujk -> bm', quads_1, Rx2)),  A(tf.einsum('bmjk, bujk -> bm', quads_2, Rx2))
    quad_dipo = S(Q1_Rx1, dipos_2)
    dipo_quad = S(Q2_Rx1, dipos_1)
    quad_mono = Q1_Rx2 * monos_2
    mono_quad = Q2_Rx2 * monos_1
    quad_quad = A(tf.einsum('bmjk, bmjk -> bm', quads_1, quads_2)) 
    quad_R = S(Q1_Rx1, Q2_Rx1)
    quad_dipo_Rx2 = -Q1_Rx2 * D2_Rx1
    dipo_quad_Rx2 = Q2_Rx2 * D1_Rx1  
    return tf.concat((monos_1, monos_2, 
                      dipo_mono, mono_dipo, dipo_dipo, D1_Rx1, D2_Rx1, dipo_R, #remove mono-x terms?
                      Q1_Rx2, Q2_Rx2, quad_dipo, dipo_quad, quad_mono, mono_quad, quad_quad, quad_R,quad_dipo_Rx2, dipo_quad_Rx2), axis=-1)