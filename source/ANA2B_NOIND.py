import graph_nets as gn
import numpy as np
import tensorflow as tf

from Utilities import S, A, build_Rx2, ff_module, switch, gaussian
from TopoGNN import TopoGNN

class ANA2B(tf.keras.Model):
    def __init__(self, cutoff=10.0, n_units=128, node_size=64, n_layers=2, n_steps=1, activation=tf.nn.swish, n_gaussians=5, debug=False):
        super(ANA2B, self).__init__()   
        self.cutoff = cutoff
        self.n_units = n_units
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.node_size = node_size        
        self.activation = activation
        self.n_gaussians = n_gaussians
        self.debug = debug
        
        self.S_static = ff_module(self.n_units, self.n_layers, activation=self.activation, output_size=2, final_activation=tf.nn.softplus)
        self.K = ff_module(self.n_units, self.n_layers, activation=self.activation, output_size=3, final_activation=tf.nn.softplus)
        self.feature_embedding = ff_module(self.n_units, self.n_layers, activation=self.activation)
        self.TopoGNN = TopoGNN(node_size=self.node_size, n_layers=1, n_steps=self.n_steps)

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, graph_1, graph_2, coords_1, coords_2, distance_matrices, multipoles, batch_size):          
        S_features, K_features, R1, R2, switch_function, indices_b =\
                     self._build_features(coords_1, coords_2, distance_matrices, graph_1, graph_2, multipoles)
        V_ex, V_at = self._calculate_energy_terms(S_features, K_features, R1, R2, switch_function)        
        return tf.math.unsorted_segment_sum(V_ex - V_at, indices_b, num_segments=batch_size)[:, 0]

    @tf.function(experimental_relax_shapes=True)
    def _calculate_energy_terms(self, S_features, K_features, R1, R2, switch_function):
        S2_static, S2_at = tf.split(self.S_static(S_features) * switch_function, 2, axis=-1)
        K1_static, K2_static, K_at = tf.split(self.K(K_features), 3, axis=-1)
        V_ex_static = (K1_static * S2_static) / R1 + (K2_static * S2_static) / R2
        V_ex = V_ex_static
        V_at = K_at * S2_at # tf.convert_to_tensor(coords_1.shape[0:1])
        return V_ex, V_at
    
    @tf.function(experimental_relax_shapes=True)
    def _build_features(self, coords_1, coords_2, distance_matrices, graph_1, graph_2, multipoles):
        graph_1, graph_2 = self.TopoGNN(graph_1), self.TopoGNN(graph_2)
        indices_b, indices_1, indices_2, R1, R2, Rx1, Rx2 = prepare_distances(coords_1, coords_2, distance_matrices, cutoff=self.cutoff)
        node_features_1, node_features_2 = tf.gather(graph_1.nodes, indices_1[:, 1]), tf.gather(graph_2.nodes, indices_2[:, 1])
        monopoles_1, monopoles_2, dipoles_1, dipoles_2, quadrupoles_1, quadrupoles_2 = multipoles[:6]
        monos_1, monos_2 = tf.gather_nd(monopoles_1, indices_1), tf.gather_nd(monopoles_2, indices_2)
        dipos_1, dipos_2 = tf.gather_nd(dipoles_1, indices_1), tf.gather_nd(dipoles_2, indices_2)
        quads_1, quads_2 = tf.gather_nd(quadrupoles_1, indices_1), tf.gather_nd(quadrupoles_2, indices_2)
        feature_12 = tf.concat((node_features_1, node_features_2), axis=-1)
        feature_21 = tf.concat((node_features_2, node_features_1), axis=-1)
        node_features = self.feature_embedding(feature_12) + self.feature_embedding(feature_21)    
        switch_function = switch(R1, self.cutoff - 1.0, self.cutoff)
        distance_features = gaussian(R1, np.logspace(-1, 0, self.n_gaussians)) * switch_function
        G_features = G_matrices_sym(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)
        S_features = tf.concat((G_features, node_features, distance_features), axis=-1)
        K_features = tf.concat((node_features, distance_features), axis=-1)  
        return S_features, K_features, R1, R2, switch_function, indices_b

@tf.function(experimental_relax_shapes=True)
def G_matrices_sym(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2):
    D1_Rx1, D2_Rx1 = S(dipos_1, Rx1), S(dipos_2, Rx1)
    Q1_Rx1, Q2_Rx1 = tf.einsum('ijk, ik -> ij', quads_1, Rx1), tf.einsum('ijk, ik -> ij', quads_2, Rx1)
    Q1_Rx2, Q2_Rx2 = A(tf.einsum('ijk, ijk -> i', quads_1, Rx2)),  A(tf.einsum('ijk, ijk -> i', quads_2, Rx2))
    dipo_mono = D1_Rx1 * monos_2 
    mono_dipo = D2_Rx1 * monos_1
    dipo_dipo = S(dipos_1, dipos_2) 
    dipo_R = D1_Rx1 * D2_Rx1
    G0 = monos_1 * monos_2
    G1 = tf.concat((dipo_mono -mono_dipo, dipo_dipo), axis=-1)
    G2 = tf.concat((-dipo_R, 
                    2 * S(Q1_Rx1, dipos_2) -2 * S(Q2_Rx1, dipos_1), # sym
                    Q1_Rx2 * monos_2 + Q2_Rx2 * monos_1, # sym
                    2 * A(tf.einsum('ijk, ijk -> i', quads_1, quads_2))), axis=-1) # bnxy, bnxy -> bn
    G3 = tf.concat((-4 * S(Q1_Rx1, Q2_Rx1), 
                    -Q1_Rx2 * D2_Rx1 + Q2_Rx2 * D1_Rx1), axis=-1) # sym
    G4 = Q1_Rx2 * Q2_Rx2
    return tf.concat((G0, G1, G2, G3, G4), axis=-1)

@tf.function(experimental_relax_shapes=True)
def prepare_distances(coords_1, coords_2, distance_matrices, cutoff):
    cutoff_indices = tf.where(distance_matrices < cutoff)
    indices_b, indices_i, indices_j = tf.unstack(cutoff_indices, axis=-1)
    indices_1 = tf.stack((indices_b, indices_i), axis=-1)
    indices_2 = tf.stack((indices_b, indices_j), axis=-1)
    R1 = A(tf.gather_nd(distance_matrices, cutoff_indices))
    R2 = tf.square(R1)
    Rx1 = (tf.gather_nd(coords_2, indices_2) - tf.gather_nd(coords_1, indices_1)) / R1
    Rx2 = build_Rx2(Rx1)    
    return indices_b, indices_1, indices_2, R1, R2, Rx1, Rx2

def validate(model, db_key, REF_DATA):
    references, predictions = [], []
    for system_key in REF_DATA[db_key]:
        graph_1, graph_2 = REF_DATA[db_key][system_key]['graphs']
        coords_1, coords_2 = REF_DATA[db_key][system_key]['coordinates']
        #coords_1, coords_2 = tf.convert_to_tensor(coords_1), tf.convert_to_tensor(coords_2)
        die_term = REF_DATA[db_key][system_key]['die_term']
        distance_matrices = REF_DATA[db_key][system_key]['distance_matrix']
        multipoles = REF_DATA[db_key][system_key]['multipoles']
        V_terms = model(graph_1, graph_2, coords_1, coords_2, distance_matrices, multipoles, coords_1.shape[0])
        predictions.append(die_term + V_terms)
        references.append(REF_DATA[db_key][system_key]['ref_energy'])
    return np.hstack(references), np.hstack(predictions)


