from Utilities import S, A, cdist_tf_batch, build_Rx2, envelope, build_sin_kernel

import numpy as np
import tensorflow as tf

class Graph:
    def __init__(self, nodes=None, edges=None, n_node=None, senders=None, receivers=None,
                 batch_indices=None, Rx1=None, Rx2=None, monos=None, dipos=None, quads=None):
        self.nodes = nodes
        self.n_node = n_node
        self.edges = edges
        self.senders = senders
        self.receivers = receivers
        self.batch_indices = batch_indices
        self.Rx1 = Rx1
        self.Rx2 = Rx2
        self.monos = monos
        self.dipos = dipos
        self.quads = quads
        
        
ONEHOTS_ELEMENTS = {
                        'H': tf.convert_to_tensor(np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
                        'C': tf.convert_to_tensor(np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float32)),
                        'N': tf.convert_to_tensor(np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float32)),
                        'O': tf.convert_to_tensor(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)),                   
                        'F': tf.convert_to_tensor(np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32)),
                        'S': tf.convert_to_tensor(np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32)),
                        'CL': tf.convert_to_tensor(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)),
                        'Cl': tf.convert_to_tensor(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)),
                    }

EYE = tf.eye(3, dtype=np.float32)[None]


@tf.function(experimental_relax_shapes=True)
def neutralize(monos, n_node):
    indices = tf.repeat(tf.range(len(n_node)), n_node)
    corrections = tf.math.segment_mean(monos, indices)    
    return monos - A(tf.repeat(corrections, n_node))

@tf.function(experimental_relax_shapes=True)
def G_matrices(monos, dipos, quads, Rx1, Rx2, senders, receivers):
    monos_1, monos_2 = tf.gather(monos, senders), tf.gather(monos, receivers)
    dipos_1, dipos_2 = tf.gather(dipos, senders), tf.gather(dipos, receivers)
    quads_1, quads_2 = tf.gather(quads, senders), tf.gather(quads, receivers)
    D1_Rx1, D2_Rx1 = S(dipos_1, Rx1), S(dipos_2, Rx1)
    dipo_mono = D1_Rx1 * monos_2 
    mono_dipo = -D2_Rx1 * monos_1
    dipo_dipo = S(dipos_1, dipos_2) 
    Q1_Rx1, Q2_Rx1 = tf.einsum('ijk, ik -> ij', quads_1, Rx1), tf.einsum('ijk, ik -> ij', quads_2, Rx1)
    Q1_Rx2, Q2_Rx2 = A(tf.einsum('ijk, ijk -> i', quads_1, Rx2)),  A(tf.einsum('ijk, ijk -> i', quads_2, Rx2))
    quad_dipo = S(Q1_Rx1, dipos_2)
    dipo_quad = S(Q2_Rx1, dipos_1)
    quad_mono = Q1_Rx2 * monos_2
    mono_quad = Q2_Rx2 * monos_1
    quad_quad = A(tf.einsum('ijk, ijk -> i', quads_1, quads_2)) 
    quad_R = S(Q1_Rx1, Q2_Rx1)
    quad_dipo_Rx2 = -Q1_Rx2 * D2_Rx1
    dipo_quad_Rx2 = Q2_Rx2 * D1_Rx1  
    return tf.concat((monos_1, monos_2, monos_1 * monos_2,
                      dipo_mono, mono_dipo, dipo_dipo, D1_Rx1, D2_Rx1, D1_Rx1 * D2_Rx1,
                      Q1_Rx2, Q2_Rx2, quad_dipo, dipo_quad, quad_mono, mono_quad, quad_quad, quad_R,
                      quad_dipo_Rx2, dipo_quad_Rx2, Q1_Rx2 * Q2_Rx2), axis=-1)

@tf.function(experimental_relax_shapes=True)
def build_features(features_ij, edges, aniso_feature):
    edge_features = tf.reshape(A(edges) * A(aniso_feature, 1), [tf.shape(edges)[0], -1]) 
    mp_feature_aniso = tf.concat((features_ij, edge_features), axis=-1)
    return mp_feature_aniso, edge_features

@tf.function(experimental_relax_shapes=True)
def build_poles(Rx1, Rx2, receivers, coefficients, n_segments):   
    mono_coeff, dipo_coeff, quad_coeff = tf.split(coefficients * 1e-3, 3, axis=-1)
    monos = tf.math.unsorted_segment_sum(mono_coeff, receivers, num_segments=n_segments)
    dipos = tf.math.unsorted_segment_sum(dipo_coeff * Rx1, receivers, num_segments=n_segments)
    quads = tf.math.unsorted_segment_sum(A(quad_coeff) * Rx2, receivers, num_segments=n_segments)
    return monos, dipos, quads

def reshape_coeffs(graph, shape): 
    graph.monos = tf.stack(tf.split(graph.monos, shape[0]))
    graph.dipos = tf.stack(tf.split(graph.dipos, shape[0]))
    graph.quads = tf.stack(tf.split(graph.quads, shape[0]))
    graph.ratios = tf.stack(tf.split(graph.ratios, shape[0]))  
    return graph

def build_graph(coordinates, elements, cutoff):
    n_molecules, mol_size = coordinates.shape[:2]    
    node_features = tf.stack([ONEHOTS_ELEMENTS[e] for e in elements])
    node_features = tf.tile(node_features, [n_molecules, 1])  
    dmat = cdist_tf_batch(coordinates, coordinates)
    mol_id, senders, receivers, Rx1, Rx2, edge_weights, dmat = prepare_coords_graph(dmat, coordinates, cutoff=cutoff)
    shift = mol_size * mol_id
    senders, receivers = senders + shift, receivers + shift
    #n_node = tf.cast(tf.fill(1, node_features.shape[0]), dtype=tf.int32)
    n_node = tf.cast(tf.fill(n_molecules, mol_size), dtype=tf.int32)
    return Graph(nodes=node_features, edges=edge_weights, senders=senders, receivers=receivers, n_node=n_node, Rx1=Rx1, Rx2=Rx2, batch_indices=mol_id), dmat

#@tf.function(experimental_relax_shapes=True)
def prepare_coords_graph(dmat, coordinates, cutoff):
    indices = tf.where(tf.math.logical_and(dmat > 0, dmat < cutoff))  
    mol_id, senders, receivers = tf.unstack(indices, 3, axis=-1)
    coords_a = tf.gather_nd(coordinates, tf.stack((mol_id, senders), axis=-1)) 
    coords_b = tf.gather_nd(coordinates, tf.stack((mol_id, receivers), axis=-1))
    #R1 = A(tf.gather_nd(dmat, indices))
    Rx1 = coords_b - coords_a
    R1 = tf.linalg.norm(Rx1, axis=-1, keepdims=True)
    Rx1 /= R1
    Rx2 = build_Rx2(Rx1)
    edge_weights = build_sin_kernel(R1, cutoff)
    return mol_id, senders, receivers, Rx1, Rx2, edge_weights, dmat    
