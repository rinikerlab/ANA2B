from Utilities import cdist_tf_batch, A, S, build_Rx2
from Constants import KC

import numpy as np
import tensorflow as tf

#@tf.function(experimental_relax_shapes=True)
def esp_dimer(coords_1, coords_2, distance_matrices, multipoles, with_field=False):    
    indices_1, indices_2, R1, R2, Rx1, Rx2 = prepare_distances(distance_matrices, coords_1, coords_2)
    B0, B1, B2, B3, B4 = B_matrices(R1, R2)
    if with_field:
        (monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2) = multipoles
        dipos_1, dipos_2 = tf.convert_to_tensor(dipos_1), tf.convert_to_tensor(dipos_2)
        multipoles = (monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(multipoles[2:4])
            V_esp = es_multipole(R1, R2, Rx1, Rx2, B0, B1, B2, B3, B4, multipoles, indices_1, indices_2)
        return V_esp, -tape.gradient(V_esp, multipoles[2]) / KC, -tape.gradient(V_esp, multipoles[3]) / KC
    return es_multipole(R1, R2, Rx1, Rx2, B0, B1, B2, B3, B4, multipoles, indices_1, indices_2)

def esp_monomer(coordinates, R1, multipoles, indices):
    monos, dipos, quads = multipoles[:3]
    nb_indices_1, nb_indices_2 = indices
    monos_1, monos_2 = tf.gather_nd(monos, nb_indices_1), tf.gather_nd(monos, nb_indices_2)
    dipos_1, dipos_2 = tf.gather_nd(dipos, nb_indices_1), tf.gather_nd(dipos, nb_indices_2)
    quads_1, quads_2 = tf.gather_nd(quads, nb_indices_1), tf.gather_nd(quads, nb_indices_2)
    R2 = tf.square(R1)
    Rx1 = tf.gather_nd(coordinates, nb_indices_2) - tf.gather_nd(coordinates, nb_indices_1)
    Rx2 = build_Rx2(Rx1)
    B0, B1, B2, B3, B4 = B_matrices(R1, R2)
    G0, G1, G2, G3, G4 = G_matrices(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)
    es_terms = (G0 * B0 + G1 * B1 + G2 * B2 + G3 * B3 + G4 * B4)
    return tf.reduce_sum(es_terms, axis=1) * KC

def esp_monomer_damped(coordinates, multipoles):    
    batch_size, n_atoms = coordinates.shape[:2]
    indices_1, indices_2 = np.triu_indices(n_atoms, k=1)
    batch_indices = np.tile(np.arange(batch_size)[:, None], [1, indices_1.shape[0]])
    indices_1 = np.stack((batch_indices, np.tile(indices_1[None], [batch_size, 1])), axis=-1)
    indices_2 = np.stack((batch_indices, np.tile(indices_2[None], [batch_size, 1])), axis=-1)
    indices_nb = np.concatenate((indices_1, indices_2[:, :, 1:]), axis=-1)
    monos, dipos, quads = multipoles[:3]
    nb_indices_1, nb_indices_2 = indices_1, indices_2
    monos_1, monos_2 = tf.gather_nd(monos, nb_indices_1), tf.gather_nd(monos, nb_indices_2)
    dipos_1, dipos_2 = tf.gather_nd(dipos, nb_indices_1), tf.gather_nd(dipos, nb_indices_2)
    quads_1, quads_2 = tf.gather_nd(quads, nb_indices_1), tf.gather_nd(quads, nb_indices_2)    
    Rx1 = tf.gather_nd(coordinates, nb_indices_2) - tf.gather_nd(coordinates, nb_indices_1)
    Rx2 = build_Rx2(Rx1)
    R1 = tf.linalg.norm(Rx1, axis=-1, keepdims=True)
    R2 = tf.square(R1)
    B0, B1, B2, B3, B4 = B_matrices(R1, R2)
    G0, G1, G2, G3, G4 = G_matrices(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)
    es_terms = (G0 * B0 + G1 * B1 + G2 * B2 + G3 * B3 + G4 * B4)
    damping_terms = (1 - tf.math.exp(-0.25 * R2)) ** 2 
    return tf.reduce_sum(es_terms * damping_terms, axis=1) * KC

@tf.function(experimental_relax_shapes=True)
def G_matrices(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2):
    D1_Rx1, D2_Rx1 = S(dipos_1, Rx1), S(dipos_2, Rx1)
    Q1_Rx1, Q2_Rx1 = tf.einsum('bijk, bik -> bij', quads_1, Rx1), tf.einsum('bijk, bik -> bij', quads_2, Rx1)
    Q1_Rx2, Q2_Rx2 = A(tf.einsum('bijk, bijk -> bi', quads_1, Rx2)),  A(tf.einsum('bijk, bijk -> bi', quads_2, Rx2))
    dipo_mono = D1_Rx1 * monos_2 # G1
    mono_dipo = D2_Rx1 * monos_1
    dipo_dipo = S(dipos_1, dipos_2) # G1
    dipo_R = D1_Rx1 * D2_Rx1
    G0 = monos_1 * monos_2
    G1 = dipo_mono - mono_dipo + dipo_dipo
    G2 = -dipo_R + 2 * S(Q1_Rx1, dipos_2) - 2 * S(Q2_Rx1, dipos_1)\
         + Q1_Rx2 * monos_2 + Q2_Rx2 * monos_1\
         + 2 * A(tf.einsum('bijk, bijk -> bi', quads_1, quads_2)) # bnxy, bnxy -> bn
    G3 = -4 * S(Q1_Rx1, Q2_Rx1) - Q1_Rx2 * D2_Rx1 + Q2_Rx2 * D1_Rx1
    G4 = Q1_Rx2 * Q2_Rx2
    return G0, G1, G2, G3, G4

@tf.function(experimental_relax_shapes=True)
def B_matrices(R1, R2=None):
    if R2 is None:
        R2 = tf.square(R1)
    B0 = 1 / R1
    B1 = B0 / R2
    B2 = 3 * B1 / R2
    B3 = 5 * B2 / R2
    B4 = 7 * B3 / R2
    return B0, B1, B2, B3, B4

@tf.function(experimental_relax_shapes=True)
def es_multipole(R1, R2, Rx1, Rx2, B0, B1, B2, B3, B4, multipoles, indices_1, indices_2):
    monopoles_1, monopoles_2, dipoles_1, dipoles_2, quadrupoles_1, quadrupoles_2 = multipoles[:6]
    monos_1, monos_2 = tf.gather(monopoles_1, indices_1, axis=1), tf.gather(monopoles_2, indices_2, axis=1)
    dipos_1, dipos_2 = tf.gather(dipoles_1, indices_1, axis=1), tf.gather(dipoles_2, indices_2, axis=1)
    quads_1, quads_2 = tf.gather(quadrupoles_1, indices_1, axis=1), tf.gather(quadrupoles_2, indices_2, axis=1)
    G0, G1, G2, G3, G4 = G_matrices(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)
    return tf.math.reduce_sum(G0 * B0 + G1 * B1 + G2 * B2 + G3 * B3 + G4 * B4, axis=[-1, -2]) * KC

#@tf.function(experimental_relax_shapes=True)
def prepare_distances(distance_matrices, coords_1, coords_2):
    n_nodes_1, n_nodes_2 = coords_1.shape[1], coords_2.shape[1]#distance_matrices.shape[1:3]
    interaction_indices = np.indices((n_nodes_1, n_nodes_2)).reshape((2, -1)).T
    indices_1, indices_2 = tf.unstack(interaction_indices, axis=-1)
    R1 = tf.reshape(distance_matrices, [distance_matrices.shape[0], -1, 1])
    R2 = tf.square(R1)
    Rx1 = tf.gather(coords_2, indices_2, batch_dims=0, axis=1) - tf.gather(coords_1, indices_1, batch_dims=0, axis=1)
    Rx2 = build_Rx2(Rx1)
    return indices_1, indices_2, R1, R2, Rx1, Rx2