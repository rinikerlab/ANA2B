from Utilities import cdist_tf_batch, A
from Constants import KC

import numpy as np
import tensorflow as tf

@tf.function(experimental_relax_shapes=True)
def damping_thole(dmats, alpha_1, alpha_2, smear=-0.39):
    U = A(dmats / tf.math.pow(alpha_1 * tf.linalg.matrix_transpose(alpha_2), 1/6))
    exponent = smear * tf.math.pow(U, 3)
    coeff = tf.exp(exponent)
    L3 = 1.0 - coeff
    L5 = 1.0 - (1.0 - exponent) * coeff
    return L3, L5

#@tf.function(experimental_relax_shapes=True)
def prepare_Rs(coords):
    Rs = A(coords, [2]) - A(coords, [1])
    R1 = cdist_tf_batch(coords, coords)
    R2 = tf.square(R1)
    R5 = R2 * R2 * R1
    RxR = A(Rs, -1) * A(Rs, -2)
    return R1, R2, R5, RxR

#@tf.function(experimental_relax_shapes=True)
def build_thole_matrix(coords_1, coords_2, alpha_1, alpha_2, smear=-0.39):
    coords = tf.concat((coords_1, coords_2), axis=1)
    alphas = tf.concat((alpha_1, alpha_2), axis=1)
    R1, R2, R5, RxR = prepare_Rs(coords)
    batch_size, n_atoms = R1.shape[:2]
    L3, L5 = damping_thole(R1, alphas, alphas, smear=smear)
    R2_term = L3[..., tf.newaxis] * R2[..., tf.newaxis, tf.newaxis] * tf.eye(3, dtype=tf.float32)[tf.newaxis, tf.newaxis]
    RxR_term = -3 * L5[..., tf.newaxis] * RxR
    T_thole = tf.math.divide_no_nan(R2_term + RxR_term, R5[..., tf.newaxis, tf.newaxis])
    T_thole = tf.transpose(T_thole, [0, 1, 3, 2, 4])
    T_thole = tf.reshape(T_thole, (batch_size, 3 * n_atoms, 3 * n_atoms))
    T_thole += tf.linalg.diag(tf.repeat(1 / alphas[..., 0], 3, axis=1))
    return T_thole

#@tf.function(experimental_relax_shapes=True)
def ind_dimer(coords_1, coords_2, alpha_1, alpha_2, F1, F2, smear=-0.39):    
    B = tf.linalg.inv(build_thole_matrix(coords_1, coords_2, alpha_1, alpha_2, smear=smear))
    batch_size, n3 = B.shape[:2]
    F = tf.reshape(tf.concat((F1, F2), axis=1), B.shape[:2])
    mu_induced = tf.einsum('bij, bj->bi', B, F)
    in_term = -0.5 * tf.reduce_sum(mu_induced * F, axis=-1) * KC
    mu_induced = tf.reshape(mu_induced, (batch_size, n3 // 3, 3))
    mu_ind_1, mu_ind_2 = mu_induced[:, :coords_1.shape[1]], mu_induced[:, coords_1.shape[1]:]
    return in_term, mu_ind_1, mu_ind_2

@tf.function(experimental_relax_shapes=True)
def ind_dimer0(alpha_1, alpha_2, F1, F2):    
    mu_ind_1, mu_ind_2 = alpha_1 * F1, alpha_2 * F2
    in_term_1 = -0.5 * tf.reduce_sum(mu_ind_1 * F1, axis=[-1, -2]) * KC
    in_term_2 = -0.5 * tf.reduce_sum(mu_ind_2 * F2, axis=[-1, -2]) * KC
    return in_term_1 + in_term_2, mu_ind_1, mu_ind_2