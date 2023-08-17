from pymbd.pymbd import vdw_params, freq_grid
from Utilities import H_TO_KJ, BOHR_TO_ANGSTROM, cdist_tf_batch

import numpy as np
import tensorflow as tf

class MBD:
    def __init__(self, 
                 beta=0.8,
                 d=6.0,
                 nfreq=15,
                 dtype_tf=tf.float32, 
                 dtype_np=np.float32):
        self._d = d
        self._beta = beta
        self._dtype_tf = dtype_tf
        self._dtype_np = dtype_np
        self._freq, self._freq_w = freq_grid(nfreq)
        self._R = lambda x: tf.cast(x, self._dtype_tf)
        self._freq, self._freq_w = self._R(self._freq), self._R(self._freq_w)
        
    def initialize(self, elements):
        alpha_0, C6, R_vdw = (np.array([vdw_params[sp][param] for sp in elements], dtype=self._dtype_np)
                                                    for param in 'alpha_0(TS) C6(TS) R_vdw(TS)'.split())
        self._alpha0, self._C6_0, self._R_vdw0 = A(alpha_0, ks=[0]), A(C6, ks=[0]), A(R_vdw, ks=[0])
        
    def _volumes_to_params(self, volumes):
        volumes = self._R(volumes[..., 0])
        alpha_0 = self._alpha0 * volumes
        C6 = self._C6_0 * tf.square(volumes)
        R_vdw = self._R_vdw0 * tf.pow(volumes, 1 / 3)
        return alpha_0, C6, R_vdw
        
    #@tf.function(experimental_relax_shapes=True)
    def __call__(self, coordinates, volumes, grad=False):
        coordinates = tf.convert_to_tensor(coordinates / BOHR_TO_ANGSTROM)
        if grad:
            with tf.GradientTape() as tape:
                tape.watch(coordinates)                
                energy = self._mbd_energy(coordinates, volumes)
            gradient = tape.gradient(energy, coordinates)
            return energy * H_TO_KJ, gradient * (H_TO_KJ / BOHR_TO_ANGSTROM)
        return self._mbd_energy(coordinates, volumes) * H_TO_KJ
    
    def _mbd_energy(self, coordinates, volumes):
        alpha_0, C6, R_vdw = self._volumes_to_params(volumes)
        omega = 4 / 3 * C6 / alpha_0**2
        alpha_dyn = A(alpha_0, [-1]) / (1 + (A(self._freq, [0, 0]) / A(omega, [-1])) ** 2)
        sigma = (tf.sqrt(self._dtype_np(2. / np.pi)) * alpha_dyn / 3) ** (1 / 3)
        sigma_ij = tf.sqrt(A(sigma, [2]) ** 2 + A(sigma, [1]) ** 2)
        Rs, R1, R2, R5, RxR = prepare_geometry(coordinates)
        T_bare = T_bare_tf(Rs, R1, R2, R5, RxR, dtype=self._dtype_tf)
        T_erf_coulomb = T_erf_coulomb_tf(Rs, R1, R5, RxR, sigma_ij, T_bare, dtype_np=self._dtype_np)
        dipmats = prepare_screened_dipole_matrix_tf(R1, R_vdw, T_erf_coulomb, beta=self._beta)
        alpha_rsscs, C6_rsscs, R_vdw_rsscs, omega_rsscs, S_vdw_rsscs = screen_parameters(dipmats, alpha_dyn, alpha_0, R_vdw, 
                                                                                         self._freq_w, beta=self._beta, 
                                                                                         dtype_np=self._dtype_np)
        eigs = decompose_eigenvalues_tf(R1, S_vdw_rsscs, omega_rsscs, alpha_rsscs, T_bare, d=self._d)
        energy = tf.reduce_sum(tf.sqrt(eigs), axis=-1) / 2 - 3 * tf.reduce_sum(omega_rsscs, axis=-1) / 2
        return energy[..., tf.newaxis] 

def A(x, ks):
    for k in ks:
        x = tf.expand_dims(x, axis=k)
    return x

#@tf.function(experimental_relax_shapes=True)
def damping_fermi_tf(R, S_vdw, d):
    return 1 / (1 + tf.exp(-d * (R / S_vdw - 1)))

#@tf.function(experimental_relax_shapes=True)
def prepare_geometry(coords):
    Rs = A(coords, [2]) - A(coords, [1])    
    R1 = tf.linalg.norm(Rs, axis=-1)
    R2 = tf.square(R1)
    R5 = R2 * R2 * R1
    RxR = A(Rs, [-1]) * A(Rs, [-2])
    return Rs, R1, R2, R5, RxR

#@tf.function(experimental_relax_shapes=True)
def T_bare_tf(Rs, R1, R2, R5, RxR, dtype=tf.float64):
    DIAG = A(R2, [-1, -1]) * A(tf.eye(3, dtype=dtype), [0, 0])
    return tf.math.divide_no_nan((-3 * RxR + DIAG), A(R5, [-1, -1]))

#@tf.function(experimental_relax_shapes=True)
def T_erf_coulomb_tf(Rs, R1, R5, RxR, sigma_ij, bare, dtype_np=np.float64):
    RxR = A(Rs, [-1]) * A(Rs, [-2])
    RxR5 = tf.math.divide_no_nan(RxR,  A(R5, [-1, -1]))
    zeta = tf.math.divide_no_nan(A(R1, [-1]), sigma_ij) # tf.math.divide_no_nan(dists, sigma_ij[..., 0])
    Z2 = tf.square(zeta)
    theta = 2 * zeta / tf.sqrt(dtype_np(np.pi)) * tf.exp(-Z2)
    erf_theta = tf.math.erf(zeta) - theta
    return A(erf_theta, [-2, -2]) * A(bare, [-1]) + A(2 * Z2 * theta, [-2, -2]) * A(RxR5, [-1])

#@tf.function(experimental_relax_shapes=True)
def prepare_screened_dipole_matrix_tf(R1, R_vdw, erf_coulomb, beta, d=6.0):
    batch_size, n_atoms = erf_coulomb.shape[:2]
    S_vdw = beta * (A(R_vdw, [2]) + A(R_vdw, [1]))    
    dipmats = A(1 - damping_fermi_tf(R1, S_vdw, d=d), [-1, -1, -1]) * erf_coulomb
    dipmats = tf.transpose(dipmats, [0, 1, 3, 2, 4, 5])    
    return tf.transpose(tf.reshape(dipmats, (batch_size, 3 * n_atoms, 3 * n_atoms, -1)), [0, 3, 1, 2])

#@tf.function(experimental_relax_shapes=True)
def screen_parameters(dipmats, alpha_dyn, alpha_0, R_vdw, freq_w, beta, dtype_np=np.float64):   
    diag_part = tf.repeat(1 / tf.transpose(alpha_dyn, [0, 2, 1]), 3, axis=-1)
    a_nlc = tf.linalg.inv(dipmats + tf.linalg.diag(diag_part))
    alpha_dyn_rsscs = tf.reduce_sum([tf.reduce_sum(a_nlc[:, :, i::3, i::3], -1) for i in range(3)], axis=0) / 3
    alpha_rsscs = alpha_dyn_rsscs[:, 0]
    C6_rsscs = dtype_np(3 / np.pi) * tf.reduce_sum(A(freq_w, [0, -1]) * tf.square(alpha_dyn_rsscs), 1)
    R_vdw_rsscs = R_vdw * (alpha_rsscs / alpha_0) ** (1 / 3)
    omega_rsscs = 4 / 3 * C6_rsscs / alpha_rsscs ** 2
    S_vdw_rsscs = beta * (R_vdw_rsscs[:, :, tf.newaxis] + R_vdw_rsscs[:, tf.newaxis])
    return alpha_rsscs, C6_rsscs, R_vdw_rsscs, omega_rsscs, S_vdw_rsscs

#@tf.function(experimental_relax_shapes=True)
def decompose_eigenvalues_tf(R1, S_vdw_rsscs, omega_rsscs, alpha_rsscs, bare, d=6.0):
    pre = tf.repeat(tf.sqrt(alpha_rsscs) * omega_rsscs, 3, axis=-1)
    batch_size, n_atoms = bare.shape[:2]    
    dipmats_rsscs = A(damping_fermi_tf(R1, S_vdw_rsscs, d=d), [-1, -1]) * bare
    dipmats_rsscs = tf.reshape(tf.transpose(dipmats_rsscs, [0, 1, 3, 2, 4]), (batch_size, 3 * n_atoms, 3 * n_atoms))    
    dipmats_rsscs = tf.linalg.diag(tf.repeat(tf.square(omega_rsscs), 3, axis=-1)) + A(pre, [-1]) * A(pre, [-2]) * dipmats_rsscs
    return tf.linalg.eigvalsh(dipmats_rsscs)