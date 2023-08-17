import numpy as np
import tensorflow as tf

from Utilities import S, build_Rx2, to_Z
from Electrostatics import G_matrices, B_matrices
from Constants import KC, H_TO_KJ, BOHR_TO_ANGSTROM, EV_TO_KJ

SQRT_PI = np.sqrt(np.pi)
from itertools import product


class Ewald:
    def __init__(self, 
                 cutoff=10.0, 
                 cutoff_pol=10.0,
                 alpha=None, 
                 smear=-0.39, 
                 tolerance=1e-4, 
                 tolerance_pol=5e-3,
                 knorm=30, 
                 dtype=np.float32,
                 dtype_tf=tf.float32, 
                 dtype_tf_c=tf.complex64):
        self._dtype = dtype
        self._dtype_tf = dtype_tf
        self._dtype_tf_c = dtype_tf_c
        self._tolerance = tolerance
        self._tolerance_pol = tolerance_pol
        self._cutoff = cutoff    
        self._cutoff_pol = cutoff_pol
        self._pi = self._dtype(np.pi)
        self._sqrt_pi = np.sqrt(self._pi)
        self._pi_sq = np.square(np.pi)
        self._knorm = knorm
        self._smear = smear
        self._eye =  A(tf.eye(3, dtype=self._dtype_tf), [0, 0])
        self._C = lambda x: tf.cast(x, self._dtype_tf_c)
        self._R = lambda x: tf.cast(x, self._dtype_tf)
        if alpha is None:
            self._update_parameters()    
        else:
            self.alpha = alpha
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self._alpha_sq = self._alpha ** 2
        
    @property
    def tolerance(self):
        return self._tolerance
    
    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = tolerance
        self._update_parameters()
        
    @property
    def cutoff(self):
        return self._cutoff
    
    @cutoff.setter
    def cutoff(self, cutoff):
        self._cutoff = cutoff
        self._update_parameters()        
    
    def _update_parameters(self):
        self._alpha = self._R(np.sqrt(-np.log(2 * self._tolerance)) / self._cutoff)
        self._alpha_sq = self._alpha ** 2        
        self._alpha_pol = self._R(np.sqrt(-np.log(2 * self._tolerance_pol)) / self._cutoff_pol)
        self._alpha_sq_pol = self._alpha_pol ** 2
    
    ###################
    # Reciprocal Part #
    ###################
    def reciprocal(self, coordinates, cell, monos, dipos=None, quads=None):
        kpoints = self.kpoints(cell, knorm=self._knorm)
        rho, exp = self._rho(coordinates, kpoints, monos, dipos, quads)
        Ak = self._Ak(kpoints)
        pre_rec = (2 * self._pi / tf.linalg.det(cell)) 
        V_rec = pre_rec * self._reciprocal_potential(Ak, rho)
        E_rec = -2 * pre_rec * self._reciprocal_field(Ak, rho, exp, kpoints)
        return V_rec, E_rec
    
    @tf.function(experimental_relax_shapes=True)
    def _reciprocal_field(self, Ak, rho, exp, kpoints):
        real_terms = self._R(tf.math.real(1j * exp * tf.math.conj(rho)[..., None]))
        return tf.reduce_sum(tf.expand_dims((Ak[..., None] * kpoints), axis=-2) * real_terms[..., None], axis=-3)
    
    @tf.function(experimental_relax_shapes=True)
    def _reciprocal_potential(self, Ak, rho):
        rho_sq = self._R(rho * tf.math.conj(rho))
        return tf.reduce_sum(Ak * rho_sq, axis=-1)    
    
    @tf.function(experimental_relax_shapes=True)
    def _Ak(self, kpoints):
        k2 = tf.reduce_sum(kpoints * kpoints, axis=-1)
        return tf.exp(-k2 / (4 * self._alpha_sq)) / k2
    
    @tf.function(experimental_relax_shapes=True)
    def _rho(self, coordinates, kpoints, monos, dipos=None, quads=None):
        exp = tf.exp(1j * self._C(tf.einsum('bki, bni -> bkn', kpoints, coordinates)))
        mono_term = self._C(tf.einsum('bin->bni', monos))
        dipo_term, quad_term = 0, 0
        if dipos is not None:
            dipo_term = 1j * self._C((tf.einsum('bki, bni -> bkn', kpoints, dipos)))
        if quads is not None:
            quad_term = -self._C(tf.einsum('bki, bnij, bkj -> bkn', kpoints, quads, kpoints) / 3)
        # Term Shape: KxN
        rho_i = (mono_term + dipo_term + quad_term)
        return tf.reduce_sum(rho_i * exp, axis=-1), exp
    
    def kpoints(self, cell, xmax=None, ymax=None, zmax=None, knorm=30, include_pi=True):
        xmax0, ymax0, zmax0 = np.amax(np.ceil(knorm / tf.linalg.norm(cell, axis=-1)), axis=0)
        if xmax is None:
            xmax = xmax0
        if ymax is None:
            ymax = ymax0
        if zmax is None:
            zmax = zmax0
        nx = np.arange(-xmax, xmax + 1)
        ny = np.arange(-ymax, ymax + 1)
        nz = np.arange(-zmax, zmax + 1)
        ns = np.array(np.meshgrid(nx, ny, nz), dtype=np.float64).T.reshape([-1, 3])
        ns = ns[np.where(np.sum(np.abs(ns), axis=-1) != 0)]
        if include_pi:
            ns *= 2 * self._pi
        return self._R(tf.einsum('kj, bij -> bki', ns, tf.linalg.inv(cell)))
    
    ###############
    # Direct Part #
    ###############        
    def direct_multi(self, R, Rx1, monos, dipos, quads, indices_1, indices_2, erf=True):
        G0, G1, G2, G3, G4 = self.G_matrices(Rx1, monos, dipos, quads, indices_1, indices_2)
        if erf:
            B0, B1, B2, B3, B4 = self.B_matrices_erf(R)
        else:
            B0, B1, B2, B3, B4 = B_matrices(R)
        return tf.reduce_sum(G0 * B0 + G1 * B1 + G2 * B2 + G3 * B3 + G4 * B4, axis=[-1, -2])

    def G_matrices(self, Rx1, monos, dipos, quads, indices_1, indices_2):
        n_atoms = monos.shape[1]
        indices_1, indices_2 = indices_1 % n_atoms, indices_2 % n_atoms
        monos_1, monos_2 = tf.gather(monos, indices_1, axis=1), tf.gather(monos, indices_2, axis=1)
        dipos_1, dipos_2 = tf.gather(dipos, indices_1, axis=1), tf.gather(dipos, indices_2, axis=1)
        quads_1, quads_2 = tf.gather(quads, indices_1, axis=1), tf.gather(quads, indices_2, axis=1)
        Rx2 = build_Rx2(Rx1)        
        return G_matrices(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)

    @tf.function(experimental_relax_shapes=True)
    def B_matrices_erf(self, R):
        SQRT_PI_ALPHA = self._sqrt_pi * self._alpha
        R2 = R * R
        EXP_A2R2 = tf.math.exp(-self._alpha_sq * R2)
        A_SQ_2 = 2. * self._alpha_sq
        # SMITH: http://physics.ujep.cz/~mlisal/md/es_multipole.pdf
        B0 = tf.math.erfc(self._alpha * R) / R
        B1 = (1. * B0 + ((A_SQ_2**1 / SQRT_PI_ALPHA) * EXP_A2R2)) / R2
        B2 = (3. * B1 + ((A_SQ_2**2 / SQRT_PI_ALPHA) * EXP_A2R2)) / R2
        B3 = (5. * B2 + ((A_SQ_2**3 / SQRT_PI_ALPHA) * EXP_A2R2)) / R2
        B4 = (7. * B3 + ((A_SQ_2**4 / SQRT_PI_ALPHA) * EXP_A2R2)) / R2   
        return B0, B1, B2, B3, B4

    #############
    # Self Part #
    #############    
    def correction(self, coordinates, monos, dipos=None, quads=None):
        pre_self = -self.alpha / self._sqrt_pi # fterm
        pre_self_sq = 2.0 * self._alpha_sq # term
        self_term_mono = tf.reduce_sum(monos * monos, axis=[-1])
        self_term_dipo, self_term_quad = 0, 0
        E_self = tf.zeros_like(coordinates)
        if dipos is not None:
            dipo_norm = tf.reduce_sum(dipos * dipos, axis=[-1])
            self_term_dipo = dipo_norm / 3.0
            pre_field = (4. * self.alpha ** 3.) / (3. * self._sqrt_pi)
            E_self = pre_field * dipos
        if quads is not None:
            quad_norm = tf.reduce_sum(quads * quads, axis=[-1, -2])
            self_term_quad =  (2.0 * pre_self_sq * quad_norm) / 5.0
        V_self = pre_self * tf.reduce_sum(self_term_mono + pre_self_sq * (self_term_dipo + self_term_quad), axis=[-1])
        return V_self, E_self
    
    ##############
    # Intra Part #
    ##############       
    def esp_intra(self, coords_uc, monos, dipos, quads, mol_size):
        n_molecules_uc = coords_uc.shape[1] // mol_size
        intra_indices = np.stack(np.triu_indices(mol_size, k=1), axis=-1)
        shifts = np.repeat(np.arange(n_molecules_uc), intra_indices.shape[0]) * mol_size
        intra_indices = np.tile(intra_indices, (n_molecules_uc, 1))
        intra_indices += shifts[:, None]
        intra_coords_a, intra_coords_b = tf.unstack(tf.gather(coords_uc, intra_indices, axis=1), axis=-2)
        intra_index_1, intra_index_2 = tf.unstack(intra_indices, axis=-1)
        Rx1_intra = intra_coords_b - intra_coords_a
        R1_intra = tf.linalg.norm(Rx1_intra, axis=-1, keepdims=True)
        Rx2_intra = build_Rx2(Rx1_intra)
        return self.direct_multi(R1_intra, Rx1_intra, monos, dipos, quads, intra_index_1, intra_index_2, erf=False) 
    
    def V_esp(self, coords_uc, lattice, mol_size, monos, dipos=None, quads=None, field=True):
        if dipos is None:
            dipos = tf.zeros_like(coords_uc)
        if quads is None:
            quads = tf.zeros((*coords_uc.shape, 3), dtype=dipos.dtype)
        coords_sc, lattice_sc = expand_unit_cell(coords_uc, lattice, cutoff=2.0 * self._cutoff)
        R, Rx1, index_1, index_2 = get_distances_cutoff(coords_sc, lattice_sc, self._cutoff)
        n_atoms_sc, n_atoms_uc = coords_sc.shape[1], coords_uc.shape[1]
        n_cells = (n_atoms_sc / n_atoms_uc)
        V_rec, E_rec = self.reciprocal(coords_uc, lattice, monos, dipos, quads)
        V_self, E_self = self.correction(coords_uc, monos, dipos, quads)
        if field:
            with tf.GradientTape(persistent=True) as tape_field:
                dipos = tf.convert_to_tensor(dipos)
                tape_field.watch(dipos)
                V_dir = self.direct_multi(R, Rx1, monos, dipos, quads, index_1, index_2, erf=True) / n_cells
                V_intra = self.esp_intra(coords_uc, monos, dipos, quads, mol_size)
            E_dir = -tape_field.gradient(V_dir, dipos)
            E_intra = -tape_field.gradient(V_intra, dipos)
            E_static = (E_rec + E_self + E_dir) - E_intra
            return (V_rec + V_self + V_dir, V_intra), E_static
        V_dir = self.direct_multi(R, Rx1, monos, dipos, quads, index_1, index_2, erf=True) / n_cells
        V_intra = self.esp_intra(coords_uc, monos, dipos, quads, mol_size)
        return (V_rec + V_self + V_dir), V_intra
        
    #######################
    # Polarization Matrix #
    #######################    
    #@tf.function(experimental_relax_shapes=True)
    def build_T(self, coords_uc, lattices, pol):
        Rx1, RxR, R1, R2, R5 = prepare_geometry(coords_uc[0], lattice=lattices[0], cutoff=self._cutoff_pol, dtype_tf=self._dtype_tf)
        T_dir = self.build_T_erfc(R1, R2, R5, RxR)
        T_corr = self.build_T_corr(R1, R2, R5, RxR, pol)
        diagonal = self.build_diagonal(pol)
        T_thole = tf.transpose(T_corr + T_dir, [0, 2, 1, 3])   
        return tf.reshape(T_thole, diagonal.shape) + diagonal
    
    #@tf.function(experimental_relax_shapes=True)
    def build_T_corr(self, R1, R2, R5, RxR, pol):
        L3, L5 = thole_damping(R1, pol, smear=self._smear)
        R2_term = A(L3 * R2, [-1, -1]) * self._eye
        RxR_term = -3. * A(L5, [-1, -1]) * RxR
        T_thole = tf.math.divide_no_nan(R2_term + RxR_term, A(R5, [-1, -1]))
        T_bare = self.build_T_bare(R1, R2, R5, RxR)
        return tf.reduce_sum(T_thole - T_bare, axis=-5)
    
    #@tf.function(experimental_relax_shapes=True)
    def build_T_erfc(self, R1, R2, R5, RxR):
        R3 = R1 * R2
        AR1 = self._alpha_pol * R1
        AR1_SQ = tf.square(AR1)
        erfc_AR1 = tf.math.erfc(AR1)
        R2_term = A(tf.math.divide_no_nan(erfc_AR1 + (2.0 * AR1 / self._sqrt_pi) * tf.exp(-AR1_SQ), R3), [-1, -1])
        RxR_term = A(tf.math.divide_no_nan(3. * erfc_AR1 + (2.0 * AR1 / self._sqrt_pi) * (3. + 2. * AR1_SQ) * tf.exp(-AR1_SQ), R5), [-1, -1])
        return tf.reduce_sum(-RxR_term * RxR + R2_term * A(self._eye, [0]), axis=-5)
    
    #@tf.function(experimental_relax_shapes=True)
    def build_diagonal(self, pol):
        self_corr = -(4 * self._alpha_pol ** 3) / (3 * self._sqrt_pi)
        return tf.linalg.diag(tf.repeat(1 / pol, 3) + self_corr)
    
    #@tf.function(experimental_relax_shapes=True)
    def build_T_bare(self, R1, R2, R5, RxR):
        diagonal = A(R2, [-1, -1]) * self._eye
        return tf.math.divide_no_nan((-3 * RxR + diagonal), A(R5, [-1, -1]))    
    
    #@tf.function(experimental_relax_shapes=True)
    def V_pol(self, E_static, pol, coords_uc, lattice):   
        B = tf.linalg.inv(self.build_T(coords_uc, lattice, pol))
        mu_ind = tf.reshape(tf.einsum('ij, j->i', B, tf.reshape(E_static, [-1])), [1, -1, 3])      
        V_ind = -0.5 * KC * tf.reduce_sum(mu_ind * E_static, [-1, -2])
        return V_ind, mu_ind
            
    #@tf.function(experimental_relax_shapes=True)
    def V_pol0(self, E_static, pol):
        mu_ind = pol * E_static
        V_ind = -0.5 * KC * tf.reduce_sum(mu_ind * E_static, axis=[-1, -2])
        return V_ind, mu_ind
    
    def __call__(self, coords_uc, lattice, mol_size, monos, dipos=None, quads=None, pol=None, pol0=False):   
        (V_esp, V_intra), E_static = self.V_esp(coords_uc, lattice, mol_size, monos, dipos, quads, field=True)
        V_esp, V_intra = V_esp * KC, V_intra * KC
        if pol0:
            V_ind, mu_ind = self.V_pol0(E_static, pol)
        else:
            V_ind, mu_ind = self.V_pol(E_static, pol, coords_uc, lattice)        
        return V_esp - V_intra + V_ind, mu_ind    
    
def A(x, ks=[-1]):
    for k in ks:
        x = tf.expand_dims(x, axis=k)
    return x

def thole_damping(R1, pol, smear=-0.39):
    U = R1 / tf.math.pow(tf.linalg.matrix_transpose(pol) * pol, 1/6)
    exponent = smear * tf.math.pow(U, 3)
    coeff = tf.exp(exponent)
    L3 = 1.0 - coeff
    L5 = 1.0 - (1.0 - exponent) * coeff
    return L3, L5

def prepare_geometry(coordinates, lattice=None, cutoff=None, dtype_tf=tf.float32):
    if lattice is None:
        R_cells = tf.zeros((1, 3), dtype=dtype_tf)
    else:
        x_, y_, z_ = np.ceil(cutoff / np.linalg.norm(lattice, axis=-1))
        cell_idxs = tf.cast(np.mgrid[-x_:x_+1, -y_:y_+1, -z_:z_+1].T.reshape([-1, 3]), dtype_tf)
        R_cells = tf.matmul(cell_idxs, lattice)
    Rx1 = A(A(coordinates, [1]) - A(coordinates, [0]), [0]) + A(R_cells, [1, 1])
    RxR = A(Rx1, [-1]) * A(Rx1, [-2])
    R1 = tf.linalg.norm(Rx1, axis=-1)
    R2 = R1 * R1
    R5 = R2 * R2 * R1
    return Rx1, RxR, R1, R2, R5

def get_distances_cutoff(sc_coords, sc_cell, cutoff, index_1=None, index_2=None):
    if index_1 is None or index_2 is None:
        index_1, index_2 = np.triu_indices(sc_coords.shape[1], k=1)
    R, Rx1 = get_pbc_distances(sc_coords, sc_cell, index_1, index_2)
    cutoff_indices = tf.where(R < cutoff)
    index_1 = tf.gather(index_1, cutoff_indices[:, 1])
    index_2 = tf.gather(index_2, cutoff_indices[:, 1])
    R = tf.gather(R, cutoff_indices[:, 1], axis=1)[..., tf.newaxis]
    Rx1 = tf.gather(Rx1, cutoff_indices[:, 1], axis=1)
    return R, Rx1, index_1, index_2

@tf.function
def get_pbc_distances(sc_coords, sc_cell, index_1, index_2):
    sc_fractional = fractional_coords(sc_coords, sc_cell)
    Rx1 = tf.gather(sc_fractional, index_2, axis=1) - tf.gather(sc_fractional, index_1, axis=1)
    Rx1 -= tf.math.floor(Rx1 + 0.5)
    Rx1 = tf.matmul(Rx1, sc_cell)
    return tf.linalg.norm(Rx1, axis=-1), Rx1

def generate_lattice_points(multipliers, lattice_vectors, no_self=True):
    x_max, y_max, z_max = multipliers[0], multipliers[1], multipliers[2]
    xx, yy, zz = tf.experimental.numpy.meshgrid(tf.range(x_max), tf.range(y_max), tf.range(z_max))
    lattice_points = tf.stack((tf.reshape(xx, [-1]), tf.reshape(yy, [-1]), tf.reshape(zz, [-1])), axis=1)
    if no_self:
        lattice_points = tf.gather_nd(lattice_points, tf.where(tf.reduce_sum(tf.math.abs(lattice_points), axis=-1) != 0),)
    return tf.matmul(lattice_points, lattice_vectors)

@tf.function(experimental_relax_shapes=True)
def expand_unit_cell(uc_coords, cells, multipliers=None, cutoff=8):
    if multipliers is None:
        multipliers = tf.reduce_max(tf.math.ceil(cutoff / tf.linalg.norm(cells, axis=-1)), axis=0)
    sc_cell = cells * multipliers[:, None]
    lattice_points = tf.expand_dims(generate_lattice_points(multipliers, cells, no_self=True), axis=2)
    sc_coords = tf.expand_dims(uc_coords, axis=1) + lattice_points
    return tf.concat((uc_coords, tf.reshape(sc_coords, (lattice_points.shape[0], -1, 3))), axis=1), sc_cell

@tf.function(experimental_relax_shapes=True)
def fractional_coords(asymmetric_units, cells):
    return tf.transpose(tf.linalg.solve(cells, tf.transpose(asymmetric_units, [0, 2, 1]), adjoint=True), [0, 2, 1])
