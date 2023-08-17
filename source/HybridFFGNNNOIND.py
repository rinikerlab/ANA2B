import os

import numpy as np
import tensorflow as tf

from ANA2B_NOIND import ANA2B, G_matrices_sym
from ase import Atoms
from ase.calculators.calculator import Calculator
from dftd3.interface import Structure, RationalDampingParam, DispersionModel
from Ewald import Ewald, expand_unit_cell, fractional_coords
from GraphBuilder import GraphBuilder
from Intra import Intra
from Utilities import to_Z, switch, gaussian, A, build_Rx2
from Constants import KC, H_TO_KJ, BOHR_TO_ANGSTROM, EV_TO_KJ


class HybridFF(Calculator):
    def __init__(self, 
                 cutoff=10.0, 
                 tolerance=1e-4, 
                 knorm=30, 
                 use_D3=True,
                 functional='PBE0',
                 D3_2B_cutoff=15,#31.75063265418,
                 D3_3B_cutoff=8,#21.167088436119997,
                 D3_CN_cutoff=15,#21.167088436119997,
                 n_alpha=2,
                 ana_model=None,
                 debug=False):
        super().__init__()
        module_dir = os.path.dirname(__file__)
        weights_dir = os.path.join(module_dir, 'weights_models')
        self.Builder = GraphBuilder()
        self.Ewald = Ewald(cutoff=cutoff, tolerance=tolerance, knorm=knorm)  
        #if ana_model is None:
        #    ana_model = ANA2B(cutoff=6.5, n_units=64, n_steps=1)
        #    file_path = os.path.join(weights_dir, 'ANA2BGNN1_2_D3PBE0_FULLMSE_CUTOFF2B6.5_S392R_E182')
        #    ana_model.load_weights(file_path)
        self.ANA = ana_model
        self.Intra = Intra()
        self._use_D3 = use_D3
        self._cutoff_ana = ana_model.cutoff
        self._functional = functional
        self._D3_2B_cutoff = D3_2B_cutoff / BOHR_TO_ANGSTROM
        self._D3_3B_cutoff = D3_3B_cutoff / BOHR_TO_ANGSTROM
        self._D3_CN_cutoff = D3_CN_cutoff / BOHR_TO_ANGSTROM
        if self._use_D3:
            self._damping_params = RationalDampingParam(method=self._functional)
        self.debug = debug
        self.implemented_properties = ['forces', 'energy']
    
    def initialize(self, elements_uc, coords_uc, lattice, mol_size, shared_topo=False):    
        self._mol_size, self._uc_size = mol_size, coords_uc.shape[1]   
        self._n_molecules_uc = self._uc_size // self._mol_size 
        coords_monomers = tf.reshape(coords_uc, (self._n_molecules_uc, self._mol_size, 3))
        elements_monomer = elements_uc[:self._mol_size] 
        self._elements_uc = elements_uc
        self._elements_monomer = elements_monomer 
        self._graph_topo = self.Builder.from_coords(coords_monomers[0].numpy(), elements_monomer)  
        self._nodes_topo = self.ANA.TopoGNN(self._graph_topo).nodes
        self.Intra.initialize(coords_monomers, elements_monomer)
        self._Z = to_Z(elements_uc)
        if self._use_D3:
            self._D3_model = DispersionModel(self._Z, np.array(coords_uc[0]) / BOHR_TO_ANGSTROM, lattice=np.array(lattice[0]) / BOHR_TO_ANGSTROM)
            self._D3_model.set_realspace_cutoff(self._D3_2B_cutoff, self._D3_3B_cutoff, self._D3_CN_cutoff)
        atoms = Atoms(positions=coords_uc[0], symbols=elements_uc, cell=lattice[0])
        atoms.calc = self
        return atoms        
        
    def D3(self, coords, lattice, grad=False):
        self._D3_model.update(positions=np.array(coords[0]) / BOHR_TO_ANGSTROM, lattice=np.array(lattice[0]) / BOHR_TO_ANGSTROM)  
        output = self._D3_model.get_dispersion(self._damping_params, grad=grad)
        if grad:
            return output['energy'] * H_TO_KJ, output['gradient'] * (H_TO_KJ / BOHR_TO_ANGSTROM)
        return output['energy'] * H_TO_KJ

    def _calc_gradient(self, coords, lattice):
        with tf.GradientTape() as tape:
            tape.watch(coords)
            coords_monomers = tf.reshape(coords, (self._n_molecules_uc, self._mol_size, 3))
            V_intra, (monos, dipos, quads, ratio) = self.Intra(coords_monomers, grad=False)
            V_intra = tf.reduce_sum(V_intra)
            monos = tf.experimental.numpy.vstack(monos)[tf.newaxis]
            dipos = tf.experimental.numpy.vstack(dipos)[tf.newaxis]
            quads = tf.experimental.numpy.vstack(quads)[tf.newaxis]
            ratio = tf.experimental.numpy.vstack(ratio)[tf.newaxis]
            V_esp, V_esp_intra = self.Ewald.V_esp(coords, lattice, self._mol_size, monos, dipos, quads, field=False) 
            V_esp, V_esp_intra = V_esp * KC, V_esp_intra * KC
            V_esp_inter = V_esp - V_esp_intra
            V_ana = self._calc_energy_ana(coords, lattice, monos, dipos, quads)
            V_total = V_esp_inter + V_intra + V_ana 
        dV_total = tape.gradient(V_total, coords)
        if self._use_D3:
            V_D3, dV_D3 = self.D3(coords, lattice, grad=True)        
            V_total = V_total + V_D3
            dV_total = dV_total + dV_D3 #dV_intra + dV_espind + dV_D3 + dV_ana
        return V_total, dV_total    
    
    def _calc_energy(self, coords, lattice):
        coords_monomers = tf.reshape(coords, (self._n_molecules_uc, self._mol_size, 3))
        V_intra, (monos, dipos, quads, ratio) = self.Intra(coords_monomers, grad=False)
        V_intra = tf.reduce_sum(V_intra)
        monos = tf.experimental.numpy.vstack(monos)[tf.newaxis]
        dipos = tf.experimental.numpy.vstack(dipos)[tf.newaxis]
        quads = tf.experimental.numpy.vstack(quads)[tf.newaxis]
        ratio = tf.experimental.numpy.vstack(ratio)[tf.newaxis]
        V_esp, V_esp_intra = self.Ewald.V_esp(coords, lattice, self._mol_size, monos, dipos, quads, field=False) 
        V_esp, V_esp_intra = V_esp * KC, V_esp_intra * KC
        V_esp_inter = V_esp - V_esp_intra      
        V_ana = self._calc_energy_ana(coords, lattice, monos, dipos, quads)
        if self._use_D3:
            V_dispersion = self.D3(coords, lattice, grad=False)
        else:
            V_dispersion = 0.0
        return V_intra + V_esp_inter + V_dispersion + V_ana
        
    def _prepare_features_ana_pbc(self, coords_uc, lattice, monos, dipos, quads):
        coords_sc, lattice_sc = expand_unit_cell(coords_uc, lattice, cutoff=self.ANA.cutoff * 2)
        R1, R2, Rx1, Rx2, indices, batch_indices, n_molecules_sc = prepare_distances(coords_sc, 
                                                                                    lattice_sc, 
                                                                                    self._mol_size, 
                                                                                    self._uc_size, 
                                                                                    self.ANA.cutoff)
        indices_1, indices_2 = tf.unstack(indices, axis=-1)
        node_features_1 = tf.gather(self._nodes_topo, indices_1 % self._mol_size)
        node_features_2 = tf.gather(self._nodes_topo, indices_2 % self._mol_size)
        node_features = node_features_1 + node_features_2
        monos_1, monos_2 = tf.gather(monos[0], indices_1), tf.gather(monos[0], indices_2)
        dipos_1, dipos_2 = tf.gather(dipos[0], indices_1), tf.gather(dipos[0], indices_2)
        quads_1, quads_2 = tf.gather(quads[0], indices_1), tf.gather(quads[0], indices_2)
        feature_12 = tf.concat((node_features_1, node_features_2), axis=-1)
        feature_21 = tf.concat((node_features_2, node_features_1), axis=-1)
        node_features = self.ANA.feature_embedding(feature_12) + self.ANA.feature_embedding(feature_21)    
        switch_function = switch(R1, self.ANA.cutoff - 1.0, self.ANA.cutoff)
        distance_features = gaussian(R1, np.logspace(-1, 0, self.ANA.n_gaussians)) * switch_function
        G_features = G_matrices_sym(monos_1, monos_2, dipos_1, dipos_2, quads_1, quads_2, Rx1, Rx2)
        S_features = tf.concat((G_features, node_features, distance_features), axis=-1)
        K_features = tf.concat((node_features, distance_features), axis=-1)  
        return G_features, S_features, K_features, R1, R2, switch_function, n_molecules_sc
    
    def _calc_energy_ana(self, coords, lattice, monos, dipos, quads):
        G_features, S_features, K_features, R1, R2, switch_function, n_molecules_sc =\
            self._prepare_features_ana_pbc(coords, lattice, monos, dipos, quads)
        S2_static, S2_at = tf.split(self.ANA.S_static(S_features) * switch_function, 2, axis=-1)
        K1_static, K2_static, K_at = tf.split(self.ANA.K(K_features), 3, axis=-1)
        V_ex = (K1_static * S2_static) / R1 + (K2_static * S2_static) / R2
        V_at = K_at * S2_at 
        quotient = self._n_molecules_uc / n_molecules_sc
        return tf.reduce_sum(V_ex - V_at) * quotient
    
    def _calc_gradient_ana(self, coords, lattice, monos, dipos, quads):
        with tf.GradientTape() as tape:
            tape.watch(coords)
            V_ana = self._calc_energy_ana(coords, lattice, monos, dipos, quads)
        dV_ana = tape.gradient(V_ana, coords)
        return V_ana, dV_ana
    
    def __call__(self, coords, lattice, grad=False):
        coords = tf.convert_to_tensor(tf.cast(coords, dtype=tf.float32))
        lattice = tf.cast(lattice, dtype=tf.float32)
        if grad:
            return self._calc_gradient(coords, lattice)
        else:
            return self._calc_energy(coords, lattice)
        
    def get_components(self, coords, elements, lattice):
        coords_monomers = tf.reshape(coords, (self._n_molecules_uc, self._mol_size, 3))
        V_intra, (monos, dipos, quads, ratio) = self.Intra(coords_monomers, grad=False)
        self._ratio = ratio
        V_intra = tf.reduce_sum(V_intra)
        if self._use_D3:
            V_dispersion_uc, V_dispersion_monos = D3(coords, elements, lattice, coords_monomers, elements[:self._mol_size])
            V_dispersion_inter = V_dispersion_uc - V_dispersion_monos
        else:
            V_dispersion_uc, V_dispersion_inter = 0, 0
        monos = tf.experimental.numpy.vstack(monos)[tf.newaxis]
        dipos = tf.experimental.numpy.vstack(dipos)[tf.newaxis]
        quads = tf.experimental.numpy.vstack(quads)[tf.newaxis]
        ratio = tf.experimental.numpy.vstack(ratio)[tf.newaxis]

        V_esp, V_esp_intra = self.Ewald.V_esp(coords, lattice, self._mol_size, monos, dipos, quads, field=False) 
        V_esp, V_esp_intra = V_esp * KC, V_esp_intra * KC
        V_esp_inter = V_esp - V_esp_intra      
        V_ana = self._calc_energy_ana(coords, lattice, monos, dipos, quads)        
        output = {}
        output['V_total'] = (V_esp_inter + V_ana + V_dispersion_uc + V_intra) / self._n_molecules_uc
        output['V_inter'] = (V_esp_inter + V_ana + V_dispersion_inter) / self._n_molecules_uc
        output['V_intra'] = V_intra / self._n_molecules_uc
        output['V_esp_inter'] = V_esp_inter / self._n_molecules_uc
        output['V_ana'] = V_ana / self._n_molecules_uc
        output['V_D3_inter'] = V_dispersion_inter / self._n_molecules_uc
        return output
    
    def calculate(self, atoms, properties, system_changes):
        self.atoms = atoms.copy()
        energy, forces_x = self.get_energy_forces(atoms)
        self.results['energy'] = energy
        self.results['forces'] = forces_x
        
    def get_potential_energy(self, atoms, force_consistent=None):
        coords = atoms.get_positions().astype(np.float32)[None]
        lattice = np.array(atoms.get_cell()).astype(dtype=np.float32)[None]
        V = self(coords, lattice, grad=False).numpy().astype(np.float64)[0]
        return V / EV_TO_KJ
    
    def get_forces(self, atoms):
        coords = atoms.get_positions().astype(np.float32)[None]
        lattice = np.array(atoms.get_cell()).astype(dtype=np.float32)[None]
        V, dV = self(coords, lattice, grad=True)
        if self.debug:
            print(V[0].numpy() / self._n_molecules_uc, np.linalg.norm(dV), np.amax(np.abs(dV)))
        return -dV.numpy().astype(np.float64)[0] / EV_TO_KJ
    
    def get_energy_forces(self, atoms):
        coords = atoms.get_positions().astype(np.float32)[None]
        lattice = np.array(atoms.get_cell()).astype(dtype=np.float32)[None]
        V, dV = self(coords, lattice, grad=True)
        if self.debug:
            print(V[0].numpy() / self._n_molecules_uc, np.linalg.norm(dV), np.amax(np.abs(dV)))
        return V / EV_TO_KJ, -dV.numpy().astype(np.float64)[0] / EV_TO_KJ
    
def prepare_distances(sc_coords, sc_cell, mol_size, uc_size, cutoff):    
    sc_fractional = fractional_coords(sc_coords, sc_cell)
    num_molecules_sc = sc_fractional.shape[1] // mol_size
    intermolecular_indices = tf.where(get_mask(mol_size, num_molecules_sc, num_molecules_sc))
    index_1, index_2 = tf.squeeze(tf.split(intermolecular_indices, 2, axis=-1))
    vectors = tf.gather(sc_fractional, index_2, axis=1) - tf.gather(sc_fractional, index_1, axis=1)
    vectors -= tf.math.floor(vectors + 0.5)
    Rx1 = tf.matmul(vectors, sc_cell)
    R1 = tf.linalg.norm(Rx1, axis=-1)
    cutoff_indices = tf.where(R1 < cutoff)
    batch_indices, uc_indices = tf.squeeze(tf.split(cutoff_indices, 2, axis=-1))    
    parameter_indices = tf.gather(intermolecular_indices, cutoff_indices[:, 1]) % uc_size
    R1 = A(tf.gather_nd(R1, cutoff_indices))
    R2 = tf.square(R1)
    Rx1 = tf.gather_nd(Rx1, cutoff_indices) / R1
    Rx2 = build_Rx2(Rx1)
    return R1, R2, Rx1, Rx2, parameter_indices, batch_indices, num_molecules_sc

def get_mask(mol_size, num_molecules_axis0, num_molecules_axis1, dtype_tf=tf.float32):
    ones = tf.ones(mol_size, dtype=dtype_tf)[tf.newaxis]
    mol_ids_uc = tf.reshape(tf.range(1, num_molecules_axis0 + 1, dtype=dtype_tf)[:, tf.newaxis] * ones, [-1])
    mol_ids_sc = tf.reshape(tf.range(1, num_molecules_axis1 + 1, dtype=dtype_tf)[:, tf.newaxis] * ones, [-1])  
    return tf.math.minimum(tf.math.maximum(mol_ids_sc[tf.newaxis, :] - mol_ids_uc[:, tf.newaxis], 0), 1)

def D3(coords_uc, elements_uc, cells, monomers_uc=None, elements_monomer=None, functional='PBE0'):
    coords_uc, cells, monomers_uc = np.array(coords_uc), np.array(cells), np.array(monomers_uc)
    damping_params = RationalDampingParam(method=functional)
    dm = DispersionModel(to_Z(elements_uc), coords_uc[0] / BOHR_TO_ANGSTROM, lattice=cells[0] / BOHR_TO_ANGSTROM)
    output = dm.get_dispersion(damping_params, grad=False)
    energy_uc = output['energy'] * H_TO_KJ
    if monomers_uc is None:
        return energy_uc, gradient_uc, None
    #monomers_uc = np.stack(tf.split(coords_uc[0], n_molecules_uc, axis=0), axis=0)
    energies_mono = []
    elements_monomer = to_Z(elements_monomer)
    for monomer in monomers_uc:
        dm = DispersionModel(elements_monomer, monomer / BOHR_TO_ANGSTROM)
        output = dm.get_dispersion(damping_params, grad=False)
        energies_mono.append(output['energy'] * H_TO_KJ)
    energies_mono = np.sum(energies_mono)
    return energy_uc, energies_mono

    