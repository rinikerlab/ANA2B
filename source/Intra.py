import os

import numpy as np
import tensorflow as tf

from Utilities import A, cdist_tf_batch, build_Rx2, reshape_coeffs, to_Z
from Constants import H_TO_KJ, BOHR_TO_ANGSTROM, EV_TO_KJ
from AMPParams import AMPParams
from AMPMulti import AMP as AMPMulti
from AMPMulti import ONEHOTS_ELEMENTS, build_sin_kernel
from HelpersParams import build_graph, Graph 
from dftd3.interface import Structure, RationalDampingParam, DispersionModel
from ase.calculators.calculator import Calculator
from ase import Atoms


class Intra(Calculator):
    def __init__(self, 
                 cutoff=5.0, 
                 use_D3=False, 
                 functional='PBE0',
                 D3_2B_cutoff=31.75063265418,#,
                 D3_3B_cutoff=21.167088436119997,#,
                 D3_CN_cutoff=21.167088436119997):#,):
        super().__init__()        
        module_dir = os.path.dirname(__file__)
        weights_dir = os.path.join(module_dir, 'weights_models')
        self.use_D3 = use_D3
        self.functional = functional
        self.cutoff = cutoff        
        self.amp_params = AMPParams()
        self.amp_multi = AMPMulti(n_channels=32, node_size=128, cutoff=self.cutoff, order=2, num_steps=3)
        self._D3_2B_cutoff = D3_2B_cutoff / BOHR_TO_ANGSTROM
        self._D3_3B_cutoff = D3_3B_cutoff / BOHR_TO_ANGSTROM
        self._D3_CN_cutoff = D3_CN_cutoff / BOHR_TO_ANGSTROM
        
        file_path_multi = os.path.join(weights_dir, 'AMP2MULTI32_A0.9_2048x2097152x5_T2000_D3RAW_FINAL')
        self.amp_multi.load_weights(file_path_multi)
    
    def initialize(self, coords, elements):        
        coords = tf.convert_to_tensor(coords)
        self.batch_size, self.n_atoms, _ = coords.shape 
        self.elements = elements
        if self.use_D3:
            self.damping_params = RationalDampingParam(method=self.functional)
            self.Z = to_Z(elements)
            self.D3_model = DispersionModel(self.Z, coords[0].numpy() / BOHR_TO_ANGSTROM)
            self.D3_model.set_realspace_cutoff(self._D3_2B_cutoff, self._D3_3B_cutoff, self._D3_CN_cutoff)
        atoms = Atoms(positions=coords[0], symbols=elements)
        atoms.calc = self
        return atoms        
        
    def __call__(self, coords, grad=False, hess=False):
        coords = tf.convert_to_tensor(coords)
        if hess:
            return self._hessian(coords)            
        elif grad:
            return self._gradient(coords)
        else:
            return self._energy(coords)   

    def _energy(self, coords):  
        V_model, multipoles, _ = self._total_model(coords, grad=False)
        V_dispersion = 0.0
        if self.use_D3:
            V_dispersion, _ = self.D3(coords.numpy(), grad=False)
        return V_model + V_dispersion, multipoles

    def _gradient(self, coords):
        V_model, dV_model, multipoles, _ = self._total_model(coords, grad=True)
        V_dispersion, dV_dispersion = 0.0, 0.0
        if self.use_D3:
            V_dispersion, dV_dispersion = self.D3(coords.numpy(), grad=True)
        return V_model + V_dispersion, dV_model + dV_dispersion, multipoles
        
    def _hessian(self, coords):
        with tf.GradientTape(persistent=True) as hessian_tape:            
            hessian_tape.watch(coords)            
            V_model, dV_model, multipoles, graph = self._total_model(coords, grad=True)
        ddV_model = hessian_tape.batch_jacobian(dV_model, coords, experimental_use_pfor=False)
        return V_model, dV_model, ddV_model, multipoles    
    
    def D3(self, coords, grad=False):
        energies_mono, gradients_mono = [], []
        for monomer in coords:            
            self.D3_model.update(positions=monomer / BOHR_TO_ANGSTROM)            
            output = self.D3_model.get_dispersion(self.damping_params, grad=grad)
            energies_mono.append(output['energy'] * H_TO_KJ)
            if grad:
                gradients_mono.append(output['gradient'] * (H_TO_KJ / BOHR_TO_ANGSTROM))
        return np.array(energies_mono)[:, tf.newaxis], np.array(gradients_mono)

    
    def _delta_energy(self, coords, graph, grad=False):
        batch_size, n_atoms, _ = coords.shape        
        if grad:
            V_delta, dV_delta = self.amp_multi.direct_call(tf.experimental.numpy.vstack(coords), 
                                                        graph.senders, 
                                                        graph.receivers, 
                                                        graph.nodes, 
                                                        tf.repeat(tf.range(batch_size), [n_atoms]), 
                                                        graph.nodes.shape[0])
            return V_delta, tf.reshape(dV_delta, coords.shape)
        V_delta = self.amp_multi(tf.experimental.numpy.vstack(coords), graph.nodes, graph.senders, graph.receivers, graph.nodes.shape[0])
        return tf.math.segment_sum(V_delta, tf.repeat(tf.range(batch_size), [coords.shape[1]]))
    
    def _get_multipoles(self, coords):   
        graph, _ = build_graph(coords, self.elements, self.cutoff)
        multipoles = reshape_coeffs(self.amp_params(graph.nodes, graph.edges, graph.senders, graph.receivers, 
                                                    graph.n_node, graph.Rx1, graph.Rx2), coords.shape[0])
        return multipoles, graph
    
    def _total_model(self, coords, grad=False):
        multipoles, graph = self._get_multipoles(coords)
        if grad:            
            V_delta, dV_delta = self._delta_energy(coords, graph, grad=True)   
            return V_delta, dV_delta, multipoles, graph
        V_delta = self._delta_energy(coords, graph, grad=False)   
        return V_delta, multipoles, graph
    
    def mu(self, coords, grad=False):
        if grad:
            coords = tf.convert_to_tensor(coords)
            with tf.GradientTape() as tape:
                tape.watch(coords)
                graph, _ = build_graph(coords, self.elements, self.cutoff)
                multipoles = reshape_coeffs(self.amp_params(graph.nodes, graph.edges,                          
                                                   graph.senders, graph.receivers, 
                                                   graph.n_node, graph.Rx1, graph.Rx2), coords.shape[0])
                mu = tf.reduce_sum(multipoles[0] * coords + multipoles[1], axis=1)
            dmu = tape.gradient(mu, coords)[0]
            return mu, dmu
        else:
            graph, _ = build_graph(coords, self.elements, self.cutoff)
            multipoles = reshape_coeffs(self.amp_params(graph.nodes, graph.edges,                          
                                               graph.senders, graph.receivers, 
                                               graph.n_node, graph.Rx1, graph.Rx2), coords.shape[0])
            mu = tf.reduce_sum(multipoles[0] * coords + multipoles[1], axis=1)
            return mu
        
    def calculate(self, atoms, properties, system_changes):
        self.atoms = atoms.copy()
        energy, forces_x = self.get_energy_forces(atoms)
        self.results['energy'] = energy
        self.results['forces'] = forces_x
        
    def get_potential_energy(self, atoms, force_consistent=None):
        coords = atoms.get_positions().astype(np.float32)[None]
        V, _ = self(coords, grad=False)
        return V.numpy().astype(np.float64)[0] / EV_TO_KJ
    
    def get_forces(self, atoms):
        coords = atoms.get_positions().astype(np.float32)[None]
        V, dV, _ = self(coords, grad=True)
        return -dV.numpy().astype(np.float64)[0] / EV_TO_KJ
    
    def get_energy_forces(self, atoms):
        coords = atoms.get_positions().astype(np.float32)[None]
        V, dV, _ = self(coords, grad=True)
        return V.numpy().astype(np.float64)[0] / EV_TO_KJ, -dV.numpy().astype(np.float64)[0] / EV_TO_KJ