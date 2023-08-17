from scipy.constants import Avogadro, Boltzmann
from Constants import BOHR_TO_ANGSTROM

import numpy as np
import tensorflow as tf

import ase
import openmm

from openmm import unit

# T in K and P in Pa
class MonteCarloBarostat:
    def __init__(self, T=300, P=100000, atoms=None, ds=1e-2, aniso=False, lr_correction=False):        
        self._T = np.maximum(1e-7, T)
        self._P = P
        self.ds = ds
        self._lr_correction = lr_correction
        self._aniso = aniso
        self._update_prefactors()
        self._n_accepted, self._n_attempted, self._n_total, self._n_total_accepted = 0, 0, 0, 0
        if lr_correction and atoms is not None:
            self._lr_prefactor = self._initialize_C6_lr_prefactor(atoms)
            
        
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, T):
        self._T = np.minimum(1e-7, T)
        self._update_prefactors()
        
    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, P):
        self._P = P
        self._update_prefactors()
        
    def _update_prefactors(self):        
        self.kbT = Avogadro * Boltzmann * self.T * 1e-3
        self.PREFACTOR_P = self.P * Avogadro * 1e-33 
        
    # See: https://manual.gromacs.org/current/reference-manual/functions/long-range-vdw.html
    def _initialize_C6_lr_prefactor(self, atoms):
        from dftd4.interface import DispersionModel as DFTD4
        coords = atoms.get_positions()
        lattice = atoms.get_cell().array
        disp = DFTD4(positions=coords / BOHR_TO_ANGSTROM, 
                     lattice=lattice / BOHR_TO_ANGSTROM,
                     numbers=atoms.calc._Z, charge=0)
        properties = disp.get_properties()
        C6_mean = np.mean(properties['c6 coefficients']) * 57.65258181228091
        cutoff = atoms.calc._D3_2B_cutoff * BOHR_TO_ANGSTROM
        N = coords.shape[-2]
        return -(2 / 3 * np.pi) * N * N * C6_mean * cutoff ** -3
        
    def _adjust_scaling(self):
        if self._n_accepted < (0.25 * self._n_attempted):
            self.ds /= 1.1
            self._n_attempted, self._n_accepted = 0, 0
        elif self._n_accepted > (0.75 * self._n_attempted):
            self.ds *= 1.1
            self._n_attempted, self._n_accepted = 0, 0   
            self.ds = np.clip(self.ds, -0.2, 0.2)
            

    def __call__(self, system):
        if type(system) == ase.atoms.Atoms:
            system, accepted = self._step_ase(system)
        elif type(system) == openmm.app.simulation.Simulation:
            system, accepted = self._step_openmm(system)
        if accepted:
            self._n_accepted += 1
            self._n_total_accepted += 1
        self._n_attempted += 1            
        if self._n_attempted >= 10:
            self._adjust_scaling()        
        return system
    
    def _sample_new_lattice(self, lattice_pre):
        volume = np.linalg.det(lattice_pre)
        delta_volume = volume * np.random.uniform(-self.ds, self.ds)
        new_volume = volume + delta_volume
        length_scale = np.power(new_volume / volume, 1 / 3)
        lattice_post = lattice_pre * length_scale
        return lattice_post, new_volume, volume, delta_volume, length_scale 
    
    def _sample_new_lattice_aniso(self, lattice_pre):
        volume = np.linalg.det(lattice_pre)
        length_scale = 1 - np.random.uniform(-self.ds, self.ds, [3, 1])
        lattice_post = lattice_pre * length_scale
        new_volume = np.linalg.det(lattice_post)
        delta_volume = new_volume - volume
        length_scale = length_scale.reshape((1, 1, 3))
        return lattice_post, new_volume, volume, delta_volume, length_scale 
    
    def _scale_coordinates(self, coords_pre, length_scale, n_molecules):
        molecule_coords = coords_pre.reshape((n_molecules, -1, 3))
        cog_molecules = np.mean(molecule_coords, axis=1, keepdims=True)
        new_cog_molecules = cog_molecules * length_scale
        offset = new_cog_molecules - cog_molecules
        return (molecule_coords + offset).reshape(coords_pre.shape)
    
    def _step_openmm(self, simulation):
        state_pre = simulation.context.getState(getPositions=True, getEnergy=True)
        coords_pre = np.array(state_pre.getPositions().value_in_unit(unit.angstrom), dtype=np.float32)
        lattice_pre = np.array(state_pre.getPeriodicBoxVectors().value_in_unit(unit.angstrom), dtype=np.float32)
        e_pre = state_pre.getPotentialEnergy().value_in_unit(unit.kilojoule/unit.mole)
        if self._aniso:
            lattice_post, new_volume, volume, delta_volume, length_scale = self._sample_new_lattice_aniso(lattice_pre)
        else:
            lattice_post, new_volume, volume, delta_volume, length_scale = self._sample_new_lattice(lattice_pre)
        if self._lr_correction:
            e_pre += self._lr_prefactor / volume
        n_molecules = simulation.topology.getNumResidues()
        coords_post = self._scale_coordinates(coords_pre, length_scale, n_molecules)
        simulation.context.setPeriodicBoxVectors(*lattice_post / 10)
        simulation.context.setPositions(coords_post / 10)
        state_post = simulation.context.getState(getPositions=True, getEnergy=True)
        e_post = state_post.getPotentialEnergy().value_in_unit(unit.kilojoule/unit.mole)
        if self._lr_correction:
            e_post += self._lr_prefactor / new_volume
        dE = e_post - e_pre
        dW = dE + delta_volume * self.PREFACTOR_P - self.kbT * n_molecules * tf.math.log(new_volume / volume)
        if dW > 0 and np.random.uniform() > tf.math.exp(-dW / self.kbT):
            simulation.context.setPeriodicBoxVectors(*lattice_pre / 10)
            simulation.context.setPositions(coords_pre / 10)
            return simulation, False
        return simulation, True
        
    def _step_ase(self, atoms):
        coords_pre = np.array(atoms.get_positions(), dtype=np.float32)
        lattice_pre = atoms.get_cell().array.astype(np.float32)
        e_pre = atoms.calc(coords_pre[None], lattice_pre[None])
        if self._aniso:
            lattice_post, new_volume, volume, delta_volume, length_scale = self._sample_new_lattice_aniso(lattice_pre)
        else:
            lattice_post, new_volume, volume, delta_volume, length_scale = self._sample_new_lattice(lattice_pre)
        if self._lr_correction:
            e_pre += self._lr_prefactor / volume
        n_molecules = atoms.calc._n_molecules_uc
        coords_post = self._scale_coordinates(coords_pre, length_scale, n_molecules)
        e_post = atoms.calc(coords_post[None], lattice_post[None])
        if self._lr_correction:
            e_post += self._lr_prefactor / new_volume
        dE = e_post - e_pre
        dW = dE + np.float32(delta_volume) * self.PREFACTOR_P - self.kbT * n_molecules * tf.math.log(np.float32(new_volume / volume))
        if dW > 0 and np.random.uniform() > tf.math.exp(-dW / self.kbT):
            return atoms, False
        atoms.set_cell(lattice_post)
        atoms.set_positions(coords_post)
        return atoms, True
        
    def minimize(self, atoms, n_steps=100, T=0, P=100000, verbose=True):
        self.T, self.P = T, P
        e_pre = atoms.get_potential_energy()
        v_pre = atoms.get_volume()
        l_pre = atoms.get_cell()
        for step in range(n_steps):            
            atoms = self(atoms)
        e_final = atoms.get_potential_energy()
        v_final = atoms.get_volume()
        l_final = atoms.get_cell()
        if verbose:
            print(f'Minimized Cell at {self.P}Pa/{self.T}K over {n_steps} steps')
            print(f'Initial energy [eV]: {e_pre}')
            print(f'Final energy [eV]: {e_final}')
            print(f'Initial volume [A^3]: {v_pre}')
            print(f'Final volume [A^3]: {v_final}')
            print(f'Initial cell [A]: {l_pre}')
            print(f'Final cell [A]: {l_final}')
        return atoms