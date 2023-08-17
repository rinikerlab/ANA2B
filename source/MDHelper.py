import os

from Intra import Intra
from packmol import pack_box

from ase import Atoms
from ase.optimize import LBFGS
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToPDBFile, MolToMolFile
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem.AllChem import EmbedMolecule
from rdkit.Chem.Descriptors import MolWt
from openmm import unit
from openmm.app import PDBFile
from openmoltools import packmol
from openmmforcefields.generators import SMIRNOFFTemplateGenerator, SystemGenerator
from openff.toolkit.utils.toolkits import RDKitToolkitWrapper
from openff.toolkit.topology import Molecule, Topology
from scipy.spatial.distance import cdist
from sys import stdout

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import openmm as mm
import openmoltools
import tensorflow as tf


def build_atoms_system(hybrid_ff, pdb_file_name_box, pdb_file_name_monomer):
    molecule = md.load(pdb_file_name_monomer)
    mol_size = molecule.topology.n_atoms
    md_trajectory = md.load(pdb_file_name_box)
    cells = md_trajectory.unitcell_vectors * 10
    coords_uc = md_trajectory.xyz * 10
    elements_uc = [atom.element.symbol for atom in md_trajectory.topology.atoms]
    return hybrid_ff.initialize(elements_uc, coords_uc, cells, mol_size)  

def wrap_molecule_cog(atoms):
    coords = atoms.get_positions()
    cell = atoms.get_cell().array
    molecule_coords = coords.reshape((atoms.calc._n_molecules_uc, -1, 3))
    cog_molecules = np.mean(molecule_coords, axis=1) * 1.2
    fractional = np.linalg.solve(cell.T, np.asarray(cog_molecules).T).T
    shift = np.dot(((fractional % 1) - fractional), cell)[:, None]
    wrapped_coords = (molecule_coords + shift).reshape((-1, 3))
    atoms.set_positions(wrapped_coords)
    return atoms

def build_box_and_equilibrate(smile, system_name, folder_path, 
              density=1.0, target_box_length=25, temperature=300, n_steps=100000, dt=2, margin=1):
    rdkit_mol = AddHs(MolFromSmiles(smile))
    EmbedMolecule(rdkit_mol)
    MolToMolFile(rdkit_mol, os.path.join(folder_path, f'{system_name}.mol'))
    MolToPDBFile(rdkit_mol, os.path.join(folder_path, f'{system_name}.pdb'))
    #molecule = Molecule.from_file(os.path.join(folder_path, f'{system_name}.mol'))
    molecule = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
    molecule_md = md.load(os.path.join(folder_path, f'{system_name}.pdb'))
    mass = MolWt(rdkit_mol) * unit.gram / unit.mole    
    density = density * unit.gram / (unit.milliliter)
    vol_per_molecule = ((mass / density).in_units_of(unit.angstrom**3 / unit.mole) / unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.angstrom**3)
    n_molecules = int((target_box_length ** 3) / vol_per_molecule)  
    openff = SMIRNOFFTemplateGenerator(molecules=[molecule], forcefield='openff-2.0.0-rc.2') 
    openff.add_molecules([molecule])
    forcefield = mm.app.ForceField()
    forcefield.registerTemplateGenerator(openff.generator)    
    mdtraj_molecule = md.load(os.path.join(folder_path, f'{system_name}.pdb'))
    box = packmol.pack_box(mdtraj_molecule, [n_molecules], box_size=target_box_length * margin)
    box_topo = box.topology
    openmm_topo = box_topo.to_openmm()
    openmm_topo.setPeriodicBoxVectors(box.unitcell_vectors[0] * unit.nanometer)
    system = forcefield.createSystem(openmm_topo, nonbondedMethod=mm.app.PME, constraints=mm.app.HBonds, rigidWater=False)
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, temperature * unit.kelvin, 25))
    modeller = mm.app.Modeller(openmm_topo, box.xyz[0])    
    integrator = mm.LangevinIntegrator(temperature * unit.kelvin, 1 / unit.picosecond, dt * unit.femtosecond)
    simulation = mm.app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    PDBFile.writeFile(simulation.topology, box.xyz[0], open(os.path.join(folder_path, f'{system_name}_EQBOX_INITIAL.pdb'), 'w'))
    simulation.minimizeEnergy()    
    simulation.reporters.append(mm.app.DCDReporter(os.path.join(folder_path, f'{system_name}_EQBOX.dcd'), 1000, enforcePeriodicBox=True))
    simulation.reporters.append(mm.app.StateDataReporter(stdout, 1000 , step=True,
            potentialEnergy=True, temperature=True, progress=True, remainingTime=True, volume=True,
            speed=True, totalSteps=n_steps, separator='\t'))
    simulation.step(n_steps)
    simulation.minimizeEnergy()
    simulation.step(1)
    trajectory = md.load(os.path.join(folder_path, f'{system_name}_EQBOX.dcd'), 
                         top=os.path.join(folder_path, f'{system_name}_EQBOX_INITIAL.pdb'))[-1]
    trajectory.save_pdb(os.path.join(folder_path, f'{system_name}_EQBOX.pdb'))

# Builss a 'pure liquid' box from a smile.
def build_box(smile_string, system_name, density=1.0, target_box_length=25, margin=1.1, folder_path='', optimize_mol=True, conv=0.01):
    model = Intra(use_D3=True)
    rdkit_mol = AddHs(MolFromSmiles(smile_string))
    EmbedMolecule(rdkit_mol)
    mass = MolWt(rdkit_mol) * unit.gram / unit.mole    
    density = density * unit.gram / (unit.milliliter)
    vol_per_molecule = ((mass / density).in_units_of(unit.angstrom**3 / unit.mole) / unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.angstrom**3)
    n_molecules = int((target_box_length ** 3) / vol_per_molecule)     
    conformation = rdkit_mol.GetConformer().GetPositions()    
    if optimize_mol:
        elements = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
        atoms = model.initialize(conformation[None], elements)
        optimizer = LBFGS(atoms)
        optimizer.run(fmax=conv)
        conformation = atoms.get_positions().astype(np.float32)        
    MolToPDBFile(rdkit_mol, os.path.join(folder_path, f'{system_name}.pdb'))
    md_topo = md.load(os.path.join(folder_path, f'{system_name}.pdb')).topology
    system_topo = [md.Trajectory(conformation * 0.1, md_topo) for idx in range(n_molecules)]
    print('Building Box')
    box = pack_box(system_topo, [1 for _ in range(n_molecules)], box_size=target_box_length * margin)
    box.save_pdb(os.path.join(folder_path, f'{system_name}_BOX.pdb'))
    return box