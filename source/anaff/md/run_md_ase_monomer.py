import os
import sys

sys.path.append('../..')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import tensorflow as tf

from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.md import MDLogger
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,Stationary, ZeroRotation)
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.md.nvtberendsen import NVTBerendsen
from ase.units import fs
from ase.io import read
from ase.io.trajectory import Trajectory

from Intra import Intra
from Utilities import show_results, write_xyz, validate
from Constants import EV_TO_KJ, H_TO_KJ, BOHR_TO_ANGSTROM

from rdkit.Chem.rdmolfiles import MolFromSmiles, MolFromPDBFile, MolToMolFile
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem.AllChem import EmbedMolecule


DT = 0.5
CONV = 10 / EV_TO_KJ
N_STEPS_EQ = 2000
SYSTEM_NAME = sys.argv[1]
REPLICA = int(sys.argv[2])
T = float(sys.argv[3])
N_STEPS = int(sys.argv[4])
READOUT_FREQUENCY = int(sys.argv[5])

print(SYSTEM_NAME, REPLICA, T, N_STEPS, READOUT_FREQUENCY)

OUTPUT_FOLDER = f'data_ase_monomer/{SYSTEM_NAME}/'
try:
    os.mkdir(OUTPUT_FOLDER)
except:
    pass

MONOMER_FILE = f'systems/{SYSTEM_NAME}.pdb'
rdkit_mol = MolFromPDBFile(MONOMER_FILE, removeHs=False)
EmbedMolecule(rdkit_mol)
conformation = rdkit_mol.GetConformer().GetPositions()    
elements = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
#conformation = md.load(MONOMER_FILE)
#elements = [atom.element.symbol for atom in conformation.topology.atoms]

model = Intra(use_D3=True, D3_3B_cutoff=5.0, D3_2B_cutoff=10.0, D3_CN_cutoff=5.0)
atoms = model.initialize(conformation[None], elements)
optimizer = LBFGS(atoms)
optimizer.run(fmax=CONV) 

NAME_LOGFILE = f'{OUTPUT_FOLDER}{SYSTEM_NAME}_MONOMER{REPLICA}.log'
NAME_TRAJFILE = f'{OUTPUT_FOLDER}{SYSTEM_NAME}_MONOMER{REPLICA}.traj'
NAME_LOGFILE_EQ = f'{OUTPUT_FOLDER}{SYSTEM_NAME}_EQ_MONOMER{REPLICA}.log'

MaxwellBoltzmannDistribution(atoms, temperature_K=T, force_temp=True)
Stationary(atoms)
ZeroRotation(atoms)
with NVTBerendsen(atoms, DT * fs, temperature_K=T, taut=10*fs, fixcm=False) as dyn:
    for step in dyn.irun(N_STEPS_EQ): 
        pass
Stationary(atoms)
ZeroRotation(atoms)
with Langevin(atoms, DT * fs, temperature_K=T, friction=1e-0, fixcm=False, # 1e-2
          logfile=NAME_LOGFILE, trajectory=NAME_TRAJFILE,
          loginterval=READOUT_FREQUENCY) as dyn:
    for step in dyn.irun(N_STEPS):   
        pass  



