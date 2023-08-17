import os
import sys

sys.path.append('../..')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import tensorflow as tf

from ase import Atoms
from ase.build import make_supercell
from ase.optimize import BFGS, LBFGS
from ase.vibrations import Vibrations
from ase.md import MDLogger
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary, ZeroRotation)
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.units import fs
from ase.io import read
from ase.io.trajectory import Trajectory
from Barostat import MonteCarloBarostat

from HybridFF import HybridFF
from MDHelper import wrap_molecule_cog, build_atoms_system
from Utilities import show_results, write_xyz, validate
from Constants import EV_TO_KJ, H_TO_KJ, BOHR_TO_ANGSTROM


SYSTEM_NAME = sys.argv[1]
REPLICA = int(sys.argv[2])
T = float(sys.argv[3])
P = float(sys.argv[4])
N_STEPS = int(sys.argv[5])
BAROSTAT_FREQUENCY = int(sys.argv[6])
READOUT_FREQUENCY = int(sys.argv[7])
DT = 0.5
CONV = 50 / EV_TO_KJ
N_STEPS_EQ = 2000

print(SYSTEM_NAME, REPLICA, T, P, N_STEPS, BAROSTAT_FREQUENCY, READOUT_FREQUENCY)

OUTPUT_FOLDER = f'data_ase/{SYSTEM_NAME}/'
try:
    os.mkdir(OUTPUT_FOLDER)
except:
    pass

BOX_FILE = f'systems/{SYSTEM_NAME}_EQBOX.pdb'
MONOMER_FILE = f'systems/{SYSTEM_NAME}.pdb'
hybrid_ff = HybridFF(debug=True, pol0=True, D3_3B_cutoff=5.0, D3_2B_cutoff=10.0, D3_CN_cutoff=5.0)
atoms = build_atoms_system(hybrid_ff, BOX_FILE, MONOMER_FILE)
barostat = MonteCarloBarostat(T=T, P=P, aniso=False, atoms=atoms, lr_correction=False)

optimizer = LBFGS(atoms) 
optimizer.run(fmax=CONV)  

NAME_LOGFILE = f'{OUTPUT_FOLDER}{SYSTEM_NAME}_{REPLICA}.log'
NAME_TRAJFILE = f'{OUTPUT_FOLDER}{SYSTEM_NAME}_{REPLICA}.traj'
NAME_LOGFILE_EQ = f'{OUTPUT_FOLDER}{SYSTEM_NAME}_EQ_{REPLICA}.log'

MaxwellBoltzmannDistribution(atoms, temperature_K=T, force_temp=True)
Stationary(atoms)
ZeroRotation(atoms)

with Andersen(atoms, DT * fs, temperature_K=T, andersen_prob=1e-1, 
              logfile=NAME_LOGFILE_EQ, loginterval=READOUT_FREQUENCY) as dyn:
    for _ in dyn.irun(N_STEPS_EQ):   
        atoms = wrap_molecule_cog(atoms)
        if dyn.nsteps % 10 == 0:
            atoms = barostat(atoms)

with Andersen(atoms, DT * fs, temperature_K=T, andersen_prob=1e-2, 
              trajectory=NAME_TRAJFILE, logfile=NAME_LOGFILE, loginterval=READOUT_FREQUENCY) as dyn:
    for _ in dyn.irun(N_STEPS):    
        atoms = wrap_molecule_cog(atoms)
        if dyn.nsteps % BAROSTAT_FREQUENCY == 0:
            atoms = barostat(atoms)
            



