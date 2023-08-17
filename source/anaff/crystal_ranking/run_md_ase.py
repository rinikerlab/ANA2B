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


SYSTEM = sys.argv[1]
POLY_KEY = sys.argv[2]
REPLICA = int(sys.argv[3])
T = float(sys.argv[4])
P = float(sys.argv[5])
N_STEPS = int(sys.argv[6])
BAROSTAT_FREQUENCY = int(sys.argv[7])
READOUT_FREQUENCY = int(sys.argv[8])
DT = 0.5
N_STEPS_EQ = 2000

print(SYSTEM, REPLICA, T, P, N_STEPS, BAROSTAT_FREQUENCY, READOUT_FREQUENCY)


DATA = np.load('../../../data/test_sets/DATA_CSP_ranking.npy', allow_pickle=True).item()
MIN_STRUCTURES = np.load(f'data/STRUCTURES_{SYSTEM}.npy', allow_pickle=True).item()

mol_size = len(DATA[SYSTEM][POLY_KEY]['monomer_symbols'])
elements_uc = DATA[SYSTEM][POLY_KEY]['uc_symbols']
coords_uc = MIN_STRUCTURES[POLY_KEY]['coordinates']
cell = MIN_STRUCTURES[POLY_KEY]['cell']

hybrid_ff = HybridFF(debug=True, pol0=False)
barostat = MonteCarloBarostat(T=T, P=P, aniso=True)
#hybrid_ff = HybridFF(debug=True, pol0=True, D3_3B_cutoff=5.0, D3_2B_cutoff=10.0, D3_CN_cutoff=5.0)
atoms = hybrid_ff.initialize(elements_uc, coords_uc, cell, mol_size)  

optimizer = LBFGS(atoms) # BFGS
optimizer.run(fmax=4 / EV_TO_KJ)

#atoms = atoms.repeat(np.ceil(10 / np.linalg.norm(atoms.get_cell().array, axis=-1)).astype(np.int32))
#elements = atoms.get_chemical_symbols()
#cell = atoms.get_cell().array[None].astype(np.float32)
#coords = atoms.get_positions()[None].astype(np.float32)
#atoms = hybrid_ff.initialize(elements, coords, cell, mol_size)  

FOLDER_PATH = f'data_md/{SYSTEM}/'
try:
    os.mkdir(FOLDER_PATH)
except:
    pass
NAME_LOGFILE = f'{FOLDER_PATH}{SYSTEM}_{POLY_KEY}_{REPLICA}.log'
NAME_TRAJFILE = f'{FOLDER_PATH}{SYSTEM}_{POLY_KEY}_{REPLICA}.traj'
NAME_LOGFILE_EQ = f'{FOLDER_PATH}{SYSTEM}_{POLY_KEY}_{REPLICA}_EQ.log'

MaxwellBoltzmannDistribution(atoms, temperature_K=T, force_temp=True)
Stationary(atoms)
ZeroRotation(atoms)
with Andersen(atoms, DT * fs, temperature_K=T, andersen_prob=1e-1, 
              logfile=NAME_LOGFILE_EQ, loginterval=READOUT_FREQUENCY) as dyn:
    for _ in dyn.irun(N_STEPS_EQ):   
        #atoms = wrap_molecule_cog(atoms)
        if dyn.nsteps % BAROSTAT_FREQUENCY == 0:
            atoms = barostat(atoms)

with Andersen(atoms, DT * fs, temperature_K=T, andersen_prob=1e-1, 
              trajectory=NAME_TRAJFILE, logfile=NAME_LOGFILE, loginterval=READOUT_FREQUENCY) as dyn:
    for _ in dyn.irun(N_STEPS):   
        #atoms = wrap_molecule_cog(atoms)
        if dyn.nsteps % BAROSTAT_FREQUENCY == 0:
            atoms = barostat(atoms)
