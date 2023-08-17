import os
import sys
import shutil

sys.path.append('../..')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import ase
from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.vibrations import Vibrations
from ase_harmonic import HarmonicForceField, HarmonicCalculator
from HybridFF import HybridFF
from Utilities import show_results, write_xyz, validate
from Constants import EV_TO_KJ, H_TO_KJ, BOHR_TO_ANGSTROM

from ase.calculators.mixing import MixedCalculator
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.md.andersen import Andersen
from ase.units import fs

SYSTEM = sys.argv[1]
POLY_KEY = sys.argv[2]
T = float(sys.argv[3])
N_STEPS = int(sys.argv[4])
READOUT_FREQUENCY = int(sys.argv[5])
DT = 0.5
N_STEPS_EQ = 1000
N_STEPS_EQ_AH = 200
OUTPUT_FOLDER = 'data_ah'
N_LAMBDA = 11
lambdas = np.linspace(0, 1, N_LAMBDA)

print(SYSTEM, POLY_KEY, T, N_STEPS, READOUT_FREQUENCY)

DATA = np.load('../../../data/test_sets/DATA_CSP_ranking.npy', allow_pickle=True).item()
THERMAL_STRUCTURES = np.load(f'data/MINIMIZED_THERMAL_STRUCTURES_{SYSTEM}.npy', allow_pickle=True).item()
hybrid_ff = HybridFF(debug=False, pol0=False)
coords_uc = THERMAL_STRUCTURES[POLY_KEY]['coordinates'] 
cells = THERMAL_STRUCTURES[POLY_KEY]['cell']
elements_uc = THERMAL_STRUCTURES[POLY_KEY]['uc_symbols'] 
mol_size = len(DATA[SYSTEM][POLY_KEY]['monomer_symbols'])
atoms = hybrid_ff.initialize(elements_uc, coords_uc, cells, mol_size) 
ah_calc = hybrid_ff
optimizer = LBFGS(atoms) # BFGS
optimizer.run(fmax=1 / EV_TO_KJ)

vib = Vibrations(atoms, name=os.path.join(OUTPUT_FOLDER, SYSTEM, POLY_KEY, 'VIB_CALC'))
vib.run()
vib_data = vib.get_vibrations()
hessian_x = vib_data.get_hessian_2d()
ref_atoms = atoms.copy()
ref_energy = ah_calc.get_potential_energy(ref_atoms)

hff = HarmonicForceField(ref_atoms=ref_atoms, ref_energy=ref_energy, hessian_x=hessian_x)
hf_calc = HarmonicCalculator(hff)


FOLDER_PATH = os.path.join(OUTPUT_FOLDER, SYSTEM)
try:
    os.mkdir(FOLDER_PATH)
except:
    pass
np.save(os.path.join(FOLDER_PATH, f'{SYSTEM}_{POLY_KEY}_vib_data'), vib_data)
NAME_LOGFILE = os.path.join(FOLDER_PATH, f'{SYSTEM}_{POLY_KEY}.log')
NAME_TRAJFILE = os.path.join(FOLDER_PATH, f'{SYSTEM}_{POLY_KEY}.traj')
NAME_LOGFILE_EQ = os.path.join(FOLDER_PATH, f'{SYSTEM}_{POLY_KEY}_EQ.log')
MaxwellBoltzmannDistribution(atoms, temperature_K=T, force_temp=True)
Stationary(atoms)
ZeroRotation(atoms)
with Andersen(atoms, DT * fs, temperature_K=T, andersen_prob=1e-1, 
              logfile=NAME_LOGFILE_EQ, loginterval=READOUT_FREQUENCY) as dyn:
    for _ in dyn.irun(N_STEPS_EQ):   
        if dyn.nsteps % 10 == 0:
            pass

ediffs = {}
for lamb in lambdas:
    print(lamb)
    ediffs[lamb] = []
    calc_linearCombi = MixedCalculator(hf_calc, ah_calc, 1 - lamb, lamb)
    atoms.calc = calc_linearCombi
    with Andersen(atoms, DT * fs, temperature_K=T, andersen_prob=1e-1) as dyn:
        for _ in dyn.irun(N_STEPS_EQ_AH): 
            pass
    with Andersen(atoms, DT * fs, temperature_K=T, andersen_prob=1e-1,
                 logfile=NAME_LOGFILE, trajectory=NAME_TRAJFILE, loginterval=READOUT_FREQUENCY) as dyn:
        for _ in dyn.irun(N_STEPS): 
            e0, e1 = calc_linearCombi.get_energy_contributions(atoms)
            ediffs[lamb].append(float(e1) - float(e0))
        ediffs[lamb] = np.mean(ediffs[lamb])    
    print(ediffs[lamb])
dA = np.trapz([ediffs[lamb] for lamb in lambdas], x=lambdas)  # anharm. corr.
print(dA * EV_TO_KJ)
np.save(os.path.join(FOLDER_PATH, f'{SYSTEM}_{POLY_KEY}_ediffs'), ediffs)





