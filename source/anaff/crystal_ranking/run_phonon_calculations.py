import os
import sys

sys.path.append('../..')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ase.build import make_supercell
from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, CrystalThermo
from ase.phonons import Phonons
from HybridFF import HybridFF
from Utilities import show_results, write_xyz, validate
from Constants import EV_TO_KJ, H_TO_KJ, BOHR_TO_ANGSTROM


SYSTEM = sys.argv[1]
POLY_KEY = sys.argv[2]

DATA = np.load('../../../data/test_sets/DATA_CSP_ranking.npy', allow_pickle=True).item()
MIN_STRUCTURES = np.load(f'data/STRUCTURES_{SYSTEM}.npy', allow_pickle=True).item()
hybrid_ff = HybridFF(debug=True, pol0=False)

mol_size = len(DATA[SYSTEM][POLY_KEY]['monomer_symbols'])
elements_uc = DATA[SYSTEM][POLY_KEY]['uc_symbols']
coords_uc = MIN_STRUCTURES[POLY_KEY]['coordinates']
cell = MIN_STRUCTURES[POLY_KEY]['cell']
atoms = hybrid_ff.initialize(elements_uc, coords_uc, cell, mol_size)  

ph = Phonons(atoms, atoms.calc, name=f'vibdata_phonon/{SYSTEM}/{POLY_KEY}')
ph.run()
ph.read(acoustic=True)
phonon_energies, phonon_DOS = ph.dos(kpts=(20, 20, 20), npts=2000)#kpts=(10, 10, 10), npts=1000,delta=5e-4)
RESULTS = {}
RESULTS['phonon_energies'] = phonon_energies
RESULTS['phonon_DOS'] = phonon_DOS
np.save(f'vibdata_phonon/{SYSTEM}/phonons_{POLY_KEY}', RESULTS)