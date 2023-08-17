import os
import sys
import time

sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

from ANA2B import ANA2B 
from Utilities import MARE, loss_weight, show_results, validate

def get_batch(key):
    targets = DATA[key]['energies']
    distance_matrices = DATA[key]['distance_matrices']
    elements_1, elements_2 = DATA[key]['elements']
    coords_1, coords_2 = DATA[key]['coordinates']
    graph_1, graph_2 = DATA[key]['graphs']
    multipoles = MULTIPOLES[key]
    die_term = LR_TERMS[key][0] + LR_TERMS[key][1] + D3_TERMS[key]
    induced_mus = INDUCED_MUS[key]
    return targets, distance_matrices, coords_1, coords_2, multipoles,\
                    induced_mus, die_term, graph_1, graph_2

FUNCTIONAL = 'PBE0'
SMEAR = int(sys.argv[2])
FOLDER = '../../data/'
DATA = np.load(f'{FOLDER}DES5M.npy', allow_pickle=True).item()
D3_TERMS = np.load(f'{FOLDER}DES5M_D3_{FUNCTIONAL}.npy', allow_pickle=True).item()
LR_TERMS = np.load(f'{FOLDER}LR_TERMS_S{SMEAR}2R.npy', allow_pickle=True).item()
INDUCED_MUS = np.load(f'{FOLDER}MU_IND_S{SMEAR}2R.npy', allow_pickle=True).item()
REF_DATA = np.load(f'{FOLDER}test_sets/BENCHMARK_DATA_S{SMEAR}2R_D3{FUNCTIONAL}.npy', allow_pickle=True).item()
MULTIPOLES = np.load(f'{FOLDER}MULTIPOLES_DES5M.npy', allow_pickle=True).item()

KEYS = list(DATA.keys())


N_SAMPLES = 2048
N_EPOCHS = 512
N_UNITS = 64
N_STEPS = 1
N_LAYERS = 2
CUTOFF = float(sys.argv[1])
print(CUTOFF)


mae_energies_test = tf.keras.metrics.MeanAbsoluteError()
mae_energies_train = tf.keras.metrics.MeanAbsoluteError()
mae_energies_test_weighted = tf.keras.metrics.MeanAbsoluteError()
mae_energies_train_weighted = tf.keras.metrics.MeanAbsoluteError()


model = ANA2B(cutoff=CUTOFF, n_units=N_UNITS, n_steps=N_STEPS)
lr_fn = tf.optimizers.schedules.ExponentialDecay(5e-4, int(N_SAMPLES *  N_EPOCHS), 2e-2) # 4e-4, 1e-2
optimizer = tf.keras.optimizers.Adam(lr_fn)
cur_time = str(int(time.time()))
folder_path = f'summaries/{cur_time}_ANA2BGNN{N_STEPS}_{N_LAYERS}_{N_UNITS}_CUTOFF2B{CUTOFF}_{N_SAMPLES}x{N_EPOCHS}_'
folder_string = f'{folder_path}D3{FUNCTIONAL}_FULLMSE_S{SMEAR}2R'
os.mkdir(folder_string)
summary_writer = tf.summary.create_file_writer(folder_string)

with summary_writer.as_default():
    for epoch in range(N_EPOCHS):
        start = time.time()
        sampled_keys = np.random.choice(KEYS, N_SAMPLES, replace=False)
        for key in sampled_keys:
            targets, distance_matrices, coords_1, coords_2, multipoles,\
                induced_mus, die_term, graph_1, graph_2 = get_batch(key)
            N = np.product(distance_matrices.shape[1:])
            with tf.GradientTape() as tape:
                V_terms = model(graph_1, graph_2, coords_1, coords_2, distance_matrices,\
                                multipoles, induced_mus, coords_1.shape[0])
                total = V_terms + die_term
                loss = tf.reduce_mean(tf.math.squared_difference(targets, total))
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, norm = tf.clip_by_global_norm(gradients, 1)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            weights = loss_weight(targets, temperature=300, offset=10)
            mae_energies_train.update_state(targets, total)
            mae_energies_train_weighted.update_state(weights * targets, weights * total)
        print('Epoch {}'.format(epoch))
        print(time.time() - start)
        print('MAE Energy Train [kJ/mol]: {}'.format(mae_energies_train.result()))
        print('MAE Energy Train Weighted [kJ/mol] [kJ/mol]: {}'.format(mae_energies_train_weighted.result()))
        tf.summary.scalar('MAE Energy Train [kJ/mol]', mae_energies_train.result(), step=epoch)
        tf.summary.scalar('MAE Energy Train Weighted [kJ/mol]', mae_energies_train_weighted.result(), step=epoch)
        mae_energies_train.reset_states()
        mae_energies_train_weighted.reset_states()
        for ref_key in ['S66x8', 'S7L_CC']:
            energy_target, energy_predicted = validate(model, ref_key, REF_DATA)
            show_results(energy_target, energy_predicted, ref_key, show_plot=False)
            diffs_total = energy_predicted - energy_target
            if ref_key == 'S66x8':
                ref_key = 'S66'
                mae_S66 = np.mean(np.abs(diffs_total))
            if ref_key == 'S7L_CC':
                mae_S7L = np.mean(np.abs(diffs_total))
            tf.summary.scalar(f'MAE Energy {ref_key} [kJ/mol]', np.mean(np.abs(diffs_total)), step=epoch)
            tf.summary.scalar(f'ME Energy {ref_key} [kJ/mol]', np.mean(diffs_total), step=epoch)
            tf.summary.scalar(f'MAX Energy {ref_key} [kJ/mol]', np.amax(np.abs(diffs_total)), step=epoch)
        save_path = f'weights_alphagnn_S{SMEAR}_2R/ANA2BGNN{N_STEPS}_{N_LAYERS}_D3{FUNCTIONAL}_FULLMSE_CUTOFF2B{CUTOFF}_S{SMEAR}2R_E{epoch}'
        if mae_S66 < 0.85 and mae_S7L < 3.5:
            model.save_weights(save_path)
        if epoch == 350:
            model.save_weights(save_path)
model.save_weights(save_path)
