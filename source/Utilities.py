import numpy as np
import tensorflow as tf

E_TO_Z = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'CL': 17,
}

def to_Z(elements):
    return np.array([E_TO_Z[e] for e in elements])

S = lambda x, y: tf.reduce_sum(x * y, axis=-1, keepdims=True)
A = lambda x, k=-1: tf.expand_dims(x, axis=k)

def ff_module(node_size=128, n_layers=2, activation=tf.nn.swish, output_size=None, with_bias=True, final_activation=None, final_bias=False, bias=0):
    modules = []
    for _ in range(n_layers):
        modules.append(tf.keras.layers.Dense(
                                                units=node_size,
                                                activation=activation,
                                                use_bias=with_bias,
                                            ))
    if output_size is not None:
        modules.append(tf.keras.layers.Dense(
                                                units=output_size,
                                                activation=final_activation,
                                                use_bias=final_bias,
                                                bias_initializer=tf.keras.initializers.Constant(bias)           
                                            ))
    return tf.keras.Sequential(modules)

def cdist_tf_batch(A, B):
    na = tf.reduce_sum(tf.square(A), axis=-1, keepdims=True)
    nb = tf.transpose(tf.reduce_sum(tf.square(B), axis=-1, keepdims=True), [0, 2, 1])
    dmat = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return tf.linalg.set_diag(dmat, tf.zeros(dmat.shape[:2], dtype=A.dtype)) 

def gaussian(x, alpha=0.1):
    return tf.exp(-alpha * tf.square(x))

@tf.function(experimental_relax_shapes=True)
def detrace(Rx2):
    return tf.linalg.set_diag(Rx2, tf.linalg.diag_part(Rx2) - tf.expand_dims((tf.linalg.trace(Rx2) / 3), axis=-1))

@tf.function(experimental_relax_shapes=True)
def build_Rx2(Rx1):
    return detrace(A(Rx1, -2) * A(Rx1, -1))

def switch(R1, r_switch=4.0, r_cutoff=5.0):
    X = (R1 - r_switch) / (r_cutoff - r_switch)
    X3 = tf.math.pow(X, 3)
    X4 = X3 * X
    X5 = X4 * X
    return tf.clip_by_value(1 - 6 * X5 + 15 * X4 - 10 * X3, 0, 1)

NUM_KERNELS = 20
FREQUENCIES = np.pi * tf.range(1, NUM_KERNELS + 1, dtype=np.float32)[None]
#@tf.function(experimental_relax_shapes=True)
def envelope(R1):
    p = 5 + 1
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    env_val = 1.0 / R1 + a * R1 ** (p - 1) + b * R1 ** p + c * R1 ** (p + 1)
    return tf.where(R1 < 1, env_val, 0)

#@tf.function(experimental_relax_shapes=True)
def build_sin_kernel(R1, cutoff):
    d_scaled = R1 * (1 / cutoff)
    d_cutoff = envelope(d_scaled)
    return d_cutoff * tf.sin(FREQUENCIES * d_scaled)

def MARE(targets, predictions, shift=1):
    targets_rel = targets - np.amin(targets) + shift
    return tf.reduce_mean((tf.abs((targets - predictions) / targets_rel) * 100))

def loss_weight(target_energy, temperature=300, offset=0):
    scaling = 0.0083144 * temperature
    target = np.copy(target_energy)
    min_index = np.argmin(target)
    target -= target[min_index]
    target -= offset
    weights = np.minimum(1.0, np.exp(-np.divide(target, scaling)))    
    weights[min_index:] = 1
    return weights

def plot_progress(targets, totals, distance_matrices, weights=None, ymin=None, ymax=None):
    import matplotlib.pyplot as plt
    plt.figure(0, figsize=(20, 10), dpi=100)
    ax = plt.gca()
    if weights is None:
        weights = 1
    plt.text(.7, .92, 'MAE: {:6.2f} kJ/mol'.format(np.mean(np.abs(targets - totals) * weights)), 
             horizontalalignment='center', verticalalignment='center', transform= ax.transAxes)
    ds = [np.amin(dm) for dm in distance_matrices]
    #print(len(ds), len(targets), len(total))
    plt.scatter(ds, targets, label='Reference', s=10)      
    plt.plot(ds, totals, color='orange', label='ML-FF') 
    plt.scatter(ds, totals, color='blue', label='ML-FF', s=5)  
    ax.set_xlabel('Min Distance [A]')
    ax.set_ylabel('Interaction Energy [kJ/mol]')
    ax.legend()  
    if ymax is not None:
        ax.set_ylim(ymin, ymax)
    plt.show()
    
def validate(model, db_key, REF_DATA):
    references, predictions = [], []
    for system_key in REF_DATA[db_key]:
        graph_1, graph_2 = REF_DATA[db_key][system_key]['graphs']
        coords_1, coords_2 = REF_DATA[db_key][system_key]['coordinates']
        #coords_1, coords_2 = tf.convert_to_tensor(coords_1), tf.convert_to_tensor(coords_2)
        die_term = REF_DATA[db_key][system_key]['die_term']
        distance_matrices = REF_DATA[db_key][system_key]['distance_matrix']
        induced_mus = REF_DATA[db_key][system_key]['mu_ind']
        multipoles = REF_DATA[db_key][system_key]['multipoles']
        V_terms = model(graph_1, graph_2, coords_1, coords_2, distance_matrices, multipoles, induced_mus, coords_1.shape[0])
        predictions.append(die_term + V_terms)
        references.append(REF_DATA[db_key][system_key]['ref_energy'])
    return np.hstack(references), np.hstack(predictions)

def show_results(target_total=None, 
                 pred_total=None, 
                 dataset_name='', 
                 names=None, 
                 print_out=True, 
                 show_plot=False,
                 show_chem_acc=False,
                 show_mae=True,
                 cutoff_txt=0,
                 n_digits=2,
                 s=15,
                 unit='kJ/mol'):
    import matplotlib.pyplot as plt
    diffs_total = target_total - pred_total
    mae = np.round(np.mean(np.abs(diffs_total)), n_digits)
    me = np.round(np.mean(diffs_total), n_digits)
    rmse = np.round(np.sqrt(np.mean(np.square(diffs_total))), n_digits)
    if print_out:
        print(dataset_name)
        print(f'Total - {len(diffs_total)}')
        print(f'MAE:  {mae:{4}.{n_digits}f}')
        print(f'RMSE: {rmse:{4}.{n_digits}f}')
        print(f'ME:   {me:{4}.{n_digits}f}')   
    fig = plt.figure(0, figsize=(6, 6), dpi=400)
    if show_plot:        
        plt.scatter(target_total, pred_total, s=s, color='#3E9BBD', edgecolors='black', linewidths=.1)
        ax = plt.gca()
        min_ = min(np.amin(target_total), np.amin(pred_total)) 
        max_ = max(np.amax(target_total), np.amax(pred_total))
        max_spread = max_ - min_
        shift = 0.1 * max_spread
        min_, max_ = min_ - shift, max_ + shift
        xs = np.linspace(min_, max_, 4)
        ys = np.linspace(min_, max_, 4)
        plt.plot(xs, ys, color='black', linewidth=1)
        if show_chem_acc:
            plt.plot(xs, ys + 4.184, linewidth=.4, color='grey')
            plt.plot(xs, ys - 4.184, linewidth=.4, color='grey')
        ax.set_xlabel(f'Reference [{unit}]', labelpad=10)
        ax.set_ylabel(f'Prediction [{unit}]', labelpad=10)
        ax.set_title(dataset_name)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlim(min_, max_)
        ax.set_ylim(min_, max_)
        if show_mae:
            ax.text(min_ + shift, max_ - shift, f'MAE: {mae:{4}.{n_digits}f} {unit}')
        if names is not None:
            for i, txt in enumerate(names):
                if abs(pred_total[i] - target_total[i]) > cutoff_txt:
                    ax.annotate(txt, ( pred_total[i] + 0.1 * shift, target_total[i] + 0.1 * shift), fontsize=10 )
    return mae, me, rmse, fig

def write_xyz(coords, symbols, file_name='test.xyz'):
    num_atoms = len(symbols)
    assert len(coords) == num_atoms    
    with open(file_name, 'w') as file:
        file.write(str(num_atoms) + '\n')
        file.write('\n')
        for ida in range(num_atoms):
            file.write(symbols[ida] + ' ' + str(coords[ida][0]) + ' ' + str(coords[ida][1]) + ' ' + str(coords[ida][2]) + '\n')
    return file_name

def reshape_coeffs(multipoles, batch_size):
    monos = tf.stack(tf.split(multipoles[0], batch_size))
    dipos = tf.stack(tf.split(multipoles[1], batch_size))
    quads = tf.stack(tf.split(multipoles[2], batch_size))
    ratios = tf.stack(tf.split(multipoles[3], batch_size))  
    return (monos, dipos, quads, ratios)
