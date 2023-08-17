import os

import numpy as np
import tensorflow as tf

from TopoGNN import TopoGNN
from Utilities import ff_module


def shifted_softplus(x):
    return tf.nn.softplus(x) + 1e-3

ALPHA0 = np.array([0.666793, 1.76, 1.10, 0.802, 0.556, 3.630, 2.90, 2.18])
# H, C, N, O, F, P, S, Cl

class Alpha(tf.keras.Model):
    def __init__(self, n_units=64, n_layers=2, n_steps=2, node_size=64, edge_size=64, activation=tf.nn.swish, load_weights=True):
        super(Alpha, self).__init__()   
        self.n_steps = n_steps
        self.n_units = n_units
        self.n_layers = n_layers        
        self.node_size = node_size
        self.edge_size = edge_size        
        self.activation = activation
                
        self.alpha = ff_module(self.node_size, self.n_layers, activation=self.activation, output_size=1, final_activation=shifted_softplus)   
        self.topo_gnn = TopoGNN(n_units=self.n_units, node_size=self.node_size, edge_size=self.edge_size, n_layers=1, n_steps=self.n_steps)
        
        module_dir = os.path.dirname(__file__)
        weights_dir = os.path.join(module_dir, 'weights_models')
        if load_weights:
            if n_steps == 2:                
                file_path = os.path.join(weights_dir, 'ALPHA2R')
                self.load_weights(file_path)
                print('loaded ALPHA2R')
            if n_steps == 3:
                file_path = os.path.join(weights_dir, 'ALPHA3R')
                self.load_weights(file_path)
                print('loaded ALPHA3R')
                
    def __call__(self, graph, ratio=None):
        alpha0 = ALPHA0[np.argmax(graph.nodes, axis=-1)][:, None]
        alphas = self.alpha(self.topo_gnn(graph).nodes) * alpha0
        if ratio is None:
            return alphas
        return alphas * ratio