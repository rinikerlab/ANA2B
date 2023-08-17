import os

import tensorflow as tf
import numpy as np

from Utilities import ff_module
from HelpersParams import Graph, neutralize, G_matrices, build_poles, reshape_coeffs, build_graph


class AMPParams(tf.keras.Model):
    def __init__(self, node_size=128, activation=tf.nn.swish, num_steps=3, order=2, cutoff=5.0):
        super(AMPParams, self).__init__()
        self.num_steps = num_steps
        self.node_size = node_size
        self.order = order
        self.cutoff = cutoff

        self.embedding_nodes = tf.keras.layers.Dense(units=self.node_size, activation=activation, use_bias=False)
        self.in_message_layers = [ff_module(self.node_size, 2) for _ in range(self.num_steps)]
        self.in_update_layers = [ff_module(self.node_size, 2) for _ in range(self.num_steps)]
        self.eq_message_layers = [ff_module(self.node_size, 2, output_size=self.order + 1, final_activation=None)\
                                                                                              for _ in range(self.num_steps+1)]
        self.ratios = ff_module(self.node_size, 2, output_size=1, final_activation=tf.nn.softplus)
        self.monos = ff_module(self.node_size, 2, output_size=1, final_activation=None)
    
        module_dir = os.path.dirname(__file__)
        weights_dir = os.path.join(module_dir, 'weights_models')
        file_path = os.path.join(weights_dir, 'AMP_PARAMS')
        self.load_weights(file_path)

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, nodes, edges, senders, receivers, n_node, Rx1, Rx2):
        nodes, _, dipos, quads = self._pass_messages(nodes, edges, senders, receivers, Rx1, Rx2, tf.reduce_sum(n_node)) 
        ratios, monos = self.ratios(nodes), self.monos(nodes)
        monos = neutralize(monos, n_node)
        return monos, dipos, quads, ratios
    
    def predict(self, coordinates, elements):
        graph, dmat = build_graph(coordinates, elements, cutoff=self.cutoff)
        graph.monos, graph.dipos, graph.quads, graph.ratios = self(graph.nodes, graph.edges, 
                                                                   graph.senders, graph.receivers, 
                                                                   graph.n_node, graph.Rx1, graph.Rx2)              
        graph = reshape_coeffs(graph, coordinates.shape)
        return graph, dmat
    
    @tf.function(experimental_relax_shapes=True)
    def _pass_messages(self, nodes0, edges, senders, receivers, Rx1, Rx2, n_segments):
        edge_features = edges 
        nodes = self.embedding_nodes(nodes0)
        for eq_message_layer, in_message_layer, in_update_layer in zip(self.eq_message_layers, self.in_message_layers, self.in_update_layers):
            features_ij = tf.concat((tf.gather(nodes, receivers), tf.gather(nodes, senders)), axis=-1)
            coefficients = eq_message_layer(tf.concat((features_ij, edge_features), axis=-1))           
            monos, dipos, quads = build_poles(Rx1, Rx2, receivers, coefficients, n_segments)         
            aniso_feature = G_matrices(monos, dipos, quads, Rx1, Rx2, senders, receivers)
            edge_features = tf.concat((edges, aniso_feature), axis=-1)
            mp_feature_aniso = tf.concat((features_ij, edge_features), axis=-1)
            messages = tf.math.unsorted_segment_sum(in_message_layer(mp_feature_aniso), receivers, num_segments=n_segments)
            nodes += in_update_layer(tf.concat((nodes0, nodes, messages), axis=-1))
        features_ij = tf.concat((tf.gather(nodes, receivers), tf.gather(nodes, senders)), axis=-1)
        coefficients = self.eq_message_layers[-1](tf.concat((features_ij, edge_features), axis=-1))           
        monos, dipos, quads = build_poles(Rx1, Rx2, receivers, coefficients, n_segments)      
        return nodes, monos, dipos, quads
    


