import graph_nets as gn
import tensorflow as tf

from Utilities import ff_module

class TopoGNN(tf.keras.Model):
    def __init__(self, n_units=64, node_size=64, edge_size=64, n_layers=1, n_steps=1, activation=tf.nn.swish):
        super(TopoGNN, self).__init__()   
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.node_size = node_size
        self.edge_size = edge_size
        self.activation = activation
        
        self.embedding = gn.modules.GraphIndependent(
                            edge_model_fn=lambda: ff_module(self.edge_size, self.n_layers, activation=self.activation),
                            node_model_fn=lambda: ff_module(self.node_size, self.n_layers, activation=self.activation),     
                        )   

        self.gns = [gn.modules.InteractionNetwork( 
                        edge_model_fn=lambda: ff_module(self.edge_size, self.n_layers, activation=self.activation),
                        node_model_fn=lambda: ff_module(self.node_size, self.n_layers, activation=self.activation)) for _ in range(self.n_steps)]
        
    def __call__(self, graph):
        return self._update_graph(graph)
        
    def _update_graph(self, graph):
        nodes0 = tf.identity(graph.nodes)
        graph = self.embedding(graph)
        for layer in self.gns: 
            graph = layer(graph)
            graph = graph.replace(nodes=tf.concat((nodes0, graph.nodes), axis=-1))
        return graph