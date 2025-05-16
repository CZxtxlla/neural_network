from neuron import Neuron

class Network:
    def __init__(self):
        self.nodes = {} # Dictionary to hold nodes
        self.edges = {} # Dictionary to hold edges

    def add_node(self, node_id: str, num_inputs: int):
        """
        Add a node to the network.
        """
        self.nodes[node_id] = Neuron(num_inputs)
        self.edges[node_id] = []
    
    def add_edge(self, from_node: str, to_node: str):
        """
        Add an edge from one node to another.
        """
        if from_node in self.nodes and to_node in self.nodes:
            self.edges[from_node].append(to_node)
        else:
            raise ValueError("Both nodes must exist in the network.")