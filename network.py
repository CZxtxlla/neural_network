from neuron import Neuron
import matplotlib.pyplot as plt
import numpy as np

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
            self.edges[to_node].append(from_node)
        else:
            raise ValueError("Both nodes must exist in the network.")
        
    def topological_sort(self):
        visited = set()
        order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in self.edges[node]:
                dfs(neighbor)
            order.append(node)

        for node in self.nodes:
            dfs(node)


        return list(order)
        
    def forward(self, inputs: dict) -> dict:
        """
        Perform a forward pass through the network.
        """
        outputs = {}
        #print(self.topological_sort())
        for node_id in self.topological_sort():
            neuron = self.nodes[node_id]
            if node_id in inputs:
                outputs[node_id] = neuron.activate(inputs[node_id])
                #print(f"Node {node_id} activated with input {inputs[node_id]}: output {outputs[node_id]}")
            else:
                # If the node is not an input node, compute its output based on its edges
                #edge_inputs = [outputs[edge] for edge in self.edges[node_id]]
                edge_inputs = [outputs[edge].item() if isinstance(outputs[edge], np.ndarray) else outputs[edge] for edge in self.edges[node_id]]
                outputs[node_id] = neuron.activate(edge_inputs)
                #print(f"Node {node_id} activated with inputs {edge_inputs}: output {outputs[node_id]}")
        return outputs
    
    def backward(self, target: dict, learning_rate: float):
        """
        Perform a backward pass through the network.
        """
        for node_id, neuron in self.nodes.items():
            if node_id in target:
                neuron.backpropogation(target[node_id], learning_rate)


if __name__ == "__main__":
    # Example usage
    network = Network()
    network.add_node("input1", 0)
    network.add_node("input2", 0)
    network.add_node("hidden1", 2)
    network.add_node("hidden2", 2)
    network.add_node("output", 2)

    network.add_edge("input1", "hidden1")
    network.add_edge("input2", "hidden1")
    network.add_edge("input1", "hidden2")
    network.add_edge("input2", "hidden2")
    network.add_edge("hidden1", "output")
    network.add_edge("hidden2", "output")
    """
    inputs = {"input1": 0.5, "input2": 0.8}
    target = {"output": 1.0}
    print("Inputs:", inputs)
    
    outputs = network.forward(inputs)
    print("Outputs:", outputs)
    
    network.backward(target, learning_rate=0.01)
    """
    losses = []
    iterations = 10000

    for i in range(iterations):
        # train for xor
        inputs = {"input1": np.random.randint(0,2), "input2": np.random.randint(0,2)}
        target = {"output": int(inputs["input1"] != inputs["input2"])}
        outputs = network.forward(inputs)
        loss = sum((target[node] - outputs[node]) ** 2 for node in target)
        losses.append(loss)
        print(f"Iteration {i+1}/{iterations}, Loss: {loss}")

        network.backward(target, learning_rate=0.01)

    for x in [0, 1]:
        for y in [0, 1]:
            output = network.forward({"input1": x, "input2": y})
            print(f"{x} XOR {y} => {output['output']}")

    # We plot losses to see how our network is doing
    plt.plot(losses)
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss value")
    plt.title("Loss over epochs")
    plt.show()