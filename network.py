from neuron import Neuron
import matplotlib.pyplot as plt
import numpy as np

class Network:
    def __init__(self):
        self.nodes = {} # Dictionary to hold nodes
        self.edges = {} # Dictionary to hold edges
        self.last_outputs = {} # To store the last outputs for backpropagation

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
        
    def generate(self, num_inputs: int, num_hidden):
        """
        Generate a network with a given number of input and hidden nodes.
        """
        for i in range(num_inputs):
            self.add_node(f"input{i+1}", 0)
        
        for i in range(num_hidden):
            self.add_node(f"hidden{i+1}", num_inputs)
        
        self.add_node("output", num_hidden)

        # Connect input nodes to hidden nodes
        for i in range(num_inputs):
            for j in range(num_hidden):
                self.add_edge(f"input{i+1}", f"hidden{j+1}")

        # Connect hidden nodes to output node
        for i in range(num_hidden):
            self.add_edge(f"hidden{i+1}", "output")
        
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


    def sigmoid_derivative(self, x: float) -> float:
        """
        Derivative of the sigmoid function.
        """
        return x * (1 - x)
        
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
        # Store the last outputs for backpropagation
        self.last_outputs = outputs.copy()
        return outputs
    
    def backward(self, target: dict, learning_rate: float):
        """
        Perform a backward pass through the network.
        """
        outputs = self.last_outputs

        y_o = outputs["output"]
        delta_o = (target["output"] - y_o) * self.sigmoid_derivative(y_o)

        # Update the output neuron
        self.nodes["output"].backprop(delta_o, learning_rate)

        # Update the hidden neurons
        for i, h in enumerate(self.edges["output"]):
            y_h = outputs[h]
            w = self.nodes["output"].weights[i]
            delta_h = w * delta_o * self.sigmoid_derivative(y_h)
            self.nodes[h].backprop(delta_h, learning_rate)

    def predict(self, inputs: dict) -> dict:
        """
        Predict the output for given inputs.
        """
        outputs = self.forward(inputs)
        return outputs["output"]
    
    def train(self, patterns: list, iterations: int, learning_rate: float):
        """
        Train the network using the given patterns.
        """
        losses = []
        required_iterations = 0

        for i in range(iterations):
            # decrease learning rate
            learning_rate *= 0.95 if i % 500 == 0 else 1
            # shuffle patterns
            np.random.shuffle(patterns)
            temp_losses = []
            for inputs, target in patterns:
                # train for xor
                outputs = self.forward(inputs)
                
                loss = sum((target[node] - outputs[node]) ** 2 for node in target)
                temp_losses.append(loss)
                losses.append(loss)
                #print(f"Iteration {i+1}/{iterations}, Loss: {loss}, learning_rate: {learning_rate}")
                self.backward(target, learning_rate=learning_rate)
            for x in temp_losses:
                if x > 0.01:
                    required_iterations += 1
                    break
        print(f"Required iterations: {required_iterations}")
        return losses, required_iterations


if __name__ == "__main__":
    average_required = 0
    for i in range(50):

        # Create network
        network = Network()
        network.generate(2, 8)

        # Parameters for training
        iterations = 1000
        learning_rate = 1.0
        patterns = [({"input1":0,"input2":0}, {"output":0}), ({"input1":0,"input2":1}, {"output":1}), ({"input1":1,"input2":0}, {"output":1}), ({"input1":1,"input2":1}, {"output":0})]

        # Train network
        train = network.train(patterns, iterations, learning_rate)
        losses = train[0]
        required_iterations = train[1]
        average_required += required_iterations
    average_required /= 50
    print(f"Average required iterations: {average_required}")

    
    # Test network
    for x in [0, 1]:
        for y in [0, 1]:
            output = network.predict({"input1": x, "input2": y})
            print(f"{x} XOR {y} => {output}")

    # Plot losses to see how network performs
    plt.plot(losses)
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss value")
    plt.title("Loss over epochs")
    plt.show()