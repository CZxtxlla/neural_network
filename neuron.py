# simple neuron to be used in a neural netowrk

import numpy as np


class Neuron:
    def __init__(self, num_inputs: int):
        """
        Initialize the neuron with a given number of inputs.
        Each input has an associated weight initialized randomly.
        """
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1) # Initialized to a random bias
        self.output = None
        self.num_inputs = num_inputs

    def activate(self, inputs: np.ndarray) -> float:
        """
        Compute the output of the neuron using the sigmoid activation function.
        """

        if len(self.weights) == 0:
            self.output = inputs
            return self.output
        
        #inputs = np.array(inputs).flatten() # Ensure inputs are a 1D array
    
        #print(f"Neuron weights: {self.weights}, inputs: {inputs}, bias: {self.bias}")
        #print(self.num_inputs)
        z = np.dot(self.weights, inputs) + self.bias # Weighted sum of inputs + bias
        self.output = self.sigmoid(z)
        #print(f"Neuron activated with inputs {inputs}: weighted sum {z}, output {self.output}")
        return self.output.item()
    
    def sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: float) -> float:
        """
        Derivative of the sigmoid function.
        """
        return x * (1 - x)
    
    def backpropogation(self, target: float, learning_rate: float):
        """
        Update the weights and bias using backpropagation.
        """
        error = target - self.output 
        gradient = self.sigmoid_derivative(self.output)
        
        # Update weights
        for i in range(len(self.weights)): 
            self.weights[i] += learning_rate * error * gradient
        
        # Update bias
        self.bias += learning_rate * error * gradient

