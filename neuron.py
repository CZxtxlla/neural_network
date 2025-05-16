# simple neuron to be used in a neural netowrk

import numpy as np


class Neuron:
    def __init__(self, num_inputs: int):
        """
        Initialize the neuron with a given number of inputs.
        Each input has an associated weight initialized randomly.
        """
        self.weights = np.random.rand(num_inputs) # Initialized to random weights for each input
        self.bias = np.random.rand(1) # Initialized to a random bias
        self.output = None

    def activate(self, inputs: np.ndarray) -> float:
        """
        Compute the output of the neuron using the sigmoid activation function.
        """
        z = np.dot(self.weights, inputs) + self.bias # Weighted sum of inputs + bias
        self.output = self.sigmoid(z)
        return self.output
    
    def sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))