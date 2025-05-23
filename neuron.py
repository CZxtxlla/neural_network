# simple neuron to be used in a neural netowrk

import numpy as np


class Neuron:
    def __init__(self, num_inputs: int):
        """
        Initialize the neuron with a given number of inputs.
        Each input has an associated weight initialized randomly.
        """
        #limit = np.sqrt(6 / (num_inputs + 1))
        #self.weights = np.random.uniform(-limit, limit, size=(num_inputs,))
        #self.bias    = 0.0
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.randn() # Initialized to a random bias
        self.output = None
        self.num_inputs = num_inputs
        self.last_inputs = None

    def activate(self, inputs: np.ndarray) -> float:
        """
        Compute the output of the neuron using the sigmoid activation function.
        """

        inputs = np.array(inputs, dtype=float)

        if len(self.weights) == 0:
            self.output = inputs
            return self.output
        
        self.last_inputs = inputs # Store the last inputs for backpropagation
        
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
    
    def linear(self, x: float) -> float:
        """
        Linear activation function.
        """
        return x
    
    def linear_derivative(self, x: float) -> float:
        """
        Derivative of the linear function.
        """
        return 1.0
    
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

    def backprop(self, delta: float, learning_rate: float):
        """
        Update the weights and bias using backpropagation.
        """
        self.weights += learning_rate * delta * self.last_inputs
        self.bias += learning_rate * delta

