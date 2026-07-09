from layer import Layer
from loss import Loss
import numpy as np

class Dense(Layer):
    def __init__(self, size: int, next_size: int):
        """
        Creates a new dense layer.
        """
        self.size = size
        self.next_size = next_size
        self.weights = np.ones((next_size, size))
        self.biases = np.zeros(size)
        self.input = np.zeros(size)

    def forward(self, input_arr: np.array) -> np.array:
        """
        Computes the output of the layer
        """
        # if self.next_size == 0: # if final layer
        #     return input_arr
        
        output_arr = np.zeros((input_arr.shape[0], self.next_size))
        self.input = input_arr
        self.input += self.biases

        for inputs, output in zip(self.input, output_arr): # for every input
            for weights in self.weights: # go over all of the weights to the output nodes
                for input, weight in zip(inputs, weights): # go over each input and its weight to the output node
                    output += input * weight

        return output_arr
    
    def backward(self, input: np.array, output_gradient: float, learning_rate: float) -> float:
        """
        Calculates the gradients for the weights and biases
        """
        
        weights_gradient = np.dot(output_gradient, input.T)
        bias_gradient = output_gradient

        input_gradient = np.dot(output_gradient, self.weights)

        self.weights = self.weights - weights_gradient * learning_rate
        self.biases = self.biases - bias_gradient * learning_rate

        return input_gradient