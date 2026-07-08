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
        self.weights = np.zeros((next_size, size))
        self.biases = np.zeros(size)
        self.input = np.zeros(size)

    def forward(self, input_arr: np.array) -> np.array:
        """
        Computes the output of the layer
        """
        if self.next_size == 0: # if final layer
            return input_arr
        
        output_arr = np.zeros((input_arr.shape[0], self.next_size))
        self.input = input_arr
        self.input += self.biases

        for output, weights in zip(output_arr, self.weights):
            for input, weight in zip(self.input, weights):
                output += input * weight

        return output_arr
    
    def backward(self, input: np.array, output_gradient: np.array, learning_rate: float) -> float:
        """
        Calculates the gradients for the weights and biases
        """
        
        weights_gradient = np.dot(input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
    
        input_gradient = np.dot(output_gradient, self.weights.T) 

        self.weights -= weights_gradient * learning_rate
        self.biases -= bias_gradient * learning_rate

        return input_gradient