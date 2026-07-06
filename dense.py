from layer import Layer
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
        output_arr = np.zeros(self.weights.next_size)
        self.input = input_arr
        self.input += self.biases

        for output, weights in zip(output_arr, self.weights):
            for input, weight in zip(self.input, weights):
                output += input * weight

        return output_arr
    
    def backward(self, output_gradient: np.array, learning_rate: float) -> float:
        """
        Calculates the gradients for the weights and biases
        """

        # Biases
        bias_gradient = sum(output_gradient)
        self.biases += -bias_gradient * learning_rate

        # Weights
        weights_gradient = self.input * output_gradient
        self.weights += -weights_gradient * learning_rate

        # Calculate Loss
        

