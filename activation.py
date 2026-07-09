from layer import Layer
from enum import Enum
import numpy as np

class ActivationFuncs(Enum):
    """
    enum of possible functions
    """
    relu = 'relu'
    sigmoid = 'sigmoid'

class Activation(Layer):
    
    def __init__(self, activation_func: ActivationFuncs):
        """
        creates a new activation object
        """

        self.activation_func = Activation.func_dict[activation_func.value]
        self.derivative_func = Activation.derivative_dict[activation_func.name]
        self.input = None

    def forward(self, input: np.array) -> np.array:
        """
        pass all activations in the input vector through the activation function
        """

        self.input = input
        output = self.activation_func(self.input)
        return output
    
    def backward(self, input: np.array, output_gradient: float, learning_rate: float) -> np.array:
        """
        makes the backward pass and changes the values, returns the new gradient
        """
        next_gradient = -output_gradient * self.derivative_func(input.squeeze())
        return next_gradient