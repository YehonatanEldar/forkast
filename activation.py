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
        func_dict = {'relu': lambda x: np.maximum(x, 0),
                     'sigmoid': lambda x: 1/(1+ np.exp(-x))}
        derivative_dict = {'relu': lambda x: 1 if x > 0 else 0,
                           'sigmoid': lambda x: func_dict['sigmoid'](x) * (1 - func_dict['sigmoid'])(x)}

        self.activation_func = func_dict[activation_func.value]
        self.derivative_func = derivative_dict[activation_func.name]
        self.input = None

    def forward(self, input: np.array) -> np.array:
        """
        pass all activations in the input vector through the activation function
        """
        self.input = input
        output = self.activation_func(self.input)
        return output