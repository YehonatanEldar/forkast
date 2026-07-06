from layer import Layer
from dense import Dense
from activation import Activation, ActivationFuncs
import numpy as np

class NeuralNetwork:
    def __init__(self, loss: Layer):
        """
        intialize a neural network object
        """
        self.layers = []
        self.loss = loss

    def add(self, new_layer: Dense):
        """
        append a new layer to the neural network
        """
        self.layers.append(new_layer)
        self.layers.append(Activation(ActivationFuncs.relu))

    def predict(self, input_data: np.array) -> float:
        """
        Applies the network on a given input
        """
        for curr_layer in self.layers:
            input_data = curr_layer.forward(input_data)
        return input_data



