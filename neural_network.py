from layer import Layer
from dense import Dense
from loss import Loss
from activation import Activation, ActivationFuncs
import numpy as np
import json

class NeuralNetwork:
    def __init__(self):
        """
        intialize a neural network object
        """
        self.layers = []
        self.loss = Loss()

    def add(self, new_layer: Layer):
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
            print(input_data)
        return input_data
    
    def train(self, x_train: np.array, y_train: np.array, epochs: int, learning_rate: float): # choo choo
        """
        Trains the network on the given batch
        """

        for _ in range(epochs):
            # predict
            results = self.predict(x_train)

            # calc loss
            print("Error: " + self.loss.forward(results, y_train))

            # propagate backwards
            current_gradient = self.loss.backward()

            for layer in self.layers[::-1]:
                current_gradient = layer.backward(current_gradient, learning_rate)

    def save(self, filepath: str): # TODO make save work
        """
        Save a neural network to a file
        """
        with open(filepath + '.json', 'w') as save_file:
            json.dump(save_file, self) 



