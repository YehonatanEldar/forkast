import json

import numpy as np

from activation import Activation, ActivationFuncs
from dense import Dense
from layer import Layer
from loss import Loss


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
        self.layers.append(Activation(ActivationFuncs.sigmoid))

    def predict(self, input_data: np.array) -> float:
        """
        Applies the network on a given input
        """
        # TODO: this function somehow returns one value
        for curr_layer in self.layers:
            input_data = curr_layer.forward(input_data)
        return input_data
    
    def train(self, x_train: np.array, y_train: np.array, epochs: int, learning_rate: float): # choo choo
        """
        Trains the network on the given batch
        """
        # x_train = [sub for sub in x_train if len(sub) > 0]

        for i in range(epochs):
            print(f"    Epoch: #{i + 1}")
            # predict
            results = self.predict(x_train)
            # calc loss
            current_gradient = self.loss.backward(results, y_train)

            print(f"Error: {self.loss.forward(results, y_train)}")

            # propagate backwards
            for layer in self.layers[:-1][::-1]: # iterate backwards
                current_gradient = layer.backward(results, np.mean(current_gradient, axis=0).squeeze(), learning_rate)

    def save(self, filepath: str): # TODO make save work
        """
        Save a neural network to a file
        """
        layer_data = []
        for layer in self.layers:
            if type(layer) == Dense:
                bias_list = layer.biases.tolist()
                weight_list = layer.weights.tolist()
                layer_data.append({'biases': bias_list, 'weights': weight_list}) #add lists of weights and biases

        with open(filepath + '.json', 'w') as save_file:
            json.dump(save_file, layer_data)




            json.dump(save_file, self) 