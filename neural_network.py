from layer import Layer
from dense import Dense
from loss import Loss
from activation import Activation, ActivationFuncs, func_dict
import numpy as np
import json

class NeuralNetwork:
    def __init__(self, loss: Loss):
        """
        intialize a neural network object
        """
        self.layers = []
        self.loss = loss

    def add(self, new_layer: Dense):
        """
        append a new layer to the neural network
        """
        new_layer.biases = np.random.uniform(low=-10.0, high=11.0, size=new_layer.biases.shape) # randomize baises
        new_layer.weights = np.random.uniform(low=-10.0, high=11.0, size=new_layer.weights.shape) # randomize weights

        self.layers.append(new_layer)
        self.layers.append(Activation(ActivationFuncs.sigmoid))

    def predict(self, input_data: np.array) -> float:
        """
        Applies the network on a given input
        """
        for curr_layer in self.layers:
            input_data = curr_layer.forward(input_data)
        return ActivationFuncs.sigmoid(self.layers[-1].input) # Apply sigmoid to final result
    
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

    def save(self, filepath: str):
        """
        save a neural network to a file
        """
        flipped_dict = {value: key for key, value in func_dict.items()} #get a dict of the function as key
        layer_data = []
        for layer in self.layers:
            if type(layer) == Dense:
                bias_list = layer.biases.tolist()
                weight_list = layer.weights.tolist()
                layer_data.append({'biases': bias_list, 'weights': weight_list}) #add lists of weights and biases

        with open(filepath + '.json', 'w') as save_file:
            json.dump(save_file, layer_data)



