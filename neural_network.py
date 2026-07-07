from layer import Layer
from dense import Dense
from loss import Loss
from activation import Activation, ActivationFuncs
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
        layer_data = []
        for layer in self.layers:
            if type(layer) == Dense:
                bias_list = layer.biases.tolist()
                weight_list = layer.weights.tolist()
                layer_data.append({'biases': bias_list, 'weights': weight_list}) #add lists of weights and biases

        with open(filepath + '.json', 'w') as save_file:
            json.dump(save_file, layer_data)

    def load(self, savefile: str):
        """
        load neural network from file
        """
        layer_data = []
        try:
            with open(savefile, 'r') as file:
                layer_data = json.load(file)
        except Exception as e:
            print('Error while opening file:', e)
            return # close load after fail
        
        for layer in layer_data:
            # get length of nodes on curr and next layer
            next_size = len(layer['weights'][0])
            size = len(layer['biases'])

            new_layer = Dense(size, next_size) 
            self.add(new_layer) #add randomized layer

            weights = np.array(layer['weights']) #get biases
            biases = np.array(layer['biases']) #get weights

            self.layers[-1].weights = weights
            self.layers[-1].biases = biases
            self.layers[-1].size = size
            self.layers[-1].size = next_size

            self.add(Activation(ActivationFuncs.sigmoid)) #add sigmoid after every layer


