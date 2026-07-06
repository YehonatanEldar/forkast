from neural_network import NeuralNetwork
from dense import Dense
from activation import Activation, ActivationFuncs
from loss import Loss

import pandas as pd

import numpy as np
import struct

# Constants
TARGET_NUM = 3
TRAINING_PORTION = 0.8
BATCH_SIZE = 100
EPOCHS = 10
LEARNING_RATE = 0.001

LAYER_SIZES = [784, 128, 64, 1, 0]

# Splitting
slice_idx = int(len(images_flat)*TRAINING_PORTION)
x_train, x_test = images_flat[:slice_idx], images_flat[slice_idx:]
y_train, y_test = new_labels[:slice_idx], new_labels[slice_idx:]

# Batching
x_batches = []
y_batches = []
for i in range(0, len(x_train), BATCH_SIZE):
    x_batches.append(x_train[i:min(i + BATCH_SIZE, len(x_train) - 1)])

for i in range(0, len(y_train), BATCH_SIZE):
    y_batches.append(y_train[i:min(i + BATCH_SIZE, len(y_train) - 1)])

# Create network
network = NeuralNetwork()

for i in range(len(LAYER_SIZES) - 1):
    dense = Dense(LAYER_SIZES[i], LAYER_SIZES[i + 1])
    activation = Activation(ActivationFuncs.sigmoid)

    dense.biases = np.random.uniform(low=-10.0, high=11.0, size=dense.biases.shape) # randomize baises
    dense.weights = np.random.uniform(low=-10.0, high=11.0, size=dense.weights.shape) # randomize weights

    network.add(dense)
    network.add(activation)

# Training
num_batch = 1
for x_batch, y_batch in zip(x_batches, y_batches):
    print("Batch: " + str(num_batch))
    network.train(x_batch, y_batch, EPOCHS, LEARNING_RATE)
    num_batch += 1

# Testing
results = np.fromiter(map(network.predict, x_test))
mean_error = network.loss.forward(results, y_test)
print("Mean Error: " + mean_error)