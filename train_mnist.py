from neural_network import NeuralNetwork
from dense import Dense
from activation import Activation, ActivationFuncs
from loss import Loss
from pathlib import Path
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

DATASET_PATH = 'data/mnist_dataset.csv'

def load_mnist_to_file(images_path, labels_path, file_path, target_num):
    """Loads MNIST binary files and returns a clean pandas DataFrame.

    Converts images to normalized flat rows and creates a binary target column.
    """
    # 1. Read and parse images
    with open(images_path, "rb") as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8)
        # Normalize and flatten directly
        images_flat = images.astype("float32") / 255.0
        images_flat = images_flat.reshape(num_images, rows * cols)

    # 2. Read and parse labels
    with open(labels_path, "rb") as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    # Quick sanity check
    assert (
        num_images == num_labels
    ), "The number of images does not match the number of labels."

    # 3. Create descriptive column names for pixels (pixel_0, pixel_1, ...)
    pixel_cols = [f"pixel_{i}" for i in range(rows * cols)]

    # 4. Build the DataFrame
    df = pd.DataFrame(images_flat, columns=pixel_cols)

    # 5. Add the binary target column (1 if target_num, else 0) using efficient numpy vectorized operations
    df["target"] = (labels == target_num).astype(int)

    df.to_csv(file_path, index=False)

def train_test_split(df: pd.DataFrame, test_size: int) -> tuple:
    pass

def batch_split(df: pd.DataFrame, batch_size: int):
    pass

def train_on_batches(network: NeuralNetwork, batches: list):
    pass

def test_network(network: NeuralNetwork, test_data: pd.DataFrame):
    pass

def main():
    if not Path(DATASET_PATH).is_file():
        load_mnist_to_file("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", DATASET_PATH, TARGET_NUM)

    df = pd.read_csv(DATASET_PATH)

if __name__ == '__main__':
    main()

# TODO: import the pandas df and work with it instead
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