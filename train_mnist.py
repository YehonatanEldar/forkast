import struct
from pathlib import Path

import numpy as np
import pandas as pd

from activation import Activation, ActivationFuncs
from dense import Dense
from loss import Loss
from neural_network import NeuralNetwork

# Constants
TARGET_NUM = 3
TRAIN_SIZE = 0.8
BATCH_SIZE = 100
EPOCHS = 10
LEARNING_RATE = 0.001
LAYER_SIZES = [784, 128, 64, 1]

DATASET_PATH = 'data/mnist_dataset.csv'

def load_mnist_to_file(images_path, labels_path, file_path, target_num):
    """
    Loads the mnist dataset to a csv
    """
    # read and parse images
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

    clean_df = df.dropna()
    clean_df.to_csv(file_path, index=False)

def create_network(layer_sizes: list[int]) -> NeuralNetwork:
    """
    Creates a network and fills it with random values
    """
    network = NeuralNetwork()

    for i in range(len(layer_sizes) - 1):
        # Create layers
        dense = Dense(layer_sizes[i], layer_sizes[i + 1])

        dense.biases = np.random.uniform(low=-10.0, high=11.0, size=dense.biases.shape) # randomize baises
        dense.weights = np.random.uniform(low=-10.0, high=11.0, size=dense.weights.shape) # randomize weights

        network.add(dense)

    # Add last one
    dense = Dense(layer_sizes[-1], 1)
    network.add(dense)

    return network

def target_split(df: pd.DataFrame) -> tuple:
    """
    Splits the df to the X and y
    """
    return df.drop(columns=['target']).to_numpy(), df['target'].to_numpy()

def train_test_split(df: pd.DataFrame, train_size: float) -> tuple:
    """
    Splits the data to train and test sets
    """
    train_df = df.sample(frac=train_size, random_state=42)
    test_df = df.drop(train_df.index)

    return train_df, test_df

def train_network(network: NeuralNetwork, df: pd.DataFrame, batch_size: int, epochs: int, learning_rate: float):
    """
    Trains the network on the given data
    """
    X, y = target_split(df)
    num_batch = 1
    
    for i in range(0, len(df), batch_size):
        print(f"Batch {num_batch}")
        # Split to batches
        batch_x = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        network.train(batch_x, batch_y, epochs, learning_rate)
        num_batch += 1

def test_network(network: NeuralNetwork, test_data: pd.DataFrame) -> float:
    """
    Tests the network on the given test data
    """
    X, y = target_split(test_data)
    results = network.predict(X)
    mean_error = network.loss.forward(results, y)

    return mean_error

def main():
    if not Path(DATASET_PATH).is_file():
        load_mnist_to_file("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", DATASET_PATH, TARGET_NUM)

    df = pd.read_csv(DATASET_PATH)

    network = create_network(LAYER_SIZES)

    train, test = train_test_split(df, TRAIN_SIZE)

    train_network(network, train, BATCH_SIZE, EPOCHS, LEARNING_RATE)
    print("Finished training!")

    print("Test Error Rate: " + test_network(network, test))

if __name__ == '__main__':
    main()