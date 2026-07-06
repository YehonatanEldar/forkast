import numpy as np
import struct
import pandas as pd

def load_mnist_to_df(images_path, labels_path, target_num=5):
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

    df.to_csv('data/mnist.csv', index=False)

load_mnist_to_df("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 3)