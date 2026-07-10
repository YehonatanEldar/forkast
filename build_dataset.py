import numpy as np
from PIL import Image
import os

IMAGE_SHAPE = (32, 32)
IMAGE_SIZE = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
DATASET_SIZE = 1000

def get_image_array(file_path: str, new_size=IMAGE_SHAPE) -> np.array:
    """
    Reads an image from the given path and returns it as a 
    grayscaled and resized numpy array.
    """
    print(f"Processing {file_path}...")
    img = Image.open(file_path)
    # Resize and grayscale
    new_img = img.resize(new_size, Image.Resampling.LANCZOS).convert('L')

    return np.array(new_img).flatten()

def get_filenames(dir_path: str) -> int:
    """
    Returns names of files in the directory (excluding folders).
    """
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

def create_image_matrix(dir_path: str):
    """
    Fills a numpy array with all of the images in a given directory
    """
    filenames = get_filenames(dir_path)
    arr = np.zeros((len(filenames), IMAGE_SIZE))

    array_list = [get_image_array(os.path.join(dir_path, filename)) for filename in filenames]

    # Convert to array
    return np.vstack(array_list)

def main():
    # TODO: take images according to positive/negative/hard/wasy split
    positive_images = create_image_matrix('images/positive')
    negative_images = create_image_matrix('images/negative')
    positive_labels = np.ones(positive_images.shape[0])
    negative_labels = np.zeros(negative_images.shape[0])

    X = np.vstack(positive_images, negative_images)
    y = np.vstack(positive_labels, negative_labels)

    # Save to file
    np.savez("fork_dataset.npz", X=X, y=y)

if __name__ == '__main__':
    main()