from pathlib import Path

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
# from os.path import join
# from viz.visualization import display_grid
import os

from config import normal_dataset_path, tuberculosis_dataset_path
from utils.io import read_as_csv

label_map = {
    "Normal": 1,
    "Tuberculosis": 2,
}

data_root = "data/chest_database"
index_to_label_dict = {index: label for label, index in label_map.items()}


def get_file_path(file, label):
    return os.path.join(normal_dataset_path if (label == "Normal") else tuberculosis_dataset_path, file)


def clean_transforms(transforms):
    cleaned_transforms = []
    for transform in transforms:
        if transform.size == transforms[0].size:
            cleaned_transforms.append(transform)
    return cleaned_transforms


def clean_label_transforms(img_transforms, lbl_transforms):
    cleaned_transforms = []
    for index, values in enumerate(img_transforms):
        if values.size == img_transforms[0].size:
            cleaned_transforms.append(lbl_transforms[index])
    return cleaned_transforms


def image_transforms(file_path, label) -> np.ndarray:
    array = read_image(file_path, size=(256, 256), grayscale=True)
    flatten_image = array.flatten()
    return flatten_image


def label_transforms(label) -> int:
    # label_to_index
    return label_to_index(label)


def label_to_index(label: str):
    if label not in label_map:
        raise KeyError("label in not valid")
    return label_map[label]


def index_to_label(inx: int):
    if inx not in index_to_label_dict:
        raise KeyError("index is not valid")
    return index_to_label_dict[inx]


def read_image(image_path: str, size: tuple = (256, 256), grayscale: bool = False) -> np.ndarray:
    """ reads image from the given path and returns as a numpy array
    TODO: resize the image and implement the mode of zoom or paddding 
    args:
    -----
    image_path: the image which we want to read
    mode: either 'zoom' or 'pad'
    size:the size of the image we want to set to
    """
    image = Image.open(image_path)
    # image= image.resize(size)
    height, width = image.size

    # if mode == "padding":
    #    if height== width:
    #      pass
    #    else:
    #     image=ImageOps.pad(image, (256, 256), color=None, centering=(0.5, 0.5))

    # if mode== "zoom":
    #     diff= height-width
    #     if diff>0:
    #         right = width
    #         (left, upper, right, lower) = (0, diff//2, right, height-(diff//2))
    #         image= image.crop((left, upper, right, lower))

    #     else:
    #         lower= height
    #         diff = abs(diff)
    #         (left, upper, right, lower) = (diff//2, 0, width-(diff//2), lower)
    #         image = image.crop((left, upper, right, lower))
    image = Image.open(image_path)
    image = image.resize(size)  # Resize all images to the specified size
    img_array = np.asarray(image)
    return img_array


def validate_image_dimensions(image_paths, target_size):
    """
    Validate that all images in the given list have the same dimensions.

    Args:
        image_paths (list): List of file paths to the images.
        target_size (tuple): The target size (height, width) for images.

    Returns:
        bool: True if all images have the same dimensions, False otherwise.
    """
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.size != target_size:
            return False
    return True


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = os.path.join(BASE_DIR, "data", "train.csv")
    train_files, train_labels = read_as_csv(csv_path)
    zipped = zip(train_files, train_labels)
    # file_path_array = []
    # for file, label in zipped:
    #     path = os.path.join(BASE_DIR, "data", "chest_database", "Normal" if (label == "Normal") else "Tuberculosis",
    #                         file)
    #     file_path_array.append(path)
    transforms = [image_transforms(
        os.path.join(BASE_DIR, "data", "chest_database", "Normal" if (label == "Normal") else "Tuberculosis", file),
        label) for file, label in zipped]
    print(transforms)
    # X_train = np.array([transforms])
# print("Flattened image lengths in X_train:", [len(image) for image in X_train])
