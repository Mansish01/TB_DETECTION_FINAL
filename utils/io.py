import os
import csv
from sklearn.model_selection import train_test_split


def list_files(dir: str, file_extension: str) -> list:
    """given a dorectory, list all files with given extension"""

    if not os.path.isdir(dir):
        return None

    files = os.listdir(dir)
    return files


def get_image_label_pairs(dir: str, label: str) -> tuple:
    filenames = list_files(dir, "")
    labels = [label] * len(filenames)
    return filenames, labels


def save_as_csv(img_path: str, label: str, outfile):
    """save image path and save as csv fille"""
    with open(outfile, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        for image_path, label in zip(img_path, label):
            writer.writerow([image_path, label])


def read_as_csv(csv_file):
    image_path = []
    labels = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_path.append(row[0])
            labels.append(row[1])
    return image_path, labels
