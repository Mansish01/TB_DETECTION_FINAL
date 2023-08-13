import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from config import model_path, data_path, train_csv_path, test_csv_path
from utils.io import read_as_csv
from utils.pre_processing import image_transforms, label_transforms, clean_transforms, clean_label_transforms, \
    get_file_path


def train(model, checkpoint_path):
    # load csv
    train_files, train_labels = read_as_csv(train_csv_path)
    test_files, test_labels = read_as_csv(test_csv_path)
    zipped = zip(train_files, train_labels)
    test_zipped = zip(test_files, test_labels)

    # Transforms for train data
    train_transforms = [image_transforms(get_file_path(file=file, label=label), label) for file, label in zipped]
    train_cleaned_transforms = clean_transforms(train_transforms)
    train_transforms_labels = [label_transforms(lab) for lab in train_labels]
    train_cleaned_label_transforms_labels = clean_label_transforms(train_transforms, train_transforms_labels)
    print(f"Train | cleaned image transformed length ::  {len(train_cleaned_transforms)}")
    print(f"Train | cleaned label transformed length ::  {len(train_cleaned_label_transforms_labels)}")
    x_train = np.array(train_cleaned_transforms)
    y_train = np.array(train_cleaned_label_transforms_labels)

    # Transforms for test data
    test_transforms = [image_transforms(get_file_path(file=file, label=label), label) for file, label in test_zipped]
    test_cleaned_transforms = clean_transforms(test_transforms)
    test_transforms_labels = [label_transforms(lab) for lab in test_labels]
    test_cleaned_label_transforms_labels = clean_label_transforms(test_transforms, test_transforms_labels)
    print(f"Test | cleaned image transformed length ::  {len(test_cleaned_transforms)}")
    print(f"Test | cleaned label transformed length ::  {len(test_cleaned_label_transforms_labels)}")
    x_test = np.array(test_cleaned_transforms)
    y_test = np.array(test_cleaned_label_transforms_labels)

    clf = model()
    clf.fit(x_train, y_train)

    print("Train score:", clf.score(x_train, y_train))
    print("Test score:", clf.score(x_test, y_test))

    # Save the model
    joblib.dump(clf, checkpoint_path)


if __name__ == "__main__":
    train(data_path, "train.csv", "test.csv", KNeighborsClassifier, model_path)
