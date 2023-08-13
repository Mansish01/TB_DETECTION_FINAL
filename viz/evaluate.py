import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score

# from config import MODEL_CHECKPOINT_PATH
from config import model_path, test_csv_path
from utils.io import read_as_csv
from utils.pre_processing import label_map, get_file_path, clean_transforms, \
    clean_label_transforms
from utils.pre_processing import label_transforms, image_transforms


def get_prediction():
    loaded_knn_model = joblib.load(model_path)

    test_files, test_labels = read_as_csv(test_csv_path)
    test_zipped = zip(test_files, test_labels)
    # Transforms for test data
    test_transforms = [image_transforms(get_file_path(file=file, label=label), label) for file, label in test_zipped]
    test_cleaned_transforms = clean_transforms(test_transforms)
    test_transforms_labels = [label_transforms(lab) for lab in test_labels]
    test_cleaned_label_transforms_labels = clean_label_transforms(test_transforms, test_transforms_labels)
    print(f"Test | cleaned image transformed length ::  {len(test_cleaned_transforms)}")
    print(f"Test | cleaned label transformed length ::  {len(test_cleaned_label_transforms_labels)}")
    x_test = np.array(test_cleaned_transforms)
    y_test = np.array(test_cleaned_label_transforms_labels)
    y_pred = loaded_knn_model.predict(x_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map)
    display_cm.plot()
    plt.show()

    # Compute precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average='macro')  # 'macro' averages the scores for each class
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


if __name__ == '__main__':
    get_prediction()
