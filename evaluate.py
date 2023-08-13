import joblib
from config import MODEL_CHECKPOINT_PATH 
from utils.io import read_as_csv
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
from utils.pre_processing import label_map, index_to_label , label_to_index
from viz.visualization import display_grid
import numpy as np
from metrics.accuracy import accuracy
from metrics.confusion_metrix import confusionmatrix , confusion_metrics_calculate
from utils.load_config import config_laoder
from utils.io import save_predictions
import os 
from utils.pre_processing import label_transforms,image_transforms
from argparse import ArgumentParser


def get_prediction(model_checkpoint_path, csv_test_file, out_dir):
    res_out_dir = f'results/{out_dir}'
    
    os.makedirs( res_out_dir, exist_ok = True)
    loaded_knn_model = joblib.load(model_checkpoint_path)
    test_files, test_labels = read_as_csv(csv_test_file)
   # test_label =  label_to_index(test_labels)
    
    
    X_test = np.array([image_transforms(file, label) for file, label in zip(test_files, test_labels)])
    Y_test = np.array([label_transforms(lab) for lab in test_labels])
    y_pred = loaded_knn_model.predict(X_test)
    save_predictions(test_files , Y_test, y_pred, f'{res_out_dir}/predict.csv')
    print("prediction save")
    
if __name__ == '__main__' :
    parser = ArgumentParser(
        prog="test",
        description="Script to test model",
    )
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    config=config_laoder(r"configs\test.yaml")
       
    configs = config_laoder("configs/test.yaml")
    get_prediction(model_checkpoint_path= configs["model_checkpoint"], csv_test_file=configs["data_path"],
                   out_dir=config["out_dir"])    
# Load the model

# Test filenames, labels
# Predict


# Compute accuracy
#calculated_accuracy = accuracy_score(Y_test, y_pred)
#print("Accuracy:", accuracy)

#defined_accuracy= accuracy( y_pred, Y_test)
#print("Defined_accuracy :", defined_accuracy)


# matrix = confusionmatrix( y_pred, Y_test)
# print(matrix)

# new_matrix = confusion_metrics_calculate(y_pred, Y_test)
# print(new_matrix) 

# Compute confusion matrix
#cm = confusion_matrix(Y_test, y_pred)
#print("Confusion Matrix:\n", cm)

# Display confusion matrix
#labels = label_map.keys()
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#disp.plot()
#plt.show()

# Compute precision, recall, and F1-score
#precision = precision_score(Y_test, y_pred, average='macro')  # 'macro' averages the scores for each class
#recall = recall_score(Y_test, y_pred, average='macro')
#f1 = f1_score(Y_test, y_pred, average='macro')

#print("Precision:", precision)
#print("Recall:", recall)
#print("F1-score:", f1)

# Compute specificity (True Negative Rate)
#specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#print("Specificity (True Negative Rate):", specificity)

# Compute Cohen's Kappa 
#cohen_kappa = cohen_kappa_score(Y_test, y_pred)
#print("Cohen's Kappa:", cohen_kappa)

#print(classification_report(Y_test, y_pred, target_names=labels))


#DATA_DIR ='data/middle-ear-dataset'
#predicted_label = np.array([index_to_label(i) for i in y_pred])


#display_grid(image_dir=DATA_DIR, images=test_files, labels=test_labels, n_rows=3, n_cols=3, title='asom') 
  
