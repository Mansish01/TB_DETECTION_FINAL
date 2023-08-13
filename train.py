import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.io import read_as_csv
from utils.pre_processing import read_image, label_to_index
import os.path
import joblib
from config  import  MODEL_CHECKPOINT_PATH   

from utils.pre_processing import image_transforms, label_transforms


data_root = "data"

def train(data_root, train_csv, test_csv, model, checkpoint_path):
# load csv
    train_path = os.path.join(data_root, train_csv)
    test_path = os.path.join(data_root, test_csv)
    train_files, train_labels = read_as_csv(train_path)
    test_files, test_labels = read_as_csv(test_path)
    
    X_train=[]
    Y_train=[]
    X_test =[]
    Y_test =[]
    # Apply the image_transforms function to train_files and test_files
    for file, label in zip(train_files, train_labels):
        flatten_image = image_transforms(file, label)
        if len(flatten_image) !=  65536:
          lab = label_transforms(label)
          Y_train. append(lab)
          #Y_train = np.array([label_transforms(lab) for lab in label])
          #print(len(flatten_image))
          X_train.append(flatten_image)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    #print("Flattened image lengths in X_train:", [len(image) for image in X_train])
    

    
    print(Y_train)

    # X_test = np.array(
    #     [image_transforms(file, label) for file, label in zip(test_files, test_labels)])
    
    for file, label in zip(test_files, test_labels):
        flatten_image = image_transforms(file, label)
        if len(flatten_image) !=  65536:
          lab = label_transforms(label)
          Y_test. append(lab)
          #Y_test = np.array([label_transforms(lab) for lab in test_labels])
          #print(len(flatten_image))
          X_test.append(flatten_image)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    



    clf = model()
    clf.fit(X_train, Y_train)


    print("Train score:", clf.score(X_train, Y_train))
    print("Test score:", clf.score(X_test, Y_test))

  
    # Save the model
    joblib.dump(clf, checkpoint_path)

    # Load the model from the file
    loaded_knn_model = joblib.load(checkpoint_path)


    # print(loaded_knn_model.predict(X_test))
    
if __name__ == "__main__":
   
    
    train(data_root , "train.csv", "test.csv", KNeighborsClassifier, MODEL_CHECKPOINT_PATH)