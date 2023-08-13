import os
from os.path import join 
from utils.io import get_image_label_pairs , save_as_csv
from sklearn.model_selection import train_test_split


DATA_DIR= 'data/tuber_dataset'
#print(DATA_DIR)

data_folders= os.listdir(DATA_DIR)

x=[]
y=[]

for folder in data_folders:
    data_path= join(DATA_DIR, folder)
    label = folder
    
    files_names, label = get_image_label_pairs(data_path , label)
    x.extend(files_names)
    y.extend(label)
    
#print(len(x), len(y))


x_train, x_test , y_train , y_test= train_test_split(x , y, test_size=0.2, random_state=42)
# print(len(x_train), len(x_test), len(y_train), len (y_test))

save_as_csv(x_train , y_train, 'data/train.csv')
save_as_csv(x_test , y_test , "data/test.csv")


