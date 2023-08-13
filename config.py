import os.path
from pathlib import Path

base_dir = Path(__file__).resolve().parent

data_path = os.path.join(base_dir, "data")
dataset_path = os.path.join(base_dir, "data", "chest_database")
normal_dataset_path = os.path.join(base_dir, "data", "chest_database", "Normal")
tuberculosis_dataset_path = os.path.join(base_dir, "data", "chest_database", "Tuberculosis")
model_path = os.path.join(base_dir, "model.h5")

test_csv_path = os.path.join(data_path, "test.csv")
train_csv_path = os.path.join(data_path, "train.csv")
