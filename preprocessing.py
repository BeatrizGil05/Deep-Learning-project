import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from keras.layers import Dropout

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from PIL import Image

# Load Data
df = pd.read_csv(r"C:\Users\inesb\Downloads\Deep-Learning-project\metadata.csv")

# Add the root directory with the unzipped images.
data_root_path = r'C:\Users\inesb\Downloads\rare_species' 
df['full_path'] = df['file_path'].apply(lambda x: os.path.join(data_root_path, x))

# Check for null values
df.info()
# There are no null values

# Encode each category in the target variable
df['class_encoded'] = pd.factorize(df['family'])[0]
print(df['family'].nunique())
df.head(3)

# Check the class distribution throughout the dataset
class_distribution = df['full_path'].groupby(df["class_encoded"]).count()
class_distribution.describe().T
# The count of images per class varies from 29 to 300, so we can consider this an imbalanced dataset

# Check for duplicates
df.info()
restricted_df = df.drop(columns = ["rare_species_id", "eol_content_id", "eol_page_id", "kingdom",
                               "phylum", "family", "file_path"]).drop_duplicates(keep='first')
restricted_df.info()
# There remain 11983 non-null rows, so there are no duplicates

# Function to find the largest and smallest images in a directory, took 3m43s to run
def find_extreme_images(data_root_path):
    largest_image = None
    smallest_image = None
    largest_size = 0
    smallest_size = float('inf')
    # Iteration loop for each folder to compare the image sizes
    for folder in os.listdir(data_root_path):
        folder_path = os.path.join(data_root_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                with Image.open(file_path) as img:
                    img_size = img.size[0] * img.size[1]

                    if img_size > largest_size:
                        largest_size = img_size
                        largest_image = file_path, img.size
                    if img_size < smallest_size:
                        smallest_size = img_size
                        smallest_image = file_path, img.size

    return largest_image, smallest_image

largest_image, smallest_image = find_extreme_images(data_root_path)
print("Largest image in the directory:")
print("File Path:", largest_image[0])
print("Size (Width x Height):", largest_image[1])
# Size (Width x Height): (17000, 6800)
print("\nSmallest image in the directory:")
print("File Path:", smallest_image[0])
print("Size (Width x Height):", smallest_image[1])
# Size (Width x Height): (193, 129)









# 3. Stratified Split (Crucial for rare species)
# Split: 70% Train, 15% Validation, 15% Test
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['family'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df['family'], random_state=42) 
# (0.1765 of the remaining 85% is roughly 15% of the total)

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")
