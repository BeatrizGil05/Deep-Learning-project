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

# Load metadata
df = pd.read_csv('metadata.csv')
data_root_path = r'C:\Users\inesb\Downloads\rare_species'

def preprocess(df):
    # Add the root directory with the unzipped images.
    df['full_path'] = df['file_path'].apply(lambda x: os.path.join(data_root_path, x))

    # Remove rows with missing file paths or family labels
    df = df.dropna(subset=['file_path', 'family']).reset_index(drop=True)

    # Check for missing files
    df['exists'] = df['full_path'].apply(os.path.exists)
    missing = df[df['exists'] == False]

    print("Missing images:", len(missing))
    if len(missing) > 0:
        print(missing[['file_path']].head())

    # Drop rows with missing images
    df = df[df['exists'] == True].reset_index(drop=True)

    # Duplicate rows in metadata
    duplicate_rows = df[df.duplicated()]
    print("Duplicate metadata rows:")
    print(duplicate_rows)
    df = df.drop_duplicates().reset_index(drop=True)

    # Duplicate image paths
    duplicate_paths = df[df.duplicated(subset='full_path')]
    print("Duplicate file paths:")
    print(duplicate_paths)
    df = df.drop_duplicates(subset='full_path').reset_index(drop=True)

    # Encode each category in the target variable
    df['family_encoded'] = pd.factorize(df['family'])[0]
    unique_families = df['family'].unique()
    print(df['family'].nunique()) # 202
    df.head(3)

    # Check the class distribution throughout the dataset
    target_distribution = df['full_path'].groupby(df["family_encoded"]).count()
    print(target_distribution.describe().T)
    # The count of images per class varies from 29 to 300, so we can consider this an imbalanced dataset

    # Stratified Split: 70% Train, 15% Validation, 15% Test
    train_df, test_df = train_test_split(df, test_size = 0.15, stratify = df['family'], random_state = 42)
    train_df, val_df = train_test_split(train_df, test_size = 0.1765, stratify = train_df['family'], random_state = 42) 
    # (0.1765 of the remaining 85% is roughly 15% of the total)
    print(f"Train shape: {train_df.shape}") # Train shape: (8387, 10)
    print(f"Val shape: {val_df.shape}") # Val shape: (1798, 10)
    print(f"Test shape: {test_df.shape}") # Test shape: (1798, 10)
    return train_df, test_df, val_df



# Function to find the largest and smallest images in a directory
def find_extreme_images(data_root_path):
  # Define the variables of smalles and biggest images
    largest_image = None
    smallest_image = None
    largest_size = 0
    smallest_size = float('inf') ## infinit number

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