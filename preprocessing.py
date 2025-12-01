import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load metadata
df = pd.read_csv('metadata.csv')

# Clean metadata: Remove rows with missing file paths or family labels
df = df.dropna(subset=['file_path', 'family']).reset_index(drop=True)

# Duplicate rows in metadata
duplicate_rows = df[df.duplicated()]
print("Duplicate metadata rows:")
print(duplicate_rows)
df = df.drop_duplicates().reset_index(drop=True)

# Adjust File Paths
data_root_path = '/Users/jakubb/Desktop/rare_species' 
df['full_path'] = df['file_path'].apply(lambda x: os.path.join(data_root_path, x))

# Check for missing files
df['exists'] = df['full_path'].apply(os.path.exists)
missing = df[df['exists'] == False]

print("Missing images:", len(missing))
if len(missing) > 0:
    print(missing[['file_path']].head())

# Drop rows with missing images
df = df[df['exists'] == True].reset_index(drop=True)

# Duplicate image paths
duplicate_paths = df[df.duplicated(subset='full_path')]
print("Duplicate file paths:")
print(duplicate_paths)
df = df.drop_duplicates(subset='full_path').reset_index(drop=True)

# Encode each category in the target variable
df['class_encoded'] = pd.factorize(df['family'])[0]
unique_families = df['family'].unique()
print(df['family'].nunique())
df.head(3)

# Check class imbalance
print(df['family'].value_counts()) # para o report

# Eploratory: Finding largest/smallest images

# Stratified Split 
# Split: 70% Train, 15% Validation, 15% Test
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['family'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df['family'], random_state=42) 
# (0.1765 of the remaining 85% is roughly 15% of the total)

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

