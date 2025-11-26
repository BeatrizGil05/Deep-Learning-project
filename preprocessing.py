import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Load Data
df = pd.read_csv('metadata.csv')

# 2. Adjust File Paths
# You mentioned images are in folders named 'phylum_family'. 
# The CSV 'file_path' column already includes 'phylum_family/filename.jpg'.
# You just need to add the root directory where you unzipped the images.
data_root_path = '/Users/jakubb/Desktop/rare_species' 
df['full_path'] = df['file_path'].apply(lambda x: os.path.join(data_root_path, x))

# 3. Stratified Split (Crucial for rare species)
# Split: 70% Train, 15% Validation, 15% Test
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['family'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df['family'], random_state=42) 
# (0.1765 of the remaining 85% is roughly 15% of the total)

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")
