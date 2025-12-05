import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths (set in .env file)
#DATA_DIR = os.getenv("DATA_DIR", "data/raw")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")  # Adjust as needed
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# Image settings
IMG_SIZE = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training settings
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.001
PATIENCE = 10  # For early stopping

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Model settings
NUM_CLASSES = None  # Will be set from data
CLASS_NAMES = None  # Will be set from data

# Augmentation settings
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.2