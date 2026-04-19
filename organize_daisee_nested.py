import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# SCRIPT 2 FOR DAiSEE Dataset
# Description: This script prepares the DAiSEE dataset for modeling.
# It reads the master label file (AllLabels_Custom.csv),
# performs a stratified split into train/val/test sets
# based on the CustomLabel column, and copies each video clip
# into its respective split folder under DAiSEE_prepared/.
#
# OUTPUT STRUCTURE:
# DAiSEE_prepared/
# ├── train/
# │   ├── Distracted/
# │   ├── Fatigued/
# │   └── Focused/
# ├── val/
# └── test/

# Define dataset and label paths
DATASET_DIR = os.path.join("DAiSEE", "DataSet")   # Root folder containing all video clips (Train/Test/Validation mixed)
LABEL_PATH = os.path.join("DAiSEE", "Labels", "AllLabels_Custom.csv")  # Path to the custom label CSV file
OUTPUT_DIR = "DAiSEE_prepared"                    # Destination folder for the reorganized dataset

# Load the CSV file containing clip labels
df = pd.read_csv(LABEL_PATH)

# Clean up any whitespace from column names to prevent key errors
df.columns = df.columns.str.strip()

# Perform stratified dataset splitting to maintain label balance
#  - 70% for training
#  - 20% for validation
#  - 10% for testing
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["CustomLabel"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.333, stratify=temp_df["CustomLabel"], random_state=42
)

# Print summary of the number of clips in each split
print("Split summary:")
print(f"Train: {len(train_df)} clips")
print(f"Validation: {len(val_df)} clips")
print(f"Test: {len(test_df)} clips")

# Combine all splits into a dictionary for easier iteration later
splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

# Define a function to copy videos into their respective folders
def copy_videos(split_name, subset_df):
    print(f"\nCopying {split_name} set...")

    # Iterate through each row in the split DataFrame with a progress bar
    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        clip_id = row["ClipID"]           # Name of the video file
        label = row["CustomLabel"]        # Assigned custom label (Fatigued, Focused, Distracted)

        found = False  # Track if the video file was located

        # Search across all possible DAiSEE parent folders (Train, Validation, Test)
        for parent in ["Train", "Validation", "Test"]:
            parent_dir = os.path.join(DATASET_DIR, parent)

            # Walk through the directory to find the specific clip
            for root, _, files in os.walk(parent_dir):
                for file in files:
                    # Check if the current file matches the clip ID
                    if file == clip_id:
                        src_path = os.path.join(root, file)  # Source file path
                        dst_dir = os.path.join(OUTPUT_DIR, split_name, label)  # Destination folder by split and label
                        os.makedirs(dst_dir, exist_ok=True)  # Create folder if not yet existing
                        shutil.copy2(src_path, os.path.join(dst_dir, file))  # Copy file with metadata preserved
                        found = True
                        break
                if found:
                    break

        # Notify if a clip listed in the labels file was not found in the dataset
        if not found:
            print(f"Clip not found: {clip_id}")

# Execute the reorganization process for each dataset split
for split_name, subset_df in splits.items():
    copy_videos(split_name, subset_df)

# Final confirmation message once copying is complete
print("\nDataset successfully reorganized with stratified 70/20/10 split!")
print(f"Output saved to: {OUTPUT_DIR}")
