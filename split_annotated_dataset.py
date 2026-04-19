import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# SCRIPT 4 FOR COLLECTED DATASET
# Description: This script splits the annotated dataset into training, validation, and test sets.

# ==============
# CONFIGURATION
# ==============
SOURCE_FOLDER = "collected_annotated_3sec"
DEST_FOLDER = "collected_split_3sec"   # Output: train/val/test
RATIOS = [0.8, 0.1, 0.1]          # 80/10/10 split

# Collect all labeled samples
records = []
for label in os.listdir(SOURCE_FOLDER):
    label_path = os.path.join(SOURCE_FOLDER, label)
    if not os.path.isdir(label_path):
        continue
    for f in os.listdir(label_path):
        if f.lower().endswith(".mp4"):
            records.append({"filename": f, "label": label, "path": os.path.join(label_path, f)})

df = pd.DataFrame(records)
print(f"Total samples found: {len(df)}")
print(df["label"].value_counts())

# ================
# STRATIFIED SPLIT
# ================
# 1. First split: Separate the main training set (train_df) from the validation/test pool (temp_df).
train_df, temp_df = train_test_split(
    df, stratify=df["label"], test_size=(1 - RATIOS[0]), random_state=42
)
# 2. Second split: Divide the validation/test pool into separate validation (val_df) and test (test_df) sets.
val_df, test_df = train_test_split(
    temp_df, stratify=temp_df["label"],
    test_size=(RATIOS[2] / (RATIOS[1] + RATIOS[2])), random_state=42
)

splits = {"train": train_df, "val": val_df, "test": test_df}


# =============================
# COPY FILES INTO NEW STRUCTURE
# =============================
# 1. Create Destination Folders: For each split (train, val, test) and for every unique attention state label,
#    a corresponding directory is created (DEST_FOLDER/train/Focused).
for split_name, split_df in splits.items():
    for label in df["label"].unique():
        os.makedirs(os.path.join(DEST_FOLDER, split_name, label), exist_ok=True)

# 2. Copy files: Each file from the split DataFrame is copied to its corresponding directory.
    for _, row in split_df.iterrows():
        dst = os.path.join(DEST_FOLDER, split_name, row["label"], row["filename"])
        shutil.copy(row["path"], dst)

# Print how many clips were copied for each split
    print(f"{split_name.upper()}: {len(split_df)} clips")

# ==============================
# CHECK FINAL CLASS DISTRIBUTION
# ==============================
for split_name, split_df in splits.items():
    print(f"\n{split_name.upper()} CLASS DISTRIBUTION:")
    print(split_df["label"].value_counts(normalize=True))
