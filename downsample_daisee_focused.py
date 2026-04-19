import os
import random
import shutil
from tqdm import tqdm

# STEP 4 FOR DAiSEE Dataset
# Description: This script reduces the "Focused" class in the DAiSEE dataset
# to approximately 4.5% of its original size in each of the class splits, while keeping all samples of other classes.
# The reduced dataset is saved in a new directory structure.

# =============
# CONFIGURATION
# =============
SOURCE_ROOT = "DAiSEE_prepared"             # path to your DAiSEE dataset
TARGET_ROOT = "DAiSEE_reduced"              # output path for the reduced dataset
FOCUSED_RATIO = 0.045                       # keep 4.5% of original Focused dataset
SEED = 42                                   # random seed for reproducibility

random.seed(SEED)

# Helper to copy selected files
def copy_files(file_list, src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    # Copy each file from source to destination directory
    for f in tqdm(file_list, desc=f"Copying to {dest_dir}", leave=False):
        shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))

# Main Process
for split in ["train", "val", "test"]:
    print(f"\nProcessing {split} split...")
    # Define source and target directories for the current split
    split_dir = os.path.join(SOURCE_ROOT, split)
    target_split_dir = os.path.join(TARGET_ROOT, split)
    os.makedirs(target_split_dir, exist_ok=True)

    # Process each class label within the current split
    for label in os.listdir(split_dir):
        label_dir = os.path.join(split_dir, label)
        if not os.path.isdir(label_dir):    # Skip non-directory files
            continue

        # Define target directory for the current class label
        target_label_dir = os.path.join(target_split_dir, label)
        os.makedirs(target_label_dir, exist_ok=True)

        # List all video files in the current class label directory
        files = [f for f in os.listdir(label_dir) if f.endswith((".mp4", ".avi"))]

        # Reduce "Focused" class videos
        if label.lower() == "focused":
            keep_count = int(len(files) * FOCUSED_RATIO)
            selected = random.sample(files, keep_count)     # Randomly select a subset of Focused videos
            print(f" - {label}: keeping {keep_count}/{len(files)} ({FOCUSED_RATIO*100:.0f}%)")    # Print reduction info
        # Keep all videos for other classes
        else:
            selected = files
            print(f" - {label}: keeping all {len(files)} files")

        copy_files(selected, label_dir, target_label_dir)

print("\nDAiSEE Focused reduction complete!")
print(f"Reduced dataset saved in: {os.path.abspath(TARGET_ROOT)}")
