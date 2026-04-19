import os
import shutil
import pandas as pd

# SCRIPT 3 FOR COLLECTED DATASET
# Description: This script organizes annotated video segments into class-specific folders.

# === CONFIGURATION ===
ANNOTATION_CSV = "segment_annotations_3sec.csv"
SOURCE_FOLDER = "output_segments_3sec"
DEST_FOLDER = "collected_annotated_3sec"  # Structure will be created here

# Load annotations
df = pd.read_csv(ANNOTATION_CSV)

# Create class subfolders
for label in df["label"].unique():
    os.makedirs(os.path.join(DEST_FOLDER, label), exist_ok=True)

# Move files based on their label
for _, row in df.iterrows():
    src_path = os.path.join(SOURCE_FOLDER, row["filename"])
    dst_path = os.path.join(DEST_FOLDER, row["label"], row["filename"])

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Missing file: {src_path}")

print("\nAll annotated videos organized into:")
# Print how many clips are in the class folders
for label in df["label"].unique():
    path = os.path.join(DEST_FOLDER, label)
    print(f" - {label}: {len(os.listdir(path))} clips")
