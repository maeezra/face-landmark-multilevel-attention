import os
import csv

# SCRIPT 3 FOR DAiSEE Dataset
# Description: This script counts the number of video samples in each split and class
# of the prepared DAiSEE dataset, and saves a summary CSV file.

# Root directory of the prepared dataset
#ROOT_DIR = "DAiSEE_prepared"
ROOT_DIR = "DAiSEE_reduced"

# Define expected structure
SPLITS = ["train", "val", "test"]
CLASSES = ["Focused", "Distracted", "Fatigued"]

# Count summary dictionary
counts = {}

print("\nCounting video samples...\n")

total_files = 0

# Iterate through each split and class to count video files
for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(ROOT_DIR, split, cls)
        if not os.path.exists(folder):
            print(f"Missing folder: {folder}") # Folder does not exist, skipping count
            continue

        # Count all video files (supports .mp4, .avi, .mov)
        num_videos = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ])

        # Store the count in the dictionary
        counts[(split, cls)] = num_videos
        total_files += num_videos

        print(f"{split}/{cls:<10}: {num_videos} videos")

# Summary table
print("\n============================")
print(f"Total video files: {total_files}")
print("============================")

# Save to CSV
csv_path = os.path.join(ROOT_DIR, "dataset_counts.csv")
with open(csv_path, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Split", "Class", "Count"])
    for (split, cls), count in counts.items():
        writer.writerow([split, cls, count])

print(f"\nSaved detailed counts to: {csv_path}")
