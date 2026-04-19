import os
import shutil
import pandas as pd

# SCRIPT 6 FOR COLLECTED DATASET AND DAiSEE DATASET
# Description: This script merges the original and augmented datasets into a final dataset.

# Define paths
ORIGINAL_DATASET_DIR = "collected_split_3sec"   # original dataset path
AUGMENTED_DATASET_DIR = "collected_split_aug_3sec" # augmented dataset path
OUTPUT_DIR = "collected_final_3sec"                # where merged data will go
SUMMARY_CSV = "merge_collected_summary_3sec.csv"

# ORIGINAL_DATASET_DIR = "DAiSEE_reduced"   # original dataset path
# AUGMENTED_DATASET_DIR = "DAiSEE_reduced_aug" # augmented dataset path
# OUTPUT_DIR = "DAiSEE_final"                # where merged data will go
# SUMMARY_CSV = "merge_daisee_summary.csv"

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset splits and class names
splits = ["train", "val", "test"]
classes = ["Focused", "Distracted", "Fatigued"]

summary_data = []

print("Merging datasets...\n")

# Iterate through each split and class to merge datasets
for split in splits:
    for cls in classes:
        # Define source and target folders
        orig_dir = os.path.join(ORIGINAL_DATASET_DIR, split, cls)
        aug_dir = os.path.join(AUGMENTED_DATASET_DIR, split, cls)
        out_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(out_dir, exist_ok=True)

        # Copy original files
        orig_files = []
        if os.path.exists(orig_dir):
            for f in os.listdir(orig_dir):
                src = os.path.join(orig_dir, f)
                dst = os.path.join(out_dir, f)
                shutil.copy2(src, dst)
                orig_files.append(f)

        # Copy augmented files
        aug_files = []
        if os.path.exists(aug_dir):
            for f in os.listdir(aug_dir):
                src = os.path.join(aug_dir, f)
                dst = os.path.join(out_dir, f)
                # Avoid overwriting
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                aug_files.append(f)

        # Count total files
        total_files = len(os.listdir(out_dir))
        print(f"{split}/{cls}: {len(orig_files)} original + {len(aug_files)} augmented = {total_files} total")

        # Add to summary
        summary_data.append({
            "Split": split,
            "Class": cls,
            "Original Count": len(orig_files),
            "Augmented Count": len(aug_files),
            "Total Count": total_files
        })

# Save summary to CSV
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"\nMerge complete! Summary saved to: {SUMMARY_CSV}")

# Display summary
print("\n--- Dataset Summary ---")
print(summary_df)
