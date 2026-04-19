import os
import cv2
import albumentations as A
import random
import csv
from tqdm import tqdm
import math

# SCRIPT 5 FOR COLLECTED DATASET AND DAiSEE DATASET
# Description: This script performs data augmentation on the collected dataset to balance class distributions.

# =============
# CONFIGURATION
# =============

SOURCE_ROOT = "collected_split_3sec"
AUGMENTED_ROOT = "collected_split_aug_3sec"

# SOURCE_ROOT = "DAiSEE_reduced"
# AUGMENTED_ROOT = "DAiSEE_reduced_aug"

LOG_FILE = os.path.join(AUGMENTED_ROOT, "augmentation_log.csv")  # CSV log path
TARGET_CLASS = "Focused"   # The majority class
SPLITS = ["train"]

# ======================================================
# DEFINE INDIVIDUAL AUGMENTERS (safe and realistic only)
# ======================================================
# For the collected dataset
# AUGMENTERS = {
#     "brightness": A.Compose([
#         A.RandomBrightnessContrast(
#             p=1.0,
#             brightness_limit=0.15,   # slightly reduced for realism
#             contrast_limit=0.1
#         )
#     ]),
#     "flip": A.Compose([
#         A.HorizontalFlip(p=1.0)
#     ]),
#     "noise": A.Compose([
#         A.GaussNoise(var_limit=(10, 25), p=1.0)
#     ]),
#     "blur": A.Compose([
#         A.MotionBlur(blur_limit=(3, 7), p=1.0)
#     ]),
#     "color": A.Compose([
#         A.ColorJitter(
#             brightness=0.1,
#             contrast=0.1,
#             saturation=0.1,
#             hue=0.02,
#             p=1.0
#         )
#     ])
# }

# For the DAiSEE dataset
AUGMENTERS = {
    "brightness": A.Compose([
        A.RandomBrightnessContrast(
            p=1.0,
            brightness_limit=0.1,
            contrast_limit=0.05
        )
    ]),
    "flip": A.Compose([A.HorizontalFlip(p=1.0)]),
    "noise": A.Compose([A.GaussNoise(var_limit=(5.0, 20.0), p=1.0)]),
    "blur": A.Compose([A.MotionBlur(blur_limit=(3, 5), p=1.0)]),
    "color": A.Compose([
        A.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05,
            hue=0.01,
            p=1.0
        )
    ])
}

# Weighted probabilities (sum = 1)
AUG_WEIGHTS = {
    "brightness": 0.25,
    "flip": 0.25,
    "noise": 0.2,
    "blur": 0.15,
    "color": 0.15
}

# ==================
# Initialize CSV Log
# ==================
os.makedirs(AUGMENTED_ROOT, exist_ok=True)
# This log tracks which original file led to which augmented file, its split (train),
# its class, and the specific augmentation applied (augmentation_type).
with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["split", "class", "original_path", "augmented_path", "augmentation_type"])

# ======================================
# MAIN LOOP FOR HANDLING CLASS IMBALANCE
# ======================================

# This block iterates through the defined data splits (e.g., "train", "val", "test")
# to prepare for the data augmentation process.
for split in SPLITS:
    print(f"\nAugmenting {split} set...")
    split_dir = os.path.join(SOURCE_ROOT, split)
    
    # Count samples per class: The inner loop scans the directory structure (split_dir)
    # to determine how many video files (ending in .mp4 or .avi) belong to each
    # attention state label (class) within the current split. This is 
    # done to identify class imbalance and determine the augmentation strategy.
    class_counts = {}
    for label in os.listdir(split_dir):
        label_dir = os.path.join(split_dir, label)
        if os.path.isdir(label_dir):
            class_counts[label] = len([
                f for f in os.listdir(label_dir)
                if f.endswith((".mp4", ".avi"))
            ])
    
    # Prints the calculated counts for verification.
    print(f"Class counts: {class_counts}")
    
    # Determine target count for minority classes based on Focused
    target_count = class_counts.get(TARGET_CLASS, 0)
    print(f"Target samples per class: {target_count}")

    # Start the augmentation loop: Iterate through each attention state class (label) and its current count.
    for label, count in class_counts.items():
        if label == TARGET_CLASS:
            print(f"Skipping majority class: {label}") # Skip Focused class
            continue

        num_needed = int(target_count - count)  # Calculate how many new samples need to be generated

        if num_needed <= 0:
            print(f"No augmentation needed for {label}") # No augmentation needed if count meets or exceeds target
            continue

        # Setup directories for processing and output.
        label_dir = os.path.join(split_dir, label)
        output_label_dir = os.path.join(AUGMENTED_ROOT, split, label)
        os.makedirs(output_label_dir, exist_ok=True)

        # Get a list of all original video files in the current class directory
        source_files = [
            f for f in os.listdir(label_dir)
            if f.endswith((".mp4", ".avi"))
        ]
        num_source = len(source_files)

        # Print augmentation summary for the current class
        print(f"\nAugmenting class '{label}':")
        print(f" - Original: {num_source}")
        print(f" - Target: {target_count}")
        print(f" - Need to generate: {num_needed}")

        # Calculate how many augmentations per video (2–5 random per sample)
        num_per_video = math.ceil(num_needed / num_source)
        num_per_video = min(max(2, num_per_video), 5)  # clamp to 2–5 augmentations

        total_generated = 0
        aug_index = 1

        # Start augmenting videos for the current class
        for file in tqdm(source_files, desc=f"Augmenting {label}", leave=False):
            if total_generated >= num_needed:
                break

            # Prepare paths and filenames for augmentation
            src_path = os.path.join(label_dir, file)
            base_name, ext = os.path.splitext(file)

            # Randomly decide how many augmentations to apply to this file
            current_augs = random.randint(2, 5)
            # Clamp to ensure we don't generate more than needed
            for _ in range(current_augs):
                if total_generated >= num_needed:
                    break

                # Randomly choose one augmentation type (weighted)
                aug_name = random.choices(
                    list(AUG_WEIGHTS.keys()),
                    weights=list(AUG_WEIGHTS.values()),
                    k=1
                )[0]
                AUGMENTER = AUGMENTERS[aug_name]

                # Prepare output filename and path
                out_filename = f"{base_name}_aug{aug_index:02d}_{aug_name}{ext}"
                out_path = os.path.join(output_label_dir, out_filename)

                # Open the source video for reading and prepare the output video writer
                cap = cv2.VideoCapture(src_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                # Apply augmentation frame by frame
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    aug_frame = AUGMENTER(image=frame)["image"]
                    out.write(aug_frame)

                cap.release()
                out.release()

                # Log augmentation details to CSV
                with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([split, label, src_path, out_path, aug_name])

                # Increment counters for augmentation indexing and total generated samples
                aug_index += 1
                total_generated += 1

        print(f"Done augmenting {label}: generated {total_generated} new clips")

# Print final summary
print("\nAll augmentations completed successfully!")
print(f"Output saved in: {os.path.abspath(AUGMENTED_ROOT)}")
print(f"Log saved in: {os.path.abspath(LOG_FILE)}")
