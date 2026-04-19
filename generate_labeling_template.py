import os
import pandas as pd

# SCRIPT 2 FOR COLLECTED DATASET
# Description: This script generates a CSV template for labeling the 10-second video segments.

# ==============
# CONFIGURATION
# ==============
SEGMENT_FOLDER = "output_segments_3sec"
OUTPUT_CSV = "segment_annotations_3sec.csv" 

# Collect all video segment filenames
segment_files = sorted([
    f for f in os.listdir(SEGMENT_FOLDER)
    if f.lower().endswith(".mp4")
])

# Create DataFrame with empty labels
df = pd.DataFrame({
    "filename": segment_files,
    "label": [""] * len(segment_files)  # Empty column for manual labeling
})

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Annotation template created: {OUTPUT_CSV}")
print(f"Total segments listed: {len(segment_files)}")
