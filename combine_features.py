import os
import pandas as pd

# SCRIPT 8
# Description: This script combines feature CSV files from DAiSEE and collected datasets into a single dataset per split.

# =============
# CONFIGURATION
# =============
DAISEE_FEATURES = "DAiSEE_features"         # folder containing train_features.csv, val_features.csv, test_features.csv
COLLECTED_FEATURES = "collected_features"   # folder containing train_features.csv, val_features.csv, test_features.csv
OUTPUT_FEATURES = "combined_features"       # output folder

os.makedirs(OUTPUT_FEATURES, exist_ok=True)

# ===============
# MERGE PER SPLIT
# ===============
for split in ["train", "val", "test"]:
    daisee_csv = os.path.join(DAISEE_FEATURES, f"{split}_features.csv")
    collected_csv = os.path.join(COLLECTED_FEATURES, f"{split}_features.csv")

    # Check if both CSV files exist before proceeding
    if not os.path.exists(daisee_csv):
        print(f"Missing: {daisee_csv}")
        continue
    if not os.path.exists(collected_csv):
        print(f"Missing: {collected_csv}")
        continue

    print(f"Merging {split} features...")

    # Load CSVs
    df_daisee = pd.read_csv(daisee_csv)
    df_collected = pd.read_csv(collected_csv)

    # Check for column consistency before adding source
    if set(df_daisee.columns) != set(df_collected.columns):
        print(f"Column mismatch in {split} set!")
        print("DAiSEE columns:", df_daisee.columns.tolist())
        print("Collected columns:", df_collected.columns.tolist())
        continue

    # Add a column to trace origin
    df_daisee["source"] = "daisee"
    df_collected["source"] = "collected"

    # Check for column consistency
    daisee_cols = df_daisee.columns.tolist()
    collected_cols = df_collected.columns.tolist()

    # Check for column consistency after adding source
    if set(daisee_cols) != set(collected_cols):
        print(f"Column mismatch in {split} set!")
        print("DAiSEE columns:", daisee_cols)
        print("Collected columns:", collected_cols)
        continue

    # Align column order just to be safe
    df_collected = df_collected[daisee_cols]

    # Merge both datasets
    df_combined = pd.concat([df_daisee, df_collected], ignore_index=True)

    # Save combined CSV
    out_csv = os.path.join(OUTPUT_FEATURES, f"{split}_features.csv")
    df_combined.to_csv(out_csv, index=False)

    print(f"Saved combined {split} features → {out_csv}")
    print(f"DAiSEE rows: {len(df_daisee)}, Collected rows: {len(df_collected)}, Total: {len(df_combined)}")

print("\nAll splits processed successfully!")
