import pandas as pd
import os

# SCRIPT 1 FOR DAiSEE Dataset
# Description: This script prepares the DAiSEE label file by adding a
# "CustomLabel" column derived from the original scores.
#
# Each video clip in DAiSEE has multiple affective state ratings
# (Engagement, Boredom, Confusion, Frustration).
# This step maps them into broader behavioral categories to simplify training:
#   - Fatigued:  High boredom (≥2) + low engagement (≤1)
#   - Focused:   High engagement (≥2)
#   - Distracted: All other combinations
#
# The output is a new CSV file 'AllLabels_Custom.csv' inside DAiSEE/Labels folder
# that will be used for stratified splitting and dataset organization in the next step.


# Define the file path to the DAiSEE dataset label file
LABEL_PATH = os.path.join("DAiSEE", "Labels", "AllLabels.csv")

# Load the CSV file into a pandas DataFrame
# 'sep=","' ensures that the file is parsed using commas as delimiters
# 'engine="python"' makes parsing more robust for some edge cases
df = pd.read_csv(LABEL_PATH, sep=",", engine="python")

# Clean up any unwanted spaces or hidden characters from column names
df.columns = df.columns.str.strip()

# Print the detected column names for verification
print("Columns detected:", df.columns.tolist())

# Display the first few rows of the dataset to confirm that the data loaded correctly
print("First few rows:")
print(df.head())

# Define a function to assign custom labels based on engagement and boredom levels
def map_custom_label(row):
    # Label as "Fatigued" if boredom is high (>=2) and engagement is low (<=1)
    if row['Boredom'] >= 2 and row['Engagement'] <= 1:
        return 'Fatigued'
    # Label as "Focused" if engagement is high (>=2)
    elif row['Engagement'] >= 2:
        return 'Focused'
    # Label as "Distracted" for all other combinations
    else:
        return 'Distracted'

# Apply the mapping function to each row in the DataFrame
# axis=1 means the function operates across rows, not columns
df['CustomLabel'] = df.apply(map_custom_label, axis=1)

# Define the output path for the new CSV file with custom labels
output_path = os.path.join("DAiSEE", "Labels", "AllLabels_Custom.csv")

# Save the updated DataFrame (with the new CustomLabel column) to a new CSV file
df.to_csv(output_path, index=False)

# Print a completion message and show where the file was saved
print(f"\nCustom label mapping complete! Saved to: {output_path}")

# Display the count of each label category for quick verification
print(df['CustomLabel'].value_counts())
