import pandas as pd
import re
import sys
import os
sys.path.insert(0, ".")

# Load the data from the provided TSV file
# Get the directory path from the user
dat_dir = input("Enter the path to the data directory: ").strip()
# Concatenate directory path with the file name
file_path = os.path.join(dat_dir, 'all_metrics_scores.tsv')
# Load the data from the constructed file path
data = pd.read_csv(file_path, sep='\t')

# Remove "Excerpt" column if it exists
if "Excerpt" in data.columns:
    data = data.drop(columns=["Excerpt"])

# Function to clean cell values based on rules
def clean_cell(value):
    # Replace empty cells with "NA"
    if pd.isna(value) or value == "":
        return "NA"
    
    # Convert the value to string for regex matching
    value = str(value)
    
    # Remove trailing string if cell contains integer between 1 and 100 at the start
    match = re.match(r"^(\d{1,3})\b", value)
    if match:
        integer_value = int(match.group(1))
        if 1 <= integer_value <= 100:
            return str(integer_value)

    # Keep only "NA" if cell contains "NA" with trailing text
    if value.startswith("NA"):
        return "NA"

    # Keep only result of equation if cell contains "="
    if "=" in value:
        result_match = re.search(r"=\s*([\d.]+)", value)
        if result_match:
            return result_match.group(1)

    # Replace specific phrases with "NA"
    if value.strip() in [
        "The cat sat on the mat. It was a sunny day. The cat was happy.",
        "The text is missing. Please provide the text for evaluation."
    ]:
        return "NA"
    
    return value

# Apply the cleaning function to all columns except "Question"
for col in data.columns:
    if col != "Question":
        data[col] = data[col].apply(clean_cell)

# Convert columns to integer or NA as appropriate
for col in data.columns:
    if col != "Question":
        # Convert to integer where possible, otherwise keep as "NA"
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna("NA").astype(str)

# Save the cleaned data to a new TSV file
output_path = os.path.join(dat_dir, 'all_metrics_scores_cleaned.tsv')
data.to_csv(output_path, sep='\t', index=False)
