import pandas as pd
import ast

def process_dictionary_column(df, column_name, prefix):
    # Convert the string representation of the dictionary to an actual dictionary
    df[column_name] = df[column_name].apply(ast.literal_eval)

    # Iterate over each row and expand the dictionary into new columns
    for index, row in df.iterrows():
        for key, value in row[column_name].items():
            new_column_name = f"{prefix}_{key.lower().replace(' ', '_')}"
            if 'score:' in value.lower():
                value = value.replace('Score: ', '')
            try:
                df.at[index, new_column_name] = int(float(value))
            except (ValueError, TypeError):
                df.at[index, new_column_name] = 0  # Replace non-numeric values with zero

    # Drop the original column
    df = df.drop(columns=[column_name])
    return df

def calculate_column_averages(df):
    averages = {}

    # Calculate average for columns starting with 'zero_shot'
    zero_shot_cols = [col for col in df.columns if col.startswith('zero_shot')]
    for col in zero_shot_cols:
        averages[col] = df[col].mean()

    # Calculate average for columns starting with 'ragas'
    ragas_cols = [col for col in df.columns if col.startswith('ragas')]
    for col in ragas_cols:
        averages[col] = df[col].mean()

    return averages

# Get user input for the TSV filename
input_filename = input("Enter the TSV filename (with .tsv extension): ").strip()

# Load the CSV file
df = pd.read_csv(input_filename, sep="\t")

# Process the columns for df
df = process_dictionary_column(df, 'zero_shot', 'zero_shot')
df = process_dictionary_column(df, 'ragas', 'ragas')

# Calculate averages for df
averages = calculate_column_averages(df)

# Derive the output filenames from the input filename
base_filename = input_filename.replace('.tsv', '')

# Save the averages to a text file
with open(f'{base_filename}_averages.txt', 'w') as f:
    for key, value in averages.items():
        f.write(f"{key}: {value}\n")

# Save the transformed DataFrame to a CSV file
df.to_csv(f'{base_filename}_transformed.csv', index=False)
