import os
import re
import pandas as pd
from configs import Config

# Load the dataset
df = pd.read_csv(os.path.join(Config.DATA_DIR, 'final_dataset_with_non_proactive_answers.tsv'), sep='\t')


# Remove the 'Questions' column
df = df.drop(columns=['Questions'])

designed_columns = ['Designed_Answer_1', 'Designed_Answer_2']
non_proactive_columns = [
    'Designed_Answer_Non_Proactive_1', 'Designed_Answer_Non_Proactive_2',
    'Designed_Answer_Non_Proactive_3', 'Designed_Answer_Non_Proactive_4', 'Designed_Answer_Non_Proactive_5'
]


# Function to remove punctuation from a string
def remove_punctuation(text):
    if isinstance(text, str):  # Only process if text is a string
        return re.sub(r'[^\w\s]', '', text)
    return text  # Return the value as is if it's not a string


# Function to shift non-empty values left to fill empty columns
def shift_non_proactive_answers(row):
    non_proactive_values = [row[col] for col in non_proactive_columns if
                            isinstance(row[col], str) and row[col].strip()]  # Get non-empty strings
    non_proactive_values += [''] * (len(non_proactive_columns) - len(non_proactive_values))  # Pad with empty strings
    for i, col in enumerate(non_proactive_columns):
        row[col] = non_proactive_values[i]  # Reassign values, shifting them left
    return row


# Function to remove matching values row by row, ignoring punctuation
def remove_matching_values(row):
    # Remove punctuation from Designed_Answer_1 and Designed_Answer_2
    designed_answers = {remove_punctuation(row['Designed_Answer_1']), remove_punctuation(row['Designed_Answer_2'])}

    # Iterate through the non-proactive columns and remove matching values (ignoring punctuation)
    for col in non_proactive_columns:
        if isinstance(row[col], str) and remove_punctuation(row[col]) in designed_answers:
            row[col] = ''  # Remove matching value by setting it to an empty string

    # After removing, shift non-proactive answers to the left to fill gaps
    row = shift_non_proactive_answers(row)

    return row


# Apply the function row by row
df = df.apply(remove_matching_values, axis=1)


# Function to check if a column is completely empty (all values are empty or NaN)
def is_empty_column(col):
    if col.dtype == 'object':  # Only apply string operations to object (string) columns
        return col.apply(lambda x: isinstance(x, str) and x.strip() == '').all() or col.isnull().all()
    else:
        return col.isnull().all()  # For non-string columns, just check if all values are NaN

# Identify completely empty columns
empty_columns = df.columns[df.apply(is_empty_column)]

# Drop empty columns
df = df.drop(columns=empty_columns)

df.to_csv(os.path.join(Config.DATA_DIR, 'updated_final_dataset_with_non_proactive_answers.tsv'), sep='\t')

