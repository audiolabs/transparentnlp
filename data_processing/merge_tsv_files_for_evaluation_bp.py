import os
import pandas as pd
import re
from functools import reduce
from tqdm import tqdm  # For progress bar
import sys
sys.path.insert(0, ".")
from configs import Config

def clean_question_column(df):
    """
    Cleans the 'Question' column by:
    - Stripping leading/trailing spaces
    - Converting the text to lowercase to standardize
    - Ensuring each question ends with a question mark
    - Capitalizing the first letter of each question

    Args:
        df (pd.DataFrame): The input DataFrame with a 'Question' column.

    Returns:
        pd.DataFrame: The DataFrame with a cleaned 'Question' column.
    """
    df['Question'] = df['Question'].str.strip()  # Remove leading/trailing spaces
    df['Question'] = df['Question'].str.lower()  # Convert to lowercase to standardize

    # Ensure the question ends with a "?" and capitalize the first letter
    df['Question'] = df['Question'].apply(lambda x: x.capitalize() + '?' if not x.endswith('?') else x.capitalize())

    return df


def merge_tsv_files(data_dir):
    """
    Merges TSV files from the specified directory and its subdirectories.
    It cleans the 'Question' column and merges the data on the 'Question' column.

    Args:
        data_dir (str): The path to the directory containing TSV files.

    Returns:
        None: Saves the merged DataFrame to a TSV file.
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(data_dir, 'dataset_for_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize an empty list to hold DataFrames
    dataframes = []

    # Walk through the directory and its subdirectories
    for subdir, _, files in os.walk(data_dir):
        # Skip 'dataset_for_analysis' subdirectory
        if os.path.basename(subdir) == 'dataset_for_analysis':
            continue

        for file in files:
            if file.endswith('.tsv'):
                file_path = os.path.join(subdir, file)

                try:
                    # Read the TSV file
                    df = pd.read_csv(file_path, sep='\t')
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                # Ensure 'Question' column exists
                if 'Question' in df.columns:
                    # Clean the 'Question' column
                    df = clean_question_column(df)
                    # Append the DataFrame to the list
                    dataframes.append(df)
                else:
                    print(f"Skipping {file_path} as it does not contain the 'Question' column.")

    if not dataframes:
        print("No TSV files found in the specified directory with the 'Question' column.")
        return

    # Merge all DataFrames efficiently
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Question', how='outer', suffixes=('', '_dup')), dataframes)

    # Drop duplicate columns
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Selecting required columns
    columns_to_keep = [
        'Question', 'Answer_Alexa', 'Excerpts', 'Designed_Answer_1',
        'Designed_Answer_2', 'Designed_Answer_Non_Proactive_1',
        'Designed_Answer_Non_Proactive_2', 'Designed_Answer_Non_Proactive_3', 'VanillaRAG'
    ]

    # Add columns starting with 'RAG+RAIN_' or 'RAG+MultiRAIN_'
    rag_columns = [col for col in merged_df.columns if col.startswith('RAG+RAIN_') or col.startswith('RAG+MultiRAIN_')]
    columns_to_keep.extend(rag_columns)

    # Filter the DataFrame to keep only the selected columns
    final_df = merged_df[columns_to_keep]

    # Save the final DataFrame to a TSV file
    output_file_path = os.path.join(output_dir, 'dataset_for_evaluation.tsv')
    final_df.to_csv(output_file_path, sep='\t', index=False)
    print(f"Merged dataset saved to {output_file_path}")


if __name__ == "__main__":
    # Use the data directory specified in Config
    data_dir = Config.DATA_DIR
    print(f"Processing TSV files in directory: {data_dir}")
    merge_tsv_files(data_dir)
