import os
import pandas as pd
from configs import Config
from pathlib import Path


def merge_tsv_files(stat_dir):
    """
    Merges exactly two .tsv files in a specified directory on the 'Question' column.
    Drops duplicate columns and saves the result as 'all_metrics_scores.tsv'.

    Parameters:
        stat_dir (str): Path to the directory containing .tsv files.

    Returns:
        bool: True if merge is successful and file is saved, False otherwise.
    """
    try:
        # Get all .tsv files in the specified directory
        tsv_files = [f for f in os.listdir(stat_dir) if f.endswith('.tsv')]

        # Check if exactly two .tsv files exist
        if len(tsv_files) != 2:
            print("Error: There should be exactly two .tsv files in the STAT_DIR.")
            return False

        # Load the .tsv files as DataFrames
        df1 = pd.read_csv(Path(stat_dir) / tsv_files[0], sep='\t')
        df2 = pd.read_csv(Path(stat_dir) / tsv_files[1], sep='\t')

        # Merge the two DataFrames on 'Question' column
        merged_df = pd.merge(df1, df2, on='Question', how='outer', suffixes=('', '_dup'))

        # Drop duplicate columns (those ending with '_dup')
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]

        # Save the merged DataFrame to a new .tsv file in the current directory
        output_file = Path('all_metrics_scores.tsv')
        merged_df.to_csv(output_file, sep='\t', index=False)
        print(f"Merged file saved as {output_file}")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    # Path to the STAT_DIR
    stat_dir = Config.STAT_DIR
    merge_tsv_files(stat_dir)
