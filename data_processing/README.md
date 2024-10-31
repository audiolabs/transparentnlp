## Data Processing Directory Overview
This directory contains two main scripts:

1. json_to_tsv_processing.py
2. merge_tsv_files_for_evaluation.py


## Script Descriptions

### json_to_tsv_processing.py:
This script guides the user through a step-by-step process to convert a JSON file into a TSV format:

Step 1: Enter the path to the data directory.
Step 2: Enter the name of the JSON file (omit res_ and .json).
Step 3: Enter the desired name for the output TSV file (include .tsv).


Example Input:

Path to data directory: correctness_readability
JSON file name (without prefix or extension): correctness
Output TSV file name: my_data_on_correctness.tsv

The TSV output will be saved in the directory specified in the first step. You
will also see a progress bar, so be patient. 

### merge_tsv_files_for_evaluation.py:

This script consolidates all TSV files within the specified directory, 
retaining only columns required for further analysis. You don’t need 
to modify or filter columns manually, as the script automatically 
selects the necessary ones for evaluation in subsequent steps.