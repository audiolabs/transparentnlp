import os
import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, ".")
from configs import Config

# Custom exception for file loading errors
class FileNotFoundErrorCustom(Exception):
    pass

# Loading JSON
def load_json(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundErrorCustom(f"Error: The file '{file_path}' was not found.")

def process_data(data, data_dir, json_file):
    """Process the data and save it to a TSV file with the same base name as the JSON file, minus any 'res_' prefix."""
    
    # Remove the "res_" prefix from the json file name if it exists
    file_name_base = json_file.stem
    if file_name_base.startswith("res_"):
        file_name_base = file_name_base[len("res_"):]

    # Determine if the file name contains multirain-related variants
    file_name = json_file.name.lower()
    if 'multirain' in file_name or 'multi_rain' in file_name or 'multi+rain' in file_name:
        prefix = 'RAG+MultiRAIN_'
    else:
        prefix = 'RAG+RAIN_'

    # Clean up the data directory name for use in the column name
    formatted_data_dir = Path(json_file.parent).name  # Get the directory name
    file_name_upper = file_name_base.upper()  # Convert file name to uppercase
    
    # Construct the column name
    column_name = f'{prefix}{formatted_data_dir}_{file_name_upper}'
    
    question_list, id_list, title_list, content_list, raina_list = [], [], [], [], []

    # Compile the regex pattern once
    response_pattern = re.compile(r'(?:.*?Assistant:.*?){2}(?P<Response>.*)', re.DOTALL)

    for item in tqdm(data, desc=f"Processing items in {json_file.name}", unit="item"):
        question = item["question"]
        documents = item["documents"]
        raina = item["raina"]
        num_documents = len(documents)

        question_list += [question] * num_documents
        id_list += [doc["id"] for doc in documents]
        title_list += [doc["title"] for doc in documents]
        content_list += [doc["content"] for doc in documents]
        raina_list += [raina] * num_documents

    # Create DataFrame
    df = pd.DataFrame({
        'Question': question_list,
        'Document ID': id_list,
        'Title': title_list,
        'Content': content_list,
        'Raina': raina_list,
    })

    # Add the extracted response as a new column with the appropriate name
    df[column_name] = df['Raina'].str.extract(response_pattern)['Response'].fillna('')

    # Drop duplicates and unwanted columns
    df = df.drop_duplicates(subset='Question', keep='first').drop(columns=['Raina', 'Content'])

    # Define the output TSV filename, removing "res_" if present
    output_file_name = json_file.stem.replace("res_", "") + ".tsv"
    save_to_tsv(df, data_dir, output_file_name)

def save_to_tsv(df, data_dir, output_file_name):
    """Save the DataFrame to a TSV file."""
    output_path = Path(Config.DATA_DIR) / data_dir / output_file_name
    df.to_csv(output_path, sep='\t', index=False)
    print(f"File saved to: {output_path}")

def process_all_json_files(data_dir):
    """Process all JSON files in the specified directory."""
    data_dir_path = Path(Config.DATA_DIR) / data_dir
    if not data_dir_path.is_dir():
        raise ValueError(f"The directory '{data_dir_path}' does not exist.")

    # Find all JSON files in the directory
    json_files = list(data_dir_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in directory '{data_dir_path}'.")
        return

    for json_file in json_files:
        try:
            # Load JSON data
            data = load_json(json_file)

            # Process and save each file
            process_data(data, data_dir, json_file)
        except (FileNotFoundErrorCustom, ValueError) as e:
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred while processing {json_file}: {e}")

def main():
    data_dir = input("Enter the path to the data directory: ").strip()
    process_all_json_files(data_dir)

if __name__ == "__main__":
    main()
