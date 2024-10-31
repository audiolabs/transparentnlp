import os
import json
import pandas as pd
import re
from tqdm import tqdm  # Import tqdm for progress bar
from configs import Config


# Loading JSON
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()


def process_data(data, data_dir, user_input, pa_exists, output_file):
    # Capitalize and format `data_dir`
    formatted_data_dir = data_dir.replace("_", " ").title().replace(" ", "_")

    # Initialize lists to hold extracted data
    question_list = []
    id_list = []
    title_list = []
    content_list = []
    raina_list = []

    # Combine the data extraction in a single loop for the dataset
    for item in tqdm(data, desc="Processing items", unit="item"):  # Add progress bar for items
        question = item["question"]
        documents = item["documents"]
        raina = item["raina"]

        num_documents = len(documents)

        # Extend lists with extracted data
        question_list.extend([question] * num_documents)
        id_list.extend([doc["id"] for doc in documents])
        title_list.extend([doc["title"] for doc in documents])
        content_list.extend([doc["content"] for doc in documents])
        raina_list.extend([raina] * num_documents)

    # Create DataFrame
    df = pd.DataFrame({
        'Question': question_list,
        'Document ID': id_list,
        'Title': title_list,
        'Content': content_list,
        'Raina': raina_list,
    })

    # Extract the specific responses from the text using compiled regex
    response_pattern = re.compile(r'(?:.*?Assistant:.*?){2}(?P<Response>.*)', re.DOTALL)

    # Determine column name based on user input
    if user_input.lower() == "multirain":
        column_name = f'RAG+MultiRAIN_{formatted_data_dir}'
    else:
        column_name = f'RAG+RAIN_{user_input.capitalize()}'

    # Update progress bar for applying the RAG+RAIN column
    tqdm.pandas(desc=f'Processing {column_name}')

    df[column_name] = df['Raina'].progress_apply(
        lambda x: response_pattern.search(x).group('Response') if response_pattern.search(x) else '')


    df = df.drop_duplicates(subset='Question', keep='first')
    df = df.drop(columns=['Raina', 'Content'])

    # Save the DataFrame to a TSV file with the user-specified name
    df.to_csv(os.path.join(os.path.join(Config.DATA_DIR, data_dir), output_file), sep='\t', index=False)


# Profiling the code
def main():
    # Ask the user for the data directory
    data_dir = input("Enter the path to the data directory: ")

    user_input = input("Enter the name of the JSON file (without res_ or .json): ")
    json_file = f'res_{user_input}.json'

    file_path = os.path.join(os.path.join(Config.DATA_DIR, data_dir), json_file)
    data = load_json(file_path)

    # Check for 'pa' existence
    pa_exists = any('pa' in item for item in data)

    # Ask the user for the output TSV file name
    output_file = input("Enter the name for the output TSV file (with .tsv extension): ")

    process_data(data, data_dir, user_input, pa_exists, output_file)


if __name__ == "__main__":
    main()
