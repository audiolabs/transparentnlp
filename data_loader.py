import os
import json
import pandas as pd
import re
from configs import Config

#loading json
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()


def process_data(data, data2):
    # Initialize lists to hold extracted data
    question_list = []
    id_list = []
    title_list = []
    content_list = []
    raina_list = []
    pa_list = []
    raina_list2 = []

    # Combine the data extraction in a single loop for each dataset
    for item, item2 in zip(data, data2):
        question = item["question"]
        documents = item["documents"]
        raina = item["raina"]
        pa = item['pa']

        documents2 = item2["documents"]
        raina2 = item2["raina"]

        num_documents = len(documents)

        # Extend lists with extracted data, handling variability in document counts
        question_list.extend([question] * num_documents)
        id_list.extend([doc["id"] for doc in documents])
        title_list.extend([doc["title"] for doc in documents])
        content_list.extend([doc["content"] for doc in documents])
        raina_list.extend([raina] * num_documents)
        pa_list.extend([pa] * num_documents)
        raina_list2.extend([raina2] * num_documents)

    # Create DataFrame
    df = pd.DataFrame({
        'Question': question_list,
        'Document ID': id_list,
        'Title': title_list,
        'Content': content_list,
        'Raina': raina_list,
        'pa': pa_list,
        'Raina2': raina_list2[:len(question_list)]  # Aligning the size with the number of documents
    })

    # Extract the specific responses from the text using compiled regex
    response_pattern = re.compile(r'(?:.*?Assistant:.*?){2}(?P<Response>.*)', re.DOTALL)

    df['Rag_Rain_comp'] = df['Raina'].apply(
        lambda x: response_pattern.search(x).group('Response') if response_pattern.search(x) else '')
    df['Rag_Rain_dishon'] = df['Raina2'].apply(
        lambda x: response_pattern.search(x).group('Response') if response_pattern.search(x) else '')
    df['Rag'] = df['pa'].apply(
        lambda x: response_pattern.search(x).group('Response') if response_pattern.search(x) else '')

    # Save the DataFrame to a TSV file
    df.to_csv(os.path.join(Config.DATA_DIR,'json_files_to_tsv_loaded.tsv'), sep='\t', index=False)


# Profiling the code
def main():
    data = load_json(os.path.join(Config.DATA_DIR,'res_comprehensibility.json'))
    data2 = load_json(os.path.join(Config.DATA_DIR,'res_dishonesty.json'))
    process_data(data, data2)


if __name__ == "__main__":
    main()

