import os
import pandas as pd
import json
from metrics import *
from config import Config
import openai


def read_data(file_name):

    #Reads data from a CSV or JSON file into a DataFrame.
    file_path = os.path.join(Config.DATA_DIR, file_name)

    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_name.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")

    return df


def apply_metric_functions(df, metric_functions):
    # Applies a list of metric functions to each row of the dataframe
    for func, data_class in metric_functions:
        func_name = func.__name__.replace('calculate_', '')

        def apply_func(row):
            kwargs = {field.name: row[field.name] for field in data_class.__dataclass_fields__.values() if
                      field.name in row}
            metric_instance = data_class(**kwargs)
            return func(metric_instance)

        df[func_name] = df.apply(apply_func, axis=1)

    return df


def save_results(df, output_file_name):

    #Saves the DataFrame to a CSV or JSON file.
    Config.create_output_directory()
    output_path = os.path.join(Config.OUTPUT_DIR, output_file_name)

    if output_file_name.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_file_name.endswith('.json'):
        df.to_json(output_path, orient='records', lines=True)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")

def calculate_and_save_average_scores(df, output_dir):

    #Calculate the average score for each metric and save to a text file.
    average_scores = df.mean()

    # Handle zero-shot prompts separately
    zeroshot_columns = [col for col in df.columns if 'zeroshot_prompt' in col]
    for col in zeroshot_columns:
        average_scores[col] = df[col].apply(pd.Series).mean().mean()

    avg_scores_path = os.path.join(output_dir, 'average_scores.txt')
    with open(avg_scores_path, 'w') as f:
        for metric, score in average_scores.items():
            f.write(f"{metric}: {score:.2f}\n")

def set_openai_api_key():

    #ask users to enter their API key
    api_key = input("Enter your OpenAI API key: ").strip()
    openai.api_key = api_key

def process_files(input_file_name, output_file_name, metric_functions):

    # Read input file, add scores, save output, and calculate averages.
    df = read_data(input_file_name)
    df = apply_metric_functions(df, metric_functions)
    save_results(df, output_file_name)
    calculate_and_save_average_scores(df, Config.OUTPUT_DIR)