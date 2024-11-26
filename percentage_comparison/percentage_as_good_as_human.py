import pandas as pd
import numpy as np

def calculate_percentage_comparisons(dataset):

    data = pd.read_csv(dataset, sep='\t')

    # Get the list of all column names
    columns = data.columns.tolist()

    # Define default answer types
    all_answer_types = [
        'Designed_Answer_1',
        'Designed_Answer_2',
        'Answer_Alexa',
        'VanillaRAG',
        'RAG+RAIN_honesty_comprehensibility_DISHONESTY',
        'RAG+RAIN_honesty_comprehensibility_COMPREHENSIBILITY',
        'RAG+RAIN_correctness_readability_CORRECTNESS',
        'RAG+RAIN_correctness_readability_READABILITY',
        'RAG+MultiRAIN_correctness_readability_MULTIRAIN',
        'RAG+RAIN_Flesh-Kincaid-Readability_BERT_FLESCH_READABILITY',
        'RAG+RAIN_Flesh-Kincaid-Readability_BERT_BERT',
        'RAG+MultiRAIN_Flesh-Kincaid-Readability_BERT_MULTIRAIN_DETERMINISTIC'

    ]

    # Initialize an empty dictionary to hold the DataFrames for each answer type
    data_dict = {}

    # Loop through each answer type and collect relevant columns
    for answer_type in all_answer_types:
        relevant_columns = [col for col in columns if col.startswith(answer_type) or col == 'Question']
        df = data[relevant_columns].copy()
        df.columns = [col.replace(f"{answer_type}_", "") if col.startswith(answer_type) else col for col in df.columns]
        data_dict[answer_type] = df

    metrics = [col for col in data_dict.get('Designed_Answer_1').columns if col != 'Question' and col != 'Designed_Answer_1']

    # Initialize the thresholds DataFrame with the 'Question' column
    thresholds = pd.DataFrame()
    thresholds['Question'] = data_dict.get('Designed_Answer_1')['Question']

    # Iterate through the metrics and compute the appropriate threshold value
    for metric in metrics:
        if metric in data_dict.get('Designed_Answer_1').columns and metric in data_dict.get('Designed_Answer_2').columns:
            if metric == 'ReadabilityGrade':
                thresholds[metric] = data_dict.get('Designed_Answer_1')[metric].combine(data_dict.get('Designed_Answer_2')[metric], max)
            elif metric in ['SentenceLength', 'LexicalDiversity']:
                # Calculate the actual min and max values for each question
                thresholds[metric] = list(zip(
                data_dict.get('Designed_Answer_1')[metric].combine(data_dict.get('Designed_Answer_2')[metric], min),
                data_dict.get('Designed_Answer_1')[metric].combine(data_dict.get('Designed_Answer_2')[metric], max)
                ))
            else:
                thresholds[metric] = data_dict.get('Designed_Answer_1')[metric].combine(data_dict.get('Designed_Answer_2')[metric], min)

    # Initialize counts dictionary
    counts_dict = {answer_type: pd.DataFrame() for answer_type in all_answer_types}

    # Iterate over each answer type to calculate counts
    for answer_type in all_answer_types:
        counts = pd.DataFrame()
        counts['Question'] = thresholds['Question']

        for metric in metrics:
            counts[metric] = np.nan  # Initialize counts for each metric

            for index, row in thresholds.iterrows():
                question_id = row['Question']
                threshold_value = row[metric]

                # Check the current answer type DataFrame
                if metric in data_dict[answer_type].columns:
                    df_subset = data_dict[answer_type][data_dict[answer_type]['Question'] == question_id]
                    if not df_subset.empty:
                        metric_values = df_subset[metric].values
                        if len(metric_values) > 0:
                            if pd.isna(threshold_value) or pd.isna(metric_values).any():
                                counts.at[index, metric] = np.nan
                            else:
                                if metric == 'ReadabilityGrade':
                                    counts.at[index, metric] = int(np.all(metric_values <= threshold_value))
                                elif metric in ['SentenceLength', 'LexicalDiversity']:
                                    min_val, max_val = threshold_value  # Unpack the interval
                                    counts.at[index, metric] = int(np.all((metric_values >= min_val) & (metric_values <= max_val)))
                                else:
                                    counts.at[index, metric] = int(np.all(metric_values >= threshold_value))
                        else:
                            counts.at[index, metric] = np.nan
                    else:
                        counts.at[index, metric] = np.nan
                else:
                    counts.at[index, metric] = np.nan

        counts_dict[answer_type] = counts

    # Initialize an empty DataFrame to store percentages
    percentage_df = pd.DataFrame(index=all_answer_types, columns=metrics)

    # Iterate over each answer type and metric to compute percentages
    for answer_type in all_answer_types:
        counts_df = counts_dict[answer_type]

        for metric in metrics:
            if metric in counts_df.columns:
                num_ones = counts_df[metric].eq(1).sum()
                num_non_nan = counts_df[metric].notna().sum()
                if num_non_nan > 0:
                    percentage = (num_ones / num_non_nan) * 100
                else:
                    percentage = np.nan
                percentage_df.at[answer_type, metric] = percentage

    # Return the percentage DataFrame for use in another script
    return percentage_df