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
        'RAG+RAIN_Flesh-Kincaid-Readability_BERT_FLESCH_READABILITY',
        'RAG+RAIN_Flesh-Kincaid-Readability_BERT_BERT',
        'RAG+RAIN_correctness_readability_CORRECTNESS',
        'RAG+RAIN_correctness_readability_READABILITY',
        'RAG+MultiRAIN_correctness_readability_MULTIRAIN'
        ]

    # Initialize an empty dictionary to hold the DataFrames for each answer type
    data_dict = {}

    # Loop through each answer type and collect relevant columns
    for answer_type in all_answer_types:
        # Select columns that start with the answer_type or are named 'Question'
        relevant_columns = [col for col in columns if col.startswith(answer_type) or col == 'Question']

        # Create a DataFrame with these relevant columns
        df = data[relevant_columns].copy()

        # Rename metric columns by removing the answer_type prefix
        df.columns = [col.replace(f"{answer_type}_", "") if col.startswith(answer_type) else col for col in df.columns]

        # Store the modified DataFrame in data_dict
        data_dict[answer_type] = df

    metrics = [col for col in data_dict.get('Designed_Answer_1').columns if col != 'Question' and col != 'Designed_Answer_1']

    # Initialize the thresholds DataFrame with the 'Question' column
    thresholds = pd.DataFrame()
    thresholds['Question'] = data_dict.get('Designed_Answer_1')['Question']

    # Iterate through the metrics and compute the appropriate threshold value
    for metric in metrics:
        if metric in data_dict.get('Designed_Answer_1').columns and metric in data_dict.get('Designed_Answer_2').columns:
            if metric == 'Readability':
                thresholds[metric] = data_dict.get('Designed_Answer_1')[metric].combine(data_dict.get('Designed_Answer_2')[metric], max)  # Use max for readability_grade
            else:
                thresholds[metric] = data_dict.get('Designed_Answer_1')[metric].combine(data_dict.get('Designed_Answer_2')[metric], min)  # Use min for all other metrics

    numeric_columns = thresholds.drop(columns=['Question'])

    counts_dict = {answer_type: pd.DataFrame() for answer_type in all_answer_types}

    # Iterate over each answer type to calculate counts
    for answer_type in all_answer_types:
        # Initialize counts DataFrame for the current answer type
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
                            # Handle cases where NaN values might be present
                            if pd.isna(threshold_value) or pd.isna(metric_values).any():
                                counts.at[index, metric] = np.nan
                            else:
                                if metric == 'Readability':
                                    # For readability_grade, count instances no larger than the threshold
                                    counts.at[index, metric] = int(np.all(metric_values <= threshold_value))
                                else:
                                    # For all other metrics, count instances at least as high as the threshold
                                    counts.at[index, metric] = int(np.all(metric_values >= threshold_value))
                        else:
                            counts.at[index, metric] = np.nan
                    else:
                        counts.at[index, metric] = np.nan
                else:
                    counts.at[index, metric] = np.nan

        # Store the counts DataFrame for the current answer type in the dictionary
        counts_dict[answer_type] = counts

    # Initialize an empty DataFrame to store percentages
    percentage_df = pd.DataFrame(index=all_answer_types, columns=metrics)

    # Iterate over each answer type and metric to compute percentages
    for answer_type in all_answer_types:
        counts_df = counts_dict[answer_type]  # Retrieve the counts DataFrame for the current answer type

        for metric in metrics:
            if metric in counts_df.columns:
                # Count the number of 1s and the number of non-NaN values
                num_ones = counts_df[metric].eq(1).sum()
                num_non_nan = counts_df[metric].notna().sum()

                # Compute the percentage
                if num_non_nan > 0:
                    percentage = (num_ones / num_non_nan) * 100
                else:
                    percentage = np.nan  # If there are no non-NaN values, set the percentage to NaN

                # Save the percentage in the DataFrame
                percentage_df.at[answer_type, metric] = percentage

    # Return the percentage DataFrame for use in another script
    return percentage_df