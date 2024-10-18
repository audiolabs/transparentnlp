import os
import re
import pandas as pd
from configs import Config

# Load the dataset
df = pd.read_csv(os.path.join(Config.DATA_DIR, 'updated_readability_bertscore_thresholds.tsv'), sep='\t')




# Find columns that start with "Designed_Answer" and end with "Readability" or "bert"
readability_columns = [col for col in df.columns if col.endswith('Readability')]
bert_columns = [col for col in df.columns if col.endswith('bert')]

# Calculate row-wise average for "Readability" and "bert"
df['Readability_Avg'] = df[readability_columns].mean(axis=1)
df['bert_Avg'] = df[bert_columns].mean(axis=1)

# Display the dataframe with the new average columns
df[['Readability_Avg', 'bert_Avg']].to_csv(os.path.join(Config.DATA_DIR, 'readbility_bertscore_mean.tsv'),sep='\t')
