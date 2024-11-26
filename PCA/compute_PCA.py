import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Prompt user for the input data directory
dat_dir = input("Enter the path to the data directory: ").strip()
file_path = os.path.join(dat_dir, 'all_metrics_scores_cleaned.tsv')

# Check if the input file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist in the specified directory.")

# Load the input data
data = pd.read_csv(file_path, sep='\t')

# Define the metrics and answer types
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
metrics = [
    'context_adherence', 'completeness', 'correctness', 'answer_relevancy', 
    'readability_LLM_eval_Trott', 'BLEU_Ex', 'ROUGE_Ex', 'BERTScore_Ex', 'STS_Ex',
    'BLEU_DA1', 'ROUGE_DA1', 'BERTScore_DA1', 'STS_DA1',
    'BLEU_DA2', 'ROUGE_DA2', 'BERTScore_DA2', 'STS_DA2',
    'Readability', 'ReadabilityGrade', 'LexicalDiversity', 'SentenceLength'
]

# Reshape the data
reshaped_rows = []
for _, row in data.iterrows():
    question = row['Question']
    for answer_type in all_answer_types:
        new_row = {'Question': question, 'Answer type': answer_type}
        for metric in metrics:
            col_name = f"{answer_type}_{metric}"
            new_row[metric] = row.get(col_name, None)
        reshaped_rows.append(new_row)

reshaped_df = pd.DataFrame(reshaped_rows)
reshaped_output_path = "reshaped_metrics_scores_with_answer_type.tsv"
reshaped_df.to_csv(reshaped_output_path, index=False, sep='\t')

# Invert readability grade and normalize metrics
if 'ReadabilityGrade' in reshaped_df.columns:
    reshaped_df['ReadabilityGrade'] = 20 - reshaped_df['ReadabilityGrade']

for metric in metrics:
    if metric in reshaped_df.columns:
        reshaped_df[metric] = (reshaped_df[metric] - reshaped_df[metric].mean()) / reshaped_df[metric].std()

normalized_output_path = "normalized_metrics_scores_with_inverted_readability_grade.tsv"
reshaped_df.to_csv(normalized_output_path, index=False, sep='\t')

# PCA Analysis
pca_data = reshaped_df[metrics].fillna(reshaped_df[metrics].mean())
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_data)

pca = PCA()
pca_results = pca.fit_transform(scaled_data)
explained_variance_ratio = pca.explained_variance_ratio_

# Plot Explained Variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance by Principal Components')
plt.savefig('PCA/Explained_Variance.png')
plt.close()

# PCA Loadings
loadings = pd.DataFrame(
    pca.components_[:2].T,
    columns=['PC1', 'PC2'],
    index=metrics
)
loadings_output_path = os.path.join("PCA", "PCA_Loadings_for_Normalized_Metrics.tsv")
loadings.to_csv(loadings_output_path, sep='\t')

# Scatter Plot with Colored Metrics
loadings['Category'] = np.where(
    (loadings['PC1'] < 0) & (loadings['PC2'] < 0), 'Negative PC1 & PC2',
    np.where(
        (loadings['PC1'] > 0) & (loadings['PC2'] > 0), 'Positive PC1 & PC2',
        np.where(
            (loadings['PC1'] > 0) & (loadings['PC2'] < 0), 'Positive PC1 & Negative PC2',
            'Negative PC1 & Positive PC2'
        )
    )
)
color_map = {
    'Negative PC1 & PC2': 'red',
    'Positive PC1 & PC2': 'blue',
    'Positive PC1 & Negative PC2': 'green',
    'Negative PC1 & Positive PC2': 'purple'
}
loadings['Color'] = loadings['Category'].map(color_map)

os.makedirs('PCA', exist_ok=True)

plt.figure(figsize=(10, 6))
for category, group in loadings.groupby('Category'):
    plt.scatter(group['PC1'], group['PC2'], label=category, color=color_map[category], alpha=0.8, edgecolor='k')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Projection with Colored Metrics')
plt.legend(title='Loading Patterns')
plt.grid(True)
plt.savefig('PCA/2D_PCA_Projection_with_Colored_Metrics.png')
plt.close()

print("Outputs generated and saved in the 'PCA' directory.")