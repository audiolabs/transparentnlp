import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 24,             # General font size
    'axes.titlesize': 28,        # Title font size
    'axes.labelsize': 24,        # Axis label font size
    'xtick.labelsize': 20,       # X-tick label font size
    'ytick.labelsize': 20,       # Y-tick label font size
    'legend.fontsize': 20,       # Legend font size
    'figure.titlesize': 28       # Figure title font size
})
from adjustText import adjust_text

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
reshaped_output_path = "PCA/reshaped_metrics_scores_with_answer_type.tsv"
reshaped_df.to_csv(reshaped_output_path, index=False, sep='\t')

# Invert readability grade and normalize metrics
if 'ReadabilityGrade' in reshaped_df.columns:
    reshaped_df['ReadabilityGrade'] = 20 - reshaped_df['ReadabilityGrade']

for metric in metrics:
    if metric in reshaped_df.columns:
        reshaped_df[metric] = (reshaped_df[metric] - reshaped_df[metric].mean()) / reshaped_df[metric].std()

normalized_output_path = "PCA/normalized_metrics_scores_with_inverted_readability_grade.tsv"
reshaped_df.to_csv(normalized_output_path, index=False, sep='\t')

# PCA Analysis
pca_data = reshaped_df[metrics].fillna(reshaped_df[metrics].mean())
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_data)

pca = PCA()
pca_results = pca.fit_transform(scaled_data)
explained_variance_ratio = pca.explained_variance_ratio_

# Plot Explained Variance
plt.figure(figsize=(16, 12))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance by Principal Components')
plt.savefig('PCA/Explained_Variance.pdf')
plt.close()

# PCA Loadings
loadings = pd.DataFrame(
    pca.components_[:2].T,
    columns=['PC1', 'PC2'],
    index=metrics
)
loadings_output_path = os.path.join("PCA", "PCA_Loadings_for_Normalized_Metrics.tsv")
loadings.to_csv(loadings_output_path, sep='\t')

#### Scatter Plot with Colored Metrics
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
    'Negative PC1 & PC2': 'blue',
    'Positive PC1 & PC2': 'red',
    'Positive PC1 & Negative PC2': 'purple',
    'Negative PC1 & Positive PC2': 'green'
}
loadings['Color'] = loadings['Category'].map(color_map)

os.makedirs('PCA', exist_ok=True)

plt.figure(figsize=(16, 12))
for category, group in loadings.groupby('Category'):
    plt.scatter(group['PC1'], group['PC2'], label=category, color=color_map[category], alpha=0.8, edgecolor='k', s=200)

plt.xlabel('Principal Component 1', fontsize=28)  
plt.ylabel('Principal Component 2', fontsize=28)  
plt.xticks(fontsize=25)  # Increased tick label size
plt.yticks(fontsize=25)  # Increased tick label size
plt.title('2D PCA Projection with Colored Metrics', fontsize=28)
plt.legend(title='Loading Patterns', loc='lower right')  # Adjusted legend position
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')  # Light grey dashed grid
plt.savefig('PCA/2D_PCA_Projection_with_Colored_Metrics.pdf')
plt.close()

#### Scatter Plot with Metric Names
os.makedirs('PCA', exist_ok=True)

plt.figure(figsize=(16, 12))

# Plot data points without color coding or groups
plt.scatter(loadings['PC1'], loadings['PC2'], color='black', alpha=0.7, edgecolor='k', s=100)

# Add metric names near each point
texts = []
for i, metric in enumerate(loadings.index):
    texts.append(plt.text(loadings['PC1'][i], loadings['PC2'][i], metric, fontsize=24))

# Adjust text to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

# Formatting the plot
plt.xlabel('Principal Component 1', fontsize=28)
plt.ylabel('Principal Component 2', fontsize=28)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('2D PCA Projection with Metric Names', fontsize=28)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')  # Light grey dashed grid

# Save the plot
plt.savefig('PCA/2D_PCA_Projection_with_metric_names.pdf')
plt.close()


#### Scatter Plot with Preciseness and Comprehensibility category assignment for metrics
# Define Categories and Colors
preciseness_metrics = [
    'context_adherence', 'completeness', 'correctness', 'answer_relevancy',
    'BLEU_Ex', 'ROUGE_Ex', 'BERTScore_Ex', 'STS_Ex',
    'BLEU_DA1', 'ROUGE_DA1', 'BERTScore_DA1', 'STS_DA1',
    'BLEU_DA2', 'ROUGE_DA2', 'BERTScore_DA2', 'STS_DA2'
]
comprehensibility_metrics = [
    'Readability', 'ReadabilityGrade', 'LexicalDiversity', 'SentenceLength', 'readability_LLM_eval_Trott'
]

color_map = {
    'Assumed Comprehensibility metrics': 'orange',
    'Assumed Preciseness metrics': 'darkgreen'
}

# Scatter Plot 
os.makedirs('PCA', exist_ok=True)

plt.figure(figsize=(16, 12))

# Plot Comprehensibility Metrics
for metric in comprehensibility_metrics:
    if metric in loadings.index:
        idx = loadings.index.get_loc(metric)
        plt.scatter(
            loadings['PC1'][idx], loadings['PC2'][idx],
            color=color_map['Assumed Comprehensibility metrics'], label='Assumed Comprehensibility metrics',
            alpha=0.7, edgecolor='k', s=200
        )

# Plot Preciseness Metrics
for metric in preciseness_metrics:
    if metric in loadings.index:
        idx = loadings.index.get_loc(metric)
        plt.scatter(
            loadings['PC1'][idx], loadings['PC2'][idx],
            color=color_map['Assumed Preciseness metrics'], label='Assumed Preciseness metrics',
            alpha=0.7, edgecolor='k', s=200
        )

# Remove duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.5), fontsize=22, title='Metric Categories', title_fontsize=24)

# Format plot
plt.xlabel('Principal Component 1', fontsize=28)
plt.ylabel('Principal Component 2', fontsize=28)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('2D PCA Projection: Preciseness and Comprehensibility', fontsize=28)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')  # Leicht graues, gestricheltes Gitter

# Save Plot
plt.savefig('PCA/2D_PCA_Projection_preciseness_comprehensibility.pdf')
plt.close()

#### Scatter Plot with Covariance Clusters
# Define Categories and Colors
categories = {
    'LLM and Deterministic Metrics (Privacy Notice)': [
        'context_adherence', 'completeness', 'answer_relevancy', 'BLEU_Ex', 'ROUGE_Ex', 'BERTScore_Ex', 'STS_Ex'
    ],
    'Deterministic Correctness Metrics (Human Answers)': [
        'BLEU_DA1', 'ROUGE_DA1', 'BERTScore_DA1', 'STS_DA1',
        'BLEU_DA2', 'ROUGE_DA2', 'BERTScore_DA2', 'STS_DA2'
    ],
    'Correctness': ['correctness'],
    'Readability': ['Readability', 'ReadabilityGrade', 'readability_LLM_eval_Trott'],
    'Structural Text Features': ['LexicalDiversity', 'SentenceLength']
}

# Assign colors to categories
color_map = {
    'LLM and Deterministic Metrics (Privacy Notice)': 'blue',
    'Deterministic Correctness Metrics (Human Answers)': 'green',
    'Correctness': 'red',
    'Readability': 'purple',
    'Structural Text Features': 'orange'
}

# Prepare the figure
os.makedirs('PCA', exist_ok=True)
plt.figure(figsize=(12, 16))

# Iterate through categories and plot each metric
for category, metrics in categories.items():
    for metric in metrics:
        if metric in loadings.index:
            idx = loadings.index.get_loc(metric)
            plt.scatter(
                loadings['PC1'][idx], loadings['PC2'][idx],
                color=color_map[category], alpha=0.7, edgecolor='k', s=200,
                label=category if metric == metrics[0] else ""  # Avoid duplicate legend entries
            )

# Remove duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend = plt.legend(
    by_label.values(), by_label.keys(),
    loc='upper center', bbox_to_anchor=(0.5, 1.5), fontsize=22,
    title='Metric Categories', title_fontsize=24, ncol=1  # Adjust for multi-column layout
)

# Format the plot
plt.xlabel('Principal Component 1', fontsize=28)
plt.ylabel('Principal Component 2', fontsize=28)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('2D PCA Projection: Covariance Clusters', fontsize=28)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')  # Light grey dashed grid

# Adjust layout to provide space for the legend
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leaves space above the plot for the legend

# Save the plot
plt.savefig('PCA/2D_PCA_Projection_covariance_clusters.pdf', bbox_inches="tight")
plt.close()

#### Scatter Plot indicating Computational Costs

# Define categories for human gold standard
gold_standard_metrics = [
    'BLEU_DA1', 'ROUGE_DA1', 'BERTScore_DA1', 'STS_DA1',
    'BLEU_DA2', 'ROUGE_DA2', 'BERTScore_DA2', 'STS_DA2'
]
no_gold_standard_metrics = [
    'LexicalDiversity', 'SentenceLength', 'context_adherence', 'completeness', 
    'answer_relevancy', 'BLEU_Ex', 'ROUGE_Ex', 'BERTScore_Ex', 'STS_Ex', 
    'correctness', 'Readability', 'ReadabilityGrade', 'readability_LLM_eval_Trott'
]

# Define computational cost levels
very_low_cost = ['LexicalDiversity', 'SentenceLength', 'Readability', 'ReadabilityGrade']
low_cost = ['BLEU_DA1', 'BLEU_DA2', 'BLEU_Ex', 'ROUGE_DA1', 'ROUGE_DA2', 'ROUGE_Ex']
medium_cost = ['BERTScore_DA1', 'BERTScore_DA2', 'BERTScore_Ex', 'STS_DA1', 'STS_DA2', 'STS_Ex']
high_cost = ['readability_LLM_eval_Trott', 'context_adherence', 'completeness', 'answer_relevancy', 'correctness']

# Map computational cost to circle sizes
size_map = {
    'Very low': 50,
    'Low': 150,
    'Medium': 450,
    'High': 1350
}

# Map computational cost levels to metrics
cost_category_map = {
    'Very low': very_low_cost,
    'Low': low_cost,
    'Medium': medium_cost,
    'High': high_cost
}

# Assign colors for human gold standard categorization
color_map = {
    'Gold Standard': 'lightgrey',
    'No Gold Standard': 'black'
}

# Prepare the figure
os.makedirs('PCA', exist_ok=True)
plt.figure(figsize=(16, 12)) 

# Scatter plot for all metrics
for metric in loadings.index:
    # Determine gold standard color
    if metric in gold_standard_metrics:
        color = color_map['Gold Standard']
    elif metric in no_gold_standard_metrics:
        color = color_map['No Gold Standard']
    else:
        continue  # Skip metrics not in either category

    # Determine size based on computational cost
    size = 50  # Default size
    for cost_category, metrics in cost_category_map.items():
        if metric in metrics:
            size = size_map[cost_category]
            break

    # Plot the metric
    idx = loadings.index.get_loc(metric)
    plt.scatter(
        loadings['PC1'][idx], loadings['PC2'][idx],
        color=color, alpha=0.6, edgecolor='k', s=size, label=metric
    )

# First legend: Gold standard categorization
gold_standard_handles = [
    plt.Line2D([], [], marker='o', color='lightgrey', label='Requires Human Gold Standard', markersize=10, alpha=0.6),
    plt.Line2D([], [], marker='o', color='black', label='Does Not Require Human Gold Standard', markersize=10, alpha=0.6)
]
legend1 = plt.legend(
    handles=gold_standard_handles, loc='upper left', bbox_to_anchor=(-0.1, 1.25), 
    fontsize=22, title='Gold Standard Categorization', title_fontsize=24
)

# Add first legend to the plot
plt.gca().add_artist(legend1)

# Second legend: Computational cost
cost_handles = [
    plt.Line2D([], [], marker='o', color='black', markersize=size_map['Very low']**0.5, alpha=0.6, label='Very Low Cost'),
    plt.Line2D([], [], marker='o', color='black', markersize=size_map['Low']**0.5, alpha=0.6, label='Low Cost'),
    plt.Line2D([], [], marker='o', color='black', markersize=size_map['Medium']**0.5, alpha=0.6, label='Medium Cost'),
    plt.Line2D([], [], marker='o', color='black', markersize=size_map['High']**0.5, alpha=0.6, label='High Cost')
]
plt.legend(
    handles=cost_handles, loc='upper center', bbox_to_anchor=(0.8, 1.25),
    fontsize=22, title='Computational Cost', title_fontsize=24, ncol=2
)

# Formatting the plot
plt.xlabel('Principal Component 1', fontsize=28)
plt.ylabel('Principal Component 2', fontsize=28)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('2D PCA Projection: Gold Standard and Computational Cost', fontsize=28)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')  # Light grey dashed grid

# Save the plot with tight layout
plt.savefig('PCA/2D_PCA_Projection_computational_cost.pdf', bbox_inches="tight")
plt.close()


print("Outputs generated and saved in the 'PCA' directory.")