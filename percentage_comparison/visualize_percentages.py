import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import colors as mcolors
from percentage_as_good_as_human import calculate_percentage_comparisons

# Define constants and metric groups
GROUPS = {
    "LLM_as_judge_metrics": ['context_adherence', 'completeness', 'correctness', 'answer_relevancy', 'readability_LLM_eval_Trott'],
    "statistical_metrics_for_correctness_excerpt-baseline": ['BLEU_Ex', 'ROUGE_Ex', 'BERTScore_Ex', 'STS_Ex'],
    "statistical_metrics_for_correctness_human-answers1-baseline": ['BLEU_DA1', 'ROUGE_DA1', 'BERTScore_DA1', 'STS_DA1'],
    "statistical_metrics_for_correctness_human-answers2-baseline": ['BLEU_DA2', 'ROUGE_DA2', 'BERTScore_DA2', 'STS_DA2'],
    "statistical_metrics_for_readability": ['Readability', 'ReadabilityGrade', 'LexicalDiversity', 'SentenceLength']
}
COLOR_MAPPING = {
    'Designed Answer 1': 'gold', 'Designed Answer 2': 'gold', 'VanillaRAG': 'red',
    'Dishonesty': '#1f77b4', 'Comprehensibility': '#17becf', 'Correctness': '#98df8a',
    'Readability': '#2ca02c', 'Multirain': '#4F7942', 'Flesch Readability': '#e377c2', 'BERT': '#E0218A', 'Multirain Deterministic': '#8e3563'
}

# Define custom legend
def get_custom_legend():
    return [
        plt.Line2D([0], [0], color='#1f77b4', lw=4, label="Experiment 1 with LLM-as-a-judge\nDistinct definitions"),
        plt.Line2D([0], [0], color='#2ca02c', lw=4, label="Experiment 2 with LLM-as-a-judge\nSimilar definitions"),
        plt.Line2D([0], [0], color='#e377c2', lw=4, label="Experiment 3 with statistical metrics\nSimilar definitions")
    ]

# Prepare the data for plotting
def prepare_data(file_path):
    percentage_df = calculate_percentage_comparisons(dataset=file_path)
    ordered_columns = [metric for group in GROUPS.values() for metric in group]
    percentage_df = percentage_df[ordered_columns]
    percentage_df = percentage_df.rename(index={
        'Designed_Answer_1': 'Designed Answer 1', 'Designed_Answer_2': 'Designed Answer 2', 'Answer_Alexa': 'Answer Alexa',
        'VanillaRAG': 'VanillaRAG', 'RAG+RAIN_honesty_comprehensibility_DISHONESTY': 'Dishonesty',
        'RAG+RAIN_honesty_comprehensibility_COMPREHENSIBILITY': 'Comprehensibility',
        'RAG+RAIN_correctness_readability_CORRECTNESS': 'Correctness', 'RAG+RAIN_correctness_readability_READABILITY': 'Readability',
        'RAG+MultiRAIN_correctness_readability_MULTIRAIN': 'Multirain', 'RAG+RAIN_Flesh-Kincaid-Readability_BERT_FLESCH_READABILITY': 'Flesch Readability',
        'RAG+RAIN_Flesh-Kincaid-Readability_BERT_BERT': 'BERT', 'RAG+MultiRAIN_Flesh-Kincaid-Readability_BERT_MULTIRAIN_DETERMINISTIC': 'Multirain Deterministic'
    })
    colors = [COLOR_MAPPING.get(answer_type, 'gray') for answer_type in percentage_df.index]
    return percentage_df, colors

# Plotting functions
def plot_individual_metric(data, colors, metric, group_name, save_dir):
    plt.figure(figsize=(6, 4))
    plt.bar(data.index, data[metric], color=colors)
    plt.ylabel('Percentage (%)')
    plt.title(metric)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.xlabel('Answer Type')
    plt.savefig(os.path.join(save_dir, f"{group_name}_{metric}.png"), bbox_inches='tight')
    plt.close()

def plot_group_metrics(data, colors, group_name, metrics, save_dir):
    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 4, 5))
    if len(metrics) == 1:
        axes = [axes]  # To ensure axes is iterable

    for ax, metric in zip(axes, metrics):
        ax.bar(data.index, data[metric], color=colors)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(metric)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Answer Type')
    
    fig.suptitle(group_name.replace("_", " ").title(), fontsize=16)
    fig.legend(handles=get_custom_legend(), bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{group_name}.png"), bbox_inches='tight')
    plt.close()

def plot_combined_metrics(data, colors, save_dir):
    num_metrics = sum(len(group) for group in GROUPS.values())
    rows = (num_metrics // 4) + (1 if num_metrics % 4 != 0 else 0)  # 4 metrics per row
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    axes = axes.flatten()  # Flatten axes for simple indexing
    plot_index = 0

    for group_name, metrics in GROUPS.items():
        for metric in metrics:
            if plot_index >= len(axes):  # Avoid out-of-bounds errors
                break
            ax = axes[plot_index]
            ax.bar(data.index, data[metric], color=colors)
            ax.set_ylabel('Percentage (%)')
            ax.set_title(metric)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(range(len(data.index)))
            ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=8)
            ax.set_xlabel('Answer Type')
            plot_index += 1

    for j in range(plot_index, len(axes)):
        axes[j].set_visible(False)  # Hide unused axes

    fig.suptitle('Percentage Metrics Visualization (Combined)', fontsize=16)
    fig.legend(handles=get_custom_legend(), bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10, frameon=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(os.path.join(save_dir, 'percentage_metrics_combined.png'), bbox_inches='tight')
    plt.close()

# Main function to generate all plots
def generate_all_plots(file_path, save_dir="images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    percentage_df, colors = prepare_data(file_path)
    
    # Combined plot of all metrics
    plot_combined_metrics(percentage_df, colors, save_dir)
    
    # Separate overall plots and individual metric plots for each group
    for group_name, metrics in GROUPS.items():
        plot_group_metrics(percentage_df, colors, group_name, metrics, save_dir)
        for metric in metrics:
            plot_individual_metric(percentage_df, colors, metric, group_name, save_dir)

# Run the plotting functions
dat_dir = input("Enter the path to the data directory: ").strip()
file_path = os.path.join(dat_dir, 'all_metrics_scores_cleaned.tsv')
generate_all_plots(file_path)