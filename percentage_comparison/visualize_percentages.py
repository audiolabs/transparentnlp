import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import colors as mcolors
plt.rcParams.update({
    'font.size': 24,             # General font size
    'axes.titlesize': 28,        # Title font size
    'axes.labelsize': 24,        # Axis label font size
    'xtick.labelsize': 20,       # X-tick label font size
    'ytick.labelsize': 20,       # Y-tick label font size
    'legend.fontsize': 20,       # Legend font size
    'figure.titlesize': 28       # Figure title font size
})
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
    percentage_df = percentage_df[~percentage_df.index.str.contains('Alexa', case=False)]

    colors = [COLOR_MAPPING.get(answer_type, 'gray') for answer_type in percentage_df.index]
    return percentage_df, colors

# Plotting functions
def plot_individual_metric(data, colors, metric, group_name, save_dir):
    plt.figure(figsize=(8, 6))  # Increased plot size
    plt.bar(data.index, data[metric], color=colors)
    plt.ylabel('Percentage (%)', fontsize=16)
    plt.title(metric, fontsize=18)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotated labels
    plt.xlabel('Answer Type', fontsize=14)
    plt.tight_layout()  # Avoid overlap
    plt.savefig(os.path.join(save_dir, f"{group_name}_{metric}.pdf"), bbox_inches='tight')  # Save as PDF
    plt.close()


def plot_group_metrics(data, colors, group_name, metrics, save_dir):
    # Number of subplots (all in one row)
    cols = len(metrics)
     # Adjust the figure size: Increase the width per subplot
    width_per_subplot = 12  # Adjust this value to make subplots wider
    fig, axes = plt.subplots(1, cols, figsize=(width_per_subplot * cols, 6))  # Increase width for all subplots in one row

    if cols == 1:
        axes = [axes]  # Ensure axes is iterable even for a single plot

    for ax, metric in zip(axes, metrics):
        ax.bar(data.index, data[metric], color=colors)
        ax.set_ylabel('Percentage (%)', fontsize=30)
        ax.set_title(metric, fontsize=30)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=30)  # Larger font size for x-axis labels
        ax.set_xlabel('Answer Type', fontsize=30)
        # Increase the font size of y-tick labels
        ax.tick_params(axis='y', labelsize=30)  # Adjust this value for larger y-tick labels


    # Add legend and adjust layout
    fig.suptitle(group_name.replace("_", " ").title(), fontsize=30, y=1.05)
    fig.legend(handles=get_custom_legend(), bbox_to_anchor=(0.9, 0.5), loc='center left', fontsize=26, frameon=True)
    plt.subplots_adjust(wspace=0.3, left=0.05, right=0.85, bottom=0.2, top=0.9)  # Adjust spacing and layout
    plt.savefig(os.path.join(save_dir, f"{group_name}.pdf"), bbox_inches='tight')  # Save as PDF
    plt.close()


def plot_combined_metrics(data, colors, save_dir):
    num_metrics = sum(len(group) for group in GROUPS.values())
    rows = (num_metrics // 4) + (1 if num_metrics % 4 != 0 else 0)  # 4 metrics per row
    fig, axes = plt.subplots(rows, 4, figsize=(24, 6 * rows))  # Increased figure size
    axes = axes.flatten()  # Flatten axes for simple indexing
    plot_index = 0

    for group_name, metrics in GROUPS.items():
        for metric in metrics:
            if plot_index >= len(axes):  # Avoid out-of-bounds errors
                break
            ax = axes[plot_index]
            ax.bar(data.index, data[metric], color=colors)
            ax.set_ylabel('Percentage (%)', fontsize=16)
            ax.set_title(metric, fontsize=18)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(range(len(data.index)))
            ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=12)  # Rotated labels
            ax.set_xlabel('Answer Type', fontsize=14)
            plot_index += 1

    # Hide unused axes
    for j in range(plot_index, len(axes)):
        axes[j].set_visible(False)

    # Adjust spacing and legend
    fig.suptitle('Percentage Metrics Visualization (Combined)', fontsize=24)
    fig.legend(handles=get_custom_legend(), bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=16, frameon=True)
    plt.subplots_adjust(hspace=0.6, wspace=0.4, bottom=0.2)  # Increased vertical spacing and bottom space
    plt.savefig(os.path.join(save_dir, 'percentage_metrics_combined.pdf'), bbox_inches='tight')  # Save as PDF
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