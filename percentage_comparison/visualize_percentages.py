import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib import colors as mcolors
from percentage_as_good_as_human_bp import calculate_percentage_comparisons

# Get the directory path from the user
dat_dir = input("Enter the path to the data directory: ").strip()
# Concatenate directory path with the file name
file_path = os.path.join(dat_dir, 'all_metrics_scores_cleaned.tsv')

# Calculate the percentage DataFrame
percentage_df = calculate_percentage_comparisons(dataset=file_path)

# Define the number of columns and calculate rows dynamically based on number of metrics
num_metrics = percentage_df.shape[1]
num_cols = 5  # Adjusted to spread subplots horizontally
num_rows = (num_metrics + num_cols - 1) // num_cols  # Calculate rows needed

# Color mapping for specific answer types and unique colors for remaining types
color_mapping = {
    'Designed_Answer_1': 'gold',
    'Designed_Answer_2': 'gold',
    'VanillaRAG': 'red'
}

# Generate distinct colors for answer types that don't have predefined colors
viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(percentage_df.index) - len(color_mapping)))
remaining_colors = [mcolors.to_hex(c[:3]) for c in viridis_colors]

# Apply color to each answer type and ensure consistency across plots
colors = []
for answer_type in percentage_df.index:
    if answer_type in color_mapping:
        colors.append(color_mapping[answer_type])
    else:
        colors.append(remaining_colors.pop(0) if remaining_colors else '#000000')

# Directory to save the plots
save_dir = 'images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set up the subplots with adjusted figure size to accommodate more columns
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 4))
axes = axes.flatten()  # Flatten for easy indexing

# Plot each metric in the overall plot and save individual plots
for i, metric in enumerate(percentage_df.columns):
    ax = axes[i]

    # Plot bars with the assigned colors for each answer type
    ax.bar(percentage_df.index, percentage_df[metric], color=colors)

    # Set y-axis label, title, and limits
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'{metric}')
    ax.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Rotate x-tick labels with smaller font size and further rotation for readability
    ax.set_xticks(range(len(percentage_df.index)))
    ax.set_xticklabels(percentage_df.index, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Answer Type')

    # Create and save each individual plot with additional bottom space
    plt.figure(figsize=(5, 4))
    plt.bar(percentage_df.index, percentage_df[metric], color=colors)
    plt.ylabel('Percentage (%)')
    plt.title(f'{metric}')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(len(percentage_df.index)), percentage_df.index, rotation=45, ha='right', fontsize=8)
    plt.xlabel('Answer Type')
    
    # Save individual plot with padding for x-axis labels
    individual_plot_path = os.path.join(save_dir, f'percentage_{metric}.png')
    plt.savefig(individual_plot_path, bbox_inches='tight', pad_inches=0.4)  # Add padding
    plt.close()  # Close the figure after saving

# Hide any unused subplots in the overall plot
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Adjust spacing between subplots to prevent label overlap
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increased horizontal and vertical spacing

# Tight layout with increased spacing for title and labels
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to give space for the title

# Set the main title for the entire figure
plt.suptitle('Percentage Metrics Visualization', fontsize=16, y=1.02)

# Save the overall figure
overall_plot_path = os.path.join(save_dir, 'percentage_comparison.png')
plt.savefig(overall_plot_path)
plt.show()