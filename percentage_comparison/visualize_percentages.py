import matplotlib.pyplot as plt
from matplotlib import colors as mcolors  # Import colors from matplotlib
from percentage_as_good_as_human import calculate_percentage_comparisons
import numpy as np
import os
# Ignore FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Calculate the percentage DataFrame
percentage_df = calculate_percentage_comparisons(dataset='all_metrics_scores.tsv')

# Drop the specified rows
percentage_df = percentage_df.drop([
    'Designed_Answer_Non_Proactive_1',
    'Designed_Answer_Non_Proactive_2',
    'Designed_Answer_Non_Proactive_3'
])

# Create a color mapping for specific answer types
color_mapping = {
    'Designed_Answer_1': 'gold',
    'Designed_Answer_2': 'gold',
    'VanillaRAG': 'red'
}

# Generate a viridis colormap
viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))


# Function to convert named color to RGB
def get_rgb(color):
    if color in color_mapping.values():
        return np.array(mcolors.to_rgba(color)[:3])  # Return RGB without alpha
    else:
        return np.array(mcolors.to_rgba(color)[:3])  # Ensure all colors are in RGB format


# Function to filter out colors that are close to gold or red
def is_color_similar_to(color, target, threshold=0.1):
    return np.linalg.norm(color - target) < threshold


# Convert gold and red to RGB
gold_rgb = get_rgb('gold')  # RGB for gold
red_rgb = get_rgb('red')  # RGB for red

# Filter out similar colors
remaining_colors = []
for color in viridis_colors:
    if not is_color_similar_to(color[:3], gold_rgb) and not is_color_similar_to(color[:3], red_rgb):
        remaining_colors.append(color[:3])  # Ensure we only take RGB components

# Prepare the full color list
colors = []
for index in percentage_df.index:
    if index in color_mapping:
        colors.append(color_mapping[index])  # Append the mapped color
    else:
        # Use a unique remaining color for non-mapped answer types
        if remaining_colors:
            # Check if the next color is too close to any existing colors in 'colors'
            next_color = remaining_colors.pop(0)
            # Convert existing colors to RGB for comparison
            existing_colors_rgb = [get_rgb(existing_color) for existing_color in colors]
            if all(not is_color_similar_to(next_color, existing_color) for existing_color in existing_colors_rgb):
                colors.append(next_color)  # Append only if it's distinct enough
            else:
                colors.append('#000000')  # Default to black if all remaining colors are too close
        else:
            colors.append('#000000')  # Default to black if we run out of colors

# Number of metrics
num_metrics = percentage_df.shape[1]
num_rows = (num_metrics + 1) // 2  # Calculate rows needed

# Set up the subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 4))
axes = axes.flatten()  # Flatten to easily access axes

# Plot each metric
for i, metric in enumerate(percentage_df.columns):
    ax = axes[i]

    # Plot bars with different colors for each answer type
    ax.bar(percentage_df.index, percentage_df[metric], color=colors)

    # Set the y-axis label
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'{metric}')
    ax.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines

    # Set the x-tick labels directly with rotation
    ax.set_xticks(range(len(percentage_df.index)))
    ax.set_xticklabels(percentage_df.index, rotation=90, ha='center')
    ax.set_xlabel('Answer Type')  # Label for the x-axis

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Layout adjustments
plt.tight_layout()
plt.suptitle('Percentage Metrics Visualization', fontsize=16, y=1.02)

# Specify the directory for saving the image
save_dir = 'images'  # Change to your desired directory name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create the directory if it doesn't exist

# Save the figure in the specified directory
plt.savefig(os.path.join(save_dir, 'percentage_comparison.png'))

plt.show()
