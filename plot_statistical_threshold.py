import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import List
from dataclasses import dataclass
from configs import Config


@dataclass
class MetricAnalysis:
    metrics: List[str]
    results: pd.DataFrame
    df: pd.DataFrame

    def __post_init__(self):
        # Verify that the 'Question' column exists in df and results
        if 'Question' not in self.df.columns:
            raise ValueError("The 'Question' column is missing from the DataFrame.")
        if 'Question' not in self.results.columns:
            raise ValueError("The 'Question' column is missing from the results DataFrame.")
        # Ensure the 'Question' column is in results
        self.results['Question'] = self.df['Question']
        self.results = self.compute_metric_min_max_avg(self.results, self.metrics)
        self.standardized_df = self.standardize_metrics(self.results.copy(), self.metrics)

    def compute_metric_min_max_avg(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Compute min, max, and average for each metric."""
        for metric in metrics:
            min_values = []
            max_values = []
            avg_values = []

            for _, row in df.iterrows():
                designed_columns = [col for col in df.columns if col.startswith('Designed_') and col.endswith(metric)]
                designed_values = [row[col] for col in designed_columns if pd.notna(row[col])]

                if designed_values:
                    min_value = min(designed_values)
                    max_value = max(designed_values)
                    avg_value = (min_value + max_value) / 2
                else:
                    min_value = max_value = avg_value = float('nan')

                min_values.append(min_value)
                max_values.append(max_value)
                avg_values.append(avg_value)

            df[f'min_{metric}'] = min_values
            df[f'max_{metric}'] = max_values
            df[f'average_{metric}'] = avg_values

        return df

    def standardize_metrics(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Apply Z-score normalization to the metrics."""
        for metric in metrics:
            metric_columns = [col for col in df.columns if col.endswith(metric)]
            for col in metric_columns:
                mean_value = df[col].mean()
                std_dev = df[col].std()
                if std_dev != 0:
                    df[col] = (df[col] - mean_value) / std_dev
                else:
                    df[col] = df[col] - mean_value  # No scaling, just centering if std_dev is 0
        return df

    @staticmethod
    def adjust_text_positions(y_values: List[float], labels: List[str]) -> List[tuple]:
        """Adjust text positions to avoid overlapping."""
        adjusted_labels = []
        prev_y = None

        for y, label in zip(y_values, labels):
            if prev_y is not None and abs(y - prev_y) < 0.05:
                y += 0.05
            adjusted_labels.append((y, label))
            prev_y = y

        return adjusted_labels

    def plot_metric_groups(self, df: pd.DataFrame, save_dir: str) -> None:
        """Create and save subplots for each row in the DataFrame."""
        columns_to_plot = [col for col in df.columns if
                           not col.startswith('Designed_') and not any(x in col for x in ['min', 'max', 'average'])]

        os.makedirs(save_dir, exist_ok=True)

        for index, row in df.iterrows():
            # Verify that the 'Question' column is available
            if 'Question' not in row.index:
                raise KeyError("The 'Question' column is missing from the row data.")

            fig = make_subplots(rows=4, cols=2, subplot_titles=self.metrics)

            for i, metric in enumerate(self.metrics):
                metric_columns = [col for col in columns_to_plot if col.endswith(metric)]
                metric_values = [row[col] for col in metric_columns]
                column_names = [col.replace(f'_{metric}', '') for col in metric_columns]
                boxplot_data = [row[f'min_{metric}'], row[f'max_{metric}'], row[f'average_{metric}']]

                adjusted_labels = self.adjust_text_positions(metric_values, column_names)
                adjusted_y_values, adjusted_column_names = zip(*adjusted_labels)

                fig.add_trace(go.Box(
                    y=boxplot_data,
                    name='Answers',
                    marker_color='lightgrey',
                    line_color='black'
                ), row=i // 2 + 1, col=i % 2 + 1)

                fig.add_trace(go.Scatter(
                    x=['Answers'] * len(metric_values),
                    y=adjusted_y_values,
                    mode='markers+text',
                    text=adjusted_column_names,
                    textposition="top right",
                    marker=dict(color='blue', size=10),
                    name=f'{metric} Values'
                ), row=i // 2 + 1, col=i % 2 + 1)

            question_text = row['Question']

            fig.update_layout(
                title=f'Question: {question_text}',
                height=3000,  # Increase height
                width=1800,  # Increase width
                showlegend=False,
                margin=dict(l=25, r=25, t=25, b=25),  # Increase margins
            )

            fig.write_image(os.path.join(save_dir, f'question_{index + 1}.png'))


def main():
    # Define the metrics list
    metrics = ['BLEU', 'ROUGE', 'Readability', 'TextStat', 'LexicalDiversity', 'SentenceLength', 'bert', 'STS']

    # Load the dataset
    results = pd.read_csv(os.path.join(Config.DATA_DIR, 'thresholds_statistical.tsv'), sep='\t')
    df = pd.read_csv(os.path.join(Config.DATA_DIR, 'final_dataset_with_non_proactive_answers.tsv'), sep='\t')

    # Ensure 'Question' column is merged into results DataFrame
    if 'Question' not in df.columns:
        raise ValueError("The 'Question' column is missing from the DataFrame.")

    results = results.assign(Question=df['Question'])

    # Initialize MetricAnalysis
    analysis = MetricAnalysis(metrics=metrics, results=results, df=df)

    # Create directories for saving plots
    non_standardized_dir = os.path.join('statistical_plot_dir', 'non_standardized')
    standardized_dir = os.path.join('statistical_plot_dir', 'Z_score_standardized')

    os.makedirs(non_standardized_dir, exist_ok=True)
    os.makedirs(standardized_dir, exist_ok=True)

    # Plot metric groups and save as PNG files
    analysis.plot_metric_groups(df=analysis.results, save_dir=non_standardized_dir)
    analysis.plot_metric_groups(df=analysis.standardized_df, save_dir=standardized_dir)


if __name__ == '__main__':
    main()
