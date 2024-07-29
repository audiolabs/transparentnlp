import json
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class MetricsVisualizer:
    data: Dict
    normalized_metrics: List[str] = field(default_factory=lambda: [
        "context_adherence", "completeness", "correctness", "answer_relevancy",
        "readability_LLM_eval_Trott", "Faithfulness", "Answer relevancy",
        "Answer correctness", "Context relevancy", "Answer semantic similarity"
    ])
    to_be_normalized_metrics: List[str] = field(default_factory=lambda: [
        "bleu_score", "rouge_1", "bert_score"
    ])
    ignore_metrics: List[str] = field(default_factory=lambda: [
        "readability_score", "readability_grade", "text_length"
    ])

    def visualize_all_data(self, file_name: str = "all_data_plot.png"):
        available_metrics = [metric for metric in self.normalized_metrics if metric in self.data]
        categories = sorted(set(cat for metric in available_metrics for cat in self.data[metric].keys()))

        traces = self._create_traces(available_metrics, categories)

        fig = go.Figure(data=traces)
        fig.update_layout(
            barmode='group',
            title='Metrics Comparison (All Data)',
            xaxis=dict(title='Metrics'),
            yaxis=dict(title='Scores', range=[0, 100]),
            legend=dict(title='Categories')
        )

        pio.write_image(fig, file_name)
        fig.show()

    def visualize_data_excluding_excerpts(self, file_name_png: str = "data_excluding_excerpts_plot.png", file_name_html: str = "data_excluding_excerpts_plot.html"):
        available_metrics = [metric for metric in self.normalized_metrics if metric in self.data]
        data_excluding_excerpts = self._remove_excerpts(self.data, available_metrics)
        categories = sorted(set(cat for metric in available_metrics for cat in data_excluding_excerpts[metric].keys()))

        traces = self._create_traces(available_metrics, categories, data_excluding_excerpts)

        fig = go.Figure(data=traces)
        fig.update_layout(
            barmode='group',
            title='Metrics Comparison (Excluding Excerpts)',
            xaxis=dict(title='Metrics'),
            yaxis=dict(title='Scores', range=[0, 100]),
            legend=dict(title='Categories'),
            width=1900,
            height=1200
        )

        pio.write_image(fig, file_name_png, width=1900, height=1200)
        pio.write_html(fig, file_name_html)
        fig.show()

    def _remove_excerpts(self, data, available_metrics):
        data_copy = data.copy()
        for metric in available_metrics:
            if 'Excerpts' in data_copy[metric]:
                del data_copy[metric]['Excerpts']
        return data_copy

    def _create_traces(self, metrics, categories, data=None):
        if data is None:
            data = self.data
        traces = []
        for category in categories:
            x, y, text = [], [], []
            for metric in metrics:
                x.append(metric)
                value = data[metric].get(category, 0)
                y.append(value)
                text.append(f'{value:.2f}')
            traces.append(go.Bar(
                x=x,
                y=y,
                name=category,
                text=text,
                textposition='auto'
            ))
        return traces

@dataclass
class MetricsProcessor:
    data: Dict
    normalized_metrics: List[str] = field(default_factory=lambda: [
        "context_adherence", "completeness", "correctness", "answer_relevancy",
        "readability_LLM_eval_Trott", "Faithfulness", "Answer relevancy",
        "Answer correctness", "Context relevancy", "Answer semantic similarity"
    ])
    group_1_keys: List[str] = field(default_factory=lambda: ["Designed_Answer_1", "Answer_Alexa", "RAG", "RAG+RAIN comprehensibility", "RAG+RAIN dishonesty"])
    group_2_keys: List[str] = field(default_factory=lambda: ["Designed_Answer_2", "Answer_Alexa", "RAG", "RAG+RAIN comprehensibility", "RAG+RAIN dishonesty"])

    def divide_and_normalize_metrics(self):
        group_1, group_2 = {}, {}
        for metric in self.normalized_metrics:
            if metric in self.data:
                values = self.data[metric]
                group_1[metric] = {key: value for key, value in values.items() if key in self.group_1_keys}
                if "Designed_Answer_1" in group_1[metric]:
                    diff_1 = 100 - group_1[metric]["Designed_Answer_1"]
                    group_1[metric] = {key: value + diff_1 for key, value in group_1[metric].items()}

                group_2[metric] = {key: value for key, value in values.items() if key in self.group_2_keys}
                if "Designed_Answer_2" in group_2[metric]:
                    diff_2 = 100 - group_2[metric]["Designed_Answer_2"]
                    group_2[metric] = {key: value + diff_2 for key, value in group_2[metric].items()}

        return group_1, group_2

    def create_individual_bar_plots(self, group, group_name: str):
        if not group:
            print(f"No data to plot for {group_name}.")
            return

        num_metrics = len(group)
        num_rows = (num_metrics + 1) // 2

        fig = make_subplots(rows=num_rows, cols=2, subplot_titles=list(group.keys()))
        metrics = list(group.keys())
        keys = ["Answer_Alexa", "RAG", "RAG+RAIN comprehensibility", "RAG+RAIN dishonesty"]
        colors = {
            "Answer_Alexa": "blue",
            "RAG": "orange",
            "RAG+RAIN comprehensibility": "green",
            "RAG+RAIN dishonesty": "red"
        }

        for idx, metric in enumerate(metrics):
            row, col = idx // 2 + 1, idx % 2 + 1
            if metric not in group:
                print(f"Metric {metric} not found in group.")
                continue

            for key in keys:
                if key not in group[metric]:
                    print(f"Key {key} not found in metric {metric}.")
                    continue

                fig.add_trace(go.Bar(
                    x=[key],
                    y=[round(group[metric][key])],
                    name=key,
                    text=[str(round(group[metric][key]))],
                    textposition='inside',
                    textangle=0,
                    insidetextanchor='middle',
                    marker_color=colors.get(key, 'gray')  # Default color if key not found
                ), row=row, col=col)

        fig.update_layout(
            title=f"{group_name} Metrics",
            showlegend=False,
            height=400 * num_rows
        )

        fig.show()
        fig.write_image(f"{group_name}_Metrics.png")
        fig.write_html(f"{group_name}_Metrics.html")

def load_json(filename: str) -> Dict:
    with open(filename, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    data = load_json('metrics.json')

    # Visualize data
    visualizer = MetricsVisualizer(data)
    visualizer.visualize_all_data()
    visualizer.visualize_data_excluding_excerpts()

    # Process and plot groups
    processor = MetricsProcessor(data)
    group_1, group_2 = processor.divide_and_normalize_metrics()

    for metric in group_1:
        if "Designed_Answer_1" in group_1[metric]:
            del group_1[metric]["Designed_Answer_1"]

    for metric in group_2:
        if "Designed_Answer_2" in group_2[metric]:
            del group_2[metric]["Designed_Answer_2"]

    processor.create_individual_bar_plots(group_1, "Responses compared to the normalized Designed Answer 1 across all LLM-based")
    processor.create_individual_bar_plots(group_2, "Responses compared to the normalized Designed Answer 2 across all LLM-based")
