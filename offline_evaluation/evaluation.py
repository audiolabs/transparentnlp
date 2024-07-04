import argparse
from utils import read_data, apply_metric_functions, save_results
from metrics_dataclasses import (
    BleuMetric, RougeMetric, ReadabilityMetric, LexicalDiversityMetric,
    AvgSentenceLengthMetric, BertScoreMetric, SemanticSimilarityMetric,
    FaithfulnessMetric, AnswerRelevancyMetric, AnswerSimilarityMetric,
    AnswerCorrectnessMetric, RqugeScoreMetric, GevalMetric,
    SelfCheckCorrectnessMetric, ZeroshotPromptsMetric
)
from metrics import (
    calculate_bleu_sentence, calculate_rouge, calculate_readability,
    calculate_lexical_diversity, calculate_avg_sentence_length,
    calculate_bert_score, calculate_semantic_similarity, calculate_ragas_faithfulness,
    calculate_ragas_answer_relevancy, calculate_ragas_answer_similarity,
    calculate_ragas_answer_correctness, calculate_rquge_score,
    calculate_geval_metric, calculate_selfcheck_correctness,
    calculate_zeroshot_prompts
)
from config import Config

# Define the list of metric functions and their corresponding data classes
metric_functions = [
    (calculate_bleu_sentence, BleuMetric),
    (calculate_rouge, RougeMetric),
    (calculate_readability, ReadabilityMetric),
    (calculate_lexical_diversity, LexicalDiversityMetric),
    (calculate_avg_sentence_length, AvgSentenceLengthMetric),
    (calculate_bert_score, BertScoreMetric),
    (calculate_semantic_similarity, SemanticSimilarityMetric),
    (calculate_ragas_faithfulness, FaithfulnessMetric),
    (calculate_ragas_answer_relevancy, AnswerRelevancyMetric),
    (calculate_ragas_answer_similarity, AnswerSimilarityMetric),
    (calculate_ragas_answer_correctness, AnswerCorrectnessMetric),
    (calculate_rquge_score, RqugeScoreMetric),
    (calculate_geval_metric, GevalMetric),
    (calculate_selfcheck_correctness, SelfCheckCorrectnessMetric),
    (calculate_zeroshot_prompts, ZeroshotPromptsMetric)
]

def main(input_file_name, output_file_name):
    # Step 1: Read the data
    df = read_data(input_file_name)

    # Step 2: Apply metric functions
    df_with_scores = apply_metric_functions(df, metric_functions)

    # Step 3: Save the results
    save_results(df_with_scores, output_file_name)

    print(f"Metrics calculated and saved to {os.path.join(Config.OUTPUT_DIR, output_file_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Offline Evaluation Script')
    parser.add_argument('input_file', type=str, help='Path to the input file (CSV or JSON)')
    parser.add_argument('output_file', type=str, help='Name of the output file (CSV or JSON)')

    args = parser.parse_args()

    main(args.input_file, args.output_file)