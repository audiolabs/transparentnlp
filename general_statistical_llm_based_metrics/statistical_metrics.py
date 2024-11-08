import pandas as pd
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from textstat import textstat
from lexicalrichness import LexicalRichness
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize, word_tokenize
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os
import sys
sys.path.insert(0, ".")
# Add the parent directory to sys.path to allow relative imports from the project
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from configs import Config
import nltk

# Ensure punkt is downloaded just once
nltk.download('punkt', quiet=True)

# Load the dataset from the specified directory
data_dir = os.path.join(Config.DATA_DIR, 'dataset_for_analysis')
df = pd.read_csv(os.path.join(data_dir, 'dataset_for_evaluation.tsv'), sep='\t')

# Initialize models and tokenizers once
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
scorer = BERTScorer(model_type='bert-base-uncased', lang="en", rescale_with_baseline=True)
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    return sacrebleu.sentence_bleu(candidate, [reference]).score

# Function to calculate ROUGE score
def calculate_rouge(reference, candidate):
    scores = rouge_scorer_instance.score(reference, candidate)
    return scores['rouge1'].fmeasure

# Function to calculate readability using Flesch-Kincaid
def calculate_readability(text):
    return textstat.flesch_reading_ease(text)

# Function to calculate text statistics (using textstat)
def calculate_textstat(text):
    return textstat.text_standard(text, float_output=True)

# Function to calculate lexical diversity
def calculate_lexical_diversity(text, threshold=0.72):
    lex = LexicalRichness(text)
    return lex.mtld(threshold=threshold)

# Function to calculate average sentence length
def calculate_sentence_length(text):
    sentences = sent_tokenize(text)
    word_count = sum(len(word_tokenize(sentence)) for sentence in sentences)
    return word_count / len(sentences) if len(sentences) > 0 else 0

# Function to calculate BERTScore (optimized by loading once)
def calculate_bert_score(reference, candidate):
    P, R, F1 = scorer.score([candidate], [reference], verbose=False)
    return F1[0].item()

# Function to calculate semantic similarity (optimized by loading once)
def calculate_semantic_similarity(reference, candidate):
    embeddings = sentence_model.encode([reference, candidate], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# Define columns for evaluation
columns_to_evaluate = [
    'Answer_Alexa', 'Designed_Answer_1', 'Designed_Answer_2'
] + [col for col in df.columns if col.startswith('RAG+RAIN_') or col.startswith('RAG+MultiRAIN_') or col == 'VanillaRAG']

# Initialize a dictionary to hold the results
result_columns = {}

# Precompute the values for each row and store them in the dictionary
for column in tqdm(columns_to_evaluate, desc="Calculating metrics for each column"):
    result_columns[f'{column}_BLEU'] = df.apply(lambda x: calculate_bleu(x['Excerpts'], x[column]) if pd.notna(x['Excerpts']) and pd.notna(x[column]) else np.nan, axis=1)
    result_columns[f'{column}_ROUGE'] = df.apply(lambda x: calculate_rouge(x['Excerpts'], x[column]) if pd.notna(x['Excerpts']) and pd.notna(x[column]) else np.nan, axis=1)
    result_columns[f'{column}_Readability'] = df[column].apply(lambda x: calculate_readability(x) if pd.notna(x) else np.nan)
    result_columns[f'{column}_TextStat'] = df[column].apply(lambda x: calculate_textstat(x) if pd.notna(x) else np.nan)
    result_columns[f'{column}_LexicalDiversity'] = df[column].apply(lambda x: calculate_lexical_diversity(x) if pd.notna(x) else np.nan)
    result_columns[f'{column}_SentenceLength'] = df[column].apply(lambda x: calculate_sentence_length(x) if pd.notna(x) else np.nan)
    result_columns[f'{column}_BERTScore'] = df.apply(lambda x: calculate_bert_score(x['Excerpts'], x[column]) if pd.notna(x['Excerpts']) and pd.notna(x[column]) else np.nan, axis=1)
    result_columns[f'{column}_STS'] = df.apply(lambda x: calculate_semantic_similarity(x['Excerpts'], x[column]) if pd.notna(x['Excerpts']) and pd.notna(x[column]) else np.nan, axis=1)

# After the loop, combine the result columns into the original dataframe
results = pd.concat([df['Question'], pd.DataFrame(result_columns)], axis=1)

# Save the results to a new TSV file in the current directory

# Create the output directory if it doesn't exist
output_dir = os.path.join(data_dir, 'analysed_data')
os.makedirs(output_dir, exist_ok=True)
results.to_csv(os.path.join(output_dir, 'statistical_metrics_scores.tsv'), sep='\t', index=False)