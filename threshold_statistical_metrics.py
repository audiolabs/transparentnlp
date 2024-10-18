import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from textstat import textstat
from lexicalrichness import LexicalRichness
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize, word_tokenize
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
import os
from configs import Config

# Load the dataset
df = pd.read_csv(os.path.join(Config.DATA_DIR, 'updated_final_dataset_with_non_proactive_answers.tsv'), sep='\t')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split(), weights=(0.5, 0.5))

# Function to calculate ROUGE score
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
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

def calculate_bert_score(reference, candidate):
    scorer = BERTScorer(model_type='bert-base-uncased', lang="en", rescale_with_baseline=True)
    P, R, F1 = scorer.score([candidate], [reference], verbose=False)
    return F1[0].item()

def calculate_semantic_similarity(reference, candidate, model_name = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode([reference, candidate], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# Calculate metrics for specified columns
columns_to_evaluate = [
    'Designed_Answer_1', 'Designed_Answer_2', 'Designed_Answer_Non_Proactive_1',
    'Designed_Answer_Non_Proactive_2', 'Designed_Answer_Non_Proactive_3'
]

# Create a new DataFrame to store the results
results = pd.DataFrame(index=df.index)

for column in columns_to_evaluate:
    results[f'{column}_BLEU'] = df.apply(lambda x: calculate_bleu(x['Excerpts'], x[column])
                                         if pd.notna(x['Excerpts']) and pd.notna(x[column]) else np.nan, axis=1)
    results[f'{column}_ROUGE'] = df.apply(lambda x: calculate_rouge(x['Excerpts'], x[column])
                                           if pd.notna(x['Excerpts']) and pd.notna(x[column]) else np.nan, axis=1)
    results[f'{column}_Readability'] = df[column].apply(lambda x: calculate_readability(x)
                                                         if pd.notna(x) else np.nan)
    results[f'{column}_TextStat'] = df[column].apply(lambda x: calculate_textstat(x)
                                                     if pd.notna(x) else np.nan)
    results[f'{column}_LexicalDiversity'] = df[column].apply(lambda x: calculate_lexical_diversity(x)
                                                             if pd.notna(x) else np.nan)
    results[f'{column}_SentenceLength'] = df[column].apply(lambda x: calculate_sentence_length(x)
                                                           if pd.notna(x) else np.nan)
    results[f'{column}_bert'] = df.apply(lambda x: calculate_bert_score(x['Excerpts'], x[column])
                                        if pd.notna(x['Excerpts']) and pd.notna(x[column]) else np.nan, axis=1)
    results[f'{column}_STS'] = df.apply(lambda x: calculate_semantic_similarity(x['Excerpts'], x[column])
                                     if pd.notna(x['Excerpts']) and pd.notna(x[column]) else np.nan, axis=1)

# Save the results to a new CSV file
#results.to_csv(os.path.join(Config.DATA_DIR, 'add_new_name.tsv'),sep='\t', index=False)
