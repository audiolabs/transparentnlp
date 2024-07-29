from dataclasses import dataclass

import pandas as pd
import os
import nltk
from configs import Config

nltk.download('punkt')
from textstat import textstat
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize, word_tokenize
from lexicalrichness import LexicalRichness
from collections import Counter


# Define data class for the metrics
@dataclass
class Metrics:
    bleu_sentence: float
    bleu_corpus: float
    rouge: float
    one_gram_overlap: float
    readability: float
    readability_standard: float
    lexical_diversity: float
    avg_sentence_length: float
    bert_score: float
    semantic_similarity: float

# Define data class for a single row of data
@dataclass
class RowData:
    Excerpts: str
    Rag_Rain_dishon: str
    Rag_Rain_comp: str

def calculate_bleu_sentence(reference: str, hypothesis: str):
    return sentence_bleu([reference.split()], hypothesis.split(), weights=(0.5, 0.5))

def calculate_one_gram_overlap(reference: str, hypothesis: str):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    ref_counter = Counter(ref_words)
    hyp_counter = Counter(hyp_words)
    overlap = sum((ref_counter & hyp_counter).values())
    overlap_ratio = overlap / (len(ref_words) + len(hyp_words))
    return overlap_ratio

def calculate_bleu_corpus(reference: str, hypothesis: str):
    reference_sentences = sent_tokenize(reference)
    hypothesis_sentences = sent_tokenize(hypothesis)
    reference_corpus = [reference_sentences]
    hypothesis_corpus = [hypothesis_sentences]
    return corpus_bleu(reference_corpus, hypothesis_corpus, weights=(0.5, 0.5))

def calculate_rouge(reference: str, hypothesis: str):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure

def calculate_readability(hypothesis: str):
    return textstat.flesch_reading_ease(hypothesis)

def calculate_textstandard(hypothesis: str):
    return textstat.text_standard(hypothesis)

def calculate_lexical_diversity(hypothesis: str, threshold: float = 0.72):
    lex = LexicalRichness(hypothesis)
    return lex.mtld(threshold=threshold)

def calculate_avg_sentence_length(hypothesis: str):
    sentences = sent_tokenize(hypothesis)
    total_sentences = len(sentences)
    if total_sentences > 0:
        total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
        avg_sentence_length = total_words / total_sentences
    else:
        avg_sentence_length = 0.0
    return avg_sentence_length

def calculate_bert_score(reference: str, hypothesis: str):
    scorer = BERTScorer(model_type='bert-base-uncased', lang="en", rescale_with_baseline=True)
    P, R, F1 = scorer.score([hypothesis], [reference], verbose=False)
    return F1[0].item()

def calculate_semantic_similarity(reference: str, hypothesis: str, model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode([reference, hypothesis], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

def calculate_average_scores(df: pd.DataFrame) -> dict:
    avg_scores = {
        'Sentence BLEU average': df['bleu_sentence'].mean(),
        'One gram overlap average': df['one_gram_overlap'].mean(),
        'BLEU corpus average': df['bleu_corpus'].mean(),
        'ROUGE-1 average': df['rouge'].mean(),
        'Readability flesch': df['readability'].mean(),
        'Readability textstandard': df['readability_standard'].mean(),
        'Lexical diversity average': df['lexical_diversity'].mean(),
        'Average sentence length': df['avg_sentence_length'].mean(),
        'BERTScore average': df['bert_score'].mean(),
        'Semantic similarity average': df['semantic_similarity'].mean()
    }
    return avg_scores

def process_column(df: pd.DataFrame, column_name: str, output_file: str):
    df['bleu_sentence'] = df.apply(lambda row: calculate_bleu_sentence(row['Excerpts'], row[column_name]), axis=1)
    df['bleu_corpus'] = df.apply(lambda row: calculate_bleu_corpus(row['Excerpts'], row[column_name]), axis=1)
    df['rouge'] = df.apply(lambda row: calculate_rouge(row['Excerpts'], row[column_name]), axis=1)
    df['one_gram_overlap'] = df.apply(lambda row: calculate_one_gram_overlap(row['Excerpts'], row[column_name]), axis=1)
    df['readability'] = df.apply(lambda row: calculate_readability(row[column_name]), axis=1)
    df['readability_standard'] = df.apply(lambda row: calculate_textstandard(row[column_name]), axis=1)
    df['lexical_diversity'] = df.apply(lambda row: calculate_lexical_diversity(row[column_name]), axis=1)
    df['avg_sentence_length'] = df.apply(lambda row: calculate_avg_sentence_length(row[column_name]), axis=1)
    df['bert_score'] = df.apply(lambda row: calculate_bert_score(row['Excerpts'], row[column_name]), axis=1)
    df['semantic_similarity'] = df.apply(lambda row: calculate_semantic_similarity(row['Excerpts'], row[column_name]), axis=1)

    df.to_csv(output_file, sep='\t', index=False)

    avg_scores = calculate_average_scores(df)
    avg_scores_file = os.path.join(Config.DATA_DIR, f'average_scores_{column_name}.txt')
    with open(avg_scores_file, 'w') as f:
        f.write("Average scores of each metric:\n")
        for metric, score in avg_scores.items():
            f.write(f"{metric}: {score}\n")

def main():
    input_file = os.path.join(Config.DATA_DIR, 'processed_data.tsv')

    df = pd.read_csv(input_file, sep="\t")

    process_column(df, 'Rag_Rain_dishon', os.path.join(Config.VAL_DIR, 'processed_data_rag_rainrag_dishonesty_statistical.tsv'))
    process_column(df, 'Rag_Rain_comp', os.path.join(Config.VAL_DIR, 'processed_data_rag_rainrag_comprehensibility_statistical.tsv'))

if __name__ == "__main__":
    main()
