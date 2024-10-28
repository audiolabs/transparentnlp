import pandas as pd
import openai
import sacrebleu
from textstat import textstat
from rouge_score import rouge_scorer
import bert_score
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


os.environ["OPENAI_API_KEY"] = "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"


def call_gpt4(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",  # or the appropriate engine like "text-davinci-002"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    #return response.choices[0].message['content'].strip()
    return response.choices[0].message.content.strip()

# Calculate lexical diversity as the ratio of unique words to the total number of words in the text.
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

# Define the evaluation function to be parallelized
def evaluate_single_text(row, question, context, answer_type, scorer):
    text = row[answer_type]
    text_evaluation = {
        'Question': row['Question'],
        answer_type: text,
        'context_adherence': call_gpt4(
            f"Context adherence measures whether your model's response was purely based on the context provided. A high Context Adherence score means your response is supported by the context provided. Evaluate the context adherence of the following text: {text} given this context: {context}. The context adherence should be given as a score from 0 to 100, where 100 is perfect context adherence and 0 is lack of any context adherence. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'."
        ),
        'completeness': call_gpt4(
            f"Completeness measures how thoroughly your model's response covered the relevant information available in the context provided. Evaluate the completeness of the following text: {text} given this context: {context} and question: {question}. The completeness should be given as a score from 0 to 100, where 100 is perfect completeness and 0 is no completeness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'."
        ),
        'correctness': call_gpt4(
            f"Correctness measures whether a given model response is factual or not. Correctness (f.k.a. Factuality) is a good way of uncovering open-domain hallucinations: factual errors that don't relate to any specific documents or context. A high Correctness score means the response is more likely to be accurate vs a low response indicates a high probability for hallucination. Evaluate the correctness of this text: {text}. The Correctness should be given as a score from 0 to 100, where 100 is perfect correctness and 0 is no correctness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'."
        ),
        'answer_relevancy': call_gpt4(
            f"Measures how relevant the answer is to the user question. Higher answer relevance means that the answer is more relevant to the question. Evaluate the relevancy of this answer: {text} given this question: {question}. The answer relevancy should be given as a score from 0 to 100, where 100 is perfect answer relevancy and 0 is no answer relevancy. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'."
        ),
        'readability_LLM_eval_Trott': call_gpt4(
            f"Read the text below. Then, indicate the readability of the text, on a scale from 1 (extremely challenging to understand) to 100 (very easy to read and understand). In your assessment, consider factors such as sentence structure, vocabulary complexity, and overall clarity. Text: {text}"
        ),
        'bleu_score': sacrebleu.sentence_bleu(text, [context]).score,
        'rouge_1': scorer.score(context, text)['rouge1'].fmeasure,
        'bert_score': bert_score.score([text], [context], lang='en')[2].mean().item(),
        'readability_score': textstat.flesch_reading_ease(text),
        'readability_grade': textstat.text_standard(text, float_output=False),
        'lexical_diversity': lexical_diversity(text),
        'text_length': len(text.split())
    }
    return text_evaluation


def evaluate_text(df):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    answer_types = ['Designed_Answer_1', 'Designed_Answer_2', 'VanillaRAG', 'RAG+RAIN_Readability',
                    'RAG+RAIN_Correctness', 'RAG+MultiRAIN_Readability+Correctness']
    results = {answer_type: [] for answer_type in answer_types}

    for answer_type in answer_types:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_single_text, row, row['Question'], row['Excerpts'], answer_type, scorer)
                       for index, row in df.iterrows()]
            for future in futures:
                results[answer_type].append(future.result())

        # Convert results to DataFrame
        results_df = pd.DataFrame(results[answer_type])
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'PrivacyQA_Evaluation_{answer_type}_{timestamp}.csv'
        results_df.to_csv(filename, index=False)

    return results

df = pd.read_csv('readability_correctness_multirain_all.tsv',sep="\t")
results= evaluate_text(df)