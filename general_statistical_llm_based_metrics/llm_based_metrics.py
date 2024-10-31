# This code is now only based on Birgit's designed prompts
from tqdm import tqdm
import openai
import os
import pandas as pd
import numpy as np
from configs import Config

# Ensure you have your OpenAI API key set up in the environment
os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ").strip()

# Define data directory and read dataset
data_dir = os.path.join(Config.DATA_DIR, 'dataset_for_analysis')
df = pd.read_csv(os.path.join(data_dir, 'dataset_for_evaluation.tsv'), sep='\t')

# Define columns for evaluation
columns_to_evaluate = [
    'Answer_Alexa', 'Designed_Answer_1', 'Designed_Answer_2',
    'Designed_Answer_Non_Proactive_1', 'Designed_Answer_Non_Proactive_2',
    'Designed_Answer_Non_Proactive_3'
] + [col for col in df.columns if col.startswith('RAG+RAIN_') or col.startswith('RAG+MultiRAIN_') or col == 'VanillaRAG']



# Function to calculate zero-shot prompts
def calculate_zeroshot_prompts(question: str, excerpt: str, answer: str):
    if not isinstance(question, str):
        question = str(question)
    if not isinstance(answer, str):
        answer = str(answer)
    if not isinstance(excerpt, str):
        excerpt = str(excerpt)
    prompts = {
        'excerpt_adherence': f"Context adherence measures whether your model's response was purely based on the excerpt provided. A high Context Adherence score means your response is supported by the excerpt provided. Evaluate the excerpt adherence of the following text: {answer} excerpt: {excerpt}.The excerpt adherence should be given as a score from 0 to 100, where 100 is perfect excerpt adherence and 0 is lack of any excerpt adherence. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'completeness': f"Completeness measures how thoroughly your model's response covered the relevant information available in the excerpt provided. Evaluate the completeness of the following text: {answer} given the excerpt: {excerpt} and question: {question}. The completeness should be given as a score from 0 to 100, where 100 is perfect completeness and 0 is no completeness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'correctness': f"Correctness measures whether a given model response is factual or not. Correctness (f.k.a. Factuality) is a good way of uncovering open-domain hallucinations: factual errors that don't relate to any specific documents or excerpt. A high Correctness score means the response is more likely to be accurate vs a low response indicates a high probability for hallucination. Evaluate the correctness of this text: {answer} . The Correctness should be given as a score from 0 to 100, where 100 is perfect correctness and 0 is no correctness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'answer_relevancy': f"Measures how relevant the answer is to the user question. Higher answer relevance means that the answer is more relevant to the question. Evaluate the relevancy of this answer: {answer} given this question: {question}. The answer relevancy should be given as a score from 0 to 100, where 100 is perfect answer relevancy and 0 is no answer relevancy. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'readability_LLM_eval_Trott': f"Read the text below. Then, indicate the readability of the text, on a scale from 1 (extremely challenging to understand) to 100 (very easy to read and understand). In your assessment, consider factors such as sentence structure, vocabulary complexity, and overall clarity. Text: {answer}"
    }
    scores = {}
    for prompt_name, prompt_text in prompts.items():
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            stop=None,
            temperature=0,
            top_p=0
        )

        score_str = response.choices[0].message.content.strip().split('Overall score: ')[-1]
        scores[prompt_name] = score_str
    return scores


# Iterate over the DataFrame and apply the prompts
results = df[columns_to_evaluate].copy()

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating rows"):
    question = row['Question']
    excerpt = row['Excerpts']
    row_results = {'Question': question, 'Excerpt': excerpt}

    # Evaluate each column in columns_to_evaluate
    for column in columns_to_evaluate:
        answer = row[column] if pd.notna(row[column]) else ""
        prompt_scores = calculate_zeroshot_prompts(question, excerpt, answer)

        # Append each prompt's score with the column name
        for prompt_name, score in prompt_scores.items():
            row_results[f"{column}_{prompt_name}"] = score

    # Append row results to results DataFrame
    results = results.append(row_results, ignore_index=True)

results['Question'] = df['Question']

# Save the results to a new TSV file in the current directory
results.to_csv(os.path.join('.', 'llm_metrics_scores.tsv'), sep='\t', index=False)

# Filter out columns with non-numeric data types in the results DataFrame
numeric_results = results.select_dtypes(include=[np.number])
# Drop columns with all NaN values in numeric_results
numeric_results = numeric_results.dropna(axis=1, how='all')

statistics = []
for col in numeric_results.columns:
    statistics.append({
        'answer_columns': col,
        'mean': numeric_results[col].mean(),
        'median': numeric_results[col].median(),
        'max': numeric_results[col].max(),
        'min': numeric_results[col].min()
    })

# Create and print the statistics DataFrame
statistics_df = pd.DataFrame(statistics)
print(statistics_df[['answer_columns', 'mean', 'median', 'max', 'min']])
