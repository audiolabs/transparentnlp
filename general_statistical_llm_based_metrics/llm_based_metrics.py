import asyncio
import logging
from tqdm import tqdm
from openai import AsyncOpenAI
import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")
# Add the parent directory to sys.path to allow relative imports from the project
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from configs import Config

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

# Prompt the user for the OpenAI API key
api_key = input("Enter your OpenAI API key: ").strip()
if not api_key:
    logging.error("OpenAI API key was not provided.")
    exit(1)

# Initialize AsyncOpenAI client with the provided API key
client = AsyncOpenAI(api_key=api_key)

# Define data directory and read dataset
data_dir = os.path.join(Config.DATA_DIR, 'dataset_for_analysis')

# Try to load the dataset and handle potential errors
try:
    df = pd.read_csv(os.path.join(data_dir, 'dataset_for_evaluation.tsv'), sep='\t')
    logging.info("Successfully loaded dataset")
except Exception as e:
    logging.error("Failed to load dataset: %s", e)
    exit(1)

# Define columns for evaluation
columns_to_evaluate = [
    'Answer_Alexa', 'Designed_Answer_1', 'Designed_Answer_2'
] + [col for col in df.columns if col.startswith('RAG+RAIN_') or col.startswith('RAG+MultiRAIN_') or col == 'VanillaRAG']

# Function to calculate zero-shot prompts asynchronously
async def calculate_zeroshot_prompts(question: str, excerpt: str, answer: str):
    
    question, answer, excerpt = str(question), str(answer), str(excerpt)
    
    # Definition of prompts
    prompts = {
        'context_adherence': f"Context adherence measures whether your model's response was purely based on the excerpt provided. A high Context Adherence score means your response is supported by the excerpt provided. Evaluate the excerpt adherence of the following text: {answer} excerpt: {excerpt}.The excerpt adherence should be given as a score from 0 to 100, where 100 is perfect excerpt adherence and 0 is lack of any excerpt adherence. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'completeness': f"Completeness measures how thoroughly your model's response covered the relevant information available in the excerpt provided. Evaluate the completeness of the following text: {answer} given the excerpt: {excerpt} and question: {question}. The completeness should be given as a score from 0 to 100, where 100 is perfect completeness and 0 is no completeness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'correctness': f"Correctness measures whether a given model response is factual or not. Correctness (f.k.a. Factuality) is a good way of uncovering open-domain hallucinations: factual errors that don't relate to any specific documents or excerpt. A high Correctness score means the response is more likely to be accurate vs a low response indicates a high probability for hallucination. Evaluate the correctness of this text: {answer} . The Correctness should be given as a score from 0 to 100, where 100 is perfect correctness and 0 is no correctness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'answer_relevancy': f"Measures how relevant the answer is to the user question. Higher answer relevance means that the answer is more relevant to the question. Evaluate the relevancy of this answer: {answer} given this question: {question}. The answer relevancy should be given as a score from 0 to 100, where 100 is perfect answer relevancy and 0 is no answer relevancy. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'readability_LLM_eval_Trott': f"Read the text below. Then, indicate the readability of the text, on a scale from 1 (extremely challenging to understand) to 100 (very easy to read and understand). In your assessment, consider factors such as sentence structure, vocabulary complexity, and overall clarity. Text: {answer}"
    }

    scores = {}
    for prompt_name, prompt_text in prompts.items():
        try:
            # Use asynchronous API call to OpenAI
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0
            )
            # Access the response content correctly
            score_str = response.choices[0].message.content.strip().split('Overall score: ')[-1]
            scores[prompt_name] = score_str
        except Exception as e:
            logging.error("Error in OpenAI API call for %s: %s", prompt_name, e)
            scores[prompt_name] = "Error"

    return scores

# Asynchronous row-by-row evaluation for runtime optimization
async def process_rows():
    logging.info("Starting row evaluation")
    results_list = []
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating rows"):
        question = row.get('Question', '')
        excerpt = row.get('Excerpts', '')
        row_results = {'Question': question, 'Excerpt': excerpt}

        # Evaluate each column in columns_to_evaluate
        for column in columns_to_evaluate:
            answer = row[column] if pd.notna(row[column]) else ""
            
            # Calculate zero-shot prompts
            prompt_scores = await calculate_zeroshot_prompts(question, excerpt, answer)

            # Append each prompt's score with the column name
            for prompt_name, score in prompt_scores.items():
                row_results[f"{column}_{prompt_name}"] = score

        # Append row results to list
        results_list.append(row_results)

    return pd.DataFrame(results_list)

# Run the async function
results = asyncio.run(process_rows())

# Save results to a file in the desired directory
# Create the output directory if it doesn't exist
output_dir = os.path.join(data_dir, 'analysed_data')
os.makedirs(output_dir, exist_ok=True)

try:
    # Save the results to the desired path
    results.to_csv(os.path.join(output_dir, 'llm_metrics_scores.tsv'), sep='\t', index=False)

    logging.info("Results successfully saved to llm_metrics_scores.tsv")
except Exception as e:
    logging.error("Failed to save results: %s", e)