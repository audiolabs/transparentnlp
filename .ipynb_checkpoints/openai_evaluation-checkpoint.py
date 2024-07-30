import openai
import os
import pandas as pd

# Ensure you have your OpenAI API key set up in the environment
os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ").strip()


# Function to calculate zero-shot prompts
def calculate_zeroshot_prompts(query: str, context: str, hypothesis: str):
    if not isinstance(query, str):
        query = str(query)
    if not isinstance(hypothesis, str):
        hypothesis = str(hypothesis)
    if not isinstance(context, str):
        context = str(context)
    prompts = {
        'context_adherence': f"Context adherence measures whether your model's response was purely based on the context provided. A high Context Adherence score means your response is supported by the context provided. Evaluate the context adherence of the following text: {hypothesis} context: {context}.The context adherence should be given as a score from 0 to 100, where 100 is perfect context adherence and 0 is lack of any context adherence. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'completeness': f"Completeness measures how thoroughly your model's response covered the relevant information available in the context provided. Evaluate the completeness of the following text: {hypothesis} given the context: {context} and question: {query}. The completeness should be given as a score from 0 to 100, where 100 is perfect completeness and 0 is no completeness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'correctness': f"Correctness measures whether a given model response is factual or not. Correctness (f.k.a. Factuality) is a good way of uncovering open-domain hallucinations: factual errors that don't relate to any specific documents or context. A high Correctness score means the response is more likely to be accurate vs a low response indicates a high probability for hallucination. Evaluate the correctness of this text: {hypothesis} . The Correctness should be given as a score from 0 to 100, where 100 is perfect correctness and 0 is no correctness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'answer_relevancy': f"Measures how relevant the answer is to the user question. Higher answer relevance means that the answer is more relevant to the question. Evaluate the relevancy of this answer: {hypothesis} given this question: {query}. The answer relevancy should be given as a score from 0 to 100, where 100 is perfect answer relevancy and 0 is no answer relevancy. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'readability_LLM_eval_Trott': f"Read the text below. Then, indicate the readability of the text, on a scale from 1 (extremely challenging to understand) to 100 (very easy to read and understand). In your assessment, consider factors such as sentence structure, vocabulary complexity, and overall clarity. Text: {hypothesis}"
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


def calculate_Ragas_metrics(query: str, context: str, hypothesis: str, reference: str):
    if not isinstance(query, str):
        query = str(query)
    if not isinstance(hypothesis, str):
        hypothesis = str(hypothesis)
    if not isinstance(context, str):
        context = str(context)
    if not isinstance(reference, str):
        reference = str(reference)
    prompts = {
        'faithfulness': f"Measure the factual consistency of the hypothesis: {hypothesis} against the given context: {context}.Return a value between 0 to 100. Your response should be in this format, 'Score: number'",
        'answer relevancy': f"Measure how relevant is the hypothesis: {hypothesis} to the query: {query}.Return a value between 0 to 100. Your response should be in this format, 'Score: number'",
        'context correctness': f"Measure how correct is the hypothesis: {hypothesis} in comparison to the reference: {reference}.Return a value between 0 to 100. Your response should be in this format, 'Score: number'",
        'context relevancy': f"Measure how relevant is the hypothesis: {hypothesis} to the given context: {context}.Return a value between 0 to 100. Your response should be in this format, 'Score: number'.",
        'answer semantic similarity': f"Measure how similar is the hypothesis: {hypothesis} to the reference {reference}.Return a value between 0 to 100. Your response should be in this format, 'Score: number'"
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


def process_csv(input_file, output_dir):
    # Read input CSV into DataFrame
    df = pd.read_csv(input_file, sep='\t')

    # Define column combinations
    columns_combinations = [
        ('Rag_Rain_dishon', 'Designed_Answer_1'),
        ('Rag_Rain_dishon', 'Designed_Answer_2'),
        ('Rag_Rain_comp', 'Designed_Answer_1'),
        ('Rag_Rain_comp', 'Designed_Answer_2')
    ]

    for rag_column, answer_column in columns_combinations:
        # Apply zero-shot prompts calculations
        df['zero_shot'] = df.apply(
            lambda row: calculate_zeroshot_prompts(row['Question'], row['Excerpts'], row[rag_column]), axis=1)

        # Apply Ragas metrics calculations
        df['ragas'] = df.apply(
            lambda row: calculate_Ragas_metrics(row['Question'], row['Excerpts'], row[rag_column], row[answer_column]),
            axis=1)

        # Define output file path based on column names
        output_file = os.path.join(output_dir, f'output_{rag_column}_{answer_column}.tsv')

        # Save the DataFrame to a TSV file
        df.to_csv(output_file, sep='\t', index=False)


# Example usage for the first input file
input_file_1 = os.path.join('Config.DATA_DIR', 'processed_data.tsv')
output_dir = 'Config.VAL_DIR'
process_csv(input_file_1, output_dir)
