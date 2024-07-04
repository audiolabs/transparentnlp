import nltk
nltk.download('punkt')
from util import set_openai_api_key
set_openai_api_key()

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
from textstat import flesch_reading_ease
from nltk.tokenize import sent_tokenize, word_tokenize
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate
from evaluate import load
from qa_metrics.em import em_match
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

# Import data classes from separate file
from metrics_dataclasses import (
    BleuMetric, RougeMetric, ReadabilityMetric, LexicalDiversityMetric,
    AvgSentenceLengthMetric, BertScoreMetric, SemanticSimilarityMetric,
    FaithfulnessMetric, AnswerRelevancyMetric, AnswerSimilarityMetric,
    AnswerCorrectnessMetric, RqugeScoreMetric, GevalMetric,
    SelfCheckCorrectnessMetric, ZeroshotPromptsMetric
)

# Define metrics functions
def calculate_bleu_sentence(metric: BleuMetric):
    return sentence_bleu([metric.reference.split()], metric.hypothesis.split())

def calculate_bleu_corpus(metric: BleuMetric):
    reference_sentences = sent_tokenize(metric.reference)
    hypothesis_sentences = sent_tokenize(metric.hypothesis)
    reference_corpus = [reference_sentences]
    hypothesis_corpus = [hypothesis_sentences]
    return corpus_bleu(reference_corpus, hypothesis_corpus)

def calculate_rouge(metric: RougeMetric):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(metric.reference, metric.hypothesis)
    return scores['rouge1'].fmeasure

def calculate_readability(metric: ReadabilityMetric):
    return flesch_reading_ease(metric.hypothesis)

def calculate_lexical_diversity(metric: LexicalDiversityMetric):
    # Replace mtld with other metric
    return lex.mtld(metric.hypothesis, threshold=metric.threshold)

def calculate_avg_sentence_length(metric: AvgSentenceLengthMetric):
    sentences = sent_tokenize(metric.hypothesis)
    total_sentences = len(sentences)
    if total_sentences > 0:
        total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
        avg_sentence_length = total_words / total_sentences
    else:
        avg_sentence_length = 0.0
    return avg_sentence_length

def calculate_bert_score(metric: BertScoreMetric):
    scorer = BERTScorer(model_type='bert-base-uncased', lang="en", rescale_with_baseline=True)
    P, R, F1 = scorer.score([metric.hypothesis], [metric.reference], verbose=False)
    return F1[0].item()

def calculate_semantic_similarity(metric: SemanticSimilarityMetric):
    model = SentenceTransformer(metric.model)
    embeddings = model.encode([metric.reference, metric.hypothesis], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

def calculate_ragas_faithfulness(metric: FaithfulnessMetric):
    data_samples = {
        'question': [metric.query],
        'answer': [metric.hypothesis],
        'contexts': [[metric.context]]
    }
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset, metrics=[faithfulness])
    return score['faithfulness']

def calculate_ragas_answer_relevancy(metric: AnswerRelevancyMetric):
    data_samples = {
        'question': [metric.query],
        'answer': [metric.hypothesis],
        'contexts': [[metric.context]]
    }
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset, metrics=[answer_relevancy])
    return score['answer_relevancy']

def calculate_ragas_answer_similarity(metric: AnswerSimilarityMetric):
    data_samples = {
        'question': [metric.query],
        'answer': [metric.hypothesis],
        'reference': [metric.reference]
    }
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset, metrics=[answer_similarity])
    return score['answer_similarity']

def calculate_ragas_answer_correctness(metric: AnswerCorrectnessMetric):
    data_samples = {
        'question': [metric.query],
        'answer': [metric.hypothesis],
        'reference': [metric.reference]
    }
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset, metrics=[answer_correctness])
    return score['answer_correctness']

def calculate_rquge_score(metric: RqugeScoreMetric):
    rquge_score = load(metric.model_name)
    results = rquge_score.compute(generated_questions=[metric.query],
                                  context=[metric.context],
                                  answers=[metric.hypothesis])
    return results["mean_score"]

def calculate_geval_metric(metric: GevalMetric):
    input_to_llm = f"{metric.query} {metric.context}"
    test_case = LLMTestCase(input=input_to_llm, actual_output=metric.hypothesis)
    coherence_metric = GEval(
        name="Coherence",
        criteria="Coherence - the collective quality of sentences in the actual output",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    coherence_metric.measure(test_case)
    return coherence_metric.score

def calculate_selfcheck_correctness(metric: SelfCheckCorrectnessMetric):
    prompt = f"User Query: {metric.query}, \n, Context: {metric.context} \n Generate {metric.num_samples} comma-separated answers in a list."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": prompt}
        ],
        stop=None,
        temperature=0,
        top_p=0
    )
    generated_answers = [choice['message']['content'].strip() for choice in response['choices']]
    hypothesis_embedding = model.encode(metric.hypothesis, convert_to_tensor=True)
    sts_scores = []
    for answer in generated_answers:
        answer_embedding = model.encode(answer, convert_to_tensor=True)
        sts_score = util.pytorch_cos_sim(hypothesis_embedding, answer_embedding)
        sts_scores.append(sts_score.item())
    avg_sts_score = sum(sts_scores) / len(sts_scores)
    return avg_sts_score

# The following function is based on Dr. Birgit Popp's implementation. For more information, please refer to: https://gitlab.cc-asp.fraunhofer.de/hmi/data-governance/offline_evaluation

def calculate_zeroshot_prompts(metric: ZeroshotPromptsMetric):
    prompts = {
        'context_adherence': f"Context adherence measures whether your model's response was purely based on the context provided. A high Context Adherence score means your response is supported by the context provided. Evaluate the context adherence of the following text: {metric.hypothesis} context: {metric.context}.The context adherence should be given as a score from 0 to 100, where 100 is perfect context adherence and 0 is lack of any context adherence. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'completeness': f"Completeness measures how thoroughly your model's response covered the relevant information available in the context provided. Evaluate the completeness of the following text: {metric.hypothesis} given the context: {metric.context} and question: {metric.query}. The completeness should be given as a score from 0 to 100, where 100 is perfect completeness and 0 is no completeness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'correctness': f"Correctness measures whether a given model response is factual or not. Correctness (f.k.a. Factuality) is a good way of uncovering open-domain hallucinations: factual errors that don't relate to any specific documents or context. A high Correctness score means the response is more likely to be accurate vs a low response indicates a high probability for hallucination. Evaluate the correctness of this text: {metric.hypothesis} . The Correctness should be given as a score from 0 to 100, where 100 is perfect correctness and 0 is no correctness. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'answer_relevancy': f"Measures how relevant the answer is to the user question. Higher answer relevance means that the answer is more relevant to the question. Evaluate the relevancy of this answer: {metric.hypothesis} given this question: {metric.query}. The answer relevancy should be given as a score from 0 to 100, where 100 is perfect answer relevancy and 0 is no answer relevancy. Think step by step, and present your reasoning before giving the answer. After reasoning, provide an overall score in the following format: 'Overall score: number'. The overall score can be an average of scores that you come up with during the reasoning. If no sensible overall score can be provided, because the metric does not apply then you can provide 'Overall score: NA'.",
        'readability_LLM_eval_Trott': f"Read the text below. Then, indicate the readability of the text, on a scale from 1 (extremely challenging to understand) to 100 (very easy to read and understand). In your assessment, consider factors such as sentence structure, vocabulary complexity, and overall clarity. Text: {metric.hypothesis}"
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
        try:
            score_str = response.choices[0].message.content.strip().split('Overall score: ')[-1]
            score = float(score_str)
        except Exception as e:
            score = "NA"
            print(f"Error extracting score for '{prompt_name}': {e}")
        scores[prompt_name] = score
    return scores

