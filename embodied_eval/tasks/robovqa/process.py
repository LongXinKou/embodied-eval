'''
modified from "https://github.com/google-deepmind/robovqa/blob/main/data_loading_and_eval.ipynb"
bleu = sacrebleu.sentence_bleu(answer, pred_answer)
1. remove discrete question for BELU2-4
2. smooth_method='exp' (Method 7) <- "https://github.com/unira-zwj/PhysVLM/issues/2"
'''
import os
import numpy as np
import re
import pandas as pd

from collections import defaultdict
from openai import OpenAI
from typing import Optional
from tqdm import tqdm
from loguru import logger as eval_logger

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = ""
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_API_BASE")
)

METRICS_FOR_ROBOVQA = {
    # "BELU": "BELU_Eval"
    "llm_match_score": "llm_match"
}
ROBOVQA_QUESTION_TYPES = [
    "past_description:freeform",
    "immediate_planning_with_context20:freeform",
    "affordance:discriminative:discrete:False",
    "success:discrete:True",
    "future_prediction:freeform",
    "affordance:generative:positive:freeform",
    "success:discrete:False",
    "remaining5_planning_with_context20:freeform",
    "planning:freeform",
    "affordance:discriminative:discrete:True",
]

def robovqa_doc_to_visual(doc, dataset_kwargs=None):
    video_path = os.path.join(dataset_kwargs["video_dir"], doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def robovqa_doc_to_text(doc, dataset_kwargs=None):
    question = doc["question"]
    return question

def robovqa_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]
    pred_raw = results[0]
    
    target = doc["answer"]
    question_type = doc["question_type"]

    result_dict = {"target": target}
    result_dict["question_type"] = question_type
    for key, value in METRICS_FOR_ROBOVQA.items():
        pred = pred_raw
        # score = eval(value)(pred, target)
        # doc[key] = {'score': score.score, 'precisions': score.precisions, "bp": score.bp}
        score = eval(value)(doc["question"], target, pred)
        doc[key] = {key: score}

        result_dict[key] = doc[key]

    return result_dict

def robovqa_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}
    # key: {question_type}_{metric_name}
    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in ROBOVQA_QUESTION_TYPES:
            for metric in METRICS_FOR_ROBOVQA.keys():
                metric_data = per_question_type[metric].tolist()
                # avg_score = np.mean([x['score'] for x in metric_data])
                # avg_bp = np.mean([x['bp'] for x in metric_data])
                # avg_precisions = np.mean([x['precisions'] for x in metric_data], axis=0)  # element-wise mean for 4-gram precisions

                # output[f"{question_type}_{metric}"] = avg_score
                # output[f"{question_type}_{metric}-bp"] = avg_bp
                # output[f"{question_type}_{metric}1"] = avg_precisions[0]

                # if 'freeform' in question_type:
                #     output[f"{question_type}_{metric}2"] = avg_precisions[1]
                #     output[f"{question_type}_{metric}3"] = avg_precisions[2]
                #     output[f"{question_type}_{metric}4"] = avg_precisions[3]
                avg_score = np.mean([x[metric] for x in metric_data])
                output[f"{question_type}_{metric}"] = avg_score
    
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key:
            qtype, metric_name = key.rsplit("_", 1)
            if isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            avg_val = sum(vals) / len(vals)
            output[f"{metric_name}_average"] = avg_val

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output

def BELU_Eval(pred_answer, answer):
    import sacrebleu
    bleu = sacrebleu.sentence_bleu(
        pred_answer, 
        [answer],
        smooth_method='exp'
    )
    return bleu

def llm_match(
        question: str,
        answer: str,
        prediction: str,
        extra_answers = None,
        openai_model: str = "gpt-4o-mini",
        openai_seed: int = 1234,
        openai_max_tokens: int = 128,
        openai_temperature: float = 0.2,
        verbose: bool = False,
        max_tries: int = 3,
    ):
    import time
    
    if prediction is None:
        return 0
    
    prompt = load_prompt()

    messages = prepare_openai_messages(
        prompt.format(
            question=question,
            answer=answer,
            prediction=prediction,
            extra_answers=extra_answers,
        ),
    )
    
    for attempt in range(max_tries):
        try:
            output = call_openai_api(
                messages=messages,
                model=openai_model,
                seed=openai_seed,
                max_tokens=openai_max_tokens,
                temperature=openai_temperature,
                verbose=verbose,
            )
            return parse_score(output)
        except Exception as e:
            if attempt < max_tries - 1:
                eval_logger.warning(f"LLM evaluation failed (attempt {attempt + 1}/{max_tries}): {e}")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                eval_logger.error(f"LLM evaluation failed after {max_tries} attempts: {e}")
                return 0  # Return 0 score if all attempts fail


def load_prompt():
    prompt = """
    You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
    To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
    5 means that the response perfectly matches the answer.
    1 means that the response is completely different from the answer.

    Example 1:
    Question: Is it overcast?
    Answer: no
    Response: yes
    Your mark: 1

    Example 2:
    Question: Who is standing at the table?
    Answer: woman
    Response: Jessica
    Your mark: 3

    Example 3:
    Question: Are there drapes to the right of the bed?
    Answer: yes
    Response: yes
    Your mark: 5

    Your Turn:
    Question: {question}
    Answer: {answer}
    Response: {prediction}    
    """
    return prompt

def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]

def call_openai_api(
    messages: list,
    model: str = "gpt-4o",
    seed = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content


def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())

def post_evaluate_results(sample_file_path, results_file_path):
    import json
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    for doc in tqdm(data):
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        target = doc["target"]
        question_type = doc["question_type"]
        question = doc.get("question", doc.get("doc", ""))

        result_dict = {"target": target}
        result_dict["question_type"] = question_type
        
        for key, value in METRICS_FOR_ROBOVQA.items():
            pred = pred_raw
            # Call the evaluation function directly with the new signature
            score = eval(value)(question, target, pred)
            doc[key] = {key: score}
            result_dict[key] = doc[key]

        results.append(result_dict)

    # samples_robovqa.json
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    output = robovqa_aggregate_results(results)

    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    base_dir = "/home/lx/embodied-eval/logs/logs_robobrain2/robovqa-7b/"
    post_evaluate_results(
        sample_file_path=f"{base_dir}/samples_robovqa.json",
        results_file_path=f"{base_dir}/results_robovqa.json"
    )