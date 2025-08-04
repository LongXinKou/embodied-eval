'''
modified from "https://github.com/embodiedreasoning/ERQA/blob/main/eval_harness.py"
is_correct = response_text.replace(".", "").strip().lower() == answer.strip().lower()
'''
import os

import numpy as np
import pandas as pd

from collections import defaultdict
from loguru import logger as eval_logger

# MCA
METRICS_FOR_ERQA = {"accuracy": "exact_match"}
ERQA_QUESTION_TYPES = [
    "Action Reasoning",
    "Multi-view Reasoning",
    "Other",
    "Pointing",
    "State Estimation",
    "Spatial Reasoning",
    "Trajectory Reasoning",
    "Task Reasoning",
]

def erqa_doc_to_visual(doc, dataset_kwargs=None):
    if len(doc["visual_indices"]) == 0:
        return doc["images"]
    else:
        return (doc["images"], doc["visual_indices"])


def erqa_doc_to_text(doc, dataset_kwargs=None):
    return doc["question"]

def extract_single_word_option(text):
    """
    Extract a single word option from the prediction text.
    Handles various response formats and extracts the most likely answer.
    """
    if not text:
        return ""
    
    # Clean the text
    text = text.strip()
    
    # Common patterns for extracting answers
    import re
    
    # Pattern 1: Look for "Answer: X" or "The answer is X"
    answer_patterns = [
        r'(?:answer|Answer)(?:\s*is)?\s*:\s*([A-Za-z]+)',
        r'(?:the|The)\s+answer\s+is\s+([A-Za-z]+)',
        r'^\s*([A-Za-z])\s*[\.\):]',  # Single letter at start with punctuation
        r'\b([A-Za-z])\s*[\.\):]?\s*$',  # Single letter at end
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return text

def erqa_process_results(doc, results, dataset_kwargs=None):
    pred_raw = results[0]
    # Process the raw prediction to extract single word option
    processed_pred = extract_single_word_option(pred_raw)
    doc["prediction"] = processed_pred
    target = doc["answer"]

    result_dict = {"target": target}
    result_dict["question_type"] = doc.get("question_type", "erqa")
    for key, value in METRICS_FOR_ERQA.items():
        doc[key] = eval(value)(doc["prediction"], target)
        result_dict[key] = doc[key]

    return result_dict

def erqa_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in ERQA_QUESTION_TYPES:
            for metric in METRICS_FOR_ERQA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
    
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

def exact_match(response_text, answer):
    return response_text.replace(".", "").strip().lower() == answer.strip().lower()

def post_evaluate_results(sample_file_path, results_file_path):
    import json
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    for doc in data:
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        # Process the raw prediction to extract single word option
        processed_pred = extract_single_word_option(pred_raw)
        target = doc["target"]
        question_type = doc["question_type"]

        for key, value in METRICS_FOR_ERQA.items():
            doc[key] = eval(value)(processed_pred, target)

            result_dict = {
                "question_type": question_type,
                key: doc[key]
            }
            results.append(result_dict)
    
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    output = erqa_aggregate_results(results)

    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    post_evaluate_results(
        sample_file_path="",
        results_file_path=""
    )