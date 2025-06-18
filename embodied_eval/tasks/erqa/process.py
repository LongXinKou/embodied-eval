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
    return doc["images"]

def erqa_doc_to_text(doc, dataset_kwargs=None):
    return doc["question"]

def erqa_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]
    target = doc["answer"]

    result_dict = {"target": target}
    result_dict["question_type"] = doc.get("question_type", "erqa")
    for key, value in METRICS_FOR_ERQA.items():
        doc[key] = eval(value)(fuzzy_matching(doc["prediction"]), target)
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

def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0

def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()

def post_evaluate_results(sample_file_path, results_file_path):
    import json
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    for doc in data:
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        target = doc["target"]
        question_type = doc["question_type"]

        for key, value in METRICS_FOR_ERQA.items():
            doc[key] = eval(value)(fuzzy_matching(pred_raw), target)

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