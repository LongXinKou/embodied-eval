import os

import numpy as np
import pandas as pd

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
    
    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output

def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0