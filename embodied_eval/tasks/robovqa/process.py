
import os
import numpy as np
import re
import sacrebleu
import pandas as pd

from loguru import logger as eval_logger


METRICS_FOR_ROBOVQA = {"BELU": "BELU_Eval"}
ROBOVQA_QUESTION_TYPES = [
    "planning",
    "remaining5_planning_with_context",
    "immediate_planning_with_context",
    "future_prediction",
    "success",
    "affordance",
    "past_description"
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
    
    target = doc["answer"]
    result_dict = {"target": target}
    result_dict["question_type"] = extract_task_type_tags["task_type"][0]

    for key, value in METRICS_FOR_ROBOVQA.items():
        score = eval(value)(doc["prediction"], target)
        doc[key] = {'score': score.score, 'precisions': score.precisions, "bp": score.bp}
        result_dict[key] = doc[key]

    return result_dict

def robovqa_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in ROBOVQA_QUESTION_TYPES:
            for metric in METRICS_FOR_ROBOVQA.keys():
                metric_data = per_question_type[metric].tolist()
                avg_score = np.mean([x['score'] for x in metric_data])
                avg_bp = np.mean([x['bp'] for x in metric_data])
                avg_precisions = np.mean([x['precisions'] for x in metric_data], axis=0)  # element-wise mean for 4-gram precisions

                output[f"{question_type}_{metric}_score"] = avg_score
                output[f"{question_type}_{metric}_bp"] = avg_bp
                output[f"{question_type}_{metric}_precisions"] = avg_precisions.tolist()
    
    output["overall"] = avg_score
    eval_logger.info(f"Evaluation results: {output}")
    return output

def BELU_Eval(pred_answer, answer):
    bleu = sacrebleu.sentence_bleu(answer, pred_answer)
    return bleu

def extract_task_type_tags(task_type_string: str) -> list:
    if not task_type_string:
        return []
    tags = task_type_string.split(':')
    return tags