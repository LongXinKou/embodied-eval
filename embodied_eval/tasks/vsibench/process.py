import os

import numpy as np
import pandas as pd

from functools import partial
from loguru import logger as eval_logger

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}
WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}

def vsibench_doc_to_visual(doc, dataset_kwargs=None):
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(dataset_kwargs["video_dir"], video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def vsibench_doc_to_text(doc, dataset_kwargs=None):
    question = doc["question"]

    pre_prompt = "These are frames of a video."

    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def vsibench_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]

    result_dict = {"target": doc["ground_truth"]}
    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc["prediction"]), doc["ground_truth"])
            result_dict[key] = doc[key]
            result_dict["question_type"] = doc["question_type"]

    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc["prediction"])), to_float(doc["ground_truth"]))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
            result_dict[key] = doc[key]
            result_dict["question_type"] = doc["question_type"]
    return result_dict


def vsibench_aggregate_results(results):
    """
    Input
        - results: Optional(list(dict))
        [{question_type:, accuracy:,}, {...}]

    Output
        - {question-type_metric:, ..., overall: }
    """
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

    output["object_rel_direction_accuracy"] = (
        sum(
            [
                output.pop("object_rel_direction_easy_accuracy"),
                output.pop("object_rel_direction_medium_accuracy"),
                output.pop("object_rel_direction_hard_accuracy"),
            ]
        )
        / 3.0
    )

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output

def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()

def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred
