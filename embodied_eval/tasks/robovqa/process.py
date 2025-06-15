
import os
import numpy as np
import re


def robovqa_doc_to_visual(doc, dataset_kwargs=None):
    video_path = os.path.join(dataset_kwargs["video_dir"], parse_item(doc))
    return [video_path]

def robovqa_doc_to_text(doc, dataset_kwargs=None):
    question = parse_item(doc)["question"]
    return question

def robovqa_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]
    
    parse_doc = parse_item(doc)
    target = parse_doc["answer"]
    result_dict = {"target": target}
    result_dict["question_type"] = parse_doc.get("task", "robovqa")
    
    for key, value in METRICS_FOR_WHERE2PLACE.items():
        doc[key] = eval(value)(doc["prediction"], target)
        result_dict[key] = doc[key]

    return result_dict

def robovqa_aggregate_results(results):
    pass

def parse_item(entry):
    task_match = re.search(r"<task:(.*?)>", entry["text"])
    question_match = re.search(r"Q:\s*(.*?)(<PRED>|$)", entry["text"])
    answer_match = re.search(r"<PRED>.*?A:\s*(.*?)</PRED>", entry["text"], re.DOTALL)

    return {
        "uid": entry["uid"],
        "task": task_match.group(1) if task_match else None,
        "question": question_match.group(1).strip() if question_match else None,
        "answer": clean_text(answer_match.group(1)) if answer_match else None,
        "video": entry["video"]
    }

def clean_text(text):
    return re.sub(r"<[^>]+>", "", text).strip()