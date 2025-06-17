import numpy as np
import pandas as pd
import os
import re
import json

from collections import defaultdict
from PIL import Image
from loguru import logger as eval_logger

METRICS_FOR_WHERE2PLACE = {"accuracy": "spatial_reference"}

def where2place_doc_to_visual(doc, dataset_kwargs=None):
    image = []
    image_path = os.path.join(dataset_kwargs["image_dir"], doc["image"])
    
    image.append(Image.open(image_path).convert("RGB"))
    return image

def where2place_doc_to_text(doc, dataset_kwargs=None):
    question = doc["text"]
    if (
        "pre_prompt" in dataset_kwargs
        and dataset_kwargs["pre_prompt"] != ""
    ):
        question = f"{dataset_kwargs['pre_prompt']} {question}"
    if (
        "post_prompt" in dataset_kwargs
        and dataset_kwargs["post_prompt"] != ""
    ):
        question = f"{question} {dataset_kwargs['post_prompt']}"
    return question

def where2place_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]

    target = np.array(Image.open(os.path.join(dataset_kwargs["target_image_dir"], doc["image"]))) / 255.
    result_dict = {"target": mask_to_bbox(target)}
    result_dict["question_type"] = doc.get("question_type", "where2place")
    
    for key, value in METRICS_FOR_WHERE2PLACE.items():
        doc[key] = eval(value)(doc["prediction"], target)
        result_dict[key] = doc[key]

    return result_dict

def where2place_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        for metric in METRICS_FOR_WHERE2PLACE.keys():
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


def spatial_reference(pred, mask, width=640, height=480):
    try:
        points = text2points(pred.strip(), width=width, height=height)
        if isinstance(mask, list) and len(mask) == 4:
            x0, y0, x1, y1 = mask
            binary_mask = np.zeros((height, width), dtype=bool)
            binary_mask[y0:y1, x0:x1] = 1
            mask = binary_mask
        
        if len(points) > 0:
            in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) \
                        & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
            acc = np.concatenate([
                mask[points[in_range, 1], points[in_range, 0]],
                np.zeros(points.shape[0] - in_range.sum())
            ]).mean()
            
        return acc
    except:
        return 0

def text2points(text, width=640, height=480):
    points = []

    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    for match in matches:
        vector = [
            float(num) if '.' in num else int(num) for num in match.split(',')
        ]
        if len(vector) == 2:
            x, y = vector
            if isinstance(x, float) or isinstance(y, float):
                x = int(x * width)
                y = int(y * height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if isinstance(x0, float):
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
            y, x = np.where(mask)
            points.extend(list(np.stack([x, y], axis=1)))
    if points:
        return np.array(points)
    
    try:
        if '```' in text:
            text_clean = text.split('```json')[-1].split('```')[0].strip()
        else:
            text_clean = text

        data = json.loads(text_clean)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "point" in item:
                    pt = item["point"]
                    if isinstance(pt, list) and len(pt) == 2:
                        x = int(pt[0] * width)
                        y = int(pt[1] * height)
                        points.append((x, y))
    except Exception:
        pass  # ignore JSON errors

    return np.array(points)

def mask_to_bbox(mask, threshold=0.5):
    binary_mask = mask > threshold
    ys, xs = np.where(binary_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return (x0, y0, x1, y1)

def post_process_results(sample_file_path, results_file_path):
    import json
    from collections import defaultdict
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    type_correct = defaultdict(float)
    type_total = defaultdict(int)
    for doc in data:
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        target = doc["target"]
        
        acc = spatial_reference(pred_raw, target)
        doc["accuracy"] = acc 
        
        qtype = doc["question_type"]
        type_correct[qtype] += acc
        type_total[qtype] += 1

    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    type_success_rate = {
        f"{qtype}_accuracy": round(type_correct[qtype] / type_total[qtype], 4)
        for qtype in type_total
    }
    values = list(type_success_rate.values())
    overall = round(sum(values) / len(values), 4)
    type_success_rate["overall"] = overall

    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(type_success_rate, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    post_process_results(
        sample_file_path="",
        results_file_path=""
    )