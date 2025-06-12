import numpy as np
import os
import re

from PIL import Image

WHERE2PLACE_METRICS = {"acc"}

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
        question = f"{dataset_kwargs['pre_prompt']}{question}"
    if (
        "post_prompt" in dataset_kwargs
        and dataset_kwargs["post_prompt"] != ""
    ):
        question = f"{question}{dataset_kwargs['post_prompt']}"
    return question

def where2place_process_results(doc, results, dataset_kwargs=None):
    acc = 0
    try:
        points = text2points(results[0].strip())
    except:
        print('Failed to parse answer for question', results)
        return acc
    
    mask_image_path = os.path.join(dataset_kwargs["target_image_dir"], doc["image"])
    mask = np.array(Image.open(mask_image_path)) / 255.
    if len(points) > 0:
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) \
                    & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        acc = np.concatenate([
            mask[points[in_range, 1], points[in_range, 0]],
            np.zeros(points.shape[0] - in_range.sum())
        ]).mean()
    
    result_dict = {"acc": acc}

    return {f"where2place_{metric}": value for metric, value in result_dict.items()}

def text2points(text, width=640, height=480):
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []
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
    return np.array(points)