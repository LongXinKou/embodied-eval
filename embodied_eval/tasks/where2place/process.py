import os

from PIL import Image

def doc_to_visual(doc, dataset_kwargs=None):
    image = []
    image_path = os.path.join(dataset_kwargs["image_dir"], doc["image"])
    
    image.append(Image.open(image_path).convert("RGB"))
    return image

def doc_to_text(doc, dataset_kwargs=None):
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