import argparse
import torch
import random
import json
import os
import time
import numpy as np

from loguru import logger as eval_logger
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,)
    parser.add_argument(
        "--pretrained",
        type=str, )
    parser.add_argument(
        "--model_args",
        default="",
    )
    parser.add_argument(
        "--tasks",
        default=None,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1)
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
    )
    return parser.parse_args()

def run_inference(args):
    try:
        results, samples = run_inference_for_single_task(args)
        results_list.append(results)
        accelerator.wait_for_everyone()

    except Exception as e:
        eval_logger.error(
            f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")
        results_list.append(None)

def run_inference_for_single_task(args):
    task_list = args.tasks.split(",")
    task_names = task_list # TODO

    eval_logger.info(f"Selected Tasks: {task_names}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)