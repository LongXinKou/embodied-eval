import argparse
import datetime
import torch
import random
import json
import os
import time
import numpy as np

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm

from embodied_eval.inference import SimpleInference
from embodied_eval.tasks import TaskManager
from embodied_eval.utils import (
    get_datetime_str
)

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
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    return parser.parse_args()

def cli_evaluate(args):

    results_list = []

    # initialize Accelerator
    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
    if accelerator.is_main_process:
        is_main_process = True
    else:
        is_main_process = False

    try:
        results, samples = cli_evaluate_single(args)
        results_list.append(results)
        accelerator.wait_for_everyone()

    except Exception as e:
        eval_logger.error(
            f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")
        results_list.append(None)

def cli_evaluate_single(args):
    task_manager = TaskManager(model_name=args.model)

    task_list = args.tasks.split(",")
    task_names = task_manager.match_tasks(task_list)
    eval_logger.info(f"Selected Tasks: {task_names}")

    datetime_str = get_datetime_str(timezone=args.timezone)

    # Model
    if isinstance(args.model, str):
        if args.model_args is None:
            model_args = ""

        model =

    # Inference
    prediction = SimpleInference()

    # Evaluate
    results = SimpleEvaluate()


if __name__ == "__main__":
    args = parse_args()
    cli_evaluate(args)