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
from embodied_eval.models import get_model
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
        "--model_args",
        default="",
        help="model_name_or_path=,lora_id=",
    )
    parser.add_argument(
        "--tasks",
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
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
    task_manager = TaskManager()

    task_list = args.tasks.split(",")
    task_names = task_manager.match_tasks(task_list)
    eval_logger.info(f"Selected Tasks: {task_names}")

    datetime_str = get_datetime_str(timezone=args.timezone)

    # Model
    if isinstance(args.model, str):
        if args.model_args is None:
            model_args = ""

        model = get_model(model_name=args.model).create_from_arg_string(
            model_args = model_args,
            additional_config = {
                "batch_size": args.batch_size,
            }
        )

    # Inference
    eval_tasks = SimpleInference(
        model=model,
        tasks=task_list,
        task_manager=task_manager,
        limit=args.limit,
        seed=args.seed,
    )

    # Evaluate
    results = SimpleEvaluate(
        model=model,
        eval_tasks=eval_tasks,
    )


if __name__ == "__main__":
    args = parse_args()
    cli_evaluate(args)