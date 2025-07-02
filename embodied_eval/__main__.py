import argparse
import datetime
import os
import sys

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from embodied_eval.tasks import TaskManager
from embodied_eval.models import get_model
from embodied_eval.utils import (
    get_datetime_str
)
from embodied_eval.evaluators import EQAEvaluator

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
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1
    )
    parser.add_argument(
        "--save_results",
        default=True,
        type=bool,
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
        "--verbosity",
        type=str,
        default="DEBUG",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    return parser.parse_args()


def cli_evaluate(args):
    # reset logger
    eval_logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity

    # initialize Evaluator
    evaluator = EQAEvaluator(args)

    try:
        eval_tasks = evaluator.inference()
        results_dict = evaluator.evaluate(eval_tasks)
        evaluator.print_results(results_dict)
        if args.save_results:
            evaluator.save_results(results_dict)

    except Exception as e:
        eval_logger.error(
            f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")

if __name__ == "__main__":
    args = parse_args()
    cli_evaluate(args)