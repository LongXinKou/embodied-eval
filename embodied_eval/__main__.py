import argparse
import datetime
import os
import sys

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from embodied_eval.inference import SimpleInference
from embodied_eval.evaluate import SimpleEvaluate, SaveResult
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

    # initialize Accelerator
    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])

    try:
        datetime_str = get_datetime_str(timezone=args.timezone)

        results_dict = cli_evaluate_single(args)

        if results_dict:
            # Visualize
            print_results(results_dict, args)

            # Save
            SaveResult(
                results_dict=results_dict,
                output_path=args.output_path,
                datetime_str=datetime_str
            )

    except Exception as e:
        eval_logger.error(
            f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")

def cli_evaluate_single(args):
    task_manager = TaskManager()

    task_list = args.tasks.split(",")
    task_names = task_manager.match_tasks(task_list)
    eval_logger.info(f"Selected Tasks: {task_names}")

    # Model
    if isinstance(args.model, str):
        if args.model_args is None:
            model_args = ""

        model = get_model(model_name=args.model).create_from_arg_string(
            args.model_args,
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
    results_dict = SimpleEvaluate(
        model=model,
        eval_tasks=eval_tasks,
    )

    return results_dict

def print_results(results_dict, args):
    results = results_dict["results"]
    configs = results_dict["configs"]
    for task_name, _ in results.items():
        result, config = results[task_name], configs[task_name]
        info = f"task:{config['task']}, data:{config['dataset_path']}/{config['dataset_kwargs']},\
            generate:{config['generation_kwargs']}"
        eval_logger.info(info)
        make_table(result, args)
    
def make_table(result, args):
    from pytablewriter import MarkdownTableWriter
    import re

    model = args.model

    type_to_metrics = {}
    for key, val in result.items():
        if key == "overall" or key.endswith("_stderr"):
            continue
        match = re.match(r"(.+?)_([a-zA-Z0-9:.]+)$", key)
        if match:
            qtype, metric = match.groups()
            if qtype not in type_to_metrics:
                type_to_metrics[qtype] = {}
            type_to_metrics[qtype][metric] = val

    all_metrics = sorted({m for v in type_to_metrics.values() for m in v})

    headers = ["Metric"] + list(type_to_metrics.keys())
    value_matrix = []

    for metric in all_metrics:
        row = [metric]
        for qtype in type_to_metrics:
            val = type_to_metrics[qtype].get(metric, "")
            row.append(f"{val:.4f}" if val != "" else "")
        value_matrix.append(row)

    # Step 4: 输出 markdown 表格
    from pytablewriter import MarkdownTableWriter
    md_writer = MarkdownTableWriter()
    md_writer.table_name = f"Results for {model}"
    md_writer.headers = headers
    md_writer.value_matrix = value_matrix

    eval_logger.info("Markdown Table:\n")
    eval_logger.info(f"Overall: {result['overall']}")
    md_writer.write_table()


if __name__ == "__main__":
    args = parse_args()
    cli_evaluate(args)