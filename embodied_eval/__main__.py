import argparse
import os
import sys

from loguru import logger as eval_logger

from embodied_eval.evaluators import build_evaluator

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
        "--evaluator",
        type=str,
        default="eqa",
        help="Choose Evaluator: eqa or nav. "
    )
    parser.add_argument(
        "--tasks",
        default=None,
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Environment name for navigation tasks"
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
        "--inference_only",
        default=False,
        type=bool,
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
    evaluator = build_evaluator(args)

    try:
        inference_results = evaluator.inference()
        if args.inference_only:
            eval_logger.info("Inference completed, exiting without evaluation.")
            return
        else:
            eval_logger.info("Inference completed, proceeding to evaluation.")
            results_dict = evaluator.evaluate(inference_results)
            evaluator.print_results(results_dict)
            if args.save_results:
                eval_logger.info("Saving results...")
                evaluator.save_results(results_dict)

    except Exception as e:
        eval_logger.error(f"Error during evaluation: {e}.")

if __name__ == "__main__":
    args = parse_args()
    cli_evaluate(args)