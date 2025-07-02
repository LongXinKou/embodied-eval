from embodied_eval.evaluators.eqa_evaluator import EQAEvaluator
from embodied_eval.evaluators.nav_evaluator import NavEvaluator

def build_evaluator(args):
    """
    Build the evaluator based on the provided arguments.
    """
    if args.evaluator == 'nav':
        return NavEvaluator(args)
    else:
        return EQAEvaluator(args)