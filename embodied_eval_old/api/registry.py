from typing import Callable, Dict, Union

import evaluate as hf_evaluate
from loguru import logger as eval_logger

from embodied_eval_old.api.model import lmms

MODEL_REGISTRY = {}


FILTER_REGISTRY = {}

def register_filter(name):
    def decorate(cls):
        if name in FILTER_REGISTRY:
            eval_logger.info(f"Registering filter `{name}` that is already in Registry {FILTER_REGISTRY}")
        FILTER_REGISTRY[name] = cls
        return cls

    return decorate

def get_filter(filter_name: Union[str, Callable]) -> Callable:
    try:
        return FILTER_REGISTRY[filter_name]
    except KeyError as e:
        if callable(filter_name):
            return filter_name
        else:
            eval_logger.warning(f"filter `{filter_name}` is not registered!")
            raise e