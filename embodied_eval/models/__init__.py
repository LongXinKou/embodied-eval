import abc

from typing import List, Optional, Tuple, Type, TypeVar, Union
from loguru import logger

from embodied_eval.utils import (
    simple_parse_args_string
)

T = TypeVar("T", bound="BaseAPIModel")

class BaseAPIModel(abc.ABC):
    def __init__(self) -> None:
        # set rank and world size to a single process, by default.
        self._rank = 0
        self._world_size = 1
        self.task_dict = {}

    @abc.abstractmethod
    def generate_until(self, requests) -> List[str]:
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            generation_kwargs: dict
                Generation Kwargs
            'visual_list: list[dict]'
                Visual input to the model. Can be None.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        pass

    @property
    def rank(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._rank

    @property
    def world_size(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._world_size

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict] = None) -> T:
        """
        Creates an instance of the LMM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LMM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

AVAILABLE_MODELS = {
    "robobrain": "RoboBrain",
}

def get_model(model_name):
    """
    model_name --> model_name.py --> model_class
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"embodied_eval.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class) # module.model_class
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise
