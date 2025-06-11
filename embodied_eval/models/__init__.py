import abc

from typing import List, Optional, Tuple, Type, TypeVar, Union


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
