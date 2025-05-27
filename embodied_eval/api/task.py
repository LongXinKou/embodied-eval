import abc
import ast
import copy
import inspect
import itertools
import json
import os
import random
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from glob import glob
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import datasets
import numpy as np
from accelerate import Accelerator
from datasets import Audio, DownloadConfig, Image, Sequence
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import ImageFile
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed
from tqdm import tqdm

from embodied_eval.filters import build_filter_ensemble
from embodied_eval import utils

@dataclass
class TaskConfig(dict):
    # task naming/registry
    task: str = None
    task_alias: str = None
    tag: str = None
    group: Union[str, list] = None
    group_alias: Union[str, list] = None
    # HF dataset options.
    # which dataset to use,
    # and what splits for what purpose
    dataset_path: str = None
    dataset_name: str = None
    dataset_kwargs: dict = None
    training_split: str = None
    validation_split: str = None
    test_split: str = None
    fewshot_split: str = None  # TODO: assert that this not None if num_fewshot > 0. (?) assert if this is same split as one evaling (?)
    full_docs: bool = False
    # formatting / prompting options.
    # see docs/advanced_task_guide.md for more info
    process_results_use_image: bool = False
    process_docs: Callable = None
    doc_to_visual: Union[Callable, str] = None
    doc_to_text: Union[Callable, str] = None
    doc_to_target: Union[Callable, str] = None
    doc_to_choice: Union[Callable, str, dict, list] = None
    process_results: Union[Callable, str] = None
    use_prompt: str = None
    description: str = ""
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    fewshot_config: dict = None
    # runtime configuration options
    num_fewshot: int = None
    # scoring options
    metric_list: list = None
    output_type: str = "generate_until"
    generation_kwargs: dict = None
    repeats: int = 1
    filter_list: Union[str, list] = None
    should_decontaminate: bool = False
    doc_to_decontamination_query: str = None

    metadata: Union[str, list] = None  # by default, not used in the code. allows for users to pass arbitrary info to tasks

    embodied_eval_specific_kwargs: dict = None
    model_specific_generation_kwargs: dict = None
    model_specific_target_kwargs: dict = None

    def __post_init__(self) -> None:
        if self.dataset_path and os.path.exists(os.path.dirname(self.dataset_path)):
            import inspect
            from importlib import import_module

            # self.dataset_path = inspect.getfile(import_module(self.dataset_path))

        if self.group is not None:
            eval_logger.warning(
                "A task YAML file was found to contain a `group` key. Groups which provide aggregate scores over several subtasks now require a separate config file--if not aggregating, you may want to use the `tag` config option instead within your config. Setting `group` within a TaskConfig will be deprecated in v0.4.4. Please see https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md for more information."
            )

            if self.tag is None:
                self.tag = self.group
            else:
                raise ValueError("Got both a `group` and `tag` entry within a TaskConfig. Please use one or the other--`group` values will be deprecated in v0.4.4.")

        if self.generation_kwargs is not None:
            if "generate_until" not in self.output_type:
                eval_logger.warning(f"[{self.task}] passed `generation_kwargs`, but not using `output_type: generate_until`!")
                assert "generate_until" not in self.output_type

            if "temperature" in self.generation_kwargs:
                self.generation_kwargs["temperature"] = float(self.generation_kwargs["temperature"])

            if "until" not in self.generation_kwargs:
                self.generation_kwargs["until"] = [self.fewshot_delimiter]
        else:
            if "generate_until" in self.output_type:
                # ensure that we greedily generate in absence of explicit arguments otherwise
                self.generation_kwargs = {
                    "until": None if self.fewshot_delimiter is None else [self.fewshot_delimiter],
                    "do_sample": False,
                }

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self):
        """dumps the current config as a dictionary object, as a printable format.
        null fields will not be printed.
        Used for dumping results alongside full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        # TODO: should any default value in the TaskConfig not be printed?
        """
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif isinstance(v, Callable):
                # TODO: this should handle Promptsource template objects as a separate case?
                cfg_dict[k] = str(v)
        return cfg_dict

class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """

    VERSION = None

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    OUTPUT_TYPE: str = None

    def __init__(
            self,
            data_dir=None,
            cache_dir=None,
            download_mode=None,
            config=None,
    ) -> None:
        """
        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None
        self._instances = None

        self._config = TaskConfig({**config}) if config else TaskConfig()

        self._filters = [build_filter_ensemble("none", [["take_first", None]])]

    def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
        """Downloads and returns the task dataset.
        Override this method to download the dataset from a custom API.

        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        self.dataset_no_image = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        for doc_name in self.dataset_no_image:
            remove_cols = []
            features = self.dataset_no_image[doc_name].features
            # If it is an Image instance or a Sequence of Image instance. Remove it
            for feature in features:
                if isinstance(features[feature], Image):
                    remove_cols.append(feature)
                elif isinstance(features[feature], Sequence) and isinstance(features[feature].feature, Image):
                    remove_cols.append(feature)
            for remove_col in remove_cols:
                self.dataset_no_image[doc_name] = self.dataset_no_image[doc_name].remove_columns(remove_col)

    @property
    def config(self):
        """Returns the TaskConfig associated with this class."""
        return self._config

    @abc.abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass

    @abc.abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abc.abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    def training_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def fewshot_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        if self.has_training_docs():
            return self.training_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            if self.config.num_fewshot is not None:
                eval_logger.warning(
                    "has_training_docs and has_validation_docs are False" ", using test_docs as fewshot_docs but this is not recommended.")
            return self.test_docs()

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    @property
    def instances(self):
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc) -> None:
        print("Override doc_to_decontamination_query with document specific decontamination query.")
        assert False

    @abc.abstractmethod
    def doc_to_text(self, doc):
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc):
        pass

    # @profile
    def build_all_requests(
            self,
            *,
            limit: Union[int, None] = None,
            rank: int = 0,
            world_size: int = 1,
            cache_requests: bool = False,
            rewrite_requests_cache: bool = False,
            system_instruction: Optional[str] = None,
            apply_chat_template: bool = False,
            fewshot_as_multiturn: bool = False,
            chat_template: Optional[Callable] = None,
            tokenizer_name: str = "",
    ) -> None:
        """Build a set of Instances for a task, and store them in task.instances"""
        if self.has_test_docs():
            docs = self.test_docs()
            split = self.config.test_split
        elif self.has_validation_docs():
            docs = self.validation_docs()
            split = self.config.validation_split
        else:
            assert False, f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have valid or test docs!"

        # used with caching
        og_limit = limit

        cache_key = f"requests-{self._config.task}-{self.config.num_fewshot}shot-rank{rank}-world_size{world_size}"
        cache_key += "-chat_template" if apply_chat_template else ""
        cache_key += "-fewshot_as_multiturn" if fewshot_as_multiturn else ""
        cache_key += f"-system_prompt_hash{utils.hash_string(system_instruction)}" if system_instruction is not None else ""
        cache_key += f"-tokenizer{tokenizer_name}"

        cached_instances = load_from_cache(file_name=cache_key)