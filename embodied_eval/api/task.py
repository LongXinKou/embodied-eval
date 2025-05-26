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

    lmms_eval_specific_kwargs: dict = None
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