import abc
import os
import collections

import random
import math

from datasets import Dataset, load_dataset, load_from_disk, DownloadMode
from collections.abc import Callable
from typing import Dict, List, Mapping, Optional, Union
from dataclasses import asdict, dataclass, field

import yaml

from embodied_eval.utils import load_yaml_config, load_json

# ======================= Task Config =======================
@dataclass
class TaskConfig(dict):
    # task naming/registry
    task: str = None
    # HF dataset options: which dataset to use, and what splits for what purpose
    dataset_path: str = None
    dataset_name: str = None
    load_from_disk: bool = False
    test_split: str = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self):
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif isinstance(v, Callable):
                cfg_dict[k] = str(v)
        return cfg_dict

# ======================= Task Object =======================
class Task(abc.ABC):
    """
    A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods.
    """

    def __init__(
            self,
            config: Optional[dict] = None,
            model_name: Optional[str] = None,
    ) -> None:
        self._config = TaskConfig(**config)
        if self.config is None:
            raise ValueError("Must pass a config to Task")

        self.dataset_path = getattr(self.config, 'dataset_path', None)
        self.dataset_name = getattr(self.config, 'dataset_name', None)
        self.load_from_disk = getattr(self.config, 'load_from_disk', False)

        # TODO model_name / metric config

        self.prepare_dataset()
        self.task_docs = self.test_docs()
        self.features = list(self.task_docs.features.keys())

        # TODO MC

    @property
    def config(self):
        """Returns the TaskConfig associated with this class."""
        return self._config

    def test_docs(self) -> Dataset:
        if self.config.test_split is not None:
            return self.dataset[self.config.test_split]
        else:
            assert False, f"Task dataset (path={self.dataset_path}, name={self.dataset_name}) must have test docs!"

    def prepare_dataset(self):
        if not self.load_from_disk:
            self.dataset = load_dataset(
                path=self.dataset_path,
                name=self.dataset_name,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
        else:
            if self.dataset_path.endswith(".json"):
                self.dataset = load_json(self.dataset_path)
            elif self.dataset_path.endswith(".yaml"):
                self.dataset = []
                with open (self.dataset_path, "r") as f:
                    yaml_data = yaml.safe_load(f)
                    datasets = yaml_data.get("datasets", [])
                    # datasts:
                    #    - json_path: xxx1.json
                    #      sampling_strategy: first/end/random:1000
                    for dataset in datasets:
                        json_path = dataset.get("json_path")
                        sampling_strategy = dataset.get("sampling_strategy")
                        sampling_number = None

                        cur_dataset = load_json(json_path)
                        if ":" in sampling_strategy:
                            sampling_strategy, sampling_number = sampling_strategy.split(":")
                            if "%" in sampling_number:
                                sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_dataset) / 100)
                            else:
                                sampling_number = int(sampling_number)
                        if sampling_strategy == "first" and sampling_number is not None:
                            cur_dataset = cur_dataset[:sampling_number]
                        elif sampling_strategy == "end" and sampling_number is not None:
                            cur_dataset = cur_dataset[-sampling_number:]
                        elif sampling_strategy == "random" and sampling_number is not None:
                            random.shuffle(cur_dataset)
                            cur_dataset = cur_dataset[:sampling_number]

                        self.dataset.extend(cur_dataset)
                self.dataset = Dataset.from_list(self.dataset)
            else:
                self.dataset = load_from_disk(dataset_path=self.dataset_path)


# ======================= Task Manager =======================

class TaskManager:
    def __init__(
            self,
            model_name: Optional[str] = None,
    ) -> None:
        self.model_name = model_name

        self. _task_index = self.initialize_tasks(include_defaults=True)
        self._all_tasks = sorted(list(self._task_index.keys()))
        # TODO 评估单一task

    @property
    def all_tasks(self):
        return self._all_tasks

    @property
    def task_index(self):
        return self._task_index

    def initialize_tasks(
            self,
            include_defaults: bool = True,
    ):
        if include_defaults:
            task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        # TODO 额外的task dir
        tasks = self._get_tasks(task_dir=task_dir)
        return tasks

    def _get_tasks(self, task_dir: str):
        tasks = collections.defaultdict()
        for root, dirs, file_list in os.walk(task_dir):
            for f in file_list:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
                    config = load_yaml_config(yaml_path)

                tasks[config["task"]] = {
                    "type": "task",
                    "yaml_path": yaml_path,
                }
        return tasks

    def _get_config(self, name):
        if name not in self.task_index:
            raise ValueError("Task {} not found".format(name))
        yaml_path = self.task_index[name]["yaml_path"]
        return load_yaml_config(yaml_path)

    def _load_individual_task(
            self,
            name: Optional[str] = None,
    ):
        def _load_task(config, task):
            task_object = Task(config=config, model_name=self.model_name)
            return {task: task_object}

        if isinstance(name, str):
            task_config = self._get_config(name)
            return _load_task(task_config, task=name)

    def load_tasks(self, task_list):
        if isinstance(task_list, str):
            task_list = [task_list]
        all_loaded_tasks = dict(collections.ChainMap(*map(self._load_individual_task, task_list)))
        return all_loaded_tasks

def get_task_dict(
    task_name_list: Union[str, List[Union[str, Dict, Task]]],
    task_manager: Optional[TaskManager] = None,
):
    if isinstance(task_name_list, str):
        task_name_list = [task_name_list]
    string_task_name_list = [task for task in task_name_list if isinstance(task, str)]

    if len(string_task_name_list) > 0:
        if task_manager is None:
            task_manager = TaskManager()

        task_name_from_string_dict = task_manager.load_tasks(
            string_task_name_list,
        )

    final_task_dict = {
        **task_name_from_string_dict,
    }

    return final_task_dict