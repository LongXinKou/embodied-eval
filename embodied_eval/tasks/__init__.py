import abc
import os
import copy
import collections

import random
import math

from datasets import Dataset, load_dataset, load_from_disk, DownloadMode
from collections.abc import Callable
from typing import Dict, List, Mapping, Optional, Union
from dataclasses import asdict, dataclass, field
from loguru import logger as eval_logger
from tqdm import tqdm

import yaml

from embodied_eval.common.instance import Instance
from embodied_eval.utils import load_yaml_config, load_json, pattern_match, create_iterator

# ======================= Task Config =======================
@dataclass
class TaskConfig(dict):
    # task naming/registry
    task: str = None
    # HF dataset options: which dataset to use, and what splits for what purpose
    dataset_path: str = None
    dataset_name: str = None
    load_from_disk: bool = False
    eval_split: str = None
    # Formatting
    doc_to_visual: Union[Callable, str] = None
    doc_to_text: Union[Callable, str] = None
    doc_to_target: Union[Callable, str] = None
    doc_to_choice: Union[Callable, str, dict, list] = None
    # Inference
    output_type: str = "generate_until"
    generation_kwargs: dict = None
    repeats: int = 1

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
        self.output_type = getattr(self.config, 'output_type')

        self.prepare_dataset()
        self._instances = None

        # TODO MC

    @property
    def config(self):
        """Returns the TaskConfig associated with this class."""
        return self._config

    @property
    def instances(self):
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances

    @property
    def eval_docs(self) -> Dataset:
        if self.config.eval_split is not None:
            return self.dataset[self.config.eval_split]
        else:
            assert False, f"Task dataset (path={self.dataset_path}, name={self.dataset_name}) must have eval docs!"

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

    def build_all_requests(
        self,
        limit: Union[int, None] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Build a set of Instances for a task, and store them in task.instances"""
        eval_logger.info(f"Building contexts for {self.config.task} on rank {rank}...")

        instances = []

        doc_id_docs = create_iterator(enumerate(self.eval_docs), rank=rank, limit=limit, world_size=world_size)
        for doc_id, doc in tqdm(
            doc_id_docs,
            total=len(self.eval_docs),
        ):
            per_task_metadata = {"task": self.config["task"], "doc_id": doc_id, "repeats": self.config.repeats, "split": self.config["eval_split"]}
            inst = self.construct_requests(doc_id=doc_id, metadata=per_task_metadata)
            if not isinstance(inst, list):
                inst = [inst]

            instances.append(inst)

        sliced_instances = instances[:limit]
        flattened_instances = [instance for instance_group in sliced_instances for instance in instance_group]
        self._instances = flattened_instances


    def construct_requests(self, doc_id: int, **kwargs) -> Union[List[Instance], Instance]:
        split = kwargs.get("metadata").get("split")
        # TODO MC
        if self.output_type == "generate_until":
            arguments = (copy.deepcopy(self.config.generation_kwargs), self.doc_to_visual, doc_id,
                         self.config.task, split)

        return Instance(request_type=self.output_type, arguments=arguments, idx=0, **kwargs)


# ======================= Task Output =======================
class TaskOutput:
    def __init__(
        self,
        task=None,
        task_name=None,
    ):
        self.task = task
        self.task_name = task_name

    @classmethod
    def from_taskdict(cls, task_name: str, task):
        return cls(
            task=task,
            task_name=task_name
        )

# ======================= Task Manager =======================

class TaskManager:
    def __init__(
            self,
    ) -> None:
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

    def match_tasks(self, task_list):
        return pattern_match(task_list, self.all_tasks)

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

def get_task_list(task_dict: dict):
    outputs = []
    for task_name, task_obj in task_dict.items():
        task_output = TaskOutput.from_taskdict(task_name, task_obj)
        outputs.append(task_output)

    return outputs