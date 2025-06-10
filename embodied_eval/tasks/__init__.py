import os
import collections

from typing import Dict, List, Mapping, Optional, Union

from embodied_eval.utils import load_yaml_config

class TaskManager:
    def __init__(
            self,
            model_name: Optional[str] = None,
            tasks: Optional[List[str]] = None,
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

    