import os

from typing import Dict, List, Mapping, Optional, Union

class TaskManager:
    def __init__(
            self,
            model_name: Optional[str] = None,
            tasks: Optional[List[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.tasks = tasks
        # TODO 评估单一task

    def initialize_tasks(
            self,
            include_defaults: bool = True,
    ):
        if include_defaults:
            task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        # TODO 额外的task dir

    def _get_tasks(self, task_dir: str):
        for root, dirs, file_list in os.walk(task_dir):
            for f in file_list:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
