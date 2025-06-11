import random

import numpy as np
import torch

from loguru import logger as eval_logger
from tqdm import tqdm
from typing import List, Optional, Union

from embodied_eval.tasks import TaskManager, get_task_dict
from embodied_eval.utils import (
    positional_deprecated
)

@positional_deprecated
def SimpleInference(
    model,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    task_manager: Optional[TaskManager] = None,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_logger.info(f"Setting random seed to {seed} | \
        Setting numpy seed to {seed} | \
        Setting torch manual seed to {seed}s"
    )

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."
    task_dict = get_task_dict(tasks, task_manager)

    result = Inference(
        model=model,
        task_dict=task_dict,
    )

    return result

@positional_deprecated
def Inference(
    model,
    task_dict,
):

