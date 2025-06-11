import random

import numpy as np
import torch

from loguru import logger as eval_logger
from tqdm import tqdm
from typing import List, Optional, Union

from embodied_eval.utils import (
    positional_deprecated
)

@positional_deprecated
def SimpleInference(
    model,
    model_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    random_seed: int = 0,
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    eval_logger.info(f"Setting random seed to {random_seed} | \
        Setting numpy seed to {random_seed} | \
        Setting torch manual seed to {random_seed}s"
    )

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."

    if model_args is None:
        model_args = ""
