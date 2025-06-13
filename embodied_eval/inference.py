import random
import math
import collections
import numpy as np
import torch

from loguru import logger as eval_logger
from typing import List, Optional, Union

from embodied_eval.tasks import TaskManager, get_task_dict, get_task_list
from embodied_eval.utils import (
    positional_deprecated
)

@positional_deprecated
def SimpleInference(
    model,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    task_manager: Optional[TaskManager] = None,
    limit: Optional[Union[int, float]] = None,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_logger.info(f"Setting random seed to {seed}  \
        Setting numpy seed to {seed}  \
        Setting torch manual seed to {seed}s"
    )

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."
    task_dict = get_task_dict(tasks, task_manager)
    for task_name, task_obj in task_dict.items():
        model.task_dict[task_name] = task_obj.dataset

    eval_tasks = Inference(
        model=model,
        task_dict=task_dict,
        limit=limit,
    )

    return eval_tasks

@positional_deprecated
def Inference(
    model,
    task_dict,
    limit: Optional[Union[int, float]] = None,
):
    requests = collections.defaultdict(list)
    padding_requests = collections.defaultdict(int)

    eval_tasks = get_task_list(task_dict)

    for task_output in eval_tasks:
        task = task_output.task
        task_name = task_output.task_name

        if limit is not None:
            limit = int(math.ceil(len(task.eval_docs) * limit)) if limit < 1.0 else int(limit)
        task.build_all_requests(
            limit=limit,
            rank=model.rank,
            world_size=model.world_size,
        )
        eval_logger.debug(f"Task: {task_name}; number of requests on this rank: {len(task._instances)}")
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
        if model.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=model.device)
            gathered_item = model.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            numpad = max(gathered_item) - gathered_item[model.rank]
            padding_requests[task.output_type] += numpad

    ### Run LMM on inputs, get all outputs ###
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        # (FSDP/DDP require even batches among ranks)
        if (model.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)
                
        # Run requests
        resps = getattr(model, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if model.world_size > 1:
            model.accelerator.wait_for_everyone()

    return eval_tasks