import collections
import itertools
import torch


from typing import List, Optional, Union
from tqdm import tqdm

def SimpleEvaluate(
    model,
    eval_tasks,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: Optional[int] = 100000,
):
    RANK = model.rank
    WORLD_SIZE = model.world_size

    ### Postprocess outputs ###
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()

        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)

        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE)
            pbar = tqdm(total=len(task.eval_docs), desc=f"Postprocessing", disable=(RANK != 0))

            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(doc, [req.filtered_resps[filter_key] for req in requests])

                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)
                pbar.update(1)
            pbar.close()

    if hasattr(model, "_model"):
        del model._model
        torch.cuda.empty_cache()

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))

        torch.distributed.barrier()  # Ensure all processes are synced before proceeding

    if RANK == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)