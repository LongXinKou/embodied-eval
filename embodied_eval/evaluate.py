import collections
import itertools
import json
import numpy as np
import os
import torch

from loguru import logger as eval_logger
from typing import List, Optional, Union
from tqdm import tqdm

from embodied_eval.utils import create_iterator

def SimpleEvaluate(
    model,
    eval_tasks,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: Optional[int] = 100000,
):
    results_dict = collections.defaultdict(dict)
    results = collections.defaultdict(dict)
    configs = collections.defaultdict(dict)
    samples = collections.defaultdict(list)

    RANK = model.rank
    WORLD_SIZE = model.world_size

    ### Postprocess outputs ###
    for task_output in eval_tasks:
        task = task_output.task
        # task.apply_filters() # TODO

        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)

        # iterate over different filters used
        doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE)
        doc_iterator_for_counting = create_iterator(range(len(task.eval_docs)), rank=RANK, limit=limit, world_size=WORLD_SIZE)
        num_docs = sum(1 for _ in doc_iterator_for_counting)

        pbar = tqdm(total=num_docs, desc=f"Postprocessing", disable=(RANK != 0))

        for doc_id, doc in doc_iterator:
            requests = instances_by_doc_id[doc_id]
            metrics = task.process_results(doc, [req.resps for req in requests])
            
            target = metrics.pop('target', None)
            example = {
                "doc_id": doc_id,
                "doc": requests[0].args[0],
                "target": target,
                "resps": [req.resps for req in requests],
            }
            task_output.logged_samples.append(example)

            for metric, value in metrics.items():
                task_output.sample_metrics[metric].append(value)
            pbar.update(1)
        pbar.close()

    if hasattr(model, "_model"):
        del model._model
        torch.cuda.empty_cache()

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            full_samples = [None] * WORLD_SIZE if RANK == 0 else None
            per_rank_samples = []
            for sample in task_output.logged_samples:
                per_rank_samples.append(sample)
            torch.distributed.gather_object(
                    obj=per_rank_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )
            
            if RANK == 0:
                task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))

            eval_logger.info(f"Gathering sample across all ranks for: {task_output.task_name}")
            
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))
            
            eval_logger.info(f"Gathering results across all ranks for: {task_output.task_name}")

        torch.distributed.barrier()  # Ensure all processes are synced before proceeding

    if RANK == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            result = task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
            results[task_output.task_name] = result
            configs[task_output.task_name] = task_output.task_config
            samples[task_output.task_name] = task_output.logged_samples
        
        results_dict["results"] = dict(results)
        results_dict["samples"] = dict(samples)
        results_dict["configs"] = dict(configs)
        

        eval_logger.info(f"Aggregating results across on ranks {RANK}")
    else:
        results_dict = None

        eval_logger.info(f"Pass on ranks {RANK}")
    
    if hasattr(model, "accelerator"):
        model.accelerator.wait_for_everyone()
    
    return results_dict

def SaveResult(
    results_dict,
    output_path,
    datetime_str
):
    def handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)
    
    tasks_samples = results_dict.pop('samples')
    tasks_results = results_dict.pop('results')
    tasks_configs = results_dict.pop('configs')

    for task_name, _ in tasks_results.items():
        samples = tasks_samples[task_name]
        results = tasks_results[task_name]
        configs = tasks_configs[task_name]

        if output_path:
            os.makedirs(os.path.join(output_path, datetime_str), exist_ok=True)
            eval_logger.info(f"Saving per-sample results for: {task_name}")

            file_results_samples = os.path.join(output_path, datetime_str, f"samples_{task_name}.json")
            for sample in samples:
                sample_dump = (
                    json.dumps(
                        sample,
                        default=handle_non_serializable,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                with open(file_results_samples, "a", encoding="utf-8") as f:
                    f.write(sample_dump)
            
            file_results_aggregated = os.path.join(output_path, datetime_str, f"results_{task_name}.json")
            result_dumped = json.dumps(results, indent=4, default=handle_non_serializable)
            with open(file_results_aggregated, "a", encoding="utf-8") as f:
                f.write(result_dumped)
            
            file_configs_aggregated = os.path.join(output_path, datetime_str, f"configs_{task_name}.json")
            result_dumped = json.dumps(configs, indent=4, default=handle_non_serializable)
            with open(file_configs_aggregated, "a", encoding="utf-8") as f:
                f.write(result_dumped)

        else:
            eval_logger.info("Output path not provided, skipping saving results")


    
