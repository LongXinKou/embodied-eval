import collections
import itertools
import json
import random
import math
import os
import numpy as np
import torch

from loguru import logger as eval_logger
from tqdm import tqdm

from embodied_eval.tasks import TaskManager, get_task_dict, get_task_list
from embodied_eval.models import get_model
from embodied_eval.utils import create_iterator, get_datetime_str


class EQAEvaluator:
    def __init__(self, config):
        self._config = config

        # ========== Initialize TaskManager ==========
        self.task_manager = TaskManager()
        self.task_list = self.config.tasks.split(",")
        self.task_names = self.task_manager.match_tasks(self.task_list)
        eval_logger.info(f"Selected Tasks: {self.task_names}")
        self.task_dict = get_task_dict(self.task_list, self.task_manager)

        # ========== Initialize Model ==========
        model_name = self.config.model
        model_args = "" if self.config.model_args is None else self.config.model_args
        self.model = get_model(model_name=model_name).create_from_arg_string(
            model_args,
            additional_config={
            }
        )

        # Set task_dict for model
        for task_name, task_obj in self.task_dict.items():
            self.model.task_dict[task_name] = task_obj.dataset

        # ========== Initialize Config ==========

    @property
    def config(self):
        return self._config

    def inference(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        eval_logger.info(
            f"Setting random seed to {self.config.seed}  \
            Setting numpy seed to {self.config.seed}  \
            Setting torch manual seed to {self.config.seed}s"
         )
        requests = collections.defaultdict(list)
        padding_requests = collections.defaultdict(int)

        eval_tasks = get_task_list(self.task_dict)
        for task_output in eval_tasks:
            task = task_output.task
            task_name = task_output.task_name

            task.build_all_requests(
                rank=self.model.rank,
                world_size=self.model.world_size,
            )
            eval_logger.debug(f"Task: {task_name}; number of requests on this rank: {len(task._instances)}")

            for instance in task.instances:
                reqtype = instance.request_type
                requests[reqtype].append(instance)

                # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            if self.model.world_size > 1:
                instances_rnk = torch.tensor(len(task._instances), device=self.model.device)
                gathered_item = self.model.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
                numpad = max(gathered_item) - gathered_item[self.model.rank]
                padding_requests[task.output_type] += numpad

        for reqtype, reqs in requests.items():
            eval_logger.info(f"Running {reqtype} requests")
            cloned_reqs = []
            for req in reqs:
                cloned_reqs.extend([req] * req.repeats)

            # (FSDP/DDP require even batches among ranks)
            if (self.model.world_size > 1) and (padding_requests[reqtype] > 0):
                for _ in range(padding_requests[reqtype]):
                    cloned_reqs.extend([req] * req.repeats)

            resps = self.single_inference(cloned_reqs)

            # put responses from model into a list of length K for each request.
            for x, req in zip(resps, cloned_reqs):
                req.resps.append(x)

            if self.model.world_size > 1:
                self.model.accelerator.wait_for_everyone()

        return eval_tasks
    
    def single_inference(self, requests):
        """
        Generate responses for a list of requests.
        
        Args:
            requests: List of Instance objects
            
        Returns:
            List[str]: List of generated responses
        """
        responses = []

        def process_request(request) -> str:
            """
            Process a single request and return the response.
            
            Args:
                request: Instance object containing request information
                
            Returns:
                str: Generated response text
            """
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            gen_kwargs = gen_kwargs if gen_kwargs else {}
            
            # Get visuals for this request
            if split is not None:
                visuals = doc_to_visual(self.task_dict[task].dataset[split][doc_id])
                if isinstance(visuals, tuple):  # ([visual], [visual_index])
                    visual_list = visuals[0]
                else:
                    visual_list = visuals  # [visual]
            else:
                visuals = doc_to_visual(self.task_dict.dataset[task][doc_id])
                if isinstance(visuals, dict):
                    visual_list = visuals[0]
                else:
                    visual_list = visuals
            
            return context, visual_list, gen_kwargs
        
        eval_logger.info(f"Processing {len(requests)} requests")
        progress_bar = tqdm(total=len(requests), disable=(self.model.rank != 0), desc="Model Responding")
        for request in requests:
            try:
                context, visual_list, gen_kwargs = process_request(request)
                response = self.model.respond(context, visual_list, **gen_kwargs).strip()
                responses.append(response)
            except Exception as e:
                eval_logger.error(f"Error processing request: {e}")
                responses.append("")  # Return empty string on error
            progress_bar.update(1)
        
        progress_bar.close()
        return responses


    def evaluate(self, eval_tasks):
        results_dict = collections.defaultdict(dict)
        results = collections.defaultdict(dict)
        configs = collections.defaultdict(dict)
        samples = collections.defaultdict(list)

        RANK = self.model.rank
        WORLD_SIZE = self.model.world_size

        for task_output in eval_tasks:
            task = task_output.task

            # Pre-process task.instances to group by doc_id
            instances_by_doc_id = collections.defaultdict(list)
            for instance in task.instances:
                instances_by_doc_id[instance.doc_id].append(instance)
            for instances in instances_by_doc_id.values():
                instances.sort(key=lambda x: x.idx)

            # iterate over different filters used
            doc_iterator = task.doc_iterator(rank=RANK,world_size=WORLD_SIZE)
            doc_iterator_for_counting = create_iterator(range(len(task.eval_docs)), rank=RANK, world_size=WORLD_SIZE)
            num_docs = sum(1 for _ in doc_iterator_for_counting)
            pbar = tqdm(total=num_docs, desc=f"Postprocessing", disable=(RANK != 0))
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                sample_result = task.process_results(doc, [req.resps for req in requests])
                example = {
                    "doc_id": doc_id,
                    "doc": requests[0].args[0],
                    "resps": [req.resps for req in requests],
                }
                example.update({key: value for key, value in sample_result.items()})
                task_output.logged_samples.append(example)

                pbar.update(1)
            pbar.close()

        if hasattr(self.model, "_model"):
            del self.model._model
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
            torch.distributed.barrier()  # Ensure all processes are synced before proceeding

        if RANK == 0:
            ### Aggregate results over all datapoints ###
            # aggregate results ; run bootstrap CIs
            for task_output in eval_tasks:
                agg_result = task_output.calculate_aggregate_metric()
                results[task_output.task_name] = agg_result
                configs[task_output.task_name] = task_output.task_config
                samples[task_output.task_name] = task_output.logged_samples

            results_dict["results"] = dict(results)
            results_dict["samples"] = dict(samples)
            results_dict["configs"] = dict(configs)

            eval_logger.info(f"Aggregating results across on ranks {RANK}")
        else:
            results_dict = None

            eval_logger.info(f"Pass on ranks {RANK}")

        if hasattr(self.model, "accelerator"):
            self.model.accelerator.wait_for_everyone()

        return results_dict

    def print_results(self, results_dict):
        def make_table(result):
            from pytablewriter import MarkdownTableWriter
            import re

            model = self.config.model

            type_to_metrics = {}
            for key, val in result.items():
                if key == "overall" or key.endswith("_stderr") or key.endswith("_average"):
                    continue
                qtype, metric_name = key.rsplit("_", 1)
                if qtype not in type_to_metrics:
                    type_to_metrics[qtype] = {}
                type_to_metrics[qtype][metric_name] = val

            all_metrics = sorted({m for v in type_to_metrics.values() for m in v})
            headers = ["Metric"] + list(type_to_metrics.keys()) + ["Average"]
            value_matrix = []

            for metric in all_metrics:
                row = [metric]
                for qtype in type_to_metrics:
                    val = type_to_metrics[qtype].get(metric, "")
                    row.append(f"{val:.4f}" if val != "" else "")
                avg_key = f"{metric}_average"
                avg_val = result.get(avg_key, "")
                row.append(f"{avg_val:.4f}" if avg_val != "" else "")
                value_matrix.append(row)

            # Add the metric averages to the original result dictionary
            md_writer = MarkdownTableWriter()
            md_writer.table_name = f"Results for {model}"
            md_writer.headers = headers
            md_writer.value_matrix = value_matrix

            eval_logger.info("Markdown Table:\n")
            eval_logger.info(f"Overall: {result['overall']}")
            md_writer.write_table()

        results = results_dict["results"]
        configs = results_dict["configs"]
        for task_name, _ in results.items():
            result, config = results[task_name], configs[task_name]
            info = f"task:{config['task']}, data:{config['dataset_path']}/{config.get('dataset_kwargs')},\
                            generate:{config['generation_kwargs']}"
            eval_logger.info(info)
            make_table(result)

    def save_results(self, results_dict):
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
        datetime_str = get_datetime_str(timezone=self.config.timezone)

        for task_name, _ in tasks_results.items():
            samples = tasks_samples[task_name]
            results = tasks_results[task_name]
            configs = tasks_configs[task_name]

            if self.config.output_path:
                os.makedirs(os.path.join(self.config.output_path, datetime_str), exist_ok=True)
                eval_logger.info(f"Saving per-sample results for: {task_name}")

                file_results_samples = os.path.join(self.config.output_path, datetime_str, f"samples_{task_name}.json")
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

                file_results_aggregated = os.path.join(self.config.output_path, datetime_str, f"results_{task_name}.json")
                result_dumped = json.dumps(results, indent=4, default=handle_non_serializable)
                with open(file_results_aggregated, "a", encoding="utf-8") as f:
                    f.write(result_dumped)

                file_configs_aggregated = os.path.join(self.config.output_path, datetime_str, f"configs_{task_name}.json")
                result_dumped = json.dumps(configs, indent=4, default=handle_non_serializable)
                with open(file_configs_aggregated, "a", encoding="utf-8") as f:
                    f.write(result_dumped)

            else:
                eval_logger.info("Output path not provided, skipping saving results")
