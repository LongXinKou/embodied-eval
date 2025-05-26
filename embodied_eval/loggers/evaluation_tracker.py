# code is adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/loggers/evaluation_tracker.py
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

from embodied_eval.utils import (
    eval_logger,
    hash_string,
    sanitize_model_name,
)

@dataclass(init=False)
class GeneralConfigTracker:
    """
    Tracker for the evaluation parameters.

    Attributes:
        model_source (str): Source of the model (e.g. Hugging Face, GGUF, etc.)
        model_name (str): Name of the model.
        model_name_sanitized (str): Sanitized model name for directory creation.
        start_time (float): Start time of the experiment. Logged at class init.
        end_time (float): Start time of the experiment. Logged when calling [`GeneralConfigTracker.log_end_time`]
        total_evaluation_time_seconds (str): Inferred total evaluation time in seconds (from the start and end times).
    """
    model_source: str = None
    model_name: str = None
    model_name_sanitized: str = None
    system_instruction: str = None
    system_instruction_sha: str = None
    fewshot_as_multiturn: bool = None
    chat_template: str = None
    chat_template_sha: str = None
    start_time: float = None
    end_time: float = None
    total_evaluation_time_seconds: str = None

    def __init__(self) -> None:
        """Starts the evaluation timer."""
        self.start_time = time.perf_counter()

    @staticmethod
    def _get_model_name(model_args: str) -> str:
        """Extracts the model name from the model arguments."""

        def extract_model_name(model_args: str, key: str) -> str:
            """Extracts the model name from the model arguments using a key."""
            args_after_key = model_args.split(key)[1]
            return args_after_key.split(",")[0]

        # order does matter, e.g. peft and delta are provided together with pretrained
        prefixes = ["peft=", "delta=", "pretrained=", "model=", "path=", "engine="]
        for prefix in prefixes:
            if prefix in model_args:
                return extract_model_name(model_args, prefix)
        return ""

    def log_experiment_args(
            self,
            model_source: str,
            model_args: str,
            system_instruction: str,
            chat_template: str,
            fewshot_as_multiturn: bool,
    ) -> None:
        """Logs model parameters and job ID."""
        self.model_source = model_source
        self.model_name = GeneralConfigTracker._get_model_name(model_args)
        self.model_name_sanitized = sanitize_model_name(self.model_name)
        self.system_instruction = system_instruction
        self.system_instruction_sha = hash_string(system_instruction) if system_instruction else None
        self.chat_template = chat_template
        self.chat_template_sha = hash_string(chat_template) if chat_template else None
        self.fewshot_as_multiturn = fewshot_as_multiturn

    def log_end_time(self) -> None:
        """Logs the end time of the evaluation and calculates the total evaluation time."""
        self.end_time = time.perf_counter()
        self.total_evaluation_time_seconds = str(self.end_time - self.start_time)

class EvaluationTracker:
    """
    Keeps track and saves relevant information of the evaluation process.
    Compiles the data from trackers and writes it to files, which can be published to the Hugging Face hub if requested.
    """

    def __init__(
        self,
        output_path: str = None,
        hub_results_org: str = "",
        hub_repo_name: str = "",
        details_repo_name: str = "",
        results_repo_name: str = "",
        push_results_to_hub: bool = False,
        push_samples_to_hub: bool = False,
        public_repo: bool = False,
        token: str = "",
        leaderboard_url: str = "",
        point_of_contact: str = "",
        gated: bool = False,
    ) -> None:
        """
        Creates all the necessary loggers for evaluation tracking.

        Args:
            output_path (str): Path to save the results. If not provided, the results won't be saved.
            hub_results_org (str): The Hugging Face organization to push the results to. If not provided, the results will be pushed to the owner of the Hugging Face token.
            hub_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will be pushed to `lm-eval-results`.
            details_repo_name (str): The name of the Hugging Face repository to push the details to. If not provided, the results will be pushed to `lm-eval-results`.
            result_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will not be pushed and will be found in the details_hub_repo.
            push_results_to_hub (bool): Whether to push the results to the Hugging Face hub.
            push_samples_to_hub (bool): Whether to push the samples to the Hugging Face hub.
            public_repo (bool): Whether to push the results to a public or private repository.
            token (str): Token to use when pushing to the Hugging Face hub. This token should have write access to `hub_results_org`.
            leaderboard_url (str): URL to the leaderboard on the Hugging Face hub on the dataset card.
            point_of_contact (str): Contact information on the Hugging Face hub dataset card.
            gated (bool): Whether to gate the repository.
        """
        self.general_config_tracker = GeneralConfigTracker()

        self.output_path = output_path
        self.push_results_to_hub = push_results_to_hub
        self.push_samples_to_hub = push_samples_to_hub
        self.public_repo = public_repo
        self.leaderboard_url = leaderboard_url
        self.point_of_contact = point_of_contact
        self.api = HfApi(token=token) if token else None
        self.gated_repo = gated

        if not self.api and (push_results_to_hub or push_samples_to_hub):
            raise ValueError(
                "Hugging Face token is not defined, but 'push_results_to_hub' or 'push_samples_to_hub' is set to True. " "Please provide a valid Hugging Face token by setting the HF_TOKEN environment variable.")
        if self.api and hub_results_org == "" and (push_results_to_hub or push_samples_to_hub):
            hub_results_org = self.api.whoami()["name"]
            eval_logger.warning(f"hub_results_org was not specified. Results will be pushed to '{hub_results_org}'.")

