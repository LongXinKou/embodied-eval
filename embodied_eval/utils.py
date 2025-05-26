import collections
import datetime
import fnmatch
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import gc
from itertools import islice

import numpy as np
import pytz
import torch
import transformers
from jinja2 import BaseLoader, Environment, StrictUndefined
from loguru import logger as eval_logger

SPACING = " " * 47
HIGHER_IS_BETTER_SYMBOLS = {
    True: "↑",
    False: "↓",
}

def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()

def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]}
    return args_dict

# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    if type(patterns) == str:
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        try:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        except Exception as e:
            eval_logger.error(f"Error matching pattern {pattern}: {e}")
    return sorted(list(task_names))


def sanitize_model_name(model_name: str, full_path: bool = False) -> str:
    """
    Given the model name, returns a sanitized version of it.
    """
    if full_path:
        return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_name)
    else:
        parts = model_name.split("/")
        last_two = "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]  # accommondate for models that are in Hugging Face Hub format like lmms-lab/llava-onevision-qwen2-0.5b
        return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", last_two)

def sanitize_task_name(task_name: str) -> str:
    """
    Given the task name, returns a sanitized version of it.
    """
    return re.sub(r"\W", "_", task_name)

def ignore_constructor(loader, node):
    return node

def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)

    *module_name, function_name = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)
    module_path = os.path.normpath(os.path.join(yaml_path, "{}.py".format(module_name)))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function

def load_yaml_config(yaml_path=None, yaml_config=None, yaml_dir=None, mode="full"):
    if mode == "simple":
        constructor_fn = ignore_constructor
    elif mode == "full":
        constructor_fn = import_function

    # Add the import_function constructor to the YAML loader
    yaml.add_constructor("!function", constructor_fn)
    if yaml_config is None:
        with open(yaml_path, "rb") as file:
            yaml_config = yaml.full_load(file)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    assert yaml_dir is not None
    assert yaml_config is not None

    if "include" in yaml_config:
        include_path = yaml_config["include"]
        del yaml_config["include"]

        if isinstance(include_path, str):
            include_path = [include_path]

        # Load from the last one first
        include_path.reverse()
        final_yaml_config = {}
        for path in include_path:
            # Assumes that path is a full path.
            # If not found, assume the included yaml
            # is in the same dir as the original yaml
            if not os.path.isfile(path):
                path = os.path.join(yaml_dir, path)

            try:
                included_yaml_config = load_yaml_config(yaml_path=path, mode=mode)
                final_yaml_config.update(included_yaml_config)
            except Exception as ex:
                # If failed to load, ignore
                raise ex

        final_yaml_config.update(yaml_config)
        return final_yaml_config
    return yaml_config