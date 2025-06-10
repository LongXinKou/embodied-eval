import hashlib
import os
import pickle

import dill

from embodied_eval_old.loggers.utils import _handle_non_serializable, is_serializable
from embodied_eval_old.utils import eval_logger

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("Embodied_Eval_CACHE_PATH")


PATH = OVERRIDE_PATH if OVERRIDE_PATH else f"{MODULE_DIR}/.cache"

# This should be sufficient for uniqueness
HASH_INPUT = "embodied-evaluation"

HASH_PREFIX = hashlib.sha256(HASH_INPUT.encode("utf-8")).hexdigest()

FILE_SUFFIX = f".{HASH_PREFIX}.pickle"

def load_from_cache(file_name):
    try:
        path = f"{PATH}/{file_name}{FILE_SUFFIX}"

        with open(path, "rb") as file:
            cached_task_dict = dill.loads(file.read())
            return cached_task_dict

    except Exception:
        eval_logger.debug(f"{file_name} is not cached, generating...")
        pass