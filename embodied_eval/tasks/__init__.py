# credit to https://github.com/EleutherAI/lm-evaluation-harness
import collections
import inspect
import logging
import os
from functools import partial
from typing import Dict, List, Mapping, Optional, Union

from loguru import logger as eval_logger

from embodied_eval import utils

from embodied_eval.api.task import ConfigurableTask, Task

class TaskManager:
    def __init__(
        self,
        verbosity="INFO",
        include_path: Optional[Union[str, List]] = None,
        include_defaults: bool = True,
        model_name: Optional[str] = None,
    ) -> None:
        self.verbosity = verbosity
        self.include_path = include_path
        self.logger = eval_logger
        self.model_name = model_name

        self._task_index = self.initialize_tasks(include_path=include_path, include_defaults=include_defaults)
        self._all_tasks = sorted(list(self._task_index.keys()))

        self._all_groups = sorted([x for x in self._all_tasks if self._task_index[x]["type"] == "group"])
        self._all_subtasks = sorted([x for x in self._all_tasks if self._task_index[x]["type"] == "task"])
        self._all_tags = sorted([x for x in self._all_tasks if self._task_index[x]["type"] == "tag"])

        self.task_group_map = collections.defaultdict(list)

    def initialize_tasks(
            self,
            include_path: Optional[Union[str, List]] = None,
            include_defaults: bool = True,
    ):
        """Creates a dictionary of tasks index.

        :param include_path: Union[str, List] = None
            An additional path to be searched for tasks recursively.
            Can provide more than one such path as a list.
        :param include_defaults: bool = True
            If set to false, default tasks (those in lmms_eval/tasks/) are not indexed.
        :return
            Dictionary of task names as key and task metadata
        """
        if include_defaults:
            all_paths = [os.path.dirname(os.path.abspath(__file__)) + "/"]
        else:
            all_paths = []
        if include_path is not None:
            if isinstance(include_path, str):
                include_path = [include_path]
            all_paths.extend(include_path)

        task_index = {}
        for task_dir in all_paths:
            tasks = self._get_task_and_group(task_dir)
            task_index = {**tasks, **task_index}

        return task_index

    @property
    def all_tasks(self):
        return self._all_tasks

    @property
    def all_groups(self):
        return self._all_groups

    @property
    def all_subtasks(self):
        return self._all_subtasks

    @property
    def all_tags(self):
        return self._all_tags

    @property
    def task_index(self):
        return self._task_index

    def list_all_tasks(self, list_groups=True, list_tags=True, list_subtasks=True) -> str:
        from pytablewriter import MarkdownTableWriter

        def sanitize_path(path):
            # don't print full path if we are within the embodied_eval/tasks dir !
            # if we aren't though, provide the full path.
            if "embodied_eval/tasks/" in path:
                return "embodied_eval/tasks/" + path.split("embodied_eval/tasks/")[-1]
            else:
                return path

        group_table = MarkdownTableWriter()
        group_table.headers = ["Group", "Config Location"]
        gt_values = []
        for g in self.all_groups:
            path = self.task_index[g]["yaml_path"]
            if path == -1:
                path = "---"
            else:
                path = sanitize_path(path)
            gt_values.append([g, path])
        group_table.value_matrix = gt_values

        tag_table = MarkdownTableWriter()
        tag_table.headers = ["Tag"]
        tag_table.value_matrix = [[t] for t in self.all_tags]

        subtask_table = MarkdownTableWriter()
        subtask_table.headers = ["Task", "Config Location", "Output Type"]
        st_values = []
        for t in self.all_subtasks:
            path = self.task_index[t]["yaml_path"]

            output_type = ""

            # read the yaml file to determine the output type
            if path != -1:
                config = utils.load_yaml_config(path, mode="simple")
                if "output_type" in config:
                    output_type = config["output_type"]
                elif "include" in config:  # if no output type, check if there is an include with an output type
                    include_path = path.split("/")[:-1] + config["include"]
                    include_config = utils.load_yaml_config(include_path, mode="simple")
                    if "output_type" in include_config:
                        output_type = include_config["output_type"]

            if path == -1:
                path = "---"
            else:
                path = sanitize_path(path)
            st_values.append([t, path, output_type])
        subtask_table.value_matrix = st_values

        result = "\n"
        if list_groups:
            result += group_table.dumps() + "\n\n"
        if list_tags:
            result += tag_table.dumps() + "\n\n"
        if list_subtasks:
            result += subtask_table.dumps() + "\n\n"
        return result

    def match_tasks(self, task_list):
        return utils.pattern_match(task_list, self.all_tasks)

    def _config_is_task(self, config) -> bool:
        if ("task" in config) and isinstance(config["task"], str):
            return True
        return False

    def _config_is_group(self, config) -> bool:
        if ("task" in config) and isinstance(config["task"], list):
            return True
        return False

    def _config_is_python_task(self, config) -> bool:
        if "class" in config:
            return True
        return False

    def _get_task_and_group(self, task_dir: str):
        """
        Creates a dictionary of tasks index with the following metadata,
        - `type`, that can be either `task`, `python_task`, `group` or `tags`.
            `task` refer to regular task configs, `python_task` are special
            yaml files that only consists of `task` and `class` parameters.
            `group` are group configs. `tags` are labels that can be assigned
            to tasks to assist in sorting and calling tasks of certain themes.
        - `yaml_path`, path to the yaml file. If the entry is a `group` that
            was configured through a task config, the yaml_path will be -1
            and all subtasks will be listed in `task` (see below)
        - `task`, reserved for entries with `type` as `group`. This will list
            all subtasks. When a group config is created (as opposed to task
            config having `group` parameter set), this will be set to -1 to
            avoid recursive indexing. The whole list of subtasks will be loaded
            at evaluation.

        :param task_dir: str
            A directory to check for tasks

        :return
            Dictionary of task names as key and task metadata
        """
        # TODO: remove group in next release
        print_info = True
        ignore_dirs = [
            "__pycache__",
            ".ipynb_checkpoints",
        ]
        tasks_and_groups = collections.defaultdict()
        for root, dirs, file_list in os.walk(task_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in file_list:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
                    config = utils.load_yaml_config(yaml_path, mode="simple")
                    if self._config_is_python_task(config):
                        # This is a python class config
                        tasks_and_groups[config["task"]] = {
                            "type": "python_task",
                            "yaml_path": yaml_path,
                        }
                    elif self._config_is_group(config):
                        # This is a group config
                        tasks_and_groups[config["group"]] = {
                            "type": "group",
                            "task": -1,  # This signals that
                            # we don't need to know
                            # the task list for indexing
                            # as it can be loaded
                            # when called.
                            "yaml_path": yaml_path,
                        }

                        # # Registered the level 1 tasks from a group config
                        # for config in config["task"]:
                        #     if isinstance(config, dict) and self._config_is_task(config):
                        #         task = config["task"]
                        #         tasks_and_groups[task] = {
                        #             "type": "task",
                        #             "yaml_path": yaml_path,
                        #             }

                    elif self._config_is_task(config):
                        # This is a task config
                        task = config["task"]
                        tasks_and_groups[task] = {
                            "type": "task",
                            "yaml_path": yaml_path,
                        }

                        # TODO: remove group in next release
                        for attr in ["tag", "group"]:
                            if attr in config:
                                if attr == "group" and print_info:
                                    self.logger.debug(
                                        "`group` and `group_alias` keys in tasks' configs will no longer be used in the next release of lmms-eval. "
                                        "`tag` will be used to allow to call a collection of tasks just like `group`. "
                                        "`group` will be removed in order to not cause confusion with the new ConfigurableGroup "
                                        "which will be the offical way to create groups with addition of group-wide configuations."
                                    )
                                    print_info = False
                                    # attr = "tag"

                                attr_list = config[attr]
                                if isinstance(attr_list, str):
                                    attr_list = [attr_list]

                                for tag in attr_list:
                                    if tag not in tasks_and_groups:
                                        tasks_and_groups[tag] = {
                                            "type": "tag",
                                            "task": [task],
                                            "yaml_path": -1,
                                        }
                                    elif tasks_and_groups[tag]["type"] != "tag":
                                        self.logger.warning(
                                            f"The tag {tag} is already registered as a group, this tag will not be registered. " "This may affect tasks you want to call.")
                                        break
                                    else:
                                        tasks_and_groups[tag]["task"].append(task)
                    else:
                        self.logger.debug(f"File {f} in {root} could not be loaded as a task or group")

        return tasks_and_groups

def _check_duplicates(task_dict: dict) -> List[str]:
    """helper function solely used in validating get_task_dict output.
    Takes the output of lmms_eval.evaluator_utils.get_subtask_list and
    returns a list of all leaf subtasks contained within, and errors if any such leaf subtasks are
    "oversubscribed" to several disjoint groups.
    """
    subtask_names = []
    for key, value in task_dict.items():
        subtask_names.extend(value)

    duplicate_tasks = {task_name for task_name in subtask_names if subtask_names.count(task_name) > 1}

    # locate the potentially problematic groups that seem to 'compete' for constituent subtasks
    competing_groups = [group for group in task_dict.keys() if len(set(task_dict[group]).intersection(duplicate_tasks)) > 0]

    if len(duplicate_tasks) > 0:
        raise ValueError(
            f"Found 1 or more tasks while trying to call get_task_dict() that were members of more than 1 called group: {list(duplicate_tasks)}. Offending groups: {competing_groups}. Please call groups which overlap their constituent tasks in separate evaluation runs."
        )

def get_task_dict(
    task_name_list: Union[str, List[Union[str, Dict, Task]]],
    task_manager: Optional[TaskManager] = None,
):
    """Creates a dictionary of task objects from either a name of task, config, or prepared Task object.

    :param task_name_list: List[Union[str, Dict, Task]]
        Name of model or LM object, see lmms_eval.models.get_model
    :param task_manager: TaskManager = None
        A TaskManager object that stores indexed tasks. If not set,
        task_manager will load one. This should be set by the user
        if there are additional paths that want to be included
        via `include_path`

    :return
        Dictionary of task objects
    """

    task_name_from_string_dict = {}
    task_name_from_config_dict = {}
    task_name_from_object_dict = {}

    if isinstance(task_name_list, str):
        task_name_list = [task_name_list]
    elif isinstance(task_name_list, list):
        if not all([isinstance(task, (str, dict, Task)) for task in task_name_list]):
            raise TypeError("Expected all list items to be of types 'str', 'dict', or 'Task', but at least one entry did not match.")
    else:
        raise TypeError(f"Expected a 'str' or 'list' but received {type(task_name_list)}.")

    string_task_name_list = [task for task in task_name_list if isinstance(task, str)]
    others_task_name_list = [task for task in task_name_list if not isinstance(task, str)]
    if len(string_task_name_list) > 0:
        if task_manager is None:
            task_manager = TaskManager()

        task_name_from_string_dict = task_manager.load_task_or_group(
            string_task_name_list,
        )

    for task_element in others_task_name_list:
        if isinstance(task_element, dict):
            task_name_from_config_dict = {
                **task_name_from_config_dict,
                **task_manager.load_config(config=task_element),
            }

        elif isinstance(task_element, Task):
            task_name_from_object_dict = {
                **task_name_from_object_dict,
                get_task_name_from_object(task_element): task_element,
            }

    if not set(task_name_from_string_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys())):
        raise ValueError

    final_task_dict = {
        **task_name_from_string_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }

    # behavior can get odd if one tries to invoke several groups that "compete" for the same task.
    # (notably, because one could request several num_fewshot values at once in GroupConfig overrides for the subtask
    # and we'd be unsure which to use and report.)
    # we explicitly check and error in this case.
    _check_duplicates(get_subtask_list(final_task_dict))

    return final_task_dict