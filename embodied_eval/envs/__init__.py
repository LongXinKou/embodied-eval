import abc
import os
from typing import List, Optional, Tuple, Type, TypeVar, Union
from loguru import logger

from embodied_eval.utils import load_yaml_config, simple_parse_args_string

T = TypeVar("T", bound="BaseEnvironment")

AVAILABLE_ENVS = {
    "EBNavEnv": "EBNavigationEnv",
}

class BaseEnvironment(abc.ABC):
    """Base class for all environments."""
    
    def __init__(self, **kwargs):
        pass
    
    @abc.abstractmethod
    def reset(self):
        """Reset the environment and return initial observation."""
        pass
    
    @abc.abstractmethod
    def step(self, action, reasoning=None, is_first_action=1):
        """Execute an action and return observation, reward, done, info."""
        pass
    
    @abc.abstractmethod
    def close(self):
        """Close the environment."""
        pass

    @abc.abstractmethod
    def save_image(self, obs):
        """Save observation image and return path."""
        pass
    
    @property
    @abc.abstractmethod
    def number_of_episodes(self):
        """Return total number of episodes."""
        pass
    
    @property
    @abc.abstractmethod
    def episode_language_instruction(self):
        """Return current episode instruction."""
        pass
    
    @classmethod
    def create_from_config(
        cls: Type[T], config: dict, additional_config: Optional[dict] = None
    ) -> T:
        """Create environment from config dictionary."""
        additional_config = {} if additional_config is None else additional_config
        args = {k: v for k, v in config.items() if k != 'env'}
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

def get_env(env_name):
    """
    Load environment class and configuration from yaml file.
    env_name --> env_name.yaml --> config + env_class
    """
    if env_name not in AVAILABLE_ENVS:
        raise ValueError(f"Environment {env_name} not found in available environments: {list(AVAILABLE_ENVS.keys())}")

    # Load environment configuration from yaml file
    env_dir = os.path.join(os.path.dirname(__file__), env_name)
    yaml_path = os.path.join(env_dir, f"{env_name}.yaml")
    
    if not os.path.exists(yaml_path):
        logger.warning(f"Configuration file {yaml_path} not found, using default config")
        config = {"env": env_name}
    else:
        config = load_yaml_config(yaml_path)

    env_class_name = AVAILABLE_ENVS[env_name]
    if "." not in env_class_name:
        env_class_path = f"embodied_eval.envs.{env_name}.{env_class_name}"
    else:
        env_class_path = env_class_name

    try:
        env_module, env_class = env_class_path.rsplit(".", 1)
        module = __import__(env_module, fromlist=[env_class])
        env_class_obj = getattr(module, env_class)
        
        # Return a wrapper that can create environment instances with config
        class EnvWithConfig:
            def __init__(self):
                self.env_class = env_class_obj
                self.config = config
            
            def create_from_config(self, additional_config=None):
                """Create environment instance from yaml config."""
                # Merge config with additional_config
                final_config = dict(self.config)
                if additional_config:
                    final_config.update(additional_config)
                
                # Remove 'env' key as it's not a constructor parameter
                env_params = {k: v for k, v in final_config.items() if k != 'env'}
                return self.env_class(**env_params)
                
            def create_from_arg_string(self, arg_string="", additional_config=None):
                """Create environment instance from argument string (legacy support)."""
                args = simple_parse_args_string(arg_string) if arg_string else {}
                final_config = dict(self.config)
                final_config.update(args)
                if additional_config:
                    final_config.update(additional_config)
                
                # Remove 'env' key as it's not a constructor parameter
                env_params = {k: v for k, v in final_config.items() if k != 'env'}
                return self.env_class(**env_params)
        
        return EnvWithConfig()
        
    except Exception as e:
        logger.error(f"Failed to import environment {env_name}: {e}")
        raise
