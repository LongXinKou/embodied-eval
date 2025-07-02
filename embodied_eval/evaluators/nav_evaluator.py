import os
import json
import collections
import numpy as np
from loguru import logger as eval_logger
from tqdm import tqdm
import time

from embodied_eval.models import get_model
from embodied_eval.envs import get_env
from embodied_eval.utils import get_datetime_str

class NavEvaluator:
    """Navigation evaluator for embodied AI tasks with planner-environment interaction."""
    
    def __init__(self, args):
        self._config = args

        # ========== Initialize Planner ==========
        # Create planner model from config
        eval_logger.info('Initializing planner...')
        model_name = self.config.model
        model_args = "" if self.config.model_args is None else self.config.model_args
        self.planner = get_model(model_name=model_name).create_from_arg_string(
            model_args,
            additional_config={
                "batch_size": self.config.batch_size,
            }
        )
        eval_logger.info(f"Planner initialized: {model_name}")

        # ========== Initialize Environment ==========
        # Create navigation environment from config
        eval_logger.info('Initializing environment...')
        assert self.config.env is not None, "Environment name must be specified via --env argument"
        env_name = self.config.env
        self.env = get_env(env_name=env_name).create_from_config()
        eval_logger.info(f"Environment initialized: {env_name}")
        
        # Storage for evaluation results
        self.episode_results = []
        self.episode_samples = []

    @property
    def config(self):
        return self._config

    def inference(self):
        """Run navigation episodes and collect results."""
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            done = False
            episode_info = {'reward': []}

            # Reset planner state for new episode
            self.planner.reset()
            obs = self.env.reset()
            current_image = obs  
            user_instruction = self.env.episode_language_instruction
            eval_logger.info(f"Instruction: {user_instruction}")
            
            # Execute actions until episode ends
            while not done:
                try:
                    # Get action and reasoning from planner
                    action, reasoning = self.planner.act(current_image, user_instruction)
                    reasoning = json.loads(reasoning) if isinstance(reasoning, str) else reasoning
                    
                    # Handle multi-step actions
                    if isinstance(action, list):
                        for i, action_single in enumerate(action[:min(self.env._max_episode_steps - self.env._current_step + 1, len(action))]):
                            obs, reward, done, info = self.env.step(action_single, reasoning, int(i==0))
                            eval_logger.info(f"reward: {reward}")
                            eval_logger.info(f"terminate: {done}\n")
                            self.planner.update_info(info)
                            current_image = obs  
                            episode_info['reward'].append(reward)
                            if done:
                                break
                            # Stop for replanning if action failed
                            if info.get('last_action_success', 1) == 0:
                                eval_logger.info('invalid action, start replanning')
                                break
                    # Handle single action
                    else:
                        obs, reward, done, info = self.env.step(action, reasoning, 1)
                        eval_logger.info(f"reward: {reward}")
                        eval_logger.info(f"terminate: {done}\n")
                        self.planner.update_info(info)
                        current_image = obs
                        episode_info['reward'].append(reward)
                except Exception as e:
                    time.sleep(1)
                    eval_logger.info(f"Error {e}")
            
            # Collect episode metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = float(np.mean(episode_info['reward']))
            episode_info['task_success'] = info['task_success']
            episode_info['num_steps'] = info.get("env_step", 0)
            episode_info['planner_steps'] = getattr(self.planner, 'planner_steps', 0)
            episode_info['planner_output_error'] = getattr(self.planner, 'output_json_error', 0)
            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", 0)
            
            # Store results
            self.episode_results.append(episode_info)
            self.episode_samples.append({
                'episode_id': self.env._current_episode_num,
                'instruction': user_instruction,
                'episode_info': episode_info
            })
        
            progress_bar.update()
        
        progress_bar.close()
        return self.episode_results

    def evaluate(self, episode_results):
        episode_results = episode_results if episode_results is not None else self.episode_results
        
        if not episode_results:
            eval_logger.warning("No evaluation tasks to process")
            return {}
        
        results_dict = collections.defaultdict(dict)
        
        total_episodes = len(episode_results)
        successful_episodes = sum(1 for ep in episode_results if ep.get('task_success', False))
        avg_reward = np.mean([ep.get('reward', 0) for ep in episode_results])
        avg_steps = np.mean([ep.get('num_steps', 0) for ep in episode_results])
        avg_planner_steps = np.mean([ep.get('planner_steps', 0) for ep in episode_results])
        avg_time = np.mean([ep.get('episode_elapsed_seconds', 0) for ep in episode_results])
        
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
        
        results = {
            'total_episodes': total_episodes,
            'successful_episodes': successful_episodes,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'avg_planner_steps': avg_planner_steps,
            'avg_episode_time': avg_time,
        }
        
        results_dict["results"] = {"navigation": results}
        results_dict["samples"] = {"navigation": self.episode_samples}
        results_dict["configs"] = {"navigation": vars(self.config)}
        
        eval_logger.info(f"Navigation evaluation completed with {total_episodes} episodes")
        
        return results_dict

    def print_results(self, results_dict):
        if not results_dict or "results" not in results_dict:
            eval_logger.warning("No results to print")
            return
        
        results = results_dict["results"]
        model = self.config.model
        
        eval_logger.info(f"Results for {model}:")
        eval_logger.info("=" * 50)
        
        if "navigation" in results:
            nav_results = results["navigation"]
            eval_logger.info("Navigation Task Results:")
            eval_logger.info(f"  Total Episodes: {nav_results['total_episodes']}")
            eval_logger.info(f"  Successful Episodes: {nav_results['successful_episodes']}")
            eval_logger.info(f"  Success Rate: {nav_results['success_rate']:.4f}")
            eval_logger.info(f"  Average Reward: {nav_results['avg_reward']:.4f}")
            eval_logger.info(f"  Average Steps: {nav_results['avg_steps']:.2f}")
            eval_logger.info(f"  Average Planner Steps: {nav_results['avg_planner_steps']:.2f}")
            eval_logger.info(f"  Average Episode Time: {nav_results['avg_episode_time']:.2f}s")

    def save_results(self, results_dict):
        if not results_dict:
            eval_logger.warning("No results to save")
            return
        
        def handle_non_serializable(o):
            if isinstance(o, np.int64) or isinstance(o, np.int32):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        tasks_samples = results_dict.get('samples', {})
        tasks_results = results_dict.get('results', {})
        tasks_configs = results_dict.get('configs', {})
        datetime_str = get_datetime_str(timezone=getattr(self.config, 'timezone', 'UTC'))

        for task_name in tasks_results.keys():
            samples = tasks_samples.get(task_name, [])
            results = tasks_results.get(task_name, {})
            configs = tasks_configs.get(task_name, {})

            if hasattr(self.config, 'output_path') and self.config.output_path:
                output_dir = os.path.join(self.config.output_path, datetime_str)
                os.makedirs(output_dir, exist_ok=True)
                eval_logger.info(f"Saving results for: {task_name}")

                file_results_samples = os.path.join(output_dir, f"samples_{task_name}.json")
                with open(file_results_samples, "w", encoding="utf-8") as f:
                    json.dump(samples, f, default=handle_non_serializable, ensure_ascii=False, indent=2)

                file_results_aggregated = os.path.join(output_dir, f"results_{task_name}.json")
                with open(file_results_aggregated, "w", encoding="utf-8") as f:
                    json.dump(results, f, default=handle_non_serializable, ensure_ascii=False, indent=2)

                file_configs_aggregated = os.path.join(output_dir, f"configs_{task_name}.json")
                with open(file_configs_aggregated, "w", encoding="utf-8") as f:
                    json.dump(configs, f, default=handle_non_serializable, ensure_ascii=False, indent=2)
            else:
                eval_logger.info("Output path not provided, skipping saving results")
