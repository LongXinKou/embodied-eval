"""
modified from 
"https://github.com/EmbodiedBench/EmbodiedBench/blob/master/embodiedbench/envs/eb_navigation/EBNavEnv.py"
"""
import copy
import numpy as np
import time
import json
import os
import re
import sys
import math

from PIL import Image, ImageDraw, ImageFont
from loguru import logger as eval_logger

import ai2thor.controller
import gym
from ai2thor.platform import CloudRendering

valid_objs = ['Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD', 'Pan', 'Watch',
                'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone', 'Kettle', 'Mug', 'StoveBurner', 'Bowl', 'Spoon', 'TissueBox', 'Apple', 'TennisRacket', 'SoapBar',
                'Cloth', 'Plunger', 'FloorLamp', 'ToiletPaperHanger', 'Spatula', 'Plate',
                'Glassbottle', 'Knife', 'Tomato', 'ButterKnife', 'Dresser', 'Microwave',
                'GarbageCan', 'WateringCan', 'Vase', 'ArmChair', 'Safe', 'KeyChain', 'Pot', 'Pen', 'Newspaper', 'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock',
                'ToiletPaper', 'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle', 'Pencil',
                'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'SaltShaker', 'PepperShaker',
                'Pillow', 'Bathtub', 'SoapBottle', 'Statue', 'Fridge', 'Toaster', 'LaundryHamper']

def random_color():
    return tuple(np.random.choice(range(256), size=3))

def draw_boxes(image, classes_and_boxes, image_path):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    font.size = 8
    # Loop through each class and its associated bounding boxes
    for class_name, box in classes_and_boxes.items():
        if class_name.split('|')[0] in valid_objs:
            color = random_color()
            # if class_name in name_translation:
            #     name = name_translation[class_name]
            # else:
            #     name = class_name.split('|')[0]
            
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            # Add class name above the rectangle
            # text_position = (x1, max(0, y1 - 12))  # Position text above box
            # draw.text(text_position, name, fill=color, font=font)
    image.save(image_path)
    # return image

SUCCESS_THRESHOLD = 1

ValidEvalSets = [
    'base', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon'
]

DISCRETE_SKILLSET = [
    "Move forward by 0.25",
    "Move backward by 0.25",
    "Move rightward by 0.25",
    "Move leftward by 0.25",
    "Rotate to the right by 90 degrees.",
    "Rotate to the left by 90 degrees.",
    "Tilt the camera upward by 30 degrees.",
    "Tilt the camera downward by 30 degrees.",
    # "Crouch to be lower",
    # "Stand to be taller",
    # "Complete the current task."
]

# "Move forward by 0.25 meter.",
# "Move backward by 0.25 meter.",
# "Move right by 0.25 meter.",
# "Move left by 0.25 meter.",


class EBNavigationEnv(gym.Env):
    def __init__(
        self, 
        eval_set='base', 
        exp_name='test_base', 
        down_sample_ratio=1.0, 
        fov = 100, 
        multiview = False, 
        boundingbox = False, 
        multistep = False,  
        resolution = 500, 
        selected_indexes =[],
        **kwargs,
    ):
        """
        A wrapper for AI2-THOR ManipulaTHOR environment.

        :param config: Dictionary containing initialization parameters for the controller.
        """
        self.resolution = resolution
        self.config = {
            "agentMode": "default",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": True,
            "renderInstanceSegmentation": True,
            "width": self.resolution,
            "height": self.resolution,
            "fieldOfView": fov,
            "platform": CloudRendering
        }
        self.env = ai2thor.controller.Controller(**self.config)

        # load dataset
        assert eval_set in ValidEvalSets
        self.down_sample_ratio = down_sample_ratio
        self.data_path = os.path.join(os.path.dirname(__file__), f"datasets/{eval_set}.json")
        self.dataset = self._load_dataset(eval_set)
        if len(selected_indexes):
            self.dataset = [self.dataset[i] for i in selected_indexes]

        self.selected_indexes = selected_indexes

        # Episode tracking
        self.number_of_episodes = len(self.dataset)
        self._reset = False
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 20
        self._episode_start_time = 0
        self.is_holding = False
        self.episode_log = []
        self.episode_language_instruction = ""
        self.episode_data = None

        self._last_event = None

        self.standing = True

        # set action space
        self.language_skill_set = DISCRETE_SKILLSET
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))

        # set observation space
        self.obs_key = kwargs.get('obs_key', 'head_rgb')

        # set log and verbosity(0 for concise)
        self.feedback_verbosity = 0
        self.log_path = 'running/eb_nav/{}'.format(exp_name)

        self.system_prompt = kwargs.get('system_prompt', "")

        self.multiview = multiview
        self.boundingbox = boundingbox
        self.multistep = multistep
        self.img_paths = []
    
    @property
    def actions(self):
        return self.language_skill_set
    
    @property
    def available_action_str(self):
        return self.get_availabel_action_prompt(self.actions)
    
    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str

    def _load_dataset(self, eval_set):
        with open(self.data_path) as f:
            dataset_split = json.load(f)
        dataset = dataset_split["tasks"]
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        return dataset

    def reset(self, **kwargs):
        """
        Reset the environment.

        :param scene: Optionally set the scene for reset.
        :return: The initial observation.
        """
        # self.save_episode_log()
        assert self._current_episode_num < self.number_of_episodes

        # start reset environment 
        traj_data = self.dataset[self._current_episode_num]
        self.episode_data = traj_data
        self.episode_language_instruction = traj_data["instruction"]

        scene_name = traj_data["scene"]
        eval_logger.info(f"Restoring scene {scene_name}...")
        self._last_event = self.env.reset(
            scene=scene_name
        )

        if self.multiview:
            event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = copy.deepcopy(event.metadata["actionReturn"])
            pose["orthographic"] = True

            # add the camera to the scene
            self.env.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )

        pose = traj_data["agentPose"]
        self.env.step(
            action="Teleport",
            position={
                "x": pose["position"]["x"],
                "y": pose["position"]["y"],
                "z": pose["position"]["z"]
            },
            rotation={
                "x": 0,
                "y": pose["rotation"],
                "z": 0
            },
            horizon=pose["horizon"],
            standing=True
        )

        # finish reset environment 
        # reset episode information
        self._current_episode_num += 1
        self._current_step = 0

        self.standing = True
        obs = {
            'head_rgb': self.env.last_event.frame
        }
        obs = obs[self.obs_key] if self.obs_key and self.obs_key in obs else obs
        self._reset = True
        self.episode_log = []
        self._episode_start_time = time.time()

        self.img_paths = []

        return obs
    
    def generate_prompt(self, user_instruction, prev_act_feedback = [], **kwarg):
        user_instruction = user_instruction.rstrip('.')

        n_shot = kwarg.get('n_shot', 0)
        examples = kwarg.get('example', None)
        language_only = kwarg.get('language_only', False)
        chat_history = kwarg.get('chat_history', True)
        use_feedback = kwarg.get('use_feedback', True)   

        if len(prev_act_feedback) == 0:
            if n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(examples[:n_shot])])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            if language_only:
                prompt += f" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            else:
                prompt += f" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
        
        elif chat_history:
            prompt = f'The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        else:
            if n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(examples[:n_shot])])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        return prompt

    def parse_action_response(self, response):
        """
        Parse the action and reasoning from the model's response.

        Supports both:
        - A single JSON object: {"action_id": 1, "reasoning": "..."}
        - A list of JSON objects: [{"action_id": 1, "reasoning": "..."}, {...}]
        
        :param response: The raw string response from the model.
        :return: A tuple (action_index, reasoning) from the first object, or (None, "") if parsing fails.
        """
        try:
            blocks = re.findall(r'```json\s*([\s\S]*?)```', response)
            response_json = json.loads(blocks[-1])
            if isinstance(response_json, list):
                action_index = [int(item.get("action_id", -1)) for item in response_json]
                reasoning = [item.get("reasoning", item.get("reason", "")) for item in response_json]
                # Validate action indices in list
                for idx in action_index:
                    if idx < 0 or idx >= len(self.actions):
                        eval_logger.error(f"Invalid action index {idx}. Must be between 0 and {len(self.actions)-1}")
                        return None, ""
            else:
                action_index = int(response_json.get("action_id", -1))
                reasoning = response_json.get("reasoning", response_json.get("reason", ""))
                # Validate single action index
                if action_index < 0 or action_index >= len(self.actions):
                    eval_logger.error(f"Invalid action index {action_index}. Must be between 0 and {len(self.actions)-1}")
                    return None, ""
            return action_index, reasoning
        except (json.JSONDecodeError, ValueError) as e:
            eval_logger.error(f"Failed to parse response: {response}. Error: {e}")
            return None, ""

    def discrete_action_mapper(self, action_index):
        """
        Maps a discrete action index to the corresponding iTHOR environment action.

        Parameters:
            env: The AI2-THOR environment object.
            action_index: An integer representing the action index.

        Raises:
            ValueError: If the action index is invalid.
        """

        if action_index == 0:  # Move forward by 0.25 meter
            self._last_event = self.env.step(action="MoveAhead", moveMagnitude=0.25)
        elif action_index == 1:  # Move backward by 0.25 meter
            self._last_event = self.env.step(action="MoveBack", moveMagnitude=0.25)
        elif action_index == 2:  # Move right by 0.25 meter
            self._last_event = self.env.step(action="MoveRight", moveMagnitude=0.25)
        elif action_index == 3:  # Move left by 0.25 meter
            self._last_event = self.env.step(action="MoveLeft", moveMagnitude=0.25)
        elif action_index == 4:  # Rotate clockwise by 45 degrees
            self._last_event = self.env.step(action="RotateRight", degrees=90)
        elif action_index == 5:  # Rotate counterclockwise by 45 degrees
            self._last_event = self.env.step(action="RotateLeft", degrees=90)
        elif action_index == 6:  # Tilt the camera upward by 30 degrees
            self._last_event = self.env.step(action="LookUp", degrees=30)
        elif action_index == 7:  # Tilt the camera downward by 30 degrees
            self._last_event = self.env.step(action="LookDown", degrees=30)
        # elif action_index == 8:  # Crouch to be lower
        #     self._last_event = self.env.step(action="Crouch")
        #     self.standing = False
        # elif action_index == 9:  # Stand to be taller
        #     self._last_event = self.env.step(action="Stand")
        #     self.standing = True
        # elif action_index == 8:  # Complete the current task
        #     self._last_event = self.env.step(action="Done")
        else:
            print(f"Invalid action index: {action_index}")

    def measure_success(self):
        # success measurement
        agent_position = self.env.last_event.metadata["agent"]["position"]
        target_object_id = self.episode_data["targetObjectIds"]
        target_position = self.episode_data["target_position"]

        # for obj in self.env.last_event.metadata["objects"]:
        #     if obj["objectId"] == target_object_id:
        #         target_position = obj["position"]
        #         break

        dist = math.sqrt(
            (agent_position["x"] - target_position["x"])**2 +
            (agent_position["z"] - target_position["z"])**2
        )
        success = (dist <= SUCCESS_THRESHOLD)
        return float(success), dist

        

    def step(self, action: int, reasoning, i_flag):
        """
        Perform an action in the environment.

        :param action: The name of the action to perform.
        :param kwargs: Additional parameters for the action.
        :return: Event.
        """

        assert self._reset, 'Reset env before stepping'
        info = {}

        self._current_step += 1

        if self._current_step>=self._max_episode_steps:

            if type(action)!=int or action > 7 or action < 0:
                action = np.random.randint(8)

            self.discrete_action_mapper(action)
            reward, distance = self.measure_success()
            done = True
            info['action_description'] = self.language_skill_set[action]

        else:
            if type(action)!=int or action > 7 or action < 0:
                action = np.random.randint(8)

            self.discrete_action_mapper(action)
            reward, distance = self.measure_success()
            if reward>0:
                done = True
            else:
                done = False
            info['action_description'] = self.language_skill_set[action]

        #info['action_description'] = self.language_skill_set[action]

        obs = {
            'head_rgb': self.env.last_event.frame,
        }
        obs = obs[self.obs_key] if self.obs_key and self.obs_key in obs else obs
        reward, distance = self.measure_success()

        ## test calculate reward
        info['distance'] = distance
        info['env_feedback'] = self.get_env_feedback(self._last_event)
        info['reasoning'] = reasoning
        # info['reflection'] = reasoning['reasoning_and_reflection']
        # info['plan'] = reasoning['language_plan']
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['task_success'] = reward
        info['last_action_success'] = self.env.last_event.metadata['lastActionSuccess']
        info['action_id'] = action
        # info['reasoning'] = reasoning

        self.episode_log.append(info)

        if i_flag == 1:
            self.save_episode_log_per_step(1)
        else:
            self.save_episode_log_per_step(0)
        
        self.episode_log = []

        return obs, reward, done, info
        
    def get_env_feedback(self, event):
        """
        To extract relevant information from the event to construct a feedback dictionary.

        :param event: self._last_event
        :return: A dictionary containing structured feedback.
        """
        if self.feedback_verbosity == 1:
            feedback = {
                "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
                "errorMessage": event.metadata.get("errorMessage", None),
                "lastAction": event.metadata.get("lastAction", None),

                "agent": {
                    "position": event.metadata.get("agent", {}).get("position", {}),
                    "rotation": event.metadata.get("agent", {}).get("rotation", {}),
                    "is_standing": self.standing
                }
            }
        else:
            # Does not provide the specific reason why the action fails if so
            feedback = {
                "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
                "lastAction": event.metadata.get("lastAction", None),
                "errorMessage": event.metadata.get("errorMessage", None),

                "agent": {
                    "is_standing": self.standing
                }
            }

        msg = ''
        if feedback["lastActionSuccess"]:
            msg += f"Last action {feedback['lastAction']} executed successfully."
        else:
            msg += f"Last action {feedback['lastAction']} is invalid. {feedback['errorMessage']}"
        return msg

    def seed(self, seed=None):
        self.env.random_initilize(seed)


    def save_image(self, *args, **kwargs):
        """Save current agent view as a PNG image."""
        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if self.multiview:
            img1 = Image.fromarray(self.env.last_event.frame)
            img2 = Image.fromarray(self.env.last_event.third_party_camera_frames[-1])
            time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            # image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, time_stamp)
            image_path1 = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front.png'.format(episode_idx, self._current_step, time_stamp))
            image_path2 = os.path.join(self.log_path, 'episode_{}_step_{}_{}_top.png'.format(episode_idx, self._current_step, time_stamp))
            img1.save(image_path1)
            img2.save(image_path2)
            return [image_path1, image_path2]
        
        elif self.multistep:
            
            img = Image.fromarray(self.env.last_event.frame)
            time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            # image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, time_stamp)
            image_path = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front.png'.format(episode_idx, self._current_step, time_stamp))
            img.save(image_path)
            self.img_paths.append(image_path)
            if self._current_step<3:
                return self.img_paths
            else:
                return self.img_paths[-3:]

        else:
            if not self.boundingbox:
                img = Image.fromarray(self.env.last_event.frame)
                time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                # image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, time_stamp)
                image_path = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front.png'.format(episode_idx, self._current_step, time_stamp))
                img.save(image_path)
                return image_path
            else:
                img = Image.fromarray(self.env.last_event.frame)
                time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                # image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, time_stamp)
                image_path = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front_bb.png'.format(episode_idx, self._current_step, time_stamp))
                # if self.target_only:
                #     draw_target_box(img, self.env.last_event.instance_detections2D, self.episode_data["targetObjectIds"], image_path)
                # else:
                draw_boxes(img,self.env.last_event.instance_detections2D, image_path)
                # img.save(image_path)
                return image_path

    def save_episode_log_per_step(self, flag):

        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = 'episode_{}.json'.format(episode_idx)
        if len(self.episode_log):
            with open(os.path.join(self.log_path, filename), 'a') as f:
                if flag == 1:
                    f.write('\n\n')
                    for item in self.episode_log:
                        if 'object_states' in item:
                            item.pop('object_states')
                        try:
                            json.dump(item, f, ensure_ascii=False)
                        except:
                            import pdb;pdb.set_trace()
                        f.write('\n') 
                else:
                    for item in self.episode_log:
                        if 'object_states' in item:
                            item.pop('object_states')
                        try:
                            json.dump(item, f, ensure_ascii=False)
                        except:
                            import pdb;pdb.set_trace()
                        f.write('\n') 

    # def save_episode_log(self):
    #     if not os.path.exists(self.log_path):
    #         os.makedirs(self.log_path)
    #     time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    #     filename = 'episode_{}_step_{}_{}.json'.format(self._current_episode_num, self._current_step, time_stamp)
    #     if len(self.episode_log):
    #         with open(os.path.join(self.log_path, filename), 'w') as f:
    #             for item in self.episode_log:
    #                 if 'object_states' in item:
    #                     item.pop('object_states')
    #                 try:
    #                     json.dump(item, f, ensure_ascii=False)
    #                 except:
    #                     import pdb;pdb.set_trace()
    #                 f.write('\n') 

    def close(self):
        """Close the environment."""
        self.env.stop()


if __name__ == "__main__":

    env = EBNavigationEnv("base")
    env.reset()
    print([(i, name) for i, name in enumerate(env.language_skill_set)])
    for _ in range(30):
        # Select  action
        action = int(input('action id: ')) #env.action_space.sample()
        if action in env.language_skill_set:
            action = env.language_skill_set.index(action)
        else:
            action = int(action)
            if action < 0:
                break
        
        print(env.language_skill_set[action])
        
        # Execute action
        obs, reward, done, info = env.step(action, "", 1)
        print(reward, done, info)
        # Optional rendering and image saving
        env.save_image()
        if done:
            break
    env.close()
    env.close()