import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from packaging import version
from transformers import AutoProcessor, AutoModelForPreTraining
from typing import List, Optional, Union
from loguru import logger as eval_logger


from roboannotatorx.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images
from roboannotatorx.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from roboannotatorx.conversation import conv_templates, SeparatorStyle
from roboannotatorx.model.builder import load_roboannotator

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator


@register_model("roboannotatorx")
class RoboAnnotatorX(BaseAPIModel):
    """
    RoboAnnotatorX Model
    "https://huggingface.co/koulx/roboannotatorx"
    """

    def __init__(
            self,
            model_name_or_path: str = "koulx/roboannotatorx",
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "cuda",
            max_length: Optional[int] = 2048,
            max_new_tokens: Optional[int] = 1024,
            temperature: float = 0,
            do_sample: bool = False,
            top_p: Optional[int] = None,
            num_beams: Optional[int] = 1,
            use_cache: Optional[bool] = True,
            system_prompt: Optional[str] = None,
            model_name: Optional[str] = None,
            interval: int = 32,
            mm_use_im_start_end: bool = False,
            conv_mode: Optional[str] = "llava_v1",
            **kwargs,
    ) -> None:
        super().__init__()

        # Handle distributed setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Load model
        eval_logger.info(f"Loading RoboAnnotatorX model from {model_name_or_path}")
        model_name = "RoboAnnotatorX"
        eval_logger.info(f"Model name is {model_name}")
        self._tokenizer, self._model, self._image_processor, self._max_length = load_roboannotator(
            model_path = model_name_or_path, 
            model_base = None, 
            device_map=self.device_map, 
        )
        self.model.eval()

        # Store configuration
        self._config = self._model.config
        self._max_length = max_length if getattr(self._config, "max_length", None) else self._config.max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.num_beams = num_beams
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        
        self.model_name = model_name
        self.interval = interval
        self.mm_use_im_start_end = mm_use_im_start_end if mm_use_im_start_end is not None else self.model.config.mm_use_im_start_end
        self.conv_mode = conv_mode
        self.conv_templates = conv_templates[self.conv_mode]

        # Set up distributed evaluation
        if accelerator.num_processes > 1:
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
    
    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model
    
    @property
    def max_length(self):
        return self._max_length
    
    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
    
    def load_video(self, video_path):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        frame_idx = [i for i in range(0, len(vr))]
        if len(frame_idx) > 2000: # 4x downsampling
            frame_idx = frame_idx[::4]
        elif len(frame_idx) > 500: # 2x downsampling
            frame_idx = frame_idx[::2]
        total_frames = vr.get_batch(frame_idx).asnumpy()
        return total_frames
    
    def respond(self, context, visuals, **gen_kwargs):
        """
        Generate a text response based on the given context and visual inputs.
        Args:
            context (str): The input text context for the response.
            visuals (list): A list of visual inputs (e.g., images or videos) to process.
            gen_kwargs (dict, optional): Additional keyword arguments for text generation.
        Returns:
            str: The generated text response.
        """
        # Set default generation parameters
        gen_kwargs = dict(gen_kwargs) if gen_kwargs else {}
        if "until" in gen_kwargs:
            gen_kwargs.pop("until")
        
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = self.max_new_tokens
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = self.do_sample
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = self.temperature
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = self.top_p
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = self.num_beams
        if "use_cache" not in gen_kwargs:
            gen_kwargs["use_cache"] = self.use_cache

        try:
            # Process visual inputs
            if isinstance(visuals[0], Image.Image):
                # Process images
                image_tensor = process_images(visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            elif isinstance(visuals[0], str):
                # Process video
                image_tensor = []
                frames = self.load_video(visuals)
                frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device)
                image_tensor.append(frames)
            
            # Format the question with visual tokens
            if self.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + context
            else:
                if DEFAULT_IMAGE_TOKEN not in context:
                    question = DEFAULT_IMAGE_TOKEN + '\n' + context
                else:
                    question = context
            
            # Prepare conversation
            conv = self.conv_templates.copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # Tokenize input
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            # Set task-specific parameters
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            gen_kwargs["stopping_criteria"] = [stopping_criteria]

            # Generate response
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids, 
                    images=image_tensor, 
                    **gen_kwargs
                )
            
            text_output = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            torch.cuda.empty_cache()
            
            return text_output.strip()
            
        except Exception as e:
            eval_logger.error(f"Error during RoboAnnotatorX inference: {e}")
            return "Error processing input."