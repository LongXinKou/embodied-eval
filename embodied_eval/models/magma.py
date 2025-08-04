'''
modified from "https://github.com/microsoft/Magma/blob/main/tools/lmms-eval-magma/magma.py"
'''
import base64
from platform import processor
import decord
import torch
import numpy as np

from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import List, Optional, Union
from loguru import logger as eval_logger

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel


@register_model("magma")
class Magma(BaseAPIModel):
    """
    Magma Model
    "https://huggingface.co/microsoft/Magma-8B/"
    """

    def __init__(
            self,
            model_name_or_path: str = "microsoft/Magma-8B",
            torch_dtype = torch.bfloat16, # torch.float16
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "cuda",
            max_length: Optional[int] = 2048,
            max_new_tokens: Optional[int] = 1024,
            temperature: float = 0,
            do_sample: bool = False,
            top_p: Optional[int] = None,
            num_beams: Optional[int] = 1,
            use_cache: Optional[bool] = True,
            system_prompt: Optional[str] = "You are agent that can see, talk and act.",
            max_num_frames: int = 8,
            enable_thinking: bool = False,
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
        eval_logger.info(f"Loading Magma model from {model_name_or_path}")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
            trust_remote_code=True,
        )

        # Load processor
        self._processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

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

        self.max_num_frames = max_num_frames
        self.enable_thinking = enable_thinking
        
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
        return self._config

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
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

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)
    
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
        # Get generation parameters
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
        
        # Process the request
        if "<image>" in context:
            context = context.replace("<image>", "")

        # Process visuals - determine if we have images or videos
        if isinstance(visuals[0], Image.Image):
            images = [visual for visual in visuals]
        elif isinstance(visuals[0], np.ndarray):
            images = [to_pil_image(visual) for visual in visuals]
        elif isinstance(visuals[0], str):
            frames = self.load_video(visuals, self.max_num_frames)
            images = [to_pil_image(frame) for frame in frames]
 
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ''.join(["<image>\n"]*len(images)) + context},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        if self.model.config.mm_use_image_start_end:
            prompt = prompt.replace("<image>", "<image_start><image><image_end>")

        inputs = self.processor(
            images=images,
            texts=prompt,
            return_tensors="pt"
        )
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        inputs = inputs.to(self.device, self.model.dtype)

        self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
        response = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()

        return response
