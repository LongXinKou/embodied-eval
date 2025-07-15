import torch
import numpy as np

from PIL import Image

from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModelForPreTraining
from typing import List, Optional, Union
from loguru import logger as eval_logger

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel

DEFAULT_IMAGE_TOKEN = "<image>"


def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

@register_model("robobrain")
class RoboBrain(BaseAPIModel):
    """
    RoboBrain Model
    "https://huggingface.co/BAAI/RoboBrain"
    """

    def __init__(
            self,
            model_name_or_path: str = "BAAI/RoboBrain",
            lora_id: Optional[str] = None,
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
            max_frames_num: int = 8,
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
        eval_logger.info(f"Loading RoboBrain model from {model_name_or_path}")
        self._model = AutoModelForPreTraining.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device_map
        )

        # Load processor
        self._processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Apply LoRA if specified
        if lora_id is not None:
            try:
                from peft import PeftModel
                eval_logger.info(f"Loading LoRA weights from {lora_id}")
                self._processor.image_processor.image_grid_pinpoints = [[384, 384]]
                self._model.base_model.base_model.config.image_grid_pinpoints = [[384, 384]]
                self._model = PeftModel.from_pretrained(self._model, lora_id)
            except ImportError:
                eval_logger.error("Failed to import PeftModel. Please install peft to use LoRA.")
                raise

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

        self.max_frames_num = max_frames_num
        
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
    def eot_token_id(self):
        # Use the model's EOS token ID
        return self.processor.tokenizer.eos_token_id

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
    
    def _process_image(self, image):
        """Process images for model input."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, (np.ndarray)):
            return Image.fromarray(image).convert("RGB")
        else:
            return None
    
    def respond(self, context, visuals, **gen_kwargs):
        """
        Generate a text response based on the given context and visual inputs.
        Args:
            context (str): The input text context for the response.
            visuals (list): A list of visual inputs (e.g., images or videos) to process.
            gen_kwargs (dict, optional): Additional keyword arguments for text generation.
        Returns:
            str: The generated text response.
        Note:
            For both image and video tasks, multiple image tokens are inserted into the context, corresponding to the number of images or video frames. 
            This is because the training data uses multiple <image> tokens for video inputs, treating each frame as an image. This design ensures consistency with the training data format.
            You can see "https://huggingface.co/datasets/BAAI/ShareRobot/tree/main/planning/jsons" for more details.
        """   
        # Process the request
        if "<image>" in context:
            context = context.replace("<image>", "")

        # Process visuals - determine if we have images or videos
        first_visual = visuals[0]
        
        if isinstance(first_visual, Image.Image) or isinstance(first_visual, np.ndarray):
            if DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                context = f"{''.join(image_tokens)}\n{context}"
            images = [self._process_image(visual) for visual in visuals]
        elif isinstance(first_visual, str):
            frames = load_video(visuals, self.max_frames_num)
            if DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(frames)
                context = f"{''.join(image_tokens)}\n{context}"
            images = [self._process_image(frame) for frame in frames]
            
        messages = [{"role": "user", "content": context}]
        text = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt"
        ).to(self.device, self.model.dtype)

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

        # Generate output
        input_length = inputs['input_ids'].shape[1]
        output = self.model.generate(
            **inputs,
            **gen_kwargs
        )

        # Extract generated text
        generated_ids_trimmed = output[0][input_length:]
        answer = self.processor.tokenizer.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )
        
        return answer.strip()
