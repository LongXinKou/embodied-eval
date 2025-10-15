'''
modified from "https://github.com/FlagOpen/RoboBrain2.0/blob/main/inference.py"
'''
import base64
import decord
import torch
import numpy as np

from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from io import BytesIO
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Optional, Union
from loguru import logger as eval_logger

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel


@register_model("embodied_r1")
class EmbodiedR1(BaseAPIModel):
    """
    EmbodiedR1 Model
    "https://huggingface.co/IffYuan/Embodied-R1-3B-Stage1"
    """

    def __init__(
            self,
            model_name_or_path: str = "IffYuan/Embodied-R1-3B-Stage1",
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
            max_num_frames: int = 8,
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
        eval_logger.info(f"Loading Embodied-R1 model from {model_name_or_path}")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
        )

        # Load processor
        self._processor = AutoProcessor.from_pretrained(model_name_or_path)

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
    
    def process_visuals(self, visual):
        """Process visuals for the model."""
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
            # Video file
            vr = decord.VideoReader(visual)
            first_frame = vr[0].asnumpy()
            height, width = first_frame.shape[:2]
            processed_visual = {
                "type": "video", 
                "video": visual, 
            }
        elif isinstance(visual, Image.Image) or isinstance(visual, np.ndarray):
            if isinstance(visual, np.ndarray):
                visual = Image.fromarray(visual)
            # Handle both single and multiple images
            base64_image = visual.convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            processed_visual = {
                "type": "image", 
                "image": f"data:image/jpeg;base64,{base64_string}", 
            }
        return processed_visual

    def build_messages(self, context: str, visuals: List[dict]) -> List[dict]:
        """Build messages for the model."""
        message = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        message.append(
            {
                "role": "user",
                "content": visuals + [{"type": "text", "text": context}],
            }
        )
        return message
    
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

        # Process visuals
        processed_visuals = [self.process_visuals(visual) for visual in visuals]
        
        # Build the message
        message = self.build_messages(context, processed_visuals)

        text = self.processor.tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(message)
        if video_inputs is not None:
            total_frames = video_inputs[0].shape[0]
            if total_frames > self.max_num_frames:
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device, self.model.dtype)

        generated_ids = self.model.generate(
            **inputs,
            **gen_kwargs
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
