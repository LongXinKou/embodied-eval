import base64
import decord
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from io import BytesIO
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import List, Optional, Union

from loguru import logger as eval_logger
from qwen_vl_utils import process_vision_info

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator

@register_model("cosmos_reason1")
class CosmosReason1(BaseAPIModel):
    """
    CosmosReason1-7B Model
    "https://huggingface.co/nvidia/Cosmos-Reason1-7B"
    """

    def __init__(
            self,
            model_name_or_path: str = "nvidia/Cosmos-Reason1-7B",
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "cuda",
            max_length: Optional[int] = 2048,
            batch_size: Optional[Union[int, str]] = 1,
            temperature: float = 0,
            do_sample: bool = False,
            top_p: Optional[int] = None,
            num_beams: Optional[int] = 1,
            use_cache: Optional[bool] = True,
            system_prompt: Optional[str] = None,
            use_flash_attention_2: Optional[bool] = False,
            max_num_frames: int = 32,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Handle distributed setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Load model
        eval_logger.info(f"Loading Cosmos-Reason1-7B model from {model_name_or_path}")
        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=self.device_map
            ).eval()

        self._processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Store configuration
        self._config = self._model.config
        self._max_length = max_length if getattr(self._config, "max_length", None) else self._config.max_length
        self.batch_size_per_gpu = int(batch_size)
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
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device
    
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
        # Process the request
        if "<image>" in context:
            context = context.replace("<image>", "")

        # Process visuals
        processed_visuals = [self.process_visuals(visual) for visual in visuals]
        
        # Build the message
        message = self.build_messages(context, processed_visuals)
        
        # Apply chat template and process vision info
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([message])
        if video_inputs is not None:
            total_frames = video_inputs[0].shape[0]
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
            return_tensors="pt"
        ).to(self.device)

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

        cont = self.model.generate(
            **inputs,
            **gen_kwargs
        )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        text_output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return text_output.strip()

