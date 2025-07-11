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

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"

@register_model("spatialvlm")
class SpatialVLM(BaseAPIModel):
    """
    SpatialVLM Model
    "https://huggingface.co/remyxai/SpaceLLaVA"
    """

    def __init__(
            self,
            model_name_or_path: str = "remyxai/SpaceLLaVA",
            use_flash_attention_2: bool = True,
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "cuda",
            max_length: Optional[int] = 2048,
            batch_size: Optional[Union[int, str]] = 1,
            max_new_tokens: Optional[int] = 1024,
            temperature: float = 0,
            do_sample: bool = False,
            top_p: Optional[int] = None,
            num_beams: Optional[int] = 1,
            use_cache: Optional[bool] = True,
            system_prompt: Optional[str] = None,
            model_name: Optional[str] = "llava-v1.5",
            max_num_frames: Optional[int] = 32,
            attn_implementation: Optional[str] = best_fit_attn_implementation,
            conv_mode: Optional[str] = "vicuna_v1",
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
        
        llava_model_args = {
            "multimodal": True,
            "torch_dtype": "float16",
            "attn_implementation": attn_implementation if not use_flash_attention_2 else "flash_attention_2",
        }

        # Load model
        eval_logger.info(f"Loading SpatialVLM model from {model_name_or_path}")
        model_name = get_model_name_from_path(model_name_or_path) if model_name is None else model_name
        eval_logger.info(f"Model name is {model_name}")
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            model_path=model_name_or_path,
            model_base=None,
            model_name=model_name, 
            device_map=self.device_map, 
            **llava_model_args
        )
        self.model.eval()

        # Store configuration
        self._config = self._model.config
        self._max_length = max_length if getattr(self._config, "max_length", None) else self._config.max_length
        self.batch_size_per_gpu = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.num_beams = num_beams
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        
        self.model_name = model_name
        self.conv_mode = conv_mode
        self.conv_template = conv_templates[self.conv_mode]
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
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length
    
    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids
    
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
    
    def generate_until(self, requests) -> List[str]:
        """Generate text until a stopping sequence."""
        res = []

        def _sort_by_context_length(x):
            # Sort by context length for better batching
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]
        
        # Initialize the Collator to group requests and sort them by context length
        collator = Collator([req.args for req in requests], _sort_by_context_length, grouping=True)
        # Create batches from the sorted and grouped requests
        batches = collator.get_batched(n=self.batch_size, batch_fn=None)

        # Determine the number of iterations required to process all requests
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        progress_bar = tqdm(total=num_iters, disable=(self.rank != 0), desc="SpatialVLM Responding")

        for batch in batches:
            batch_contexts, all_gen_kwargs, batch_doc_to_visual, batch_doc_id, batch_task, batch_split = zip(*batch)
            task = batch_task[0]
            split = batch_split[0]

            gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
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

            batch_visuals = []
            for ids in batch_doc_id:
                doc_to_visual = batch_doc_to_visual[0]
                if split is not None:
                    visuals = doc_to_visual(self.task_dict[task][split][ids])
                    if isinstance(visuals, tuple): # ([visual], [visual_index])
                        batch_visuals.append(visuals[0])
                    else:
                        batch_visuals.append(visuals) # [visual]
                else:
                    visuals = doc_to_visual(self.task_dict[task][ids])
                    if isinstance(visuals, tuple):
                        batch_visuals.append(visuals[0])
                    else:
                        batch_visuals.append(visuals)
            
            question_input = []
            for visual, context in zip(batch_visuals, batch_contexts): # for multi-modal task
                # For image, multi-image tasks
                if isinstance(visual[0], Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    
                    task_type = "image"
                    placeholder_count = 1
                
                # For video task
                elif isinstance(visual[0], str):
                    image_tensor = []
                    try:
                        frames = self.load_video(visual, self.max_num_frames)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None
                    
                    task_type = "video"
                    placeholder_count = 1
                
                else:
                    image_tensor = None
                
                # For multimodal context
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                
                # For conv template
                conv = self.conv_template.copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # Input (input_ids, attention_mask, image_tensor)
            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            try:
                with torch.inference_mode():
                    cont = self.model.generate(
                        input_ids, 
                        attention_mask=attention_masks, 
                        pad_token_id=pad_token_ids, 
                        images=image_tensor, 
                        **gen_kwargs
                    )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            progress_bar.update(1)

        # Reorder results to match original order
        res = collator.get_original(res)
        progress_bar.close()
        return res