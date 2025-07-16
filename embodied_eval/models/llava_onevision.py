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

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
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

@register_model("llava_onevision")
class Llava_OneVision(BaseAPIModel):
    """
    LLaVA-OneVision Model
    "https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov"
    """

    def __init__(
            self,
            model_name_or_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
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
            max_num_frames: Optional[int] = 32,
            mm_spatial_pool_stride: Optional[int] = 2,
            mm_spatial_pool_mode: Optional[str] = "bilinear",
            token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
            attn_implementation: Optional[str] = best_fit_attn_implementation,
            conv_mode: Optional[str] = "qwen_1_5",
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
            "attn_implementation": attn_implementation,
            "torch_dtype": "float16",
        }

        # Load model
        eval_logger.info(f"Loading LLaVA-OneVision model from {model_name_or_path}")
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
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.num_beams = num_beams
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        
        self.model_name = model_name
        self.max_num_frames = max_num_frames
        self.token_strategy = token_strategy
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode

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
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

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
            # Handle multi-image aspect ratio setting
            if len(visuals) > 1 and "image_aspect_ratio" not in self._config.__dict__:  
                self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                if "image_aspect_ratio" in gen_kwargs:
                    gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

            # Process visual inputs
            if isinstance(visuals[0], Image.Image):
                # Process images
                image_tensor = process_images(visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                
                task_type = "image"
                placeholder_count = len(visuals)
            elif isinstance(visuals[0], str):
                # Process video
                image_tensor = []
                frames = self.load_video(visuals, max_frames_num=self.max_num_frames)
                frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device)
                image_tensor.append(frames)

                task_type = "video"
                placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
            else:
                raise ValueError(f"Unsupported visual type: {type(visuals[0])}")
            
            # Format the question with visual tokens
            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                question = image_tokens + "\n" + context
            else:
                question = context
            
            # Prepare conversation
            conv = self.conv_templates.copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # Tokenize input
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            attention_mask = input_ids.ne(pad_token_ids).to(self.device)

            # Set task-specific parameters
            if task_type == "image":
                gen_kwargs["image_sizes"] = [img.size for img in visuals]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # Generate response
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    pad_token_id=pad_token_ids, 
                    images=image_tensor, 
                    **gen_kwargs
                )
            
            text_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            return text_output.strip()
            
        except Exception as e:
            eval_logger.error(f"Error during LLaVA-OneVision inference: {e}")
            return "Error processing input."