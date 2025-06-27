'''
modified from
"https://github.com/NVlabs/VILA/blob/main/server.py"
'''
import numpy as np
import torch
import torchvision.transforms as T
import copy

from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from packaging import version
from peft import PeftModel
from PIL import Image
from typing import List, Optional, Union
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from loguru import logger as eval_logger

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

from llava.conversation import SeparatorStyle, conv_templates, CONVERSATION_MODE_MAPPING
from llava.constants import MEDIA_TOKENS
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.media import Video

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator


@register_model("vila")
class VILA(BaseAPIModel):
    """
    VILA Model
    "https://github.com/NVlabs/VILA"
    """

    def __init__(
        self,
        model_name_or_path: str = "Efficient-Large-Model/VILA1.5-13b",
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
        model_name: Optional[str] = None,
        conv_mode="auto",
        max_num_frames: Optional[int] = 8,
        fps: Optional[float] = 0.0,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),
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
        llava_model_args = {
            "attn_implementation": attn_implementation,
            "torch_dtype": "float16",
        }

        eval_logger.info(f"Loading VILA1.5 model from {model_name_or_path}")
        model_name = model_name if model_name is not None else get_model_name_from_path(model_name_or_path)
        eval_logger.info(f"Model name is {model_name}")
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            model_path=model_name_or_path,
            model_base=None,
            model_name=model_name, 
            device_map=self.device_map, 
            **llava_model_args
        )
        self.model.image_processor = self._image_processor
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
        self.conv_mode = conv_mode if conv_mode is not None else CONVERSATION_MODE_MAPPING[model_name.lower()]
        self.conv = conv_templates[self.conv_mode]
        self.max_num_frames = max_num_frames
        self.fps = fps

        self.image_query = MEDIA_TOKENS["image"]
        self.video_query = MEDIA_TOKENS["video"]

        # Set up distributed evaluation
        if accelerator.num_processes > 1:
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
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
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = np.linspace(0, total_frame_num - 2, max_frames_num, dtype=int)
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            return [Image.fromarray(img) for img in spare_frames]
        except Exception as e:
            eval_logger.error(f"Failed to load video {video_path} with error: {e}")
            return [Image.new("RGB", (448, 448), (0, 0, 0))] * max_frames_num
    
    def generate_until(self, requests) -> List[str]:
        """Generate text until a stopping sequence."""
        res = []
        progress_bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if split is not None:
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            else:
                visuals = doc_to_visual(self.task_dict[task][doc_id])
            if isinstance(visuals, tuple):
                visuals = visuals[0]

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
            
            generation_config = copy.deepcopy(self.model.default_generation_config)
            generation_config.max_new_tokens = gen_kwargs["max_new_tokens"]
            generation_config.temperature = gen_kwargs["temperature"]
            generation_config.top_p = gen_kwargs["top_p"]
            generation_config.do_sample = gen_kwargs["do_sample"]
            generation_config.num_beams = gen_kwargs["num_beams"]
            generation_config.use_cache = gen_kwargs["use_cache"]

            conv = self.conv.copy()
            if isinstance(visuals[0], Image.Image):
                task_type = "image"
                images = visuals
                question = f"{self.image_query}\n {contexts}"
            elif isinstance(visuals[0], str):
                task_type = "video"
                video = visuals[0]
                video = Video(video)
                self.model.config.fps = self.max_num_frames
                self.model.config.fps = self.fps
                question = f"{self.video_query}\n {contexts}"

            if task_type == "image":
                prompt = images + [question]
            elif task_type == "video":
                prompt = [video, question]
            else:
                prompt = question

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            try:
                with torch.inference_mode():
                    outputs = self.model.generate_content(
                        prompt=prompt, 
                        generation_config=generation_config
                    )
                    if outputs.endswith(stop_str):
                        outputs = outputs[: -len(stop_str)]
            except Exception as e:
                raise e

            text_outputs = [outputs.strip()]
            res.extend(text_outputs)
            progress_bar.update(1)

        # Reorder results to match original order
        progress_bar.close()
        return res