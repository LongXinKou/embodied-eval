'''
modified from
"https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch/blob/main/demo/inference.py"
'''
import numpy as np
import torch
import torchvision.transforms as T


from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from packaging import version
from peft import PeftModel
from PIL import Image
from typing import List, Optional, Union
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from loguru import logger as eval_logger

from transformers import (
    LlamaTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from robohusky.model.modeling_husky_embody2 import HuskyForConditionalGeneration

from robohusky.conversation import (
    conv_templates,
    get_conv_template,
)

from robohusky.video_transformers import (
    GroupNormalize,
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    get_index,
)

DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"

DEFAULT_VIDEO_START_TOKEN = "<vid>"
DEFAULT_VIDEO_END_TOKEN = "</vid>"

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops, encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        device = input_ids.device
        for stop in self.stops:
            stop = stop.to(device)
            stop_len = len(stop)
            for seq in input_ids:
                if stop_len <= len(seq):
                    if torch.all(stop == seq[-stop_len:]).item():
                        return True
        return False


@register_model("embodiedgpt")
class EmbodiedGPT(BaseAPIModel):
    """
    EmbodiedGPT Model
    "https://huggingface.co/Liang-ZX/Embodied_family_7b"
    """

    def __init__(
            self,
            model_name_or_path: str = "Liang-ZX/Embodied_family_7b",
            lora_weights: str = None,
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
            model_name: Optional[str] = "EmbodiedGPT",
            max_num_frames: Optional[int] = 8,
            conv_template: Optional[str] = "husky",
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
        kwargs["torch_dtype"] = kwargs.get("torch_dtype", torch.float16)

        eval_logger.info(f"Loading EmbodiedGPT model from {model_name_or_path}")
        self._tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        if lora_weights is None:
            self._model = HuskyForConditionalGeneration.from_pretrained(
                model_name_or_path, 
                device_map=self.device_map,
                low_cpu_mem_usage=True, 
                **kwargs
            )
        else:
            kwargs["device_map"] = self.device_map
            self._model = HuskyForConditionalGeneration.from_pretrained(
                model_name_or_path, 
                device_map=self.device_map,
                low_cpu_mem_usage=True, 
                **kwargs
            )
            self._model.language_model = PeftModel.from_pretrained(
                self._model.language_model,
                lora_weights,
                **kwargs
            )
        self._model.eval()

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
        self.conv_template = conv_template
        self.max_num_frames = max_num_frames

        self.conv = get_conv_template(conv_template)
        self.image_query = DEFAULT_IMG_START_TOKEN + DEFAULT_IMG_END_TOKEN
        self.video_query = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_END_TOKEN

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
    
    def load_image(self, image, input_size=224):
        crop_pct = 224 / 256
        size = int(input_size / crop_pct)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        image = transform(image)
        return image

    def load_video(self, video_path, num_segments=8):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments)

        # transform
        crop_size = 224
        scale_size = 224
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]

        transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std)
        ])

        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        video = transform(images_group)
        return video
    
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

        # Process visual inputs
        if isinstance(visuals[0], Image.Image):
            modal_type = "image"
            pixel_values = [self.load_image(img) for img in visuals]
            # Limit to max_num_frames
            if len(pixel_values) > self.max_num_frames:
                pixel_values = pixel_values[:self.max_num_frames]
            pixel_values = torch.cat(pixel_values, dim=0)
            if len(visuals) > 1:
                TC, H, W = pixel_values.shape
                pixel_values = pixel_values.reshape(TC // 3, 3, H, W).transpose(0, 1)
        elif isinstance(visuals[0], str):
            modal_type = "video"
            pixel_values = self.load_video(visuals[0], self.max_num_frames)
            TC, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape(TC // 3, 3, H, W).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported visual type: {type(visuals[0])}")

        pixel_values = pixel_values.unsqueeze(0).to(dtype=torch.float16, device=self.device)
        language_model_inputs = self.model.extract_feature(pixel_values)

        # Format the question with visual tokens
        if modal_type == "image":
            question = self.image_query + "\n" + context
        elif modal_type == "video":
            question = self.video_query + "\n" + context
        else:
            question = context
            
        # Prepare conversation
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        conversation = conv.get_prompt()
        
        # Tokenize input
        model_inputs = self.tokenizer(
            [conversation],
            return_tensors="pt",
        )
        model_inputs.pop("token_type_ids", None)

        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        # Set up stopping criteria
        stop_words = ["Human: ", "Assistant: ", "###", "\n\n"]
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # Generate response
        try:
            with torch.inference_mode():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    language_model_inputs=language_model_inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **gen_kwargs,
                )
            
            preds = generation_output.sequences
            text_output = self.tokenizer.batch_decode(preds, skip_special_tokens=True)[0]
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            return text_output.strip()
            
        except Exception as e:
            eval_logger.error(f"Error during EmbodiedGPT inference: {e}")
            return "Error processing input."