import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModelForPreTraining
from typing import List, Optional, Union
from loguru import logger as eval_logger

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator

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
            batch_size: Optional[Union[int, str]] = 1,
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
        self.batch_size_per_gpu = int(batch_size)

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
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device
    
    def flatten(self, input):
        """Helper method to flatten a list of lists into a single list."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        """Generate text until a stopping sequence."""
        res = []

        def _sort_by_context_length(x):
            # Sort by context length for better batching
            toks = self.processor.tokenizer.encode(x[0])
            return -len(toks), x[0]

        # Initialize the Collator to group requests and sort them by context length
        collator = Collator([req.args for req in requests], _sort_by_context_length, grouping=True)
        # Create batches from the sorted and grouped requests
        batches = collator.get_batched(n=self.batch_size, batch_fn=None)

        # Determine the number of iterations required to process all requests
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        progress_bar = tqdm(total=num_iters, disable=(self.rank != 0), desc="RoboBrain Responding")

        for batch in batches:
            batch_contexts, all_gen_kwargs, batch_doc_to_visual, batch_doc_id, batch_task, batch_split = zip(*batch)
            task = batch_task[0]
            split = batch_split[0]
            gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}

            # Define a stopping sequence if specified
            until = [self.processor.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until_from_kwargs = gen_kwargs.pop("until")
                if isinstance(until_from_kwargs, str):
                    until = [until_from_kwargs]

            batch_visuals = [batch_doc_to_visual[0](self.task_dict[task][split][ids]) if split is not None 
                           else batch_doc_to_visual[0](self.task_dict[task][ids]) for ids in batch_doc_id]
            
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

            # Input (input_ids, attention_mask)
            inputs_list = []
            input_lengths = []
            for visual, context in zip(batch_visuals, batch_contexts): 
                # For image
                if isinstance(visual[0], Image.Image):
                    task_type = "image"
                    if DEFAULT_IMAGE_TOKEN not in context:
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual)
                        context = f"{image_tokens}\n{context}"
                    messages = [{"role": "user", "content": context}]
                    if self.processor.tokenizer.chat_template is not None:
                        text = self.processor.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    
                    images = [self._process_image(image) for image in visual]
                    inputs = self.processor(
                        images=images,
                        text=text,
                        return_tensors="pt"
                    ).to(self.device, self.model.dtype)
                elif isinstance(visual[0], str):
                    task_type = "video"
                    frames = load_video(visual, self.max_frames_num)

                    if DEFAULT_IMAGE_TOKEN not in context:
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * len(frames)
                        context = f"{image_tokens}\n{context}"
                    messages = [{"role": "user", "content": context}]
                    if self.processor.tokenizer.chat_template is not None:
                        text = self.processor.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    
                    images = [self._process_image(frame) for frame in frames]
                    inputs = self.processor(
                        images=images,
                        text=text,
                        return_tensors="pt"
                    ).to(self.device, self.model.dtype)
                inputs_list.append(inputs)
                input_lengths.append(inputs['input_ids'].shape[1])
                
            batch_inputs = {}
            for key in inputs_list[0].keys():
                batch_inputs[key] = torch.cat([inp[key] for inp in inputs_list], dim=0)
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            # Generate output
            # import inspect
            # print(inspect.signature(self.model.generate))
            output = self.model.generate(
                **batch_inputs,
                **gen_kwargs
            )

            generated_ids_trimmed = [out_ids[in_ids:] for in_ids, out_ids in zip(input_lengths, output)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )
            
            # Apply stopping sequence if needed
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            # Cache result
            for ans, context in zip(answers, batch_contexts):
                res.append(ans)
                progress_bar.update(1)

        # Reorder results to match original order
        res = collator.get_original(res)
        progress_bar.close()
        return res

    def _process_image(self, image):
        """Process images for model input."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, (np.ndarray)):
            return Image.fromarray(image).convert("RGB")
        else:
            return None
        