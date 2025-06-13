import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator, DistributedType
from transformers import AutoProcessor, AutoModelForPreTraining
from typing import List, Optional, Union
from loguru import logger as eval_logger

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator

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
                elif isinstance(until_from_kwargs, list):
                    until = until_from_kwargs

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
            input_ids_list, attention_mask_list = [], []
            input_lengths = []
            for visual, context in zip(batch_visuals, batch_contexts): 
                if self.system_prompt:
                    message = [{"role": "system", "content": self.system_prompt}]
                else:
                    message = []
                
                content_parts = []
                # For context   
                content_parts.append({"type": "text", "text": context})

                # For image, multi-image 
                if isinstance(visual, str) and (visual.startswith("http")):
                    content_parts.append({"type": "image", "url": visual})
                elif isinstance(visual, (str, Image.Image, np.ndarray)):
                    content_parts.append({"type": "image", "image": self._process_image(visual)})
                
                message.append({"role": "user", "content": content_parts})

                inputs = self.processor.apply_chat_template(
                    message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                )
                input_ids_list.append(inputs['input_ids'])
                attention_mask_list.append(inputs.get('attention_mask',inputs['input_ids'].ne(self.processor.tokenizer.pad_token_id)))
                input_lengths.append(inputs['input_ids'].shape[1])

            input_ids = torch.cat(input_ids_list, dim=0).to(self.device)
            attention_mask = torch.cat(attention_mask_list, dim=0).to(self.device)

            # Generate output
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
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
        """Process image for model input."""
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                eval_logger.error(f"Error loading image {image}: {e}")
                return None
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return None