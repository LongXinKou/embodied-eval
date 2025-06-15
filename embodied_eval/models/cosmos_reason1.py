import base64
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
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device_map
        )

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
        progress_bar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Cosmos-Reason1-7B Responding")

        for batch in batches:
            batch_contexts, all_gen_kwargs, batch_doc_to_visual, batch_doc_id, batch_task, batch_split = zip(*batch)
            task = batch_task[0]
            split = batch_split[0]
            gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}

            until = [self.processor.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                
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

            batch_visuals = [batch_doc_to_visual[0](self.task_dict[task][split][ids]) if split is not None 
                           else batch_doc_to_visual[0](self.task_dict[task][ids]) for ids in batch_doc_id]
            
            batched_messages = []
            for visual_list, context in zip(batch_visuals, batch_contexts): 
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
                
                processed_visuals = []
                for visual in visual_list:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        processed_visuals.append({
                            "type": "video", 
                            "video": visual, 
                        })
                    elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append({
                            "type": "image", 
                            "image": f"data:image/jpeg;base64,{base64_string}", 
                        })
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]

            inputs = self.processor(
                text=texts, 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)

            output = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans
            
            text_outputs = [response.strip() for response in answers]
            res.extend(text_outputs)
            progress_bar.update(1)

        # Reorder results to match original order
        res = collator.get_original(res)
        progress_bar.close()
        return res
