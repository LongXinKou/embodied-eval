import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator, DistributedType
from transformers import AutoProcessor, AutoModelForPreTraining
from typing import List, Optional, Tuple, Union
from loguru import logger as eval_logger

from embodied_eval.api.registry import register_model
from embodied_eval.utils import Collator

@register_model("robobrain")
class RoboBrain(BaseAPIModel):
    """
    RoboBrain Model
    "https://huggingface.co/BAAI/RoboBrain"
    """

    def __init__(
            self,
            pretrained: str = "BAAI/RoboBrain",
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "cuda",
            temperature: float = 0.7,
            do_sample: bool = True,
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
        eval_logger.info(f"Loading RoboBrain model from {pretrained}")
        self._model = AutoModelForPreTraining.from_pretrained(
            pretrained,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device_map
        )

        # Load processor
        self._processor = AutoProcessor.from_pretrained(pretrained)

        # Store configuration
        self.temperature = temperature
        self.do_sample = do_sample

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
    def processor(self):
        return self._processor

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    def generate_until(self, requests: List[Instance]) -> List[str]:
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
        num_batches = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        progress_bar = tqdm(total=num_batches, disable=(self.rank != 0), desc="RoboBrain Responding")

        for batch in batches:

            gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}

            # Get generation parameters
            do_sample = gen_kwargs.get("do_sample", self.do_sample)
            temperature = gen_kwargs.get("temperature", self.temperature)
            max_new_tokens = gen_kwargs.get("max_new_tokens", 256)

            # Define a stopping sequence if specified
            until = [self.processor.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until_from_kwargs = gen_kwargs.pop("until")
                if isinstance(until_from_kwargs, str):
                    until = [until_from_kwargs]
                elif isinstance(until_from_kwargs, list):
                    until = until_from_kwargs

            contexts = list(contexts)

            batched_messages = []
            for i, context in enumerate(contexts):
                if self.system_prompt:
                    message = [{"role": "system", "content": self.system_prompt}]
                else:
                    message = []

                content_parts = []
                content_parts.append({"type": "text", "text": context})
                if i < len(visual_list) and visual_list[i] is not None:
                    visual = visual_list[i]
                    if isinstance(visual, str) and (visual.startswith("http")):
                        content_parts.append({"type": "image", "url": visual})
                    elif isinstance(visual, (str, Image.Image, np.ndarray)):
                        content_parts.append({"type": "image", "image": self._process_image(visual)})

                message.append({"role": "user", "content": content_parts})
                batched_messages.append(message)

            # Process all messages in batch
            inputs_list = []
            input_lengths = []
            for message in batched_messages:
                inputs = self.processor.apply_chat_template(
                    message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                )
                # Add attention mask if it's missing
                inputs["attention_mask"] = inputs.get('attention_mask',
                                                      inputs['input_ids'].ne(self.processor.tokenizer.pad_token_id))

                inputs_list.append(inputs)
                input_lengths.append(inputs['input_ids'].shape[1])

            batch_inputs = {}
            for key in inputs_list[0].keys():
                batch_inputs[key] = torch.cat([inp[key] for inp in inputs_list], dim=0)
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            # Generate output
            output = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature
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
            for ans, context in zip(answers, contexts):
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