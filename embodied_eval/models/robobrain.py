import torch

from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from transformers import AutoProcessor, AutoModelForPreTraining
from typing import List, Optional, Tuple, Union
from loguru import logger as eval_logger

from embodied_eval.api.registry import register_model

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

    def generate_until(self, requests) -> List[str]:
        """Generate text until a stopping sequence."""
        res = []

        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="RoboBrain Responding")
