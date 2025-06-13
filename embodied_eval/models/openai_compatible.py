import base64
import os
import numpy as np
import time

from decord import VideoReader, cpu
from io import BytesIO
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Union

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator

from openai import OpenAI


@register_model("openai_compatible")
class OpenAICompatible(BaseAPIModel):
    def __init__(
            self,
            model_name_or_path: str = "gpt-4o",
            timeout: int = 10,
            max_retries: int = 5,
            batch_size: Optional[Union[int, str]] = 1,
            max_new_tokens: int = 1024,
            temperature: float = 0,
            do_sample: bool = False,
            top_p: Optional[int] = None,
            num_beams: int = 1,
            system_prompt: Optional[str] = None,
            **kwargs,
    ) -> None:
        super().__init__()

        self.client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"), 
            base_url = os.getenv("OPENAI_API_BASE")
        )

        # Store configuration
        self.model_name_or_path = model_name_or_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.num_beams = num_beams
        self.system_prompt = system_prompt
    
    
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
        
        batches = [reg.args for reg in requests]
        progress_bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for batch in batches:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*batch)
            task = task[0]
            split = split[0]

            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) if split is not None 
                           else doc_to_visual[0](self.task_dict[task][ids]) for ids in doc_id]
            if None in visual_list:
                visual_list = []
                imgs = []
            else:
                visual_list = self.flatten(visual_list)
                imgs = []  # multiple images or frames for video
                for visual in visual_list:
                    if isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                        img = self.encode_image(visual)
                        imgs.append(img)
                    elif isinstance(visual, Image.Image):
                        img = self.encode_image(visual)
                        imgs.append(img)
                    elif isinstance(visual, str) and (".mp4" in visual or ".avi" in visual):
                        frames = self.encode_video(visual, self.max_frames_num)
                        imgs.extend(frames)
                    elif isinstance(visual, list):
                        frames = self.encode_video(visual, self.max_frames_num)
                        imgs.extend(frames)
            
            payload = {"messages": []}
            payload["model"] = self.model_name_or_path
            payload["messages"].append({"role": "user", "content": []})
            payload["messages"][0]["content"].append({"type": "text", "text": contexts})
            for img in imgs:
                payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

            gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
            do_sample = gen_kwargs.get("do_sample", self.do_sample)
            temperature = gen_kwargs.get("temperature", self.temperature)
            top_p = gen_kwargs.get("top_p", self.top_p)
            num_beams = gen_kwargs.get("num_beams", self.num_beams)

            payload["max_tokens"] = max_new_tokens
            payload["temperature"] = temperature

            if "o1" in self.model_version or "o3" in self.model_version:
                del payload["temperature"]
                payload["reasoning_effort"] = "medium"
                payload["response_format"] = {"type": "text"}
                payload.pop("max_tokens")
                payload["max_completion_tokens"] = gen_kwargs["max_tokens"]
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(**payload)
                    response_text = response.choices[0].message.content
                except Exception as e:
                    error_msg = str(e)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")

                    # On last attempt, log error and set empty response
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                        response_text = ""
                    else:
                        time.sleep(self.timeout)
            res.append(response_text)
            progress_bar.update(1)

        progress_bar.close()
        return res


    def encode_image(self, image: Union[Image.Image, str]):
        max_size = self.max_size_in_mb * 1024 * 1024  # 20MB in bytes
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        # If image is too large, resize it while maintaining aspect ratio
        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str
    
    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        if isinstance(video_path, str):
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

            # Ensure the last frame is included
            if total_frame_num - 1 not in uniform_sampled_frames:
                uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

            frame_idx = uniform_sampled_frames.tolist()
            frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames