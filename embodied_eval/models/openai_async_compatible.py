import base64
import os
import numpy as np
import time
import httpx

from decord import VideoReader, cpu
from io import BytesIO
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Union

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel

import asyncio
from openai import AsyncOpenAI

@register_model("openai_async_compatible")
class OpenAIAsyncCompatible(BaseAPIModel):
    def __init__(
            self,
            model_name_or_path: str = "gpt-4o",
            batch_size: Optional[Union[int, str]] = 1,
            max_new_tokens: int = 4096,
            temperature: float = 0,
            do_sample: bool = False,
            top_p: Optional[float] = None,
            num_beams: int = 1,
            system_prompt: Optional[str] = None,
            max_frames_num: int = 16,
            timeout: int = 10,
            max_retries: int = 1,
            max_size_in_mb: int = 20,
            **kwargs,
    ) -> None:
        super().__init__()

        self.async_client = AsyncOpenAI(
            api_key = os.getenv("OPENAI_API_KEY"), 
            base_url = os.getenv("OPENAI_API_BASE")
        )

        # Store configuration
        self.model_name_or_path = model_name_or_path
        self.batch_size = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.num_beams = num_beams
        self.system_prompt = system_prompt

        self.max_frames_num = max_frames_num
        self.max_size_in_mb = max_size_in_mb
        self.timeout = timeout
        self.max_retries = max_retries
        self.sema = asyncio.Semaphore(5)


    def generate_until(self, requests) -> List[str]:
        progress_bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        async def _batch_async():
            tasks = []
            for i, reg in enumerate(requests):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = reg.args
                # 把 index 传入 task 中
                coro = self._answer_async(contexts, gen_kwargs, doc_to_visual, doc_id, task, split)
                tasks.append((i, asyncio.create_task(coro)))

            results = [None] * len(tasks)
            for i, fut in tasks:
                result = await fut
                results[i] = result
                progress_bar.update(1)

            return results

        results = list(asyncio.run(_batch_async()))
        progress_bar.close()
        return results

    async def _answer_async(self, contexts, gen_kwargs, doc_to_visual, doc_id, task, split):
        visual_list = doc_to_visual(self.task_dict[task][split][doc_id]) if split is not None else doc_to_visual(self.task_dict[task][doc_id])

        visual_indices = []
        imgs = []

        has_index = (
            isinstance(visual_list, (list, tuple)) and
            len(visual_list) == 2 and
            all(isinstance(img, Image.Image) for img in visual_list[0]) and
            all(isinstance(i, int) for i in visual_list[1])
        )
        # has_index = False

        if has_index:
            imgs.extend(visual_list[0])
            visual_indices.extend(visual_list[1])
        else:
            for visual in visual_list:
                if isinstance(visual, Image.Image):
                    img = await asyncio.to_thread(self.encode_image, visual)
                    imgs.append(img)
                elif isinstance(visual, str) and (".mp4" in visual or ".avi" in visual):
                    frames = await asyncio.to_thread(self.encode_video, visual, self.max_frames_num)
                    imgs.extend(frames)

        payload = {
            "model": self.model_name_or_path,
            "messages": []
        }
        if self.system_prompt:
            payload["messages"].append({
                "role": "system", 
                "content": {"type": "text", "text": self.system_prompt}
            })

        content = self.build_message_content(
            question=contexts,
            pil_images=imgs,
            visual_indices=visual_indices
        )

        payload["messages"].append({
            "role": "user",
            "content": content
        })

        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        do_sample = gen_kwargs.get("do_sample", self.do_sample)
        temperature = gen_kwargs.get("temperature", self.temperature)
        top_p = gen_kwargs.get("top_p", self.top_p)
        num_beams = gen_kwargs.get("num_beams", self.num_beams)
        payload["max_tokens"] = max_new_tokens
        payload["temperature"] = temperature

        if "2.5-pro" in self.model_name_or_path:
            payload["max_tokens"] = self.max_new_tokens

        if "o1" in self.model_name_or_path or "o3" in self.model_name_or_path or "o4" in self.model_name_or_path:
            del payload["temperature"]
            payload["reasoning_effort"] = "medium"
            payload["response_format"] = {"type": "text"}
            payload.pop("max_tokens")
            payload["max_completion_tokens"] = self.max_new_tokens
        
        
        for attempt in range(self.max_retries):
            try:
                async with self.sema:
                    response = await self.async_client.chat.completions.create(**payload)
                response_text = response.choices[0].message.content
                break
            except Exception as e:
                error_msg = str(e)
                eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")

                # On last attempt, log error and set empty response
                if attempt == self.max_retries - 1:
                    response_text = ""
                    eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                else:
                    await asyncio.sleep(0.5)
        return response_text


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
    
    def build_message_content(
        self, 
        question: str, 
        pil_images: List[Image.Image], 
        visual_indices: List[int]
    ) -> List[dict]:
        def is_base64_encoded(s):
            try:
                return s.rstrip('=') == base64.b64encode(base64.b64decode(s, validate=True)).decode("utf-8").rstrip('=')
            except Exception:
                return False
        contents = []
        if len(visual_indices) == 0 or all(idx == 0 for idx in visual_indices):
            contents.extend(pil_images)
            contents.append(question)
        else:
            image_index_pairs = list(zip(pil_images, visual_indices))
            image_index_pairs.sort(key=lambda x: x[1])
            last_pos = 0
            for img, idx in image_index_pairs:
                if idx == 0:
                    contents.append(img)
                elif idx <= len(question):
                    text_segment = question[last_pos:idx]
                    if text_segment:
                        contents.append(text_segment)
                    contents.append(img)
                    last_pos = idx
                else:
                    contents.append(img)
            if last_pos < len(question):
                contents.append(question[last_pos:])
            if not contents:
                contents.append(question)
                contents.extend(img for img, _ in image_index_pairs)

        # Convert to OpenAI-style interleaved content
        interleaved = []
        for item in contents:
            if isinstance(item, str):
                if is_base64_encoded(item):
                    interleaved.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{item}"}
                    })
                else:
                    interleaved.append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                base64_img = self.encode_image(item)
                interleaved.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })
        return interleaved