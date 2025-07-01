
## 🔨 Setup

### flash-attention
```
# Directly
pip install flash-attn --no-build-isolation

# [Option] install the per-commit wheel built by that PR, "https://github.com/Dao-AILab/flash-attention/releases"
pip install flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### LLaVA-Next
```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e . --no-deps # llava 1.7.0.dev0 
```

### VILA
need to install `llava` from VILA repo instead of LLaVA or LLaVA-Next.
```
pip uninstall llava
git clone https://github.com/NVlabs/VILA.git
cd VILA
pip install -e . --no-deps # vila-2.0.0
```
(1) cannot import name 'Qwen2FlashAttention2' from 'transformers.models.qwen2.modeling_qwen2'
```
pip install transformers==4.46.0
```
(2) Error during evaluation: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!
add `chat_template` to tokenizer_config.json of VILA1.5.
```json
"chat_template": "{% if messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{% for message in messages if message['content'] is not none %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
"chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
```

### robopint
```
git clone https://github.com/wentaoyuan/RoboPoint.git
cd RoboPoint
pip install -e . --no-deps 
```

### physvlm
```
git clone https://github.com/unira-zwj/PhysVLM.git
cd PhysVLM/physvlm-main
pip install -e . --no-deps # physvlm-1.1.0 
```

### embodiedgpt
```
git clone https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch.git
cd EmbodiedGPT_Pytorch
pip install -e . --no-deps # robohusky-0.1.0
```
(1) ImportError: cannot import name 'is_flash_attn_available' from 'transformers.utils'
```python
from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
```
(2) Error during evaluation: wo was specified in the `_keep_in_fp32_modules` list, but is not part of the modules in HuskyVisionModel.
modeling_husky_embody2.py(line 392): 
```python
# _keep_in_fp32_modules = ["wo"]
```
(3) Error during evaluation: 'HuskyQFormerFlashAttention2' object has no attribute 'embed_size'.
replace embed_size with embed_dim in modeling_husky_embody2.py line 838.
```python
context_layer = attn_output.reshape(bsz, tgt_len, embed_dim).contiguous()
```