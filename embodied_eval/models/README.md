
## 🔨 Setup

### llava
```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e . --no-deps # llava 1.7.0.dev0 
```

### VILA
```
git clone https://github.com/NVlabs/VILA.git
cd VILA
pip install -e . --no-deps # vila-2.0.0
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
