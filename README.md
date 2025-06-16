# embodied-eval

## Why `embodied-eval`

## 🚀 TODO
- [ ] Open-EQA
- - [ ] EM-EQA 
- - [ ] A-EQA 
- [ ] GOAT-Bench

1. 数据和模型的绝对路径 --> 相对路径
2. ERQA --> tfrecore 2 huggingface 


## 🔨 Setup
1. Clone this repository & Install packages
```
git clone https://github.com/LongXinKou/embodied-eval.git
cd embodied-eval
conda create -n embodied-eval python==3.10
conda activate embodied-eval
pip install -r requirements.txt
```
2. Install llava & qwen_vl_utils
```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e . --no-deps # llava 1.7.0.dev0 

pip install qwen_vl_utils
```
3. RoboPoint
```
pip install sentencepiece
pip install protobuf
```

## 🎁 Benchmark
| Category               | Items     | Paper                                                                                                                                |
|------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------|
| VQA                    | EgoToM    | [EgoToM: Benchmarking Theory of Mind Reasoning from Egocentric Videos](https://arxiv.org/pdf/2503.22152)                             |
| VQA                    | VSI-Bench | [Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](https://arxiv.org/abs/2412.14171)         |
| VQA (planning)         | EgoPlan-Bench2 | [EgoPlan-Bench2: A Benchmark for Multimodal Large Language Model Planning in Real-World Scenarios](https://arxiv.org/abs/2412.04447) |
| VQA (planning)         | EgoPlan-Bench | [EgoPlan-Bench: Benchmarking Multimodal Large Language Models for Human-Level Planning](https://arxiv.org/pdf/2312.06722)            |
| VQA                    | VidEgoThink   | [VidEgoThink: Assessing Egocentric Video Understanding Capabilities for Embodied AI](https://arxiv.org/pdf/2410.11623)                                                                                                                                 |
| VQA                    | EgoThink | [EgoThink: Evaluating First-Person Perspective Thinking Capability of Vision-Language Models](https://arxiv.org/pdf/2311.15596)                                                            |
| A-EQA                  | EXPRESS-Bench | [Beyond the Destination: A Novel Benchmark for Exploration-Aware Embodied Question Answering](https://arxiv.org/pdf/2503.11117)      |
| A-EQA + VQA            | OpenEQA   | [OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://open-eqa.github.io/assets/pdfs/paper.pdf)             |
| A-EQA                  | Explore-EQA | [Explore until Confident: Efficient Exploration for Embodied Question Answering](https://arxiv.org/pdf/2403.15941)                   |
| EQA (plannig + action) | Embodied-Bench | [EMBODIEDBENCH: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents](https://arxiv.org/pdf/2502.09560) |

