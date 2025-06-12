<div align="center">
    <img src="https://github.com/SnowNation101/NYX/blob/main/assets/Nyx.webp" alt="Nyx Logo" style="width: 35%;" />
</div>

<h1 align="center"> 🌓 Nyx: Unified Multimodal Retriever for MRAG </a></h1>

<div align="center"> 

<img alt="Arxiv Paper" src="https://img.shields.io/badge/paper-arXiv-b5212f.svg?logo=arxiv">
<img alt="GitHub License" src="https://img.shields.io/github/license/SnowNation101/Nyx?color=lightgreen">
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SnowNation101/Nyx?style=flat&logo=github&color=lightblue">

</div>

**This project is still under development and is not yet the final release version.**

## TODO

- [ ] Get feedback
- [ ] Train with feedback
- [ ] Batch indexing (encode)

## Prepare

We recommend using **Conda** for package management.

```bash
conda create -n nyx python=3.11
conda activate nyx
pip install -r requirements.txt
```

Our implementation uses `torch==2.4.0`, `faiss-cpu==1.8.0`, and `transformers==4.52.2`. Please note that `faiss-cpu` and `transformers` might have `numpy` version conflicts. We prefer keeping `numpy` at version `1.26.4` (the version compatible with `faiss-cpu`), so you may need to uninstall any newer `numpy` versions.


tips: install pytorch -> faiss-cpu -> transformers -> accelerate -> deepspeed



## Running Scripts

```bash
#!/bin/bash

python3 eval_gaokao_bench.py \
    --task "gaokao_bench" \
    --generation_path "outputs/generations/generated_250415_180512.json"
```

```bash
#!/bin/bash

python3 inference.py \
    --task "aokvqa" \
    --generator_path "/fs/archive/share/Qwen2.5-VL-7B-Instruct" \
    --retrieval_path "outputs/retrievals/retrieved_aokvqa_0601.json" \

```

```bash
#!/bin/bash
python3 retrieve.py \
    --task "aokvqa" \
    --base_model_path "/fs/archive/share/Qwen2.5-VL-7B-Instruct" \
    --ckpt_path "checkpoint/ft_2025-05-26-1010.17/checkpoint-1094" \
    --build_index --index_path "outputs/indexes/aokvqa_index.faiss" \
    --output_path "outputs/retrievals/retrieved_aokvqa_0601.json" \
    --top_k 10
```