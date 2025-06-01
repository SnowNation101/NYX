# Multimodal Retriever for MRAG

## TODO

- [ ] Get feedback
- [ ] Train with feedback
- [ ] Batch indexing (encode)

## Prepare

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
    --task "gaokao_bench" \
    --generator_path "/fs/archive/share/Qwen2_5-VL-7B-Instruct" \
    --retriever_path "outputs/models/Qwen2_mme5" \
    --index_path "outputs/indexes/gaokao_bench_index.faiss" \
    --retrieval_path "outputs/retrievals/retrieved_gaokao_bench.json" \

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

