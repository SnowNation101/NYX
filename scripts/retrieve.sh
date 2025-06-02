#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

python3 retrieve.py \
    --task "aokvqa" \
    --base_model_path "/fs/archive/share/Qwen2.5-VL-7B-Instruct" \
    --ckpt_path "checkpoint/ft_2025-05-26-1010.17/checkpoint-1094" \
    --index_path "outputs/indexes/aokvqa_index.faiss" \
    --output_path "outputs/retrievals/retrieved_aokvqa_0602.json" \
    --top_k 10