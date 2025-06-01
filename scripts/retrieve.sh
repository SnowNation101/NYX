#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5

# python3 retrieve.py \
#     --task "gaokao_mm" \
#     --base_model_path "/fs/archive/share/Qwen2.5-VL-7B-Instruct" \
#     --ckpt_path "checkpoint/ft_2025-05-26-1010.17/checkpoint-1094" \
#     --index_path "outputs/indexes/gaokao_mm_index.faiss" \
#     --output_path "outputs/retrievals/retrieved_gaokao_mm_0527.json" \
#     --top_k 10

python3 retrieve.py \
    --task "aokvqa" \
    --base_model_path "/fs/archive/share/Qwen2.5-VL-7B-Instruct" \
    --ckpt_path "checkpoint/ft_2025-05-26-1010.17/checkpoint-1094" \
    --build_index --index_path "outputs/indexes/aokvqa_index.faiss" \
    --output_path "outputs/retrievals/retrieved_aokvqa_0601.json" \
    --top_k 10