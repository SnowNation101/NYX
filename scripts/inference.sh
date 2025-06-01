#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3,4,5

python3 inference.py \
    --task "aokvqa" \
    --generator_path "/fs/archive/share/Qwen2.5-VL-7B-Instruct" \
    --retrieval_path "outputs/retrievals/retrieved_aokvqa_0601.json" \
    --batch_size 8