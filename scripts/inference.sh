#!/bin/bash

python3 inference.py \
    --task "gaokao_mm" \
    --generator_path "/fs/archive/share/Qwen2.5-VL-7B-Instruct" \
    --retriever_path "outputs/models/Qwen2_mme5" \
    --index_path "outputs/indexes/gaokao_bench_index.faiss" \
    --retrieval_path "outputs/retrievals/retrieved_gaokao_mm_0527.json" \
