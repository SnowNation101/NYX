# Multimodal Retriever for MRAG

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
    --task "gaokao_bench" \
    --retriever_path "outputs/models/Qwen2_mme5" \
    --index_path "outputs/indexes/gaokao_bench_index.faiss" \
    --retrieval_path "outputs/retrievals/retrieved_gaokao_bench_0508.json" \
    --top_k 10
```