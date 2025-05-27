import argparse
import json
import os
import time

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from process_gaokao import get_gaokao_bench_data, get_gaokao_mm_data
# from src.generator import Generator
from src.retriever import EmebeddingRetriever

TASKS = [
    "gaokao_mm",
    "gaokao_bench",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the retriever model")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the faiss index file")
    parser.add_argument("--build_index", action="store_true", help="Force build of FAISS index")
    parser.add_argument("--output_path",type=str,required=True,help="Path to the retrieved data file",)
    parser.add_argument("--top_k", type=int, default=1, help="Number of top results to retrieve")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the base model(for loading processor)")
    args = parser.parse_args()

    assert (args.task in TASKS), f"Task {args.task} not supported. Supported tasks: {TASKS}"

    # Load knowledge base and test data
    if args.task == "gaokao_mm":
        kb, test = get_gaokao_mm_data()
    elif args.task == "gaokao_bench":
        kb, test = get_gaokao_bench_data()

    retriever = EmebeddingRetriever(
        ckpt_path=args.ckpt_path,
        base_model_path=args.base_model_path,
        task=args.task,
        index=faiss.read_index(args.index_path),
    )

    all_data = []

    top_k = args.top_k
    for query in tqdm(test, desc="Retrieving data"):
        if args.task == "gaokao_mm":
            qry_text = f"<|im_start|>请根据下面的高考题目检索出最相关的题目。\n{query['question']}<|im_end|>"
            qry_image_paths = query['picture']
            qry_images = [Image.open(picture) for picture in qry_image_paths]
            D, I = retriever.retrieve(
                qry_text=qry_text, 
                qry_images=qry_images, 
                top_k=top_k)
        elif args.task == "gaokao_bench":
            qry_text = f"<|im_start|>请根据下面的高考题目检索出最相关的题目。\n{query['question']}<|im_end|>"
            D, I = retriever.retrieve(
                qry_text=qry_text, 
                qry_images=None, 
                top_k=top_k)
        entry = {
            "target": query,
            "retrieved": [],
        }
        for i in range(top_k):
            kb[I[0][i]]["distance"] = float(D[0][i])
            entry["retrieved"].append(kb[I[0][i]])

        all_data.append(entry)
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print(f"Retrieval results saved to {args.output_path}")

if __name__ == "__main__":
    main()