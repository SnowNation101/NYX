import argparse
import json
import os
import time

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from config import TASKS

from nyx.utils.process_gaokao import get_gaokao_bench_data, get_gaokao_mm_data
from nyx.utils.load_aokvqa import get_aokvqa_data
from nyx.generator import Generator
from nyx.retriever import EmebeddingRetriever


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--generator_path", type=str, required=True, help="Path to the generator model")
    parser.add_argument("--retrieval_path",type=str,required=True,help="Path to the retrieved data file",)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    args = parser.parse_args()
    
    assert (args.task in TASKS), f"Task {args.task} not supported. Supported tasks: {TASKS}"

    # Load knowledge base and test data
    if args.task == "gaokao_mm":
        _, test = get_gaokao_mm_data()
    elif args.task == "gaokao_bench":
        _, test = get_gaokao_bench_data()
    elif args.task == "aokvqa":
        _, test = get_aokvqa_data()

    assert os.path.exists(args.retrieval_path), f"Retrieval result {args.retrieval_path} does not exist."
    with open(args.retrieval_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    generator = Generator(args.generator_path)
    outputs = []

    total_samples = len(test)

    batch_size = args.batch_size

    for batch_start in tqdm(range(0, total_samples, batch_size), desc="Generating outputs"):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_queries = test[batch_start:batch_end]
        batch_results = results[batch_start:batch_end]

        batch_texts = []
        batch_images = []

        for query, result in zip(batch_queries, batch_results):
            if args.task == "gaokao_mm":
                for example in result['retrieved']:
                    qry_text = (
                        f"题目：{example['question']}\n"
                        f"答案：{example['answer']}\n"
                        f"解析：{example['analysis']}\n"
                        "你可以参考上面的例题来帮助你回答下面这一道高考选择题。请描述你的思考过程，并且给出最终答案（只需给出选项）。"
                        "你需要严格按照下面的格式来进行输出：并且注意包含尖括号<>。\n"
                        "<思考过程>：<<这里是你的思考过程>>\n"
                        "<答案>：<<这里直接给出选项（A/B/C/D）>>\n"
                        f"题目：{query['question']}"
                    )
                    example_images = [Image.open(p) for p in example["picture"]]
                    qry_images = [Image.open(p) for p in query["picture"]]
                    batch_texts.append(qry_text)
                    batch_images.append(example_images + qry_images)
            elif args.task == "aokvqa":
                for example in result['retrieved']:
                    qry_text = (
                        f"Question: {example['question']}\n"
                        f"Choices: {', '.join(example['choices'])}\n"
                        f"Answer: {example['choices'][example['correct_choice_idx']]}\n"
                        f"Analysis: {', '.join(example['rationales'])}\n"
                        "You can refer to the example above to help you answer the following question. "
                        "Please describe your thought process and provide the final answer (only the option). "
                        "You need to strictly follow the format below for output, and remember to include angle brackets <>.\n"
                        "<Thought Process>: <<Here is your thought process>>\n"
                        "<Answer>: <<Here is the option>>\n"
                        f"Question: {query['question']}\n"
                        f"Choices: {', '.join(query['choices'])}\n"
                    )
                    example_images = [Image.open(example['image_path'])]
                    qry_images = [Image.open(query['image_path'])]
                    batch_texts.append(qry_text)
                    batch_images.append(example_images + qry_images)

        batch_outputs = generator.generate(texts=batch_texts, images=batch_images)
        
        idx = 0
        for query, result in zip(batch_queries, batch_results):
            generated_result = []
            generated_result.extend(batch_outputs[idx:idx + len(result['retrieved'])])
            idx += len(result['retrieved'])
            outputs.append(generated_result)
            
    outputs_file_path = os.path.join("outputs/generations", f"generated_{time.strftime('%y%m%d_%H%M%S')}.json")
    with open(outputs_file_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False to allow for non-ASCII characters (e.g. Chinese)
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    print(f"Outputs saved to {outputs_file_path}")


if __name__ == "__main__":
    main()