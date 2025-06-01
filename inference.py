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

from src.utils.process_gaokao import get_gaokao_bench_data, get_gaokao_mm_data
from src.utils.load_aokvqa import get_aokvqa_data
from src.generator import Generator
from src.retriever import EmebeddingRetriever


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
        kb, test = get_gaokao_mm_data()
    elif args.task == "gaokao_bench":
        kb, test = get_gaokao_bench_data()
    elif args.task == "aokvqa":
        kb, _, test = get_aokvqa_data()

    assert os.path.exists(args.retrieval_path), f"Retrieval result {args.retrieval_path} does not exist."
    with open(args.retrieval_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    generator = Generator(args.generator_path)
    outputs = []

    for query, result in tqdm(
        zip(test, results), total=len(test), desc="Generating outputs"
    ):
        generated_result = []
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
                example_images = [Image.open(picture) for picture in example["picture"]]
                qry_images = [Image.open(picture) for picture in query["picture"]]
                qry_images = example_images + qry_images

                output_text = generator.generate(
                    text=qry_text, 
                    images=qry_images)
                generated_result.append(output_text[0])
        elif args.task == "gaokao_bench":
            # TODO: Add support for gaokao_bench
            qry_text = (
                f"题目：{result['question']}\n"
                f"答案：{result['answer']}\n"
                f"解析：{result['analysis']}\n"
                "你可以参考上面的例题来帮助你回答下面这一道高考选择题。请认真阅读并判断该题的类型，"
                "然后**严格按照下面对应的格式进行输出**。注意必须使用尖括号（<>）和双重尖括号（<<>>），格式不能有误。\n\n"
                "【题型与格式要求如下】\n"
                "一、如果是单选题：\n"
                "<思考过程>：<<这里是你的思考过程>>\n"
                "<答案>：<<这里直接给出选项（A/B/C/D）>>\n\n"
                "二、如果是多选题（多个正确答案）：\n"
                "<思考过程>：<<这里是你的思考过程>>\n"
                "<答案>：<<如ABD，不得有空格或逗号，必须包含所有正确选项>>\n\n"
                "三、如果是多个小题（如阅读理解）：\n"
                "<思考过程>：<<这里是你的思考过程>>\n"
                "1. <答案>：<<A/B/C/D>>\n"
                "2. <答案>：<<A/B/C/D>>\n"
                "3. <答案>：<<A/B/C/D>>\n"
                "……（按实际小题数继续编号）\n\n"
                "四、如果是七选五题型：\n"
                "<思考过程>：<<这里是你的思考过程>>\n"
                "1. <答案>：<<A/B/C/D/E/F/G>>\n"
                "2. <答案>：<<A/B/C/D/E/F/G>>\n"
                "3. <答案>：<<A/B/C/D/E/F/G>>\n"
                "4. <答案>：<<A/B/C/D/E/F/G>>\n"
                "5. <答案>：<<A/B/C/D/E/F/G>>\n\n"
                "请确保：先判断题型 → 然后严格按照格式输出，不得缺漏尖括号或格式标记。\n\n"
                f"题目：{query['question']}"
            )

            qry_images = None
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
                qry_images = example_images + qry_images

                output_text = generator.generate(
                    text=qry_text, 
                    images=qry_images)
                generated_result.append(output_text[0])
        outputs.append(generated_result)

    outputs_file_path = os.path.join("outputs/generations", f"generated_{time.strftime('%y%m%d_%H%M%S')}.json")
    with open(outputs_file_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False to allow for non-ASCII characters (e.g. Chinese)
        json.dump(outputs, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()