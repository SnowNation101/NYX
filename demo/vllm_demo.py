# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""

import os
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

question = "What is the content of each image?"
images = [
        Image.open("/fs/archive/share/mm_datasets/GAOKAO-MM/Data/2010-2023_Chemistry_MCQs/2010-2023_Chemistry_MCQs_52_0.png") for _ in range(5)
    ]

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    llm_inputs: Optional[dict] = None
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


def load_qwen2_5_vl(model_path:str, questions: list[str], images_list: list[list[Image.Image]]) -> ModelRequestData:
    from qwen_vl_utils import smart_resize
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": max(len(images) for images in images_list)},
    )

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    def post_process_image(image: Image) -> Image:
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height, width, max_pixels=1024 * 28 * 28
            )
            return image.resize((resized_width, resized_height))

    llm_inputs = []
    for question, images in zip(questions, images_list):
        placeholders = [{"type": "image", "image": image} for image in images]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_data = [post_process_image(image) for image in images]
        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image_data},
        })

    return ModelRequestData(
        engine_args=engine_args,
        llm_inputs=llm_inputs
    )


def load_internvl(model_path:str, questions: str, images_list: list[list[Image.Image]]) -> ModelRequestData:
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": max(len(images) for images in images_list)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    llm_inputs = []

    for question, images in zip(questions, images_list):
        placeholders = "\n".join(
            f"Image-{i}: <image>\n" for i, _ in enumerate(images, start=1)
        )
        messages = [{"role": "user", "content": f"{placeholders}\n{question}"}]

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        })

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [
        token_id for token in stop_tokens
        if (token_id := tokenizer.convert_tokens_to_ids(token)) is not None
    ]

    return ModelRequestData(
        engine_args=engine_args,
        llm_inputs=llm_inputs,
        stop_token_ids=stop_token_ids,
    )


model_example_map = {
    "qwen2_5_vl": load_qwen2_5_vl,
    "internvl": load_internvl,
}


def run_generate(model, model_path, question: str, images: list[str], seed: Optional[int]):
    req_data = model_example_map[model](model_path, question, images)
    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=256, stop_token_ids=req_data.stop_token_ids
    )

    outputs = llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {"image": req_data.image_data},
        },
        sampling_params=sampling_params,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)



def run_generate_batch(model, model_path, 
                       question_list: list[str], images_list: list[list[Image.Image]],
                       seed: Optional[int]):
    
    req_data = model_example_map[model](model_path, question_list, images_list)
    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=256, stop_token_ids=req_data.stop_token_ids
    )

    outputs = llm.generate(
        req_data.llm_inputs,
        sampling_params=sampling_params,
    )

    print("=" * 60)
    for i, o in enumerate(outputs):
        print(f"[Sample {i}]")
        print(o.outputs[0].text)
        print("=" * 60)



def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models that support multi-image input for text "
        "generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="phi3_v",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--model-path",
        "-p",
        type=str,
        default="Qwen/Qwen2.5-VL-Chat-7B-Instruct",
        help="Path to the model on Hugging Face Hub or local path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    return parser.parse_args()


def main(args: Namespace):
    model = args.model_type
    model_path = args.model_path
    seed = args.seed

    image_path1 = "/fs/archive/share/mm_datasets/GAOKAO-MM/Data/2010-2023_Chemistry_MCQs/2010-2023_Chemistry_MCQs_52_0.png"
    image_path2 = "/fs/archive/share/mm_datasets/GAOKAO-MM/Data/2010-2023_Chemistry_MCQs/2010-2023_Chemistry_MCQs_53_0.png"
    image_a = Image.open(image_path1)
    image_b = Image.open(image_path2)


    questions = [
        "What is shown in these chemistry images?",
        "Please describe the following diagrams."
    ]
    images_list = [
        [image_a, image_b],
        [image_b, image_a]
    ]

    # run_generate(model, model_path, question, images, seed)
    run_generate_batch(model, model_path, questions, images_list, seed)


if __name__ == "__main__":
    args = parse_args()
    main(args)