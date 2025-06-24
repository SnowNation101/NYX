import os
import json
import math
from dataclasses import asdict
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams

from PIL import Image, ImageFile

# Some images in the dataset are truncated,
# importing this to avoid errors when resizing them.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Some images in the dataset are AVIF format,
# which requires the pillow_avif package to be loaded.
import pillow_avif

TEXT_PROMPT = f"""You are given a plain text document. Your task is to generate exactly **five** question-and-answer (Q&A) pairs based on the content of this document.

Please follow these instructions carefully:

1. All questions must be **based on the content**, but must be written as if they are **standalone natural questions**—that is, they should make sense independently and be meaningful even if the document is not shown.
2. **Do not ask questions that refer to the document directly**, such as “What is mentioned in the document?” or “Who appears in the text?”. Your goal is to create **real-world questions** that could plausibly be answered by retrieving this document from a large corpus.
3. The questions should be **ordered from easy to hard**, progressing from basic factual recall to deeper reasoning, interpretation, or open-ended thought.
4. The **styles of the questions should vary**, including a mix of question types such as:
    - Factual
    - Conceptual understanding
    - Inference
    - Comparison
    - Open-ended or critical thinking
5. All answers should be **textual**, concise, and clearly grounded in the document content.
6. Present each pair in the following format:
    [Q1]: [Question]
    [A1]: [Answer]
    …
    [Q5]: [Question]
    [A5]: [Answer]
"""

MULTIMODAL_PROMPT = f"""You are given a document that contains both **text and images**. Your task is to generate exactly **five** question-and-answer (Q&A) pairs based on the **combined content** of the document.

Please follow these instructions carefully:

1. All questions must be **based on the document**, but should be written as **standalone natural questions** that make sense without referencing the document itself. Avoid questions like “What is shown in this document?” or “Which image appears above?”.
2. Your goal is to write questions that, if submitted to a retrieval system, would ideally retrieve this document as the most relevant result.
3. Among the five questions:
    - Exactly **2 questions** should be based on **pure text content**.
    - Exactly **3 questions** must involve **one or more images** from the document. When doing so, refer to the images using numbered placeholders like `<image1>`, `<image2>`, etc.
    - You may reference multiple images in a question (e.g., “Compare <image1> and <image3>”), but make sure the reference is **precise**.
4. Questions should be **ordered from easy to hard**, progressing from factual recall to conceptual, inferential, comparative, or open-ended questions.
5. The **styles of the questions should vary**, including:
    - Factual
    - Conceptual understanding
    - Inference
    - Comparison
    - Critical or open-ended reasoning
6. All answers must be **pure text**, accurate, and grounded in the document content.
7. Present each pair in the following format:
    [Q1]: [Question]
    [A1]: [Answer]
    …
    [Q5]: [Question]
    [A5]: [Answer]

    For questions involving images, clearly indicate them like:
    “Based on <image2>, ...” or “Considering both <image1> and <image3>, ...” etc.
"""

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, 
    factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, 
    max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar



def main():
    model_path = "/fs/archive/share/InternVL3-78B"
    dataset_path = "/fs/archive/share/mm_datasets/obelics_chunked_dataset.json"
    images_dir = "/fs/archive/share/mm_datasets/obelics_images"

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # dataset = dataset[:100]  # For testing purposes

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 5, "video": 0},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        tensor_parallel_size=4,
        seed=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [
        token_id for token in stop_tokens
        if (token_id := tokenizer.convert_tokens_to_ids(token)) is not None
    ]

    vlm = LLM(**asdict(engine_args))
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids
    )

    llm_inputs = []
    raw_metadata = []
    for data in dataset:
        doc = data["text"].replace("<|image|>", "<image>")
        images = []
        for image_path in data['images']:
            image = Image.open(os.path.join(images_dir, image_path))
            new_h, new_w = smart_resize(image.height, image.width)
            image = image.resize((new_w, new_h))
            images.append(image)

        if images:
            user_prompt = MULTIMODAL_PROMPT
        else:
            user_prompt = TEXT_PROMPT

        messages = [{"role": "user", "content": f"{user_prompt}\n{doc}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_entry = {"prompt": prompt}
        if images:
            input_entry["multi_modal_data"] = {"image": images}

        llm_inputs.append(input_entry)
        raw_metadata.append({"text": data["text"], "images": data["images"]})

    batch_outputs = vlm.generate(llm_inputs, sampling_params=sampling_params)

    results = []
    for meta, output in zip(raw_metadata, batch_outputs):
        results.append({
            "doc": meta["text"],
            "images": meta["images"],
            "generated_qa": output.outputs[0].text,
        })

    os.makedirs("process_obelics", exist_ok=True)
    with open("process_obelics/generated_qa.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()