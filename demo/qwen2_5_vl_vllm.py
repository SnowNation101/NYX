from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/home/u2024103939/model/Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_PATH)
print(processor.tokenizer.eos_token)

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": "What is the text in the illustrate?"},
        ],
    },
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, _ = process_vision_info(messages)

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": {"image": image_inputs},
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)