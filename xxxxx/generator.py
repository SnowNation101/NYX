from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
import torch

class Generator:
    def __init__(self, model_name):
        if "Qwen2_5" in model_name:
            self.model_type = "qwen2.5-vl"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        elif "Qwen2" in model_name:
            self.model_type = "qwen2-vl"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=True
            )

    def generate(self, 
                 text: str, 
                 images: list = None, 
                 max_new_tokens: int = 2048) -> str:
        assert text != None, "Text input cannot be None"
        if self.model_type in ["qwen2-vl", "qwen2.5-vl"]:
            # Apply a chat template
            text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>\nuser\n" + text + "<|im_end|>\n<|im_start|>assistant]\n"
            inputs = self.processor(text=text, 
                                images=images, 
                                return_tensors="pt",
                                truncation=False
                                ).to(self.model.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, 
                                                                generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )

        return output_text