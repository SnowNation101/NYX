def load_processor(model_args):
    if model_args.model_backbone == "llava_next":
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True)
    elif model_args.model_backbone == "phi3_v":
        from transformers import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
        )
    elif model_args.model_backbone == "qwen2_5_vl":
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    processor.tokenizer.padding_side = "right"
    return processor
