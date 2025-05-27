import os

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2VLModel,
    Qwen2_5_VLModel,
)


class EmebeddingRetriever:
    """
    A class to retrieve embeddings from a model and build an index for retrieval.

    Args:
    model_name (str): The name of the model to use.
    index (faiss.Index): The index to use for retrieval.
    """
    
    def __init__(self, 
                 ckpt_path, 
                 base_model_path,
                 task="gaokao-bench", index=None):
        if "Qwen2.5" in base_model_path:
            self.model = Qwen2_5_VLModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        elif "Qwen2" in base_model_path:
            self.model = Qwen2VLModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        self.task = task
        self.processor = AutoProcessor.from_pretrained(base_model_path)
        self.index = index

    def _last_pooling(self, last_hidden_state, attention_mask, normalize=True):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        indices = torch.arange(batch_size, device=last_hidden_state.device)
        reps = last_hidden_state[indices, sequence_lengths]
        if normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def build_index(self, knowledge_base):
        embeddings = []
        if self.task == "gaokao_mm":
            for knowledge in tqdm(knowledge_base, desc="Processing knowledge base"):
                kb_question = knowledge["question"]
                kb_answer = knowledge["answer"]
                kb_analysis = knowledge["analysis"]
                kb_image_paths = knowledge["picture"]

                kb_text = f"""<|im_start|>题目：{kb_question}
                答案：{kb_answer}
                解析：{kb_analysis}<|im_end|>
                """

                kb_images = [Image.open(picture) for picture in kb_image_paths]

                inputs = self.processor(
                    text=kb_text,
                    images=kb_images,
                    return_tensors="pt",
                    truncation=False,
                ).to(self.model.device)
                with torch.no_grad():
                    model_output = self.model(
                        **inputs, return_dict=True, output_hidden_states=True
                    )
                    embedding = self._last_pooling(
                        model_output.hidden_states[-1], inputs["attention_mask"]
                    )
                    embeddings.append(embedding.float().cpu().numpy())
        elif self.task == "gaokao_bench":
            # TODO: complete this
            for knowledge in tqdm(knowledge_base, desc="Processing knowledge base"):
                kb_question = knowledge["question"]
                kb_answer = knowledge["answer"]
                kb_analysis = knowledge["analysis"]

                kb_text = f"""<|im_start|>题目：{kb_question}
                答案：{kb_answer}
                解析：{kb_analysis}<|im_end|>
                """

                inputs = self.processor(
                    text=kb_text,
                    images=None,
                    return_tensors="pt",
                    truncation=False,
                ).to(self.model.device)
                with torch.no_grad():
                    model_output = self.model(
                        **inputs, return_dict=True, output_hidden_states=True
                    )
                    embedding = self._last_pooling(
                        model_output.hidden_states[-1], inputs["attention_mask"]
                    )
                    embeddings.append(embedding.float().cpu().numpy())

        embeddings = np.vstack(embeddings).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        index_file_name = f"{self.task}_index.faiss"
        index_file_path = os.path.join("outputs/indexes", index_file_name)
        faiss.write_index(index, index_file_path)
        print(f"Index built, saved to {index_file_path}")
        self.index = index

        return index

    def retrieve(self, qry_text: str, qry_images: list, top_k: int = 10):
        inputs = self.processor(
            text=qry_text, 
            images=qry_images, 
            return_tensors="pt", 
            truncation=False
        ).to(self.model.device)

        with torch.no_grad():
            model_output = self.model(
                **inputs, return_dict=True, output_hidden_states=True
            )
            embedding = self._last_pooling(
                model_output.hidden_states[-1], inputs["attention_mask"]
            )
            embedding = embedding.float().cpu().numpy()
        D, I = self.index.search(embedding, top_k)
        return D, I
