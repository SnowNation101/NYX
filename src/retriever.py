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
    
    def __init__(self, task,
                 ckpt_path, 
                 base_model_path,
                 ):
        self.task = task
        if "Qwen2.5" in base_model_path:
            self.model = Qwen2_5_VLModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(
                base_model_path,
                use_fast=True)
        elif "Qwen2" in base_model_path:
            self.model = Qwen2VLModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                )
            self.processor = AutoProcessor.from_pretrained(
                base_model_path,
                use_fast=True)


    def _last_pooling(self, last_hidden_state, attention_mask, normalize=True):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        indices = torch.arange(batch_size, device=last_hidden_state.device)
        reps = last_hidden_state[indices, sequence_lengths]
        if normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    

    def build_index(self, knowledge_base, index_path):
        # Check if the index file already exists
        if os.path.exists(index_path):
            overwrite = input(f"The index file {index_path} already exists. Do you want to overwrite it? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Operation cancelled")
                return
            
        embeddings = []
        if self.task == "gaokao_mm":
            for knowledge in tqdm(knowledge_base, desc="Processing knowledge base"):
                kb_question = knowledge["question"]
                kb_answer = knowledge["answer"]
                kb_analysis = knowledge["analysis"]
                kb_image_paths = knowledge["picture"]

                kb_text = (
                    f"题目：{kb_question}\n"
                    f"答案：{kb_answer}\n"
                    f"解析：{kb_analysis}"
                )

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
            for knowledge in tqdm(knowledge_base, desc="Processing knowledge base"):
                kb_question = knowledge["question"]
                kb_answer = knowledge["answer"]
                kb_analysis = knowledge["analysis"]

                kb_text = (
                    f"题目：{kb_question}\n"
                    f"答案：{kb_answer}\n"
                    f"解析：{kb_analysis}"
                )

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
        elif self.task == "aokvqa":
            for knowledge in tqdm(knowledge_base, desc="Processing knowledge base"):
                kb_question = knowledge["question"]
                kb_choices = knowledge["choices"]
                kb_answer_idx = knowledge["correct_choice_idx"]
                kb_analysis = knowledge["rationales"]
                kb_image = Image.open(knowledge["image_path"])

                kb_text = (
                    f"Question: {kb_question}\n"
                    f"Choices: {', '.join(kb_choices)}\n"
                    f"Answer: {kb_choices[kb_answer_idx]}\n"
                    f"Analysis: {', '.join(kb_analysis)}"
                )

                inputs = self.processor(
                    text=kb_text,
                    images=[kb_image],
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

        faiss.write_index(index, index_path)
        print(f"FAISS index built and saved to {index_path}")

        return index

    def retrieve(self, index: faiss.Index,
                 qry_text: str, qry_images: list, top_k: int = 10):
        inputs = self.processor(
            text=qry_text, 
            images=qry_images, 
            return_tensors="pt", 
            truncation=False
        )
        inputs.to(self.model.device)

        # Encode the query
        with torch.no_grad():
            model_output = self.model(
                **inputs, return_dict=True, output_hidden_states=True
            )
            embedding = self._last_pooling(
                model_output.hidden_states[-1], inputs["attention_mask"]
            )
            embedding = embedding.float().cpu().numpy()
        D, I = index.search(embedding, top_k)

        return D, I
