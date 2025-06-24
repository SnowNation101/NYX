import torch
import json
import os
import numpy as np
import faiss

from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# Pooling and Normalization
def last_pooling(last_hidden_state, attention_mask, normalize=True):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), 
                             sequence_lengths]
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

model_name = "/fs/archive/share/mmE5-mllama-11b-instruct"

# Load Processor and Model
processor = AutoProcessor.from_pretrained(model_name)
model = MllamaForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

with open("baselines/t2t_corpus.json", "r") as f:
    corpus = json.load(f)

# embeddings = []
# for item in tqdm(corpus, desc="Indexing"):
#     inputs = processor(
#         text=item,
#         images=None,
#         return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         outputs = last_pooling(
#             model(
#                 **inputs, 
#                 return_dict=True, 
#                 output_hidden_states=True
#             ).hidden_states[-1], 
#             inputs['attention_mask'])
#         embeddings.append(outputs.float().cpu().numpy())

# embeddings = np.vstack(embeddings).astype("float32")

# index = faiss.IndexFlatIP(embeddings.shape[1])
# index.add(embeddings)

# faiss.write_index(index, "baselines/index/mmE5.faiss")

# index = faiss.read_index("baselines/index/mmE5.faiss")

# test_data = ""

# for item in tqdm(test_data, desc="retrieving hard negs"):
#     inputs = processor(
#         text=item["qry"],
#         images=None,
#         return_tensors="pt").to("cuda")
    
#     with torch.no_grad():
#         outputs = last_pooling(
#             model(
#                 **inputs, 
#                 return_dict=True, 
#                 output_hidden_states=True
#             ).hidden_states[-1], 
#             inputs['attention_mask'])
    
#     q_reps = outputs.float().cpu().numpy()
#     scores, indices = index.search(q_reps, k=20)