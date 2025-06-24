import json
import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from datasets import load_dataset
from generator import MMGenerator

import re
import string
from collections import Counter

model_name = "/fs/archive/share/bge-m3"
model = BGEM3FlagModel(model_name,  
                       use_fp16=True)

with open("t2t_corpus.json", "r") as f:
    corpus = json.load(f)

datasets_info = {
    "hotpotqa": ("hotpotqa", "dev"),
    "2wikimultihopqa": ("2wikimultihopqa", "dev"),
    "bamboogle": ("bamboogle", "test"),
    "musique": ("musique", "dev"),
}

index_path = "index/bge_m3.faiss"

# ===========indexing===========

# embeddings = []
# for item in tqdm(corpus, desc="Indexing"):
#     output = model.encode(
#         item,
#         batch_size=12, 
#         max_length=8192,
#         )['dense_vecs']
#     embeddings.append(output)

# embeddings = np.vstack(embeddings).astype("float32")
# index = faiss.IndexFlatIP(embeddings.shape[1])
# index.add(embeddings)


# faiss.write_index(index, index_path)
# print(f"Index saved to {index_path}")

# ===========retrieving===========

# def retrieve(index, corpus, query, top_k=10):
#     query_embedding = model.encode(
#         query,
#         batch_size=12, 
#         max_length=8192,
#         )['dense_vecs'].reshape(1, -1)
    
#     _, I = index.search(query_embedding, top_k)
#     return [corpus[i] for i in I[0]]


# index = faiss.read_index(index_path)
# print(f"Index loaded from {index_path}")

# dataset_path = "/fs/archive/share/mm_datasets/Nyx-T2T-Data"

# for name, (subset, split) in datasets_info.items():
#     print(f"\nProcessing {name}...")
#     dataset = load_dataset(dataset_path, subset, split=split)
#     results = []

#     for item in tqdm(dataset, desc=f"retrieving {name}"):
#         item["retrieved_docs"] = retrieve(index, corpus, item["qry"])
#         results.append(item)

#     output_path = f"retrieved/{name}_retrieved_docs.json"
#     with open(output_path, "w") as f:
#         json.dump(results, f, indent=2, ensure_ascii=True)
#     print(f"Saved: {output_path}")


# ===========generating===========
# model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
# vlm = MMGenerator(model_path=model_path)

# top_k = 1

# for name in datasets_info.keys():
#     with open(f"retrieved/{name}_retrieved_docs.json", "r") as f:
#         dataset = json.load(f)
    
#     for item in tqdm(dataset, desc=f"generating {name} answers"):
#         question = item["qry"]
#         retrieved_docs = item["retrieved_docs"][:top_k]

#         response = vlm.generate(
#             docs=retrieved_docs,
#             question=question,
#             images=None
#         )
        
#         item["response"] = response
#         item.pop("neg_text", None)

#     output_path = f"generated/{name}_response.json"
#     with open(output_path, "w") as f:
#         json.dump(dataset, f, indent=2, ensure_ascii=True)
#     print(f"Saved: {output_path}")

# ===========evaluating===========
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_pred, a_true):
    return int(normalize_answer(a_pred) == normalize_answer(a_true))

def compute_f1(a_pred, a_true):
    pred_tokens = normalize_answer(a_pred).split()
    true_tokens = normalize_answer(a_true).split()
    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return int(pred_tokens == true_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


for name in datasets_info.keys():
    with open(f"generated/{name}_response.json", "r") as f:
        dataset = json.load(f)

    total_em = 0
    total_f1 = 0
    count = 0

    for item in tqdm(dataset, desc=f"evaluating {name} answers"):
        pred = item["response"].strip()
        gold_list = item["ans"]

        em_scores = [compute_exact(pred, g) for g in gold_list]
        f1_scores = [compute_f1(pred, g) for g in gold_list]

        total_em += max(em_scores)
        total_f1 += max(f1_scores)
        count += 1

    avg_em = total_em / count
    avg_f1 = total_f1 / count

    print(f"\n{name} Evaluation:")
    print(f"  Exact Match: {avg_em:.4f}")
    print(f"  F1 Score:    {avg_f1:.4f}")

    
    