import json
import os
import random
from datasets import Dataset, DatasetDict, Features, Sequence, Value

data_dir = "/fs/archive/share/mm_datasets/t2t_data"

SUBSET_SPLIT = {
    "2wikimultihopqa": ['train', 'dev'],
    "bamboogle": ['test'],
    "hotpotqa": ['train', 'dev'],
    "musique": ['train', 'dev'],
}


raw_datasets = {}
for dataset_name, splits in SUBSET_SPLIT.items():
    raw_datasets.setdefault(dataset_name, {})
    for split in splits:
        dataset_path = os.path.join(data_dir, dataset_name, f"{split}_with_retrieved_docs.jsonl")

        data_from_file = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_from_file.append(json.loads(line.strip()))

        # Rename dev -> test
        actual_split = 'test' if split == 'dev' else split
        raw_datasets[dataset_name][actual_split] = data_from_file


processed_data = {}
for dataset_name, splits in raw_datasets.items():
    processed_data.setdefault(dataset_name, {})
    for split_name, examples in splits.items():
        new_split_data = []
        for item in examples:
            new_item = {
                "qry": item["question"],
                "qry_image_path": [],
                "pos_text": item["retrieved_docs"][0],
                "pos_image_path": [],
                "neg_text": item["retrieved_docs"][10:],
                "neg_image_path": [],
                "ans": item["golden_answers"],
            }
            new_split_data.append(new_item)

        # If it's test split and size > 250, subsample
        if split_name == "test" and len(new_split_data) > 250:
            random.seed(42)
            new_split_data = random.sample(new_split_data, 250)

        processed_data[dataset_name][split_name] = new_split_data


repo_id = "SnowNation/Nyx-T2T-Data"

custom_features = Features({
    "qry": Value("string"),
    "qry_image_path": Sequence(feature=Value("string")),
    "pos_text": Value("string"),
    "pos_image_path": Sequence(feature=Value("string")),
    "neg_text": Sequence(feature=Value("string")),
    "neg_image_path": Sequence(feature=Value("string")),
    "ans": Sequence(feature=Value("string")),
})

for dataset_name, splits_data in processed_data.items():
    dataset_dict = DatasetDict()
    for split_name, data_list in splits_data.items():
        dataset_dict[split_name] = Dataset.from_list(data_list, features=custom_features)

    dataset_dict.push_to_hub(repo_id, config_name=dataset_name, private=False)
