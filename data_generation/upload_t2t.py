import json
import os
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"Attempting to read: {dataset_path}")

        data_from_file = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data_from_file.append(json.loads(line.strip()))
            raw_datasets[dataset_name][split] = data_from_file
            logger.info(f"Successfully loaded {len(data_from_file)} items from {dataset_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {dataset_path}. Skipping this split.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {dataset_path}: {e}. Skipping this split.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading {dataset_path}: {e}. Skipping this split.")


processed_data = {}
for dataset_name, splits in SUBSET_SPLIT.items():
    for split in splits:
        if dataset_name not in raw_datasets or split not in raw_datasets[dataset_name]:
            logger.warning(f"Skipping processing for {dataset_name} - {split} as raw data was not loaded.")
            continue

        new_split_data = []
        processed_data.setdefault(dataset_name, {})
        for item in raw_datasets[dataset_name][split]:
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
        processed_data[dataset_name][split] = new_split_data
        logger.info(f"Processed {len(new_split_data)} items for {dataset_name} - {split}")
    logger.info("=" * 50)

repo_id = "SnowNation/Nyx-T2T-Data" 

logger.info(f"Preparing to push data to Hugging Face Hub under repository: {repo_id}")
logger.info("-" * 50)

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
        if not data_list:
            logger.warning(f"No data to create Dataset for subset '{dataset_name}', split '{split_name}'. Skipping.")
            continue
        
        dataset_dict[split_name] = Dataset.from_list(data_list, features=custom_features)
        logger.info(f"Created Dataset for subset '{dataset_name}', split '{split_name}' with {len(data_list)} items.")

    if dataset_dict:
        logger.info(f"Pushing subset '{dataset_name}' to {repo_id}...")
        try:
            dataset_dict.push_to_hub(repo_id, config_name=dataset_name, private=False)
            logger.info(f"Successfully pushed subset '{dataset_name}' to Hugging Face Hub.")
        except Exception as e:
            logger.error(f"Failed to push subset '{dataset_name}' to Hugging Face Hub: {e}")
    else:
        logger.warning(f"Skipping push for subset '{dataset_name}' as no valid splits were found.")
    logger.info("=" * 50)

logger.info(f"All relevant datasets pushed to Hugging Face Hub under {repo_id}!")
logger.info("You can view your dataset at: https://huggingface.co/datasets/" + repo_id)