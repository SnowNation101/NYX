from datasets import load_dataset
from huggingface_hub import snapshot_download

data_dir = "/fs/archive/share/mm_datasets/OBELICS"

snapshot_download(
    repo_id="HuggingFaceM4/OBELICS",
    repo_type="dataset",
    allow_patterns="data/*",
    local_dir=data_dir,
    endpoint="https://hf-mirror.com"
)

# if successfully loaded,
# the dataset is fully downloaded
obelics_dataset = load_dataset(data_dir)

print("loaded!!")