from datasets import load_dataset, DatasetDict

synthetic_dataset_path = "/fs/archive/share/mm_datasets/mmE5/mmE5-synthetic/"
subsets = ["Classification", "Retrieval", "VQA"]
repo_id = "SnowNation/Nyx-mmE5-Synthetic"
for subset in subsets:
    subset_data = load_dataset(synthetic_dataset_path, subset, split="train")

    subset_data = subset_data.filter(lambda x: x["pos_text"] != "")

    def clean_qry(example):
        if example["modality"] in ["T2I", "T2IT"] and example["qry"].count("<|image_1|>"):
            example["qry"] = example["qry"].replace("<|image_1|>\n", "")
        return example

    subset_data = subset_data.map(clean_qry)
    subset_dict = DatasetDict({"train": subset_data})
    subset_dict.push_to_hub(repo_id, config_name=subset, private=False)

mmeb_dataset_path = "/fs/archive/share/mm_datasets/mmE5/mmE5-MMEB-hardneg/"
subsets = ['TAT-DQA', 'ArxivQA', 'InfoSeek_it2t', 'InfoSeek_it2it', 'ImageNet_1K', 'N24News', 'HatefulMemes', 'SUN397', 'VOC2007', 'InfographicsVQA', 'ChartQA', 'A-OKVQA', 'DocVQA', 'OK-VQA', 'Visual7W', 'VisDial', 'CIRR', 'NIGHTS', 'WebQA', 'VisualNews_i2t', 'VisualNews_t2i', 'MSCOCO_i2t', 'MSCOCO_t2i', 'MSCOCO']
repo_id = "SnowNation/Nyx-mmE5-MMEB"
for subset in subsets:
    subset_data = load_dataset(mmeb_dataset_path, subset, split="train")

    subset_data = subset_data.filter(lambda x: x["pos_text"] != "")

    def clean_qry(example):
        if "<|image_1|>" in example["qry"] and example["qry_image_path"] == "":
            example["qry"] = example["qry"].replace("<|image_1|>\n", "")
        return example

    subset_data = subset_data.map(clean_qry)
    subset_dict = DatasetDict({"train": subset_data})
    subset_dict.push_to_hub(repo_id, config_name=subset, private=False)
