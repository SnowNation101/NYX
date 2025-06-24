from datasets import load_dataset

dataset_path = "/fs/archive/share/mm_datasets/Nyx-mmE5-Synthetic/"
subsets = ["Classification", "Retrieval", "VQA"]
sythetic_dataset =  {}
for subset in subsets:
    subset_data = load_dataset(
                        dataset_path,
                        subset,
                        split="train",
                    )
    for item in subset_data:
        if item["pos_text"] == "":
            print(f"Empty pos_text found in synthetic data, subset {subset}: {item}")
print("="*10, "Synthetic dataset loaded and checked for empty queries.")

dataset_path = "/fs/archive/share/mm_datasets/Nyx-mmE5-MMEB-hardneg/"
subsets = ['TAT-DQA', 'ArxivQA', 'InfoSeek_it2t', 'InfoSeek_it2it', 'ImageNet_1K', 'N24News', 'HatefulMemes', 'SUN397', 'VOC2007', 'InfographicsVQA', 'ChartQA', 'A-OKVQA', 'DocVQA', 'OK-VQA', 'Visual7W', 'VisDial', 'CIRR', 'NIGHTS', 'WebQA', 'VisualNews_i2t', 'VisualNews_t2i', 'MSCOCO_i2t', 'MSCOCO_t2i', 'MSCOCO']
mmeb_hardneg_dataset = {}
for subset in subsets:
    subset_data = load_dataset(
                        dataset_path,
                        subset,
                        split="train",
                    )
    mmeb_hardneg_dataset[subset] = subset_data
    for item in subset_data:
        if item["pos_text"] == "":
            print(f"Empty pos_text found in MMEB hardneg data, subset {subset}: {item}")
print("="*10, "MMEB hardneg dataset loaded and checked for empty queries.")

dataset_path = "/fs/archive/share/mm_datasets/Nyx-T2T-Data"
subsets = ["2wikimultihopqa", "hotpotqa", "musique"]
t2t_dataset = {}
for subset in subsets:
    subset_data = load_dataset(
                        dataset_path,
                        subset,
                        split="train",
                    )
    t2t_dataset[subset] = subset_data
    for item in subset_data:
        if item["pos_text"] is None or item["pos_text"] == "":
            print(f"Empty pos_text found in T2T data, subset {subset}: {item}")
print("="*10, "T2T dataset loaded and checked for empty queries.")
