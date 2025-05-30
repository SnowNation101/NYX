import os
import json

root = "/fs/archive/share/mm_datasets"

gaokao_path = os.path.join(root, "GAOKAO-Bench/Data")
gaokao_update_path = os.path.join(root, "GAOKAO-Bench-Updates/Data")
gaokao_mm_path = os.path.join(root, "GAOKAO-MM/Data")

def get_gaokao_bench_data():
    """Get the data from GAOKAO-Bench dataset."""
    """Only keep the objective questions in the Gaokao-Bench dataset."""
    paths = [
        os.path.join(gaokao_path, "Objective_Questions/"),
        os.path.join(gaokao_update_path, "GAOKAO-Bench-2023"),
        os.path.join(gaokao_update_path, "GAOKAO-Bench-2024")
    ]

    files = []
    for path in paths:
        files.extend([os.path.join(path, f) for f in os.listdir(path) 
                      if f.endswith(".json")])

    kb = []
    test = []

    for file in files:
        with open(file, "r", encoding='utf-8') as f:
            data = json.load(f)
            key = data["keywords"]
            examples = data["example"]
            for example in examples:
                example["keywords"] = key
                target_list = kb if example["year"] in map(str, range(2010, 2021)) else test
                target_list.append(example)

    return kb, test


def get_gaokao_mm_data():
    """Get the data from GAOKAO-MM dataset. Add vision padding to questions."""
    files = [f for f in os.listdir(gaokao_mm_path) if f.endswith(".json")]
    kb = []
    test = []
    for file in files:
        with open(os.path.join(gaokao_mm_path, file), "r", encoding='utf-8') as f:
            data = json.load(f)
            key = data["keywords"]
            examples = data["example"]
            for example in examples:
                example["keywords"] = key
                target_list = (
                    kb if example["year"] in map(str, range(2010, 2020)) else test
                )
                example["picture"] = [
                    os.path.join(gaokao_mm_path, picture)
                    for picture in example["picture"]
                ]
                num_pictures = len(example["picture"])
                if num_pictures < 4:
                    example["question"] = (
                        "<|vision_start|><|image_pad|><|vision_end|>" 
                        * num_pictures
                        + example["question"]
                    )
                else:
                    example["question"] = (
                        "<|vision_start|><|image_pad|><|vision_end|>"
                        * (num_pictures - 4)
                        + example["question"]
                    )
                    example["question"] += (
                        "\n" + "<|vision_start|><|image_pad|><|vision_end|>" * 4
                    )
                target_list.append(example)

    return kb, test


if __name__ == "__main__":
    gk_kb, gk_test = get_gaokao_bench_data()
    mm_kb, mm_test = get_gaokao_mm_data()

    print(len(gk_kb))
    print(len(gk_test))
    print(len(mm_kb))
    print(len(mm_test))