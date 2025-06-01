import os
import json

coco_dir = "/fs/archive/share/mm_datasets/MSCOCO2017"
aokvqa_dir = "/fs/archive/share/mm_datasets/A-OKVQA"


def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    for data in dataset:
        data.update({
            'question': f"<|vision_start|><|image_pad|><|vision_end|> {data['question']}",
            'image_path': get_coco_path(split, data['image_id'], coco_dir)
        })
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def get_aokvqa_data():
    train_dataset = load_aokvqa(aokvqa_dir, 'train')
    val_dataset = load_aokvqa(aokvqa_dir, 'val')
    test_dataset = load_aokvqa(aokvqa_dir, 'test')

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = get_aokvqa_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")