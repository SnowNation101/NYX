import os
import random
import datasets
import json
import ast

import logging
logger = logging.getLogger(__name__)

from typing import List, Tuple
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from nyx.utils.image_processing import smart_resize

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_TOKEN = "<|image|>"
PHI_IMAGE_TOKEN = "<|image_1|>"
LLAVA_IMAGE_TOKEN = "<image>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


class TrainDataset(Dataset):

    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        self.negative_ratio = self.data_args.negative_ratio
        train_data = []
        if self.data_args.synthetic_dataset_name or self.data_args.synthetic_dataset_path:
            print(f"Loading {len(data_args.synthetic_subset_name)} synthetic datasets: {data_args.synthetic_subset_name}")
            for subset in data_args.synthetic_subset_name:
                num_sample = -1
                if self.data_args.synthetic_dataset_name:
                    subset_data = load_dataset(
                        self.data_args.synthetic_dataset_name,
                        subset,
                        split=f"{self.data_args.dataset_split}[:{num_sample}]",
                    )
                elif self.data_args.synthetic_dataset_path:
                    subset_path = os.path.join(self.data_args.synthetic_dataset_path, subset) 
                    subset_data = load_from_disk(subset_path)
                    if len(subset_data) > num_sample and num_sample != -1:
                        subset_data = subset_data.select(range(num_sample))
                train_data.append(subset_data)
        if self.data_args.dataset_name or self.data_args.dataset_path:
            print(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
            for subset in data_args.subset_name:
                num_sample = data_args.num_sample_per_subset
                if self.data_args.dataset_name:
                    subset_data = load_dataset(
                        self.data_args.dataset_name,
                        subset,
                        split=f"{self.data_args.dataset_split}[:{num_sample}]",
                    )
                elif self.data_args.dataset_path:
                    subset_path = os.path.join(self.data_args.dataset_path, subset) 
                    subset_data = load_from_disk(subset_path)
                    if len(subset_data) > num_sample and num_sample != -1:
                        subset_data = subset_data.select(range(num_sample))
                train_data.append(subset_data)
        if self.data_args.t2t_dataset_name:
            print(f"Loading {len(data_args.t2t_subset_name)} T2T datasets: {data_args.t2t_subset_name}")
            for subset in data_args.t2t_subset_name:
                num_sample = self.data_args.num_sample_per_subset
                subset_data = load_dataset(
                    self.data_args.t2t_dataset_name,
                    subset,
                    split=self.data_args.dataset_split,
                )
                if len(subset_data) > num_sample and num_sample != -1:
                    subset_data = subset_data.select(range(num_sample))
                # Look, this is utterly infuriating. 
                # The "mmE5" data is flat-out string, 
                # not the sequences it should be.
                # But naturally, the Nyx data, being *properly* structured, 
                # uses sequences.
                # This mismatch is a guaranteed concatenation nightmare. 
                # So, here we are, forced to painstakingly
                # convert everything to string just to make this mess compatible. 
                # This isn't a fix; 
                # it's a desperate workaround.
                column_names = subset_data.column_names
                subset_data = subset_data.map(
                        self._to_string_converter,
                        fn_kwargs={"column_names": column_names}
                    )
                train_data.append(subset_data)
        if self.data_args.mm_dataset_path:
            import json
            from datasets import Dataset
            print("Loading mixed modal datasets")
            # TODO: finish this
        
        self.train_data = concatenate_datasets(train_data)

        print(f"Number of samples: {len(self.train_data)}")
        
    def _to_string_converter(self, example, column_names):
        """ Convert all fields in the example to string."""
        for col in column_names:
            if isinstance(example[col], list):
                example[col] = json.dumps(example[col], ensure_ascii=False)
            else:
                example[col] = str(example[col])
        return example

    def __len__(self):
        return len(self.train_data)

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((512, 512))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone == "mllama":
            if image.size[1] == 1:
                # print(f"Failed Image: {image}.")
                image = image.resize((image.size[0], 2))
        elif self.model_args.model_backbone == "llava_next":
            return self._process_image(image, "high")
        elif self.model_args.model_backbone == "qwen2_5_vl":
            new_h, new_w = smart_resize(image.height, image.width)
            return image.resize((new_w, new_h))
        else:
            return image

    def _extract_images(self, item):
        images = []
        valid_extensions = (
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
            '.webp', '.ico', '.svg'
        )

        def is_valid_image(filename):
            return isinstance(filename, str) and filename.lower().endswith(valid_extensions)

        if isinstance(item, str):
            item = item.strip()
            if is_valid_image(item):
                images.append(item)
            else:
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, list):
                        for i in parsed:
                            if is_valid_image(i):
                                images.append(i)
                except Exception:
                    pass
        elif isinstance(item, list):
            for i in item:
                if is_valid_image(i):
                    images.append(i)

        return images

    def _filter_hard_negtives(self, negs, pos, negative_ratio):
        negs = eval(negs)
        if not isinstance(negs, list):
            negs = [negs]

        if len(negs) < negative_ratio and len(negs) > 0:
            negs += [negs[-1]] * (negative_ratio - len(negs))

        negs = negs[:negative_ratio]
        return negs

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        qry_text, qry_image_path, pos_text, pos_image_path = (
            self.train_data[item]["qry"], self.train_data[item]["qry_image_path"],
            self.train_data[item]["pos_text"], self.train_data[item]["pos_image_path"],
        )

        qry_image_paths = self._extract_images(qry_image_path)
        qry_images = [self._get_image(img_path) for img_path in qry_image_paths]
        pos_image_paths = self._extract_images(pos_image_path)
        pos_images = [self._get_image(img_path) for img_path in pos_image_paths]


        neg_texts, neg_image_paths, neg_images = [], [], []
        if self.negative_ratio > 0:
            neg_text_list, neg_image_path_list = (
                self.train_data[item]["neg_text"], self.train_data[item]["neg_image_path"],
            )
            neg_texts = self._filter_hard_negtives(neg_text_list, pos_text, self.negative_ratio)
            neg_image_paths = self._filter_hard_negtives(neg_image_path_list, pos_image_path, self.negative_ratio)

        for ind, neg in enumerate(neg_texts):
            if neg == '':
                if len(set(eval(neg_text_list))) == 1:
                    neg_texts[ind] = pos_text
                else:
                    neg_texts[ind] = random.choice([text for text in eval(neg_text_list) if text != ""])
        if self.model_args.model_backbone == "llava_next":
            # Update image token
            qry_text = qry_text.replace(PHI_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN)
            pos_text = pos_text.replace(PHI_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN)
            for ind, neg in enumerate(neg_texts):
                neg_texts[ind] = neg.replace(PHI_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN)
        elif self.model_args.model_backbone == "qwen2_5_vl":
            # Update image token
            qry_text = qry_text.replace(PHI_IMAGE_TOKEN, QWEN_IMAGE_TOKEN)
            pos_text = pos_text.replace(PHI_IMAGE_TOKEN, QWEN_IMAGE_TOKEN)
            for ind, neg in enumerate(neg_texts):
                neg_texts[ind] = neg.replace(PHI_IMAGE_TOKEN, QWEN_IMAGE_TOKEN)

        for neg_img in neg_image_paths:
            neg_images.append(self._get_image(neg_img))

        ret = (qry_text, qry_images,
                pos_text, pos_images,
                neg_texts, neg_images)
        
        if qry_text.count(QWEN_IMAGE_TOKEN) != len(qry_images):
            logger.warning(f"+++Warning: The number of {QWEN_IMAGE_TOKEN} tokens ({qry_text.count(QWEN_IMAGE_TOKEN)}) does not match the number of images ({len(qry_images)}).")
            logger.warning(f"+++qry_text: {qry_text}")
            logger.warning(f"+++qry_images: {qry_image_path}")
    
        return ret


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args

        if self.data_args.dataset_name:
            self.eval_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=self.data_args.dataset_split,
                download_mode="force_redownload"
            )
        elif self.data_args.dataset_path:
            subset_path = os.path.join(self.data_args.dataset_path, subset) 
            self.eval_data = load_from_disk(subset_path)
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.model_args.model_backbone == "llava_next":
            # Update llava image token
            text = text.replace(PHI_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN)
        return text, self._get_image(img_path),

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((512, 512))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone == "llava_next":
            return self._process_image(image, "high")
        else:
            return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif isinstance(row[text_field], List):
                assert isinstance(row[img_path_field], List) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data
