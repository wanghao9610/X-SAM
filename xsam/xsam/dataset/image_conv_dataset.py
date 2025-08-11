import copy
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from PIL import Image
from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.dataset.utils import expand2square

from .base_dataset import BaseDataset
from .utils.load import load_jsonl


class ImageConvDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="imgconv",
        offline_processed_text_folder=None,
        max_dataset_length=None,
        preprocess_text_data=False,
        is_multimodal=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            task_name=task_name,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            preprocess_text_data=preprocess_text_data,
            is_multimodal=is_multimodal,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.offline_processed_text_folder = kwargs.get("offline_processed_text_folder", None)
        self.max_dataset_length = kwargs.get("max_dataset_length", None)
        self.preprocess_text_data = kwargs.get("preprocess_text_data", False)
        self.is_multimodal = kwargs.get("is_multimodal", False)

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.data:
            cur_len = (
                sum(len(conv["value"].split()) for conv in data_dict["conversations"])
                if not self.preprocess_text_data
                else len(data_dict["input_ids"])
            )
            if data_dict.get("image", None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def _load_ann_data(self):
        assert self.offline_processed_text_folder or (self.data_path and self.tokenizer)
        if self.offline_processed_text_folder and self.data_path:
            print_log(
                "Both `offline_processed_text_folder` and "
                "`data_path` are set, and we load dataset from"
                "`offline_processed_text_folder` "
                f"({self.offline_processed_text_folder})",
                logger="current",
                level=logging.WARNING,
            )

        if self.offline_processed_text_folder is not None:
            self.data = load_from_disk(self.offline_processed_text_folder)
        else:
            if self.data_path.endswith(".json"):
                json_data = json.load(open(self.data_path))
            elif self.data_path.endswith(".jsonl"):
                json_data = load_jsonl(self.data_path)
            else:
                raise NotImplementedError

            data = json_data
            if self.preprocess_text_data:
                for idx in range(len(json_data)):
                    if isinstance(json_data[idx]["id"], int):
                        json_data[idx]["id"] = str(json_data[idx]["id"])
                json_data = DatasetDict({"train": HFDataset.from_list(json_data)})
                text_data = process_hf_dataset(
                    dataset=json_data,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    dataset_map_fn=self.dataset_map_fn,
                    template_map_fn=self.template_map_fn,
                    split="train",
                    max_dataset_length=self.max_dataset_length,
                    remove_unused_columns=False,
                    pack_to_max_length=False,
                    with_image_token=True,
                )
                data = text_data

        for d in data:
            if self.preprocess_fn is not None:
                d = self.preprocess_fn(d)
            if "image" not in d:
                continue
            d["image_file"] = d.pop("image")

        return data

    def __getitem__(self, index):
        index = index % self.data_length
        data_dict = copy.deepcopy(self.data[index])
        if data_dict.get("image_file", None) is not None:
            image_file = data_dict["image_file"]
            pil_image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
            if self.image_processor is not None:
                image = pil_image
                if self.pad_image_to_square:
                    image = expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                data_dict["pixel_values"] = image
            if self.extra_image_processor is not None:
                seg_output = self.extra_image_processor.preprocess(pil_image, return_tensors="pt")
                data_dict["image_info"] = {"image_file": image_file}
                data_dict["seg_pixel_values"] = seg_output["pixel_values"][0]
                data_dict["image_size"] = seg_output["original_sizes"][0]
                data_dict["scaled_size"] = tuple(seg_output["scaled_sizes"][0].tolist())
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, with_image_token=True))
        elif self.is_multimodal:
            if hasattr(self.image_processor, "crop_size"):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict["pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            if self.extra_image_processor is not None:
                if hasattr(self.extra_image_processor, "crop_size"):
                    crop_size = self.extra_image_processor.crop_size
                elif hasattr(self.extra_image_processor, "pad_size"):
                    crop_size = self.extra_image_processor.pad_size
                else:
                    crop_size = self.extra_image_processor.size
                data_dict["seg_pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
                data_dict["image_file"] = None
                data_dict["image_size"] = {"height": crop_size["height"], "width": crop_size["width"]}
                data_dict["image_info"] = {"image_file": None}
                data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, with_image_token=False))
        else:
            data_dict.update(self._get_input_ids(data_dict, with_image_token=True))
        return data_dict
