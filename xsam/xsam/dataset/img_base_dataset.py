import copy
import os.path as osp

import torch
from PIL import Image

from .base_dataset import BaseDataset
from .utils.image import expand2square


class ImgBaseDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        index = index % self.data_length
        data_dict = copy.deepcopy(self.data[index])
        if data_dict.get("image_file", None) is not None:
            image_file = data_dict["image_file"]
            pil_image = Image.open(osp.join(self.image_folder, image_file)).convert("RGB")
            if self.image_processor is not None:
                image = pil_image
                if self.expand2square:
                    image = expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                output = self.image_processor.preprocess(image, return_tensors="pt")
                pixel_values = (
                    output["pixel_values"][0] if output["pixel_values"].ndim == 4 else output["pixel_values"]
                )
                image_grid_thw = output.get("image_grid_thw", None)
                data_dict["pixel_values"] = pixel_values
                data_dict["image_grid_thw"] = image_grid_thw
            if self.extra_image_processor is not None:
                data_dict.update(self._decode_ann_data(data_dict))
                extra_output = self.extra_image_processor.preprocess(
                    pil_image, data_dict["mask_labels"], return_tensors="pt"
                )
                data_dict["extra_pixel_values"] = extra_output["pixel_values"][0]
                data_dict["scaled_size"] = extra_output["scaled_sizes"][0].tolist()
                data_dict["mask_labels"] = extra_output.get("mask_labels", None)
                data_dict["task_name"] = self.task_name
                data_dict["data_name"] = self.data_name
            data_dict.update(self._get_input_ids(data_dict, use_vision_token=True))
            data_dict.update(self._get_cond_ids(data_dict))
            data_dict.update(self._get_seg_ids(data_dict))
        else:
            if hasattr(self.image_processor, "crop_size"):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            # placeholder for crop_size
            lengths = [1600, 1536] if self.pixel_values_ndim == 2 else [384, 384]
            crop_size = (
                {"height": lengths[0], "width": lengths[1]}
                if crop_size is None or "height" not in crop_size or "width" not in crop_size
                else crop_size
            )
            data_dict["pixel_values"] = (
                torch.zeros(3, crop_size["height"], crop_size["width"])
                if self.pixel_values_ndim == 3
                else torch.zeros(crop_size["height"], crop_size["width"])
            )
            data_dict["image_grid_thw"] = torch.tensor([[1, 40, 40]]) if self.pixel_values_ndim == 2 else None
            if self.extra_image_processor is not None:
                if hasattr(self.extra_image_processor, "crop_size"):
                    crop_size = self.extra_image_processor.crop_size
                else:
                    crop_size = self.extra_image_processor.size
                data_dict["extra_pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
                data_dict["image_info"] = {"image_file": None}
                data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
                data_dict["image_size"] = {"height": crop_size["height"], "width": crop_size["width"]}
                data_dict["mask_labels"] = torch.zeros(0, crop_size["height"], crop_size["width"])
                data_dict["class_labels"] = torch.zeros(0)
                data_dict["task_name"] = self.task_name
                data_dict["data_name"] = self.data_name
            data_dict.update(self._get_input_ids(data_dict, use_vision_token=False))
            data_dict.update(self._get_cond_ids(data_dict))
            data_dict.update(self._get_seg_ids(data_dict))
        return data_dict
