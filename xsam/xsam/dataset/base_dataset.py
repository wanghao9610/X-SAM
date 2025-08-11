import copy
import os

import torch
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string
from PIL import Image
from torch.utils.data import Dataset
from xtuner.dataset.utils import expand2square
from xtuner.registry import BUILDER, MAP_FUNC

from xsam.utils.logging import print_log

from ..utils.constants import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PSTART_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEFAULT_TASKS,
)
from .utils.catalog import MetadataCatalog
from .utils.encode import encode_fn

SPECIAL_TOKENS = [DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN, DEFAULT_CLS_TOKEN]
TASK_MODALITY_LENGTH = {k: int(i * 512) for i, k in enumerate(DEFAULT_TASKS)}

debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
debug_iter = 200


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path,
        image_folder,
        gt_image_folder=None,
        image_processor=None,
        tokenizer=None,
        task_name="seg",
        data_name="",
        data_mode="train",
        use_random_cat=False,
        special_tokens=None,
        cond_type="phrase",
        extra_image_processor=None,
        preprocess_fn=None,
        postprocess_fn=None,
        dataset_map_fn=None,
        template_map_fn=None,
        max_length=2048,
        task_length=None,
        pad_image_to_square=False,
        output_ids_with_output=True,
        ignore_label=255,
        sample_num=134,
        repeats_scale=1.0,
        **kwargs,
    ):
        super().__init__()

        assert task_name in DEFAULT_TASKS, f"Invalid dataset type: {task_name}"
        assert data_mode in ["train", "eval", "infer"], f"Invalid dataset mode: {data_mode}"
        assert cond_type in ["phrase", "cls", "all"], f"Invalid cond_type: {cond_type}"
        self.task_name = task_name
        self.data_name = data_name
        self.data_mode = data_mode
        self.use_random_cat = use_random_cat
        self.data_path = data_path
        self.image_folder = image_folder
        self.gt_image_folder = gt_image_folder
        self.pad_image_to_square = pad_image_to_square
        self.max_length = max_length
        self.task_length = TASK_MODALITY_LENGTH[task_name] if task_length is None else task_length
        self.ignore_label = ignore_label
        self.sample_num = sample_num
        self.output_ids_with_output = output_ids_with_output
        self.cond_type = cond_type
        self.repeats_scale = repeats_scale
        self.repeats = 1.0

        if isinstance(tokenizer, dict) or isinstance(tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            tokenizer = BUILDER.build(tokenizer)

        if isinstance(dataset_map_fn, str):
            map_fn_obj = MAP_FUNC.get(dataset_map_fn) or get_object_from_string(dataset_map_fn)
            if map_fn_obj is not None:
                dataset_map_fn = map_fn_obj
            else:
                raise TypeError(
                    "dataset_map_fn must be a function or a "
                    "registered function's string in MAP_FUNC, "
                    f"but got a string of '{dataset_map_fn}'"
                )
        elif (
            isinstance(dataset_map_fn, dict)
            or isinstance(dataset_map_fn, Config)
            or isinstance(dataset_map_fn, ConfigDict)
        ):
            dataset_map_fn = BUILDER.build(dataset_map_fn)

        if (
            isinstance(template_map_fn, dict)
            or isinstance(template_map_fn, Config)
            or isinstance(template_map_fn, ConfigDict)
        ):
            template_map_fn = BUILDER.build(template_map_fn)

        if (
            isinstance(postprocess_fn, dict)
            or isinstance(postprocess_fn, Config)
            or isinstance(postprocess_fn, ConfigDict)
        ):
            postprocess_fn = BUILDER.build(postprocess_fn)

        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.tokenizer = tokenizer

        if special_tokens is not None:
            assert all(
                token in SPECIAL_TOKENS for token in special_tokens
            ), f"special_tokens must be a subset of {SPECIAL_TOKENS}"
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

            self.seg_token_idx = -1
            self.cls_token_idx = -1
            self.pstart_token_idx = -1
            self.pend_token_idx = -1

            if DEFAULT_SEG_TOKEN in special_tokens:
                self.seg_token_idx = self.tokenizer(DEFAULT_SEG_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_CLS_TOKEN in special_tokens:
                self.cls_token_idx = self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_PSTART_TOKEN in special_tokens:
                self.pstart_token_idx = self.tokenizer(DEFAULT_PSTART_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_PEND_TOKEN in special_tokens:
                self.pend_token_idx = self.tokenizer(DEFAULT_PEND_TOKEN, add_special_tokens=False)["input_ids"][0]

        if (
            isinstance(image_processor, dict)
            or isinstance(image_processor, Config)
            or isinstance(image_processor, ConfigDict)
        ):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        if (
            isinstance(extra_image_processor, dict)
            or isinstance(extra_image_processor, Config)
            or isinstance(extra_image_processor, ConfigDict)
        ):
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        else:
            self.extra_image_processor = extra_image_processor

        self.custom_init(**kwargs)
        self.woann_cnt = 0
        print_log(f"Loading {self.data_name} dataset from {self.data_path}...", logger="current")
        self.data = self.load_ann_data()
        if self.woann_cnt > 0:
            print_log(f"Filtered {self.woann_cnt} images without annotations of {self.data_name}.", logger="current")

    def __len__(self):
        return int(len(self.data) * self.repeats)

    @property
    def repeats(self):
        return self._repeats * self.repeats_scale

    @property
    def modality_length(self):
        return [self.task_length] * int(len(self.data) * self.repeats)

    @property
    def source_length(self):
        return int(len(self.data) * self.repeats)

    @property
    def metadata(self):
        return self._metadata

    @repeats.setter
    def repeats(self, value=1.0):
        self._repeats = value

    def custom_init(self, **kwargs):
        pass

    def _set_metadata(self, **kwargs):
        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _get_input_ids(self, data_dict, with_image_token=True):
        if self.tokenizer is None:
            return data_dict

        if self.dataset_map_fn is not None:
            data_dict = self.dataset_map_fn(data_dict, self.output_ids_with_output)
        if self.template_map_fn is not None:
            data_dict = self.template_map_fn(data_dict)
        if self.tokenizer is not None:
            data_dict = encode_fn(
                data_dict, self.tokenizer, self.max_length, self.output_ids_with_output, with_image_token
            )
        return data_dict

    def _get_cond_ids(self, data_dict):
        if self.tokenizer is None:
            return data_dict

        input_ids = data_dict["input_ids"]
        cond_ids = [-1] * len(input_ids)
        pstart_idx = [i for i, x in enumerate(input_ids) if x == self.pstart_token_idx]
        pend_idx = [i for i, x in enumerate(input_ids) if x == self.pend_token_idx]
        cls_idx = [i for i, x in enumerate(input_ids) if x == self.cls_token_idx]

        if len(pstart_idx) == 0 and len(pend_idx) == 0 and len(cls_idx) == 0:
            return data_dict

        if self.cond_type in ["phrase", "all"]:
            for i, (ps, pe) in enumerate(zip(pstart_idx, pend_idx)):
                cond_ids[ps : pe + 1] = [i] * (pe - ps + 1)
        if self.cond_type in ["cls", "all"]:
            for i, ci in enumerate(cls_idx):
                cond_ids[ci] = i

        data_dict["cond_ids"] = cond_ids
        return data_dict

    def _get_seg_ids(self, data_dict):
        if self.tokenizer is None:
            return data_dict

        input_ids = data_dict["input_ids"]
        seg_ids = [-1] * len(input_ids)

        seg_idx = [i for i, x in enumerate(input_ids) if x == self.seg_token_idx]
        for i, idx in enumerate(seg_idx):
            seg_ids[idx] = i

        data_dict["seg_ids"] = seg_ids
        return data_dict

    def load_ann_data(self):
        data = self._load_ann_data()
        if debug_mode:
            data = data[:debug_iter] + data[-debug_iter:]
        self.data_length = len(data)
        return data

    def _load_ann_data(self):
        pass

    def _decode_mask(self):
        pass

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
                data_dict.update(self._decode_mask(data_dict))
                seg_output = self.extra_image_processor.preprocess(
                    pil_image, data_dict["mask_labels"], return_tensors="pt"
                )
                data_dict["seg_pixel_values"] = seg_output["pixel_values"][0]
                data_dict["scaled_size"] = tuple(seg_output["scaled_sizes"][0].tolist())
                data_dict["mask_labels"] = seg_output.get("mask_labels", None)
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, with_image_token=True))
            data_dict.update(self._get_cond_ids(data_dict))
            data_dict.update(self._get_seg_ids(data_dict))
        else:
            if hasattr(self.image_processor, "crop_size"):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict["pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            if self.extra_image_processor is not None:
                if hasattr(self.extra_image_processor, "crop_size"):
                    crop_size = self.extra_image_processor.crop_size
                else:
                    crop_size = self.extra_image_processor.size
                data_dict["seg_pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
                data_dict["image_info"] = {"image_file": None}
                data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
                data_dict["image_size"] = {"height": crop_size["height"], "width": crop_size["width"]}
                data_dict["mask_labels"] = torch.zeros(0, crop_size["height"], crop_size["width"])
                data_dict["class_labels"] = torch.zeros(0)
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, with_image_token=False))
            data_dict.update(self._get_cond_ids(data_dict))
            data_dict.update(self._get_seg_ids(data_dict))
        return data_dict
