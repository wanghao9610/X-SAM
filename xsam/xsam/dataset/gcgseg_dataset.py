import json
import os
from functools import reduce

import numpy as np
import torch
from PIL import Image

from .base_dataset import BaseDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.mask import decode_mask


class GCGSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="gcgseg",
        cap_data_path=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            task_name=task_name,
            cap_data_path=cap_data_path,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.cap_data_path = kwargs.get("cap_data_path", None)

    def _set_metadata(self, **kwargs):
        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path,
            data_name=self.data_name,
            cap_gt_json=self.cap_data_path,
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _remove_overlapping_intervals(self, _anns):
        anns = []
        _tokens_positive = [ann["tokens_positive"] for ann in _anns]
        sorted_tokens_positive = sorted(_tokens_positive, key=lambda x: x[0])
        tokens_positive = [
            x
            for x in reduce(
                lambda acc, x: acc + [x] if not acc or x[0] > acc[-1][1] else acc, sorted_tokens_positive, []
            )
            if x[0] < x[1] and x[0] >= 0 and x[1] >= 0
        ]
        for i, interval in enumerate(tokens_positive):
            index = _tokens_positive.index(interval)
            ann = _anns[index]
            assert ann["tokens_positive"] == interval
            ann.update({"category_id": i})
            anns.append(ann)
        return anns

    def _process_grandf_format_data(self, json_data):
        ann_id = 0
        rets = []
        for data in json_data:
            anns = []

            caption = data["caption"].strip('"').strip().lower()
            _img_info = {
                "image_id": data["image_id"],
                "file_name": data["file_name"],
            }

            grounding_items = sorted(data["groundings"].items(), key=lambda x: x[1]["token_positives"][0])
            for phrase, grounding_item in grounding_items:
                phrase = phrase.strip('"').strip().lower()
                if phrase not in caption:
                    continue

                cur_ann = {
                    "id": ann_id,
                    "phrase": phrase,
                    "tokens_positive": grounding_item["token_positives"],
                    "segmentation": grounding_item["rle_masks"],
                }
                anns.append(cur_ann)
                ann_id += 1

            anns = self._remove_overlapping_intervals(anns)

            if len(anns) == 0:
                self.woann_cnt += 1
                continue

            if data.get("height", None) is not None:
                image_size = (int(data["height"]), int(data["width"]))
            else:
                pil_image = Image.open(os.path.join(self.image_folder, data["file_name"]))
                image_size = (pil_image.height, pil_image.width)

            rets.append(
                {
                    "image_id": data["image_id"],
                    "image_file": data["file_name"],
                    "image_size": image_size,
                    "caption": caption,
                    "image_info": _img_info,
                    "annotations": anns,
                }
            )

        return rets

    def _process_refcocog_format_data(self, json_data):
        img_id = 0
        ann_id = 0
        rets = []

        for data in json_data:
            data = next(iter(data.values()))
            anns = []

            caption = data["caption"].strip('"').strip().lower()
            _img_info = {
                "image_id": data.get("image_id", img_id),
                "file_name": data["img_file_name"],
            }

            data["refs"] = sorted(data["refs"], key=lambda x: caption.find(x["sentence"].strip('"').strip().lower()))
            for ref in data["refs"]:
                phrase = ref["sentence"].strip('"').strip().lower()
                if phrase not in caption:
                    continue

                start_index = caption.find(phrase)
                end_index = start_index + len(phrase) if start_index != -1 else -1

                cur_ann = {
                    "id": ann_id,
                    "phrase": phrase,
                    "tokens_positive": [start_index, end_index],
                    "segmentation": ref["segmentation"],
                }
                anns.append(cur_ann)
                ann_id += 1

            anns = self._remove_overlapping_intervals(anns)

            if len(anns) == 0:
                self.woann_cnt += 1
                continue

            if data.get("height", None) is not None:
                image_size = (int(data["height"]), int(data["width"]))
            else:
                pil_image = Image.open(os.path.join(self.image_folder, data["img_file_name"]))
                image_size = (pil_image.height, pil_image.width)

            rets.append(
                {
                    "image_id": data.get("image_id", img_id),
                    "image_file": data["img_file_name"],
                    "image_size": image_size,
                    "caption": caption,
                    "image_info": _img_info,
                    "annotations": anns,
                }
            )
            img_id += 1

        return rets

    def _process_flickr_format_data(self, coco_data):
        rets = []
        ann_id = 0
        coco_api = COCO(dataset=coco_data)
        img_ids = sorted(coco_api.getImgIds())
        for img_id in img_ids:
            img_info = coco_api.loadImgs(img_id)[0]
            ann_ids = coco_api.getAnnIds(imgIds=[img_id])
            anns = coco_api.loadAnns(ann_ids)

            caption = img_info["caption"].strip('"').strip().lower()
            if len(caption.split(" ")) < 3 or len(anns) == 0:
                self.woann_cnt += 1
                continue

            _img_info = {
                "image_id": img_id,
                "file_name": img_info["file_name"],
            }

            new_anns = []
            anns = sorted(anns, key=lambda x: x["tokens_positive"][0][0])
            for ann in anns:
                tokens_positive = ann["tokens_positive"][0]
                phrase = caption[tokens_positive[0] : tokens_positive[1]]
                cur_ann = {
                    "id": ann.get("id", ann_id),
                    "phrase": phrase,
                    "tokens_positive": [tokens_positive[0], tokens_positive[1]],
                    "segmentation": ann["sam_mask"],
                }
                new_anns.append(cur_ann)
                ann_id += 1

            new_anns = self._remove_overlapping_intervals(new_anns)

            if img_info.get("height", None) is not None:
                image_size = (int(img_info["height"]), int(img_info["width"]))
            else:
                pil_image = Image.open(os.path.join(self.image_folder, img_info["file_name"]))
                image_size = (pil_image.height, pil_image.width)

            rets.append(
                {
                    "image_id": img_id,
                    "image_file": img_info["file_name"],
                    "image_size": image_size,
                    "caption": caption,
                    "image_info": _img_info,
                    "annotations": new_anns,
                }
            )

        return rets

    def _process_val_format_data(self, json_data):
        rets = []
        for image_data in json_data["images"]:
            image_id = image_data["id"]
            file_name = image_id + ".jpg"
            _img_info = {
                "image_id": image_id,
                "file_name": file_name,
            }

            if image_data.get("height", None) is not None:
                image_size = (int(image_data["height"]), int(image_data["width"]))
            else:
                pil_image = Image.open(os.path.join(self.image_folder, file_name))
                image_size = (pil_image.height, pil_image.width)

            rets.append(
                {
                    "image_id": image_id,
                    "image_file": file_name,
                    "image_size": image_size,
                    "image_info": _img_info,
                }
            )

        return rets

    def _load_ann_data(self):
        with open(self.data_path, "r") as f:
            json_data = json.load(f)
        self._set_metadata()

        if self.data_name in ["psg_gcgseg", "grandf_gcgseg"]:
            rets = self._process_grandf_format_data(json_data)
        elif self.data_name in ["gcgseg"]:
            rets = self._process_refcocog_format_data(json_data)
        elif self.data_name in ["flickr_gcgseg"]:
            rets = self._process_flickr_format_data(json_data)
        elif self.data_name in ["val_gcgseg", "test_gcgseg"]:
            rets = self._process_val_format_data(json_data)
        else:
            raise ValueError(f"Invalid dataset name for GCGSegDataset: {self.data_name}")

        del json_data
        return rets

    def _decode_mask(self, data_dict):
        if "annotations" not in data_dict:
            data_dict.update(
                {
                    "mask_labels": None,
                    "class_labels": None,
                }
            )
            return data_dict

        height, width = data_dict["image_size"]
        anns = data_dict["annotations"]
        mask_labels = []
        class_labels = []
        for ann in anns:
            segmentation = ann["segmentation"]
            binary_mask = decode_mask(segmentation, height, width)
            mask_labels.append(binary_mask)
            class_labels.append(ann["category_id"])

        mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_labels])
        class_labels = torch.tensor(np.array(class_labels), dtype=torch.int64)

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
            }
        )

        return data_dict
