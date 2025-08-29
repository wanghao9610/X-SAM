import json
import os
import os.path as osp
import random
import tempfile

import numpy as np
import torch
from panopticapi.utils import rgb2id
from PIL import Image
from pycocotools import mask as mask_utils

from ..structures import BoxMode
from ..utils.constants import DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN
from ..utils.logging import print_log
from ..utils.palette import get_palette
from .base_dataset import BaseDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.mask import decode_mask

SPECIAL_TOKENS = [DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN]


class GenericSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        use_variant_cat=False,
        use_full_cat=True,
        caption_data_path=None,
        panseg_map_folder=None,
        semseg_map_folder=None,
        semseg_sufix=".png",
        label_shift=0,
        **kwargs,
    ):
        super().__init__(
            *args,
            use_variant_cat=use_variant_cat,
            use_full_cat=use_full_cat,
            caption_data_path=caption_data_path,
            panseg_map_folder=panseg_map_folder,
            semseg_map_folder=semseg_map_folder,
            semseg_sufix=semseg_sufix,
            label_shift=label_shift,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.use_variant_cat = kwargs.get("use_variant_cat", False)
        self.use_full_cat = kwargs.get("use_full_cat", True)
        self.caption_data_path = kwargs.get("caption_data_path", None)
        self.panseg_map_folder = kwargs.get("panseg_map_folder", None)
        self.semseg_map_folder = kwargs.get("semseg_map_folder", None)
        self.semseg_sufix = kwargs.get("semseg_sufix", ".png")
        self.label_shift = kwargs.get("label_shift", 0)

    def _set_instance_metadata(self, coco_data, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_colors = (
            [x["color"] for x in sorted(cats, key=lambda x: x["id"])]
            if "color" in cats[0]
            else get_palette("random", len(cats))
        )
        dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(cats)}
        cat_id_to_name = {x["id"]: x["name"] for x in cats}
        cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

        thing_cats = [x for x in cats]
        thing_cat_ids = [x["id"] for x in thing_cats]
        thing_cat_id_to_contiguous_id = {
            cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in thing_cat_ids
        }
        thing_cat_id_to_name = {thing_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in thing_cats}
        thing_cat_id_to_color = {
            thing_cat_id_to_contiguous_id[x["id"]]: cat_colors[thing_cat_id_to_contiguous_id[x["id"]]]
            for x in thing_cats
        }

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path if gt_json is None else gt_json,
            data_name=self.data_name,
            dataset_classes=cat_id_to_name,
            dataset_colors=cat_id_to_color,
            thing_classes=thing_cat_id_to_name,
            thing_colors=thing_cat_id_to_color,
            dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
            thing_dataset_id_to_contiguous_id=thing_cat_id_to_contiguous_id,
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _set_semantic_metadata(self, coco_data, **kwargs):
        return self._set_panoptic_metadata(coco_data, **kwargs)

    def _set_panoptic_metadata(self, coco_data, **kwargs):
        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_colors = (
            [x["color"] for x in sorted(cats, key=lambda x: x["id"])]
            if "color" in cats[0]
            else get_palette("random", len(cats))
        )
        dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(cats)}
        cat_id_to_name = {x["id"]: x["name"] for x in cats}
        cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

        thing_cats = [x for x in cats if x.get("isthing", None) == 1]
        thing_cat_ids = [x["id"] for x in thing_cats]
        stuff_cats = [x for x in cats if x.get("isthing", None) == 0]
        stuff_cat_ids = [x["id"] for x in stuff_cats]
        thing_cat_id_to_contiguous_id = {
            cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in thing_cat_ids
        }
        stuff_cat_id_to_contiguous_id = {
            cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in stuff_cat_ids
        }
        thing_cat_id_to_name = {thing_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in thing_cats}
        stuff_cat_id_to_name = {stuff_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in stuff_cats}
        thing_cat_id_to_color = {
            thing_cat_id_to_contiguous_id[x["id"]]: cat_colors[thing_cat_id_to_contiguous_id[x["id"]]]
            for x in thing_cats
        }
        stuff_cat_id_to_color = {
            stuff_cat_id_to_contiguous_id[x["id"]]: cat_colors[stuff_cat_id_to_contiguous_id[x["id"]]]
            for x in stuff_cats
        }

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path,
            semseg_sufix=self.semseg_sufix,
            panseg_map_folder=self.panseg_map_folder,
            semseg_map_folder=self.semseg_map_folder,
            data_name=self.data_name,
            dataset_classes=cat_id_to_name,
            dataset_colors=cat_id_to_color,
            thing_classes=thing_cat_id_to_name,
            thing_colors=thing_cat_id_to_color,
            stuff_classes=stuff_cat_id_to_name,
            stuff_colors=stuff_cat_id_to_color,
            dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
            thing_dataset_id_to_contiguous_id=thing_cat_id_to_contiguous_id,
            stuff_dataset_id_to_contiguous_id=stuff_cat_id_to_contiguous_id,
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _set_metadata(self, coco_data, **kwargs):
        if "semantic" in self.data_name:
            self._set_semantic_metadata(coco_data, **kwargs)
        elif "instance" in self.data_name:
            self._set_instance_metadata(coco_data, **kwargs)
        elif "panoptic" in self.data_name:
            self._set_panoptic_metadata(coco_data, **kwargs)
        else:
            raise ValueError(f"Invalid dataset type: {self.data_name}")

    def _sample_cats(self, cat_ids, anns):
        def _sample(items, num=None):
            num = len(items) if num is None or num < 0 else min(len(items), num)
            items = random.sample(items, num) if self.use_random_cat or self.use_variant_cat else items
            return items

        anns = _sample(anns, len(anns))
        ann_cat_ids = [ann["category_id"] for ann in anns]
        pos_cat_ids = sorted(set(ann_cat_ids))
        neg_cat_ids = sorted(set(cat_ids) - set(pos_cat_ids))

        sampled_anns = anns
        if self.data_mode == "train":
            if self.use_full_cat and self.use_variant_cat:
                if random.random() < 0.5:
                    neg_num = max(self.sample_num - len(pos_cat_ids), 0)
                    neg_cat_ids = _sample(neg_cat_ids, random.randint(0, neg_num))
                    sampled_cat_ids = _sample(pos_cat_ids + neg_cat_ids)
                else:
                    sampled_cat_ids = _sample(cat_ids)
            elif self.use_full_cat and not self.use_variant_cat:
                sampled_cat_ids = _sample(cat_ids)
            elif not self.use_full_cat and self.use_variant_cat:
                neg_num = max(self.sample_num - len(pos_cat_ids), 0)
                neg_cat_ids = _sample(neg_cat_ids, random.randint(0, neg_num))
                sampled_cat_ids = _sample(pos_cat_ids + neg_cat_ids)
            elif not self.use_full_cat and not self.use_variant_cat:
                sampled_cat_ids = _sample(pos_cat_ids, self.sample_num)
                sampled_anns = [anns[ann_cat_ids.index(cat_id)] for cat_id in sampled_cat_ids]
        else:
            sampled_cat_ids = cat_ids

        for ann in sampled_anns:
            ann["category_id"] = sampled_cat_ids.index(ann["category_id"])

        return sampled_cat_ids, sampled_anns

    def _load_ann_data(self):
        def _format_caption(caption):
            caption = caption.strip().capitalize()
            if not caption.endswith("."):
                caption = caption + "."
            return caption

        def _format_cat_names(cat_names):
            formatted_cat_names = []
            for cat_name in cat_names:
                cat_splits = cat_name.strip().split(":")
                if len(cat_splits) == 1:
                    cat_name = cat_splits[0].strip().split("_(")[0]
                if len(cat_splits) > 1:
                    assert len(cat_splits) == 2
                    main, part = cat_splits
                    main = main.split("_(")[0].replace("_", " ").replace("-", " ").strip()
                    part = part.split("_(")[0].replace("_", " ").replace("-", " ").strip()
                    if random.random() < 0.5:
                        cat_name = f"{main} {part}"
                    else:
                        cat_name = f"the {part} of the {main}"
                formatted_cat_names.append(cat_name)
            return formatted_cat_names

        with open(self.data_path, "r") as f:
            coco_data = json.load(f)

        cap_coco_api = None
        if self.caption_data_path is not None:
            cap_coco_api = COCO(self.caption_data_path)

        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_ids2names = {cat["id"]: cat["name"] for cat in cats}

        rets = []
        if "panoptic" in self.data_name:
            coco_data["images"] = sorted(coco_data["images"], key=lambda x: x["id"])
            coco_data["annotations"] = sorted(coco_data["annotations"], key=lambda x: x["image_id"])
            for _img_info, _ann_info in zip(coco_data["images"], coco_data["annotations"]):
                img_id = _img_info["id"]
                assert _img_info["id"] == _ann_info["image_id"]
                seg_map_path = _img_info["file_name"].replace(".jpg", ".png")

                if cap_coco_api is not None:
                    cap_ann_ids = cap_coco_api.getAnnIds(imgIds=[img_id])
                    cap_anns = cap_coco_api.loadAnns(cap_ann_ids)
                    caption = _format_caption(random.choice(cap_anns)["caption"])
                else:
                    caption = None

                segments_info = _ann_info["segments_info"]
                if len(segments_info) == 0:
                    self.woann_cnt += 1
                    continue

                # random sample cat_ids to shuffle the order
                sampled_cat_ids, sampled_segments_info = self._sample_cats(cat_ids, segments_info)
                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

                img_info = {
                    "file_name": _img_info["file_name"],
                    "image_id": _img_info["id"],
                    "height": _img_info["height"],
                    "width": _img_info["width"],
                }

                rets.append(
                    {
                        "image_id": _img_info["id"],
                        "image_file": _img_info["file_name"],
                        "image_size": (_img_info["height"], _img_info["width"]),
                        "caption": caption,
                        "seg_map": seg_map_path,
                        "segments_info": sampled_segments_info,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "image_info": img_info,
                    }
                )
        elif "semantic" in self.data_name:
            coco_api = COCO(dataset=coco_data)
            img_ids = sorted(coco_api.getImgIds())
            for img_id in img_ids:
                _img_info = coco_api.loadImgs(img_id)[0]
                ann_ids = coco_api.getAnnIds(imgIds=[img_id])
                _anns = coco_api.loadAnns(ann_ids)
                caption = None

                anns = []
                for ann in _anns:
                    if int(ann.get("iscrowd", 0)) != 0:
                        continue

                    ann["segmentation"] = ann["segmentation"]
                    ann["bbox_mode"] = BoxMode.XYWH_ABS
                    anns.append(ann)

                if len(anns) == 0:
                    self.woann_cnt += 1
                    continue

                # random sample cat_ids to shuffle the order
                sampled_cat_ids, sampled_anns = self._sample_cats(cat_ids, anns)
                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]
                sampled_cat_names = _format_cat_names(sampled_cat_names)

                img_info = {
                    "file_name": _img_info["file_name"],
                    "image_id": _img_info["id"],
                    "height": _img_info["height"],
                    "width": _img_info["width"],
                }

                rets.append(
                    {
                        "image_id": _img_info["id"],
                        "image_file": _img_info["file_name"],
                        "image_size": (_img_info["height"], _img_info["width"]),
                        "caption": caption,
                        "annotations": sampled_anns,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "image_info": img_info,
                    }
                )
        elif "instance" in self.data_name:
            coco_api = COCO(dataset=coco_data)
            img_ids = sorted(coco_api.getImgIds())
            for img_id in img_ids:
                _img_info = coco_api.loadImgs(img_id)[0]
                ann_ids = coco_api.getAnnIds(imgIds=[img_id])
                _anns = coco_api.loadAnns(ann_ids)

                if cap_coco_api is not None and random.random() < 0.5:
                    cap_ann_ids = cap_coco_api.getAnnIds(imgIds=[img_id])
                    cap_anns = cap_coco_api.loadAnns(cap_ann_ids)
                    caption = _format_caption(random.choice(cap_anns).get("caption", ""))
                else:
                    caption = None

                anns = []
                for ann in _anns:
                    if int(ann.get("iscrowd", 0)) != 0:
                        continue

                    segmentation = ann["segmentation"]
                    if isinstance(segmentation, dict):
                        if isinstance(segmentation["counts"], list):
                            # convert to compressed RLE
                            segmentation = mask_utils.frPyObjects(segmentation["counts"], *segmentation["size"])
                        segmentation["counts"] = segmentation["counts"].decode("utf-8")
                    else:
                        # filter out invalid polygons (< 3 points)
                        segmentation = [poly for poly in segmentation if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segmentation) == 0:
                            continue  # ignore this instance

                    ann["segmentation"] = segmentation
                    ann["bbox_mode"] = BoxMode.XYWH_ABS
                    anns.append(ann)

                if len(anns) == 0:
                    self.woann_cnt += 1
                    continue

                # random sample cat_ids to shuffle the order
                sampled_cat_ids, sampled_anns = self._sample_cats(cat_ids, anns)
                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

                img_info = {
                    "file_name": _img_info["file_name"],
                    "image_id": _img_info["id"],
                    "height": _img_info["height"],
                    "width": _img_info["width"],
                }

                rets.append(
                    {
                        "image_id": _img_info["id"],
                        "image_file": _img_info["file_name"],
                        "image_size": (_img_info["height"], _img_info["width"]),
                        "caption": caption,
                        "annotations": sampled_anns,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "image_info": img_info,
                    }
                )
        else:
            raise ValueError(f"Invalid dataset type: {self.data_name}")

        if self.data_mode == "eval" and "instance" in self.data_name:
            base_tmp = tempfile.gettempdir()
            cache_dir = osp.join(base_tmp, "xsam_cache")
            os.makedirs(cache_dir, exist_ok=True)
            tmp_dir = tempfile.mkdtemp(dir=cache_dir)
            print_log(f"Writing {self.data_name} gt_json to {tmp_dir}...", logger="current")
            tmp_file = osp.join(tmp_dir, f"{self.data_name}.json")
            with open(tmp_file, "w") as f:
                json.dump(rets, f)
            self._set_metadata(coco_data, gt_json=tmp_file)
        else:
            self._set_metadata(coco_data)

        del coco_data
        return rets

    def _decode_mask(self, data_dict):
        if "panoptic" in self.data_name:
            segments_info = data_dict.get("segments_info", None)
            seg_map_path = data_dict.get("seg_map", None)
            if seg_map_path is None:
                height, width = data_dict["image_size"]
                mask_labels = torch.zeros((0, height, width))
                class_labels = torch.zeros((0,))
            else:
                # TODO: upsample the seg_map to the same size as the image
                seg_map = Image.open(os.path.join(self.panseg_map_folder, seg_map_path)).convert("RGB")
                seg_map = rgb2id(np.array(seg_map))

                mask_labels = []
                class_labels = []
                for segment_info in segments_info:
                    cat_id = segment_info["category_id"]
                    if not segment_info["iscrowd"]:
                        mask = seg_map == segment_info["id"]
                        class_labels.append(cat_id)
                        mask_labels.append(mask)
                if len(mask_labels) == 0:
                    mask_labels = torch.zeros((0, seg_map.shape[-2], seg_map.shape[-1]))
                    class_labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_labels])
                    class_labels = torch.tensor(np.array(class_labels), dtype=torch.int64)

            del data_dict["segments_info"]
            del data_dict["seg_map"]
            data_dict.update(
                {
                    "mask_labels": mask_labels,
                    "class_labels": class_labels,
                }
            )
        elif "semantic" in self.data_name:
            sampled_labels = data_dict["sampled_labels"]
            height, width = data_dict["image_size"]
            annotations = data_dict["annotations"]
            mask_labels = []
            class_labels = []

            semseg_map = None
            if self.semseg_map_folder is not None:
                semseg_map = Image.open(
                    os.path.join(self.semseg_map_folder, data_dict["image_file"].replace(".jpg", self.semseg_sufix))
                ).convert("RGB")
                semseg_map = np.array(semseg_map)
                if self.label_shift != 0:
                    semseg_map = semseg_map + self.label_shift
                    semseg_map[semseg_map == self.label_shift] = self.ignore_label

            for ann in annotations:
                segmentation = ann.get("segmentation", None)
                if segmentation is not None:
                    binary_mask = decode_mask(segmentation, height, width)
                elif segmentation is None:
                    assert semseg_map is not None
                    binary_mask = semseg_map == sampled_labels[ann["category_id"]]

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
        elif "instance" in self.data_name:
            height, width = data_dict["image_size"]
            annotations = data_dict["annotations"]
            mask_labels = []
            class_labels = []
            for ann in annotations:
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
        else:
            raise ValueError(f"Invalid dataset type: {self.task_name}")

        return data_dict
