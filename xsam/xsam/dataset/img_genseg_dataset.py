import json
import os
import os.path as osp
import random
import re
import tempfile

import numpy as np
import torch
from panopticapi.utils import rgb2id
from PIL import Image
from pycocotools import mask as mask_utils

from xsam.structures import BoxMode
from xsam.utils import comm
from xsam.utils.logging import print_log
from xsam.utils.palette import get_palette

from .img_base_dataset import ImgBaseDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.format import format_cat_name, format_parts_of_cat_name
from .utils.mask import decode_mask
from .utils.panoptic import IdGenerator


class ImgGenSegDataset(ImgBaseDataset):
    def __init__(
        self,
        *args,
        task_name="img_genseg",
        use_full_cat=False,
        use_variant_cat=False,
        use_binary_cls=False,
        caption_data_path=None,
        pan_segmap_folder=None,
        sem_segmap_folder=None,
        segmap_suffix=".png",
        label_shift=0,
        **kwargs,
    ):
        super().__init__(
            *args,
            task_name=task_name,
            use_full_cat=use_full_cat,
            use_variant_cat=use_variant_cat,
            use_binary_cls=use_binary_cls,
            caption_data_path=caption_data_path,
            pan_segmap_folder=pan_segmap_folder,
            sem_segmap_folder=sem_segmap_folder,
            segmap_suffix=segmap_suffix,
            label_shift=label_shift,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.use_full_cat = kwargs.get("use_full_cat", False)
        self.use_variant_cat = kwargs.get("use_variant_cat", False)
        self.use_binary_cls = kwargs.get("use_binary_cls", False)
        self.caption_data_path = kwargs.get("caption_data_path", None)
        self.pan_segmap_folder = kwargs.get("pan_segmap_folder", None)
        self.sem_segmap_folder = kwargs.get("sem_segmap_folder", None)
        self.segmap_suffix = kwargs.get("segmap_suffix", ".png")
        self.label_shift = kwargs.get("label_shift", 0)

    @staticmethod
    def _clean_cat_name(cat_name):
        cat_name = re.sub(r"-merged", "", cat_name)
        cat_name = re.sub(r"-stuff", "", cat_name)
        cat_name = cat_name.strip().lower()
        return cat_name

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
        cat_id_to_name = {x["id"]: format_cat_name(self._clean_cat_name(x["name"])) for x in cats}
        cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

        thing_cats = [x for x in cats]
        thing_cat_ids = [x["id"] for x in thing_cats]
        thing_cat_id_to_contiguous_id = {
            cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in thing_cat_ids
        }
        thing_cat_id_to_name = {
            thing_cat_id_to_contiguous_id[x["id"]]: format_cat_name(self._clean_cat_name(x["name"]))
            for x in thing_cats
        }
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
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            background_label=self.background_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _set_semantic_metadata(self, coco_data, **kwargs):
        return self._set_panoptic_metadata(coco_data, **kwargs)

    def _set_panoptic_metadata(self, coco_data, **kwargs):
        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats])
        if "color" not in cats[0]:
            cat_colors = get_palette("random", len(cats))
            for cat, color in zip(cats, cat_colors):
                cat["color"] = color
        cat_colors = [x["color"] for x in sorted(cats, key=lambda x: x["id"])]
        dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(cats)}
        cat_id_to_name = {x["id"]: format_cat_name(self._clean_cat_name(x["name"])) for x in cats}
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
        thing_cat_id_to_name = {
            thing_cat_id_to_contiguous_id[x["id"]]: format_cat_name(self._clean_cat_name(x["name"]))
            for x in thing_cats
        }
        stuff_cat_id_to_name = {
            stuff_cat_id_to_contiguous_id[x["id"]]: format_cat_name(self._clean_cat_name(x["name"]))
            for x in stuff_cats
        }
        thing_cat_id_to_color = {
            thing_cat_id_to_contiguous_id[x["id"]]: cat_colors[thing_cat_id_to_contiguous_id[x["id"]]]
            for x in thing_cats
        }
        stuff_cat_id_to_color = {
            stuff_cat_id_to_contiguous_id[x["id"]]: cat_colors[stuff_cat_id_to_contiguous_id[x["id"]]]
            for x in stuff_cats
        }
        id_generator = IdGenerator({cat["id"]: cat for cat in cats})

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path,
            segmap_suffix=self.segmap_suffix,
            pan_segmap_folder=self.pan_segmap_folder,
            sem_segmap_folder=self.sem_segmap_folder,
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
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            background_label=self.background_label,
            id_generator=id_generator,
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
                    neg_num = max(self.num_class - len(pos_cat_ids), 0)
                    neg_cat_ids = _sample(neg_cat_ids, random.randint(0, neg_num))
                    sampled_cat_ids = _sample(pos_cat_ids + neg_cat_ids)
                else:
                    sampled_cat_ids = _sample(cat_ids)
            elif self.use_full_cat and not self.use_variant_cat:
                sampled_cat_ids = _sample(cat_ids)
            elif not self.use_full_cat and self.use_variant_cat:
                neg_num = max(self.num_class - len(pos_cat_ids), 0)
                neg_cat_ids = _sample(neg_cat_ids, random.randint(0, neg_num))
                sampled_cat_ids = _sample(pos_cat_ids + neg_cat_ids)
            elif not self.use_full_cat and not self.use_variant_cat:
                sampled_cat_ids = _sample(pos_cat_ids, self.num_class)
                sampled_anns = [anns[ann_cat_ids.index(cat_id)] for cat_id in sampled_cat_ids]
        else:
            sampled_cat_ids = cat_ids

        for ann in sampled_anns:
            ann["category_id"] = sampled_cat_ids.index(ann["category_id"])

        return sampled_cat_ids, sampled_anns

    def _load_semantic_data(self, coco_data):
        rets = []
        coco_api = COCO(dataset=coco_data)
        cats = coco_data["categories"] if not self.use_binary_cls else [{"id": 0, "name": "foreground", "isthing": 1}]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_ids2names = {
            cat["id"]: format_cat_name(format_parts_of_cat_name(self._clean_cat_name(cat["name"]))) for cat in cats
        }
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
                if self.use_binary_cls:
                    ann["category_id"] = 0
                anns.append(ann)

            if len(anns) == 0:
                self.woann_cnt += 1
                continue

            # random sample cat_ids to shuffle the order
            sampled_cat_ids, sampled_anns = self._sample_cats(cat_ids, anns)
            sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

            img_info = {
                "image_id": _img_info["id"],
                "file_name": _img_info["file_name"],
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
        return rets

    def _load_instance_data(self, coco_data):
        rets = []
        coco_api = COCO(dataset=coco_data)
        cats = coco_data["categories"] if not self.use_binary_cls else [{"id": 0, "name": "foreground", "isthing": 1}]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_ids2names = {cat["id"]: format_cat_name(self._clean_cat_name(cat["name"])) for cat in cats}
        img_ids = sorted(coco_api.getImgIds())
        for img_id in img_ids:
            _img_info = coco_api.loadImgs(img_id)[0]
            ann_ids = coco_api.getAnnIds(imgIds=[img_id])
            _anns = coco_api.loadAnns(ann_ids)

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
                if self.use_binary_cls:
                    ann["category_id"] = 0
                anns.append(ann)

            if len(anns) == 0:
                self.woann_cnt += 1
                continue

            # random sample cat_ids to shuffle the order
            sampled_cat_ids, sampled_anns = self._sample_cats(cat_ids, anns)
            sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

            img_info = {
                "image_id": _img_info["id"],
                "file_name": _img_info.get("file_name", _img_info.get("coco_url", "").split("/")[-1]),
                "height": _img_info["height"],
                "width": _img_info["width"],
            }

            rets.append(
                {
                    "image_id": _img_info["id"],
                    "image_file": _img_info.get("file_name", _img_info.get("coco_url", "").split("/")[-1]),
                    "image_size": (_img_info["height"], _img_info["width"]),
                    "caption": None,
                    "annotations": sampled_anns,
                    "sampled_cats": sampled_cat_names,
                    "sampled_labels": sampled_cat_ids,
                    "image_info": img_info,
                }
            )
        return rets

    def _load_panoptic_data(self, coco_data):
        cats = coco_data["categories"] if not self.use_binary_cls else [{"id": 0, "name": "foreground", "isthing": 1}]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_ids2names = {cat["id"]: format_cat_name(self._clean_cat_name(cat["name"])) for cat in cats}

        rets = []
        coco_data["images"] = sorted(coco_data["images"], key=lambda x: x["id"])
        coco_data["annotations"] = sorted(coco_data["annotations"], key=lambda x: x["image_id"])
        for _img_info, _ann_info in zip(coco_data["images"], coco_data["annotations"]):
            img_id = _img_info["id"]
            assert img_id == _ann_info["image_id"]
            seg_map_path = _img_info["file_name"].replace(".jpg", ".png")

            segments_info = (
                _ann_info["segments_info"]
                if not self.use_binary_cls
                else [{**segment_info, "category_id": 0} for segment_info in _ann_info["segments_info"]]
            )
            if len(segments_info) == 0:
                self.woann_cnt += 1
                continue

            # random sample cat_ids to shuffle the order
            sampled_cat_ids, sampled_segments_info = self._sample_cats(cat_ids, segments_info)
            sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

            img_info = {
                "image_id": _img_info["id"],
                "file_name": _img_info["file_name"],
                "height": _img_info["height"],
                "width": _img_info["width"],
            }

            rets.append(
                {
                    "image_id": _img_info["id"],
                    "image_file": _img_info["file_name"],
                    "image_size": (_img_info["height"], _img_info["width"]),
                    "caption": None,
                    "seg_map": seg_map_path,
                    "segments_info": sampled_segments_info,
                    "sampled_cats": sampled_cat_names,
                    "sampled_labels": sampled_cat_ids,
                    "image_info": img_info,
                }
            )

        return rets

    def _decode_semantic_data(self, data_dict):
        sampled_labels = data_dict["sampled_labels"]
        height, width = data_dict["image_size"]
        anns = data_dict["annotations"]
        mask_labels = []
        class_labels = []

        semseg_map = None
        if self.sem_segmap_folder is not None:
            semseg_map = Image.open(
                osp.join(self.sem_segmap_folder, data_dict["image_file"].replace(".jpg", self.segmap_suffix))
            ).convert("RGB")
            semseg_map = np.array(semseg_map)
            if self.label_shift != 0:
                semseg_map = semseg_map + self.label_shift
                semseg_map[semseg_map == self.label_shift] = self.ignore_value

        for ann in anns:
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

        return mask_labels, class_labels

    def _decode_instance_data(self, data_dict):
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

        return mask_labels, class_labels

    def _decode_panoptic_data(self, data_dict):
        segments_info = data_dict.get("segments_info", None)
        seg_map_path = data_dict.get("seg_map", None)
        if seg_map_path is None:
            height, width = data_dict["image_size"]
            mask_labels = torch.zeros((0, height, width))
            class_labels = torch.zeros((0,))
        else:
            # TODO: upsample the seg_map to the same size as the image
            seg_map = Image.open(osp.join(self.pan_segmap_folder, seg_map_path)).convert("RGB")
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

        return mask_labels, class_labels

    def _load_ann_data(self):
        with open(self.data_path, "r") as f:
            coco_data = json.load(f)

        cache_dir = tmp_file = None
        if self.data_mode == "eval" and "instance" in self.data_name:
            base_tmp = tempfile.gettempdir()
            cache_dir = osp.join(base_tmp, "dataset_cache")
            os.makedirs(cache_dir, exist_ok=True)
            tmp_file = osp.join(cache_dir, f"{self.data_name}.json")
            if comm.get_local_rank() != 0:
                print_log(f"Waiting for {self.data_name} gt_json at {tmp_file}...", logger="current")
                comm.synchronize()
                print_log(f"Loading {self.data_name} gt_json from {tmp_file}...", logger="current")
                with open(tmp_file, "r") as f:
                    rets = json.load(f)
                self._set_metadata(coco_data, gt_json=tmp_file)
                del coco_data
                return rets

        if "panoptic" in self.data_name:
            rets = self._load_panoptic_data(coco_data)
        elif "semantic" in self.data_name:
            rets = self._load_semantic_data(coco_data)
        elif "instance" in self.data_name:
            rets = self._load_instance_data(coco_data)
        else:
            raise ValueError(f"Invalid dataset type: {self.data_name}")

        if tmp_file is not None:
            print_log(f"Saving {self.data_name} gt_json to {tmp_file}...", logger="current")
            with open(tmp_file, "w") as f:
                json.dump(rets, f)
            comm.synchronize()
            self._set_metadata(coco_data, gt_json=tmp_file)
        else:
            self._set_metadata(coco_data)

        del coco_data
        return rets

    def _decode_ann_data(self, data_dict):
        if "panoptic" in self.data_name:
            mask_labels, class_labels = self._decode_panoptic_data(data_dict)
        elif "semantic" in self.data_name:
            mask_labels, class_labels = self._decode_semantic_data(data_dict)
        elif "instance" in self.data_name:
            mask_labels, class_labels = self._decode_instance_data(data_dict)
        else:
            raise ValueError(f"Invalid dataset type: {self.data_name}")

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
            }
        )

        return data_dict
