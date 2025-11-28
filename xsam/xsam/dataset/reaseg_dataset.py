# Copyright (c) OpenMMLab. All rights reserved.
import json
import multiprocessing as mp
import os
import os.path as osp
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from xsam.utils.logging import print_log

from .base_dataset import BaseDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.mask import decode_mask, encode_mask


class ReaSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="reaseg",
        data_root=None,
        explain_path=None,
        explain_ratio=0.5,
        query_type="all",
        use_threads=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_path=None,
            task_name=task_name,
            data_root=data_root,
            explain_path=explain_path,
            explain_ratio=explain_ratio,
            query_type=query_type,
            use_threads=use_threads,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.data_root = kwargs.get("data_root", None)
        self.explain_path = kwargs.get("explain_path", None)
        self.explain_ratio = kwargs.get("explain_ratio", 0.5)
        self.query_type = kwargs.get("query_type", "sentence")
        self.use_threads = kwargs.get("use_threads", True)
        assert self.query_type in ["sentence", "phrase", "all"]

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=gt_json,
            data_name=self.data_name,
            query_type=self.query_type,
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _create_polygon_mask(self, mask, points, label_value=1):
        points_array = np.array([points], dtype=np.int32)
        cv2.polylines(mask, points_array, True, label_value, 1)
        cv2.fillPoly(mask, points_array, label_value)
        return mask

    def _get_ann_from_json(self, ann_json, height, width):
        try:
            with open(ann_json, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            return None, None, False

        questions = data["text"]
        shapes = [
            {
                "label": s["label"],
                "points": s["points"],
                "area": np.sum(self._create_polygon_mask(np.zeros((height, width), dtype=np.uint8), s["points"])),
            }
            for s in data["shapes"]
            if s["label"].lower() != "flag"
        ]
        shapes.sort(key=lambda x: x["area"], reverse=True)

        binary_mask = np.zeros((height, width), dtype=np.uint8)
        for shape in shapes:
            label_value = self.ignore_label if "ignore" in shape["label"].lower() else 1
            binary_mask = self._create_polygon_mask(binary_mask, shape["points"], label_value)

        ignore_mask = (binary_mask == self.ignore_label).astype(np.uint8)
        binary_mask = np.where(binary_mask == self.ignore_label, 0, binary_mask).astype(np.uint8)

        return questions, binary_mask, ignore_mask, data.get("is_sentence", False)

    def _get_ann_from_json_static(self, ann_json, height, width):
        try:
            with open(ann_json, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            return None, None, False, False
        questions = data["text"]
        shapes = [
            {
                "label": s["label"],
                "points": s["points"],
                "area": np.sum(self._create_polygon_mask(np.zeros((height, width), dtype=np.uint8), s["points"])),
            }
            for s in data["shapes"]
            if s["label"].lower() != "flag"
        ]
        shapes.sort(key=lambda x: x["area"], reverse=True)
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        for shape in shapes:
            label_value = 255 if "ignore" in shape["label"].lower() else 1
            binary_mask = self._create_polygon_mask(binary_mask, shape["points"], label_value)
        ignore_mask = (binary_mask == 255).astype(np.uint8)
        binary_mask = np.where(binary_mask == 255, 0, binary_mask).astype(np.uint8)
        return questions, binary_mask, ignore_mask, data.get("is_sentence", False)

    def _process_batch_data(self, args):
        image_folder, image_names, name2explain = args
        get_ann_from_json = self._get_ann_from_json_static

        rets = []
        for image_name in image_names:
            image_path = osp.join(image_folder, image_name)
            json_path = image_path.replace(".jpg", ".json")

            try:
                pil_image = Image.open(image_path)
                width, height = pil_image.size
                explain = name2explain[image_name] if name2explain else None
                questions, binary_mask, ignore_mask, is_sentence = get_ann_from_json(json_path, height, width)

                if binary_mask is None or binary_mask.sum() == 0:
                    continue
                if self.query_type == "sentence" and not is_sentence:
                    continue
                if self.query_type == "phrase" and is_sentence:
                    continue

                image_info = {
                    "id": image_name,
                    "file_name": image_name,
                    "height": height,
                    "width": width,
                }
                annotations = []
                for i, question in enumerate(questions):
                    annotations.append(
                        {
                            "id": f"{image_name}_{i}",
                            "image_id": image_name,
                            "category_id": i,
                            "explain": explain,
                            "question": question,
                            "is_sentence": is_sentence,
                            "segmentation": encode_mask(binary_mask),
                            "ignore_mask": encode_mask(ignore_mask),
                            "area": int(np.sum(binary_mask)),
                            "bbox": [0, 0, width, height],
                            "iscrowd": 0,
                        }
                    )
                rets.append((image_info, annotations))
            except Exception as e:
                print_log(f"Error processing {image_name}: {e}", logger="current")
                continue

        return rets if rets else None

    def _create_polygon_mask(self, mask, points, label_value=1):
        points_array = np.array([points], dtype=np.int32)

        points_array = np.array([points], dtype=np.int32)
        cv2.polylines(mask, points_array, True, label_value, 1)
        cv2.fillPoly(mask, points_array, label_value)
        return mask

    def _convert_to_coco_format(self):
        if self.explain_path:
            with open(self.explain_path, "r") as f:
                explain_data = json.load(f)
            name2explain = {item["image"]: item["outputs"] for item in explain_data}
        else:
            name2explain = None

        num_workers = min(16, max(1, mp.cpu_count() // 2))
        print_log(f"Using {num_workers} workers for processing images", logger="current")
        coco_data = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "question"}]}
        image_names = [f for f in os.listdir(self.image_folder) if f.endswith(".jpg")]

        batch_size = max(32, min(128, len(image_names) // (num_workers * 4)))
        batches = []
        for i in range(0, len(image_names), batch_size):
            batch_names = image_names[i : i + batch_size]
            batch_name2explain = (
                {name: name2explain[name] for name in batch_names if name2explain and name in name2explain}
                if name2explain
                else None
            )
            batches.append((self.image_folder, batch_names, batch_name2explain))

        print_log(
            f"Processing {len(image_names)} images in {len(batches)} batches (batch_size={batch_size})",
            logger="current",
        )

        if self.use_threads:
            print_log(f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks", logger="current")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._process_batch_data, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Processing {self.data_name}", ncols=80):
                    res = future.result()
                    if res is not None:
                        for image_info, annotations in res:
                            coco_data["images"].append(image_info)
                            coco_data["annotations"].extend(annotations)
                    else:
                        self.woann_cnt += 1
        else:
            with mp.Pool(num_workers) as pool:
                chunksize = max(1, min(4, len(batches) // num_workers))
                for res in tqdm(
                    pool.imap_unordered(self._process_batch_data, batches, chunksize=chunksize),
                    total=len(batches),
                    desc=f"Processing {self.data_name}",
                    ncols=80,
                ):
                    if res is not None:
                        for image_info, annotations in res:
                            coco_data["images"].append(image_info)
                            coco_data["annotations"].extend(annotations)
                    else:
                        self.woann_cnt += 1
        return coco_data

    def _load_ann_data(self):
        coco_data = self._convert_to_coco_format()

        rets = []
        coco_api = COCO(dataset=coco_data)
        img_ids = sorted(coco_api.getImgIds())
        for img_id in img_ids:
            img_info = coco_api.loadImgs([img_id])[0]
            ann_ids = coco_api.getAnnIds(imgIds=[img_id])
            anns = coco_api.loadAnns(ann_ids)

            if len(anns) == 0:
                self.woann_cnt += 1
                continue

            img_info = {
                "file_name": img_info["file_name"],
                "image_id": img_info["id"],
                "height": img_info["height"],
                "width": img_info["width"],
            }

            if self.data_mode == "train":
                ques = [ann.pop("question") for ann in anns]
                explain = [ann.pop("explain") for ann in anns]
                is_sentence = [ann.pop("is_sentence") for ann in anns]

                assert len(set(explain)) == 1 and len(set(is_sentence)) == 1
                rets.append(
                    {
                        "image_file": img_info["file_name"],
                        "image_id": img_info["image_id"],
                        "image_size": (img_info["height"], img_info["width"]),
                        "sampled_sents": ques,
                        "annotations": anns,
                        "image_info": {**img_info, "phrases": ques},
                        "explain": explain[0],
                        "is_sentence": is_sentence[0],
                    }
                )
            else:
                for i, ann in enumerate(anns):
                    ann["category_id"] = 0
                    que = ann.pop("question")
                    explain = ann.pop("explain")
                    is_sentence = ann.pop("is_sentence")
                    rets.append(
                        {
                            "image_file": img_info["file_name"],
                            "image_id": img_info["image_id"],
                            "image_size": (img_info["height"], img_info["width"]),
                            "sampled_sents": [que],
                            "annotations": [ann],
                            "image_info": {**img_info, "sample_id": i, "phrases": [que]},
                            "explain": explain,
                            "is_sentence": is_sentence,
                        }
                    )

        if self.data_mode != "train":
            base_temp = tempfile.gettempdir()
            cache_dir = osp.join(base_temp, "xsam_cache")
            os.makedirs(cache_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(dir=cache_dir)
            print_log(f"Writing {self.data_name} gt_json to {temp_dir}...", logger="current")
            temp_file = osp.join(temp_dir, f"{self.data_name}.json")
            with open(temp_file, "w") as f:
                json.dump(rets, f)
            self._set_metadata(gt_json=temp_file)
        else:
            self._set_metadata()

        del coco_data
        return rets

    def _decode_mask(self, data_dict):
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

        return data_dict
