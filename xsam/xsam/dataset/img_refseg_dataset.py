import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce

import numpy as np
import torch
from tqdm import tqdm

from xsam.utils import comm
from xsam.utils.logging import print_log

from .img_base_dataset import ImgBaseDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.mask import decode_mask
from .utils.refer import REFER


class ImgRefSegDataset(ImgBaseDataset):
    def __init__(
        self,
        *args,
        task_name="img_refseg",
        dataset=None,
        data_split=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_path=None,
            dataset=dataset,
            task_name=task_name,
            data_split=data_split,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.dataset = kwargs.get("dataset", None)
        self.data_split = kwargs.get("data_split", None)

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=gt_json,
            data_name=self.data_name,
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            background_label=self.background_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _convert_to_coco_format(self):
        refer_api = REFER(self.data_root, self.dataset)
        coco_data = {"images": [], "annotations": [], "categories": []}

        # images
        for img_id, img in refer_api.Imgs.items():
            ref = refer_api.imgToRefs[img_id]
            if ref[0]["split"] != self.data_split:
                continue
            coco_data["images"].append(
                {
                    "id": img_id,
                    "file_name": img["file_name"],
                    "height": img["height"],
                    "width": img["width"],
                }
            )

        # annotations
        for ann_id, ann in refer_api.Anns.items():
            assert (isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0) or isinstance(
                ann["segmentation"], dict
            )
            cur_ann = {
                "id": ann_id,
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "segmentation": ann["segmentation"],
                "area": ann["area"],
                "bbox": ann["bbox"],
                "iscrowd": ann.get("iscrowd", 0),
            }
            ref = refer_api.annToRef.get(ann_id, None)
            # NOTE: one ref may have multiple sentences, but only one annotation.
            if ref:
                if ref["split"] != self.data_split:
                    continue
                cur_ann["refer_sents"] = [sent for sent in ref["sentences"]]

                # only add the annotation if it has refer expressions
                coco_data["annotations"].append(cur_ann)

        # categories as placeholder
        for cat_id, cat_name in refer_api.Cats.items():
            coco_data["categories"].append({"id": cat_id, "name": cat_name})

        return coco_data

    def _load_ann_data(self):
        cache_dir = temp_file = None
        if self.data_mode == "eval":
            base_temp = tempfile.gettempdir()
            cache_dir = osp.join(base_temp, "dataset_cache")
            os.makedirs(cache_dir, exist_ok=True)
            temp_file = osp.join(cache_dir, f"{self.data_name}.json")
            if comm.get_local_rank() != 0:
                print_log(f"Waiting for {self.data_name} gt_json at {temp_file}...", logger="current")
                comm.synchronize()
                print_log(f"Loading {self.data_name} gt_json from {temp_file}...", logger="current")
                with open(temp_file, "r") as f:
                    rets = json.load(f)
                self._set_metadata(gt_json=temp_file)
                return rets

        coco_data = self._convert_to_coco_format()
        coco_api = COCO(dataset=coco_data)
        img_ids = sorted(coco_api.getImgIds())

        rets = []
        for img_id in img_ids:
            _img_info = coco_api.loadImgs(img_id)[0]
            ann_ids = coco_api.getAnnIds(imgIds=[img_id])
            anns = coco_api.loadAnns(ann_ids)
            _anns = [
                (
                    dict(ann, segmentation=ann["segmentation"][0])
                    if isinstance(ann["segmentation"], list) and isinstance(ann["segmentation"][0], dict)
                    else ann
                )
                for ann in anns
            ]
            if len(_anns) == 0:
                self.woann_cnt += 1
                continue

            img_info = {
                "image_id": _img_info["id"],
                "file_name": _img_info["file_name"],
                "height": _img_info["height"],
                "width": _img_info["width"],
            }

            ann_sents = [sorted(list(set(x["sent"].lower() for x in ann.pop("refer_sents")))) for ann in _anns]
            if self.data_split == "train":
                num_combinations = sum(len(x) for x in ann_sents)
                sent_combinations = list(itertools.islice(itertools.product(*ann_sents), num_combinations))
                anns = [copy.deepcopy(ann) for ann in _anns]
            else:
                sent_combinations = [sum(ann_sents, [])]
                anns = sum(
                    [[copy.deepcopy(ann) for _ in range(len(ann_sent))] for ann, ann_sent in zip(_anns, ann_sents)], []
                )

            for sent_combination in sent_combinations:
                assert len(sent_combination) == len(anns)
                sampled_anns = copy.deepcopy(anns)
                sampled_sents = list(sent_combination)

                for i in range(max(math.ceil(len(sampled_sents) / self.num_class), 1)):
                    cur_sampled_sents = (
                        copy.deepcopy(sampled_sents[i * self.num_class : (i + 1) * self.num_class])
                        if self.num_class > 0
                        else sampled_sents
                    )
                    cur_sampled_anns = (
                        copy.deepcopy(sampled_anns[i * self.num_class : (i + 1) * self.num_class])
                        if self.num_class > 0
                        else sampled_anns
                    )

                    for sampled_sent, sampled_ann in zip(cur_sampled_sents, cur_sampled_anns):
                        sampled_ann["category_id"] = cur_sampled_sents.index(sampled_sent)

                    rets.append(
                        {
                            "image_id": _img_info["id"],
                            "image_file": _img_info["file_name"],
                            "image_size": (_img_info["height"], _img_info["width"]),
                            "sampled_sents": cur_sampled_sents,
                            "annotations": cur_sampled_anns,
                            "image_info": {**img_info, "phrases": cur_sampled_sents, "sample_id": i},
                        }
                    )

        if temp_file is not None:
            print_log(f"Saving {self.data_name} gt_json to {cache_dir}...", logger="current")
            with open(temp_file, "w") as f:
                json.dump(rets, f)
            comm.synchronize()
            self._set_metadata(gt_json=temp_file)
        else:
            self._set_metadata()

        del coco_data
        return rets

    def _decode_ann_data(self, data_dict):
        height, width = data_dict["image_size"]
        sampled_anns = data_dict["annotations"]
        mask_labels = []
        class_labels = []
        for ann in sampled_anns:
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


class ImgGRefSegDataset(ImgRefSegDataset):
    def __init__(self, *args, use_threads=False, **kwargs):
        super().__init__(*args, use_threads=use_threads, **kwargs)

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.use_threads = kwargs.get("use_threads", False)

    def _process_batch_data_worker(self, batch_items, refer_api):
        """Worker function to process a batch of image references."""
        results = []

        for img_id, _img_info, refs in batch_items:
            img_info = {
                "file_name": _img_info["file_name"],
                "image_id": _img_info["id"],
                "height": _img_info["height"],
                "width": _img_info["width"],
            }

            sample_id = 0
            for ref in refs:
                ann_ids = refer_api.getAnnIds(ref_ids=[ref["ref_id"]])
                _anns = refer_api.loadAnns(ann_ids)
                _anns = [
                    (
                        dict(_ann, segmentation=_ann["segmentation"][0])
                        if isinstance(_ann["segmentation"], list) and isinstance(_ann["segmentation"][0], dict)
                        else _ann
                    )
                    for _ann in _anns
                ]
                if len(_anns) == 0:
                    continue

                # sampled_sents = [", ".join(sorted(list(set(x["sent"] for x in ref.pop("sentences")))))]
                sampled_sents = sorted(list(set(x["sent"] for x in ref.pop("sentences"))))
                if self.data_split == "train":
                    raise NotImplementedError("Training for GRefSegDataset is not implemented yet.")
                else:
                    for i in range(len(sampled_sents)):
                        sample_id += 1
                        results.append(
                            {
                                "image_file": _img_info["file_name"],
                                "image_id": _img_info["id"],
                                "image_size": (_img_info["height"], _img_info["width"]),
                                "sampled_sents": [sampled_sents[i]],
                                "annotations": _anns,
                                "image_info": {**img_info, "sample_id": sample_id, "phrases": [sampled_sents[i]]},
                            }
                        )

        return results if results else None

    def _load_ann_data(self):
        cache_dir = temp_file = None
        if self.data_split != "train":
            base_temp = tempfile.gettempdir()
            cache_dir = osp.join(base_temp, "dataset_cache")
            os.makedirs(cache_dir, exist_ok=True)
            temp_file = osp.join(cache_dir, f"{self.data_name}.json")
            if comm.get_local_rank() != 0:
                print_log(f"Waiting for {self.data_name} gt_json at {temp_file}...", logger="current")
                comm.synchronize()
                print_log(f"Loading {self.data_name} gt_json from {temp_file}...", logger="current")
                with open(temp_file, "r") as f:
                    rets = json.load(f)
                self._set_metadata(gt_json=temp_file)
                return rets

        refer_api = REFER(self.data_root, self.dataset)

        # Get all image items that match the data split
        image_items = []
        for img_id, _img_info in refer_api.Imgs.items():
            refs = refer_api.imgToRefs[img_id]
            if refs[0]["split"] == self.data_split:
                image_items.append((img_id, _img_info, refs))

        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(f"Using {num_workers} workers for processing images", logger="current")

        # Create batches
        batch_size = max(16, min(64, len(image_items) // num_workers))
        batches = [image_items[i : i + batch_size] for i in range(0, len(image_items), batch_size)]

        print_log(
            f"Processing {len(image_items)} images in {len(batches)} batches (batch_size={batch_size})",
            logger="current",
        )

        rets = []
        if self.use_threads:
            print_log(f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks", logger="current")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                process_func = partial(self._process_batch_data_worker, refer_api=refer_api)
                futures = [executor.submit(process_func, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Processing {self.data_name}", ncols=80):
                    res = future.result()
                    if res is not None:
                        rets.extend(res)
                    else:
                        self.woann_cnt += 1
        else:
            chunksize = max(1, len(batches) // num_workers // 2)
            with mp.Pool(num_workers) as pool:
                process_func = partial(self._process_batch_data_worker, refer_api=refer_api)
                for res in tqdm(
                    pool.imap_unordered(process_func, batches, chunksize=chunksize),
                    total=len(batches),
                    desc=f"Processing {self.data_name}",
                    ncols=80,
                ):
                    if res is not None:
                        rets.extend(res)
                    else:
                        self.woann_cnt += 1

        if temp_file is not None:
            print_log(f"Saving {self.data_name} gt_json to {cache_dir}...", logger="current")
            with open(temp_file, "w") as f:
                json.dump(rets, f)
            comm.synchronize()
            self._set_metadata(gt_json=temp_file)
        else:
            self._set_metadata()

        return rets

    def _decode_mask(self, data_dict):
        height, width = data_dict["image_size"]
        sampled_anns = data_dict["annotations"]
        mask_labels = []
        class_labels = [0]
        for ann in sampled_anns:
            segmentation = ann["segmentation"]
            binary_mask = decode_mask(segmentation, height, width)
            mask_labels.append(binary_mask)

        mask_labels = reduce(np.logical_or, mask_labels, np.zeros((height, width)))
        mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(mask_labels.copy()))])
        class_labels = torch.tensor(np.array(class_labels), dtype=torch.int64)

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
            }
        )

        return data_dict
