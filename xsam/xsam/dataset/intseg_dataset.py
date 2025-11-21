import json
import multiprocessing as mp
import os
import os.path as osp
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm

from xsam.utils.logging import print_log

from ..structures import BoxMode
from .utils.coco import COCO_INSTANCE_CATEGORIES
from .utils.mask import decode_mask, encode_mask
from .utils.vprompt import enhance_with_circles
from .vgdseg_dataset import VGDSegDataset


class IntSegDataset(VGDSegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process_visual_prompts(self, ann, height, width):
        visual_prompt_keys = [
            "point_visual_prompt_mask",
            "scribble_visual_prompt_mask",
            "box_visual_prompt_mask",
            "mask_visual_prompt_mask",
        ]
        visual_prompts = {}
        for key in visual_prompt_keys:
            if key in ann:
                new_key = key.replace("_mask", "")
                if key == "point_visual_prompt_mask":
                    mask = decode_mask(ann.pop(key), height, width)
                    mask = enhance_with_circles(mask, radius=self.point_radius)
                elif key == "scribble_visual_prompt_mask":
                    mask = decode_mask(ann.pop(key), height, width)
                    mask = enhance_with_circles(mask, radius=self.scribble_radius)
                else:
                    mask = decode_mask(ann.pop(key), height, width)
                if mask.sum() == 0:
                    continue
                visual_prompts[new_key] = encode_mask(mask)
        return visual_prompts

    def _process_batch_data(self, json_data_batch):
        current_process = mp.current_process()
        pid = current_process.pid

        _rets = []
        for data in tqdm(json_data_batch, desc=f"Process {pid}"):
            image_info = {
                "file_name": data["image_info"]["file_name"],
                "height": data["image_info"]["height"],
                "width": data["image_info"]["width"],
                "image_id": data["image_info"]["id"],
            }
            height, width = image_info["height"], image_info["width"]
            cur_anns = []
            for ann in data["anns"]:
                visual_prompts = self._process_visual_prompts(ann, height, width)
                if len(visual_prompts) == 0:
                    continue

                cur_ann = {
                    "id": ann["id"],
                    "image_id": image_info["image_id"],
                    "category_id": ann["category_id"],
                    "visual_prompts": visual_prompts,
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "iscrowd": ann["iscrowd"],
                    "segmentation": encode_mask(decode_mask(ann["segmentation"], height, width)),
                }
                cur_anns.append(cur_ann)

            if len(cur_anns) == 0:
                self.woann_cnt += 1
                continue

            _rets.append(
                {
                    "image_file": data["image"],
                    "image_id": image_info["image_id"],
                    "image_size": (image_info["height"], image_info["width"]),
                    "annotations": cur_anns,
                    "image_info": image_info,
                }
            )
        return _rets

    def _mp_process_ann_data(self, json_data, batch_size=1024):
        batches = [json_data[i : i + batch_size] for i in range(0, len(json_data), batch_size)]
        num_workers = min(16, max(1, mp.cpu_count() // 2))
        print_log(f"Processing {len(json_data)} samples with {num_workers} workers...", logger="current")

        results = []

        if self.use_threads:
            print_log(f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks", logger="current")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._process_batch_data, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Processing {self.data_name}", ncols=80):
                    batch_results = future.result()
                    if batch_results is not None:
                        results.extend(batch_results)
        else:
            chunksize = max(1, len(batches) // num_workers // 2)
            with mp.Pool(num_workers) as pool:
                for i, batch_results in enumerate(
                    tqdm(
                        pool.imap_unordered(self._process_batch_data, batches, chunksize=chunksize),
                        total=len(batches),
                        desc=f"Processing {self.data_name}",
                        ncols=80,
                    )
                ):
                    if batch_results is not None:
                        results.extend(batch_results)

        rets = [r for r in results if r is not None]
        return rets

    def _load_ann_data(self):
        cats = COCO_INSTANCE_CATEGORIES
        cat_ids = sorted([cat["id"] for cat in cats])

        if osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                _rets = json.load(f)
        else:
            with open(self.source_data_path, "r", encoding="utf-8", errors="ignore") as f:
                json_data = json.load(f)

            _rets = self._mp_process_ann_data(json_data)
            basedir = osp.dirname(self.data_path)
            os.makedirs(basedir, exist_ok=True)
            print_log(f"Writing {self.data_name} gt_json to {self.data_path}...", logger="current")
            with open(self.data_path, "w") as f:
                json.dump(_rets, f)

        if self.data_mode == "train":
            rets = _rets
        else:
            rets = []
            for ret in _rets:
                img_info = ret["image_info"]
                height, width = ret["image_size"]
                valid_anns = [
                    {
                        "image_file": ret["image_file"],
                        "image_id": ret["image_id"],
                        "image_info": {**img_info, "sample_id": i},
                        "image_size": ret["image_size"],
                        "annotations": [ann],
                    }
                    for i, ann in enumerate(ret["annotations"])
                    if self.visual_prompt_type in ann["visual_prompts"]
                    and decode_mask(ann["visual_prompts"][self.visual_prompt_type], height, width).sum() > 0
                ]

                if not valid_anns:
                    self.woann_cnt += 1
                    print_log(f"{ret['image_file']} has no {self.visual_prompt_type} anns", logger="current")
                    continue

                rets.extend(valid_anns)

        for ret in rets:
            sampled_cat_ids = self._sample_cats(cat_ids, ret["annotations"])
            ret["sampled_labels"] = sampled_cat_ids
            ret["contiguous_labels"] = cat_ids

        gt_json = None
        if self.data_mode == "eval":
            base_temp = tempfile.gettempdir()
            cache_dir = osp.join(base_temp, "xsam_cache")
            os.makedirs(cache_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(dir=cache_dir)
            print_log(f"Writing {self.data_name} gt_json to {temp_dir}...", logger="current")
            temp_file = osp.join(temp_dir, f"{self.data_name}.json")
            with open(temp_file, "w") as f:
                json.dump(rets, f)
            gt_json = temp_file

        self._set_metadata(cats=cats, gt_json=gt_json)

        return rets
