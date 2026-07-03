import json
import multiprocessing as mp
import os
import os.path as osp
import tempfile

from tqdm import tqdm

from xsam.structures import BoxMode
from xsam.utils import comm
from xsam.utils.logging import print_log

from .img_vgdseg_dataset import ImgVGDSegDataset
from .utils.coco import COCO_INSTANCE_CATEGORIES
from .utils.mask import decode_mask, encode_mask
from .utils.vprompt import enhance_with_circles


class ImgIntSegDataset(ImgVGDSegDataset):
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

    def _process_batch_data_worker(self, json_data_batch):
        current_process = mp.current_process()
        pid = current_process.pid

        _rets = []
        for data in tqdm(json_data_batch, desc=f"Process {pid}"):
            image_info = {
                "image_id": data["image_info"]["id"],
                "file_name": data["image_info"]["file_name"],
                "height": data["image_info"]["height"],
                "width": data["image_info"]["width"],
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
        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(f"Processing {len(json_data)} samples with {num_workers} workers...", logger="current")

        batch_size = max(512, min(128, len(json_data) // num_workers))
        batches = [json_data[i : i + batch_size] for i in range(0, len(json_data), batch_size)]

        rets = []
        chunk_size = max(1, len(batches) // num_workers // 2)
        with mp.Pool(num_workers) as pool:
            for i, batch_result in enumerate(
                tqdm(
                    pool.imap_unordered(self._process_batch_data_worker, batches, chunk_size=chunk_size),
                    total=len(batches),
                    desc=f"Processing {self.data_name}",
                    ncols=80,
                )
            ):
                if batch_result is not None:
                    rets.extend(batch_result)
        rets = [r for r in rets if r is not None]
        return rets

    def _load_ann_data(self):
        cats = COCO_INSTANCE_CATEGORIES
        cat_ids = sorted([cat["id"] for cat in cats])

        gt_json = None
        cache_dir = tmp_file = None
        if self.data_mode == "eval":
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
                self._set_metadata(cats=cats, gt_json=tmp_file)
                return rets

        if osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                _rets = json.load(f)
        else:
            with open(self.source_data_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            _rets = self._mp_process_ann_data(json_data)
            basedir = osp.dirname(self.data_path)
            os.makedirs(basedir, exist_ok=True)
            print_log(f"Saving {self.data_name} gt_json to {self.data_path}...", logger="current")
            if comm.is_main_process():
                with open(self.data_path, "w", encoding="utf-8") as f:
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

        if tmp_file is not None:
            print_log(f"Saving {self.data_name} gt_json to {cache_dir}...", logger="current")
            with open(tmp_file, "w") as f:
                json.dump(rets, f)
            comm.synchronize()
            gt_json = tmp_file

        self._set_metadata(cats=cats, gt_json=gt_json)

        return rets
