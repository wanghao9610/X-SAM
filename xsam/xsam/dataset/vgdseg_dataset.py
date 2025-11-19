import copy
import json
import multiprocessing as mp
import os
import os.path as osp
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from skimage.measure import label, regionprops
from tqdm import tqdm
from xtuner.dataset.utils import expand2square

from xsam.structures import BoxMode
from xsam.utils.logging import print_log

from ..utils.palette import get_palette
from .base_dataset import BaseDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.mask import decode_mask, encode_mask
from .utils.vprompt import (
    generate_box_vprompt,
    generate_mask_vprompt,
    generate_point_vprompt,
    generate_scribble_vprompt,
)


class VGDSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        min_area=5,
        max_retries=1000,
        point_radius=10,
        scribble_radius=5,
        source_data_path=None,
        visual_prompt_type="point_visual_prompt",
        use_negative_sample=False,
        use_threads=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            min_area=min_area,
            max_retries=max_retries,
            point_radius=point_radius,
            scribble_radius=scribble_radius,
            source_data_path=source_data_path,
            visual_prompt_type=visual_prompt_type,
            use_negative_sample=use_negative_sample,
            use_threads=use_threads,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.min_area = kwargs.get("min_area", 5)
        self.max_retries = kwargs.get("max_retries", 1000)
        self.point_radius = kwargs.get("point_radius", 10)
        self.scribble_radius = kwargs.get("scribble_radius", 5)
        self.source_data_path = kwargs.get("source_data_path", None)
        self.visual_prompt_type = kwargs.get("visual_prompt_type", "point_visual_prompt")
        self.use_negative_sample = kwargs.get("use_negative_sample", False)
        self.use_threads = kwargs.get("use_threads", True)

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        cats = kwargs.get("cats", None)
        cat_ids = sorted([x["id"] for x in cats])
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

    def _get_visual_prompts(self, mask):
        label_mask = label(mask)
        props = [prop for prop in regionprops(label_mask) if prop.area > self.min_area]
        point_visual_prompt = generate_point_vprompt(mask, props, self.max_retries, self.point_radius)
        scribble_visual_prompt = generate_scribble_vprompt(mask, props, self.max_retries, self.scribble_radius)
        box_visual_prompt = generate_box_vprompt(mask, props)
        mask_visual_prompt = generate_mask_vprompt(mask)

        return (
            encode_mask(point_visual_prompt),
            encode_mask(scribble_visual_prompt),
            encode_mask(box_visual_prompt),
            encode_mask(mask_visual_prompt),
        )

    def _process_batch_images(self, img_ids_batch, coco_api):
        current_process = mp.current_process()
        pid = current_process.pid

        results = []
        for img_id in tqdm(img_ids_batch, desc=f"Process {pid}"):
            _img_info = coco_api.loadImgs(img_id)[0]
            ann_ids = coco_api.getAnnIds(imgIds=[img_id])
            _anns = coco_api.loadAnns(ann_ids)
            if len(_anns) == 0:
                continue

            img_info = {
                "file_name": _img_info["file_name"],
                "image_id": _img_info["id"],
                "height": _img_info["height"],
                "width": _img_info["width"],
            }

            anns = []
            for ann in _anns:
                if int(ann.get("iscrowd", 0)) != 0:
                    continue

                segmentation = ann["segmentation"]
                if isinstance(segmentation, dict):
                    if isinstance(segmentation["counts"], list):
                        segmentation = mask_utils.frPyObjects(segmentation, *segmentation["size"])
                    segmentation["counts"] = segmentation["counts"].decode("utf-8")
                else:
                    segmentation = [poly for poly in segmentation if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segmentation) == 0:
                        continue

                ann["segmentation"] = segmentation
                ann["bbox_mode"] = BoxMode.XYWH_ABS
                anns.append(ann)

            for ann in anns:
                mask = decode_mask(ann["segmentation"], _img_info["height"], _img_info["width"])
                point_visual_prompt, scribble_visual_prompt, box_visual_prompt, mask_visual_prompt = (
                    self._get_visual_prompts(mask)
                )
                visual_prompts = {
                    "point_visual_prompt": point_visual_prompt,
                    "scribble_visual_prompt": scribble_visual_prompt,
                    "box_visual_prompt": box_visual_prompt,
                    "mask_visual_prompt": mask_visual_prompt,
                }
                ann["visual_prompts"] = visual_prompts

            results.append(
                {
                    "image_file": _img_info["file_name"],
                    "image_id": _img_info["id"],
                    "image_size": (_img_info["height"], _img_info["width"]),
                    "annotations": anns,
                    "image_info": img_info,
                }
            )

        return results

    def _mp_process_ann_data(self, img_ids, coco_api):
        batch_size = 1024
        img_id_batches = [img_ids[i : i + batch_size] for i in range(0, len(img_ids), batch_size)]

        num_workers = min(16, max(1, mp.cpu_count() // 2))
        print_log(
            f"Creating {self.data_name} gt_json, which will take a while, you can download the gt_json from https://huggingface.co/hao9610/X-SAM/resolve/main/vgdseg_annotations",
            logger="current",
        )
        print_log(f"Processing {len(img_ids)} images with {num_workers} workers...", logger="current")

        results = []

        if self.use_threads:
            print_log(f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks", logger="current")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                process_func = partial(self._process_batch_images, coco_api=coco_api)
                futures = [executor.submit(process_func, batch) for batch in img_id_batches]
                for future in tqdm(futures, desc=f"Processing {self.data_name}", ncols=80):
                    batch_results = future.result()
                    if batch_results is not None:
                        results.extend(batch_results)
        else:
            chunksize = max(1, len(img_id_batches) // num_workers // 2)
            with mp.Pool(num_workers) as pool:
                process_func = partial(self._process_batch_images, coco_api=coco_api)
                for i, batch_results in enumerate(
                    tqdm(
                        pool.imap_unordered(process_func, img_id_batches, chunksize=chunksize),
                        total=len(img_id_batches),
                        desc=f"Processing {self.data_name}",
                        ncols=80,
                    )
                ):
                    if batch_results is not None:
                        results.extend(batch_results)

        rets = [r for r in results if r is not None]
        return rets

    def _sample_cats(self, cat_ids, anns):
        def _sample(items, num=None):
            num = len(items) if num is None or num < 0 else min(len(items), num)
            items = random.sample(items, num)
            return items

        pos_cat_ids = sorted({ann["category_id"] for ann in anns if ann["category_id"] in cat_ids})

        if self.data_mode == "train":
            if self.use_negative_sample and random.random() < 0.5:
                neg_cat_ids = sorted(set(cat_ids) - set(pos_cat_ids))
                num_neg = random.randint(0, max(self.sample_num, self.sample_num - len(neg_cat_ids)))
                sampled_neg_cat_ids = _sample(neg_cat_ids, num_neg)
                sampled_cat_ids = _sample(pos_cat_ids + sampled_neg_cat_ids)
            else:
                sampled_cat_ids = _sample(pos_cat_ids)
        else:
            sampled_cat_ids = pos_cat_ids

        for ann in anns:
            ann["category_id"] = sampled_cat_ids.index(ann["category_id"])

        return sampled_cat_ids

    def _load_ann_data(self):
        coco_api = COCO(self.source_data_path)
        img_ids = sorted(coco_api.getImgIds())
        cats = coco_api.loadCats(coco_api.getCatIds())
        cat_ids = sorted([cat["id"] for cat in cats])

        if osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                _rets = json.load(f)

            self.woann_cnt = len(img_ids) - len(_rets)
        else:
            _rets = self._mp_process_ann_data(img_ids, coco_api)
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
                height, width = ret["image_size"]
                valid_anns = [
                    ann
                    for ann in ret["annotations"]
                    if self.visual_prompt_type in ann["visual_prompts"]
                    and decode_mask(ann["visual_prompts"][self.visual_prompt_type], height, width).sum() > 0
                ]
                ret["annotations"] = valid_anns

                if not valid_anns:
                    self.woann_cnt += 1
                    print_log(f"{ret['image_file']} has no {self.visual_prompt_type} anns", logger="current")
                    continue

                rets.append(ret)

        for ret in rets:
            sampled_cat_ids = self._sample_cats(cat_ids, ret["annotations"])
            ret["sampled_labels"] = sampled_cat_ids
            ret["contiguous_labels"] = cat_ids

        gt_json = None
        if self.data_mode == "eval":
            base_tmp = tempfile.gettempdir()
            cache_dir = osp.join(base_tmp, "xsam_cache")
            os.makedirs(cache_dir, exist_ok=True)
            tmp_dir = tempfile.mkdtemp(dir=cache_dir)
            print_log(f"Writing {self.data_name} gt_json to {tmp_dir}...", logger="current")
            tmp_file = osp.join(tmp_dir, f"{self.data_name}.json")
            with open(tmp_file, "w") as f:
                json.dump(rets, f)
            gt_json = tmp_file

        self._set_metadata(cats=cats, gt_json=gt_json)

        return rets

    def _decode_mask(self, data_dict):
        height, width = data_dict["image_size"]
        anns = data_dict["annotations"]
        mask_labels = []
        class_labels = []
        vprompt_masks = []
        for ann in anns:
            segmentation = ann["segmentation"]
            visual_prompts = ann["visual_prompts"]
            category_id = ann["category_id"]
            if self.data_mode == "train":
                keys = list(visual_prompts.keys())
                while keys:
                    key = random.choice(keys)
                    visual_masks = decode_mask(visual_prompts[key], height, width)
                    if visual_masks.sum() > 0:
                        break
                    keys.remove(key)
                if not keys:
                    print_log(
                        f"{data_dict['image_file']} has no visual prompts, ann_id: {ann['id']}",
                        logger="current",
                    )
                    continue
            else:
                visual_masks = (
                    decode_mask(visual_prompts[self.visual_prompt_type], height, width)
                    if visual_prompts.get(self.visual_prompt_type, None) is not None
                    else None
                )
                if visual_masks is None or visual_masks.sum() == 0:
                    print_log(
                        f"{data_dict['image_file']} has no {self.visual_prompt_type}, ann_id: {ann['id']}",
                        logger="current",
                    )
                    continue

            assert visual_masks.sum() > 0, f"visual_masks.sum() == 0, {data_dict['image_file']}, {ann['id']}"
            binary_mask = decode_mask(segmentation, height, width)
            vprompt_masks.append(visual_masks)
            mask_labels.append(binary_mask)
            class_labels.append(category_id)

        if len(mask_labels) == 0:
            print_log(f"image_file: {data_dict['image_file']} has no anns", logger="current")
            mask_labels = torch.zeros(0, height, width, dtype=torch.int64)
            class_labels = torch.zeros(0, dtype=torch.int64)
            vprompt_masks = torch.zeros(0, height, width, dtype=torch.int64)
        else:
            mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_labels])
            class_labels = torch.tensor(np.array(class_labels), dtype=torch.int64)
            vprompt_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in vprompt_masks])

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
                "vprompt_masks": vprompt_masks,
            }
        )

        return data_dict

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
                    pil_image, data_dict["mask_labels"], data_dict["vprompt_masks"], return_tensors="pt"
                )
                data_dict["seg_pixel_values"] = seg_output["pixel_values"][0]
                data_dict["scaled_size"] = tuple(seg_output["scaled_sizes"][0].tolist())
                data_dict["mask_labels"] = seg_output.get("mask_labels", None)
                data_dict["vprompt_masks"] = seg_output.get("vprompt_masks", None)
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
                data_dict["vprompt_masks"] = torch.zeros(0, crop_size["height"], crop_size["width"])
                data_dict["class_labels"] = torch.zeros(0)
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, with_image_token=False))
            data_dict.update(self._get_cond_ids(data_dict))
            data_dict.update(self._get_seg_ids(data_dict))
        return data_dict
