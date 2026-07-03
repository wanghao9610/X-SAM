import copy
import json
import multiprocessing as mp
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from xsam.utils.logging import print_log

from .img_base_dataset import ImgBaseDataset
from .utils.image import expand2square
from .utils.mask import decode_mask


class ImageSamDataset(ImgBaseDataset):
    def __init__(
        self,
        *args,
        task_name="img_sam",
        num_sample=float("inf"),
        num_data=float("inf"),
        num_ann=100,
        use_threads=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            task_name=task_name,
            num_sample=num_sample,
            num_data=num_data,
            imgmap_suffix=".jpg",
            ann_suffix=".json",
            use_threads=use_threads,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.num_sample = kwargs.get(
            "num_sample",
            float("inf"),
        )
        self.num_data = kwargs.get(
            "num_data",
            float("inf"),
        )
        self.num_ann = kwargs.get("num_ann", 100)
        self.imgmap_suffix = kwargs.get("imgmap_suffix", ".jpg")
        self.ann_suffix = kwargs.get("ann_suffix", ".json")
        self.use_threads = kwargs.get("use_threads", True)

    def _process_batch_data_worker(self, image_dirs_batch):
        current_process = mp.current_process()
        pid = current_process.pid

        rets = []
        rng = np.random.default_rng()
        for image_dir in tqdm(image_dirs_batch, desc=f"Process {pid}"):
            if not osp.isdir(osp.join(self.image_folder, image_dir)) or not image_dir.startswith("sa_"):
                continue
            image_files = [
                image_file
                for image_file in os.listdir(osp.join(self.image_folder, image_dir))
                if image_file.endswith(".jpg")
            ]
            rng.shuffle(image_files)
            image_files = image_files[: min(self.num_sample, len(image_files))]
            for image_file in image_files:
                try:
                    Image.open(osp.join(self.image_folder, image_dir, image_file)).convert("RGB")
                except Exception as e:
                    print_log(f"Error opening image {image_file}: {e}", logger="current")
                    continue
                ann_file = osp.join(
                    self.image_folder, image_dir, image_file.replace(self.imgmap_suffix, self.ann_suffix)
                )
                if not osp.exists(ann_file):
                    continue
                try:
                    with open(ann_file, "r") as f:
                        json.load(f)
                except Exception as e:
                    print_log(f"Error opening annotation file {ann_file}: {e}", logger="current")
                    continue
                rets.append(
                    {
                        "image_file": osp.join(image_dir, image_file),
                    }
                )

        return rets

    def _mp_process_ann_data(self, image_dirs):
        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(f"Processing {len(image_dirs)} image directories with {num_workers} workers...", logger="current")

        batch_size = max(1, min(16, len(image_dirs) // num_workers))
        image_dirs_batches = [image_dirs[i : i + batch_size] for i in range(0, len(image_dirs), batch_size)]

        rets = []
        if self.use_threads:
            print_log(f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks", logger="current")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._process_batch_data_worker, batch) for batch in image_dirs_batches]
                for future in tqdm(futures, desc="Processing SA1B dataset", ncols=80):
                    batch_results = future.result()
                    if batch_results is not None:
                        rets.extend(batch_results)
        else:
            chunk_size = max(1, len(image_dirs_batches) // num_workers // 2)
            with mp.Pool(num_workers) as pool:
                for i, batch_results in enumerate(
                    tqdm(
                        pool.imap_unordered(
                            self._process_batch_data_worker, image_dirs_batches, chunk_size=chunk_size
                        ),
                        total=len(image_dirs_batches),
                        desc="Processing SA1B dataset",
                        ncols=80,
                    )
                ):
                    if batch_results is not None:
                        rets.extend(batch_results)

        rets = [r for r in rets if r is not None]
        # assign image_id after collecting all results
        for img_id, ret in enumerate(rets):
            ret["image_id"] = img_id

        return rets

    def _load_ann_data(self):
        if self.data_path is not None and osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                rets = json.load(f)
        else:
            image_dirs = os.listdir(self.image_folder)
            rets = self._mp_process_ann_data(image_dirs)

            if self.data_path is not None:
                basedir = osp.dirname(self.data_path)
                if basedir:
                    os.makedirs(basedir, exist_ok=True)
                print_log(f"Saving SA1B data to {self.data_path}...", logger="current")
                with open(self.data_path, "w", encoding="utf-8") as f:
                    json.dump(rets, f)

        rng = np.random.default_rng()
        rng.shuffle(rets)
        rets = rets[: min(self.num_data, len(rets))]
        return rets

    def _decode_ann_data(self, data_dict):
        image_file = data_dict["image_file"]
        ann_file = osp.join(self.image_folder, image_file.replace(self.imgmap_suffix, self.ann_suffix))
        with open(ann_file, "r") as f:
            ann_data = json.load(f)
        image_info = ann_data["image"]
        annotations = ann_data["annotations"]
        height, width = image_info["height"], image_info["width"]

        anns_to_process = annotations[: self.num_ann]
        num_anns = len(anns_to_process)

        if num_anns == 0:
            mask_labels = torch.zeros((0, height, width))
            class_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Pre-allocate numpy array
            mask_labels_np = np.zeros((num_anns, height, width), dtype=np.uint8)

            # Decode directly into pre-allocated array
            for i, ann in enumerate(anns_to_process):
                mask_labels_np[i] = decode_mask(ann["segmentation"], height, width)

            # Single conversion to torch
            mask_labels = torch.from_numpy(mask_labels_np)
            class_labels = torch.zeros((num_anns,), dtype=torch.int64)

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
            }
        )

        return data_dict

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
