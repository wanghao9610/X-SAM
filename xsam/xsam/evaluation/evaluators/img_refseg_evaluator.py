import itertools
import json
import logging
import os
import os.path as osp
from functools import reduce

import numpy as np

from xsam.dataset.utils.mask import calculate_iou, decode_mask, encode_mask
from xsam.utils import comm
from xsam.utils.logging import print_log

from ..utils.iou import IouStat
from .img_base_evaluator import ImgBaseEvaluator


class ImgRefSegEvaluator(ImgBaseEvaluator):

    def __init__(
        self,
        *args,
        data_name: str = "img_refseg",
        **kwargs,
    ):
        super().__init__(
            *args,
            data_name,
            **kwargs,
        )
        self.iou_stat = IouStat()

    # follow segmentation evaluation
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred_mask, segments_info = (
                output["segmentation"],
                output["segments_info"],
            )
            pred_mask = pred_mask.cpu().numpy()
            pred_mask[pred_mask == self._metadata.ignore_value] = 0
            pred_mask = pred_mask.astype(np.uint8)
            file_name = os.path.basename(input["file_name"])
            self._predictions.append(
                {
                    "image_id": input["image_id"],
                    "sample_id": input["sample_id"],
                    "file_name": file_name,
                    "pred_mask": encode_mask(pred_mask),
                    "segments_info": segments_info,
                }
            )

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return
        else:
            predictions = self._predictions

        if not predictions and osp.exists(self._output_dir) and self.support_loading:
            file_path = osp.join(self._output_dir, "predictions.json")
            print_log(f"Loading predictions from {file_path}...", logger="current")
            with open(file_path, "r") as f:
                predictions = json.load(f)
        elif self._output_dir:
            try:
                os.makedirs(self._output_dir, exist_ok=True)
                file_path = osp.join(self._output_dir, "predictions.json")
                print_log(f"Saving {self.data_name} predictions to {self._output_dir}...", logger="current")
                with open(file_path, "w") as f:
                    json.dump(predictions, f)
            except Exception as e:
                print_log(f"Error saving {self.data_name} predictions to {self._output_dir}: {e}", logger="current")

        predictions = [
            pred for pred in predictions if isinstance(pred, dict) and "image_id" in pred and "sample_id" in pred
        ]
        if not predictions:
            print_log(
                f"{self.__class__.__name__} did not receive valid predictions.",
                logger="current",
                level=logging.WARNING,
            )
            return {}

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")

        gt_json = osp.realpath(self._metadata.gt_json)
        self._eval_predictions(predictions, gt_json)

    def _eval_predictions(self, predictions, gt_json):
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_anns = json.load(f)

        id2ann_map = {f"{data['image_id']}{data['image_info']['sample_id']}": data["annotations"] for data in gt_anns}

        for pred in predictions:
            image_id = pred["image_id"]
            sample_id = pred["sample_id"]
            pred_mask = pred["pred_mask"]
            height, width = pred_mask["size"]
            pred_mask = decode_mask(pred_mask, height, width)
            ann_maps = id2ann_map.get(f"{image_id}{sample_id}", None)
            if ann_maps is None:
                print_log(f"No ground truth found for image {image_id} and sample {sample_id}", logger="current")
                continue

            # segmentation is polygon
            if "gref" not in self.data_name:
                gt_mask = ann_maps[0]["segmentation"]
                gt_mask = decode_mask(gt_mask, height, width)
            else:
                gt_mask = [decode_mask(ann["segmentation"], height, width) for ann in ann_maps]
                gt_mask = reduce(np.logical_or, gt_mask, np.zeros((height, width)))

            intersection, union, _ = calculate_iou(pred_mask, gt_mask, 2, self._metadata.ignore_value)
            self.iou_stat.update(intersection, union, n=1)

        self.iou_stat.average()
        print_log(f"{self.data_name} evaluation results:\n{self.iou_stat}", logger="current")
