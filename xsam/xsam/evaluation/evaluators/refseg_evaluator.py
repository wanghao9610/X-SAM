import itertools
import json
import logging
import os
import os.path as osp
from typing import List, Optional

import numpy as np

from xsam.utils.logging import print_log

from ...dataset.utils.catalog import MetadataCatalog
from ...dataset.utils.mask import calculate_iou, decode_mask, encode_mask
from ..utils import comm
from ..utils.iou import IouStat
from .base_evaluator import BaseEvaluator


class RefSegEvaluator(BaseEvaluator):

    def __init__(
        self,
        data_name: str = "refseg",
        cat_names: Optional[List[str]] = ["ignore", "refer"],
        output_dir: Optional[str] = None,
        distributed: bool = True,
    ):
        """
        Args:
            metadata: metadata of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._distributed = distributed
        self._data_name = data_name
        self._metadata = MetadataCatalog.get(data_name)
        self._output_dir = output_dir
        self.iou_stat = IouStat(cat_names=cat_names)

        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def data_name(self):
        return self._data_name

    def reset(self):
        self._predictions = []

    # follow segmentation evaluation
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred_mask, segments_info = (
                output["segmentation"],
                output["segments_info"],
            )
            pred_mask = pred_mask.cpu().numpy()
            pred_mask[pred_mask == self._metadata.ignore_label] = 0
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
                return {}
        else:
            predictions = self._predictions

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")
        if len(predictions) == 0:
            logging.warning(f"{self.__class__.__name__} did not receive valid predictions.")
            return {}

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {self._output_dir}...", logger="current")
            with open(file_path, "w") as f:
                json.dump(predictions, f)

        gt_json = osp.realpath(self._metadata.gt_json)

        self._eval_predictions(self._predictions, gt_json)

    def _eval_predictions(self, predictions, gt_json):
        with open(gt_json, "r") as f:
            gt_anns = json.load(f)

        id2ann_map = {f"{data['image_id']}{data['image_info']['sample_id']}": data["annotations"] for data in gt_anns}

        for pred in predictions:
            image_id = pred["image_id"]
            sample_id = pred["sample_id"]
            pred_mask = pred["pred_mask"]
            height, width = pred_mask["size"]
            pred_mask = decode_mask(pred_mask, height, width)

            # segmentation is polygon
            gt_mask = id2ann_map[f"{image_id}{sample_id}"][0]["segmentation"]
            gt_mask = decode_mask(gt_mask, height, width)

            intersection, union, _ = calculate_iou(pred_mask, gt_mask, 2, self._metadata.ignore_label)
            self.iou_stat.update(intersection, union, n=1)

        self.iou_stat.average()
        print_log(f"{self.data_name} evaluation results:\n{self.iou_stat}", logger="current")
