import itertools
import json
import os
import os.path as osp
from typing import Optional

import numpy as np
import torch

from xsam.utils.logging import print_log

from ...dataset.utils.catalog import MetadataCatalog
from ...dataset.utils.coco import COCO
from ..utils import comm
from ..utils.map import convert_to_coco_json, derive_coco_results, evaluate_predictions_on_coco, instances_to_coco_json
from .base_seg_evaluator import BaseSegEvaluator


class VGDSegEvaluator(BaseSegEvaluator):
    def __init__(
        self,
        data_name: str = "coco_vgdseg",
        output_dir: Optional[str] = None,
        distributed: bool = True,
        show_categories: bool = False,
    ):
        """
        Args:
            metadata: metadata of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._data_name = data_name
        self._distributed = distributed
        self._metadata = MetadataCatalog.get(data_name)
        self._output_dir = output_dir
        self._show_categories = show_categories
        self._cpu_device = torch.device("cpu")
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
        self._dataset_name = self.data_name
        self._num_classes = len(self._metadata.dataset_id_to_contiguous_id)
        self._contiguous_id_to_dataset_id = {v: k for k, v in self._metadata.dataset_id_to_contiguous_id.items()}
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self._thing_contiguous_id_to_dataset_id = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
        if hasattr(self._metadata, "stuff_dataset_id_to_contiguous_id"):
            self._stuff_contiguous_id_to_dataset_id = {
                v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
            }

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
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            conf_matrix_list = comm.gather(self._conf_matrix, dst=0)
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
        else:
            predictions = self._predictions

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {self._output_dir}...", logger="current")
            with open(file_path, "w") as f:
                json.dump(predictions, f)

        with open(osp.realpath(self._metadata.gt_json), "r") as f:
            gt_anns = json.load(f)

        for gt_ann in gt_anns:
            sampled_labels = gt_ann["sampled_labels"]
            contiguous_labels = gt_ann["contiguous_labels"]
            for ann in gt_ann["annotations"]:
                ann["category_id"] = contiguous_labels.index(sampled_labels[ann["category_id"]])

        print_log(f"Trying to convert '{self.data_name}' to COCO format...", logger="current")
        cache_path = osp.join(self._output_dir, f"{self.data_name}_coco_format.json")
        convert_to_coco_json(self.data_name, cache_path, gt_anns, allow_cached=False)
        coco_api = COCO(cache_path)

        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        # unmap the category ids for COCO
        dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        num_classes = len(all_contiguous_ids)
        assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        new_coco_results = []
        for result in coco_results:
            category_id = result["category_id"]
            if category_id not in reverse_id_mapping:
                print_log(
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}].",
                    logger="current",
                )
                continue
            result["category_id"] = reverse_id_mapping[category_id]
            new_coco_results.append(result)

        coco_eval = (
            evaluate_predictions_on_coco(
                coco_api,
                new_coco_results,
                "segm",
            )
            if len(new_coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )
        table = derive_coco_results(
            coco_eval, "segm", class_names=self._metadata.get("thing_classes"), show_categories=self._show_categories
        )
        return table
