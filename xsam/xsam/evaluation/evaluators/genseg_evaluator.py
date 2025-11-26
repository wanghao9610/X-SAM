import contextlib
import io
import itertools
import json
import os
import os.path as osp
import tempfile
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from tabulate import tabulate

from xsam.utils.logging import print_log

from ...dataset.utils.catalog import MetadataCatalog
from ...dataset.utils.coco import COCO
from ..utils import comm
from ..utils.map import convert_to_coco_json, derive_coco_results, evaluate_predictions_on_coco, instances_to_coco_json
from ..utils.pq import pq_compute, print_panoptic_results
from .base_evaluator import BaseEvaluator


class GenSegEvaluator(BaseEvaluator):
    def __init__(
        self,
        data_name: str = "genseg",
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

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[segment_info["category_id"]]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[segment_info["category_id"]]
        return segment_info

    def _encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append({"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle})
        return json_list

    def semantic_process(self, inputs, outputs):
        gt_semseg_folder = osp.realpath(self._metadata.semseg_map_folder)
        semseg_sufix = self._metadata.semseg_sufix if hasattr(self._metadata, "semseg_sufix") else ".png"
        label_shift = self._metadata.label_shift if hasattr(self._metadata, "label_shift") else 0
        for input, output in zip(inputs, outputs):
            segmentation = output["segmentation"].to(self._cpu_device)
            sampled_labels = output["sampled_labels"]
            pred = np.array(segmentation, dtype=int)

            if sampled_labels is not None:
                unique_labels = np.unique(pred).tolist()
                for unique_label in unique_labels:
                    if sampled_labels is not None:
                        pred[pred == unique_label] = sampled_labels[unique_label] - label_shift

            file_name = input["file_name"]
            file_name_semseg = os.path.splitext(file_name)[0] + semseg_sufix
            gt = np.array(Image.open(os.path.join(gt_semseg_folder, file_name_semseg)), dtype=np.uint32)
            gt[gt == self._metadata.ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self._encode_json_sem_seg(pred, input["file_name"]))

    def instance_process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def panoptic_process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            segmentation, segments_info = (
                output["segmentation"],
                output["segments_info"],
            )
            segmentation = segmentation.to(self._cpu_device)
            segmentation = np.array(segmentation, dtype=int)
            if segments_info is None:
                # If "segments_info" is None, we assume "segmentation" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to segmentation since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(segmentation):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                segmentation += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(segmentation)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def semantic_evaluate(self, predictions):
        gt_json = osp.realpath(self._metadata.gt_json) if self._metadata.gt_json is not None else None
        if gt_json is not None:
            with tempfile.TemporaryDirectory(prefix="semantic_eval") as pred_dir:
                with open(gt_json, "r") as f:
                    json_data = json.load(f)
                json_data["annotations"] = predictions

                output_dir = self._output_dir or pred_dir
                print_log(f"Writing {self.data_name} predictions to {output_dir}...", logger="current")
                predictions_json = os.path.join(output_dir, "predictions.json")
                with open(predictions_json, "w") as f:
                    json.dump(json_data, f)
        else:
            print_log("Ground truth JSON file is not provided, skipping annotation writing.", logger="current")

        acc = np.full(self._num_classes, np.nan, dtype=np.float32)
        iou = np.full(self._num_classes, np.nan, dtype=np.float32)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float32)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float32)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float32)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        data = []
        headers = ["Metric", "Value (%)"]
        data.extend(
            [
                ["mIoU", f"{100 * miou:.2f}"],
                ["fwIoU", f"{100 * fiou:.2f}"],
                ["mACC", f"{100 * macc:.2f}"],
                ["pACC", f"{100 * pacc:.2f}"],
            ]
        )

        if self._show_categories:
            class_names = [self._metadata.dataset_classes[k] for k in sorted(self._metadata.dataset_classes)]
            for i, name in enumerate(class_names):
                data.extend([[f"IoU-{name}", f"{100 * iou[i]:.2f}"], [f"ACC-{name}", f"{100 * acc[i]:.2f}"]])

        table = tabulate(
            data,
            headers=headers,
            tablefmt="outline",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        return table

    def instance_evaluate(self, predictions):
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {self._output_dir}...", logger="current")
            with open(file_path, "w") as f:
                json.dump(predictions, f)

        with open(osp.realpath(self._metadata.gt_json), "r") as f:
            gt_anns = json.load(f)

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
                # print_log(
                #     f"A prediction has class={category_id}, "
                #     f"but the dataset only has {num_classes} classes and "
                #     f"predicted class id should be in [0, {num_classes - 1}].",
                #     logger="current",
                # )b
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

    def panoptic_evaluate(self, predictions):
        # PanopticApi requires local files
        gt_json = osp.realpath(self._metadata.gt_json)
        gt_panseg_folder = osp.realpath(self._metadata.panseg_map_folder)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            for p in predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {output_dir}...", logger="current")
            with open(predictions_json, "w") as f:
                json.dump(json_data, f)

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    osp.realpath(predictions_json),
                    gt_folder=gt_panseg_folder,
                    pred_folder=pred_dir,
                )

        table = print_panoptic_results(pq_res)
        return table

    def process(self, inputs, outputs):
        if "semantic" in self.data_name:
            self.semantic_process(inputs, outputs)
        elif "instance" in self.data_name:
            self.instance_process(inputs, outputs)
        elif "panoptic" in self.data_name:
            self.panoptic_process(inputs, outputs)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

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

        if "semantic" in self.data_name:
            table = self.semantic_evaluate(predictions)
        elif "instance" in self.data_name:
            table = self.instance_evaluate(predictions)
        elif "panoptic" in self.data_name:
            table = self.panoptic_evaluate(predictions)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

        print_log(f"{self.data_name} evaluation results:\n{table}", logger="current")
