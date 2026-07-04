import contextlib
import io
import itertools
import json
import logging
import os
import os.path as osp
import tempfile
from collections import defaultdict

import numpy as np
from panopticapi.utils import id2rgb
from PIL import Image
from pycocotools import mask as mask_util
from tabulate import tabulate

from xsam.dataset.utils.coco import COCO
from xsam.utils import comm
from xsam.utils.logging import print_log

from ..utils.map import convert_to_coco_json, derive_coco_results, evaluate_predictions_on_coco, instances_to_coco_json
from ..utils.pq import pq_compute, print_panoptic_results
from .img_base_evaluator import ImgBaseEvaluator


class ImgGenSegEvaluator(ImgBaseEvaluator):

    def __init__(
        self,
        *args,
        data_name: str = "img_genseg",
        evaluation_metrics: list[str] = ["PQ", "mIoU", "mAP"],
        show_categories: bool = False,
        map_labels: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_name,
            evaluation_metrics,
            **kwargs,
        )
        self._show_categories = show_categories
        self._map_labels = map_labels
        
    @ImgBaseEvaluator.metadata.setter
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

    def reset(self):
        self._predictions = []
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)

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

    def _encode_json_sem_seg(self, sem_seg, image_id, input_file_name):
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
            json_list.append(
                {
                    "image_id": image_id,
                    "file_name": input_file_name,
                    "category_id": dataset_id,
                    "segmentation": mask_rle,
                }
            )
        return json_list

    def semantic_process(self, inputs, outputs):
        gt_semseg_folder = osp.realpath(self._metadata.sem_segmap_folder)
        segmap_suffix = self._metadata.segmap_suffix if hasattr(self._metadata, "segmap_suffix") else ".png"
        label_shift = self._metadata.label_shift if hasattr(self._metadata, "label_shift") else 0
        for input, output in zip(inputs, outputs):
            segmentation = output["segmentation"].to(self._cpu_device)
            sampled_labels = output["sampled_labels"]
            pred = np.array(segmentation, dtype=int)

            # map contiguous id to dataset id
            if sampled_labels is not None and self._map_labels:
                unique_labels = np.unique(pred).tolist()
                for unique_label in unique_labels:
                    if sampled_labels is not None:
                        pred[pred == unique_label] = sampled_labels[unique_label] - label_shift

            file_name = input["file_name"]
            file_name_semseg = os.path.splitext(file_name)[0] + segmap_suffix
            gt = np.array(Image.open(osp.join(gt_semseg_folder, file_name_semseg)), dtype=np.uint32)
            gt[gt == self._metadata.ignore_value] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self._encode_json_sem_seg(pred, input["image_id"], input["file_name"]))

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
                # category_id * label_divisor + instance_id.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(segmentation):
                    if panoptic_label == 0:
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
        # deduplicate the predictions by file_name when distributed evaluation
        merged_predictions = defaultdict(list)
        for prediction in predictions:
            file_name = prediction["file_name"]
            merged_predictions[file_name].append(prediction)

        # Deduplicate by `file_name` within each `image_id` group and keep the latest entry.
        predictions = [
            next(iter({ann["file_name"]: ann for ann in anns}.values())) for anns in merged_predictions.values()
        ]

        gt_json = osp.realpath(self._metadata.gt_json) if self._metadata.gt_json is not None else None
        if gt_json is not None:
            with tempfile.TemporaryDirectory(prefix="semantic_eval") as pred_dir:
                with open(gt_json, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                json_data["annotations"] = predictions
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

        headers = ["Metric", "mIoU", "fwIoU", "mACC", "pACC"]
        data = [["Value (%)", f"{100 * miou:.2f}", f"{100 * fiou:.2f}", f"{100 * macc:.2f}", f"{100 * pacc:.2f}"]]

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
        with open(osp.realpath(self._metadata.gt_json), "r") as f:
            gt_anns = json.load(f)

        print_log(f"Trying to convert '{self.data_name}' to COCO format...", logger="current")
        cache_path = osp.join(self._output_dir, f"{self.data_name}_coco_format.json")
        convert_to_coco_json(self.data_name, cache_path, gt_anns, allow_cached=False)
        coco_api = COCO(cache_path)

        gt_img_ids = sorted(coco_api.getImgIds())
        pred_img_ids = sorted({pred["image_id"] for pred in predictions})
        if pred_img_ids != gt_img_ids:
            print_log(
                f"The number of images in predictions and ground truth is not the same. There are {len(pred_img_ids)} predictions and {len(gt_img_ids)} ground truth images.",
                logger="current",
            )

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
                # )
                continue
            result["category_id"] = reverse_id_mapping[category_id]
            new_coco_results.append(result)

        coco_eval = (
            evaluate_predictions_on_coco(
                coco_api,
                new_coco_results,
                "segm",
                img_ids=pred_img_ids,
            )
            if len(new_coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )
        table = derive_coco_results(
            coco_eval, "segm", class_names=self._metadata.get("thing_classes"), show_categories=self._show_categories
        )
        return table

    def panoptic_evaluate(self, predictions):
        # deduplicate the predictions by file_name when distributed evaluation
        merged_predictions = defaultdict(list)
        for prediction in predictions:
            image_id = prediction["image_id"]
            merged_predictions[image_id].append(prediction)

        # Deduplicate by `file_name` within each `image_id` group and keep the latest entry.
        predictions = [
            next(iter({ann["file_name"]: ann for ann in anns}.values())) for anns in merged_predictions.values()
        ]

        # PanopticApi requires local files
        gt_json = osp.realpath(self._metadata.gt_json)
        gt_panseg_folder = osp.realpath(self._metadata.pan_segmap_folder)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            for pred in predictions:
                assert not osp.exists(
                    osp.join(pred_dir, pred["file_name"])
                ), f"File {pred['file_name']} already exists"
                with open(osp.join(pred_dir, pred["file_name"]), "wb") as f:
                    f.write(pred.pop("png_string"))

            with open(gt_json, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            json_data["annotations"] = predictions

            pred_json = osp.join(pred_dir, "predictions.json")
            print_log(f"Saving {self.data_name} predictions to {pred_json}...", logger="current")
            with open(pred_json, "w") as f:
                json.dump(json_data, f)

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    osp.realpath(pred_json),
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
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
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
                print_log(f"Error writing {self.data_name} predictions to {self._output_dir}: {e}", logger="current")

        predictions = [pred for pred in predictions if isinstance(pred, dict) and "image_id" in pred]
        if not predictions:
            print_log(
                f"{self.__class__.__name__} did not receive valid predictions.",
                logger="current",
                level=logging.WARNING,
            )
            return {}

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")

        if "semantic" in self.data_name:
            table = self.semantic_evaluate(predictions)
        elif "instance" in self.data_name:
            table = self.instance_evaluate(predictions)
        elif "panoptic" in self.data_name:
            table = self.panoptic_evaluate(predictions)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

        print_log(f"{self.data_name} evaluation results:\n{table}", logger="current")
