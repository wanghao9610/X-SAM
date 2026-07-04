import itertools
import json
import logging
import os
import os.path as osp

from xsam.dataset.utils.coco import COCO
from xsam.utils import comm
from xsam.utils.logging import print_log

from ..utils.map import convert_to_coco_json, derive_coco_results, evaluate_predictions_on_coco, instances_to_coco_json
from .img_base_evaluator import ImgBaseEvaluator


class ImgVGDSegEvaluator(ImgBaseEvaluator):
    def __init__(
        self,
        *args,
        data_name: str = "img_vgdseg",
        reverse_cat_ids: bool = True,
        show_categories: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_name,
            **kwargs,
        )
        self._show_categories = show_categories
        self._reverse_cat_ids = reverse_cat_ids

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
            print_log(f"Gathering {self.data_name} predictions...", logger="current")

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
            try:
                os.makedirs(self._output_dir, exist_ok=True)
                file_path = osp.join(self._output_dir, "predictions.json")
                print_log(f"Saving {self.data_name} predictions to {self._output_dir}...", logger="current")
                with open(file_path, "w") as f:
                    json.dump(predictions, f)
            except Exception as e:
                print_log(f"Error saving {self.data_name} predictions to {self._output_dir}: {e}", logger="current")

        predictions = [pred for pred in predictions if isinstance(pred, dict) and "instances" in pred]

        if not predictions:
            print_log(
                f"{self.__class__.__name__} did not receive valid predictions.",
                logger="current",
                level=logging.WARNING,
            )
            return {}

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")

        self._evaluate_predictions(predictions)

    def _evaluate_predictions(self, predictions):
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

        gt_img_ids = sorted(coco_api.getImgIds())
        pred_img_ids = sorted({pred["image_id"] for pred in predictions})
        if pred_img_ids != gt_img_ids:
            print_log(
                f"The number of images in predictions and ground truth is not the same. There are {len(pred_img_ids)} predictions and {len(gt_img_ids)} ground truth images.",
                logger="current",
            )

        coco_results = list(itertools.chain(*[x.get("instances", []) for x in predictions]))
        # unmap the category ids for COCO
        dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        num_classes = len(all_contiguous_ids)
        assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        new_coco_results = []
        if self._reverse_cat_ids:
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
        else:
            new_coco_results = coco_results

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

        print_log(f"{self.data_name} evaluation results:\n{table}", logger="current")
