import json

from xsam.utils.logging import print_log

from ...dataset.utils.mask import calculate_iou, decode_mask
from .refer_seg_evaluator import ReferSegEvaluator


class InterSegEvaluator(ReferSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cat_names=["ignore", "inter"], **kwargs)

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

            intersection, union, _ = calculate_iou(pred_mask, gt_mask, self._metadata.ignore_label)
            self.iou_stat.update(intersection, union, n=1)

        self.iou_stat.average()
        print_log(f"{self.data_name} evaluation results:\n{self.iou_stat}", logger="current")
