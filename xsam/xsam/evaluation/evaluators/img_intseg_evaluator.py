import json

from xsam.dataset.utils.mask import calculate_iou, decode_mask
from xsam.utils.logging import print_log

from .img_refseg_evaluator import ImgRefSegEvaluator


class ImgIntSegEvaluator(ImgRefSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            gt_mask = ann_maps[0]["segmentation"]
            gt_mask = decode_mask(gt_mask, height, width)

            intersection, union, _ = calculate_iou(pred_mask, gt_mask, 2, self._metadata.ignore_value)
            self.iou_stat.update(intersection, union, n=1)

        self.iou_stat.average()
        print_log(f"{self.data_name} evaluation results:\n{self.iou_stat}", logger="current")
