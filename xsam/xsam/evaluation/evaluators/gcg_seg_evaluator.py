import copy
import itertools
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import traceback
from functools import partial
from typing import Optional

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from xsam.utils.logging import print_log

from ...dataset.utils.catalog import MetadataCatalog
from ...dataset.utils.coco import COCO
from ...dataset.utils.coco_cap_eval import COCOEvalCap
from ...dataset.utils.mask import decode_mask, encode_mask
from ..utils import comm
from ..utils.miou import compute_iou_matrix, compute_miou
from .base_seg_evaluator import BaseSegEvaluator


class GCGSegEvaluator(BaseSegEvaluator):

    def __init__(
        self,
        data_name: str = "gcgseg",
        text_model: str = "bert-base-uncased",
        # metrics: list[str] = ["miou", "map", "caption", "recall"],
        metrics: list[str] = ["miou", "map", "caption"],
        output_dir: Optional[str] = None,
        distributed: bool = True,
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
        self.metrics = metrics
        self._cpu_device = torch.device("cpu")
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model)
        self.text_model.eval()

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

    @property
    def data_name(self):
        return self._data_name

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    def reset(self):
        self._predictions = []

    def _encode_text_embeddings(self, text):
        input_dict = self.text_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        text_embeddings = self.text_model(**input_dict)
        text_embeddings = torch.mean(text_embeddings.last_hidden_state[0], dim=0).detach().numpy()
        return text_embeddings

    def _get_text_similarity(self, text1, text2):
        text1_embeddings = self._encode_text_embeddings(text1)
        text2_embeddings = self._encode_text_embeddings(text2)
        return cosine_similarity([text1_embeddings], [text2_embeddings])[0, 0]

    def _get_rle_masks(self, segmentation):
        rle_masks = []
        labels, areas = np.unique(segmentation, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(lambda l: l < self.metadata.ignore_label, labels):
            binary_mask = (segmentation == label).astype(np.uint8)
            rle_mask = encode_mask(binary_mask)
            rle_masks.append(rle_mask)
        return rle_masks

    # follow segmentation evaluation
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            segmentation, gcg_phrases, gcg_caption = (
                output["segmentation"],
                output["gcg_phrases"],
                output["gcg_caption"],
            )
            segmentation = segmentation.to(self._cpu_device)
            segmentation = np.array(segmentation, dtype=int)
            segmentation = self._get_rle_masks(segmentation)
            file_name = os.path.basename(input["file_name"])
            self._predictions.append(
                {
                    "image_id": input["image_id"],
                    "file_name": file_name,
                    "segmentation": segmentation,
                    "gcg_phrases": gcg_phrases,
                    "gcg_caption": gcg_caption,
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
        cap_gt_json = osp.realpath(self._metadata.cap_gt_json)

        mask_preds = []
        caption_preds = []
        for pred in predictions:
            for seg in pred["segmentation"]:
                cur_mask_pred = {
                    "image_id": pred["image_id"],
                    "category_id": 1,
                    "segmentation": seg,
                    "score": 1.0,
                }
                mask_preds.append(cur_mask_pred)
            cur_caption_pred = {
                "image_id": pred["image_id"],
                "caption": pred["gcg_caption"],
                "labels": pred["gcg_phrases"],
            }
            caption_preds.append(cur_caption_pred)

        for metric in self.metrics:
            if metric == "miou":
                self._eval_miou(copy.deepcopy(mask_preds), gt_json)
            elif metric == "map":
                self._eval_map(copy.deepcopy(mask_preds), gt_json)
            elif metric == "caption":
                self._eval_caption(copy.deepcopy(caption_preds), cap_gt_json)
            elif metric == "recall":
                self._eval_recall(copy.deepcopy(mask_preds), copy.deepcopy(caption_preds), gt_json, cap_gt_json)
            else:
                raise ValueError(f"Metric {metric} not supported")

    def _eval_miou(self, preds, gt_json):
        coco_gt = COCO(gt_json)
        coco_dt = coco_gt.loadRes(preds)
        imgids = sorted(list(set([pred["image_id"] for pred in preds])))

        mious = []
        for imgid in imgids:
            imginfo = coco_gt.loadImgs([imgid])[0]
            height, width = imginfo["height"], imginfo["width"]

            gt_ann_ids = coco_gt.getAnnIds(imgIds=[imgid])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            dt_ann_ids = coco_dt.getAnnIds(imgIds=[imgid])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            gt_masks = [decode_mask(ann["segmentation"], height, width) for ann in gt_anns]
            dt_masks = [decode_mask(ann["segmentation"], height, width) for ann in dt_anns]
            mious.append(compute_miou(dt_masks, gt_masks))

        miou_res = float(np.mean(mious) * 100) if mious else 0.0
        data = [["mIoU", f"{miou_res:.2f}"]]
        headers = ["Metric", "Value (%)"]
        table = tabulate(
            data,
            headers=headers,
            tablefmt="outline",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        print_log(f"{self.data_name} mIoU results:\n{table}", logger="current")

    def _eval_map(self, preds, gt_json):
        coco_gt = COCO(gt_json)
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        imgids = sorted(list(set([pred["image_id"] for pred in preds])))
        coco_eval.params.imgIds = imgids
        coco_eval.params.catIds = [1]
        print_log(f"{self.data_name} map results:", logger="current")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def _eval_caption(self, preds, gt_json):
        coco_gt = COCO(gt_json)
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOEvalCap(coco_gt, coco_dt)
        imgids = sorted(list(set([pred["image_id"] for pred in preds])))
        coco_eval.params["image_id"] = imgids
        coco_eval.evaluate()

        data = []
        headers = ["Metric", "Value (%)"]
        for metric, value in coco_eval.eval.items():
            data.append([metric, f"{value * 100:.2f}"])
        table = tabulate(
            data,
            headers=headers,
            tablefmt="outline",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )

        print_log(f"{self.data_name} caption results:\n{table}", logger="current")

    def _process_single_image(self, imgid, coco_gt, cap_coco_gt, coco_dt, cap_coco_dt):
        try:
            imginfo = coco_gt.loadImgs([imgid])[0]
            height, width = imginfo["height"], imginfo["width"]

            gt_ann_ids = coco_gt.getAnnIds(imgIds=[imgid])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)
            dt_ann_ids = coco_dt.getAnnIds(imgIds=[imgid])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            cap_gt_ann_ids = cap_coco_gt.getAnnIds(imgIds=[imgid])
            cap_gt_anns = cap_coco_gt.loadAnns(cap_gt_ann_ids)[0]
            cap_dt_ann_ids = cap_coco_dt.getAnnIds(imgIds=[imgid])
            cap_dt_anns = cap_coco_dt.loadAnns(cap_dt_ann_ids)[0]

            cap_gt_labels = cap_gt_anns["labels"]
            cap_dt_labels = cap_dt_anns["labels"]

            best_matches = self._find_best_matches(gt_anns, cap_gt_labels, dt_anns, cap_dt_labels, height, width)
            return len(cap_gt_labels), len(best_matches)
        except Exception as e:
            print_log(f"Error processing image {imgid}: {e}\n{traceback.format_exc()}", logger="current")
            return 0, 0

    def _eval_recall(self, mask_preds, caption_preds, gt_json, cap_gt_json):
        coco_gt = COCO(gt_json)
        cap_coco_gt = COCO(cap_gt_json)
        coco_dt = coco_gt.loadRes(mask_preds)
        cap_coco_dt = cap_coco_gt.loadRes(caption_preds)
        imgids = sorted(list(set([pred["image_id"] for pred in mask_preds])))

        # Initialize multiprocessing pool
        num_workers = min(16, len(imgids))
        with mp.Pool(num_workers) as pool:
            process_func = partial(
                self._process_single_image,
                coco_gt=coco_gt,
                cap_coco_gt=cap_coco_gt,
                coco_dt=coco_dt,
                cap_coco_dt=cap_coco_dt,
            )
            results = list(tqdm(pool.imap(process_func, imgids), total=len(imgids), desc="Calculating recall"))

        # Sum up results
        p_cnt = sum(p for p, _ in results)
        tp_cnt = sum(tp for _, tp in results)

        recall = tp_cnt / p_cnt if p_cnt > 0 else 0
        data = [["recall", f"{recall * 100:.2f}"]]
        headers = ["Metric", "Value (%)"]
        table = tabulate(
            data,
            headers=headers,
            tablefmt="outline",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )

        print_log(f"{self.data_name} recall results:\n{table}", logger="current")

    def _find_best_matches(
        self, gt_anns, gt_labels, dt_anns, dt_labels, height, width, iou_threshold=0.5, text_sim_threshold=0.5
    ):
        best_matches = []

        # Compute pair - wise IoU
        pred_masks = [decode_mask(ann["segmentation"], height, width) for ann in dt_anns]
        gt_masks = [decode_mask(ann["segmentation"], height, width) for ann in gt_anns]
        ious = compute_iou_matrix(gt_masks, pred_masks)

        text_sims = np.zeros((len(gt_labels), len(dt_labels)))

        for i, gt_label in enumerate(gt_labels):
            for j, dt_label in enumerate(dt_labels):
                text_sims[i, j] = self._get_text_similarity(gt_label, dt_label)

        # Find one-to-one matches satisfying both IoU and text similarity thresholds
        while ious.size > 0:
            max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
            if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
                break  # No admissible pair found

            best_matches.append(max_iou_idx)

            # Remove selected annotations from consideration
            ious[max_iou_idx[0], :] = 0
            ious[:, max_iou_idx[1]] = 0
            text_sims[max_iou_idx[0], :] = 0
            text_sims[:, max_iou_idx[1]] = 0

        return best_matches  # List of index pairs [(gt_idx, dt_idx), ...]
