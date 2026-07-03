import copy
import itertools
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import traceback
from functools import partial

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from xsam.dataset.utils.coco import COCO
from xsam.dataset.utils.mask import decode_mask, encode_mask
from xsam.utils import comm
from xsam.utils.logging import print_log

from ..utils.cococapeval import COCOEvalCap
from ..utils.map import create_small_table, derive_coco_results
from ..utils.miou import compute_iou_matrix, compute_miou
from .img_base_evaluator import ImgBaseEvaluator


class ImgGCGSegEvaluator(ImgBaseEvaluator):

    def __init__(
        self,
        *args,
        data_name: str = "img_gcgseg",
        text_model: str = "bert-base-uncased",
        evaluation_metrics: list[str] = ["miou", "map", "caption", "recall"],
        **kwargs,
    ):
        super().__init__(
            *args,
            data_name,
            evaluation_metrics,
            **kwargs,
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model)
        self.text_model.eval()

    def _encode_text_embeddings(self, text):
        input_dict = self.text_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        text_embeddings = self.text_model(**input_dict)
        text_embeddings = torch.mean(text_embeddings.last_hidden_state[0], dim=0).detach().numpy()
        return text_embeddings

    def _get_text_similarity(self, text1, text2):
        text1_embeddings = self._encode_text_embeddings(text1)
        text2_embeddings = self._encode_text_embeddings(text2)
        return cosine_similarity([text1_embeddings], [text2_embeddings])[0, 0]

    def _get_rle_masks(self, segmentation, segments_info):
        rle_masks = []
        for seg_info in segments_info:
            binary_mask = (segmentation == seg_info["id"]).astype(np.uint8)
            rle_masks.append(encode_mask(binary_mask))
        return rle_masks

    # follow segmentation evaluation
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            segmentation, segments_info, gcg_phrases, gcg_caption = (
                output["segmentation"],
                output["segments_info"],
                output["gcg_phrases"],
                output["gcg_caption"],
            )
            segmentation = segmentation.to(self._cpu_device)
            segmentation = np.array(segmentation, dtype=np.uint8)
            segmentation = self._get_rle_masks(segmentation, segments_info)
            file_name = os.path.basename(input["file_name"])
            self._predictions.append(
                {
                    "image_id": input["image_id"],
                    "file_name": file_name,
                    "segmentation": segmentation,
                    "segments_info": segments_info,
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
            pred
            for pred in predictions
            if isinstance(pred, dict) and ("segmentation" in pred) and ("segments_info" in pred)
        ]
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
        gt_json = osp.realpath(self._metadata.gt_json)
        cap_gt_json = osp.realpath(self._metadata.cap_gt_json)

        mask_preds = []
        caption_preds = []
        mask_ann_id = 0
        for i, pred in enumerate(predictions):
            for seg, seg_info in zip(pred["segmentation"], pred["segments_info"]):
                cur_mask_pred = {
                    "id": mask_ann_id,
                    "image_id": pred["image_id"],
                    "category_id": seg_info["category_id"],
                    "segmentation": seg,
                    "score": seg_info["score"],
                }
                mask_preds.append(cur_mask_pred)
                mask_ann_id += 1
            cur_caption_pred = {
                "id": i,
                "image_id": pred["image_id"],
                "caption": pred["gcg_caption"],
                "labels": pred["gcg_phrases"],
            }
            caption_preds.append(cur_caption_pred)

        for metric in self._evaluation_metrics:
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
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt_data["info"] = {"description": "GCGSeg dataset"}
        pred_img_ids = set([pred["image_id"] for pred in preds])
        gt_img_ids = set([img["id"] for img in gt_data["images"]])
        if set(pred_img_ids) != set(gt_img_ids):
            print_log(
                f"The number of images in predictions and ground truth is not the same. There are {len(pred_img_ids)} predictions and {len(gt_img_ids)} ground truth images.",
                logger="current",
            )
            gt_data["images"] = [img for img in gt_data["images"] if img["id"] in pred_img_ids]
            gt_data["annotations"] = [ann for ann in gt_data["annotations"] if ann["image_id"] in pred_img_ids]

        coco_gt = COCO(dataset=gt_data)
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
            mious.append(compute_miou(gt_masks, dt_masks))

        miou_res = float(np.mean(mious) * 100) if mious else 0.0
        data = [["Value (%)", f"{miou_res:.2f}"]]
        headers = ["Metric", "mIoU"]
        table = tabulate(
            data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center"
        )
        print_log(f"{self.data_name} mIoU results:\n{table}", logger="current")

    def _eval_map(self, preds, gt_json):
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt_data["info"] = {"description": "GCGSeg dataset"}
        pred_img_ids = set([pred["image_id"] for pred in preds])
        gt_img_ids = set([img["id"] for img in gt_data["images"]])
        if set(pred_img_ids) != set(gt_img_ids):
            print_log(
                f"The number of images in predictions and ground truth is not the same. There are {len(pred_img_ids)} predictions and {len(gt_img_ids)} ground truth images.",
                logger="current",
            )
            gt_data["images"] = [img for img in gt_data["images"] if img["id"] in pred_img_ids]
            gt_data["annotations"] = [ann for ann in gt_data["annotations"] if ann["image_id"] in pred_img_ids]

        for pred in preds:
            pred["category_id"] = 1

        coco_gt = COCO(dataset=gt_data)
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        imgids = sorted(list(set([pred["image_id"] for pred in preds])))
        coco_eval.params.imgIds = imgids
        coco_eval.params.catIds = [1]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results = derive_coco_results(coco_eval, "segm")
        table = create_small_table(results)
        print_log(f"{self.data_name} mAP results:\n{table}", logger="current")

    def _eval_caption(self, preds, gt_json):
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt_data["info"] = {"description": "GCGSeg dataset"}

        pred_img_ids = set([pred["image_id"] for pred in preds])
        gt_img_ids = set([img["id"] for img in gt_data["images"]])
        if set(pred_img_ids) != set(gt_img_ids):
            print_log(
                f"The number of images in predictions and ground truth is not the same. There are {len(pred_img_ids)} predictions and {len(gt_img_ids)} ground truth images.",
                logger="current",
            )
            gt_data["images"] = [img for img in gt_data["images"] if img["id"] in pred_img_ids]
            gt_data["annotations"] = [ann for ann in gt_data["annotations"] if ann["image_id"] in pred_img_ids]

        coco_gt = COCO(dataset=gt_data)
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOEvalCap(coco_gt, coco_dt)
        imgids = sorted(list(set([pred["image_id"] for pred in preds])))
        coco_eval.params["image_id"] = imgids
        coco_eval.evaluate()

        metrics = list(coco_eval.eval.keys())
        headers = ["Metric", *metrics]
        data = [["Value (%)", *[f"{coco_eval.eval[metric] * 100:.2f}" for metric in metrics]]]
        table = tabulate(
            data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center"
        )

        print_log(f"{self.data_name} caption results:\n{table}", logger="current")

    def _eval_recall(self, mask_preds, caption_preds, gt_json, cap_gt_json):
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt_data["info"] = {"description": "GCGSeg dataset"}
        with open(cap_gt_json, "r") as f:
            cap_gt_data = json.load(f)
            cap_gt_data["info"] = {"description": "GCGSeg dataset"}
        mask_pred_img_ids = set([pred["image_id"] for pred in mask_preds])
        gt_img_ids = set([img["id"] for img in gt_data["images"]])
        cap_gt_img_ids = set([img["id"] for img in cap_gt_data["images"]])
        if set(mask_pred_img_ids) != set(gt_img_ids):
            print_log(
                f"The number of images in predictions and ground truth is not the same. There are {len(mask_pred_img_ids)} predictions and {len(gt_img_ids)} ground truth images.",
                logger="current",
            )
            gt_data["images"] = [img for img in gt_data["images"] if img["id"] in mask_pred_img_ids]
            gt_data["annotations"] = [ann for ann in gt_data["annotations"] if ann["image_id"] in mask_pred_img_ids]

        caption_pred_img_ids = set([pred["image_id"] for pred in caption_preds])
        if set(caption_pred_img_ids) != set(cap_gt_img_ids):
            print_log(
                f"The number of images in predictions and ground truth is not the same. There are {len(caption_pred_img_ids)} predictions and {len(cap_gt_img_ids)} ground truth images.",
                logger="current",
            )
            cap_gt_data["images"] = [img for img in cap_gt_data["images"] if img["id"] in caption_pred_img_ids]
            cap_gt_data["annotations"] = [
                ann for ann in cap_gt_data["annotations"] if ann["image_id"] in caption_pred_img_ids
            ]

        coco_gt = COCO(dataset=gt_data)
        cap_coco_gt = COCO(dataset=cap_gt_data)
        coco_dt = coco_gt.loadRes(mask_preds)
        cap_coco_dt = cap_coco_gt.loadRes(caption_preds)
        imgids = sorted(list(set([pred["image_id"] for pred in mask_preds])))

        # Pre-compute text similarities for all images to avoid PyTorch model in workers
        text_sims = {}
        for imgid in imgids:
            cap_gt_ann_ids = cap_coco_gt.getAnnIds(imgIds=[imgid])
            cap_gt_anns = cap_coco_gt.loadAnns(cap_gt_ann_ids)[0]
            cap_dt_ann_ids = cap_coco_dt.getAnnIds(imgIds=[imgid])
            cap_dt_anns = cap_coco_dt.loadAnns(cap_dt_ann_ids)[0]

            cap_gt_labels = cap_gt_anns["labels"]
            cap_dt_labels = cap_dt_anns["labels"]

            text_sim = np.zeros((len(cap_gt_labels), len(cap_dt_labels)))
            for i, gt_label in enumerate(cap_gt_labels):
                for j, dt_label in enumerate(cap_dt_labels):
                    text_sim[i, j] = self._get_text_similarity(gt_label, dt_label)
            text_sims[imgid] = text_sim

        # Initialize multiprocessing pool
        num_workers = min(16, len(imgids))
        with mp.Pool(num_workers) as pool:
            process_func = partial(
                self._process_single_image,
                coco_gt=coco_gt,
                cap_coco_gt=cap_coco_gt,
                coco_dt=coco_dt,
                cap_coco_dt=cap_coco_dt,
                text_sims=text_sims,
            )
            results = list(
                tqdm(pool.imap_unordered(process_func, imgids), total=len(imgids), desc="Calculating recall")
            )

        # Sum up results
        p_cnt = sum(p for p, _ in results)
        tp_cnt = sum(tp for _, tp in results)

        recall = tp_cnt / p_cnt if p_cnt > 0 else 0
        headers = ["Metric", "Recall"]
        data = [["Value (%)", f"{recall * 100:.2f}"]]
        table = tabulate(
            data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center"
        )

        print_log(f"{self.data_name} recall results:\n{table}", logger="current")

    def _process_single_image(self, imgid, coco_gt, cap_coco_gt, coco_dt, cap_coco_dt, text_sims=None):
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

            text_sim = text_sims[imgid]
            best_matches = self._find_best_matches(
                gt_anns, cap_gt_labels, dt_anns, cap_dt_labels, height, width, text_sims=text_sim
            )
            return len(cap_gt_labels), len(best_matches)
        except Exception as e:
            print_log(f"Error processing image {imgid}: {e}\n{traceback.format_exc()}", logger="current")
            return 0, 0

    def _find_best_matches(
        self,
        gt_anns,
        gt_labels,
        dt_anns,
        dt_labels,
        height,
        width,
        text_sims=None,
        iou_threshold=0.5,
        text_sim_threshold=0.5,
    ):
        best_matches = []

        # Compute pair - wise IoU
        gt_masks = [decode_mask(ann["segmentation"], height, width) for ann in gt_anns]
        pred_masks = [decode_mask(ann["segmentation"], height, width) for ann in dt_anns]
        ious = compute_iou_matrix(gt_masks, pred_masks)

        if text_sims is None:
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
