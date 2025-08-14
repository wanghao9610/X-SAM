import json
import multiprocessing
import os
import time

import numpy as np
from panopticapi.utils import rgb2id
from PIL import Image
from tabulate import tabulate


class MiouStat:
    """
    Class to store and aggregate mIoU statistics
    """

    def __init__(self):
        self.confusion_matrix = None
        self.num_classes = 0

    def __iadd__(self, other):
        if self.confusion_matrix is None:
            self.confusion_matrix = other.confusion_matrix
            self.num_classes = other.num_classes
        else:
            self.confusion_matrix += other.confusion_matrix
        return self

    def compute_miou(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / (union + np.finfo(np.float32).eps)
        return np.mean(iou), iou


def miou_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    miou_stat = MiouStat()
    num_classes = len(categories)
    miou_stat.num_classes = num_classes
    miou_stat.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print("Core: {}, {} from {} images processed".format(proc_id, idx, len(annotation_set)))
        idx += 1

        # Load ground truth and prediction images
        gt_path = os.path.join(gt_folder, gt_ann["file_name"])
        pred_path = os.path.join(pred_folder, pred_ann["file_name"])

        pan_gt = np.array(Image.open(gt_path), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(pred_path), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        # Get segment information
        gt_segms = {el["id"]: el for el in gt_ann["segments_info"]}
        pred_segms = {el["id"]: el for el in pred_ann["segments_info"]}

        # Convert instance IDs to category IDs
        gt_cat = np.zeros_like(pan_gt, dtype=np.int64)
        pred_cat = np.zeros_like(pan_pred, dtype=np.int64)

        for gt_id, gt_info in gt_segms.items():
            if gt_info["iscrowd"] == 1:
                continue
            gt_cat[pan_gt == gt_id] = gt_info["category_id"]

        for pred_id, pred_info in pred_segms.items():
            pred_cat[pan_pred == pred_id] = pred_info["category_id"]

        # Update confusion matrix
        mask = (gt_cat >= 0) & (gt_cat < num_classes)
        hist = np.bincount(
            num_classes * gt_cat[mask].astype(int) + pred_cat[mask],
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)
        miou_stat.confusion_matrix += hist

    return miou_stat


def miou_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, "r") as f:
        gt_json = json.load(f)
    with open(pred_json_file, "r") as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace(".json", "")
    if pred_folder is None:
        pred_folder = pred_json_file.replace(".json", "")
    categories = {el["id"]: el for el in gt_json["categories"]}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el["image_id"]: el for el in pred_json["annotations"]}
    matched_annotations_list = []
    for gt_ann in gt_json["annotations"]:
        image_id = gt_ann["image_id"]
        if image_id not in pred_annotations:
            print("no prediction for the image with id: {}".format(image_id))
            continue
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    results = miou_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)
    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


def miou_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))

    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []

    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(
            miou_compute_single_core, (proc_id, annotation_set, gt_folder, pred_folder, categories)
        )
        processes.append(p)

    miou_stat = MiouStat()
    for p in processes:
        miou_stat += p.get()

    # Compute final mIoU
    mean_iou, class_ious = miou_stat.compute_miou()

    # Prepare results dictionary
    results = {
        "mIoU": mean_iou,
        "IoUs": {cat["name"]: iou for cat, iou in zip(categories.values(), class_ious)},
    }

    return results


def print_miou_results(miou_res, display_cats=False):
    """
    Print mIoU evaluation results in a formatted table
    Args:
        miou_res (dict): Dictionary containing mIoU results with keys:
            - mIoU: float value of mean IoU
            - IoUs: dict mapping category names to their IoU values
    Returns:
        str: Formatted table string
    """
    headers = ["", "IoU"]
    data = []

    # Add mean IoU row
    data.append(["All", miou_res["mIoU"] * 100])

    # Add per-category IoUs
    if display_cats:
        # Add separator
        data.append(["-" * 20, "-" * 10])
        for cat_name, iou in miou_res["IoUs"].items():
            data.append([cat_name, iou * 100])

    table = tabulate(
        data,
        headers=headers,
        tablefmt="outline",
        floatfmt=".2f",
        stralign="center",
        numalign="center",
    )

    return table


def compute_iou(mask1, mask2):
    """Computing IoU between two masks"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    return iou


def compute_miou(pred_masks, gt_masks):
    """Computing mIoU between predicted masks and ground truth masks"""
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    # One-to-one pairing and mean IoU calculation
    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        paired_iou.append(iou_matrix[max_iou_idx])
        iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
        iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)

    return np.mean(paired_iou) if paired_iou else 0.0


def compute_iou_matrix(pred_masks, gt_masks):
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    return iou_matrix
