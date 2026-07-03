from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .panoptic import IdGenerator


def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        mask1: Binary mask tensor of shape (H, W)
        mask2: Binary mask tensor of shape (H, W)

    Returns:
        IoU value as float
    """
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    if union == 0:
        return 0.0
    return intersection / union


class VideoInstanceTracker:
    """
    Track instances across video frames using IoU-based matching.
    This maintains consistent instance IDs for VPQ evaluation with k > 1.
    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: Minimum IoU to consider a match between frames
        """
        self.iou_threshold = iou_threshold
        # Dict mapping (category_id, instance_key) -> segment_id for previous frame
        self.prev_instances: Dict[int, Dict] = {}  # segment_id -> {mask, category_id}
        self.id_to_track: Dict[int, int] = {}  # current frame segment_id -> track_id

    def reset(self):
        """Reset tracker for a new video."""
        self.prev_instances = {}
        self.id_to_track = {}

    def match_and_track(
        self,
        segmentation: torch.Tensor,
        segments: List[Dict],
        label_ids_to_fuse: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Match current frame instances with previous frame and update IDs for tracking.

        Args:
            segmentation: Segmentation map of shape (H, W) with segment IDs
            segments: List of segment info dicts
            label_ids_to_fuse: Set of stuff class IDs (don't track these, already fused)

        Returns:
            Updated segmentation and segments with tracked IDs
        """
        if label_ids_to_fuse is None:
            label_ids_to_fuse = set()

        # Build current frame instance info
        current_instances = {}
        for seg in segments:
            seg_id = seg["id"]
            cat_id = seg["category_id"]
            # Get binary mask for this segment
            mask = segmentation == seg_id
            current_instances[seg_id] = {
                "mask": mask,
                "category_id": cat_id,
                "segment": seg,
                "is_thing": cat_id not in label_ids_to_fuse,
            }

        # If no previous frame, just store current and return
        if not self.prev_instances:
            self.prev_instances = {
                seg_id: {"mask": info["mask"].clone(), "category_id": info["category_id"]}
                for seg_id, info in current_instances.items()
                if info["is_thing"]  # Only track things, not stuff
            }
            return segmentation, segments

        # Match current instances to previous frame using IoU
        id_mapping = {}  # old_id -> new_id (tracked)
        matched_prev = set()

        # For each current thing instance, find best match in previous frame
        for cur_id, cur_info in current_instances.items():
            if not cur_info["is_thing"]:
                continue  # Skip stuff classes

            best_match_id = None
            best_iou = self.iou_threshold

            for prev_id, prev_info in self.prev_instances.items():
                if prev_id in matched_prev:
                    continue
                # Only match same category
                if cur_info["category_id"] != prev_info["category_id"]:
                    continue

                iou = compute_mask_iou(cur_info["mask"], prev_info["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = prev_id

            if best_match_id is not None:
                # Found a match - reuse the previous ID
                id_mapping[cur_id] = best_match_id
                matched_prev.add(best_match_id)

        # Apply ID mapping to segmentation and segments
        if id_mapping:
            new_segmentation = segmentation.clone()
            new_segments = []

            for seg in segments:
                old_id = seg["id"]
                if old_id in id_mapping:
                    new_id = id_mapping[old_id]
                    new_segmentation[segmentation == old_id] = new_id
                    seg = seg.copy()
                    seg["id"] = new_id
                new_segments.append(seg)

            segmentation = new_segmentation
            segments = new_segments

        # Update previous instances for next frame
        self.prev_instances = {
            seg["id"]: {"mask": (segmentation == seg["id"]).clone(), "category_id": seg["category_id"]}
            for seg in segments
            if seg["category_id"] not in label_ids_to_fuse
        }

        return segmentation, segments


# Copied from transformers.models.detr.image_processing_detr.remove_low_and_no_objects
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


# Copied from transformers.models.detr.image_processing_detr.check_segment_validity
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # Get the mask associated with the k class
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # Compute the area of all the stuff in query k
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # Eliminate disconnected tiny segments
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# Copied from transformers.models.detr.image_processing_detr.compute_segments
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
    id_generator: Optional[IdGenerator] = None,
    contiguous_id_to_dataset_id: Optional[Dict[int, int]] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []

    if target_size is not None:
        mask_probs = F.interpolate(
            mask_probs.unsqueeze(0),
            size=(target_size[0], target_size[1]),
            mode="bilinear",
            align_corners=False,
        )[0]

    segment_id = 0

    # Weigh each mask by its prediction score
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]
    # Keep track of instances of each class
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # Check if mask exists and large enough to be a segment
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                # Reuse existing segment ID for stuff classes
                segment_id = stuff_memory_list[pred_class]
            else:
                # Create new segment ID
                segment_id = (
                    id_generator.get_id(
                        contiguous_id_to_dataset_id[pred_class]
                        if contiguous_id_to_dataset_id is not None
                        else pred_class
                    )
                    if id_generator is not None
                    else segment_id + 1
                )

            # Add pixels to segmentation map
            segmentation[mask_k] = segment_id

            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": segment_id,
                    "category_id": pred_class,
                    "isthing": not should_fuse,
                    "score": segment_score,
                    "iscrowd": 0,
                    "area": mask_k.sum().item(),
                }
            )
            if should_fuse:
                stuff_memory_list[pred_class] = segment_id

    return segmentation, segments


def sem_seg_postprocess(result, img_size, output_height, output_width, mode="bilinear"):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.
        mode (str): the interpolation mode.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    if mode == "bilinear":
        result = F.interpolate(result, size=(output_height, output_width), mode=mode, align_corners=False)[0]
    elif mode == "nearest":
        result = F.interpolate(result.float(), size=(output_height, output_width), mode=mode)[0]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return result


def cond_id_postprocess(cond_id, label_ids=None, ignore_label=-100, background_label=-1, training=True):
    label_ids = label_ids if label_ids is not None and label_ids.shape[0] > 0 else torch.unique(cond_id)
    if ((label_ids != ignore_label) & (label_ids != background_label)).sum() == 0:
        return torch.zeros((0, cond_id.shape[0]), dtype=torch.float32, device=cond_id.device)
    cond_id_map = torch.stack([cond_id == x for x in label_ids if x != background_label and x != ignore_label]).to(
        torch.float32
    )
    if ignore_label in label_ids and training:
        null_cond_id_map = (cond_id == ignore_label).unsqueeze(0).to(torch.float32)
        cond_id_map = torch.cat([cond_id_map, null_cond_id_map], dim=0)
    if background_label in label_ids:
        bg_cond_id_map = (cond_id == background_label).unsqueeze(0).to(torch.float32)
        cond_id_map = torch.cat([cond_id_map, bg_cond_id_map], dim=0)

    cond_id_map = cond_id_map / (cond_id_map.sum(dim=-1)[:, None] + 1e-6)
    return cond_id_map
