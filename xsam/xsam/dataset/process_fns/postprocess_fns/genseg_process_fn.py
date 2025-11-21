from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from xsam.structures import BitMasks, Instances
from xsam.utils.logging import print_log

from ...utils.mask import convert_segmentation_to_rle
from ...utils.process import compute_segments, remove_low_and_no_objects, sem_seg_postprocess


def genseg_postprocess_fn(
    outputs,
    image_sizes,
    task_name: str = "genseg",
    scaled_sizes: Optional[List[TensorType]] = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    **kwargs,
) -> List[Dict]:

    def _semantic_genseg_postprocess(outputs, image_sizes, scaled_sizes, sampled_labels=None, **kwargs):
        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits
        scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes

        batch_size = class_queries_logits.shape[0]
        mask_classes = class_queries_logits.softmax(dim=-1)[:, :, :-1]
        mask_probs = masks_queries_logits.sigmoid()
        segmentations = torch.einsum("bqc,bqhw->bchw", mask_classes, mask_probs).cpu()

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            image_size = image_sizes[i]
            scaled_size = scaled_sizes[i]
            segmentation = segmentations[i]

            segmentation = sem_seg_postprocess(segmentation, scaled_size, image_size[0], image_size[1])
            segmentation = segmentation.argmax(dim=0)

            results.append(
                {
                    "segmentation": segmentation,
                    "sampled_labels": sampled_labels[i] if sampled_labels is not None else None,
                }
            )

        return results

    def _instance_genseg_postprocess(
        outputs,
        image_sizes,
        scaled_sizes,
        threshold,
        return_coco_annotation: Optional[bool] = False,
        return_binary_maps: Optional[bool] = False,
        **kwargs,
    ):
        if return_coco_annotation and return_binary_maps:
            raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        device = masks_queries_logits.device
        batch_size = class_queries_logits.shape[0]
        num_classes = class_queries_logits.shape[-1] - 1
        num_queries = class_queries_logits.shape[-2]

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_pred = masks_queries_logits[i]
            mask_cls = class_queries_logits[i]
            image_size = image_sizes[i]
            scaled_size = scaled_sizes[i]

            mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

            scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
            mask_pred = mask_pred[topk_indices]
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image

            segmentation = torch.zeros((image_size[0], image_size[1])) - 1

            instance_maps, segments = [], []
            current_segment_id = 0
            for j in range(num_queries):
                score = pred_scores[j].item()

                if not torch.all(pred_masks[j] == 0) and score >= threshold:
                    segmentation[pred_masks[j] == 1] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[j].item(),
                            "was_fused": False,
                            "score": round(score, 6),
                        }
                    )
                    current_segment_id += 1
                    instance_maps.append(pred_masks[j])

            # Return segmentation map in run-length encoding (RLE) format
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)

            # Return a concatenated tensor of binary instances maps
            if return_binary_maps and len(instance_maps) != 0:
                segmentation = torch.stack(instance_maps, dim=0)

            # Return the instances for d2
            keep = pred_scores >= threshold
            instances = Instances(image_size)
            instances.pred_masks = pred_masks[keep]
            instances.scores = pred_scores[keep]
            instances.pred_classes = pred_classes[keep]
            instances.pred_boxes = BitMasks(pred_masks[keep]).get_bounding_boxes()

            results.append(
                {
                    "segmentation": segmentation,
                    "segments_info": segments,
                    "instances": instances,
                }
            )
        return results

    def _genseg_postprocess(
        outputs,
        image_sizes,
        scaled_sizes,
        threshold,
        mask_threshold,
        overlap_mask_area_threshold,
        label_ids_to_fuse,
        **kwargs,
    ):
        # label_ids_to_fuse is the stuff_class_contiguous_ids
        if label_ids_to_fuse is None:
            print_log("`label_ids_to_fuse` unset. No instance will be fused.", logger="current")
            label_ids_to_fuse = set()

        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits
        scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_pred = masks_queries_logits[i]
            mask_cls = class_queries_logits[i]
            image_size = image_sizes[i]
            scaled_size = scaled_sizes[i]

            mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

            mask_prob = mask_pred.sigmoid()
            # the last class is __background__
            scores = F.softmax(mask_cls, dim=-1)[:, :-1]
            pred_score, pred_label = scores.max(-1)

            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_prob, pred_score, pred_label, threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = image_size if image_sizes is not None else mask_probs_item.shape[1:]
                # Official evaluation script uses 0 for VOID label.
                segmentation = torch.zeros((height, width), device=mask_pred.device, dtype=torch.long)
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item
            target_size = image_size if image_sizes is not None else None
            segmentation, segments_info = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=label_ids_to_fuse,
                target_size=target_size,
            )

            results.append({"segmentation": segmentation, "segments_info": segments_info})

        return results

    if "sem" in task_name:
        sampled_labels = kwargs.pop("sampled_labels", None)
        return _semantic_genseg_postprocess(outputs, image_sizes, scaled_sizes, sampled_labels=sampled_labels)
    elif "ins" in task_name:
        return_coco_annotation = kwargs.pop("return_coco_annotation", True)
        return_binary_maps = kwargs.pop("return_binary_maps", False)
        return _instance_genseg_postprocess(
            outputs,
            image_sizes,
            scaled_sizes,
            threshold,
            return_coco_annotation,
            return_binary_maps,
            **kwargs,
        )
    else:
        metadata = kwargs.pop("metadata", None)
        label_ids_to_fuse = None
        if metadata is not None and hasattr(metadata, "stuff_dataset_id_to_contiguous_id"):
            label_ids_to_fuse = metadata.stuff_dataset_id_to_contiguous_id.values()
        return _genseg_postprocess(
            outputs,
            image_sizes,
            scaled_sizes,
            threshold,
            mask_threshold,
            overlap_mask_area_threshold,
            label_ids_to_fuse,
            **kwargs,
        )
