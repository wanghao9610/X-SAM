from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from xsam.structures import BitMasks, Instances

from ...utils.mask import convert_segmentation_to_rle
from ...utils.process import sem_seg_postprocess


def vgd_seg_postprocess_fn(
    outputs,
    image_sizes,
    scaled_sizes,
    vprompt_masks=None,
    threshold=0.5,
    return_coco_annotation: Optional[bool] = False,
    return_binary_maps: Optional[bool] = False,
    return_contiguous_labels: Optional[bool] = False,
    sampled_labels: Optional[List[int]] = None,
    **kwargs,
) -> List[Dict]:
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

    metadata = kwargs.get("metadata", None)
    contiguous_labels = None
    if metadata is not None and hasattr(metadata, "dataset_id_to_contiguous_id"):
        contiguous_labels = list(metadata.dataset_id_to_contiguous_id.keys())

    # Loop over items in batch size
    results: List[Dict[str, TensorType]] = []

    for i in range(batch_size):
        mask_pred = masks_queries_logits[i]
        mask_cls = class_queries_logits[i]
        image_size = image_sizes[i]
        scaled_size = scaled_sizes[i]

        mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])
        vprompt_mask = (
            sem_seg_postprocess(
                vprompt_masks[i], scaled_sizes[i], image_sizes[i][0], image_sizes[i][1], mode="nearest"
            )
            if vprompt_masks is not None
            else None
        )

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

        sampled_label = None
        if return_contiguous_labels:
            assert contiguous_labels is not None and sampled_labels is not None
            sampled_label = sampled_labels[i]
            pred_classes = torch.tensor(
                [contiguous_labels.index(sampled_label[pred_class]) for pred_class in pred_classes], device=device
            )
            sampled_label = [contiguous_labels.index(label) for label in sampled_label]

        segmentation = torch.full((image_size[0], image_size[1]), 255, dtype=torch.long, device=mask_pred.device)

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
                "vprompt_masks": vprompt_mask,
                "sampled_labels": sampled_label,
                "num_classes": num_classes,
                "return_contiguous_labels": return_contiguous_labels,
            }
        )
    return results
