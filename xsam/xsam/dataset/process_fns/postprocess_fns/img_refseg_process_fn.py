from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from ...utils.process import cond_id_postprocess, sem_seg_postprocess


def img_refseg_postprocess_fn(
    outputs,
    image_sizes,
    cond_ids: Optional[List[TensorType]] = None,
    scaled_sizes: Optional[List[TensorType]] = None,
    mask_threshold: float = 0.5,
    use_bg_embeds: Optional[bool] = False,
    ignore_value: int = 255,
    **kwargs,
) -> List[Dict]:
    # [batch_size, num_queries, num_classes]
    class_queries_logits = outputs.class_queries_logits
    # [batch_size, num_queries, height, width]
    masks_queries_logits = outputs.masks_queries_logits
    scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes

    batch_size = class_queries_logits.shape[0]

    # Loop over items in batch size
    results: List[Dict[str, TensorType]] = []

    for i in range(batch_size):
        mask_pred = masks_queries_logits[i]
        mask_cls = class_queries_logits[i]
        image_size = image_sizes[i]
        scaled_size = scaled_sizes[i]
        cond_id = cond_ids[i] if cond_ids is not None else None

        mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

        if cond_id is not None and not use_bg_embeds:
            cond_id_map = cond_id_postprocess(cond_id, training=False).to(mask_cls.dtype)
            scores = mask_cls.sigmoid() @ cond_id_map.T
            # NOTE: refseg postprocess does not use sqrt
        elif cond_id is not None and use_bg_embeds:
            cond_id_map = cond_id_postprocess(cond_id, training=False).to(mask_cls.dtype)
            scores = mask_cls @ cond_id_map.T
            scores = F.softmax(scores, dim=-1)[:, :-1]
        else:
            # the last class is __background__
            scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        num_classes = scores.shape[-1]
        assert num_classes == 1
        top_score, top_index = scores.max(dim=0)
        mask_pred = mask_pred[top_index]
        mask_prob = mask_pred.sigmoid()

        # 255 is the ignore index
        segmentation = torch.full(
            (image_size[0], image_size[1]), ignore_value, dtype=torch.long, device=mask_pred.device
        )
        segmentation[mask_prob[0] > mask_threshold] = 1

        segments_info = {
            "id": 0,
            "label_id": 0,
            "was_fused": False,
            "score": round(top_score.item(), 6),
        }

        results.append({"segmentation": segmentation, "segments_info": segments_info})
    return results


def img_grefseg_postprocess_fn(
    outputs,
    image_sizes,
    cond_ids: Optional[List[TensorType]] = None,
    scaled_sizes: Optional[List[TensorType]] = None,
    mask_threshold: float = 0.5,
    cls_threshold: float = 0.6,
    use_bg_embeds: Optional[bool] = False,
    ignore_value: int = 255,
    **kwargs,
) -> List[Dict]:
    # [batch_size, num_queries, num_classes+1]
    class_queries_logits = outputs.class_queries_logits
    # [batch_size, num_queries, height, width]
    masks_queries_logits = outputs.masks_queries_logits
    scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes

    batch_size = class_queries_logits.shape[0]
    num_labels = class_queries_logits.shape[-1] - 1
    assert num_labels == 1

    # Loop over items in batch size
    results: List[Dict[str, TensorType]] = []

    for i in range(batch_size):
        mask_pred = masks_queries_logits[i].cpu()
        mask_cls = class_queries_logits[i].cpu()
        image_size = image_sizes[i]
        scaled_size = scaled_sizes[i]
        cond_id = cond_ids[i] if cond_ids is not None else None

        mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

        if cond_id is not None and not use_bg_embeds:
            cond_id_map = cond_id_postprocess(cond_id, training=False).to(mask_cls.dtype)
            scores = mask_cls.sigmoid() @ cond_id_map.T
            # NOTE: refseg postprocess does not use sqrt
        elif cond_id is not None and use_bg_embeds:
            cond_id_map = cond_id_postprocess(cond_id, training=False).to(mask_cls.dtype)
            scores = mask_cls @ cond_id_map.T
            scores = F.softmax(scores, dim=-1)[:, :-1]
        else:
            # the last class is __background__
            scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        num_classes = scores.shape[-1]
        assert num_classes == 1
        mask_prob = mask_pred.sigmoid()
        keep = (scores >= cls_threshold) | (scores == scores.max())
        pred_scores = scores[keep].mean()
        mask_pred = mask_pred[keep[:, 0]]
        mask_prob = mask_pred.sigmoid()

        # 255 is the ignore index
        segmentation = torch.full(
            (image_size[0], image_size[1]), ignore_value, dtype=torch.long, device=mask_pred.device
        )
        for _mask_prob in mask_prob:
            segmentation[_mask_prob > mask_threshold] = 1

        segments_info = {
            "id": 0,
            "label_id": 0,
            "was_fused": False,
            "score": round(pred_scores.item(), 6),
        }

        results.append({"segmentation": segmentation, "segments_info": segments_info})
    return results
