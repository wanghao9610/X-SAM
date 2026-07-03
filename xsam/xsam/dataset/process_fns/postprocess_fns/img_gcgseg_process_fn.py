from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from ...utils.process import cond_id_postprocess, sem_seg_postprocess


def img_gcgseg_postprocess_fn(
    outputs,
    image_sizes,
    scaled_sizes: Optional[List[TensorType]] = None,
    cond_ids: Optional[List[TensorType]] = None,
    mask_threshold: float = 0.5,
    use_bg_embeds: Optional[bool] = False,
    return_coco_annotation: Optional[bool] = False,
    return_binary_maps: Optional[bool] = False,
    ignore_value: int = 255,
    **kwargs,
) -> List[Dict]:

    if return_coco_annotation and return_binary_maps:
        raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

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

        segment_id = 1
        segments_info = []
        # 255 is the ignore index
        segmentation = torch.full(
            (image_size[0], image_size[1]), ignore_value, dtype=torch.long, device=mask_pred.device
        )

        if cond_id is not None and not use_bg_embeds:
            cond_id_map = cond_id_postprocess(cond_id, training=False).to(mask_cls.dtype)
            scores = mask_cls.sigmoid() @ cond_id_map.T
            # NOTE: gcgseg postprocess does not use sqrt
        elif cond_id is not None and use_bg_embeds:
            cond_id_map = cond_id_postprocess(cond_id, training=False).to(mask_cls.dtype)
            scores = mask_cls @ cond_id_map.T
            scores = F.softmax(scores, dim=-1)[:, :-1]
        else:
            scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        num_classes = scores.shape[-1]
        top_scores, top_indexs = scores.max(dim=0)

        for label_id in range(num_classes):
            top_score = top_scores[label_id]
            top_index = top_indexs[label_id]
            cur_mask_pred = mask_pred[top_index]
            cur_mask_prob = cur_mask_pred.sigmoid()

            segmentation[cur_mask_prob > mask_threshold] = segment_id
            segments_info.append(
                {
                    "id": segment_id,
                    "category_id": label_id,
                    "isthing": True,
                    "score": round(top_score.item(), 6),
                }
            )
            segment_id += 1

        results.append({"segmentation": segmentation, "segments_info": segments_info})
    return results
