from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from ...utils.process import sem_seg_postprocess


def refer_seg_postprocess_fn(
    outputs,
    image_sizes,
    scaled_sizes: Optional[List[TensorType]] = None,
    mask_threshold: float = 0.5,
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
        mask_pred = masks_queries_logits[i]
        mask_cls = class_queries_logits[i]
        image_size = image_sizes[i]
        scaled_size = scaled_sizes[i]

        mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

        mask_prob = mask_pred.sigmoid()
        # the last class is __background__
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        top_score, top_index = scores.max(dim=0)
        mask_pred = mask_pred[top_index]
        mask_prob = mask_pred.sigmoid()

        # 255 is the ignore index
        segmentation = torch.full((image_size[0], image_size[1]), 255, dtype=torch.long, device=mask_pred.device)
        segmentation[mask_prob[0] > mask_threshold] = 1

        segments_info = {
            "id": 0,
            "label_id": 0,
            "was_fused": False,
            "score": round(top_score.item(), 6),
        }

        results.append({"segmentation": segmentation, "segments_info": segments_info})
    return results
