from typing import List, Optional

import torch
from transformers import PreTrainedModel
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX

from ...utils.constants import REGION_TOKEN_INDEX


def prepare_inputs_labels_for_multimodal(
    llm: PreTrainedModel,
    input_ids: torch.LongTensor = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    seg_pixel_values: Optional[torch.FloatTensor] = None,
    cond_ids: Optional[torch.LongTensor] = None,
    seg_ids: Optional[torch.LongTensor] = None,
    vprompt_feats: Optional[torch.FloatTensor] = None,
    **kwargs,
):
    if pixel_values is None:
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": None,
            "cond_ids": cond_ids,
            "vprompt_feats": vprompt_feats,
            "seg_ids": seg_ids,
            "labels": labels,
        }

    if seg_pixel_values is not None:
        pixel_values = torch.cat([pixel_values, seg_pixel_values], dim=1)

    _input_ids = input_ids
    _cond_ids = cond_ids
    _seg_ids = seg_ids
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
    else:
        attention_mask = attention_mask.bool().to(device=input_ids.device)
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if cond_ids is None:
        cond_ids = torch.full_like(input_ids, -1, dtype=torch.long, device=input_ids.device)
    if seg_ids is None:
        seg_ids = torch.full_like(input_ids, -1, dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX, device=input_ids.device)
    if vprompt_feats is None:
        vprompt_feats = [None] * len(input_ids)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    cond_ids = [cur_cond_ids[cur_attention_mask] for cur_cond_ids, cur_attention_mask in zip(cond_ids, attention_mask)]
    seg_ids = [cur_seg_ids[cur_attention_mask] for cur_seg_ids, cur_attention_mask in zip(seg_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_inputs_embeds = []
    new_input_ids = []
    new_cond_ids = []
    new_seg_ids = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        num_regions = (cur_input_ids == REGION_TOKEN_INDEX).sum()

        if num_images == 0 and num_regions == 0:
            cur_inputs_embeds = llm.get_input_embeddings()(cur_input_ids)
            cur_inputs_embeds = torch.cat([cur_inputs_embeds, cur_pixel_values[0:0]], dim=0)
            cur_cond_ids = cond_ids[batch_idx]
            cur_seg_ids = seg_ids[batch_idx]
            new_inputs_embeds.append(cur_inputs_embeds)
            new_input_ids.append(cur_input_ids)
            new_cond_ids.append(cur_cond_ids)
            new_seg_ids.append(cur_seg_ids)
            new_labels.append(labels[batch_idx])
            continue

        image_token_indices = (
            [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        )
        region_token_indices = (
            [-1] + torch.where(cur_input_ids == REGION_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        )

        # Process both image and region tokens
        all_special_token_indices = sorted(
            [(idx, "image") for idx in image_token_indices[1:-1]]
            + [(idx, "region") for idx in region_token_indices[1:-1]]
        )
        all_special_token_indices = [(-1, "none")] + all_special_token_indices + [(cur_input_ids.shape[0], "none")]

        cur_input_ids_nospecial = []
        cur_cond_ids_nospecial = []
        cur_seg_ids_nospecial = []
        cur_cond_ids = cond_ids[batch_idx]
        cur_seg_ids = seg_ids[batch_idx]
        cur_labels = labels[batch_idx]
        cur_vprompt_feats = vprompt_feats[batch_idx]
        cur_labels_nospecial = []

        for i in range(len(all_special_token_indices) - 1):
            start_idx = all_special_token_indices[i][0] + 1
            end_idx = all_special_token_indices[i + 1][0]

            if start_idx < end_idx:
                cur_input_ids_nospecial.append(cur_input_ids[start_idx:end_idx])
                cur_labels_nospecial.append(cur_labels[start_idx:end_idx])
                cur_cond_ids_nospecial.append(cur_cond_ids[start_idx:end_idx])
                cur_seg_ids_nospecial.append(cur_seg_ids[start_idx:end_idx])

        split_sizes = [x.shape[0] for x in cur_labels_nospecial]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_nospecial)
            if cur_input_ids_nospecial
            else torch.tensor([], device=llm.device, dtype=llm.dtype)
        )

        if cur_inputs_embeds.numel() > 0:
            cur_inputs_embeds_no_special = torch.split(cur_inputs_embeds, split_sizes, dim=0)
            cur_cond_ids_nospecial = (
                torch.split(
                    (
                        torch.cat(cur_cond_ids_nospecial)
                        if cur_cond_ids_nospecial
                        else torch.tensor([], device=cur_cond_ids.device, dtype=cur_cond_ids.dtype)
                    ),
                    split_sizes,
                    dim=0,
                )
                if split_sizes
                else []
            )
            cur_seg_ids_nospecial = (
                torch.split(
                    (
                        torch.cat(cur_seg_ids_nospecial)
                        if cur_seg_ids_nospecial
                        else torch.tensor([], device=cur_seg_ids.device, dtype=cur_seg_ids.dtype)
                    ),
                    split_sizes,
                    dim=0,
                )
                if split_sizes
                else []
            )
            cur_labels_nospecial = (
                torch.split(
                    (
                        torch.cat(cur_labels_nospecial)
                        if cur_labels_nospecial
                        else torch.tensor([], device=cur_labels.device, dtype=cur_labels.dtype)
                    ),
                    split_sizes,
                    dim=0,
                )
                if split_sizes
                else []
            )
        else:
            cur_inputs_embeds_no_special = []
            cur_cond_ids_nospecial = []
            cur_seg_ids_nospecial = []
            cur_labels_nospecial = []

        cur_new_inputs_embeds = []
        cur_new_input_ids = []
        cur_new_cond_ids = []
        cur_new_seg_ids = []
        cur_new_labels = []

        segment_idx = 0
        cur_region_idx = 0
        for i in range(len(all_special_token_indices) - 1):
            if segment_idx < len(cur_inputs_embeds_no_special):
                cur_new_inputs_embeds.append(cur_inputs_embeds_no_special[segment_idx])
                cur_new_input_ids.append(cur_input_ids_nospecial[segment_idx])
                cur_new_cond_ids.append(cur_cond_ids_nospecial[segment_idx])
                cur_new_labels.append(cur_labels_nospecial[segment_idx])
                cur_new_seg_ids.append(cur_seg_ids_nospecial[segment_idx])
                segment_idx += 1

            # Insert special token (image or region) if present
            if i < len(all_special_token_indices) - 1 and all_special_token_indices[i + 1][1] != "none":
                token_type = all_special_token_indices[i + 1][1]

                if token_type == "image":
                    cur_pixel_values = pixel_values[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_inputs_embeds.append(
                        cur_pixel_values.to(
                            dtype=cur_inputs_embeds.dtype if cur_inputs_embeds.numel() > 0 else torch.float32
                        )
                    )
                    cur_new_input_ids.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            IMAGE_TOKEN_INDEX,
                            device=cur_input_ids.device,
                            dtype=cur_input_ids.dtype,
                        )
                    )
                    cur_new_cond_ids.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            -1,
                            device=cur_cond_ids.device,
                            dtype=cur_cond_ids.dtype,
                        )
                    )
                    cur_new_seg_ids.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            -1,
                            device=cur_seg_ids.device,
                            dtype=cur_seg_ids.dtype,
                        )
                    )
                    cur_new_labels.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                elif token_type == "region" and cur_vprompt_feats is not None:
                    cur_region_feats = cur_vprompt_feats[cur_region_idx][None, :]
                    cur_new_inputs_embeds.append(
                        cur_region_feats.to(
                            dtype=cur_inputs_embeds.dtype if cur_inputs_embeds.numel() > 0 else torch.float16
                        )
                    )
                    cur_new_input_ids.append(
                        torch.full(
                            (cur_region_feats.shape[0],),
                            REGION_TOKEN_INDEX,
                            device=cur_input_ids.device,
                            dtype=cur_input_ids.dtype,
                        )
                    )
                    cur_new_cond_ids.append(
                        torch.full(
                            (cur_region_feats.shape[0],),
                            cur_region_idx,
                            device=cur_cond_ids.device,
                            dtype=cur_cond_ids.dtype,
                        )
                    )
                    cur_new_seg_ids.append(
                        torch.full(
                            (cur_region_feats.shape[0],),
                            -1,
                            device=cur_seg_ids.device,
                            dtype=cur_seg_ids.dtype,
                        )
                    )
                    cur_new_labels.append(
                        torch.full(
                            (cur_region_feats.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    cur_region_idx += 1

        if cur_new_inputs_embeds:
            cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
            cur_new_input_ids = torch.cat(cur_new_input_ids)
            cur_new_cond_ids = torch.cat(cur_new_cond_ids)
            cur_new_seg_ids = torch.cat(cur_new_seg_ids)
            cur_new_labels = torch.cat(cur_new_labels)

            new_inputs_embeds.append(cur_new_inputs_embeds)
            new_input_ids.append(cur_new_input_ids)
            new_cond_ids.append(cur_new_cond_ids)
            new_seg_ids.append(cur_new_seg_ids)
            new_labels.append(cur_new_labels)
        else:
            # Handle empty case
            device = input_ids[0].device
            dtype = input_ids[0].dtype
            empty_embeds = torch.zeros((0, llm.config.hidden_size), device=device, dtype=dtype)
            new_inputs_embeds.append(empty_embeds)
            new_input_ids.append(torch.tensor([], device=device, dtype=dtype))
            new_cond_ids.append(torch.tensor([], device=device, dtype=dtype))
            new_seg_ids.append(torch.tensor([], device=device, dtype=dtype))
            new_labels.append(torch.tensor([], device=device, dtype=dtype))

    # Combine them
    if not new_inputs_embeds:
        batch_size = _input_ids.shape[0]
        hidden_size = llm.config.hidden_size
        device = _input_ids.device
        dtype = torch.float32

        new_inputs_embeds = torch.zeros((batch_size, 0, hidden_size), device=device, dtype=dtype)
        return {
            "input_ids": _input_ids,
            "position_ids": _position_ids,
            "attention_mask": _attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": new_inputs_embeds,
            "cond_ids": _cond_ids,
            "seg_ids": _seg_ids,
            "labels": _labels,
        }

    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    new_inputs_embeds_padded = []
    new_input_ids_padded = torch.full(
        (batch_size, max_len),
        IGNORE_INDEX,
        dtype=new_input_ids[0].dtype,
        device=new_input_ids[0].device,
    )
    new_cond_ids_padded = torch.full(
        (batch_size, max_len),
        -1,
        dtype=new_cond_ids[0].dtype,
        device=new_cond_ids[0].device,
    )
    new_seg_ids_padded = torch.full(
        (batch_size, max_len),
        -1,
        dtype=new_seg_ids[0].dtype,
        device=new_seg_ids[0].device,
    )
    new_labels_padded = torch.full(
        (batch_size, max_len),
        IGNORE_INDEX,
        dtype=new_labels[0].dtype,
        device=new_labels[0].device,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_input_ids, cur_new_cond_ids, cur_new_seg_ids, cur_new_labels) in enumerate(
        zip(new_inputs_embeds, new_input_ids, new_cond_ids, new_seg_ids, new_labels)
    ):
        cur_len = cur_new_embed.shape[0]
        new_inputs_embeds_padded.append(
            torch.cat(
                (
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                ),
                dim=0,
            )
        )
        if cur_len > 0:
            new_input_ids_padded[i, :cur_len] = cur_new_input_ids
            new_cond_ids_padded[i, :cur_len] = cur_new_cond_ids
            new_seg_ids_padded[i, :cur_len] = cur_new_seg_ids
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

    if _input_ids is None:
        new_input_ids = None
    else:
        new_input_ids = new_input_ids_padded

    if _cond_ids is None:
        new_cond_ids = None
    else:
        new_cond_ids = new_cond_ids_padded

    if _seg_ids is None:
        new_seg_ids = None
    else:
        new_seg_ids = new_seg_ids_padded

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return {
        "input_ids": new_input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "inputs_embeds": new_inputs_embeds,
        "cond_ids": new_cond_ids,
        "seg_ids": new_seg_ids,
        "labels": new_labels,
    }
