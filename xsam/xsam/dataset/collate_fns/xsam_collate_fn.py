from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xsam.structures.data_sample import DataSample
from xsam.utils.constants import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from xsam.utils.dist import get_sequence_parallel_world_size

from ..utils.collate import pad_for_sequence_parallel


def xsam_collate_fn(
    instances: Sequence[Dict],
    pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
    return_hf_format: bool = False,
    use_varlen_attn: bool = False,
):
    seq_parallel_world_size = get_sequence_parallel_world_size()

    data_samples = DataSample()
    metainfo = {}
    has_input_ids = any(inst.get("input_ids", None) is not None for inst in instances)
    has_image = any(inst.get("pixel_values", None) is not None for inst in instances)
    has_image_grid_thw = any(inst.get("image_grid_thw", None) is not None for inst in instances)
    has_seg_image = any(inst.get("extra_pixel_values", None) is not None for inst in instances)
    has_cond_id = any(inst.get("cond_ids", None) is not None for inst in instances)
    has_vprompt_mask = any(inst.get("vprompt_masks", None) is not None for inst in instances)
    has_seg_id = any(inst.get("seg_ids", None) is not None for inst in instances)
    has_mask_label = any(inst.get("mask_labels", None) is not None for inst in instances)
    has_class_label = any(inst.get("class_labels", None) is not None for inst in instances)
    has_sampled_labels = any(inst.get("sampled_labels", None) is not None for inst in instances)
    has_contiguous_labels = any(inst.get("contiguous_labels", None) is not None for inst in instances)

    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f"If utilizing varlen attention, the batch size should be" f" set to 1, but got {len(instances)}"
        )
        assert not has_image, "Currently, it is not configured to "
        "accommodate the use of varlen Attention in multimodal training"

    if has_input_ids:
        input_ids = []
        labels = []
    if has_image:
        pixel_values = []
    if has_image or has_seg_image:
        image_files = []
        image_sizes = []
    if has_image_grid_thw:
        image_grid_thw = []
    if has_seg_image:
        extra_pixel_values = []
        scaled_sizes = []
        image_infos = []
        task_names = []
        data_names = []
    if has_cond_id:
        cond_ids = []
    if has_seg_id:
        seg_ids = []
    if has_vprompt_mask:
        vprompt_masks = []
        vprompt_indices = []
    if has_mask_label:
        mask_labels = []
    if has_class_label:
        class_labels = []
    if has_sampled_labels:
        sampled_labels = []
    if has_contiguous_labels:
        contiguous_labels = []

    for example in instances:
        if has_input_ids:
            input_ids.append(torch.LongTensor(example["input_ids"]))
            labels.append(torch.LongTensor(example["labels"]))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example["cumulative_len"]))
            position_ids.append(torch.LongTensor(example["position_ids"]))

        if has_image:
            pixel_values.append(example["pixel_values"])
            image_files.append(example.get("image_file", None))
            image_sizes.append(example.get("image_size", None))
        if has_image_grid_thw:
            image_grid_thw.append(example["image_grid_thw"])
        if has_seg_image:
            if not has_image:
                image_files.append(example.get("image_file", None))
                image_sizes.append(example.get("image_size", None))
            extra_pixel_values.append(example["extra_pixel_values"])
            scaled_sizes.append(example.get("scaled_size", None))
            image_infos.append(example.get("image_info", None))
            task_names.append(example.get("task_name", None))
            data_names.append(example.get("data_name", None))
        if has_cond_id:
            cond_ids.append(torch.LongTensor(example["cond_ids"]))
        if has_seg_id:
            seg_ids.append(torch.LongTensor(example["seg_ids"]))
        if has_vprompt_mask:
            vprompt_masks.append(example["vprompt_masks"])
            vprompt_indices.append(example.get("vprompt_index", 0))
        if has_mask_label:
            mask_labels.append(example["mask_labels"])
        if has_class_label:
            class_labels.append(example["class_labels"])
        if has_sampled_labels:
            sampled_labels.append(example["sampled_labels"])
        if has_contiguous_labels:
            contiguous_labels.append(example["contiguous_labels"])

    if len(instances) > 1:
        if has_input_ids:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if has_cond_id:
            cond_ids = pad_sequence(cond_ids, batch_first=True, padding_value=-1)
        if has_seg_id:
            seg_ids = pad_sequence(seg_ids, batch_first=True, padding_value=-1)
    else:
        if has_input_ids:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
        if has_cond_id:
            cond_ids = torch.stack(cond_ids)
        if has_seg_id:
            seg_ids = torch.stack(seg_ids)

    if has_input_ids:
        ori_length = [len(ids) for ids in input_ids]
        if use_varlen_attn:
            assert input_ids.size(1) % seq_parallel_world_size == 0
            attention_mask = None
            position_ids = torch.stack(position_ids, dim=0)
        else:
            # Some tokenizers have the same eos token and pad token, so input_ids
            # cannot be masked directly based on the pad token id.
            attention_mask = torch.zeros_like(input_ids).bool()
            for i, length in enumerate(ori_length):
                attention_mask[i, :length] = True

            bs, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

        if seq_parallel_world_size > 1:
            input_ids = pad_for_sequence_parallel(input_ids, pad_index)
            labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
            position_ids = pad_for_sequence_parallel(position_ids, 0)
            if attention_mask is not None:
                attention_mask = pad_for_sequence_parallel(attention_mask, 0)

        if use_varlen_attn:
            max_seqlen = (cumulative_len[0][1:] - cumulative_len[0][:-1]).max().item()  # noqa: W504
            data_dict = {
                "input_ids": input_ids,
                "cumulative_len": cumulative_len,
                "position_ids": position_ids,
                "labels": labels,
                "max_seqlen": max_seqlen,
            }
        else:
            data_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "labels": labels,
            }
    else:
        data_dict = {}

    if has_image:
        # For CLIP-like image processing
        if pixel_values[0].ndim == 3 and all(x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = torch.stack(pixel_values, dim=0)
        # For qwen_vl image processing
        elif pixel_values[0].ndim == 2 and all(x.shape[1] == pixel_values[0].shape[1] for x in pixel_values):
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            raise ValueError(
                f"Pixel_values have different shapes: {[pixel_value.shape for pixel_value in pixel_values]}"
            )
        data_dict["pixel_values"] = pixel_values

    if has_image or has_seg_image:
        metainfo = {"image_files": image_files, "image_sizes": image_sizes}

    if has_image_grid_thw:
        data_dict["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

    if has_seg_image:
        if all(x.shape == extra_pixel_values[0].shape for x in extra_pixel_values):
            extra_pixel_values = torch.stack(extra_pixel_values, dim=0)
        data_dict["extra_pixel_values"] = extra_pixel_values
        metainfo["scaled_sizes"] = scaled_sizes
        metainfo["image_infos"] = image_infos
        metainfo["task_names"] = task_names
        metainfo["data_names"] = data_names

    if has_cond_id:
        data_dict["cond_ids"] = cond_ids

    if has_seg_id:
        data_dict["seg_ids"] = seg_ids

    if has_vprompt_mask:
        data_dict["vprompt_masks"] = vprompt_masks
        metainfo["vprompt_indices"] = vprompt_indices

    if has_mask_label:
        data_samples.mask_labels = mask_labels

    if has_class_label:
        data_samples.class_labels = class_labels

    if has_sampled_labels:
        data_samples.sampled_labels = sampled_labels

    if has_contiguous_labels:
        data_samples.contiguous_labels = contiguous_labels

    data_samples.set_metainfo(metainfo)

    if return_hf_format:
        return data_dict
    else:
        return {"data_dict": data_dict, "data_samples": data_samples}


@dataclass
class XSamCollator:
    pad_index: int = DEFAULT_PAD_TOKEN_INDEX
    return_hf_format: bool = False
    use_varlen_attn: bool = False
    collate_fn: Callable = xsam_collate_fn

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        return self.collate_fn(instances, self.pad_index, self.return_hf_format, self.use_varlen_attn)
