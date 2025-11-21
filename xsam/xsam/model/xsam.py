import logging
import math
import os.path as osp
from collections import OrderedDict
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.dist import get_rank
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig
from transformers.file_utils import ModelOutput
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import get_parameter_dtype
from xtuner.model.modules import dispatch_modules
from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from xtuner.model.utils import (
    find_all_linear_names,
    get_peft_model_state_dict,
    guess_load_checkpoint,
    make_inputs_require_grad,
    traverse_dict,
)
from xtuner.registry import BUILDER
from xtuner.utils.device import get_device, get_torch_device

from ..model.modules import (
    ConnectorConfig,
    ConnectorModel,
    DynamicProjectorConfig,
    DynamicProjectorModel,
    SamplerConfig,
    SamplerModel,
)
from ..utils.constants import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PSTART_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
    DEFAULT_TASKS,
)
from ..utils.misc import data_sample_to_device
from .utils import prepare_inputs_labels_for_multimodal


@dataclass
class XSamOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None


class XSamModel(BaseModel):
    def __init__(
        self,
        llm=None,
        tokenizer=None,
        visual_encoder=None,
        postprocess_fn=None,
        segmentor=None,
        special_tokens=None,
        freeze_llm=False,
        freeze_visual_encoder=False,
        freeze_segmentor_encoder=False,
        freeze_segmentor_connector=False,
        visual_select_layer=-2,
        visual_select_indx=0,  # 1 for clip, 0 for siglip
        seg_select_layers=[8, 16, 24, 32],
        extract_seg_embeds=True,
        s1_pretrained_pth=None,
        s2_pretrained_pth=None,
        projector_depth=2,
        downsample_ratio=0.5,
        llm_lora=None,
        visual_encoder_lora=None,
        segmentor_lora=None,
        connector_type=None,
        connector_hidden_dim=256,
        connector_scale_factor=[4, 2, 1, 0.5],
        sampler_type="naive",
        sampler_input_feat="extra_pixel_values",
        cond_type: Literal["phrase", "cls", "all"] = "phrase",
        use_dual_encoder=False,
        use_vision_sampler=False,
        use_activation_checkpointing=True,
        max_position_embeddings=None,
        llm_loss_weight: float = 1.0,
        seg_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_segmentor_encoder = freeze_segmentor_encoder
        self.freeze_segmentor_connector = freeze_segmentor_connector

        assert (
            llm is not None or visual_encoder is not None or segmentor is not None
        ), "llm, visual_encoder, and segmentor cannot be all None"

        if isinstance(llm, dict):
            llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)
        self.llm = self._build_from_cfg_or_module(llm)
        self.tokenizer = self._build_from_cfg_or_module(tokenizer)
        self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)
        self.segmentor = self._build_from_cfg_or_module(segmentor)

        if self.llm is not None:
            self.llm.config.use_cache = False
            dispatch_modules(self.llm)

        self.postprocess_fn = postprocess_fn
        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        if self.visual_encoder is not None:
            self.projector_depth = projector_depth
            visual_projector_config = DynamicProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=self.projector_depth,
            )
            self.visual_projector = DynamicProjectorModel(visual_projector_config).to(self.visual_encoder.dtype)

        if self.segmentor is not None:
            if self.llm is not None and self.segmentor.decoder is not None:
                llm_projector_config = DynamicProjectorConfig(
                    visual_hidden_size=self.llm.config.hidden_size,
                    llm_hidden_size=self.segmentor.dec_config.hidden_size,
                    depth=self.projector_depth,
                )
                self.llm_projector = DynamicProjectorModel(llm_projector_config).to(self.llm.dtype)

            if use_dual_encoder and self.segmentor.encoder is not None:
                seg_projector_config = DynamicProjectorConfig(
                    visual_hidden_size=self.segmentor.enc_config.hidden_size,
                    llm_hidden_size=self.llm.config.hidden_size,
                    downsample_ratio=downsample_ratio,
                    depth=self.projector_depth,
                )
                self.seg_projector = DynamicProjectorModel(seg_projector_config).to(self.segmentor.dtype)

            if extract_seg_embeds and connector_type is not None and self.segmentor.pixel_decoder is not None:
                seg_select_layers = seg_select_layers[-self.segmentor.dec_config.num_feature_levels :]
                connector_config = ConnectorConfig(
                    segmentor_encoder_channels=[self.segmentor.enc_config.hidden_size]
                    * self.segmentor.dec_config.num_feature_levels,
                    hidden_channels=connector_hidden_dim,
                    scale_factor=connector_scale_factor[-self.segmentor.dec_config.num_feature_levels :],
                    connector_type=connector_type,
                )
                self.seg_connector = ConnectorModel(connector_config).to(self.segmentor.dtype)

            if use_vision_sampler and self.segmentor.decoder is not None:
                sampler_config = SamplerConfig(
                    sampler_type=sampler_type,
                    num_sample_point=256,
                    input_dim=self.llm.config.hidden_size,
                    output_dim=self.segmentor.dec_config.hidden_size,
                )
                self.vision_sampler = SamplerModel(sampler_config).to(self.segmentor.dtype)

            if self.segmentor.open_cls and self.segmentor.decoder is not None:
                self.bg_embeds = nn.Embedding(1, self.segmentor.dec_config.hidden_size).to(self.segmentor.dtype)

        if self.freeze_llm and self.llm is not None:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder and self.visual_encoder is not None:
            self.visual_encoder.requires_grad_(False)
        if self.freeze_segmentor_encoder and self.segmentor is not None:
            self.segmentor.encoder.requires_grad_(False)
        if self.freeze_segmentor_connector and self.segmentor is not None:
            self.seg_connector.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if self.llm is not None:
                if hasattr(self.llm, "enable_input_require_grads"):
                    self.llm.enable_input_require_grads()
                else:
                    self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            if self.visual_encoder is not None:
                if hasattr(self.visual_encoder, "enable_input_require_grads"):
                    self.visual_encoder.enable_input_require_grads()
                else:
                    self.visual_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                self.visual_projector.enable_input_require_grads()

            if self.segmentor is not None:
                if hasattr(self.segmentor, "enable_input_require_grads"):
                    self.segmentor.enable_input_require_grads()
                else:
                    self.segmentor.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                if hasattr(self, "seg_projector"):
                    self.seg_projector.enable_input_require_grads()
                if hasattr(self, "llm_projector"):
                    self.llm_projector.enable_input_require_grads()
                if hasattr(self, "seg_connector"):
                    self.seg_connector.enable_input_require_grads()
            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.use_segmentor_encoder_lora = segmentor_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora, use_activation_checkpointing)
        if self.use_segmentor_encoder_lora:
            self._prepare_segmentor_for_lora(segmentor_lora, use_activation_checkpointing)

        state_dict = super().state_dict()
        if s1_pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(s1_pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)

            matched_keys = [k for k in pretrained_state_dict.keys() if k in state_dict.keys()]
            mismatched_keys = [k for k in pretrained_state_dict.keys() if k not in state_dict.keys()]
            missed_keys = [k for k in state_dict.keys() if k not in pretrained_state_dict.keys()]
            print_log(f"Load s1_pretrained_pth from {s1_pretrained_pth}", logger="current")
            print_log(f"Matched keys: {len(matched_keys)} / {len(pretrained_state_dict.keys())}", logger="current")
            if len(mismatched_keys) > 0:
                print_log(f"Mismatched keys: {mismatched_keys}", logger="current", level=logging.WARNING)
            if len(missed_keys) > 0:
                print_log(f"Missed keys: {missed_keys}", logger="current", level=logging.WARNING)

        if s2_pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(s2_pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)

            matched_keys = [k for k in pretrained_state_dict.keys() if k in state_dict.keys()]
            mismatched_keys = [k for k in pretrained_state_dict.keys() if k not in state_dict.keys()]
            missed_keys = [k for k in state_dict.keys() if k not in pretrained_state_dict.keys()]
            print_log(f"Load s2_pretrained_pth from {s2_pretrained_pth}", logger="current")
            print_log(f"Matched keys: {len(matched_keys)} / {len(pretrained_state_dict.keys())}", logger="current")
            if len(mismatched_keys) > 0:
                print_log(f"Mismatched keys: {mismatched_keys}", logger="current", level=logging.WARNING)
            if len(missed_keys) > 0:
                print_log(f"Missed keys: {missed_keys}", logger="current", level=logging.WARNING)

        self.visual_select_layer = visual_select_layer
        self.visual_select_indx = visual_select_indx
        self.seg_select_layers = seg_select_layers
        self.extract_seg_embeds = extract_seg_embeds
        self.sampler_input_feat = sampler_input_feat
        self.cond_type = cond_type
        self.llm_loss_weight = llm_loss_weight
        self.seg_loss_weight = seg_loss_weight

    @property
    def device(self):
        return get_device()

    @property
    def dtype(self):
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def _add_special_tokens(self, special_tokens):
        assert all(token in DEFAULT_SPECIAL_TOKENS for token in special_tokens)
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.seg_token_idx = -1
        self.cls_token_idx = -1
        self.pstart_token_idx = -1
        self.pend_token_idx = -1

        if DEFAULT_SEG_TOKEN in special_tokens:
            self.seg_token_idx = self.tokenizer(DEFAULT_SEG_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_CLS_TOKEN in special_tokens:
            self.cls_token_idx = self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_PSTART_TOKEN in special_tokens:
            self.pstart_token_idx = self.tokenizer(DEFAULT_PSTART_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_PEND_TOKEN in special_tokens:
            self.pend_token_idx = self.tokenizer(DEFAULT_PEND_TOKEN, add_special_tokens=False)["input_ids"][0]

    def _get_index_embeds(self, input_embeds, embed_ids):
        output_embeds = []
        for input_embed, embed_id in zip(input_embeds, embed_ids):
            unique_ids = torch.unique(embed_id[embed_id != -1])
            if len(unique_ids) == 0:
                continue

            embeds = torch.stack([input_embed[embed_id == idx].mean(dim=0) for idx in unique_ids])
            output_embeds.append(embeds)

        return output_embeds if len(output_embeds) > 0 else None

    def _process_embeds(self, cond_embeds, seg_embeds, task_name="genseg"):
        B = len(cond_embeds)
        embed_masks = None
        local_cond_lens = None
        global_cond_lens = None
        bg_embeds = self.bg_embeds.weight
        if task_name in ["genseg", "vgdseg", "gcgseg", "ovseg", "intseg"]:
            max_cond_len = max([x.shape[0] for x in cond_embeds])
            embed_masks = []
            for i, cond_embed in enumerate(cond_embeds):
                cond_embeds[i] = torch.cat(
                    [cond_embed, bg_embeds.clone().repeat(max_cond_len - cond_embed.shape[0], 1) + -1e9],
                    dim=0,
                )
                embed_masks.append(
                    torch.cat(
                        [
                            torch.ones(cond_embed.shape[0], device=cond_embed.device),
                            torch.zeros(max_cond_len - cond_embed.shape[0], device=cond_embed.device),
                        ]
                    )
                )
            bg_embeds = bg_embeds[None, ...].repeat(B, 1, 1)
            cond_embeds = torch.cat([torch.stack(cond_embeds), bg_embeds], dim=1)
            seg_embeds = torch.stack(seg_embeds) if seg_embeds is not None else None
            embed_masks = torch.cat([torch.stack(embed_masks), torch.ones((B, 1), device=cond_embeds.device)], dim=1)
        elif task_name in ["refseg", "reaseg"]:
            local_cond_lens = [x.shape[0] for x in cond_embeds]
            cond_embeds = torch.cat([torch.cat(cond_embeds), bg_embeds])
            cond_embeds = cond_embeds[None, ...].repeat(sum(local_cond_lens), 1, 1)
            seg_embeds = torch.cat(seg_embeds).unsqueeze(1) if seg_embeds is not None else None
        else:
            raise ValueError(f"Task name {task_name} is not supported in _process_embeds")

        return cond_embeds, seg_embeds, embed_masks, local_cond_lens, global_cond_lens

    def _get_vgd_labels(self, data_samples):
        def _get_attr_from_data_samples(data_samples, attr):
            return getattr(data_samples, attr, None) if data_samples is not None else None

        class_labels = _get_attr_from_data_samples(data_samples, "class_labels")
        sampled_labels = _get_attr_from_data_samples(data_samples, "sampled_labels")
        contiguous_labels = _get_attr_from_data_samples(data_samples, "contiguous_labels")

        if class_labels is not None:
            class_labels = [class_label.cpu().numpy().tolist() for class_label in class_labels]

        if contiguous_labels is not None:
            # convert labels to contiguous labels
            assert class_labels is not None and sampled_labels is not None
            class_labels = [
                [ordered_label.index(sampled_label[label]) for label in class_label]
                for ordered_label, sampled_label, class_label in zip(contiguous_labels, sampled_labels, class_labels)
            ]
            sampled_labels = [
                [ordered_label.index(label) for label in sampled_label]
                for ordered_label, sampled_label in zip(contiguous_labels, sampled_labels)
            ]
        return class_labels, sampled_labels

    def _get_vprompt_feats_and_masks(
        self, vprompt_feats, vprompt_masks, class_labels, contiguous_labels, sampled_labels
    ):
        sampled_feats = []
        sampled_masks = []
        new_sampled_labels = []

        # Process each batch
        for batch_idx, (
            batch_vprompt_feats,
            batch_vprompt_masks,
            batch_class_labels,
            batch_contiguous_labels,
        ) in enumerate(zip(vprompt_feats, vprompt_masks, class_labels, contiguous_labels)):
            batch_sampled_feats = torch.zeros(
                (len(batch_contiguous_labels), batch_vprompt_feats.shape[1]),
                dtype=batch_vprompt_feats.dtype,
                device=batch_vprompt_feats.device,
            )
            batch_sampled_masks = torch.zeros(
                (len(batch_contiguous_labels), batch_vprompt_masks.shape[1], batch_vprompt_masks.shape[2]),
                dtype=batch_vprompt_masks.dtype,
                device=batch_vprompt_masks.device,
            )
            new_batch_sampled_labels = []

            # Track used labels to avoid duplicate sampling
            used_labels = []
            used_poses = []

            for i, target_label in enumerate(batch_contiguous_labels):
                # Find matching positions across all batches
                pos_matches = [
                    (b_idx, pos)
                    for b_idx, batch_labels in enumerate(class_labels)
                    for pos, label in enumerate(batch_labels)
                    if label == target_label and (b_idx, pos) not in used_poses
                ]
                neg_matches = [
                    (b_idx, pos)
                    for b_idx, batch_labels in enumerate(class_labels)
                    for pos, label in enumerate(batch_labels)
                    if label not in used_labels and (b_idx, pos) not in used_poses and label not in batch_class_labels
                ]

                matches = pos_matches if pos_matches else neg_matches

                if matches:
                    selected_batch, selected_pos = matches[torch.randint(len(matches), (1,)).item()]
                    batch_sampled_feats[i] = vprompt_feats[selected_batch][selected_pos]
                    batch_sampled_masks[i] = vprompt_masks[selected_batch][selected_pos]
                    new_batch_sampled_labels.append(
                        sampled_labels[selected_batch][
                            contiguous_labels[selected_batch].index(class_labels[selected_batch][selected_pos])
                        ]
                    )
                    used_labels.append(class_labels[selected_batch][selected_pos])
                    used_poses.append((selected_batch, selected_pos))
                else:
                    # If no matches found, use default embedding
                    batch_sampled_feats[i] = torch.zeros_like(batch_vprompt_feats[0])
                    batch_sampled_masks[i] = torch.zeros_like(batch_vprompt_masks[0])
                    new_batch_sampled_labels.append(-1)

            sampled_feats.append(batch_sampled_feats)
            sampled_masks.append(batch_sampled_masks)
            new_sampled_labels.append(new_batch_sampled_labels)

        return sampled_feats, sampled_masks, new_sampled_labels

    def _get_attrs_from_data_samples(self, data_samples, attrs, **kwargs):
        if isinstance(attrs, str):
            attrs = [attrs]
        return [getattr(data_samples, attr, None) if data_samples is not None else None for attr in attrs]

    def forward(self, data_dict, data_samples=None, mode="loss", **kwargs):
        if data_samples is not None:
            data_samples = data_sample_to_device(data_samples, device=get_device())

        extra_data_dict = {}
        if "pixel_values" in data_dict and self.visual_encoder is not None:
            visual_outputs = self.visual_encoder(
                data_dict["pixel_values"].to(self.visual_encoder.dtype),
                output_hidden_states=True,
            )
            pixel_values = self.visual_projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, self.visual_select_indx :]
            )
            data_dict["pixel_values"] = pixel_values.to(self.llm.dtype)

        if "extra_pixel_values" in data_dict and self.segmentor is not None:
            if self.extract_seg_embeds:
                seg_visual_outputs = self.segmentor.encoder(
                    data_dict["extra_pixel_values"].to(self.segmentor.dtype),
                    output_hidden_states=True,
                    output_attentions=False,
                )
                seg_image_embeddings = (
                    seg_visual_outputs.last_hidden_state
                    if hasattr(seg_visual_outputs, "last_hidden_state")
                    else seg_visual_outputs.hidden_states[-1].transpose(1, 2)
                )
                extra_pixel_values = None
                if hasattr(self, "seg_projector"):
                    extra_pixel_values = self.seg_projector(seg_visual_outputs.hidden_states[self.visual_select_layer])
                    extra_pixel_values = extra_pixel_values.to(self.llm.dtype)

                if hasattr(self, "seg_connector"):
                    seg_image_embeddings = self.seg_connector(
                        [seg_visual_outputs.hidden_states[i] for i in self.seg_select_layers]
                    )
                elif self.segmentor.pixel_decoder is not None and hasattr(seg_visual_outputs, "feature_maps"):
                    seg_image_embeddings = seg_visual_outputs.feature_maps

                # here, extra_pixel_values is seg_projector output
                data_dict["extra_pixel_values"] = extra_pixel_values
                extra_data_dict = {
                    "extra_pixel_values": None,
                    "seg_image_embeddings": seg_image_embeddings,
                }
                del seg_visual_outputs
            else:
                # here, extra_pixel_values is image_processor output
                extra_data_dict = {
                    "extra_pixel_values": data_dict["extra_pixel_values"].to(self.segmentor.dtype),
                    "seg_image_embeddings": None,
                }
                data_dict["extra_pixel_values"] = None
        else:
            data_dict["extra_pixel_values"] = None

        if data_dict.get("vprompt_masks", None) is not None and hasattr(self, "vision_sampler"):
            vprompt_masks = data_dict.pop("vprompt_masks")
            class_labels, contiguous_labels = self._get_vgd_labels(data_samples)
            sampled_labels = self._get_attrs_from_data_samples(data_samples, ["sampled_labels"])[0]
            sampled_feats = self.vision_sampler(data_dict[self.sampler_input_feat], vprompt_masks)
            assert all(
                sampled_feat is not None for sampled_feat in sampled_feats
            ), f"{data_dict[self.sampler_input_feat]}, {vprompt_masks}"
            vprompt_feats, vprompt_masks, new_sampled_labels = self._get_vprompt_feats_and_masks(
                sampled_feats, vprompt_masks, class_labels, contiguous_labels, sampled_labels
            )
            data_dict["vprompt_feats"] = vprompt_feats
            kwargs["vprompt_masks"] = vprompt_masks
            kwargs["sampled_labels"] = sampled_labels

        if self.llm is not None:
            data_dict = prepare_inputs_labels_for_multimodal(llm=self.llm, **data_dict)

        data_dict.update(extra_data_dict)

        if mode == "loss":
            return self.compute_loss(data_dict, data_samples, **kwargs)
        elif mode == "predict":
            return self.predict(data_dict, data_samples, **kwargs)
        elif mode == "tensor":
            return self._forward(data_dict, data_samples, **kwargs)
        else:
            raise NotImplementedError

    def _forward(
        self,
        data_dict,
        data_samples=None,
        **kwargs,
    ):
        if data_dict.get("inputs_embeds", None) is not None:
            data_dict["input_ids"] = None

        cond_ids = data_dict.pop("cond_ids", None)
        seg_ids = data_dict.pop("seg_ids", None)
        extra_pixel_values = data_dict.pop("extra_pixel_values", None)
        seg_image_embeddings = data_dict.pop("seg_image_embeddings", None)
        task_names, image_size, scaled_size, mask_labels, class_labels = self._get_attrs_from_data_samples(
            data_samples,
            [
                "task_names",
                "image_sizes",
                "scaled_sizes",
                "mask_labels",
                "class_labels",
            ],
            **kwargs,
        )
        task_names = task_names if task_names is not None else ["genseg"]
        assert (
            len(set(task_names)) == 1 and task_names[0] in DEFAULT_TASKS
        ), f"Task name {task_names} is not in {DEFAULT_TASKS}"

        seg_embeds = None
        cond_embeds = None
        embed_masks = None
        llm_outputs = None
        seg_outputs = None
        local_cond_lens = None
        global_cond_lens = None

        if self.llm is not None:
            llm_outputs = self.llm(**data_dict, output_hidden_states=True)

        if self.segmentor is None or self.segmentor.decoder is None:
            return llm_outputs, None

        if llm_outputs is not None:
            llm_hidden_states = llm_outputs.hidden_states
            llm_last_hidden_state = llm_hidden_states[-1]
            llm_embeds = self.llm_projector(llm_last_hidden_state)
            if cond_ids is not None:
                cond_embeds = self._get_index_embeds(llm_embeds, cond_ids)
            if seg_ids is not None:
                seg_embeds = self._get_index_embeds(llm_embeds, seg_ids)
            if cond_embeds is not None and seg_embeds is not None:
                cond_embeds, seg_embeds, embed_masks, local_cond_lens, global_cond_lens = self._process_embeds(
                    cond_embeds, seg_embeds, task_names[0]
                )

        if (local_cond_lens or global_cond_lens) is not None and mask_labels is not None:
            cur_rank = get_rank()
            mask_labels = list(chain(*[mask_label.split(1) for mask_label in mask_labels]))
            if global_cond_lens is not None:
                label_offsets = (
                    list(accumulate([sum(torch.cat(global_cond_lens[:cur_rank])).item()] + local_cond_lens[:-1]))
                    if cur_rank > 0
                    else list(accumulate([0] + local_cond_lens[:-1]))
                )
            else:
                label_offsets = list(accumulate([0] + local_cond_lens[:-1]))

            class_labels = list(
                chain(
                    *[
                        (class_label + label_offset).split(1)
                        for label_offset, class_label in zip(label_offsets, class_labels)
                    ]
                )
            )

        if seg_embeds is not None or llm_outputs is None:
            if seg_embeds is not None and seg_embeds.shape[1] != 1:
                seg_outputs = None
            else:
                seg_outputs = self.segmentor(
                    pixel_values=extra_pixel_values,
                    image_embeddings=seg_image_embeddings,
                    cond_embeddings=cond_embeds,
                    seg_embeddings=seg_embeds,
                    embed_masks=embed_masks,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                    cond_lens=local_cond_lens,
                    return_dict=True,
                )
                if kwargs.pop("do_postprocess", False):
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        **kwargs,
                    )

        return llm_outputs, seg_outputs

    @torch.no_grad()
    def predict(self, data_dict, data_samples=None, **kwargs):
        if data_dict.get("inputs_embeds", None) is not None:
            data_dict["input_ids"] = None

        if data_dict.get("labels", None) is not None:
            data_dict["labels"] = None

        if data_dict.get("position_ids", None) is not None:
            data_dict["position_ids"] = None

        if data_dict.get("attention_mask", None) is not None:
            data_dict["attention_mask"] = None

        seg_ids = data_dict.pop("seg_ids", None)
        extra_pixel_values = data_dict.pop("extra_pixel_values", None)
        seg_image_embeddings = data_dict.pop("seg_image_embeddings", None)
        input_cond_ids = data_dict.pop("cond_ids", None)
        task_names, image_size, scaled_size = self._get_attrs_from_data_samples(
            data_samples,
            ["task_names", "image_sizes", "scaled_sizes"],
            **kwargs,
        )
        task_names = task_names if task_names is not None else ["genseg"]
        assert (
            len(task_names) == 1 and task_names[0] in DEFAULT_TASKS
        ), f"Task name {task_names} is not in {DEFAULT_TASKS}"

        generation_config = kwargs.pop("generation_config", None)
        stopping_criteria = kwargs.pop("stopping_criteria", None)

        seg_embeds = None
        cond_embeds = None
        llm_outputs = None
        seg_outputs = None
        local_cond_lens = None

        if self.llm is not None:
            llm_outputs = self.llm.generate(
                **data_dict,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )

        if self.segmentor is None or self.segmentor.decoder is None:
            return llm_outputs, None

        if llm_outputs is not None:
            llm_output_ids = llm_outputs.sequences
            llm_hidden_states = llm_outputs.hidden_states
            input_hidden_states = llm_hidden_states[0][-1]
            llm_last_hidden_state = torch.cat([x[-1] for x in llm_hidden_states], dim=1)
            llm_input_embeds = self.llm_projector(input_hidden_states)
            llm_output_embeds = self.llm_projector(llm_last_hidden_state)

            L = input_hidden_states.shape[1]
            if input_cond_ids is not None:
                cond_embeds = self._get_index_embeds(llm_input_embeds, input_cond_ids)

            # update cond_embeds if there is pstart and pend token in the output
            pstart_idx = (llm_output_ids[..., :-1] == self.pstart_token_idx).nonzero()[:, 1]
            pend_idx = (llm_output_ids[..., :-1] == self.pend_token_idx).nonzero()[:, 1]
            cls_idx = (llm_output_ids[..., :-1] == self.cls_token_idx).nonzero()[:, 1]
            if len(pstart_idx) > 0 or len(cls_idx) > 0:
                output_cond_ids = torch.full(
                    llm_last_hidden_state.shape[:2], -1, dtype=torch.long, device=input_hidden_states.device
                )
                shift = llm_input_embeds.shape[1]
                if self.cond_type in ["phrase", "all"]:
                    for i, (pstart, pend) in enumerate(zip(pstart_idx, pend_idx)):
                        output_cond_ids[:, shift + pstart : shift + pend + 1] = i
                if self.cond_type in ["cls", "all"]:
                    for i, ci in enumerate(cls_idx):
                        output_cond_ids[:, shift + ci] = i

                cond_embeds = self._get_index_embeds(llm_output_embeds, output_cond_ids)

            # update seg_ids if there is seg token in the output
            seg_idx = (llm_output_ids[..., :-1] == self.seg_token_idx).nonzero()[:, 1]
            if len(seg_idx) > 0:
                # fmt: off
                B = (seg_image_embeddings.shape[0] if isinstance(seg_image_embeddings, torch.Tensor) 
                    else seg_image_embeddings[0].shape[0]) if self.extract_seg_embeds else extra_pixel_values.shape[0]
                assert B == 1, "Only support batch size 1 for prediction"
                # fmt: on
                seg_ids = torch.full_like(
                    llm_output_ids[..., :-1], -1, dtype=torch.long, device=input_hidden_states.device
                )
                for i, idx in enumerate(seg_idx):
                    seg_ids[:, idx] = i
                seg_ids = torch.cat(
                    [torch.full((B, L), -1, dtype=torch.long, device=input_hidden_states.device), seg_ids], dim=-1
                )
                seg_embeds = self._get_index_embeds(llm_output_embeds, seg_ids)

            if cond_embeds is not None and seg_embeds is not None:
                cond_embeds, seg_embeds, embed_masks, local_cond_lens, _ = self._process_embeds(
                    cond_embeds, seg_embeds, task_names[0]
                )

        if (cond_embeds is not None and seg_embeds is not None) or llm_outputs is None:
            if seg_embeds is not None and seg_embeds.shape[1] != 1:
                seg_outputs = None
            else:
                seg_outputs = self.segmentor(
                    pixel_values=extra_pixel_values,
                    image_embeddings=seg_image_embeddings,
                    cond_embeddings=cond_embeds,
                    seg_embeddings=seg_embeds,
                    embed_masks=embed_masks,
                    cond_lens=local_cond_lens,
                    return_dict=True,
                )
                if kwargs.pop("do_postprocess", True):
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        **kwargs,
                    )
        return llm_outputs, seg_outputs

    def compute_loss(self, data_dict, data_samples=None, **kwargs):
        llm_outputs, seg_outputs = self._forward(data_dict, data_samples, **kwargs)
        loss, loss_llm, loss_seg = 0.0, 0.0, 0.0
        if llm_outputs is not None and seg_outputs is None:
            loss_llm = llm_outputs.loss * self.llm_loss_weight
            loss = loss_llm
            loss_dict = {"loss": loss, "loss_llm": loss_llm}
        elif llm_outputs is None and seg_outputs is not None:
            loss_seg = seg_outputs.loss * self.seg_loss_weight
            loss_seg_dict = {k: v * self.seg_loss_weight for k, v in seg_outputs.loss_dict.items()}
            loss = loss_seg
            loss_dict = {"loss": loss, "loss_seg": loss_seg}
            loss_dict.update(loss_seg_dict)
        elif llm_outputs is not None and seg_outputs is not None:
            loss_llm = llm_outputs.loss * self.llm_loss_weight
            loss_seg = seg_outputs.loss * self.seg_loss_weight
            loss_seg_dict = {k: v * self.seg_loss_weight for k, v in seg_outputs.loss_dict.items()}
            loss = loss_llm + loss_seg
            loss_dict = {"loss": loss, "loss_llm": loss_llm, "loss_seg": loss_seg}
            loss_dict.update(loss_seg_dict)
        else:
            raise ValueError("llm_outputs and seg_outputs are both None")

        return loss_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.visual_encoder is not None:
            if self.use_visual_encoder_lora:
                to_return.update(get_peft_model_state_dict(self.visual_encoder, state_dict=state_dict))
            elif not self.freeze_visual_encoder:
                to_return.update({k: v for k, v in state_dict.items() if "visual_encoder." in k})
        # Step 2. segmentor
        if self.segmentor is not None:
            if self.use_segmentor_encoder_lora:
                to_return.update(get_peft_model_state_dict(self.segmentor.encoder, state_dict=state_dict))
            elif not self.freeze_segmentor_encoder:
                to_return.update({k: v for k, v in state_dict.items() if "segmentor.encoder" in k})

            # segmentor other parts except encoder
            to_return.update(
                {k: v for k, v in state_dict.items() if "segmentor" in k and "segmentor.encoder" not in k}
            )
        # Step 3. LLM
        if self.llm is not None:
            if self.use_llm_lora:
                to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
            elif not self.freeze_llm:
                to_return.update({k: v for k, v in state_dict.items() if "llm." in k})
        # Step 4. Projector
        to_return.update({k: v for k, v in state_dict.items() if "visual_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "seg_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "llm_projector." in k})
        # Step 5. seg_connector
        to_return.update({k: v for k, v in state_dict.items() if "seg_connector." in k})
        # Step 6. other embeds
        to_return.update({k: v for k, v in state_dict.items() if "bg_embeds." in k})
        to_return.update({k: v for k, v in state_dict.items() if "vgd_embeds." in k})
        # Step 7. vision_sampler
        to_return.update({k: v for k, v in state_dict.items() if "vision_sampler." in k})
        return to_return

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def _prepare_segmentor_for_lora(self, lora_config, use_activation_checkpointing=True):
        if self.segmentor is None:
            return
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.segmentor.encoder)
            lora_config.target_modules = modules
        self.segmentor = get_peft_model(self.segmentor.encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        if self.llm is not None:
            self.llm.gradient_checkpointing_enable()
        if self.visual_encoder is not None:
            self.visual_encoder.gradient_checkpointing_enable()
            self.visual_projector.gradient_checkpointing_enable()
        if self.segmentor is not None:
            self.segmentor.gradient_checkpointing_enable({"use_reentrant": False})
            if hasattr(self, "seg_projector"):
                self.seg_projector.gradient_checkpointing_enable()
            if hasattr(self, "llm_projector"):
                self.llm_projector.gradient_checkpointing_enable()
            if hasattr(self, "seg_connector"):
                self.seg_connector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        if self.llm is not None:
            self.llm.gradient_checkpointing_disable()
        if self.visual_encoder is not None:
            self.visual_encoder.gradient_checkpointing_disable()
            self.visual_projector.gradient_checkpointing_disable()
        if self.segmentor is not None:
            self.segmentor.gradient_checkpointing_disable()
            if hasattr(self, "seg_projector"):
                self.seg_projector.gradient_checkpointing_disable()
            if hasattr(self, "llm_projector"):
                self.llm_projector.gradient_checkpointing_disable()
            if hasattr(self, "seg_connector"):
                self.seg_connector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg, max_position_embeddings):
        orig_rope_scaling = getattr(llm_cfg, "rope_scaling", None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {"factor": 1}

        orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, "max_position_embeddings", None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {"type": "linear", "factor": scaling_factor}

        # hardcode for internlm2
        llm_cfg.attn_implementation = "flash_attention_2"
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = (
            "LlamaConfig",
            "GemmaConfig",
            "MistralConfig",
            "MixtralConfig",
            "Qwen2Config",
            "Qwen2MoeConfig",
            "Qwen3Config",
            "Qwen3MoEConfig",
            "Starcoder2Config",
            "Starcoder2Config",
            "Phi3Config",
        )
        SUPPORT_FLASH_ATTN2 = (
            "InternLM2Config",
            "LlamaConfig",
            "GemmaConfig",
            "MistralConfig",
            "MixtralConfig",
            "Qwen2Config",
            "Qwen2MoeConfig",
            "Qwen3Config",
            "Qwen3MoEConfig",
            "Starcoder2Config",
            "Starcoder2Config",
            "Phi3Config",
        )

        torch_dtype = (
            torch.bfloat16
            if (get_torch_device().is_available() and get_torch_device().is_bf16_supported())
            else torch.float16
        )

        if getattr(cfg, "attn_implementation", None) is not None:
            # Flash Attention 2.0 only supports torch.float16 and
            # torch.bfloat16 dtypes
            if cfg.attn_implementation == "flash_attention_2":
                cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = "flash_attention_2"
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = "sdpa"

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(cfg, "quantization_config")):
            return cfg

        torch_dtype = (
            torch.bfloat16
            if (get_torch_device().is_available() and get_torch_device().is_bf16_supported())
            else torch.float16
        )

        cfg.torch_dtype = torch_dtype
        quantization_config = cfg.quantization_config
        quantization_config.bnb_4bit_compute_dtype = torch_dtype
        quantization_config.bnb_4bit_quant_storage = torch_dtype

        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if cfg_or_mod is None:
            return None

        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    def to_hf(
        self,
        cfg,
        save_dir,
        fp32=False,
        save_pretrained_kwargs={},
        save_format="xtuner",
        **kwargs,
    ):
        assert save_format == "xtuner", "Only support xtuner format for now"
        self.to_xtuner(cfg, save_dir, fp32, save_pretrained_kwargs)

    def to_xtuner(self, cfg, save_dir, fp32=False, save_pretrained_kwargs={}):
        # Only save the model weights of: LLM, Visual Encoder, Segment Encoder, Visual Projector, Segmentor Projector
        # LLM
        if self.llm is not None:
            self.llm.config.use_cache = True
            if not fp32:
                print_log("Convert LLM to float16", "current")
                self.llm.half()
            if self.use_llm_lora:
                llm_path = osp.join(save_dir, "llm_adapter")
                print_log(f"Saving LLM adapter to {llm_path}", "current")
                self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
            elif not self.freeze_llm:
                llm_path = osp.join(save_dir, "llm")
                print_log(f"Saving LLM tokenizer to {llm_path}", "current")
                tokenizer = BUILDER.build(cfg.tokenizer)
                tokenizer.save_pretrained(llm_path, **save_pretrained_kwargs)
                print_log(f"Saving LLM to {llm_path}", "current")
                self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
            self.llm.config.use_cache = False

        # Visual Encoder
        if self.visual_encoder is not None:
            if self.use_visual_encoder_lora:
                visual_encoder_path = osp.join(save_dir, "visual_encoder_adapter")
                print_log(f"Saving visual_encoder adapter to {visual_encoder_path}", "current")
                self.visual_encoder.save_pretrained(visual_encoder_path, **save_pretrained_kwargs)
            elif not self.freeze_visual_encoder:
                visual_encoder_path = osp.join(save_dir, "visual_encoder")
                print_log(
                    "Saving visual_encoder image_processor to" f"{visual_encoder_path}",
                    "current",
                )
                image_processor = BUILDER.build(cfg.image_processor)
                image_processor.save_pretrained(visual_encoder_path, **save_pretrained_kwargs)
                print_log(f"Saving visual_encoder to {visual_encoder_path}", "current")
                self.visual_encoder.save_pretrained(visual_encoder_path, **save_pretrained_kwargs)

            # Visual Projector
            visual_projector_path = osp.join(save_dir, "visual_projector")
            print_log(f"Saving visual_projector to {visual_projector_path}", "current")
            self.visual_projector.save_pretrained(visual_projector_path, **save_pretrained_kwargs)

        # Segmentor Encoder
        if self.segmentor is not None:
            # TODO: add segmentor_encoder_adapter
            if self.use_segmentor_encoder_lora:
                segmentor_encoder_path = osp.join(save_dir, "segmentor_encoder_adapter")
                print_log(f"Saving segmentor_encoder adapter to {segmentor_encoder_path}", "current")
                self.segmentor.encoder.save_pretrained(segmentor_encoder_path, **save_pretrained_kwargs)
            elif not self.freeze_segmentor_encoder:
                segmentor_encoder_path = osp.join(save_dir, "segmentor_encoder")
                print_log(f"Saving segmentor image_processor to {segmentor_encoder_path}", "current")
                extra_image_processor = BUILDER.build(cfg.extra_image_processor)
                extra_image_processor.save_pretrained(segmentor_encoder_path, **save_pretrained_kwargs)
                print_log(f"Saving segmentor_encoder to {segmentor_encoder_path}", "current")
                state_dict = {
                    k.replace("segmentor.encoder.", "vision_encoder."): v
                    for k, v in self.state_dict().items()
                    if "segmentor.encoder" in k
                }
                self.segmentor.save_pretrained(segmentor_encoder_path, state_dict=state_dict, **save_pretrained_kwargs)

            # Segmentor Projector
            if hasattr(self, "seg_projector"):
                seg_projector_path = osp.join(save_dir, "segmentor_projector")
                print_log(f"Saving segmentor_projector to {seg_projector_path}", "current")
                self.seg_projector.save_pretrained(seg_projector_path, **save_pretrained_kwargs)
            if hasattr(self, "seg_projector"):
                seg_projector_path = osp.join(save_dir, "segmentor_projector")
                print_log(f"Saving segmentor_projector to {seg_projector_path}", "current")
                self.seg_projector.save_pretrained(seg_projector_path, **save_pretrained_kwargs)
