import logging
import math
import os.path as osp
from collections import OrderedDict
from dataclasses import dataclass
from itertools import accumulate
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig
from transformers.file_utils import ModelOutput
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import get_parameter_dtype

from xsam.registry import BUILDER
from xsam.utils.checkpoint import guess_load_checkpoint
from xsam.utils.device import get_device, get_torch_device
from xsam.utils.logging import print_log
from xsam.utils.tensor import pad_tensors

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
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2, dispatch_modules
from .utils import (
    find_all_linear_names,
    get_peft_model_state_dict,
    make_inputs_require_grad,
    prepare_inputs_labels_for_mlm,
    traverse_dict,
)


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
        vlm=None,
        tokenizer=None,
        vision_encoder=None,
        postprocess_fn=None,
        temporal_process_fn=lambda x: x,
        extra_temporal_process_fn=lambda x: x,
        segmentor=None,
        special_tokens=None,
        freeze_llm=False,
        freeze_vlm=True,
        freeze_vision_encoder=False,
        freeze_mask_encoder=False,
        freeze_segmentor_connector=False,
        visual_select_layer=-2,
        visual_select_indx=0,  # 1 for clip, 0 for siglip
        extra_select_layers=[8, 16, 24, 32],
        extra_select_channels=None,
        extract_seg_embeds=True,
        s1_pretrained_pth=None,
        s2_pretrained_pth=None,
        projector_depth=2,
        downsample_ratio=0.5,
        llm_lora=None,
        vlm_lora=None,
        vision_encoder_lora=None,
        segmentor_lora=None,
        connector_type=None,
        connector_hidden_dim=256,
        connector_scale_factor=[4, 2, 1, 0.5],
        sampler_type="naive",
        sampler_input_feat="extra_pixel_values",
        sampler_pooling_mode="mean",
        sampler_pooling_kernel_size=None,
        sampler_pooling_output_size=None,
        cond_type: Literal["phrase", "cls", "all"] = "phrase",
        use_flatten_cond=False,
        use_dual_encoder=False,
        use_vision_sampler=False,
        use_pad_embeds=False,
        use_activation_checkpointing=True,
        max_position_embeddings=None,
        llm_loss_weight: float = 1.0,
        seg_loss_weight: float = 1.0,
        ignore_label: int = -100,
        background_label: int = -1,
        ptoken_shift: int = 0,
    ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_vlm = freeze_vlm
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_mask_encoder = freeze_mask_encoder
        self.freeze_segmentor_connector = freeze_segmentor_connector
        self.projector_depth = projector_depth

        assert (
            llm is not None or vlm is not None or vision_encoder is not None or segmentor is not None
        ), "llm, vlm, vision_encoder, and segmentor cannot be all None"

        if isinstance(llm, dict):
            llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)
        self.llm = self._build_from_cfg_or_module(llm)
        self.tokenizer = self._build_from_cfg_or_module(tokenizer)
        self.vision_encoder = self._build_from_cfg_or_module(vision_encoder)
        self.segmentor = self._build_from_cfg_or_module(segmentor)
        self.vlm = self._build_from_cfg_or_module(vlm)

        if self.llm is not None:
            self.llm.config.use_cache = False
            dispatch_modules(self.llm)

        if self.vlm is not None:
            self.vlm.config.text_config.use_cache = False
            dispatch_modules(self.vlm)

        self.postprocess_fn = postprocess_fn
        if isinstance(temporal_process_fn, dict):
            temporal_process_fn = BUILDER.build(temporal_process_fn)
        if isinstance(extra_temporal_process_fn, dict):
            extra_temporal_process_fn = BUILDER.build(extra_temporal_process_fn)
        self.temporal_process_fn = temporal_process_fn
        self.extra_temporal_process_fn = extra_temporal_process_fn

        if self.vision_encoder is not None:
            visual_projector_config = DynamicProjectorConfig(
                visual_hidden_size=self.vision_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth,
            )
            self.visual_projector = DynamicProjectorModel(visual_projector_config).to(self.vision_encoder.dtype)

        if self.segmentor is not None:
            if self.llm is not None and self.segmentor.decoder is not None:
                llm_projector_config = DynamicProjectorConfig(
                    visual_hidden_size=self.llm.config.hidden_size,
                    llm_hidden_size=self.segmentor.dec_config.hidden_size,
                    depth=projector_depth,
                )
                self.llm_projector = DynamicProjectorModel(llm_projector_config).to(self.llm.dtype)

            if self.vlm is not None and self.segmentor.decoder is not None:
                llm_projector_config = DynamicProjectorConfig(
                    visual_hidden_size=self.vlm.config.text_config.hidden_size,
                    llm_hidden_size=self.segmentor.dec_config.hidden_size,
                    depth=projector_depth,
                )
                self.llm_projector = DynamicProjectorModel(llm_projector_config).to(self.vlm.dtype)

            if (use_dual_encoder and self.segmentor.encoder is not None) or (
                use_vision_sampler and sampler_input_feat == "extra_pixel_values"
            ):
                seg_projector_config = DynamicProjectorConfig(
                    visual_hidden_size=self.segmentor.enc_config.hidden_size,
                    llm_hidden_size=(
                        self.llm.config.hidden_size
                        if self.llm is not None
                        else self.vlm.config.text_config.hidden_size
                    ),
                    downsample_ratio=downsample_ratio,
                    depth=projector_depth,
                )
                self.seg_projector = DynamicProjectorModel(seg_projector_config).to(self.segmentor.dtype)

            if extract_seg_embeds and connector_type is not None and self.segmentor.pixel_decoder is not None:
                extra_select_layers = extra_select_layers[-self.segmentor.dec_config.num_feature_levels :]
                if extra_select_channels is None:
                    extra_select_channels = [self.segmentor.enc_config.hidden_size] * len(extra_select_layers)
                connector_config = ConnectorConfig(
                    segmention_encoder_channels=extra_select_channels[-self.segmentor.dec_config.num_feature_levels :],
                    hidden_channels=connector_hidden_dim,
                    scale_factor=connector_scale_factor[-self.segmentor.dec_config.num_feature_levels :],
                    connector_type=connector_type,
                )
                self.seg_connector = ConnectorModel(connector_config).to(self.segmentor.dtype)

            if use_vision_sampler and self.segmentor.decoder is not None:
                sampler_config = SamplerConfig(
                    sampler_type=sampler_type,
                    num_sample_point=256,
                    input_dim=(
                        self.llm.config.hidden_size
                        if self.llm is not None
                        else self.vlm.config.text_config.hidden_size
                    ),
                    output_dim=self.segmentor.dec_config.hidden_size,
                    pooling_mode=sampler_pooling_mode,
                    pooling_kernel_size=sampler_pooling_kernel_size,
                    pooling_output_size=sampler_pooling_output_size,
                )
                self.vision_sampler = SamplerModel(sampler_config).to(self.segmentor.dtype)

            if (
                self.segmentor.decoder is not None
                and self.segmentor.decoder.config.head_cls_type != "linear"
                and "ce_loss" in self.segmentor.decoder.config.loss_cls_type
            ):
                self.bg_embeds = nn.Embedding(1, self.segmentor.dec_config.hidden_size).to(self.segmentor.dtype)

            if use_pad_embeds:
                self.pad_embeds = nn.Embedding(1, self.segmentor.dec_config.hidden_size).to(self.segmentor.dtype)

        self.use_llm_lora = llm_lora is not None
        self.use_vlm_lora = vlm_lora is not None
        self.use_vision_encoder_lora = vision_encoder_lora is not None
        self.use_mask_encoder_lora = segmentor_lora is not None

        if self.freeze_llm and self.llm is not None and not self.use_llm_lora:
            self.llm.requires_grad_(False)
        if self.freeze_vlm and self.vlm is not None and not self.use_vlm_lora:
            self.vlm.requires_grad_(False)
            if not self.freeze_vision_encoder:
                self.vlm.visual.requires_grad_(True)
            if not self.freeze_llm:
                self.vlm.language_model.requires_grad_(True)
        if self.freeze_vision_encoder and self.vision_encoder is not None and not self.use_vision_encoder_lora:
            self.vision_encoder.requires_grad_(False)
        if self.freeze_mask_encoder and self.segmentor is not None and not self.use_mask_encoder_lora:
            self.segmentor.encoder.requires_grad_(False)
        if self.freeze_segmentor_connector and self.segmentor is not None and hasattr(self, "seg_connector"):
            self.seg_connector.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if self.llm is not None:
                if hasattr(self.llm, "enable_input_require_grads"):
                    self.llm.enable_input_require_grads()
                else:
                    self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            if self.vlm is not None:
                if hasattr(self.vlm, "enable_input_require_grads"):
                    self.vlm.enable_input_require_grads()
                else:
                    self.vlm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            if self.vision_encoder is not None:
                if hasattr(self.vision_encoder, "enable_input_require_grads"):
                    self.vision_encoder.enable_input_require_grads()
                else:
                    self.vision_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
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

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_vlm_lora:
            self._prepare_vlm_for_lora(vlm_lora)
        if self.use_vision_encoder_lora:
            self._prepare_vision_encoder_for_lora(vision_encoder_lora)
        if self.use_mask_encoder_lora:
            self._prepare_segmentor_for_lora(segmentor_lora)
        if use_activation_checkpointing:
            self.gradient_checkpointing_enable()

        # manually enable gradient for visual merger and deepstack_merger_list in vlm
        if self.vlm is not None:
            if self.use_vlm_lora:
                self.vlm.base_model.model.model.visual.merger.requires_grad_(True)
                self.vlm.base_model.model.model.visual.deepstack_merger_list.requires_grad_(True)
            else:
                self.vlm.model.visual.merger.requires_grad_(True)
                self.vlm.model.visual.deepstack_merger_list.requires_grad_(True)

        self.special_tokens = special_tokens
        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        state_dict = super().state_dict()
        if s1_pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(s1_pretrained_pth)
            matched_state_dict = {
                k: v
                for k, v in pretrained_state_dict.items()
                if k in state_dict.keys() and v.shape == state_dict[k].shape
            }
            self.load_state_dict(matched_state_dict, strict=False)

            matched_keys = [k for k in state_dict.keys() if k in matched_state_dict.keys()]
            mismatched_keys = [k for k in pretrained_state_dict.keys() if k not in matched_state_dict.keys()]
            missing_keys = [k for k in state_dict.keys() if k not in matched_state_dict.keys()]
            print_log(f"Loaded s1_pretrained_pth from {s1_pretrained_pth}", logger="current")
            print_log(
                f"Matched keys: {len(matched_keys)} / {len(pretrained_state_dict.keys())}",
                logger="current",
            )
            if len(mismatched_keys) > 0:
                print_log(
                    f"Mismatched keys: {len(mismatched_keys)} / {len(pretrained_state_dict.keys())}\n{mismatched_keys}",
                    logger="current",
                    level=logging.WARNING,
                )
            if len(missing_keys) > 0:
                print_log(
                    f"Missing keys: {len(missing_keys)} / {len(state_dict.keys())}\n{missing_keys}",
                    logger="current",
                    level=logging.WARNING,
                )

        if s2_pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(s2_pretrained_pth)
            matched_state_dict = {
                k: v
                for k, v in pretrained_state_dict.items()
                if k in state_dict.keys() and v.shape == state_dict[k].shape
            }
            self.load_state_dict(matched_state_dict, strict=False)

            matched_keys = [k for k in state_dict.keys() if k in matched_state_dict.keys()]
            mismatched_keys = [k for k in pretrained_state_dict.keys() if k not in matched_state_dict.keys()]
            missing_keys = [k for k in state_dict.keys() if k not in matched_state_dict.keys()]
            print_log(f"Loaded s2_pretrained_pth from {s2_pretrained_pth}", logger="current")
            print_log(
                f"Matched keys: {len(matched_keys)} / {len(pretrained_state_dict.keys())}\n{matched_keys}",
                logger="current",
            )
            if len(mismatched_keys) > 0:
                print_log(
                    f"Mismatched keys: {len(mismatched_keys)} / {len(pretrained_state_dict.keys())}\n{mismatched_keys}",
                    logger="current",
                    level=logging.WARNING,
                )
            if len(missing_keys) > 0:
                print_log(
                    f"Missing keys: {len(missing_keys)} / {len(state_dict.keys())}\n{missing_keys}",
                    logger="current",
                    level=logging.WARNING,
                )

        self.cond_type = cond_type
        self.use_flatten_cond = use_flatten_cond
        self.use_dual_encoder = use_dual_encoder
        self.visual_select_layer = visual_select_layer
        self.visual_select_indx = visual_select_indx
        self.extra_select_layers = extra_select_layers
        self.extra_select_channels = extra_select_channels
        self.extract_seg_embeds = extract_seg_embeds
        self.sampler_input_feat = sampler_input_feat
        self.llm_loss_weight = llm_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.ignore_label = ignore_label
        self.background_label = background_label
        self.ptoken_shift = ptoken_shift
        assert self.ptoken_shift in [0, 1], f"ptoken_shift must be 0 or 1, but got {self.ptoken_shift}"

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
        if num_new_tokens > 0 and self.llm is not None:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.llm.get_output_embeddings().requires_grad_(True)
            self.llm.get_input_embeddings().requires_grad_(True)

        if num_new_tokens > 0 and self.vlm is not None:
            self.vlm.resize_token_embeddings(len(self.tokenizer))
            if self.use_vlm_lora:
                self.vlm.model.lm_head.requires_grad_(True)
                self.vlm.model.language_model.embed_tokens.requires_grad_(True)
            else:
                self.vlm.lm_head.requires_grad_(True)
                self.vlm.language_model.embed_tokens.requires_grad_(True)

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
        cond_embeds = []
        token_embeds = []
        cond_ids = []
        for input_embed, embed_id in zip(input_embeds, embed_ids):
            ids = embed_id[embed_id != -1]
            unique_ids = torch.unique(ids)
            if len(unique_ids) == 0:
                continue

            embeds = [input_embed[embed_id == unique_id] for unique_id in unique_ids]
            cond_embeds.append(torch.stack([embed.mean(dim=0) for embed in embeds]))
            token_embeds.append(torch.cat(embeds))
            cond_ids.append(ids)

        return (
            (cond_embeds, token_embeds, None) if not self.use_flatten_cond else (token_embeds, token_embeds, cond_ids)
        )

    def _process_embeds(self, cond_embeds, token_embeds, seg_embeds, cond_ids, task_name="img_genseg"):
        B = len(cond_embeds)
        embed_masks = None
        cond_lens = None
        bg_embeds = self.bg_embeds.weight if hasattr(self, "bg_embeds") else None
        pad_embeds = self.pad_embeds.weight if hasattr(self, "pad_embeds") else None
        if task_name in [
            "img_genseg",
            "img_vgdseg",
            "img_gcgseg",
            "img_ovseg",
            "img_intseg",
            "vid_genseg",
            "vid_vgdseg",
            "vid_objseg",
            "vid_gcgseg",
            "vid_ovseg",
            "vid_intseg",
        ]:
            embed_masks = []
            token_masks = []
            max_cond_len = max([x.shape[0] for x in cond_embeds])
            max_token_len = max([x.shape[0] for x in token_embeds])
            for i, (cond_embed, token_embed) in enumerate(zip(cond_embeds, token_embeds)):
                embed_masks.append(
                    torch.cat(
                        [
                            torch.ones(cond_embed.shape[0], device=cond_embed.device),
                            torch.zeros(
                                max_cond_len - cond_embed.shape[0],
                                device=cond_embed.device,
                            ),
                        ]
                    )
                    if cond_embed.shape[0] < max_cond_len
                    else torch.ones(max_cond_len, device=cond_embed.device)
                )
                token_masks.append(
                    torch.cat(
                        [
                            torch.ones(token_embed.shape[0], device=token_embed.device),
                            torch.zeros(
                                max_token_len - token_embed.shape[0],
                                device=token_embed.device,
                            ),
                        ]
                    )
                    if token_embed.shape[0] < max_token_len
                    else torch.ones(max_token_len, device=token_embed.device)
                )
                cond_embeds[i] = (
                    torch.cat(
                        [
                            cond_embed,
                            (
                                pad_embeds.clone().repeat(max_cond_len - cond_embed.shape[0], 1)
                                if pad_embeds is not None
                                else torch.zeros(
                                    (max_cond_len - cond_embed.shape[0], cond_embed.shape[1]),
                                    device=cond_embed.device,
                                    dtype=cond_embed.dtype,
                                )
                            ),
                        ],
                        dim=0,
                    )
                    if cond_embed.shape[0] < max_cond_len
                    else cond_embed[:max_cond_len]
                )
                token_embeds[i] = (
                    torch.cat(
                        [
                            token_embed,
                            (
                                pad_embeds.clone().repeat(max_token_len - token_embed.shape[0], 1)
                                if pad_embeds is not None
                                else torch.zeros(
                                    (max_token_len - token_embed.shape[0], token_embed.shape[1]),
                                    device=token_embed.device,
                                    dtype=token_embed.dtype,
                                )
                            ),
                        ],
                        dim=0,
                    )
                    if token_embed.shape[0] < max_token_len
                    else token_embed[:max_token_len]
                )
                if cond_ids is not None:
                    cond_ids[i] = (
                        torch.cat(
                            [
                                cond_ids[i],
                                torch.full(
                                    (max_cond_len - cond_ids[i].shape[0],),
                                    fill_value=self.ignore_label,
                                    device=cond_ids[i].device,
                                ),
                            ],
                            dim=0,
                        )
                        if cond_ids[i].shape[0] < max_cond_len
                        else cond_ids[i][:max_cond_len]
                    )
            cond_embeds = torch.stack(cond_embeds)
            token_embeds = torch.stack(token_embeds)
            cond_ids = torch.stack(cond_ids) if cond_ids is not None else None
            embed_masks = torch.stack(embed_masks)
            token_masks = torch.stack(token_masks)
            seg_embeds = torch.stack(seg_embeds) if seg_embeds is not None else None
            if bg_embeds is not None:
                bg_embeds = bg_embeds[None, ...].repeat(B, 1, 1)
                cond_embeds = torch.cat([cond_embeds, bg_embeds], dim=1)
                cond_ids = (
                    torch.cat(
                        [
                            cond_ids,
                            torch.full_like(
                                bg_embeds[..., 0], fill_value=self.background_label, device=cond_ids.device
                            ),
                        ],
                        dim=1,
                    )
                    if cond_ids is not None
                    else None
                )
                embed_masks = torch.cat(
                    [
                        embed_masks,
                        torch.ones((B, 1), device=cond_embeds.device),
                    ],
                    dim=1,
                )
        elif task_name in ["img_refseg", "img_reaseg", "vid_refseg", "vid_reaseg"]:
            if cond_ids is not None:
                cond_lens = [len(x.unique()) for x in cond_ids]
                id_offsets = list(accumulate([0] + cond_lens[:-1]))
                cond_ids = torch.cat([cond_id + label_offset for cond_id, label_offset in zip(cond_ids, id_offsets)])
            else:
                cond_lens = [x.shape[0] for x in cond_embeds]

            cond_embeds = torch.cat(cond_embeds)
            token_embeds = torch.cat(token_embeds)
            embed_masks = torch.ones(cond_embeds.shape[0], device=cond_embeds.device)
            token_masks = torch.ones(token_embeds.shape[0], device=token_embeds.device)

            if bg_embeds is not None:
                cond_embeds = torch.cat([cond_embeds, bg_embeds])
                cond_ids = (
                    torch.cat(
                        [
                            cond_ids,
                            torch.full_like(
                                bg_embeds[..., 0], fill_value=self.background_label, device=cond_ids.device
                            ),
                        ]
                    )
                    if cond_ids is not None
                    else None
                )
                embed_masks = torch.cat(
                    [
                        embed_masks,
                        torch.ones_like(bg_embeds[..., 0], device=cond_embeds.device),
                    ]
                )

            cond_embeds = cond_embeds[None, ...].repeat(B, 1, 1)
            token_embeds = token_embeds[None, ...].repeat(B, 1, 1)
            token_masks = token_masks[None, ...].repeat(B, 1)
            cond_ids = cond_ids[None, ...].repeat(B, 1) if cond_ids is not None else None
            seg_embeds = torch.cat(seg_embeds).unsqueeze(1) if seg_embeds is not None else None
        else:
            raise ValueError(f"Task name {task_name} is not supported in _process_embeds")

        return cond_embeds, token_embeds, cond_ids, seg_embeds, embed_masks, token_masks, cond_lens

    def _get_vgd_labels(self, data_samples):
        def _get_attr_from_data_samples(data_samples, attr):
            return getattr(data_samples, attr, None) if data_samples is not None else None

        class_labels = _get_attr_from_data_samples(data_samples, "class_labels")
        sampled_labels = _get_attr_from_data_samples(data_samples, "sampled_labels")
        contiguous_labels = _get_attr_from_data_samples(data_samples, "contiguous_labels")
        vprompt_indices = _get_attr_from_data_samples(data_samples, "vprompt_indices")

        if class_labels is not None:
            if class_labels[0].ndim == 2:
                class_labels = [
                    class_label[vprompt_ind] for class_label, vprompt_ind in zip(class_labels, vprompt_indices)
                ]
            class_labels = [class_label.cpu().numpy().tolist() for class_label in class_labels]

        if contiguous_labels is not None:
            # convert labels to contiguous labels
            assert class_labels is not None and sampled_labels is not None
            class_labels = [
                [ordered_label.index(sampled_label[label]) for label in class_label if label != self.ignore_label]
                for ordered_label, sampled_label, class_label in zip(contiguous_labels, sampled_labels, class_labels)
            ]
            sampled_labels = [
                [ordered_label.index(label) for label in sampled_label]
                for ordered_label, sampled_label in zip(contiguous_labels, sampled_labels)
            ]
        return class_labels, sampled_labels, vprompt_indices

    def _get_vprompt_feats_and_masks(
        self,
        vprompt_feats,
        vprompt_masks,
        class_labels,
        contiguous_labels,
        sampled_labels,
    ):
        sampled_feats = []
        sampled_masks = []
        new_sampled_labels = []

        # Process each batch
        for batch_vprompt_feats, batch_vprompt_masks, batch_class_labels, batch_contiguous_labels in zip(
            vprompt_feats, vprompt_masks, class_labels, contiguous_labels
        ):
            batch_sampled_feats = torch.zeros(
                (len(batch_contiguous_labels), batch_vprompt_feats.shape[1], batch_vprompt_feats.shape[2]),
                dtype=batch_vprompt_feats.dtype,
                device=batch_vprompt_feats.device,
            )
            batch_sampled_masks = torch.zeros(
                (
                    len(batch_contiguous_labels),
                    batch_vprompt_masks.shape[1],
                    batch_vprompt_masks.shape[2],
                ),
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
                    new_batch_sampled_labels.append(self.ignore_label)

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

        extend_data_dict = {}
        image_grid_thw = None
        if "pixel_values" in data_dict and self.vision_encoder is not None:
            pixel_values = []
            padded_pixel_values, _ = pad_tensors(data_dict.pop("pixel_values"))
            padded_pixel_values = padded_pixel_values.to(self.vision_encoder.dtype)
            for _padded_pixel_values in padded_pixel_values:
                visual_outputs = self.vision_encoder(
                    _padded_pixel_values,
                    output_hidden_states=True,
                )
                _pixel_values = self.visual_projector(
                    visual_outputs.hidden_states[self.visual_select_layer][:, self.visual_select_indx :]
                )
                pixel_values.append(_pixel_values)
            # [B, P, D]*T -> [T, B, P, D]
            data_dict["pixel_values"] = torch.stack(pixel_values).to(self.llm.dtype)

        if "pixel_values" in data_dict and self.vlm is not None:
            pixel_values = data_dict["pixel_values"].to(self.vlm.dtype)
            image_grid_thw = data_dict.get("image_grid_thw", None)
            image_embeds, deepstack_image_embeds = self.vlm.get_image_features(pixel_values, image_grid_thw)
            data_dict["pixel_values"] = image_embeds
            extend_data_dict.update(
                {
                    "image_grid_thw": image_grid_thw,
                    "image_embeds": torch.cat(image_embeds, dim=0).to(self.vlm.device, self.vlm.dtype),
                    "deepstack_image_embeds": deepstack_image_embeds,
                }
            )

        if "extra_pixel_values" in data_dict and self.segmentor is not None:
            # extra_padded_pixel_values: [T, B, C, H, W], extra_padded_masks: [T, B]
            extra_padded_pixel_values, extra_padded_masks = pad_tensors(data_dict["extra_pixel_values"])
            extra_padded_pixel_values = extra_padded_pixel_values.to(self.segmentor.dtype)
            extra_pixel_values = []
            seg_image_embeds = []
            if self.extract_seg_embeds:
                for _extra_padded_pixel_values in extra_padded_pixel_values:
                    seg_visual_outputs = self.segmentor.encoder(
                        _extra_padded_pixel_values,
                        output_hidden_states=True,
                        output_attentions=False,
                    )
                    _seg_image_embeds = (
                        seg_visual_outputs.last_hidden_state
                        if hasattr(seg_visual_outputs, "last_hidden_state")
                        else seg_visual_outputs.hidden_states[-1].transpose(1, 2)
                    )
                    _extra_pixel_values = None
                    if hasattr(self, "seg_projector"):
                        _extra_pixel_values = self.seg_projector(
                            seg_visual_outputs.hidden_states[self.visual_select_layer]
                        )
                        _extra_pixel_values = _extra_pixel_values.to((self.llm or self.vlm).dtype)

                    if hasattr(self, "seg_connector"):
                        _seg_image_embeds = self.seg_connector(
                            [seg_visual_outputs.hidden_states[i] for i in self.extra_select_layers]
                        )
                    elif hasattr(seg_visual_outputs, "fpn_hidden_states"):
                        _seg_image_embeds = seg_visual_outputs.fpn_hidden_states
                    elif hasattr(seg_visual_outputs, "feature_maps"):
                        _seg_image_embeds = seg_visual_outputs.feature_maps

                    extra_pixel_values.append(_extra_pixel_values)
                    seg_image_embeds.append(_seg_image_embeds)

                # here, extra_pixel_values is seg_projector output
                # [B, P, D]*T -> [T, B, P, D]
                data_dict["extra_pixel_values"] = (
                    torch.stack(extra_pixel_values)
                    if len(extra_pixel_values) > 0 and all(x is not None for x in extra_pixel_values)
                    else None
                )
                if len(seg_image_embeds) > 0 and isinstance(seg_image_embeds[0], (list, tuple)):
                    stacked_seg_image_embeds = tuple([torch.stack(x) for x in zip(*seg_image_embeds)])
                else:
                    stacked_seg_image_embeds = torch.stack(seg_image_embeds) if len(seg_image_embeds) > 0 else None
                extend_data_dict.update(
                    {
                        "extra_pixel_values": None,
                        "seg_image_embeds": stacked_seg_image_embeds,
                        "extra_padded_masks": extra_padded_masks,
                    }
                )
                del seg_visual_outputs
            else:
                # here, extra_pixel_values is image_processor output
                extend_data_dict.update(
                    {
                        "extra_pixel_values": extra_padded_pixel_values.to(self.segmentor.dtype),
                        "extra_padded_masks": extra_padded_masks,
                        "seg_image_embeds": None,
                    }
                )
                data_dict["extra_pixel_values"] = None
        else:
            data_dict["extra_pixel_values"] = None

        if data_dict.get("vprompt_masks", None) is not None and hasattr(self, "vision_sampler"):
            vprompt_masks = data_dict.pop("vprompt_masks")
            class_labels, contiguous_labels, vprompt_indices = self._get_vgd_labels(data_samples)
            sampled_labels = self._get_attrs_from_data_samples(data_samples, ["sampled_labels"])[0]
            grid_thw = image_grid_thw if self.sampler_input_feat == "pixel_values" else None
            sampled_feats = self.vision_sampler(
                data_dict[self.sampler_input_feat],
                vprompt_masks,
                grid_thw=grid_thw,
                vprompt_indices=vprompt_indices,
                spatial_merge_size=(
                    self.vlm.config.vision_config.spatial_merge_size if self.vlm is not None else None
                ),
            )
            assert all(
                sampled_feat is not None for sampled_feat in sampled_feats
            ), f"{data_dict[self.sampler_input_feat]}, {vprompt_masks}"
            vprompt_feats, vprompt_masks, _ = self._get_vprompt_feats_and_masks(
                sampled_feats, vprompt_masks, class_labels, contiguous_labels, sampled_labels
            )
            data_dict["vprompt_feats"] = vprompt_feats
            kwargs["vprompt_masks"] = vprompt_masks

        if self.llm or self.vlm is not None:
            data_dict = prepare_inputs_labels_for_mlm(
                mlm=self.llm or self.vlm,
                use_dual_encoder=self.use_dual_encoder,
                temporal_process_fn=self.temporal_process_fn,
                extra_temporal_process_fn=self.extra_temporal_process_fn,
                **data_dict,
            )

        data_dict.update(extend_data_dict)

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
        seg_image_embeds = data_dict.pop("seg_image_embeds", None)
        extra_padded_masks = data_dict.pop("extra_padded_masks", None)
        image_embeds = data_dict.pop("image_embeds", None)
        image_masks = data_dict.pop("image_masks", None)
        task_names, image_size, scaled_size, mask_labels, class_labels, sampled_labels = (
            self._get_attrs_from_data_samples(
                data_samples,
                [
                    "task_names",
                    "image_sizes",
                    "scaled_sizes",
                    "mask_labels",
                    "class_labels",
                    "sampled_labels",
                ],
                **kwargs,
            )
        )
        task_names = task_names if task_names is not None else ["img_genseg"]
        assert (
            len(set(task_names)) == 1 and task_names[0] in DEFAULT_TASKS
        ), f"Task name {task_names} is not in {DEFAULT_TASKS}"

        seg_embeds = None
        cond_embeds = None
        token_embeds = None
        embed_masks = None
        token_masks = None
        llm_outputs = None
        vlm_outputs = None
        seg_outputs = None
        cond_lens = None

        if self.llm is not None:
            llm_outputs = self.llm(**data_dict, output_hidden_states=True)

        if self.vlm is not None:
            vlm_outputs = self.vlm(
                **data_dict,
                image_embeds=image_embeds,
                image_masks=image_masks,
                output_hidden_states=True,
            )

        mlm_outputs = llm_outputs or vlm_outputs
        if self.segmentor is None or self.segmentor.decoder is None:
            return mlm_outputs, None

        if mlm_outputs is not None:
            mlm_hidden_states = mlm_outputs.hidden_states
            mlm_last_hidden_state = mlm_hidden_states[-1]
            mlm_embeds = self.llm_projector(mlm_last_hidden_state)
            if cond_ids is not None:
                cond_embeds, token_embeds, cond_ids = self._get_index_embeds(mlm_embeds, cond_ids)
            if seg_ids is not None:
                seg_embeds, _, _ = self._get_index_embeds(mlm_embeds, seg_ids)

            if cond_embeds and seg_embeds:
                (
                    cond_embeds,
                    token_embeds,
                    cond_ids,
                    seg_embeds,
                    embed_masks,
                    token_masks,
                    cond_lens,
                ) = self._process_embeds(
                    cond_embeds, token_embeds, seg_embeds, cond_ids, task_names[0]
                )

        if seg_embeds is not None or mlm_outputs is None:
            if seg_embeds is not None and seg_embeds.shape[1] != 1:
                seg_outputs = None
            else:
                seg_outputs = self.segmentor(
                    pixel_values=extra_pixel_values,
                    padded_masks=extra_padded_masks,
                    image_embeds=seg_image_embeds,
                    cond_embeds=cond_embeds,
                    cond_ids=cond_ids,
                    token_embeds=token_embeds,
                    seg_embeds=seg_embeds,
                    embed_masks=embed_masks,
                    token_masks=token_masks,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                    cond_lens=cond_lens,
                    return_dict=True,
                )
                if kwargs.pop("do_postprocess", False):
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        cond_ids=cond_ids,
                        use_bg_embeds=hasattr(self, "bg_embeds"),
                        sampled_labels=sampled_labels,
                        **kwargs,
                    )

        return mlm_outputs, seg_outputs

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
        extra_padded_masks = data_dict.pop("extra_padded_masks", None)
        seg_image_embeds = data_dict.pop("seg_image_embeds", None)
        input_cond_ids = data_dict.pop("cond_ids", None)
        image_embeds = data_dict.pop("image_embeds", None)
        image_masks = data_dict.pop("image_masks", None)
        task_names, image_size, scaled_size, sampled_labels = self._get_attrs_from_data_samples(
            data_samples,
            ["task_names", "image_sizes", "scaled_sizes", "sampled_labels"],
            **kwargs,
        )
        task_names = task_names if task_names is not None else ["img_genseg"]
        assert (
            len(set(task_names)) == 1 and task_names[0] in DEFAULT_TASKS
        ), f"Task name {task_names} is not in {DEFAULT_TASKS}"

        generation_config = kwargs.pop("generation_config", None)
        stopping_criteria = kwargs.pop("stopping_criteria", None)

        seg_embeds = None
        cond_embeds = None
        token_embeds = None
        embed_masks = None
        token_masks = None
        cond_ids = None
        llm_outputs = None
        vlm_outputs = None
        seg_outputs = None
        cond_lens = None

        if self.llm is not None:
            llm_outputs = self.llm.generate(
                **data_dict,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )

        if self.vlm is not None:
            vlm_outputs = self.vlm.generate(
                **data_dict,
                image_embeds=image_embeds,
                image_masks=image_masks,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )

        mlm_outputs = llm_outputs or vlm_outputs
        if (self.segmentor is None or self.segmentor.decoder is None) and self.llm is not None:
            return mlm_outputs, None

        if (self.segmentor is None or self.segmentor.decoder is None) and self.vlm is not None:
            return mlm_outputs, None

        if mlm_outputs is not None:
            mlm_output_ids = mlm_outputs.sequences
            mlm_hidden_states = mlm_outputs.hidden_states
            input_hidden_states = mlm_hidden_states[0][-1]
            mlm_last_hidden_state = torch.cat([x[-1] for x in mlm_hidden_states], dim=1)
            mlm_input_embeds = self.llm_projector(input_hidden_states)
            mlm_output_embeds = self.llm_projector(mlm_last_hidden_state)

            L = input_hidden_states.shape[1]
            if input_cond_ids is not None:
                cond_embeds, token_embeds, cond_ids = self._get_index_embeds(mlm_input_embeds, input_cond_ids)

            # update cond_embeds if there is pstart and pend token in the output
            pstart_idx = (mlm_output_ids[..., :-1] == self.pstart_token_idx).nonzero()[:, 1]
            pend_idx = (mlm_output_ids[..., :-1] == self.pend_token_idx).nonzero()[:, 1]
            cls_idx = (mlm_output_ids[..., :-1] == self.cls_token_idx).nonzero()[:, 1]
            if len(pstart_idx) > 0 or len(cls_idx) > 0:
                output_cond_ids = torch.full(
                    mlm_last_hidden_state.shape[:2], -1, dtype=torch.long, device=input_hidden_states.device
                )
                shift = mlm_input_embeds.shape[1]
                if self.cond_type in ["phrase", "all"]:
                    for i, (pstart, pend) in enumerate(zip(pstart_idx, pend_idx)):
                        output_cond_ids[
                            :, shift + pstart + self.ptoken_shift : shift + pend + 1 - self.ptoken_shift
                        ] = i
                if self.cond_type in ["cls", "all"]:
                    for i, ci in enumerate(cls_idx):
                        output_cond_ids[:, shift + ci] = i

                cond_embeds, token_embeds, cond_ids = self._get_index_embeds(mlm_output_embeds, output_cond_ids)

            # update seg_ids if there is seg token in the output
            seg_idx = (mlm_output_ids[..., :-1] == self.seg_token_idx).nonzero()[:, 1]
            if len(seg_idx) > 0:
                B = mlm_output_ids.shape[0]
                assert B == 1, "Only support batch size 1 for prediction"
                seg_ids = torch.full_like(
                    mlm_output_ids[..., :-1], -1, dtype=torch.long, device=input_hidden_states.device
                )
                for i, idx in enumerate(seg_idx):
                    seg_ids[:, idx] = i
                seg_ids = torch.cat(
                    [torch.full((B, L), -1, dtype=torch.long, device=input_hidden_states.device), seg_ids], dim=-1
                )
                seg_embeds, _, _ = self._get_index_embeds(mlm_output_embeds, seg_ids)

            has_cond_and_seg = (cond_embeds is not None and len(cond_embeds) > 0) and (
                seg_embeds is not None and len(seg_embeds) > 0
            )
            if has_cond_and_seg:
                (
                    cond_embeds,
                    token_embeds,
                    cond_ids,
                    seg_embeds,
                    embed_masks,
                    token_masks,
                    cond_lens,
                ) = self._process_embeds(
                    cond_embeds, token_embeds, seg_embeds, cond_ids, task_names[0]
                )
        else:
            has_cond_and_seg = False

        if has_cond_and_seg or mlm_outputs is None:
            if seg_embeds is not None and seg_embeds.shape[1] != 1:
                seg_outputs = None
            else:
                seg_outputs = self.segmentor(
                    pixel_values=extra_pixel_values,
                    padded_masks=extra_padded_masks,
                    image_embeds=seg_image_embeds,
                    cond_embeds=cond_embeds,
                    token_embeds=token_embeds,
                    cond_ids=cond_ids,
                    seg_embeds=seg_embeds,
                    embed_masks=embed_masks,
                    token_masks=token_masks,
                    cond_lens=cond_lens,
                    return_dict=True,
                )
                if kwargs.pop("do_postprocess", True):
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        cond_ids=cond_ids,
                        use_bg_embeds=hasattr(self, "bg_embeds"),
                        sampled_labels=sampled_labels,
                        **kwargs,
                    )
        return mlm_outputs, seg_outputs

    def compute_loss(self, data_dict, data_samples=None, **kwargs):
        mlm_outputs, seg_outputs = self._forward(data_dict, data_samples, **kwargs)
        loss, loss_llm, loss_seg = 0.0, 0.0, 0.0
        if mlm_outputs is not None and seg_outputs is None:
            loss_llm = mlm_outputs.loss * self.llm_loss_weight
            loss = loss_llm
            loss_dict = {"loss": loss, "loss_llm": loss_llm}
        elif mlm_outputs is None and seg_outputs is not None:
            loss_seg = seg_outputs.loss * self.seg_loss_weight
            loss_seg_dict = {k: v * self.seg_loss_weight for k, v in seg_outputs.loss_dict.items()}
            loss = loss_seg
            loss_dict = {"loss": loss, "loss_seg": loss_seg}
            loss_dict.update(loss_seg_dict)
        elif mlm_outputs is not None and seg_outputs is not None:
            loss_llm = mlm_outputs.loss * self.llm_loss_weight
            loss_seg = seg_outputs.loss * self.seg_loss_weight
            loss_seg_dict = {k: v * self.seg_loss_weight for k, v in seg_outputs.loss_dict.items()}
            loss = loss_llm + loss_seg
            loss_dict = {"loss": loss, "loss_llm": loss_llm, "loss_seg": loss_seg}
            loss_dict.update(loss_seg_dict)
        else:
            raise ValueError("mlm_outputs and seg_outputs are both None")

        return loss_dict

    def full_state_dict(self, *args, **kwargs):
        """Return all model parameters, including frozen base modules."""
        return super().state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        if kwargs.pop("full_model", False):
            return self.full_state_dict(*args, **kwargs)

        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. vision_encoder
        if self.vision_encoder is not None:
            if self.use_vision_encoder_lora:
                to_return.update(get_peft_model_state_dict(self.vision_encoder, state_dict=state_dict))
            elif not self.freeze_vision_encoder:
                to_return.update({k: v for k, v in state_dict.items() if "vision_encoder." in k})
        # Step 2. segmentor
        if self.segmentor is not None:
            if self.use_mask_encoder_lora:
                to_return.update(get_peft_model_state_dict(self.segmentor.encoder, state_dict=state_dict))
            elif not self.freeze_mask_encoder:
                to_return.update({k: v for k, v in state_dict.items() if "segmentor.encoder" in k})

            # segmentor other parts except encoder
            to_return.update(
                {k: v for k, v in state_dict.items() if "segmentor" in k and "segmentor.encoder" not in k}
            )
        # Step 3. LLM
        if self.llm is not None:
            if self.use_llm_lora:
                to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
                if self.special_tokens is not None:
                    to_return.update({k: v for k, v in state_dict.items() if "lm_head." in k or "embed_tokens." in k})
            elif not self.freeze_llm:
                to_return.update({k: v for k, v in state_dict.items() if "llm." in k})
        # Step 4. vlm
        if self.vlm is not None:
            if self.use_vlm_lora:
                to_return.update(get_peft_model_state_dict(self.vlm, state_dict=state_dict))
                to_return.update(
                    {
                        k: v
                        for k, v in state_dict.items()
                        if "vlm.base_model.model.model.visual.merger." in k
                        or "vlm.base_model.model.model.visual.deepstack_merger_list." in k
                    }
                )
                if self.special_tokens is not None:
                    to_return.update({k: v for k, v in state_dict.items() if "lm_head." in k or "embed_tokens." in k})
                if not self.freeze_vision_encoder:
                    to_return.update(
                        {k: v for k, v in state_dict.items() if "vlm.base_model.model.model.visual." in k}
                    )
            elif not (self.freeze_vlm and self.freeze_vision_encoder and self.freeze_llm):
                to_return.update(
                    {
                        k: v
                        for k, v in state_dict.items()
                        if "vlm.model.visual.merger." in k or "vlm.model.visual.deepstack_merger_list." in k
                    }
                )
                if not self.freeze_vlm:
                    to_return.update({k: v for k, v in state_dict.items() if "vlm." in k})
                if not self.freeze_vision_encoder:
                    to_return.update({k: v for k, v in state_dict.items() if "vlm.model.visual." in k})
                if not self.freeze_llm:
                    to_return.update(
                        {
                            k: v
                            for k, v in state_dict.items()
                            if "vlm.model.language_model." in k or "vlm.lm_head." in k
                        }
                    )
        # Step 5. Projector
        to_return.update({k: v for k, v in state_dict.items() if "visual_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "seg_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "llm_projector." in k})
        # Step 6. seg_connector
        to_return.update({k: v for k, v in state_dict.items() if "seg_connector." in k})
        # Step 7. other embeds
        to_return.update({k: v for k, v in state_dict.items() if "bg_embeds." in k})
        to_return.update({k: v for k, v in state_dict.items() if "pad_embeds." in k})
        to_return.update({k: v for k, v in state_dict.items() if "vgd_embeds." in k})
        # Step 8. vision_sampler
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

    def _prepare_vlm_for_lora(self, lora_config, **kwargs):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.vlm)
            lora_config.target_modules = modules
        self.vlm = get_peft_model(self.vlm, lora_config, **kwargs)

        if (
            not self.freeze_vision_encoder
            and lora_config.exclude_modules is not None
            and "visual" in lora_config.exclude_modules
        ):
            self.vlm.base_model.model.visual.requires_grad_(True)

    def _prepare_vision_encoder_for_lora(self, lora_config, **kwargs):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.vision_encoder or self.vlm.model.visual)
            lora_config.target_modules = modules
        self.vision_encoder = get_peft_model(self.vision_encoder, lora_config, **kwargs)

    def _prepare_segmentor_for_lora(self, lora_config, **kwargs):
        if self.segmentor is None:
            return
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.segmentor.encoder)
            lora_config.target_modules = modules
        self.segmentor.encoder = get_peft_model(self.segmentor.encoder, lora_config, **kwargs)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        kwargs = {"use_reentrant": False}
        if self.llm is not None:
            self.llm.gradient_checkpointing_enable(kwargs)
        if self.vlm is not None:
            self.vlm.gradient_checkpointing_enable(kwargs)
        if self.vision_encoder is not None:
            self.vision_encoder.gradient_checkpointing_enable(kwargs)
            self.visual_projector.gradient_checkpointing_enable(kwargs)
        if self.segmentor is not None:
            self.segmentor.gradient_checkpointing_enable(kwargs)
            if hasattr(self, "seg_projector"):
                self.seg_projector.gradient_checkpointing_enable(kwargs)
            if hasattr(self, "llm_projector"):
                self.llm_projector.gradient_checkpointing_enable(kwargs)
            if hasattr(self, "seg_connector"):
                self.seg_connector.gradient_checkpointing_enable(kwargs)

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        if self.llm is not None:
            self.llm.gradient_checkpointing_disable()
        if self.vlm is not None:
            self.vlm.gradient_checkpointing_disable()
        if self.vision_encoder is not None:
            self.vision_encoder.gradient_checkpointing_disable()
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
        save_format="pytorch",
        **kwargs,
    ):
        assert save_format == "pytorch", "Only support pytorch format for now"
        self.to_pytorch(cfg, save_dir, fp32, save_pretrained_kwargs)

    def to_pytorch(self, cfg, save_dir, fp32=False, save_pretrained_kwargs={}):
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

        # VLM
        if self.vlm is not None:
            self.vlm.config.text_config.use_cache = True
            if not fp32:
                print_log("Convert vlm to float16", "current")
                self.vlm.half()
            if self.use_vlm_lora:
                vlm_path = osp.join(save_dir, "vlm_adapter")
                print_log(f"Saving vlm adapter to {vlm_path}", "current")
                self.vlm.save_pretrained(vlm_path, **save_pretrained_kwargs)
            elif not (self.freeze_vlm and self.freeze_vision_encoder and self.freeze_llm):
                vlm_path = osp.join(save_dir, "vlm")
                print_log(f"Saving vlm to {vlm_path}", "current")
                self.vlm.save_pretrained(vlm_path, **save_pretrained_kwargs)

            print_log(f"Saving vlm image_processor to {vlm_path}", "current")
            image_processor = BUILDER.build(cfg.image_processor)
            image_processor.save_pretrained(vlm_path, **save_pretrained_kwargs)
            print_log(f"Saving vlm tokenizer to {vlm_path}", "current")
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(vlm_path, **save_pretrained_kwargs)
            self.vlm.config.text_config.use_cache = False

        # Visual Encoder
        if self.vision_encoder is not None:
            if self.use_vision_encoder_lora:
                vision_encoder_path = osp.join(save_dir, "vision_encoder_adapter")
                print_log(f"Saving vision_encoder adapter to {vision_encoder_path}", "current")
                self.vision_encoder.save_pretrained(vision_encoder_path, **save_pretrained_kwargs)
            elif not self.freeze_vision_encoder:
                vision_encoder_path = osp.join(save_dir, "vision_encoder")
                print_log(
                    "Saving vision_encoder image_processor to" f"{vision_encoder_path}",
                    "current",
                )
                image_processor = BUILDER.build(cfg.image_processor)
                image_processor.save_pretrained(vision_encoder_path, **save_pretrained_kwargs)
                print_log(f"Saving vision_encoder to {vision_encoder_path}", "current")
                self.vision_encoder.save_pretrained(vision_encoder_path, **save_pretrained_kwargs)

            # Visual Projector
            visual_projector_path = osp.join(save_dir, "visual_projector")
            print_log(f"Saving visual_projector to {visual_projector_path}", "current")
            self.visual_projector.save_pretrained(visual_projector_path, **save_pretrained_kwargs)

        # Segmentor Encoder
        if self.segmentor is not None:
            if self.use_mask_encoder_lora:
                mask_encoder_path = osp.join(save_dir, "mask_encoder_adapter")
                print_log(f"Saving segmention_encoder adapter to {mask_encoder_path}", "current")
                self.segmentor.encoder.save_pretrained(mask_encoder_path, **save_pretrained_kwargs)
            elif self.use_dual_encoder and not self.freeze_mask_encoder:
                mask_encoder_path = osp.join(save_dir, "segmention_encoder")
                print_log(f"Saving segmentor image_processor to {mask_encoder_path}", "current")
                extra_image_processor = BUILDER.build(cfg.extra_image_processor)
                extra_image_processor.save_pretrained(mask_encoder_path, **save_pretrained_kwargs)
                print_log(f"Saving segmention_encoder to {mask_encoder_path}", "current")
                state_dict = {
                    k.replace("segmentor.encoder.", "vision_encoder."): v
                    for k, v in self.state_dict().items()
                    if "segmentor.encoder" in k
                }
                self.segmentor.save_pretrained(mask_encoder_path, state_dict=state_dict, **save_pretrained_kwargs)

            # Segmentor Projector
            if self.use_dual_encoder and hasattr(self, "seg_projector"):
                seg_projector_path = osp.join(save_dir, "segmentor_projector")
                print_log(f"Saving segmentor_projector to {seg_projector_path}", "current")
                self.seg_projector.save_pretrained(seg_projector_path, **save_pretrained_kwargs)
