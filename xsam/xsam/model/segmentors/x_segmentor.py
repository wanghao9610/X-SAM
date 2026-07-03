import copy
import math
from dataclasses import dataclass
from itertools import accumulate, chain, zip_longest
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel, get_parameter_dtype
from transformers.models.swin import SwinBackbone

from xsam.utils.logging import print_log

from .mask2former import (
    Mask2FormerConfig,
    Mask2FormerLoss,
    Mask2FormerMaskedAttentionDecoderLayer,
    Mask2FormerModel,
    Mask2FormerPixelDecoder,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
    Mask2FormerPixelDecoderEncoderOnly,
    Mask2FormerPixelLevelModule,
    Mask2FormerTransformerModule,
)
from .sam import SamMaskDecoder, SamModel, SamVisionEncoder
from .sam2 import Sam2MaskDecoder, Sam2Model, Sam2VisionModel

# a large negative value as a placeholder score for missing objects, following sam2
NO_OBJ_SCORE = -1024.0


@dataclass
class XSegmentorOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ClassPredictor(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.head_cls_type = config.head_cls_type

        bias_value = -math.log((1 - 0.01) / 0.01)
        if self.head_cls_type == "linear":
            self.predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)
        elif self.head_cls_type == "learn":
            # scale with learnable logit_scale
            self.logit_scale = nn.Parameter(torch.tensor([math.log(1 / 0.07)]), requires_grad=True)
        elif self.head_cls_type == "auto":
            self.cls_bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)
        else:
            raise ValueError(f"Unsupported head_cls_type: {self.head_cls_type}")

    def forward(self, query_embeds: Tensor, cond_embeds: Tensor = None, embed_masks: Tensor = None) -> Tensor:
        if self.head_cls_type == "linear":
            return self.predictor(query_embeds)

        if cond_embeds is None:
            return None

        if self.head_cls_type == "learn":
            cls_pred = self.logit_scale.exp() * torch.einsum(
                "bqd,bcd->bqc",
                F.normalize(query_embeds, dim=-1),
                F.normalize(cond_embeds, dim=-1),
            )
        elif self.head_cls_type == "auto":
            cls_pred = (
                torch.einsum(
                    "bqd,bcd->bqc",
                    query_embeds,
                    cond_embeds,
                )
                / math.sqrt(query_embeds.shape[-1])
                + self.cls_bias
            )

        cls_pred = torch.clamp(cls_pred, min=-50, max=50)

        if embed_masks is not None:
            if embed_masks.ndim == 2:
                embed_masks = embed_masks.unsqueeze(1)
            cls_pred = cls_pred.masked_fill(~embed_masks.bool(), -1e6)

        return cls_pred


class XSegmentor(PreTrainedModel):
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_attention_backend = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        encoder: Literal[SamModel, Sam2Model, Mask2FormerModel],
        decoder=None,
        torch_dtype=torch.float32,
        use_prompt_encoder=False,
        use_memory=True,
        use_decoder=True,
        init_memory=True,
        init_decoder=True,
    ):  # type: ignore
        PreTrainedModel.__init__(self, encoder.config)

        # Sam encoder + Sam decoder
        if isinstance(encoder, SamModel) and decoder is None:
            self.enc_config = encoder.config.vision_config
            self.dec_config = encoder.config.mask_decoder_config
            self.prompt_enc_config = encoder.config.prompt_encoder_config

            self.shared_image_embedding = encoder.shared_image_embedding
            self.encoder = encoder.vision_encoder
            self.positional_encoding = None  # TODO
            self.prompt_encoder = encoder.prompt_encoder if use_prompt_encoder else None

            self.pixel_decoder = None
            self.decoder = encoder.mask_decoder if use_decoder else None
            
            self.memory_encoder = None
            self.memory_attention = None
            self.no_memory_embedding = None
        # Sam encoder + Mask2Former decoder
        elif isinstance(encoder, SamModel) and isinstance(decoder, Mask2FormerModel):
            self.enc_config = encoder.config.vision_config
            self.dec_config = decoder.config
            self.prompt_enc_config = encoder.config.prompt_encoder_config

            self.shared_image_embedding = encoder.shared_image_embedding
            self.encoder = encoder.vision_encoder
            self.positional_encoding = None  # TODO
            self.prompt_encoder = encoder.prompt_encoder if use_prompt_encoder else None
            
            self.pixel_decoder = decoder.pixel_level_module.decoder if use_decoder else None
            self.decoder = decoder.transformer_module if use_decoder else None
            
            self.memory_encoder = None
            self.memory_attention = None
            self.no_memory_embedding = None
        # Mask2Former encoder + Mask2Former decoder
        elif isinstance(encoder, Mask2FormerModel) and decoder is None:
            self.enc_config = encoder.config
            self.dec_config = copy.deepcopy(encoder.config)
            self.enc_config.hidden_size = encoder.config.backbone_config.hidden_size

            self.shared_image_embedding = None
            self.encoder = encoder.pixel_level_module.encoder
            self.positional_encoding = None
            self.prompt_encoder = None

            self.pixel_decoder = encoder.pixel_level_module.decoder if use_decoder else None
            self.decoder = encoder.transformer_module if use_decoder else None
            
            self.memory_encoder = None
            self.memory_attention = None
            self.no_memory_embedding = None
        # Mask2Former encoder + Sam decoder
        elif isinstance(encoder, Mask2FormerModel) and isinstance(decoder, SamModel):
            # TODO: check if this is correct
            self.enc_config = encoder.config
            self.dec_config = decoder.config

            self.shared_image_embedding = None
            self.encoder = encoder.pixel_level_module
            self.positional_encoding = None
            self.prompt_encoder = None

            self.pixel_decoder = encoder.pixel_level_module.decoder if use_decoder else None
            self.decoder = decoder.mask_decoder if use_decoder else None
            
            self.memory_encoder = None
            self.memory_attention = None
            self.no_memory_embedding = None
        # Sam2 encoder + Sam2 decoder
        elif isinstance(encoder, Sam2Model) and decoder is None:
            self.enc_config = encoder.config.vision_config
            self.dec_config = encoder.config.mask_decoder_config
            self.prompt_enc_config = encoder.config.prompt_encoder_config

            self.shared_image_embedding = encoder.shared_image_embedding
            self.encoder = encoder.vision_encoder
            self.positional_encoding = encoder.vision_encoder.neck.position_encoding

            self.prompt_encoder = encoder.prompt_encoder if use_prompt_encoder else None
            self.decoder = encoder.mask_decoder if use_decoder else None
            self.pixel_decoder = None

            self.num_maskmem = encoder.config.num_maskmem
            self.maskmem_dim = encoder.config.memory_encoder_output_channels
            self.memory_encoder = encoder.memory_encoder if use_memory else None
            self.memory_attention = encoder.memory_attention if use_memory else None
            self.no_memory_embedding = encoder.no_memory_embedding if use_memory else None
        # Sam2 encoder + Mask2Former decoder
        elif isinstance(encoder, Sam2Model) and isinstance(decoder, Mask2FormerModel):
            self.enc_config = encoder.config.vision_config
            self.dec_config = decoder.config
            self.prompt_enc_config = encoder.config.prompt_encoder_config

            self.shared_image_embedding = encoder.shared_image_embedding
            self.encoder = encoder.vision_encoder
            self.positional_encoding = encoder.vision_encoder.neck.position_encoding

            self.prompt_encoder = encoder.prompt_encoder if use_prompt_encoder else None
            self.pixel_decoder = decoder.pixel_level_module.decoder
            self.decoder = decoder.transformer_module if use_decoder else None

            self.num_maskmem = encoder.config.num_maskmem
            self.maskmem_dim = encoder.config.memory_encoder_output_channels
            self.memory_encoder = encoder.memory_encoder if use_memory else None
            self.memory_attention = encoder.memory_attention if use_memory else None
            self.no_memory_embedding = encoder.no_memory_embedding if use_memory else None
        else:
            raise ValueError(f"Unsupported encoder and decoder type: {type(encoder)} and {type(decoder)}")

        if not use_decoder:
            self.encoder = self.encoder.to(torch_dtype)
            self.shared_image_embedding = None
            self.prompt_encoder = None
            self.pixel_decoder = None

            self.decoder = None
            self.memory_encoder = None
            self.memory_attention = None
            self.no_memory_embedding = None

            return

        if init_decoder and use_decoder:
            print_log(f"Reinitializing decoder of {self.decoder.__class__.__name__}.", logger="current")
            # means decoder and pixel_decoder are not from pretained model, so we need to initialize the weights
            self.decoder.apply(self._init_decoder_weights)
            self.pixel_decoder.apply(self._init_decoder_weights)

        if init_memory and use_memory:
            print_log(f"Reinitializing memory encoder of {self.memory_encoder.__class__.__name__}.", logger="current")
            self.memory_encoder.apply(self._init_memory_weights)
            print_log(
                f"Reinitializing memory attention of {self.memory_attention.__class__.__name__}.", logger="current"
            )
            self.memory_attention.apply(self._init_memory_weights)

        self.weight_dict: Dict[str, float] = {
            "loss_cls": self.dec_config.class_weight,
            "loss_mask": self.dec_config.mask_weight,
            "loss_dice": self.dec_config.dice_weight,
        }
        self.criterion = Mask2FormerLoss(config=self.dec_config, weight_dict=self.weight_dict)
        self.class_predictor = ClassPredictor(config=self.dec_config)

        self.encoder = self.encoder.to(torch_dtype)
        self.decoder = self.decoder.to(torch_dtype)
        self.class_predictor = self.class_predictor.to(torch_dtype)
        self.positional_encoding = (
            self.positional_encoding.to(torch_dtype) if self.positional_encoding is not None else None
        )
        self.pixel_decoder = self.pixel_decoder.to(torch_dtype) if self.pixel_decoder is not None else None
        self.memory_encoder = self.memory_encoder.to(torch_dtype) if self.memory_encoder is not None else None
        self.memory_attention = self.memory_attention.to(torch_dtype) if self.memory_attention is not None else None
        self.shared_image_embedding = (
            self.shared_image_embedding.to(torch_dtype).requires_grad_(False)
            if self.shared_image_embedding is not None
            else None
        )
        self.no_memory_embedding = (
            self.no_memory_embedding.to(torch_dtype) if self.no_memory_embedding is not None else None
        )
        self.criterion = self.criterion.to(torch_dtype)
        self.ignore_label = self.dec_config.ignore_label
        self.background_label = self.dec_config.background_label
        self.use_repeat_cond = self.dec_config.use_repeat_cond

        self.gradient_checkpointing = False

    @property
    def config_class(self):
        return self.enc_config.__class__

    def enable_input_require_grads(self):
        def make_inputs_require_grad(module, input, output):
            if isinstance(output, Tensor):
                output.requires_grad_(True)
            elif isinstance(output, tuple):
                output[0].requires_grad_(True)

        self.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def get_dummy_tensor(self, module: Optional[nn.Module], ref_tensor: Tensor) -> Tensor:
        """
        Create a zero-valued scalar that depends on *all* parameters of `module`.

        This is useful when some branches skip a submodule entirely (e.g. early-exit)
        but we still want DDP to see those parameters as "used" in the autograd graph.
        """
        if module is None:
            return ref_tensor.new_zeros(())
        vals = [p.reshape(-1)[0] for p in module.parameters() if p.requires_grad]
        if not vals:
            return ref_tensor.new_zeros(())
        dummy = torch.stack(vals).sum()
        # Multiply by zero so it has no numerical effect while keeping graph edges.
        return dummy.to(dtype=ref_tensor.dtype) * ref_tensor.new_zeros(())

    def get_input_embeddings(self) -> nn.Module:
        # sam
        if hasattr(self.encoder, "patch_embed"):
            return self.encoder.patch_embed
        # sam2
        if hasattr(self.encoder, "backbone"):
            return self.encoder.backbone.patch_embed
        # mask2former
        elif hasattr(self.encoder, "embeddings"):
            return self.encoder.embeddings.patch_embeddings
        else:
            raise ValueError(f"Unsupported encoder: {type(self.encoder)}")

    def _init_decoder_weights(self, module: nn.Module):
        xavier_std = self.dec_config.init_xavier_std if hasattr(self.dec_config, "init_xavier_std") else 1.0
        std = self.dec_config.init_std if hasattr(self.dec_config, "init_std") else 0.02

        if isinstance(module, Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            if module.value_proj.weight.data.dim() > 1:
                nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            if module.output_proj.weight.data.dim() > 1:
                nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)

            if module.use_text_cross_attn and module.use_zero_init:
                nn.init.zeros_(module.text_cross_attn.out_proj.weight)
                nn.init.zeros_(module.text_cross_attn.out_proj.bias)

        elif isinstance(module, Mask2FormerPixelLevelModule):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()

        elif isinstance(module, Mask2FormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if hasattr(module, "reference_points"):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)

    def _init_memory_weights(self, module: nn.Module):
        """
        Initialization for SAM2 memory modules.

        Note: `XSegmentor._init_weights` is primarily tailored for Mask2Former components
        and uses `dec_config` hyper-parameters. For SAM2 memory encoder/attention, using
        SAM2's `initializer_range` and LayerNorm init is more appropriate and avoids
        partially-initialized transformer blocks.
        """
        std = getattr(self.config, "initializer_range", None)
        if std is None:
            std = self.dec_config.init_std if hasattr(self.dec_config, "init_std") else 0.02

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    @torch.no_grad()
    def get_image_wide_positional_embeddings(self):
        size = (
            self.enc_config.image_size // self.enc_config.patch_size,
            self.enc_config.image_size // self.enc_config.patch_size,
        )
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones(size, device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size[0]
        x_embed = x_embed / size[1]

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        cond_ids: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            cond_ids=cond_ids,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []  # type: ignore

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append(
                {
                    "masks_queries_logits": aux_binary_masks,
                    "class_queries_logits": aux_classes,
                }
            )

        return auxiliary_logits

    def postprocess_masks_preds(self, masks_preds):
        # upscale the mask preds to image_size
        new_masks_preds = []
        # multi-level masks_preds
        for masks_pred in masks_preds:
            masks_pred = F.interpolate(
                masks_pred,
                size=(
                    self.enc_config.image_size,
                    self.enc_config.image_size,
                ),
                mode="bilinear",
                align_corners=False,
            )
            new_masks_preds.append(masks_pred)

        return new_masks_preds

    def get_temporal_encoding(self, distance: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / half)
        angles = distance * freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1).view(1, 1, dim)

    def retrieve_memory(
        self, memory_bank: List[Dict], temporal_idx: int, num_scales: int = 3
    ) -> Tuple[List[Tensor], List[Tensor]]:
        multi_scale_combined_embeds = []
        multi_scale_combined_pos_embeds = []

        available_memory = [m for m in memory_bank if m["temporal_idx"] < temporal_idx]

        if not available_memory:
            return None, None

        for i in range(num_scales):
            scale_embeds = []
            scale_pos = []

            for m in available_memory:
                m_embed = m["memory_embeds"][i]  # [L, B, C]
                m_pos = m["memory_positional_embeddings"][i]  # [L, B, C]

                t_gap = temporal_idx - m["temporal_idx"]
                t_enc = self.get_temporal_encoding(t_gap, self.maskmem_dim, m_pos.device, m_pos.dtype)  # [1, 1, C]

                scale_embeds.append(m_embed)
                scale_pos.append(m_pos + t_enc)  # spatial position + temporal encoding

            multi_scale_combined_embeds.append(torch.cat(scale_embeds, dim=0))
            multi_scale_combined_pos_embeds.append(torch.cat(scale_pos, dim=0))

        return multi_scale_combined_embeds, multi_scale_combined_pos_embeds

    def encode_memory(
        self,
        multi_scale_features: Tuple[Tensor],
        class_logits: Tensor,
        mask_logits: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor]]:

        multi_scale_embeds = []
        multi_scale_positional_embeddings = []
        for i, single_scale_features in enumerate(multi_scale_features):
            single_scale_memory_embeds, single_scale_memory_positional_embeddings = self.memory_encoder(
                single_scale_features, class_logits, mask_logits, i
            )
            multi_scale_embeds.append(single_scale_memory_embeds)
            multi_scale_positional_embeddings.append(single_scale_memory_positional_embeddings)
        return multi_scale_embeds, multi_scale_positional_embeddings

    def _select_temporal_prompt_tensor(
        self, tensor: Optional[Tensor], temporal_idx: int, num_frames: int
    ) -> Optional[Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 0 or tensor.shape[0] != num_frames:
            return tensor
        return tensor[temporal_idx][None, ...]

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        padded_masks: Optional[Tensor] = None,
        image_embeds: Optional[Tuple[Tensor]] = None,
        seg_embeds: Optional[Tensor] = None,
        cond_embeds: Optional[Tensor] = None,
        token_embeds: Optional[Tensor] = None,
        cond_ids: Optional[Tensor] = None,
        embed_masks: Optional[Tensor] = None,
        token_masks: Optional[Tensor] = None,
        cond_lens: Optional[List] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        output_auxiliary_logits: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        output_attentions: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> XSegmentorOutput:

        frame_lens = [1] * (len(mask_labels) if mask_labels is not None else 1)
        if cond_embeds is not None and mask_labels is not None and mask_labels[0].ndim == 4:
            # flatten the cond_embeds, embed_masks, mask_labels and class_labels
            cond_embeds = torch.cat(
                [
                    cond_embed[None, ...].repeat(len(class_label), 1, 1)
                    for cond_embed, class_label in zip(cond_embeds, class_labels)
                ]
            )
            embed_masks = (
                torch.cat(
                    [
                        embed_mask[None, ...].repeat(len(class_label), 1)
                        for embed_mask, class_label in zip(embed_masks, class_labels)
                    ]
                )
                if embed_masks is not None
                else None
            )
            cond_ids = (
                torch.cat(
                    [
                        cond_id[None, ...].repeat(len(class_label), 1)
                        for cond_id, class_label in zip(cond_ids, class_labels)
                    ]
                )
                if cond_ids is not None
                else None
            )
            frame_lens = [len(class_label) for class_label in class_labels]
            # mask_labels: B x [T, H, W] -> T*B x [H, W], class_labels: B x [T] -> T*B
            mask_labels, class_labels = zip(
                *[
                    (
                        __mask_label[__class_label != self.ignore_label],
                        __class_label[__class_label != self.ignore_label],
                    )
                    for _mask_labels, _class_labels in zip(
                        zip_longest(*mask_labels, fillvalue=None), zip_longest(*class_labels, fillvalue=None)
                    )
                    for __mask_label, __class_label in zip(_mask_labels, _class_labels)
                    if __mask_label is not None and __class_label is not None
                ]
            )

        if cond_lens is not None and mask_labels is not None:
            label_offsets = [
                label_offset
                for frame_len, cond_len in zip(frame_lens, list(accumulate([0] + cond_lens[:-1])))
                for label_offset in [cond_len] * frame_len
            ]
            class_labels = [
                torch.where(
                    (class_label != self.ignore_label) & (class_label != self.background_label),
                    class_label + label_offset,
                    class_label,
                )
                for class_label, label_offset in zip(class_labels, label_offsets)
            ]
            if self.use_repeat_cond:
                # flatten the cond_embeds, cond_ids and embed_masks, each phrase as unique category
                cond_embeds = torch.cat(
                    [
                        cond_embed[None, ...].repeat(cond_len, *([1] * cond_embed.ndim))
                        for cond_embed, cond_len in zip(cond_embeds, cond_lens)
                    ]
                )
                cond_ids = (
                    torch.cat(
                        [
                            cond_id[None, ...].repeat(cond_len, *([1] * cond_id.ndim))
                            for cond_id, cond_len in zip(cond_ids, cond_lens)
                        ]
                    )
                    if cond_ids is not None
                    else None
                )
                embed_masks = (
                    torch.cat(
                        [
                            embed_mask[None, ...].repeat(cond_len, *([1] * embed_mask.ndim))
                            for embed_mask, cond_len in zip(embed_masks, cond_lens)
                        ]
                    )
                    if embed_masks is not None
                    else None
                )
                mask_labels = list(chain(*[m.split(1) for m in mask_labels]))
                class_labels = list(chain(*[c.split(1) for c in class_labels]))
            else:
                # select the last seg_embed for each frame/image
                seg_embeds = torch.stack([seg_embeds[cond_len - 1] for cond_len in cond_lens])

        # sam_enc + sam_dec
        if isinstance(self.encoder, SamVisionEncoder) and isinstance(self.decoder, SamMaskDecoder):
            if image_embeds is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # TODO: multi-scale image_embeds
                image_embeds = encoder_outputs.last_hidden_state

            image_positional_embeddings = self.get_image_wide_positional_embeddings()
            batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]
            image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
            seg_embeds = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                input_points=None,
                input_labels=None,
                input_boxes=None,
                input_masks=None,
                input_embeds=seg_embeds,
            )
            decoder_outputs = self.decoder(
                image_embeds=image_embeds,
                image_positional_embeddings=image_positional_embeddings,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                attention_similarity=None,
                cond_lens=cond_lens if self.use_repeat_cond else None,
                target_embedding=None,
                output_attentions=output_attentions,
            )

        # sam_enc + mask2former_dec
        elif isinstance(self.encoder, SamVisionEncoder) and isinstance(self.decoder, Mask2FormerTransformerModule):
            if image_embeds is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # TODO: multi-scale image_embeds
                image_embeds = [encoder_outputs.last_hidden_state] * self.dec_config.num_feature_levels

            if isinstance(image_embeds, (list, tuple)) and image_embeds[0].ndim == 5:
                image_embeds = tuple(image_embed[0] for image_embed in image_embeds)
            pixel_decoder_outputs = self.pixel_decoder(
                image_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_outputs = self.decoder(
                multi_scale_features=pixel_decoder_outputs.multi_scale_features,
                mask_features=pixel_decoder_outputs.mask_features,
                seg_embeds=seg_embeds,
                token_embeds=token_embeds,
                cond_lens=cond_lens if self.use_repeat_cond else None,
                output_attentions=output_attentions,
            )
            intermediate_hidden_states = (
                decoder_outputs.intermediate_hidden_states
                if return_dict
                else decoder_outputs[0]
            )
            masks_queries_logits = (
                decoder_outputs.masks_queries_logits if return_dict else decoder_outputs[1]
            )
            last_hidden_state = decoder_outputs.last_hidden_state if return_dict else decoder_outputs[2]
            hidden_states = decoder_outputs.hidden_states if return_dict else decoder_outputs[3]
            attentions = decoder_outputs.attentions if return_dict else decoder_outputs[4]
        # mask2former_enc(swin) + mask2former_dec
        elif isinstance(self.encoder, SwinBackbone) and isinstance(self.decoder, Mask2FormerTransformerModule):
            if image_embeds is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                image_embeds = encoder_outputs.feature_maps
            if isinstance(image_embeds, (list, tuple)) and image_embeds[0].ndim == 5:
                image_embeds = tuple(image_embed[0] for image_embed in image_embeds)
            pixel_decoder_outputs = self.pixel_decoder(
                image_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_outputs = self.decoder(
                multi_scale_features=pixel_decoder_outputs.multi_scale_features,
                mask_features=pixel_decoder_outputs.mask_features,
                seg_embeds=seg_embeds,
                token_embeds=token_embeds,
                cond_lens=cond_lens if self.use_repeat_cond else None,
                output_attentions=output_attentions,
            )
            intermediate_hidden_states = (
                decoder_outputs.intermediate_hidden_states
                if return_dict
                else decoder_outputs[0]
            )
            masks_queries_logits = (
                decoder_outputs.masks_queries_logits if return_dict else decoder_outputs[1]
            )
            last_hidden_state = decoder_outputs.last_hidden_state if return_dict else decoder_outputs[2]
            hidden_states = decoder_outputs.hidden_states if return_dict else decoder_outputs[3]
            attentions = decoder_outputs.attentions if return_dict else decoder_outputs[4]
        # sam2_enc + sam2_dec
        elif isinstance(self.encoder, Sam2VisionModel) and isinstance(self.decoder, Sam2MaskDecoder):
            if image_embeds is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # TODO: multi-scale image_embeds
                image_embeds = encoder_outputs.fpn_hidden_states

            intermediate_hidden_states = []
            mask_queries_logits = []
            last_hidden_state = []
            hidden_states = []
            attentions = []

            assert all(padded_masks.sum(-1) > 0)

            memory_bank = []
            for temporal_idx, batch_padded_masks in enumerate(padded_masks):
                batch_padded_masks = (
                    torch.cat(
                        [
                            batched_padded_mask.repeat(cond_len)
                            for batched_padded_mask, cond_len in zip(batch_padded_masks, cond_lens)
                        ]
                    )
                    if cond_lens is not None and self.use_repeat_cond
                    else batch_padded_masks
                )
                # tuple doesn't support inplace operation
                batch_image_embeds = [image_embed[temporal_idx] for image_embed in image_embeds]

                image_positional_embeddings = self.get_image_wide_positional_embeddings()
                batch_size = pixel_values.shape[0] if pixel_values is not None else batch_image_embeds[0].shape[0]
                image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
                batch_image_embeds[0] = self.decoder.conv_s0(batch_image_embeds[0])
                batch_image_embeds[1] = self.decoder.conv_s1(batch_image_embeds[1])
                # Only add no_memory_embedding for the first frame (no prior memory available)
                if temporal_idx == 0 and self.no_memory_embedding is not None:
                    batch_image_embeds = tuple(
                        [
                            image_embed + self.no_memory_embedding.transpose(1, 2)[..., None]
                            for image_embed in batch_image_embeds
                        ]
                    )
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    input_points=None,
                    input_labels=None,
                    input_boxes=None,
                    input_masks=None,
                    input_embeds=seg_embeds,
                )
                multi_scale_positional_embeds = tuple(
                    [
                        self.positional_encoding(image_embed.shape, device=image_embed.device, dtype=image_embed.dtype)
                        for image_embed in batch_image_embeds
                    ]
                )
                # 1. retrieve memory (all entries in bank are fully encoded)
                (
                    multi_scale_combined_memory_embeds,
                    multi_scale_combined_memory_positional_embeds,
                ) = (
                    self.retrieve_memory(memory_bank, temporal_idx)
                    if temporal_idx > 0 and self.memory_encoder is not None
                    else (None, None)
                )
                # 2. memory attention
                if self.gradient_checkpointing and self.training:
                    memory_attention_outputs = (
                        [
                            (
                                self._gradient_checkpointing_func(
                                    self.memory_attention.__call__,
                                    single_scale_features.flatten(2).permute(2, 0, 1),
                                    combined_memory_embed,
                                    positional_embed.flatten(2).permute(2, 0, 1),
                                    combined_memory_positional_embed,
                                    scale_idx=scale_idx,
                                )
                                .squeeze(0)
                                .permute(0, 2, 1)
                                .reshape(single_scale_features.shape[0], -1, *single_scale_features.shape[2:])
                                if combined_memory_embed.shape[0] > 0
                                else single_scale_features
                                + self.get_dummy_tensor(self.memory_attention, single_scale_features)
                            )
                            for scale_idx, (
                                single_scale_features,
                                combined_memory_embed,
                                positional_embed,
                                combined_memory_positional_embed,
                            ) in enumerate(
                                zip(
                                    batch_image_embeds,
                                    multi_scale_combined_memory_embeds,
                                    multi_scale_positional_embeds,
                                    multi_scale_combined_memory_positional_embeds,
                                )
                            )
                        ]
                        if temporal_idx > 0 and self.memory_attention is not None
                        else None
                    )
                else:
                    memory_attention_outputs = (
                        [
                            (
                                self.memory_attention(
                                    single_scale_features.flatten(2).permute(2, 0, 1),
                                    combined_memory_embed,
                                    positional_embed.flatten(2).permute(2, 0, 1),
                                    combined_memory_positional_embed,
                                    scale_idx=scale_idx,
                                )
                                .squeeze(0)
                                .permute(0, 2, 1)
                                .reshape(single_scale_features.shape[0], -1, *single_scale_features.shape[2:])
                                if combined_memory_embed.shape[0] > 0
                                else single_scale_features
                                + self.get_dummy_tensor(self.memory_attention, single_scale_features)
                            )
                            for scale_idx, (
                                single_scale_features,
                                combined_memory_embed,
                                positional_embed,
                                combined_memory_positional_embed,
                            ) in enumerate(
                                zip(
                                    batch_image_embeds,
                                    multi_scale_combined_memory_embeds,
                                    multi_scale_positional_embeds,
                                    multi_scale_combined_memory_positional_embeds,
                                )
                            )
                        ]
                        if temporal_idx > 0 and self.memory_attention is not None
                        else None
                    )
                # 3. mask decoder
                if self.gradient_checkpointing and self.training:
                    decoder_outputs = self._gradient_checkpointing_func(
                        self.decoder.__call__,
                        memory_attention_outputs or batch_image_embeds[-1],
                        image_positional_embeddings,
                        sparse_embeddings,
                        dense_embeddings,
                        memory_attention_outputs or batch_image_embeds[-1],
                        cond_lens if self.use_repeat_cond else None,
                        output_attentions,
                        output_hidden_states,
                        return_dict,
                        **kwargs,
                    )
                else:
                    decoder_outputs = self.decoder(
                        image_embeds=memory_attention_outputs or batch_image_embeds[-1],
                        image_positional_embeddings=image_positional_embeddings,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        high_resolution_features=memory_attention_outputs or batch_image_embeds[:-1],
                        cond_lens=cond_lens if self.use_repeat_cond else None,
                        token_embeds=token_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        **kwargs,
                    )
                # 4. encode memory
                (
                    multi_scale_memory_embeds,
                    multi_scale_memory_positional_embeddings,
                ) = (
                    self.encode_memory(
                        batch_image_embeds,
                        self.class_predictor(
                            decoder_outputs.intermediate_hidden_states[-1].transpose(0, 1),
                            self._select_temporal_prompt_tensor(cond_embeds, temporal_idx, len(padded_masks)),
                            self._select_temporal_prompt_tensor(embed_masks, temporal_idx, len(padded_masks)),
                        ),
                        decoder_outputs.masks_queries_logits[-1],
                    )
                    if self.memory_encoder is not None
                    else (None, None)
                )
                # 5. store memory with FIFO management (evict oldest if at capacity)
                if self.memory_encoder is not None:
                    if memory_bank and self.num_maskmem > 0 and len(memory_bank) >= self.num_maskmem:
                        memory_bank.pop(0)

                    memory_bank.append(
                        {
                            "temporal_idx": temporal_idx,
                            "memory_embeds": (
                                [m.flatten(2).permute(2, 0, 1) for m in multi_scale_memory_embeds]
                                if multi_scale_memory_embeds is not None
                                else None
                            ),
                            "memory_positional_embeddings": (
                                [p.flatten(2).permute(2, 0, 1) for p in multi_scale_memory_positional_embeddings]
                                if multi_scale_memory_positional_embeddings is not None
                                else None
                            ),
                        }
                    )

                batch_intermediate_hidden_states = (
                    decoder_outputs[0] if not return_dict else decoder_outputs.intermediate_hidden_states
                )
                batch_mask_queries_logits = (
                    decoder_outputs[1] if not return_dict else decoder_outputs.masks_queries_logits
                )
                batch_last_hidden_state = decoder_outputs[2] if not return_dict else decoder_outputs.last_hidden_state
                batch_hidden_states = decoder_outputs[3] if not return_dict else decoder_outputs.hidden_states
                batch_attentions = decoder_outputs[4] if not return_dict else decoder_outputs.attentions

                intermediate_hidden_states.append(
                    tuple(
                        [
                            intermediate_hidden_state[:, batch_padded_masks, ...]
                            for intermediate_hidden_state in batch_intermediate_hidden_states
                        ]
                    )
                )
                mask_queries_logits.append(
                    tuple(
                        [
                            mask_queries_logit[batch_padded_masks, ...]
                            for mask_queries_logit in batch_mask_queries_logits
                        ]
                    )
                )
                last_hidden_state.append(batch_last_hidden_state[batch_padded_masks, ...])
                hidden_states.append(
                    tuple([hidden_state[:, batch_padded_masks, ...] for hidden_state in batch_hidden_states])
                    if batch_hidden_states is not None
                    else None
                )
                attentions.append(
                    tuple([attention[batch_padded_masks, ...] for attention in batch_attentions])
                    if batch_attentions is not None
                    else None
                )

            intermediate_hidden_states = tuple(
                torch.cat(intermediate_hidden_state, dim=1)
                for intermediate_hidden_state in zip(*intermediate_hidden_states)
            )
            masks_queries_logits = tuple(
                torch.cat(mask_queries_logit, dim=0) for mask_queries_logit in zip(*mask_queries_logits)
            )
            last_hidden_state = torch.cat(last_hidden_state, dim=0)
            hidden_states = (
                tuple(torch.cat(hidden_state, dim=1) for hidden_state in zip(*hidden_states))
                if hidden_states[0] is not None
                else None
            )
            attentions = (
                tuple(torch.cat(attention, dim=0) for attention in zip(*attentions))
                if attentions[0] is not None
                else None
            )
        # sam2_enc + mask2former_dec
        elif isinstance(self.encoder, Sam2VisionModel) and isinstance(self.decoder, Mask2FormerTransformerModule):
            if image_embeds is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # TODO: multi-scale image_embeds
                image_embeds = [encoder_outputs.last_hidden_state] * 4

            intermediate_hidden_states = []
            mask_queries_logits = []
            last_hidden_state = []
            hidden_states = []
            attentions = []

            assert all(padded_masks.sum(-1) > 0)
            memory_bank = []
            for temporal_idx, batch_padded_masks in enumerate(padded_masks):
                batch_padded_masks = (
                    torch.cat(
                        [
                            batched_padded_mask.repeat(cond_len)
                            for batched_padded_mask, cond_len in zip(batch_padded_masks, cond_lens)
                        ]
                    )
                    if cond_lens is not None and self.use_repeat_cond
                    else batch_padded_masks
                )
                batch_image_embeds = tuple([image_embed[temporal_idx] for image_embed in image_embeds])
                if temporal_idx == 0 and self.no_memory_embedding is not None:
                    batch_image_embeds = tuple(
                        [
                            image_embed + self.no_memory_embedding.transpose(1, 2)[..., None]
                            for image_embed in batch_image_embeds
                        ]
                    )
                pixel_decoder_outputs = self.pixel_decoder(
                    batch_image_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                mask_features = pixel_decoder_outputs[0] if not return_dict else pixel_decoder_outputs.mask_features
                multi_scale_features = (
                    pixel_decoder_outputs[1] if not return_dict else pixel_decoder_outputs.multi_scale_features
                )

                multi_scale_positional_embeds = tuple(
                    [
                        self.positional_encoding(image_embed.shape, device=image_embed.device, dtype=image_embed.dtype)
                        for image_embed in multi_scale_features
                    ]
                )
                # 1. retrieve memory (all entries in bank are fully encoded)
                (
                    multi_scale_combined_memory_embeds,
                    multi_scale_combined_memory_positional_embeds,
                ) = (
                    self.retrieve_memory(memory_bank, temporal_idx, len(multi_scale_features))
                    if temporal_idx > 0 and self.memory_encoder is not None
                    else (None, None)
                )
                # 2. memory attention
                if self.gradient_checkpointing and self.training:
                    memory_attention_outputs = (
                        [
                            (
                                (
                                    self._gradient_checkpointing_func(
                                        self.memory_attention.__call__,
                                        single_scale_features.flatten(2).permute(2, 0, 1),
                                        combined_memory_embed,
                                        positional_embed.flatten(2).permute(2, 0, 1),
                                        combined_memory_positional_embed,
                                        scale_idx=scale_idx,
                                    )
                                    .squeeze(0)
                                    .permute(0, 2, 1)
                                    .reshape(single_scale_features.shape[0], -1, *single_scale_features.shape[2:])
                                )
                                if combined_memory_embed.shape[0] > 0
                                else single_scale_features
                                + self.get_dummy_tensor(self.memory_attention, single_scale_features)
                            )
                            for scale_idx, (
                                single_scale_features,
                                combined_memory_embed,
                                positional_embed,
                                combined_memory_positional_embed,
                            ) in enumerate(
                                zip(
                                    multi_scale_features,
                                    multi_scale_combined_memory_embeds,
                                    multi_scale_positional_embeds,
                                    multi_scale_combined_memory_positional_embeds,
                                )
                            )
                        ]
                        if temporal_idx > 0 and self.memory_attention is not None
                        else None
                    )
                else:
                    memory_attention_outputs = (
                        [
                            (
                                (
                                    self.memory_attention(
                                        single_scale_features.flatten(2).permute(2, 0, 1),
                                        combined_memory_embed,
                                        positional_embed.flatten(2).permute(2, 0, 1),
                                        combined_memory_positional_embed,
                                        scale_idx=scale_idx,
                                    )
                                    .squeeze(0)
                                    .permute(0, 2, 1)
                                    .reshape(single_scale_features.shape[0], -1, *single_scale_features.shape[2:])
                                )
                                if combined_memory_embed.shape[0] > 0
                                else single_scale_features
                                + self.get_dummy_tensor(self.memory_attention, single_scale_features)
                            )
                            for scale_idx, (
                                single_scale_features,
                                combined_memory_embed,
                                positional_embed,
                                combined_memory_positional_embed,
                            ) in enumerate(
                                zip(
                                    multi_scale_features,
                                    multi_scale_combined_memory_embeds,
                                    multi_scale_positional_embeds,
                                    multi_scale_combined_memory_positional_embeds,
                                )
                            )
                        ]
                        if temporal_idx > 0 and self.memory_attention is not None
                        else None
                    )
                # 3. mask decoder
                if self.gradient_checkpointing and self.training:
                    decoder_outputs = self._gradient_checkpointing_func(
                        self.decoder.__call__,
                        memory_attention_outputs or multi_scale_features,
                        mask_features,
                        seg_embeds,
                        token_embeds,
                        token_masks,
                        cond_lens if self.use_repeat_cond else None,
                        output_attentions,
                    )
                else:
                    decoder_outputs = self.decoder(
                        multi_scale_features=memory_attention_outputs or multi_scale_features,
                        mask_features=mask_features,
                        seg_embeds=seg_embeds,
                        token_embeds=token_embeds,
                        token_masks=token_masks,
                        cond_lens=cond_lens if self.use_repeat_cond else None,
                        output_attentions=output_attentions,
                    )
                # 4. encode memory
                (
                    multi_scale_memory_embeds,
                    multi_scale_memory_positional_embeddings,
                ) = (
                    self.encode_memory(
                        multi_scale_features,
                        self.class_predictor(
                            decoder_outputs.intermediate_hidden_states[-1].transpose(0, 1),
                            self._select_temporal_prompt_tensor(cond_embeds, temporal_idx, len(padded_masks)),
                            self._select_temporal_prompt_tensor(embed_masks, temporal_idx, len(padded_masks)),
                        ),
                        decoder_outputs.masks_queries_logits[-1],
                    )
                    if self.memory_encoder is not None
                    else (None, None)
                )
                # 5. store memory with FIFO management (evict oldest if at capacity)
                if self.memory_encoder is not None:
                    if memory_bank and self.num_maskmem > 0 and len(memory_bank) >= self.num_maskmem:
                        memory_bank.pop(0)

                    # update memory
                    memory_bank.append(
                        {
                            "temporal_idx": temporal_idx,
                            "memory_embeds": (
                                [
                                    m.flatten(2).permute(2, 0, 1) if m is not None else None
                                    for m in multi_scale_memory_embeds
                                ]
                                if multi_scale_memory_embeds is not None
                                else None
                            ),
                            "memory_positional_embeddings": (
                                [
                                    p.flatten(2).permute(2, 0, 1) if p is not None else None
                                    for p in multi_scale_memory_positional_embeddings
                                ]
                                if multi_scale_memory_positional_embeddings is not None
                                else None
                            ),
                        }
                    )

                batch_intermediate_hidden_states = (
                    decoder_outputs[0] if not return_dict else decoder_outputs.intermediate_hidden_states
                )
                batch_mask_queries_logits = (
                    decoder_outputs[1] if not return_dict else decoder_outputs.masks_queries_logits
                )
                batch_last_hidden_state = decoder_outputs[2] if not return_dict else decoder_outputs.last_hidden_state
                batch_hidden_states = decoder_outputs[3] if not return_dict else decoder_outputs.hidden_states
                batch_attentions = decoder_outputs[4] if not return_dict else decoder_outputs.attentions

                intermediate_hidden_states.append(
                    tuple(
                        [
                            intermediate_hidden_state[:, batch_padded_masks, ...]
                            for intermediate_hidden_state in batch_intermediate_hidden_states
                        ]
                    )
                )
                mask_queries_logits.append(
                    tuple(
                        [
                            mask_queries_logit[batch_padded_masks, ...]
                            for mask_queries_logit in batch_mask_queries_logits
                        ]
                    )
                )
                last_hidden_state.append(batch_last_hidden_state[batch_padded_masks, ...])
                hidden_states.append(
                    tuple([hidden_state[:, batch_padded_masks, ...] for hidden_state in batch_hidden_states])
                    if batch_hidden_states is not None
                    else None
                )
                attentions.append(
                    tuple([attention[batch_padded_masks, ...] for attention in batch_attentions])
                    if batch_attentions is not None
                    else None
                )

            intermediate_hidden_states = tuple(
                torch.cat(intermediate_hidden_state, dim=1)
                for intermediate_hidden_state in zip(*intermediate_hidden_states)
            )
            masks_queries_logits = tuple(
                torch.cat(mask_queries_logit, dim=0) for mask_queries_logit in zip(*mask_queries_logits)
            )
            last_hidden_state = torch.cat(last_hidden_state, dim=0)
            hidden_states = (
                tuple(torch.cat(hidden_state, dim=1) for hidden_state in zip(*hidden_states))
                if hidden_states[0] is not None
                else None
            )
            attentions = (
                tuple(torch.cat(attention, dim=0) for attention in zip(*attentions))
                if attentions[0] is not None
                else None
            )
        else:
            raise ValueError(f"Unsupported encoder and decoder type: {type(self.encoder)} and {type(self.decoder)}")

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in intermediate_hidden_states:
            # class_predition shape: [batch_size, num_queries, num_classes]
            class_logits = self.class_predictor(decoder_output.transpose(0, 1), cond_embeds, embed_masks)
            class_queries_logits += (class_logits,)

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and self.training:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                cond_ids=cond_ids,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)
        else:
            class_queries_logits = tuple([class_queries_logits[-1]])
            masks_queries_logits = tuple(
                (
                    self.postprocess_masks_preds([masks_queries_logits[-1]])
                    if masks_queries_logits[-1] is not None
                    else None
                )
            )

        output_auxiliary_logits = (
            self.dec_config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = XSegmentorOutput(
            loss=loss,
            loss_dict=loss_dict,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            decoder_last_hidden_state=last_hidden_state,
            decoder_hidden_states=hidden_states,
            decoder_attentions=attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss) + output
        return output
