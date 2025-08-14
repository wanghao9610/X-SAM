import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel, get_parameter_dtype
from transformers.models.swin import SwinBackbone

from xsam.utils.logging import print_log

from .mask2former import (
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


class XSegmentor(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, encoder: Literal[SamModel, Mask2FormerModel], decoder=None, torch_dtype=torch.float32, reinit_decoder=False, drop_decoder=False, close_cls=False, open_cls=False):  # type: ignore
        PreTrainedModel.__init__(self, encoder.config)

        if isinstance(encoder, SamModel) and decoder is None:
            self.enc_config = encoder.config.vision_config
            self.dec_config = encoder.config.mask_decoder_config
            self.prompt_enc_config = encoder.config.prompt_encoder_config

            self.shared_image_embedding = encoder.shared_image_embedding
            self.encoder = encoder.vision_encoder
            self.prompt_encoder = encoder.prompt_encoder
            self.pixel_decoder = None
            self.decoder = encoder.mask_decoder
        elif isinstance(encoder, SamModel) and isinstance(decoder, Mask2FormerModel):
            self.enc_config = encoder.config.vision_config
            self.dec_config = decoder.config
            self.prompt_enc_config = encoder.config.prompt_encoder_config

            self.shared_image_embedding = encoder.shared_image_embedding
            self.encoder = encoder.vision_encoder
            self.pixel_decoder = decoder.pixel_level_module.decoder
            self.decoder = decoder.transformer_module
        elif isinstance(encoder, Mask2FormerModel) and decoder is None:
            self.enc_config = encoder.config
            self.dec_config = copy.deepcopy(encoder.config)
            self.enc_config.hidden_size = encoder.config.backbone_config.hidden_size

            self.shared_image_embedding = None
            self.encoder = encoder.pixel_level_module.encoder
            self.pixel_decoder = encoder.pixel_level_module.decoder
            self.decoder = encoder.transformer_module
        elif isinstance(encoder, Mask2FormerModel) and isinstance(decoder, SamModel):
            # TODO: check if this is correct
            self.enc_config = encoder.config
            self.dec_config = decoder.config

            self.shared_image_embedding = None
            self.encoder = encoder.pixel_level_module
            self.pixel_decoder = encoder.pixel_level_module.decoder
            self.decoder = decoder.mask_decoder
        else:
            raise ValueError(f"Unsupported encoder and decoder type: {type(encoder)} and {type(decoder)}")

        if drop_decoder:
            self.encoder = self.encoder.to(torch_dtype)
            self.shared_image_embedding = None
            self.prompt_encoder = None
            self.pixel_decoder = None
            self.decoder = None

            return

        if reinit_decoder:
            print_log(f"Reinitializing decoder of {self.decoder.__class__.__name__}.", logger="current")
            # means decoder and pixel_decoder are not from pretained model, so we need to initialize the weights
            self.decoder.apply(self._init_weights)
            if self.pixel_decoder is not None:
                print_log(
                    f"Reinitializing pixel_decoder of {self.pixel_decoder.__class__.__name__}.", logger="current"
                )
                self.pixel_decoder.apply(self._init_weights)

        self.weight_dict: Dict[str, float] = {
            "loss_cls": self.decoder.config.class_weight,
            "loss_mask": self.decoder.config.mask_weight,
            "loss_dice": self.decoder.config.dice_weight,
        }
        self.criterion = Mask2FormerLoss(config=self.decoder.config, weight_dict=self.weight_dict)

        if close_cls:
            self.class_predictor = nn.Linear(self.decoder.config.hidden_dim, self.decoder.config.num_labels + 1).to(
                torch_dtype
            )
        if open_cls:
            self.logit_scale = nn.Parameter(torch.ones([1], dtype=torch_dtype) * np.log(1 / 0.07), requires_grad=True)

        self.close_cls = close_cls
        self.open_cls = open_cls
        self.use_cls = open_cls or close_cls

        self.encoder = self.encoder.to(torch_dtype)
        self.decoder = self.decoder.to(torch_dtype)
        self.pixel_decoder = self.pixel_decoder.to(torch_dtype) if self.pixel_decoder is not None else None
        self.shared_image_embedding = (
            self.shared_image_embedding.to(torch_dtype).requires_grad_(False)
            if self.shared_image_embedding is not None
            else None
        )
        self.criterion = self.criterion.to(torch_dtype)

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

    def get_input_embeddings(self) -> nn.Module:
        if hasattr(self.encoder, "patch_embed"):
            return self.encoder.patch_embed
        elif hasattr(self.encoder, "embeddings"):
            return self.encoder.embeddings.patch_embeddings
        else:
            raise ValueError(f"Unsupported encoder: {type(self.encoder)}")

    def _init_weights(self, module: nn.Module):
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
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)

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

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    @torch.no_grad()
    def get_image_wide_positional_embeddings(self):
        size = self.prompt_enc_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
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

    def get_class_prediction(self, query_embeddings, cond_embeddings, embed_masks=None):
        if cond_embeddings is None:
            return self.class_predictor(query_embeddings)

        query_embeddings = F.normalize(query_embeddings, dim=-1)
        cond_embeddings = F.normalize(cond_embeddings, dim=-1)
        cls_pred = self.logit_scale.exp() * torch.einsum("bqd,bcd->bqc", query_embeddings, cond_embeddings)
        cls_pred = torch.clamp(cls_pred, min=-500, max=500)

        if embed_masks is not None:
            if embed_masks.ndim == 2:
                embed_masks = embed_masks[:, None, :]
            embed_masks = embed_masks.to(torch.bool)
            cls_pred = cls_pred.masked_fill(~embed_masks, -1e9)

        return cls_pred

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

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        image_embeddings: Optional[Tuple[Tensor]] = None,
        seg_embeddings: Optional[Tensor] = None,
        cond_embeddings: Optional[Tensor] = None,
        embed_masks: Optional[Tensor] = None,
        cond_lens: Optional[List] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        output_hidden_states: Optional[bool] = False,
        output_auxiliary_logits: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> XSegmentorOutput:

        # sam_enc + sam_dec
        if isinstance(self.encoder, SamVisionEncoder) and isinstance(self.decoder, SamMaskDecoder):
            if image_embeddings is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # TODO: multi-scale image_embeddings
                image_embeddings = encoder_outputs.last_hidden_state

            image_positional_embeddings = self.get_image_wide_positional_embeddings()
            batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
            image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
            seg_embeddings = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                input_points=None,
                input_labels=None,
                input_boxes=None,
                input_masks=None,
                input_embeds=seg_embeddings,
            )
            decoder_outputs = self.decoder(
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                attention_similarity=None,
                cond_lens=cond_lens,
                target_embedding=None,
                output_attentions=output_attentions,
            )

        # sam_enc + mask2former_dec
        elif isinstance(self.encoder, SamVisionEncoder) and isinstance(self.decoder, Mask2FormerTransformerModule):
            if image_embeddings is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # TODO: multi-scale image_embeddings
                image_embeddings = [encoder_outputs.last_hidden_state] * 4

            pixel_decoder_outputs = self.pixel_decoder(
                image_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_outputs = self.decoder(
                multi_scale_features=pixel_decoder_outputs.multi_scale_features,
                mask_features=pixel_decoder_outputs.mask_features,
                seg_embeddings=seg_embeddings,
                cond_lens=cond_lens,
                output_attentions=output_attentions,
            )
        # mask2former_enc(swin) + mask2former_dec
        elif isinstance(self.encoder, SwinBackbone) and isinstance(self.decoder, Mask2FormerTransformerModule):
            if image_embeddings is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                image_embeddings = encoder_outputs.feature_maps
            pixel_decoder_outputs = self.pixel_decoder(
                image_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_outputs = self.decoder(
                multi_scale_features=pixel_decoder_outputs.multi_scale_features,
                mask_features=pixel_decoder_outputs.mask_features,
                seg_embeddings=seg_embeddings,
                cond_lens=cond_lens,
                output_attentions=output_attentions,
            )
        else:
            raise ValueError(f"Unsupported encoder and decoder type: {type(self.encoder)} and {type(self.decoder)}")

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in decoder_outputs.intermediate_hidden_states:
            # class_predition shape: [batch_size, num_queries, num_classes]
            class_prediction = (
                self.get_class_prediction(decoder_output.transpose(0, 1), cond_embeddings, embed_masks)
                if self.use_cls
                else None
            )
            class_queries_logits += (class_prediction,)

        masks_queries_logits = decoder_outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and self.training:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)
        else:
            masks_queries_logits = (
                self.postprocess_masks_preds(masks_queries_logits) if masks_queries_logits[-1] is not None else None
            )

        output_auxiliary_logits = (
            self.decoder.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = XSegmentorOutput(
            loss=loss,
            loss_dict=loss_dict,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss) + output
        return output
