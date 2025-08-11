from typing import List

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...segmentors.sam import SamLayerNorm
from ...utils import maybe_pad, pixel_shuffle
from .configuration_connector import ConnectorConfig


class LinearConnectorLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        scale_factor: int,
        hidden_channels: int = 512,
        bias=False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor

        self.model = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * self.scale_factor), bias=bias),
            nn.GELU(),
            nn.Linear(int(in_channels * self.scale_factor), int(in_channels * self.scale_factor), bias=bias),
        )

    def forward(self, x):
        # x: B, H, W, C
        x = maybe_pad(x)
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        x = self.model(x)
        x = x.view(B, H, W, -1)
        x = pixel_shuffle(x, self.scale_factor)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class ConvConnectorLayer(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int, hidden_channels: int = 512, bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor

        # bottle neck
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias),
            SamLayerNorm(hidden_channels, data_format="channels_first"),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            SamLayerNorm(hidden_channels, data_format="channels_first"),
            nn.Conv2d(hidden_channels, int(in_channels * self.scale_factor), kernel_size=1, bias=bias),
            SamLayerNorm(int(in_channels * self.scale_factor), data_format="channels_first"),
        )

    def forward(self, x):
        # x: B, H, W, C
        x = maybe_pad(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.model(x)
        x = pixel_shuffle(x, self.scale_factor, data_format="channels_first")

        return x


class ConnectorModel(PreTrainedModel):
    _auto_class = "AutoModel"
    config_class = ConnectorConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, SamLayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)

        if config.connector_type == "linear":
            self.model = nn.ModuleList(
                [
                    LinearConnectorLayer(
                        config.segmentor_encoder_channels[i],
                        config.scale_factor[i],
                        config.hidden_channels,
                        config.bias,
                    )
                    for i in range(len(config.scale_factor))
                ]
            )
        elif config.connector_type == "conv":
            self.model = nn.ModuleList(
                [
                    ConvConnectorLayer(
                        config.segmentor_encoder_channels[i],
                        config.scale_factor[i],
                        config.hidden_channels,
                        config.bias,
                    )
                    for i in range(len(config.scale_factor))
                ]
            )
        else:
            raise ValueError(f"Unsupported connector type: {config.connector_type}")

        self.post_init()

        self.gradient_checkpointing = False

    def enable_input_require_grads(self):
        def make_inputs_require_grad(module, input, outputs):
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        output.requires_grad_(True)
            else:
                outputs.requires_grad_(True)

        for layer in self.model:
            layer.register_forward_hook(make_inputs_require_grad)

    def forward(self, seg_outputt: List[torch.Tensor]):
        outputs = []

        if self.gradient_checkpointing and self.training:

            def custom_forward(model, seg_outputt_i):
                return model(seg_outputt_i)

            for i, seg_outputt_i in enumerate(seg_outputt):
                output = self._gradient_checkpointing_func(
                    custom_forward, self.model[i], seg_outputt_i, use_reentrant=False
                )
                outputs.append(output)
        else:
            for i, seg_outputt_i in enumerate(seg_outputt):
                outputs.append(self.model[i](seg_outputt_i))

        return outputs
