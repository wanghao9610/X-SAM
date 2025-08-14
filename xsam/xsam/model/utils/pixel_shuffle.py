import torch.nn as nn


def maybe_pad(input_feature, height=None, width=None, data_format="channels_last"):
    # expeacted data format: N, H, W, C
    if data_format == "channels_first":
        input_feature = input_feature.permute(0, 2, 3, 1).contiguous()

    if height is None or width is None:
        height, width = input_feature.shape[1:-1]

    should_pad = (height % 2 == 1) or (width % 2 == 1)
    if should_pad:
        pad_values = (0, 0, 0, width % 2, 0, height % 2)
        input_feature = nn.functional.pad(input_feature, pad_values)

    if data_format == "channels_first":
        input_feature = input_feature.permute(0, 3, 1, 2).contiguous()

    return input_feature


def pixel_shuffle(x, scale_factor=0.5, data_format="channels_last"):
    # expected data format: B, H, W, C
    if data_format == "channels_first":
        x = x.permute(0, 2, 3, 1).contiguous()

    B, H, W, C = x.size()
    # B, H, W, C --> B, H, W * scale, C // scale
    x = x.reshape(B, H, int(W * scale_factor), int(C // scale_factor))
    # B, H, W * scale, C // scale --> B, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # B, H * scale, W, C // scale --> B, H * scale, W * scale, C // (scale ** 2)
    x = x.view(
        B,
        int(W * scale_factor),
        int(H * scale_factor),
        int(C // (scale_factor * scale_factor)),
    )
    x = x.permute(0, 2, 1, 3).contiguous()

    if data_format == "channels_first":
        x = x.permute(0, 3, 1, 2).contiguous()

    return x
