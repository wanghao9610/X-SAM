from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all([isinstance(t, torch.Tensor) for t in x]), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


@torch.jit.script_if_tracing
def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst.device)


def pad_temporal_tensors(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0,
) -> torch.Tensor:
    if isinstance(tensors, torch.Tensor):
        return tensors

    assert len(tensors) > 0
    assert isinstance(tensors, (tuple, list))
    for t in tensors:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[-2:] == tensors[0].shape[-2:], t.shape

    max_size = max([t.shape[0] for t in tensors])
    batch_shape = [len(tensors), max_size, *tensors[0].shape[1:]]
    padded_tensors = tensors[0].new_full(batch_shape, pad_value)
    padded_masks = torch.zeros((len(tensors), max_size), dtype=torch.bool)
    for i, t in enumerate(tensors):
        padded_masks[i, : t.shape[0]] = True
        padded_tensors[i, : t.shape[0], ...].copy_(t)
    return padded_tensors.transpose(0, 1), padded_masks.transpose(0, 1)


def pad_spatial_tensors(
    tensors: List[torch.Tensor],
    size_divisibility: int = 0,
    pad_value: float = 0.0,
    padding_constraints: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    Args:
        tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
            (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
            to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad.
        padding_constraints (optional[Dict]): If given, it would follow the format as
            {"size_divisibility": int, "square_size": int}, where `size_divisibility` will
            overwrite the above one if presented and `square_size` indicates the
            square padding size if `square_size` > 0.
    Returns:
        an `ImageList`.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors

    assert len(tensors) > 0
    assert isinstance(tensors, (tuple, list))
    for t in tensors:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

    image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
    image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
    max_size = torch.stack(image_sizes_tensor).max(0).values

    if padding_constraints is not None:
        square_size = padding_constraints.get("square_size", 0)
        if square_size > 0:
            # pad to square.
            max_size[0] = max_size[1] = square_size
        if "size_divisibility" in padding_constraints:
            size_divisibility = padding_constraints["size_divisibility"]
    if size_divisibility > 1:
        stride = size_divisibility
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

    # handle weirdness of scripting and tracing ...
    if torch.jit.is_scripting():
        max_size: List[int] = max_size.to(dtype=torch.long).tolist()
    else:
        if torch.jit.is_tracing():
            image_sizes = image_sizes_tensor

    if len(tensors) == 1:
        # This seems slightly (2%) faster.
        # TODO: check whether it's faster for multiple images as well
        image_size = image_sizes[0]
        padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
        batched_tensors = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
    else:
        # max_size can be a tensor in tracing mode, therefore convert to list
        batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
        device = None if torch.jit.is_scripting() else ("cpu" if torch.jit.is_tracing() else None)
        batched_tensors = tensors[0].new_full(batch_shape, pad_value, device=device)
        batched_tensors = move_device_like(batched_tensors, tensors[0])
        for i, img in enumerate(tensors):
            # Use `batched_tensors` directly instead of `img, pad_img = zip(tensors, batched_tensors)`
            # Tracing mode cannot capture `copy_()` of temporary locals
            batched_tensors[i, ..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_tensors


def pad_tensors(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0,
    size_divisibility: int = 0,
    padding_constraints: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    assert isinstance(tensors, (torch.Tensor, list))
    if isinstance(tensors, torch.Tensor) and tensors.ndim == 5:
        # [B, T, C, H, W] -> [T, B, C, H, W]
        return tensors.transpose(0, 1), torch.ones((tensors.shape[1], tensors.shape[0]), dtype=torch.bool)
    elif isinstance(tensors, torch.Tensor) and tensors.ndim == 4:
        # [B, C, H, W] -> [1, B, C, H, W]
        return tensors[None, ...], torch.ones((1, tensors.shape[0]), dtype=torch.bool)
    elif isinstance(tensors, List) and tensors[0].ndim == 3:
        # [[C, H, W], [C, H, W], ...] -> [1, B, C, H, W]
        return pad_spatial_tensors(
            tensors, size_divisibility=size_divisibility, pad_value=pad_value, padding_constraints=padding_constraints
        )[None, ...], torch.ones((1, len(tensors)), dtype=torch.bool)
    elif isinstance(tensors, List) and tensors[0].ndim == 4:
        # [[T_0, C, H, W], [T_1, C, H, W], ..., [T_B, C, H, W]] -> [T_max, B, C, H, W]
        return pad_temporal_tensors(tensors, pad_value=pad_value)
    else:
        raise ValueError(f"Unsupported tensor input: {type(tensors[0])} with shape {tensors[0].shape}")
