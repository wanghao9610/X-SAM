from functools import partial
from typing import Callable

import torch


def temporal_process_fn_factory(fn: Callable, **kwargs) -> Callable:
    return partial(fn, **kwargs)


def frame_select_temporal_process_fn(pixel_values: torch.Tensor | tuple[torch.Tensor], select_indx: int = 0) -> torch.Tensor | tuple[torch.Tensor]:
    # pixel_values: [T, B, P, D] -> [B, P, D]
    return pixel_values[select_indx] if isinstance(pixel_values, tuple) else pixel_values[select_indx]

def frame_transpose_temporal_process_fn(pixel_values: torch.Tensor | tuple[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor]:
    # pixel_values: [T, B, P, D] -> [B, T*P, D]
    if isinstance(pixel_values, tuple):
        # tuple of T tensors, each [B, P, D] -> [B, T*P, D]
        return tuple(torch.cat(pixel_values, dim=1))

    # [T, B, P, D] -> [B, T, P, D] -> [B, T*P, D]
    return pixel_values.permute(1, 0, 2, 3).flatten(1, 2)

