import logging
import os.path as osp

import torch
from mmengine.fileio import PetrelBackend, get_file_backend

from xsam.utils.logging import print_log


def guess_load_checkpoint(pth_model):
    if osp.isfile(pth_model):
        state_dict = torch.load(pth_model, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    elif osp.isdir(pth_model):
        try:
            from xsam.utils.dszero import get_state_dict_from_zero_checkpoint
        except ImportError:
            raise ImportError(
                "The provided PTH model appears to be a DeepSpeed checkpoint. "
                "However, DeepSpeed library is not detected in current "
                "environment. This suggests that DeepSpeed may not be "
                "installed or is incorrectly configured. Please verify your "
                "setup."
            )
        state_dict = get_state_dict_from_zero_checkpoint(osp.dirname(pth_model), osp.basename(pth_model))
    else:
        raise FileNotFoundError(f"Cannot find {pth_model}")
    return state_dict


def load_checkpoint(model, pth_model: str) -> None:
    """Load model checkpoint."""
    if not osp.exists(pth_model):
        return

    backend = get_file_backend(pth_model)
    if isinstance(backend, PetrelBackend):
        from xsam.utils.fileio import patch_fileio

        with patch_fileio():
            pretrained_state_dict = guess_load_checkpoint(pth_model)
    else:
        pretrained_state_dict = guess_load_checkpoint(pth_model)

    matched_state_dict = {
        k: v
        for k, v in pretrained_state_dict.items()
        if k in model.state_dict().keys() and v.shape == model.state_dict()[k].shape
    }
    matched_keys = [k for k in model.state_dict().keys() if k in matched_state_dict.keys()]
    mismatched_keys = [k for k in pretrained_state_dict.keys() if k not in matched_state_dict.keys()]
    missing_keys = [k for k in model.state_dict().keys() if k not in matched_state_dict.keys()]
    model.load_state_dict(matched_state_dict, strict=False)
    print_log(f"Loaded checkpoint: {pth_model}", logger="current")
    print_log(f"Matched keys: {len(matched_keys)} / {len(pretrained_state_dict.keys())}", logger="current")
    if len(mismatched_keys) > 0:
        print_log(
            f"Mismatched keys: {len(mismatched_keys)} / {len(pretrained_state_dict.keys())}\n{mismatched_keys}",
            logger="current",
            level=logging.WARNING,
        )
    if len(missing_keys) > 0:
        print_log(
            f"Missing keys: {len(missing_keys)} / {len(model.state_dict().keys())}\n{missing_keys}",
            logger="current",
            level=logging.WARNING,
        )
    if len(missing_keys) == 0 and len(mismatched_keys) == 0:
        print_log("All state_dict keys matched!", logger="current")
