import os.path as osp

from mmengine.fileio import PetrelBackend, get_file_backend
from xtuner.model.utils import guess_load_checkpoint

from xsam.utils.logging import print_log


def load_checkpoint(model, pth_model: str) -> None:
    """Load model checkpoint."""
    if not osp.exists(pth_model):
        return

    backend = get_file_backend(pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio

        with patch_fileio():
            state_dict = guess_load_checkpoint(pth_model)
    else:
        state_dict = guess_load_checkpoint(pth_model)

    model.load_state_dict(state_dict, strict=True)
    matched_keys = [k for k in state_dict.keys() if k in model.state_dict().keys()]
    print_log(f"Load checkpoint from {pth_model}", logger="current")
    print_log(f"Matched keys: {len(matched_keys)} / {len(state_dict.keys())}", logger="current")
