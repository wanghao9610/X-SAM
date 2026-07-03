import argparse
import os.path as osp

import torch
from mmengine.config import Config

from xsam.registry import BUILDER
from xsam.utils.configs import cfgs_name_path
from xsam.utils.dszero import convert_zero_checkpoint_to_state_dict, get_state_dict_from_zero_checkpoint
from xsam.utils.state_dict import merge_partial_state_dict_into_model
from xsam.utils.utils import register_function, set_model_resource

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _resolve_tag(work_dir, tag_dir=None):
    if tag_dir is not None:
        return tag_dir
    path = osp.join(work_dir, "last_checkpoint")
    if not osp.isfile(path):
        raise ValueError(f"Last checkpoint file {path} does not exist")
    with open(path) as f:
        return f.read().strip()


def _build_model(config_path):
    if not osp.isfile(config_path):
        try:
            config_path = cfgs_name_path[config_path]
        except KeyError as exc:
            raise FileNotFoundError(f"Cannot find {config_path}") from exc
    cfg = Config.fromfile(config_path)
    set_model_resource(cfg)
    register_function(cfg._cfg_dict)
    model = BUILDER.build(cfg.model)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--tag-dir", default=None)
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), default="fp16")
    parser.add_argument(
        "--config",
        default=None,
        help="Build the full model and merge ZeRO weights (keeps frozen/pretrained params).",
    )
    args = parser.parse_args()

    output_file = osp.join(args.work_dir, "pytorch_model.bin")
    dtype = DTYPE_MAP[args.dtype]

    if args.config is None:
        tag = _resolve_tag(args.work_dir, args.tag_dir)
        convert_zero_checkpoint_to_state_dict(args.work_dir, output_file, tag=tag, dtype=dtype)
        return

    try:
        tag = _resolve_tag(args.work_dir, args.tag_dir)
        partial = get_state_dict_from_zero_checkpoint(args.work_dir, tag=tag, dtype=dtype)
    except ValueError:
        if not osp.isfile(output_file):
            raise
        print(f"No ZeRO checkpoint found. Loading partial state dict from {output_file}")
        ckpt = torch.load(output_file, map_location="cpu", weights_only=False)
        partial = ckpt.get("state_dict", ckpt)

    model = _build_model(args.config)
    n_ckpt = len(partial)
    state_dict, missing = merge_partial_state_dict_into_model(model, partial)
    print(
        f"Merged {n_ckpt} checkpoint tensors; "
        f"{len(missing)} tensors kept from initialized/pretrained model."
    )
    print(f"Saving full model state dict to {output_file}")
    torch.save(state_dict, output_file)


if __name__ == "__main__":
    main()
