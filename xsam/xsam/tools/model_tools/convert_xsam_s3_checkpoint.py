"""Convert X-SAM s3 mixed-finetune checkpoint to xsam s3_train format.

Key differences between X-SAM and xsam state dicts for this model family:

1. SigLIP vision tower prefix:
     visual_encoder.*  ->  vision_encoder.*

2. Open-vocabulary class head (open_cls / head_cls_type="learn"):
     segmentor.logit_scale  ->  segmentor.class_predictor.logit_scale

Weights that exist only in xsam (text cross-attn, n_points_scale buffers, etc.)
are left to model initialization when loading the converted checkpoint.

Usage:
  python -m xsam.tools.model_tools.convert_xsam_s3_checkpoint \
    --src /path/to/X-SAM/s3_mixed_finetune/.../pytorch_model.bin \
    --dst /path/to/converted/pytorch_model.bin \
    --validate \
    --config xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from xsam.registry import BUILDER
from xsam.utils.checkpoint import guess_load_checkpoint

OLD_LOGIT_SCALE_KEY = "segmentor.logit_scale"
NEW_LOGIT_SCALE_KEY = "segmentor.class_predictor.logit_scale"
OLD_VISION_PREFIX = "visual_encoder."
NEW_VISION_PREFIX = "vision_encoder."


def parse_args():
    parser = argparse.ArgumentParser(description="Convert X-SAM s3 checkpoint to xsam format")
    parser.add_argument(
        "--src",
        required=True,
        help="Source pytorch_model.bin from X-SAM s3_mixed_finetune",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Output path for converted pytorch_model.bin",
    )
    parser.add_argument(
        "--config",
        default="xsam/configs/xsam/s3_train/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_vit_large_m2f_e1_gpu16.py",
        help="xsam config used for optional --validate",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Build model from config and report load compatibility after conversion",
    )
    return parser.parse_args()


def convert_state_dict(state_dict: dict) -> tuple[dict, list[str]]:
    converted = {}
    logs = []

    vision_renamed = 0
    for key, value in state_dict.items():
        new_key = key
        if key.startswith(OLD_VISION_PREFIX):
            new_key = NEW_VISION_PREFIX + key[len(OLD_VISION_PREFIX) :]
            vision_renamed += 1
        elif key == OLD_LOGIT_SCALE_KEY:
            new_key = NEW_LOGIT_SCALE_KEY

        if new_key in converted:
            raise KeyError(f"Duplicate target key after conversion: {new_key}")
        converted[new_key] = value

    logs.append(
        f"Renamed {vision_renamed} keys: {OLD_VISION_PREFIX}* -> {NEW_VISION_PREFIX}*"
    )
    if OLD_LOGIT_SCALE_KEY in state_dict:
        logs.append(
            f"Renamed class head scale: {OLD_LOGIT_SCALE_KEY} -> {NEW_LOGIT_SCALE_KEY}"
        )
    else:
        logs.append(f"No {OLD_LOGIT_SCALE_KEY} found; skipped logit_scale rename.")

    return converted, logs


def compare_with_model(state_dict: dict, config_path: str) -> dict:
    cfg = Config.fromfile(config_path)
    cfg.model.s1_pretrained_pth = None
    cfg.model.s2_pretrained_pth = None
    init_default_scope("xsam")
    model = BUILDER.build(cfg.model)
    model_sd = model.state_dict()

    matched, shape_mismatch, missing_in_ckpt, extra_in_ckpt = [], [], [], []
    for key, value in model_sd.items():
        if key not in state_dict:
            missing_in_ckpt.append((key, tuple(value.shape)))
        elif state_dict[key].shape != value.shape:
            shape_mismatch.append(
                (key, tuple(value.shape), tuple(state_dict[key].shape))
            )
        else:
            matched.append(key)

    for key in state_dict:
        if key not in model_sd:
            extra_in_ckpt.append((key, tuple(state_dict[key].shape)))

    return {
        "matched": matched,
        "shape_mismatch": shape_mismatch,
        "missing_in_ckpt": missing_in_ckpt,
        "extra_in_ckpt": extra_in_ckpt,
    }


def main():
    args = parse_args()
    src = osp.abspath(args.src)
    dst = osp.abspath(args.dst)

    if not osp.isfile(src):
        raise FileNotFoundError(f"Source checkpoint not found: {src}")

    state_dict = guess_load_checkpoint(src)
    print(f"Loaded {len(state_dict)} keys from {src}")

    state_dict, logs = convert_state_dict(state_dict)
    for line in logs:
        print(line)

    os.makedirs(osp.dirname(dst) or ".", exist_ok=True)
    torch.save(state_dict, dst)
    print(f"Saved converted checkpoint to {dst} ({len(state_dict)} keys)")

    if args.validate:
        report = compare_with_model(state_dict, args.config)
        print(f"\nValidation against {args.config}:")
        print(f"  Matched: {len(report['matched'])}")
        print(f"  Shape mismatch: {len(report['shape_mismatch'])}")
        print(f"  Missing in ckpt (xsam-only modules): {len(report['missing_in_ckpt'])}")
        print(f"  Extra in ckpt: {len(report['extra_in_ckpt'])}")

        if report["shape_mismatch"]:
            print("\n  Shape mismatches:")
            for item in report["shape_mismatch"]:
                print(f"    {item}")

        if report["missing_in_ckpt"]:
            print("\n  Missing in ckpt (expected for new xsam modules):")
            for item in report["missing_in_ckpt"]:
                print(f"    {item}")

        if report["extra_in_ckpt"]:
            print("\n  Extra in ckpt:")
            for item in report["extra_in_ckpt"]:
                print(f"    {item}")


if __name__ == "__main__":
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if "INIT_DIR" not in os.environ:
        default_init_dir = osp.abspath(osp.join(repo_root, "..", "inits"))
        if osp.isdir(default_init_dir):
            os.environ["INIT_DIR"] = default_init_dir + osp.sep
    import xsam  # noqa: F401

    main()
