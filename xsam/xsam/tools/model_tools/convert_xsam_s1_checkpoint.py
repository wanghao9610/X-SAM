"""Convert X-SAM s1 seg-finetune checkpoint to xsam s1_train format.

X-SAM (close_cls=True) stores class head weights at:
  segmentor.class_predictor.{weight,bias}  -> shape [num_labels+1, hidden_dim]

xsam (head_cls_type="linear") stores them at:
  segmentor.class_predictor.predictor.{weight,bias}  -> shape [num_labels+1, hidden_dim]

num_labels is inferred automatically:
  - old: from the legacy class_predictor tensor shape in the source checkpoint
  - new: from model.segmentor.decoder.config.num_labels in --config (default 133)

When num_labels changes (e.g. 133 -> 1 for binary cls), the class head is remapped:
  - foreground row: mean of all thing-class rows from the old checkpoint
  - no-object row: last row from the old checkpoint (standard Mask2Former no-object class)
"""

import argparse
import os
import os.path as osp
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from xsam.registry import BUILDER
from xsam.utils.checkpoint import guess_load_checkpoint


def get_num_labels_from_config(config_path: str) -> int:
    cfg = Config.fromfile(config_path)
    try:
        return cfg.model["segmentor"]["decoder"]["config"]["num_labels"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Cannot read num_labels from {config_path}. "
            "Expected cfg.model.segmentor.decoder.config.num_labels."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Convert X-SAM s1 checkpoint to xsam format")
    parser.add_argument(
        "--src",
        required=True,
        help="Source pytorch_model.bin from X-SAM s1_seg_finetune",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Output path for converted pytorch_model.bin",
    )
    parser.add_argument(
        "--config",
        default="xsam/configs/xsam/s1_train/xsam_sam_large_m2f_e36_gpu16.py",
        help="xsam config; num_labels is read from this file unless --new-num-labels is set",
    )
    parser.add_argument(
        "--new-num-labels",
        type=int,
        default=None,
        help="Override target num_labels (default: read from --config)",
    )
    parser.add_argument(
        "--skip-class-predictor",
        action="store_true",
        help="Drop class_predictor weights instead of remapping them",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Build model from config and report load compatibility after conversion",
    )
    return parser.parse_args()


def convert_class_predictor(
    state_dict: dict,
    new_num_labels: int,
    skip: bool = False,
) -> tuple[dict, list[str]]:
    old_weight_key = "segmentor.class_predictor.weight"
    old_bias_key = "segmentor.class_predictor.bias"
    new_weight_key = "segmentor.class_predictor.predictor.weight"
    new_bias_key = "segmentor.class_predictor.predictor.bias"

    logs = []
    if old_weight_key not in state_dict:
        logs.append("No legacy class_predictor weights found; skipping class head conversion.")
        return state_dict, logs

    old_weight = state_dict.pop(old_weight_key)
    old_bias = state_dict.pop(old_bias_key)
    old_num_labels = old_weight.shape[0] - 1
    logs.append(
        f"Detected old class_predictor num_labels={old_num_labels} "
        f"(shape [{old_weight.shape[0]}, {old_weight.shape[1]}])"
    )

    if skip:
        logs.append("Skipped class_predictor conversion (--skip-class-predictor).")
        return state_dict, logs

    new_dim = new_num_labels + 1
    if old_num_labels == new_num_labels and old_weight.shape[0] == new_dim:
        state_dict[new_weight_key] = old_weight
        state_dict[new_bias_key] = old_bias
        logs.append(
            f"Renamed class_predictor: [{old_weight.shape[0]}, {old_weight.shape[1]}] "
            f"(num_labels={old_num_labels}, direct copy)"
        )
        return state_dict, logs

    new_weight = torch.zeros(new_dim, old_weight.shape[1], dtype=old_weight.dtype)
    new_bias = torch.zeros(new_dim, dtype=old_bias.dtype)

    if new_num_labels == 1:
        new_weight[0] = old_weight[:old_num_labels].mean(dim=0)
        new_bias[0] = old_bias[:old_num_labels].mean(dim=0)
        new_weight[-1] = old_weight[old_num_labels]
        new_bias[-1] = old_bias[old_num_labels]
        remap_desc = "foreground=mean of thing classes, no-object=last row"
    else:
        raise ValueError(
            f"Unsupported class_predictor remap: old_num_labels={old_num_labels}, "
            f"new_num_labels={new_num_labels}. "
            "Only same-label rename or 133->1 binary remap is supported."
        )

    state_dict[new_weight_key] = new_weight
    state_dict[new_bias_key] = new_bias
    logs.append(
        f"Remapped class_predictor: [{old_num_labels + 1}, {old_weight.shape[1]}] -> "
        f"[{new_dim}, {old_weight.shape[1]}] ({remap_desc})"
    )
    return state_dict, logs


def compare_with_model(state_dict: dict, config_path: str) -> dict:
    cfg = Config.fromfile(config_path)
    init_default_scope("xsam")
    model = BUILDER.build(cfg.model)
    model_sd = model.state_dict(full_model=True)

    matched, shape_mismatch, missing_in_ckpt, extra_in_ckpt = [], [], [], []
    for k, v in model_sd.items():
        if k not in state_dict:
            missing_in_ckpt.append((k, tuple(v.shape)))
        elif state_dict[k].shape != v.shape:
            shape_mismatch.append((k, tuple(v.shape), tuple(state_dict[k].shape)))
        else:
            matched.append(k)

    for k in state_dict:
        if k not in model_sd:
            extra_in_ckpt.append((k, tuple(state_dict[k].shape)))

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

    new_num_labels = args.new_num_labels
    if new_num_labels is None:
        new_num_labels = get_num_labels_from_config(args.config)
    source = "--new-num-labels" if args.new_num_labels is not None else args.config
    print(f"Target num_labels={new_num_labels} (from {source})")

    state_dict, logs = convert_class_predictor(
        state_dict,
        new_num_labels=new_num_labels,
        skip=args.skip_class_predictor,
    )
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
        print(f"  Missing in ckpt: {len(report['missing_in_ckpt'])}")
        print(f"  Extra in ckpt: {len(report['extra_in_ckpt'])}")

        if report["shape_mismatch"]:
            print("\n  Shape mismatches:")
            for item in report["shape_mismatch"]:
                print(f"    {item}")

        if report["missing_in_ckpt"]:
            print("\n  Missing in ckpt (auto-initialized buffers are OK):")
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
    import xsam  # noqa: F401

    main()
