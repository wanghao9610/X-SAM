#!/usr/bin/env python

import argparse
import os
import os.path as osp
import re
import traceback
import warnings
from typing import List

import mmcv
import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.utils.process import sem_seg_postprocess
from xsam.registry import BUILDER
from xsam.utils.configs import cfgs_name_path
from xsam.utils.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_PLACEHOLDER_TOKEN, INDEX2TOKEN
from xsam.utils.dist import setup_distributed
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function, set_model_resource, split_list

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore model predictions")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--output-dir", help="directory to save logs and explorations")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint for exploration",
    )
    parser.add_argument("--subset", type=str, default="train", choices=["train", "val"], help="subset to explore")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--max-samples", type=int, default=200, help="maximum samples to explore")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    return parser.parse_args()


def get_phrases_ids(input_ids: torch.Tensor, pstart_token_idx: int, pend_token_idx: int) -> List[torch.Tensor]:
    """Extract phrase IDs from input IDs using start and end tokens."""
    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases_ids = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrases_ids.append(input_ids[ps + 1 : pe - 1])
    return phrases_ids


def decode_phrases_ids(tokenizer, phrases_ids: List[torch.Tensor]) -> List[str]:
    """Decode phrase IDs to text."""
    phrases = []
    for phrase_id in phrases_ids:
        if (phrase_id < 0).any():
            phrase = ""
        else:
            phrase = tokenizer.decode(phrase_id).strip()
        phrases.append(phrase)
    return phrases


def decode_input_ids(tokenizer, input_ids: List[torch.Tensor]) -> List[str]:
    input_ids = split_list(input_ids, INDEX2TOKEN.keys())
    text = ""
    for ids in input_ids:
        if len(ids) == 1 and ids[0] in INDEX2TOKEN:
            text += INDEX2TOKEN[ids[0]]
        else:
            text += tokenizer.decode(ids)

    text = re.sub(f"({re.escape(DEFAULT_PLACEHOLDER_TOKEN)}\\s*)+", DEFAULT_IMAGE_TOKEN, text)

    return text


def explore_dataset(
    dataset,
    visualizer,
    output_dir: str,
    max_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
) -> None:
    """Explore model predictions on a single dataset."""
    data_name = dataset.data_name
    tokenizer = dataset.tokenizer

    # Setup dataloader
    sampler = DistributedSampler(dataset=dataset, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, sampler=sampler, collate_fn=xsam_collate_fn
    )

    os.makedirs(output_dir, exist_ok=True)

    # Exploration loop
    exp_cnt = 0
    with tqdm(total=max_samples, desc=f"Exploring {data_name}", disable=rank != 0) as pbar:
        for data in dataloader:
            if exp_cnt >= max_samples:
                break

            input_ids = data["data_dict"].get("input_ids", None)
            vprompt_masks = data["data_dict"].get("vprompt_masks", None)
            input_text = decode_input_ids(tokenizer, input_ids[0].tolist())
            phrases_ids = get_phrases_ids(input_ids[0], dataset.pstart_token_idx, dataset.pend_token_idx)
            phrases = decode_phrases_ids(tokenizer, phrases_ids)
            mask_labels = data["data_samples"].mask_labels
            class_labels = data["data_samples"].class_labels
            scaled_sizes = data["data_samples"].scaled_sizes
            image_sizes = data["data_samples"].image_sizes

            if image_sizes is not None and isinstance(image_sizes[0][0], int):
                image_sizes = [image_sizes]
            if scaled_sizes is not None and isinstance(scaled_sizes[0][0], int):
                scaled_sizes = [scaled_sizes]
            if mask_labels is not None and mask_labels[0].ndim == 3:
                mask_labels = [mask_label[None] for mask_label in mask_labels]
            if class_labels is not None and class_labels[0].ndim == 1:
                class_labels = [class_label[None] for class_label in class_labels]

            input_infos = data["data_samples"].metainfo["image_infos"]
            vprompt_indices = getattr(data["data_samples"], "vprompt_indices", [0])
            input_texts = []

            for i, (input_info, image_size, scaled_size, mask_label, class_label) in enumerate(
                zip(input_infos, image_sizes, scaled_sizes, mask_labels, class_labels)
            ):
                file_name = (
                    [input_info["file_name"]] if "file_name" in input_info else input_info.get("file_names", None)
                )
                extra_file_name = (
                    [input_info["extra_file_name"]]
                    if "extra_file_name" in input_info
                    else input_info.get("extra_file_names", None)
                )
                file_name = extra_file_name if extra_file_name is not None else file_name

                mask_label = mask_label if mask_label.ndim == 4 else mask_label[None]
                class_label = class_label if class_label.ndim == 2 else class_label[None]
                sample_id = input_info.get("sample_id", "")

                vprompt_mask = None
                vprompt_image = None
                if vprompt_masks is not None:
                    vprompt_mask = vprompt_masks[i]
                    vprompt_index = vprompt_indices[i]
                    vprompt_image = mmcv.imread(osp.join(dataset.image_folder, file_name[vprompt_index]))
                    vprompt_image = mmcv.imconvert(vprompt_image, "bgr", "rgb")
                    phrases = [f"vprompt_{i}" for i in range(vprompt_mask.shape[0])]

                for _file_name, _mask_label, _class_label, _scaled_size, _image_size in zip(
                    file_name, mask_label, class_label, scaled_size, image_size
                ):
                    input_texts.append((_file_name, input_text))
                    _mask_label, _class_label = (
                        _mask_label[_class_label != dataset.ignore_label],
                        _class_label[_class_label != dataset.ignore_label],
                    )
                    image = mmcv.imread(osp.join(dataset.image_folder, _file_name))
                    image = mmcv.imconvert(image, "bgr", "rgb")
                    _mask_label = sem_seg_postprocess(
                        _mask_label[None, ...], _scaled_size, _image_size[0], _image_size[1], mode="nearest"
                    )
                    vprompt_mask = (
                        sem_seg_postprocess(
                            vprompt_mask[None, ...], _scaled_size, _image_size[0], _image_size[1], mode="nearest"
                        )
                        if vprompt_mask is not None
                        else None
                    )

                    try:
                        visualizer.draw_labels(
                            image,
                            aux_img_rgb=vprompt_image,
                            mask_labels=_mask_label,
                            class_labels=_class_label,
                            class_names=phrases,
                            vprompt_masks=vprompt_mask,
                            output_file=osp.join(
                                output_dir, f"{osp.splitext(osp.basename(_file_name))[0]}{sample_id}.png"
                            ),
                            **input_info,
                        )
                    except Exception as e:
                        print_log(
                            f"Error exploring {_file_name}: {e}\n{traceback.format_exc()}",
                            logger="current",
                        )
                        continue

                    exp_cnt += 1
                    pbar.update(1)

                input_texts = list(dict((item[1], item) for item in input_texts).values())
                with open(osp.join(output_dir, "input_texts.txt"), "w") as f:
                    for input_text in input_texts:
                        f.write(f"{input_text[0]}: \n{input_text[1]}\n")


def main():
    """Main exploration function."""
    args = parse_args()
    rank, _, world_size = setup_distributed(args, kwargs={"timeout": 18000})

    # Load and process config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.seed is not None:
        # Use args.seed
        set_random_seed(args.seed)
        print_log(
            f"Set the random seed to {args.seed}.",
            logger="current",
        )
    register_function(cfg._cfg_dict)

    # Explore all datasets
    if args.subset == "train":
        dataset_cfgs = cfg.train_datasets.datasets
    elif args.subset == "val":
        dataset_cfgs = cfg.val_datasets
    else:
        raise ValueError(f"Invalid subset: {args.subset}")
    print_log(f"Exploring {len(dataset_cfgs)} datasets...", logger="current")
    for dataset_cfg in dataset_cfgs:
        try:
            output_dir = osp.join(args.output_dir, dataset_cfg.data_name)
            if osp.exists(output_dir):
                continue

            dataset_cfg.output_ids_with_output = True
            if "template_map_fn" in dataset_cfg and hasattr(dataset_cfg.template_map_fn, "output_suffix"):
                dataset_cfg.template_map_fn.output_suffix = True
            dataset = BUILDER.build(dataset_cfg)
            visualizer = BUILDER.build(cfg.visualizer)

            explore_dataset(
                dataset,
                visualizer,
                output_dir,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
            )
        except Exception as e:
            print_log(f"Error exploring {dataset_cfg.data_name}: {e}\n{traceback.format_exc()}", logger="current")
            continue


if __name__ == "__main__":
    main()
