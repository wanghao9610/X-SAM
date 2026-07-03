#!/usr/bin/env python


import argparse
import logging
import os
import os.path as osp
import traceback
import warnings
from typing import Dict, List, Optional, Tuple

import mmcv
import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.registry import BUILDER
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.colormap import random_color
from xsam.utils.config import setup_model_config
from xsam.utils.configs import cfgs_name_path
from xsam.utils.device import get_device
from xsam.utils.dist import setup_distributed
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function, set_model_resource

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and visualizations")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint for visualization",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="directory to save visualizations")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--max-samples", type=int, default=200, help="maximum samples to visualize")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--rerun", action="store_true", help="rerun the visualization")
    parser.add_argument("--concat-aux-img", action="store_true", help="concat auxiliary image with main image")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher type",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
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


def process_batch(
    model,
    data: Dict,
    data_name: str,
    metadata: Dict,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    mode: str = "tensor",
) -> Tuple[bool, Optional[torch.Tensor], List[str], str]:
    """Process a single batch of data.

    Args:
        model: The model to evaluate
        data: Input data dictionary
        data_name: Name of the dataset
        generation_config: Generation configuration for LLM
        stop_criteria: Stopping criteria for LLM
        mode: Mode of the model

    Returns:
        Tuple of (success status, segmentation outputs, phrases, llm_generation_output)
    """
    data_samples = data["data_samples"]
    image_files = data_samples.image_files

    data_dict = {
        "input_ids": data["data_dict"].get("input_ids", None),
        "attention_mask": data["data_dict"].get("attention_mask", None),
        "pixel_values": data["data_dict"].get("pixel_values", None),
        "extra_pixel_values": data["data_dict"].get("extra_pixel_values", None),
        "image_grid_thw": data["data_dict"].get("image_grid_thw", None),
        "cond_ids": data["data_dict"].get("cond_ids", None),
        "seg_ids": data["data_dict"].get("seg_ids", None),
        "vprompt_masks": data["data_dict"].get("vprompt_masks", None),
    }

    llm_question_input = ""
    if data_dict["input_ids"] is not None:
        _input_ids = data_dict["input_ids"]
        llm_question_input = model.tokenizer.decode(_input_ids[_input_ids > 0])

    data_dict = data_dict_to_device(data_dict, device=model.device, dtype=model.dtype)

    # Get input phrases
    input_phrases = []
    if hasattr(model, "pstart_token_idx") and hasattr(model, "pend_token_idx"):
        input_phrases_ids = get_phrases_ids(data_dict["input_ids"][0], model.pstart_token_idx, model.pend_token_idx)
        input_phrases = decode_phrases_ids(model.tokenizer, input_phrases_ids)

    with torch.no_grad():
        llm_outputs, seg_outputs = model(
            data_dict,
            data_samples,
            mode=mode,
            generation_config=generation_config,
            stopping_criteria=stop_criteria,
            metadata=metadata,
            do_postprocess=True,
            do_loss=False,
        )

    # Process outputs
    llm_generation_output = ""
    output_phrases = []
    if "gcg" in data_name and llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        output_ids = llm_outputs.sequences
        llm_generation_output = model.tokenizer.batch_decode(output_ids)[0]

        if hasattr(model, "pstart_token_idx") and hasattr(model, "pend_token_idx"):
            output_phrases_ids = get_phrases_ids(output_ids[0], model.pstart_token_idx, model.pend_token_idx)
            output_phrases = decode_phrases_ids(model.tokenizer, output_phrases_ids)

    if seg_outputs is None:
        print_log(
            rf"Failed to get segmentation outputs: {image_files}, "
            rf"llm question_input: {repr(llm_question_input)}, "
            rf"llm generation_output: {repr(llm_generation_output)}",
            logger="current",
        )
        return False, None, [], llm_generation_output

    phrases = output_phrases or input_phrases
    return True, seg_outputs, phrases, [llm_question_input + "\n" + llm_generation_output]


def visualize_dataset(
    model,
    dataset,
    visualizer,
    output_dir: str,
    max_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    concat_aux_img: Optional[bool] = True,
) -> None:
    """Visualize model predictions on a single dataset."""
    data_name = dataset.data_name
    metadata = dataset.metadata
    output_ids_with_output = dataset.output_ids_with_output
    mode = "tensor" if output_ids_with_output else "predict"

    # Setup dataloader
    sampler = DistributedSampler(dataset=dataset, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, sampler=sampler, collate_fn=xsam_collate_fn
    )

    # Visualization loop
    vis_cnt = 0
    with tqdm(total=max_samples, desc=f"Visualizing {data_name}", disable=rank != 0) as pbar:
        input_texts = []
        for data in dataloader:
            if vis_cnt >= max_samples:
                break

            success, seg_outputs, phrases, text_inputs = process_batch(
                model, data, data_name, metadata, generation_config, stop_criteria, mode
            )
            if not success:
                continue

            input_infos = data["data_samples"].metainfo["image_infos"]
            vprompt_indices = getattr(data["data_samples"], "vprompt_indices", None)
            scaled_sizes = data["data_samples"].scaled_sizes
            image_sizes = data["data_samples"].image_sizes

            if image_sizes is not None and isinstance(image_sizes[0][0], int):
                image_sizes = [image_sizes]
            if scaled_sizes is not None and isinstance(scaled_sizes[0][0], int):
                scaled_sizes = [scaled_sizes]

            for i, (input_info, seg_output, text_input) in enumerate(zip(input_infos, seg_outputs, text_inputs)):
                seg_output = [seg_output] if not isinstance(seg_output, list) else seg_output
                file_name = [input_info["file_name"]] if "file_name" in input_info else input_info["file_names"]
                extra_file_name = (
                    [input_info["extra_file_name"]]
                    if "extra_file_name" in input_info
                    else input_info.get("extra_file_names", None)
                )
                file_name = extra_file_name if extra_file_name is not None else file_name
                sample_id = input_info.get("sample_id", "")
                subdir = sample_id or osp.splitext(osp.basename(file_name[0]))[0]
                os.makedirs(osp.join(output_dir, subdir), exist_ok=True)

                vprompt_image = None
                if vprompt_indices is not None:
                    vprompt_index = vprompt_indices[i]
                    vprompt_image = mmcv.imread(osp.join(dataset.image_folder, file_name[vprompt_index]))
                    vprompt_image = mmcv.imconvert(vprompt_image, "bgr", "rgb")

                if len(phrases) > 0:
                    colors = [random_color(rgb=True, maximum=1) for _ in range(len(phrases))]
                elif seg_output[0].get("vprompt_masks") is not None:
                    vprompt_masks = seg_output[0].get("vprompt_masks")
                    colors = [random_color(rgb=True, maximum=1) for _ in range(len(vprompt_masks))]
                else:
                    colors = [random_color(rgb=True, maximum=1)]
                for _file_name, _seg_output in tqdm(
                    zip(file_name, seg_output), desc=f"{subdir}", disable=rank != 0
                ):
                    input_texts.append((_file_name, text_input))
                    image = mmcv.imread(osp.join(dataset.image_folder, _file_name))
                    image = mmcv.imconvert(image, "bgr", "rgb")
                    output_file = osp.join(
                        output_dir, subdir, f"{osp.splitext(osp.basename(_file_name))[0]}.png"
                    )
                    if "phrases" not in input_info:
                        input_info.update({"phrases": phrases})

                    try:
                        visualizer.draw_predictions(
                            image,
                            aux_img_rgb=vprompt_image,
                            concat_aux_img=concat_aux_img,
                            data_name=data_name,
                            output_file=output_file,
                            colors=colors,
                            **input_info,
                            **_seg_output,
                        )
                    except Exception as e:
                        print_log(
                            f"Error visualizing {osp.join(subdir, _file_name)}: {e}\n{traceback.format_exc()}",
                            logger="current",
                        )
                        continue

                vis_cnt += 1
                pbar.update(1)

            input_texts = list(dict((item[1], item) for item in input_texts).values())
            with open(osp.join(output_dir, "input_texts.txt"), "w", encoding="utf-8") as f:
                for input_text in input_texts:
                    f.write(f"{input_text[0]}: \n {input_text[1]}\n\n")


def main():
    """Main visualization function."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args, kwargs={"timeout": 18000})

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

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        else:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")

    # Build and setup model
    model = BUILDER.build(cfg.model)
    if "llm" in cfg.model:
        model.llm.to(cfg.model.llm.torch_dtype)
    if "vlm" in cfg.model:
        model.vlm.to(cfg.model.vlm.torch_dtype)
    model.eval()
    model = model.to(get_device())

    if args.pth_model is not None:
        load_checkpoint(model, args.pth_model)
    else:
        print_log("No checkpoint provided, using random initialization", logger="current", level=logging.WARNING)
    stop_criteria, generation_config = setup_model_config(model, cfg)

    # Visualize all datasets
    print_log(f"Visualizing {len(cfg.vis_datasets)} datasets...", logger="current")
    for dataset_cfg in cfg.vis_datasets:
        try:
            base_output_dir = args.output_dir or osp.join(args.work_dir, "vis_data")
            output_dir = osp.join(base_output_dir, dataset_cfg.data_name)
            print(f"output_dir: {output_dir}")
            if osp.exists(output_dir) and not args.rerun:
                print_log(f"{dataset_cfg.data_name} is already visualized, skipping...", logger="current")
                continue

            dataset = BUILDER.build(dataset_cfg)
            model.postprocess_fn = dataset.postprocess_fn

            visualizer = BUILDER.build(cfg.visualizer)
            visualizer.metadata = dataset.metadata

            visualize_dataset(
                model,
                dataset,
                visualizer,
                output_dir,
                args.max_samples,
                args.batch_size,
                rank,
                world_size,
                generation_config,
                stop_criteria,
                args.concat_aux_img,
            )
        except Exception as e:
            print_log(f"Error visualizing {dataset_cfg.data_name}: {e}\n{traceback.format_exc()}", logger="current")
            continue


if __name__ == "__main__":
    main()
