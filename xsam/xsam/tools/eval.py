#!/usr/bin/env python

import argparse
import os.path as osp
import re
import traceback
import warnings
from typing import Dict, Optional, Tuple

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.constants import DEFAULT_SEG_TOKEN
from xsam.utils.dist import setup_distributed
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and models")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint for evaluation",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
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


def get_gcg_phrases(input_ids, tokenizer, pstart_token_idx, pend_token_idx):
    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrase_ids = input_ids[ps + 1 : pe - 1]
        if (phrase_ids < 0).any():
            phrase = ""
        else:
            phrase = tokenizer.decode(phrase_ids).strip()
        phrases.append(phrase)
    return phrases


def get_gcg_caption(llm_generation_output):
    if DEFAULT_SEG_TOKEN not in llm_generation_output:
        return ""

    parts = llm_generation_output.split(".")
    sents = [part.strip() for part in parts if DEFAULT_SEG_TOKEN not in part]
    caption = ". ".join(sents)
    caption = re.sub(r"<.*?>", "", caption)
    caption = " ".join(caption.split()).strip("'").strip()
    return caption


def process_batch(
    model,
    data: Dict,
    data_name: str,
    metadata: Dict,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    mode: str = "tensor",
) -> Tuple[bool, Optional[torch.Tensor]]:
    """Process a single batch of data.

    Args:
        model: The model to evaluate
        data: Input data dictionary
        data_name: Name of the dataset
        generation_config: Generation configuration for LLM
        stop_criteria: Stopping criteria for LLM
        mode: Mode of the model

    Returns:
        Tuple of (success status, segmentation outputs)
    """
    data_samples = data["data_samples"]
    image_files = data_samples.image_files

    data_dict = {
        "input_ids": data["data_dict"].get("input_ids", None),
        "pixel_values": data["data_dict"].get("pixel_values", None),
        "extra_pixel_values": data["data_dict"].get("extra_pixel_values", None),
        "cond_ids": data["data_dict"].get("cond_ids", None),
        "seg_ids": data["data_dict"].get("seg_ids", None),
        "vprompt_masks": data["data_dict"].get("vprompt_masks", None),
    }

    llm_question_input = ""
    if data_dict["input_ids"] is not None:
        _input_ids = data_dict["input_ids"]
        llm_question_input = model.tokenizer.decode(_input_ids[_input_ids > 0])

    data_dict = data_dict_to_device(data_dict, device=model.device, dtype=model.dtype)

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

    if seg_outputs is None:
        llm_generation_output = ""
        if llm_outputs is not None and hasattr(llm_outputs, "sequences"):
            llm_generation_output = model.tokenizer.batch_decode(llm_outputs.sequences)

        print_log(
            rf"Failed to get segmentation outputs: {image_files}, "
            rf"llm question_input: {repr(llm_question_input)}, "
            rf"llm generation_output: {repr(llm_generation_output)}",
            logger="current",
        )
        return False, None

    if "gcg" in data_name and llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        llm_generation_output = model.tokenizer.batch_decode(llm_outputs.sequences)
        gcg_phrases = [
            get_gcg_phrases(output_ids, model.tokenizer, model.pstart_token_idx, model.pend_token_idx)
            for output_ids in llm_outputs.sequences
        ]
        gcg_captions = [get_gcg_caption(output) for output in llm_generation_output]
        for i, segmentation_output in enumerate(seg_outputs):
            segmentation_output.update({"gcg_phrases": gcg_phrases[i], "gcg_caption": gcg_captions[i]})

    return True, seg_outputs


def evaluate_dataset(
    model,
    dataset,
    evaluator,
    rank: int,
    world_size: int,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
) -> None:
    """Evaluate model on a single dataset."""
    data_name = evaluator.data_name
    metadata = dataset.metadata
    output_ids_with_output = dataset.output_ids_with_output
    mode = "tensor" if output_ids_with_output else "predict"

    # Setup dataloader
    sampler = DistributedSampler(dataset=dataset, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=4, sampler=sampler, shuffle=False, collate_fn=xsam_collate_fn
    )

    # Evaluation loop
    failed_cnt = 0
    evaluator.reset()
    print_log(f"Evaluating {data_name}...", logger="current")

    for data in tqdm(dataloader, desc=f"Evaluating {data_name}", disable=rank != 0):
        success, seg_outputs = process_batch(model, data, data_name, metadata, generation_config, stop_criteria, mode)
        if not success:
            failed_cnt += 1
            continue

        image_infos = data["data_samples"].metainfo["image_infos"]
        evaluator.process(image_infos, seg_outputs)

    print_log(f"Failed number of {data_name}: {failed_cnt}", logger="current")
    print_log(f"Evaluating {data_name} done!", logger="current")
    evaluator.evaluate()


def main():
    """Main evaluation function."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)

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
    register_function(cfg._cfg_dict)
    if args.seed is not None:
        # Use args.seed
        set_random_seed(args.seed)
        print_log(
            f"Set the random seed to {args.seed}.",
            logger="current",
        )

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
    model.eval()
    model = model.to(get_device())
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank]).module

    load_checkpoint(model, args.pth_model)
    stop_criteria, generation_config = setup_model_config(model, cfg)

    # Evaluate on all datasets
    assert len(cfg.val_datasets) == len(
        cfg.val_evaluators
    ), f"len(cfg.val_datasets) = {len(cfg.val_datasets)}, len(cfg.val_evaluators) = {len(cfg.val_evaluators)}"
    print_log(f"Evaluating {len(cfg.val_datasets)} datasets...", logger="current")
    for dataset_cfg, evaluator_cfg in zip(cfg.val_datasets, cfg.val_evaluators):
        try:
            dataset = BUILDER.build(dataset_cfg)
            model.postprocess_fn = dataset.postprocess_fn

            evaluator = BUILDER.build(evaluator_cfg)
            evaluator.metadata = dataset.metadata
            evaluator.output_dir = osp.join(args.work_dir, "pred_data", evaluator.data_name)
            evaluate_dataset(model, dataset, evaluator, rank, world_size, generation_config, stop_criteria)
        except Exception as e:
            print_log(f"Error evaluating {dataset_cfg.data_name}\n: {e}\n{traceback.format_exc()}", logger="current")
            continue


if __name__ == "__main__":
    main()
