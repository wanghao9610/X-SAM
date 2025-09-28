#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os
import os.path as osp
import random
import re
import traceback
import warnings
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.dataset.utils import expand2square
from xtuner.model.utils import traverse_dict
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import (
    dataset_map_fn_factory,
    gcg_seg_map_fn,
    generic_seg_map_fn,
    image_conv_map_fn,
    inter_seg_map_fn,
    reason_seg_map_fn,
    refer_seg_map_fn,
    template_map_fn_factory,
    vgd_seg_map_fn,
)
from xsam.dataset.process_fns import (
    gcg_seg_postprocess_fn,
    generic_seg_postprocess_fn,
    inter_seg_postprocess_fn,
    process_map_fn_factory,
    reason_seg_postprocess_fn,
    refer_seg_postprocess_fn,
    vgd_seg_postprocess_fn,
)
from xsam.dataset.utils.catalog import MetadataCatalog
from xsam.dataset.utils.encode import encode_fn
from xsam.dataset.utils.load import load_image
from xsam.engine.utils.util import split_list
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.constants import DEFAULT_IMAGE_TOKEN, INDEX2TOKEN
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Single image demo for X-SAM model")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--image", type=str, required=True, help="path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="user prompt for the task_name")
    parser.add_argument("--task_name", type=str, required=True, help="task_name name (e.g., segmentation, referring)")
    parser.add_argument("--vprompt_masks", type=str, required=False, help="path to vprompt masks")
    parser.add_argument("--score_thr", type=float, default=0.5, help="score threshold for the task_name")
    parser.add_argument("--work-dir", type=str, required=False, help="directory to save logs and visualizations")
    parser.add_argument("--output_dir", type=str, required=False, help="directory to save output images")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    return parser.parse_args()


def build_from_cfg_or_module(cfg_or_mod):
    if cfg_or_mod is None:
        return None

    if isinstance(cfg_or_mod, nn.Module):
        return cfg_or_mod
    elif callable(cfg_or_mod):
        return cfg_or_mod
    elif isinstance(cfg_or_mod, dict):
        traverse_dict(cfg_or_mod)
        return BUILDER.build(cfg_or_mod)
    else:
        raise NotImplementedError


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


class XSamDemo:
    def __init__(
        self,
        cfg,
        pth_model=None,
        output_ids_with_output=True,
        max_length=4096,
        cond_type="phrase",
        pad_image_to_square=True,
        **kwargs,
    ):
        self.cfg = cfg
        self.device = get_device()
        self.cpu_device = torch.device("cpu")

        self.model = BUILDER.build(cfg.model)
        if "llm" in cfg.model:
            self.model.llm.to(cfg.model.llm.torch_dtype)
        self.model.eval()
        self.model = self.model.to(self.device)
        if pth_model is not None:
            print_log(f"Loading checkpoint from {pth_model}", logger="current")
            assert osp.exists(pth_model), f"Checkpoint file {pth_model} does not exist"
            load_checkpoint(self.model, pth_model)
        self.stop_criteria, self.generation_config = setup_model_config(self.model, cfg)

        self.tokenizer = self.model.tokenizer
        self.visualizer = build_from_cfg_or_module(cfg.visualizer)
        self.image_processor = build_from_cfg_or_module(cfg.image_processor)
        self.extra_image_processor = build_from_cfg_or_module(cfg.extra_image_processor)

        self.cond_type = cond_type
        self.output_ids_with_output = output_ids_with_output
        self.max_length = max_length
        self.pad_image_to_square = pad_image_to_square
        self.metadata = MetadataCatalog.get("default")
        self.metadata.set(ignore_label=255, label_divisor=1000)
        self.dtype = self.model.dtype

        self.task_map_fns = self.build_map_fns()
        self.template_map_fns = self.build_template_map_fns()
        self.postprocess_fns = self.build_postprocess_fn()

    def build_template_map_fns(self):
        template_map_fns = {
            "imgconv": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "genseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "refseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "reaseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "gcgseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=False,
            ),
            "interseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "vgdseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
        }
        template_map_fns = {
            task_name: build_from_cfg_or_module(template_map_fn)
            for task_name, template_map_fn in template_map_fns.items()
        }
        return template_map_fns

    def build_map_fns(self):
        task_map_fns = {
            "imgconv": image_conv_map_fn,
            "genseg": dict(
                type=dataset_map_fn_factory,
                fn=generic_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "refseg": dict(
                type=dataset_map_fn_factory,
                fn=refer_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "reaseg": dict(
                type=dataset_map_fn_factory,
                fn=reason_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "gcgseg": dict(
                type=dataset_map_fn_factory,
                fn=gcg_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "interseg": dict(
                type=dataset_map_fn_factory,
                fn=inter_seg_map_fn,
                cond_type=self.cond_type,
            ),
            "vgdseg": dict(
                type=dataset_map_fn_factory,
                fn=vgd_seg_map_fn,
                cond_type=self.cond_type,
            ),
        }
        task_map_fns = {
            task_name: build_from_cfg_or_module(task_map_fn) for task_name, task_map_fn in task_map_fns.items()
        }
        return task_map_fns

    def build_postprocess_fn(self):
        postprocess_fns = {
            "imgconv": None,
            "genseg(pan)": dict(
                type=process_map_fn_factory,
                fn=generic_seg_postprocess_fn,
                task_name="genseg(pan)",
                threshold=0.5,
            ),
            "genseg(sem)": dict(
                type=process_map_fn_factory,
                fn=generic_seg_postprocess_fn,
                task_name="genseg(sem)",
            ),
            "genseg(ins)": dict(
                type=process_map_fn_factory,
                fn=generic_seg_postprocess_fn,
                task_name="genseg(ins)",
            ),
            "refseg": refer_seg_postprocess_fn,
            "reaseg": reason_seg_postprocess_fn,
            "gcgseg": gcg_seg_postprocess_fn,
            "interseg": inter_seg_postprocess_fn,
            "vgdseg": vgd_seg_postprocess_fn,
        }
        postprocess_fns = {
            task_name: build_from_cfg_or_module(postprocess_fn)
            for task_name, postprocess_fn in postprocess_fns.items()
        }
        return postprocess_fns

    def _get_input_ids(self, data_dict, task_name, with_image_token=True, next_needs_bos_token=False):
        if self.tokenizer is None:
            return data_dict

        if self.task_map_fns[task_name] is not None:
            data_dict = self.task_map_fns[task_name](data_dict, self.output_ids_with_output)
        if self.template_map_fns[task_name] is not None:
            data_dict = self.template_map_fns[task_name](data_dict)
        if self.tokenizer is not None:
            data_dict = encode_fn(
                data_dict,
                self.tokenizer,
                self.max_length,
                self.output_ids_with_output,
                with_image_token,
                next_needs_bos_token,
            )
        return data_dict

    def _get_cond_ids(self, data_dict):
        if self.tokenizer is None:
            return data_dict

        input_ids = data_dict["input_ids"]
        cond_ids = [-1] * len(input_ids)
        pstart_idx = [i for i, x in enumerate(input_ids) if x == self.model.pstart_token_idx]
        pend_idx = [i for i, x in enumerate(input_ids) if x == self.model.pend_token_idx]
        cls_idx = [i for i, x in enumerate(input_ids) if x == self.model.cls_token_idx]

        if len(pstart_idx) == 0 and len(pend_idx) == 0 and len(cls_idx) == 0:
            return data_dict

        if self.cond_type in ["phrase", "all"]:
            for i, (ps, pe) in enumerate(zip(pstart_idx, pend_idx)):
                cond_ids[ps : pe + 1] = [i] * (pe - ps + 1)
        if self.cond_type in ["cls", "all"]:
            for i, ci in enumerate(cls_idx):
                cond_ids[ci] = i

        data_dict["cond_ids"] = cond_ids
        return data_dict

    def _get_phrases_ids(self, input_ids):
        pstart_idx = [i for i, x in enumerate(input_ids) if x == self.model.pstart_token_idx]
        pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == self.model.pend_token_idx]
        phrases_ids = []
        for ps, pe in zip(pstart_idx, pend_idx):
            phrases_ids.append(input_ids[ps + 1 : pe - 1])
        return phrases_ids

    def _get_seg_ids(self, data_dict):
        if self.tokenizer is None:
            return data_dict

        input_ids = data_dict["input_ids"]
        seg_ids = [-1] * len(input_ids)

        seg_idx = [i for i, x in enumerate(input_ids) if x == self.model.seg_token_idx]
        for i, idx in enumerate(seg_idx):
            seg_ids[idx] = i

        data_dict["seg_ids"] = seg_ids
        return data_dict

    def _get_vgd_labels(self, data_dict):
        vprompt_masks = data_dict.get("vprompt_masks", None)
        if vprompt_masks is None:
            return data_dict

        class_labels = [i for i in range(len(vprompt_masks))]
        sampled_labels = [i for i in range(len(vprompt_masks))]
        contiguous_labels = [i for i in range(len(vprompt_masks))]

        data_dict["class_labels"] = torch.tensor(class_labels, dtype=torch.int64)
        data_dict["sampled_labels"] = sampled_labels
        data_dict["contiguous_labels"] = contiguous_labels
        return data_dict

    def _get_classes_from_prompt(self, prompt, task_name):
        if "genseg" not in task_name:
            return ([], [], []), task_name

        ins_match = re.search(r"ins:\s*([^;\n]+)", prompt)
        sem_match = re.search(r"sem:\s*([^;\n]+)", prompt)

        thing_classes = [x.strip() for x in ins_match.group(1).split(",") if len(x.strip()) > 0] if ins_match else []
        stuff_classes = [x.strip() for x in sem_match.group(1).split(",") if len(x.strip()) > 0] if sem_match else []
        all_classes = thing_classes + stuff_classes
        all_classes = random.sample(all_classes, len(all_classes))
        assert len(all_classes) > 0, "Please provide at least one thing or stuff class"
        if len(thing_classes) > 0 and len(stuff_classes) > 0:
            task_name = "genseg(pan)"
        elif len(thing_classes) > 0 and len(stuff_classes) == 0:
            task_name = "genseg(ins)"
        elif len(thing_classes) == 0 and len(stuff_classes) > 0:
            task_name = "genseg(sem)"
        return (all_classes, thing_classes, stuff_classes), task_name

    def _process_prompt(self, prompt, task_name, classes=None):
        if task_name == "imgconv":
            example = {
                "conversations": [
                    {"from": "human", "value": DEFAULT_IMAGE_TOKEN + prompt},
                    {"from": "gpt", "value": ""},
                ]
            }
        elif "genseg" in task_name:
            example = {
                "sampled_cats": classes[0],
                "caption": None,
            }
        elif task_name == "refseg":
            example = {
                "sampled_sents": [prompt],
            }
        elif task_name == "reaseg":
            example = {
                "sampled_sents": [prompt],
                "explain": None,
                "is_sentence": True,
            }
        elif task_name == "gcgseg":
            example = {}
        elif task_name == "interseg":
            # TODO: add interseg example
            example = {
                "sampled_labels": [0],
            }
        elif task_name == "vgdseg":
            # TODO: add vgdseg example
            example = {
                "sampled_labels": [0],
            }
        else:
            raise ValueError(f"Unsupported task_name: {task_name}")

        return example

    def _process_image(self, image):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image = np.array(pil_image)
        height, width = image.shape[:2]
        _image_info = {
            "height": height,
            "width": width,
            "image_size": (height, width),
        }
        image_info = {
            "image_info": _image_info,
            "image_size": (height, width),
        }
        return image_info

    def _process_data_dict(self, data_dict):
        data_dict["image_file"] = None
        pil_image = data_dict["pil_image"]
        if self.image_processor is not None:
            image = pil_image
            if self.pad_image_to_square:
                image = expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            data_dict["pixel_values"] = image
        if self.extra_image_processor is not None:
            seg_output = self.extra_image_processor.preprocess(
                pil_image, condition_maps=data_dict["vprompt_masks"], return_tensors="pt"
            )
            data_dict["seg_pixel_values"] = seg_output["pixel_values"][0]
            data_dict["scaled_size"] = tuple(seg_output["scaled_sizes"][0].tolist())
            data_dict["vprompt_masks"] = seg_output.get("vprompt_masks", None)

        data_dict.update(self._get_vgd_labels(data_dict))
        data_dict.update(self._get_input_ids(data_dict, data_dict["task_name"], with_image_token=True))
        data_dict.update(self._get_cond_ids(data_dict))
        data_dict.update(self._get_seg_ids(data_dict))

        return data_dict

    def _process_input_dict(self, data_dict):
        input_dict = xsam_collate_fn([data_dict])
        input_dict = data_dict_to_device(input_dict, device=self.device, dtype=self.dtype)
        data_dict = input_dict["data_dict"]
        data_samples = input_dict["data_samples"]
        data_dict.pop("labels", None)
        data_dict.pop("position_ids", None)
        data_dict.pop("attention_mask", None)

        return data_dict, data_samples

    def _decode_phrases_ids(self, phrases_ids):
        phrases = []
        for phrase_id in phrases_ids:
            if (phrase_id < 0).any():
                phrase = ""
            else:
                phrase = self.tokenizer.decode(phrase_id).strip()
            phrases.append(phrase)
        return phrases

    def _decode_input_ids(self, input_ids):
        input_ids = split_list(input_ids, INDEX2TOKEN.keys())
        text = ""
        for ids in input_ids:
            if len(ids) == 1 and ids[0] in INDEX2TOKEN:
                text += INDEX2TOKEN[ids[0]]
            else:
                text += self.tokenizer.decode(ids)
        ignore_tokens = ["<image>\n", "<p> ", "</p> ", "<|user|>", "<|assistant|>", "<|end|>"]
        for ignore_token in ignore_tokens:
            text = text.replace(ignore_token, "")
        return text

    def _set_metadata(self, task_name, classes=None):
        MetadataCatalog.reset()
        metadata = MetadataCatalog.get(task_name)
        metadata.set(
            label_divisor=1000,
            ignore_label=255,
            data_name=task_name,
        )
        if "genseg" in task_name:
            all_classes, thing_classes, stuff_classes = classes
            metadata.set(
                dataset_id_to_contiguous_id={i: i for i, _ in enumerate(all_classes)},
                thing_dataset_id_to_contiguous_id={i: i for i, c in enumerate(all_classes) if c in thing_classes},
                stuff_dataset_id_to_contiguous_id={i: i for i, c in enumerate(all_classes) if c in stuff_classes},
                thing_classes={i: c for i, c in enumerate(all_classes) if c in thing_classes},
                stuff_classes={i: c for i, c in enumerate(all_classes) if c in stuff_classes},
            )

        return metadata

    def run_on_image(self, image, prompt, task_name, vprompt_masks=None, **kwargs):
        mode = "tensor" if self.output_ids_with_output else "predict"
        data_dict = {"pil_image": image, "vprompt_masks": vprompt_masks, "task_name": task_name}

        classes, task_name_postprocess = self._get_classes_from_prompt(prompt, task_name)
        self.model.postprocess_fn = self.postprocess_fns[task_name_postprocess]
        self._set_metadata(task_name, classes)
        data_dict.update(self._process_prompt(prompt, task_name, classes))
        data_dict.update(self._process_image(image))
        data_dict.update(self._process_data_dict(data_dict))
        data_dict, data_samples = self._process_input_dict(data_dict)
        input_ids = data_dict["input_ids"]

        metadata = MetadataCatalog.get(f"{task_name}") if task_name in MetadataCatalog.list() else self.metadata

        with torch.no_grad():
            try:
                llm_outputs, seg_outputs = self.model(
                    data_dict,
                    data_samples,
                    mode=mode,
                    metadata=metadata,
                    generation_config=self.generation_config,
                    stopping_criteria=self.stop_criteria,
                    do_postprocess=True,
                    do_loss=False,
                    **kwargs,
                )
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                print_log(f"Error in {task_name} prediction: {e}\n{traceback.format_exc()}", logger="current")
                return None, None, None

        output_ids = llm_outputs.sequences
        generation_output = self.tokenizer.decode(output_ids[0]).strip()
        generation_output = generation_output.replace("<|end|>", "").replace("<p> ", "<p>").replace("</p> ", "</p>")
        if task_name != "gcgseg":
            generation_output = generation_output.replace("<p>", "").replace("</p>", "")
        llm_input = self._decode_input_ids(input_ids[0].tolist())

        input_phrases = []
        output_phrases = []
        if hasattr(self.model, "pstart_token_idx") and hasattr(self.model, "pend_token_idx"):
            input_phrases_ids = self._get_phrases_ids(input_ids[0])
            input_phrases = self._decode_phrases_ids(input_phrases_ids)

        if hasattr(self.model, "pstart_token_idx") and hasattr(self.model, "pend_token_idx"):
            output_phrases_ids = self._get_phrases_ids(output_ids[0])
            output_phrases = self._decode_phrases_ids(output_phrases_ids)
        phrases = output_phrases or input_phrases

        print_log(f"Sample output of {task_name}:\n" f"{llm_input + generation_output}\n", logger="current")
        self.visualizer.metadata = metadata

        try:
            visualized_image = self.visualizer.draw_predictions(
                image,
                data_name=task_name_postprocess,
                phrases=phrases,
                **(seg_outputs[0]),
            )
        except Exception as e:
            print_log(f"Error in {task_name} visualization: {e}\n{traceback.format_exc()}", logger="current")
            return llm_input, generation_output, None

        return llm_input, generation_output, visualized_image.get_image()


def main():
    """Main demo function for single image processing."""
    args = parse_args()

    # Validate input image exists
    if not osp.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

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
        set_random_seed(args.seed)
        print_log(f"Set the random seed to {args.seed}.", logger="current")
    register_function(cfg._cfg_dict)

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if args.work_dir and osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        elif args.work_dir:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        else:
            raise ValueError("work_dir must be specified when using 'latest' checkpoint")
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")

    # Create demo instance
    demo = XSamDemo(cfg, args.pth_model, output_ids_with_output=False)

    if args.vprompt_masks and osp.exists(args.vprompt_masks):
        if osp.isdir(args.vprompt_masks):
            vprompt_masks = [
                load_image(osp.join(args.vprompt_masks, file), mode="L") for file in os.listdir(args.vprompt_masks)
            ]
        else:
            vprompt_masks = [load_image(args.vprompt_masks, mode="L")]
    else:
        vprompt_masks = None

    if osp.isdir(args.image):
        output_dir = args.image + "_vis" if args.output_dir is None else args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(args.image):
            pil_image = Image.open(osp.join(args.image, file))
            llm_input, llm_output, seg_output = demo.run_on_image(
                pil_image, args.prompt, args.task_name, vprompt_masks=vprompt_masks, threshold=args.score_thr
            )
            # Save seg_output image
            if seg_output is not None:
                print(f"llm_input: {llm_input}\nllm_output: {llm_output}")
                cv2.imwrite(f"{output_dir}/{file[:-4]}.png", cv2.cvtColor(seg_output, cv2.COLOR_RGB2BGR))
    elif osp.isfile(args.image):
        pil_image = Image.open(args.image)
        llm_input, llm_output, seg_output = demo.run_on_image(
            pil_image, args.prompt, args.task_name, vprompt_masks=vprompt_masks, threshold=args.score_thr
        )
        # Save seg_output image
        output_dir = osp.dirname(args.image) + "_vis" if args.output_dir is None else args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if seg_output is not None:
            print(f"llm_input: {llm_input}\nllm_output: {llm_output}")
            cv2.imwrite(
                f"{output_dir}/{osp.basename(args.image)[:-4]}.png", cv2.cvtColor(seg_output, cv2.COLOR_RGB2BGR)
            )
    else:
        raise ValueError(f"Invalid image: {args.image}")


if __name__ == "__main__":
    main()
