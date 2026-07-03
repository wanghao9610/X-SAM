#!/usr/bin/env python

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import logging
import os.path as osp
import random
import re
import traceback
import warnings
from typing import List, Optional

import cv2
import torch
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import (
    dataset_map_fn_factory,
    img_chat_map_fn,
    img_gcgseg_map_fn,
    img_genseg_map_fn,
    img_intseg_map_fn,
    img_reaseg_map_fn,
    img_refseg_map_fn,
    img_vgdseg_map_fn,
    template_map_fn_factory,
)
from xsam.dataset.process_fns import (
    img_gcgseg_postprocess_fn,
    img_genseg_postprocess_fn,
    img_intseg_postprocess_fn,
    img_reaseg_postprocess_fn,
    img_refseg_postprocess_fn,
    img_vgdseg_postprocess_fn,
    process_map_fn_factory,
)
from xsam.dataset.utils.catalog import MetadataCatalog
from xsam.dataset.utils.encode import encode_fn
from xsam.dataset.utils.image import expand2square
from xsam.dataset.utils.load import load_image
from xsam.model.utils import traverse_dict
from xsam.registry import BUILDER
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.configs import cfgs_name_path
from xsam.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PLACEHOLDER_TOKEN,
    DEFAULT_PSTART_TOKEN,
    INDEX2TOKEN,
)
from xsam.utils.device import get_device
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function, set_model_resource, split_list

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Single image demo for X-SAM model")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--image", type=str, required=True, help="path to input image")
    parser.add_argument("--work-dir", type=str, required=False, help="directory to save logs and visualizations")
    parser.add_argument("--output-dir", type=str, required=False, help="directory to save output images")
    parser.add_argument("--prompt", type=str, required=False, default="", help="user prompt for the task name")
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="image task name (e.g., img_vgdseg)",
    )
    parser.add_argument(
        "--vprompt-masks",
        nargs="+",
        required=False,
        help="paths to vprompt mask files or directories",
    )
    parser.add_argument("--score-thr", type=float, default=0.5, help="score threshold for the task name")
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
            assert osp.exists(pth_model), f"Checkpoint file {pth_model} does not exist"
            load_checkpoint(self.model, pth_model)
        else:
            print_log("No checkpoint file provided, using default checkpoint", logger="current", level=logging.WARNING)
        self.stop_criteria, self.generation_config = setup_model_config(self.model, cfg)

        self.tokenizer = self.model.tokenizer
        self.visualizer = build_from_cfg_or_module(cfg.visualizer)
        self.image_processor = build_from_cfg_or_module(cfg.image_processor)
        self.extra_image_processor = build_from_cfg_or_module(cfg.extra_image_processor)

        self.cond_type = cfg.cond_type
        self.image_token = cfg.image_token
        self.max_length = cfg.max_length
        self.expand2square = cfg.expand2square
        self.use_placeholder = cfg.use_placeholder
        self.use_vision_token = "vision" in self.image_token
        self.output_ids_with_output = output_ids_with_output
        self.metadata = MetadataCatalog.get("default")
        self.metadata.set(ignore_value=255, label_divisor=1000)
        self.dtype = self.model.dtype

        self.task_map_fns = self.build_map_fns()
        self.template_map_fns = self.build_template_map_fns()
        self.postprocess_fns = self.build_postprocess_fn()
        self._last_infer_error = None

    def build_template_map_fns(self):
        template_map_fns = {
            "img_chat": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "img_genseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "img_refseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "img_reaseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=self.output_ids_with_output,
            ),
            "img_gcgseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=False,
            ),
            "img_vgdseg": dict(
                type=template_map_fn_factory,
                template=self.cfg.prompt_template,
                output_suffix=False,
            ),
            "img_intseg": dict(
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
            "img_chat": dict(
                type=dataset_map_fn_factory,
                fn=img_chat_map_fn,
                image_token=self.image_token,
            ),
            "img_genseg": dict(
                type=dataset_map_fn_factory,
                fn=img_genseg_map_fn,
                cond_type=self.cond_type,
                image_token=self.image_token,
            ),
            "img_refseg": dict(
                type=dataset_map_fn_factory,
                fn=img_refseg_map_fn,
                cond_type=self.cond_type,
                image_token=self.image_token,
            ),
            "img_reaseg": dict(
                type=dataset_map_fn_factory,
                fn=img_reaseg_map_fn,
                cond_type=self.cond_type,
                image_token=self.image_token,
            ),
            "img_gcgseg": dict(
                type=dataset_map_fn_factory,
                fn=img_gcgseg_map_fn,
                cond_type=self.cond_type,
                image_token=self.image_token,
            ),
            "img_vgdseg": dict(
                type=dataset_map_fn_factory,
                fn=img_vgdseg_map_fn,
                cond_type=self.cond_type,
                image_token=self.image_token,
            ),
            "img_intseg": dict(
                type=dataset_map_fn_factory,
                fn=img_intseg_map_fn,
                cond_type=self.cond_type,
                image_token=self.image_token,
            ),
        }
        task_map_fns = {
            task_name: build_from_cfg_or_module(task_map_fn) for task_name, task_map_fn in task_map_fns.items()
        }
        return task_map_fns

    def build_postprocess_fn(self):
        postprocess_fns = {
            "img_chat": None,
            "img_genseg|pan": dict(
                type=process_map_fn_factory,
                fn=img_genseg_postprocess_fn,
                task_name="img_genseg_panoptic",
                threshold=0.5,
            ),
            "img_genseg|sem": dict(
                type=process_map_fn_factory,
                fn=img_genseg_postprocess_fn,
                task_name="img_genseg_semantic",
            ),
            "img_genseg|ins": dict(
                type=process_map_fn_factory,
                fn=img_genseg_postprocess_fn,
                task_name="img_genseg_instance",
            ),
            "img_refseg": img_refseg_postprocess_fn,
            "img_reaseg": img_reaseg_postprocess_fn,
            "img_gcgseg": img_gcgseg_postprocess_fn,
            "img_vgdseg": img_vgdseg_postprocess_fn,
            "img_intseg": img_intseg_postprocess_fn,
        }
        postprocess_fns = {
            task_name: build_from_cfg_or_module(postprocess_fn)
            for task_name, postprocess_fn in postprocess_fns.items()
        }
        return postprocess_fns

    def _get_input_ids(self, data_dict, task_name, next_needs_bos_token=False):
        if self.tokenizer is None:
            return data_dict

        if self.task_map_fns[task_name] is not None:
            data_dict.update(
                self.task_map_fns[task_name](data_dict, output_ids_with_output=self.output_ids_with_output)
            )
        if self.template_map_fns[task_name] is not None:
            data_dict.update(self.template_map_fns[task_name](data_dict))
        if self.tokenizer is not None:
            data_dict = encode_fn(
                data_dict,
                self.tokenizer,
                self.max_length,
                self.image_processor,
                input_ids_with_output=self.output_ids_with_output,
                use_placeholder=self.use_placeholder,
                use_vision_token=self.use_vision_token,
                next_needs_bos_token=next_needs_bos_token,
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
            task_name = f"{task_name}|pan"
        elif len(thing_classes) > 0 and len(stuff_classes) == 0:
            task_name = f"{task_name}|ins"
        elif len(thing_classes) == 0 and len(stuff_classes) > 0:
            task_name = f"{task_name}|sem"
        return (all_classes, thing_classes, stuff_classes), task_name

    def _process_prompt(self, prompt, task_name, classes=None):
        if task_name == "img_chat":
            example = {
                "conversations": [
                    {"from": "human", "value": DEFAULT_IMAGE_TOKEN + prompt},
                    {"from": "gpt", "value": ""},
                ]
            }
        elif "genseg" in task_name:
            example = {
                "sampled_cats": classes[0],
            }
        elif "refseg" in task_name:
            example = {
                "sampled_sents": [prompt],
            }
        elif "reaseg" in task_name:
            example = {
                "sampled_sents": [prompt],
                "is_sentence": True,
            }
        elif "gcgseg" in task_name:
            example = {}
        elif "intseg" in task_name or "objseg" in task_name:
            example = {
                "sampled_labels": [0],
            }
        elif "vgdseg" in task_name:
            example = {
                "sampled_labels": [0],
            }
        else:
            raise ValueError(f"Unsupported task_name: {task_name}")

        return example

    def _process_image(self, image):
        image = load_image(image, to_numpy=True)
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

    def _process_image_data_dict(self, data_dict):
        data_dict["image_file"] = None
        pil_image = data_dict["pil_image"]
        if self.image_processor is not None:
            image = pil_image
            if self.expand2square:
                image = expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            output = self.image_processor.preprocess(image, return_tensors="pt")
            pixel_values = output["pixel_values"][0] if output["pixel_values"].ndim == 4 else output["pixel_values"]
            image_grid_thw = output.get("image_grid_thw", None)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw
        if self.extra_image_processor is not None:
            extra_output = self.extra_image_processor.preprocess(
                pil_image, vprompt_masks=data_dict.get("vprompt_masks", None), return_tensors="pt"
            )
            data_dict["extra_pixel_values"] = extra_output["pixel_values"][0]
            data_dict["scaled_size"] = extra_output["scaled_sizes"][0].tolist()
            data_dict["vprompt_masks"] = extra_output.get("vprompt_masks", None)

        data_dict.update(self._get_vgd_labels(data_dict))
        data_dict.update(self._get_input_ids(data_dict, data_dict["task_name"]))
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
        ignore_tokens = [
            f"{DEFAULT_IMAGE_TOKEN}",
            f"{DEFAULT_IMAGE_TOKEN}\n",
            f"{DEFAULT_PSTART_TOKEN} ",
            f"{DEFAULT_PEND_TOKEN} ",
            "<|user|>",
            "<|assistant|>",
            "<|im_start|>"
            "<|im_end|>",
        ]
        for ignore_token in ignore_tokens:
            text = text.replace(ignore_token, "")
        return text

    def _set_metadata(self, task_name, classes=None):
        MetadataCatalog.reset()
        metadata = MetadataCatalog.get(task_name)
        metadata.set(
            label_divisor=1000,
            ignore_value=255,
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

    def _visualize_image_predictions(
        self, image, seg_output, task_name, phrases=None, output_dir=None, file_prefix="image"
    ):
        visualized_image = self.visualizer.draw_predictions(
            image,
            data_name=task_name,
            phrases=phrases,
            **seg_output,
        )
        visualized_image = visualized_image.get_image()
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_file = osp.join(output_dir, f"{file_prefix}.png")
            cv2.imwrite(output_file, cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
        return visualized_image

    @staticmethod
    def _is_cuda_oom(exc: Optional[BaseException]) -> bool:
        if exc is None:
            return False
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda oom" in msg

    @staticmethod
    def _dispose_mlm_outputs(mlm_outputs) -> None:
        """Drop generation hidden states; they can occupy multiple GB on GPU."""
        if mlm_outputs is None:
            return
        for attr in ("hidden_states", "past_key_values", "attentions", "scores"):
            if hasattr(mlm_outputs, attr):
                setattr(mlm_outputs, attr, None)

    @staticmethod
    def _release_gpu_memory(aggressive: bool = False) -> None:
        """Return cached GPU memory to the driver (also needed after OOM)."""
        gc.collect()
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        if aggressive:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def _tensor_to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: XSamDemo._tensor_to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [XSamDemo._tensor_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(XSamDemo._tensor_to_cpu(v) for v in value)
        return value

    def _finalize_inference(
        self,
        data_dict,
        data_samples,
        infer_error=None,
        mlm_outputs=None,
    ) -> None:
        """Release per-request GPU tensors; model weights stay loaded."""
        if infer_error is not None:
            self._dispose_mlm_outputs(mlm_outputs)
        self._clear_inference_buffers(data_dict, data_samples)
        self._release_gpu_memory(aggressive=self._is_cuda_oom(infer_error))

    @staticmethod
    def _clear_inference_buffers(data_dict, data_samples):
        """Drop references to GPU tensors from a failed or finished forward pass."""
        if isinstance(data_dict, dict):
            data_dict.clear()
        if isinstance(data_samples, list):
            data_samples.clear()

    def run_on_image(self, image, prompt, task_name, vprompt_masks=None, **kwargs):
        mode = "tensor" if self.output_ids_with_output else "predict"
        output_dir = kwargs.pop("output_dir", None)
        file_prefix = kwargs.pop("file_prefix", "image")
        image = load_image(image, mode="RGB")
        data_dict = {"pil_image": image, "vprompt_masks": vprompt_masks, "task_name": task_name}

        classes, task_name_postprocess = self._get_classes_from_prompt(prompt, task_name)
        self.model.postprocess_fn = self.postprocess_fns[task_name_postprocess]
        self._set_metadata(task_name, classes)
        data_dict.update(self._process_prompt(prompt, task_name, classes))
        data_dict.update(self._process_image(image))
        data_dict.update(self._process_image_data_dict(data_dict))
        data_dict, data_samples = self._process_input_dict(data_dict)
        input_ids = self._tensor_to_cpu(data_dict["input_ids"])

        metadata = MetadataCatalog.get(f"{task_name}") if task_name in MetadataCatalog.list() else self.metadata

        mlm_outputs = None
        seg_outputs = None
        infer_error = None
        with torch.no_grad():
            try:
                mlm_outputs, seg_outputs = self.model(
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
            except Exception as e:
                infer_error = e
                self._last_infer_error = e
                print_log(f"Error in {task_name} prediction: {e}\n{traceback.format_exc()}", logger="current")
            finally:
                self._finalize_inference(
                    data_dict,
                    data_samples,
                    infer_error=infer_error,
                    mlm_outputs=mlm_outputs,
                )
                if infer_error is not None:
                    mlm_outputs = None
                    seg_outputs = None

        if infer_error is not None:
            return None, None, None

        mlm_input = self._decode_input_ids(input_ids[0].tolist())
        mlm_input = re.sub(f"({re.escape(DEFAULT_PLACEHOLDER_TOKEN)}\\s*)+", DEFAULT_IMAGE_TOKEN, mlm_input)
        mlm_input = re.sub(r" {2,}", " ", mlm_input)

        output_ids = mlm_outputs.sequences.detach().cpu()
        generation_output = self.tokenizer.decode(output_ids[0]).strip()
        generation_output = re.sub(r" {2,}", " ", generation_output)

        input_phrases = []
        output_phrases = []
        if hasattr(self.model, "pstart_token_idx") and hasattr(self.model, "pend_token_idx"):
            input_phrases_ids = self._get_phrases_ids(input_ids[0])
            input_phrases = self._decode_phrases_ids(input_phrases_ids)

        if hasattr(self.model, "pstart_token_idx") and hasattr(self.model, "pend_token_idx"):
            output_phrases_ids = self._get_phrases_ids(output_ids[0])
            output_phrases = self._decode_phrases_ids(output_phrases_ids)
        phrases = output_phrases or input_phrases

        print_log(f"Sample output of {task_name}:\n" f"{mlm_input + generation_output}\n", logger="current")
        self.visualizer.metadata = metadata

        self._dispose_mlm_outputs(mlm_outputs)
        mlm_outputs = None
        self._release_gpu_memory()

        if seg_outputs is None:
            return mlm_input, generation_output, None

        try:
            visualized_image = self._visualize_image_predictions(
                image,
                seg_outputs[0],
                task_name_postprocess,
                phrases=phrases,
                output_dir=output_dir,
                file_prefix=file_prefix,
            )
        except Exception as e:
            print_log(f"Error in {task_name} visualization: {e}\n{traceback.format_exc()}", logger="current")
            self._release_gpu_memory(aggressive=self._is_cuda_oom(e))
            return mlm_input, generation_output, None
        finally:
            seg_outputs = None
            self._release_gpu_memory()

        return mlm_input, generation_output, visualized_image


def main():
    """Main demo function for image processing."""
    args = parse_args()

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

    # Create demo
    demo = XSamDemo(cfg, args.pth_model, output_ids_with_output=False)

    vprompt_masks = []
    for vprompt_mask_path in args.vprompt_masks or []:
        if not osp.exists(vprompt_mask_path):
            print_log(f"Vprompt mask path not found: {vprompt_mask_path}", logger="current", level=logging.WARNING)
            continue
        if osp.isdir(vprompt_mask_path):
            vprompt_masks.extend(
                load_image(osp.join(vprompt_mask_path, file), mode="L") for file in sorted(os.listdir(vprompt_mask_path))
            )
        else:
            vprompt_masks.append(load_image(vprompt_mask_path, mode="L"))
    vprompt_masks = vprompt_masks or None

    if args.image and osp.isdir(args.image):
        output_dir = args.image + "_demo" if args.output_dir is None else args.output_dir
        output_dir = osp.join(output_dir, args.task_name)
        for file in os.listdir(args.image):
            demo.run_on_image(
                osp.join(args.image, file),
                args.prompt,
                args.task_name,
                vprompt_masks=vprompt_masks,
                threshold=args.score_thr,
                output_dir=output_dir,
                file_prefix=file[:-4],
            )
    elif args.image and osp.isfile(args.image):
        output_dir = osp.dirname(args.image) + "_demo" if args.output_dir is None else args.output_dir
        output_dir = osp.join(output_dir, args.task_name)
        demo.run_on_image(
            args.image,
            args.prompt,
            args.task_name,
            vprompt_masks=vprompt_masks,
            threshold=args.score_thr,
            output_dir=output_dir,
            file_prefix=osp.basename(args.image)[:-4],
        )
    else:
        raise ValueError(f"Invalid image: {args.image}")


if __name__ == "__main__":
    main()
