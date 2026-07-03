import os.path as osp
import re
import traceback
import warnings

import torch
from mmengine.config import Config, ConfigDict
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils import mkdir_or_exist
from mmengine.utils.misc import get_object_from_string
from transformers import GenerationConfig, StoppingCriteriaList

from xsam.dataset.utils.catalog import MetadataCatalog
from xsam.dataset.utils.image import expand2square
from xsam.dataset.utils.load import load_image
from xsam.registry import BUILDER
from xsam.structures.data_sample import DataSample
from xsam.utils.constants import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PLACEHOLDER_TOKEN,
    DEFAULT_PSTART_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
    TOKEN2INDEX,
)
from xsam.utils.criteria import StopWordStoppingCriteria
from xsam.utils.logging import print_log


class GenerationChatHook(Hook):
    priority = "LOW"

    def __init__(
        self,
        tokenizer,
        image_inputs=None,
        image_task_names=None,
        special_tokens=None,
        generation_images=None,
        reference_images=None,
        image_vprompt_masks=None,
        image_postprocess_fns=None,
        image_processor=None,
        extra_image_processor=None,
        visualizer=None,
        system="",
        image_token=None,
        expand2square=False,
        use_placeholder=False,
        prompt_template=None,
        every_n_iters=None,
        max_new_tokens=2048,
        stop_word=None,
        stop_words=[],
        generation_kwargs={},
    ):
        self.image_inputs = image_inputs
        self.image_task_names = image_task_names
        self.image_postprocess_fns = image_postprocess_fns
        self.image_token = image_token
        self.expand2square = expand2square
        self.use_placeholder = use_placeholder
        if self.image_postprocess_fns is not None:
            assert isinstance(self.image_postprocess_fns, list)
            for i, postprocess_fn in enumerate(self.image_postprocess_fns):
                if (
                    isinstance(postprocess_fn, dict)
                    or isinstance(postprocess_fn, Config)
                    or isinstance(postprocess_fn, ConfigDict)
                ):
                    self.image_postprocess_fns[i] = BUILDER.build(postprocess_fn)
                elif callable(postprocess_fn):
                    self.image_postprocess_fns[i] = postprocess_fn.build()
                else:
                    self.image_postprocess_fns[i] = None
        if isinstance(self.image_inputs, str):
            self.image_inputs = [self.image_inputs]
        if isinstance(self.image_task_names, str):
            self.image_task_names = [self.image_task_names]
        if self.image_task_names is not None:
            assert len(self.image_task_names) == len(self.image_inputs)
        self.generation_images = generation_images
        self.reference_images = reference_images
        self.image_vprompt_masks = image_vprompt_masks
        if isinstance(self.generation_images, str):
            self.generation_images = [self.generation_images]
        if isinstance(self.reference_images, str):
            self.reference_images = [self.reference_images]
        if isinstance(self.image_vprompt_masks, str):
            self.image_vprompt_masks = [(self.image_vprompt_masks,)]
        if self.generation_images is not None:
            assert len(self.generation_images) in [1, len(self.image_inputs)]
            if len(self.generation_images) == 1:
                self.generation_images = [self.generation_images[0]] * len(self.image_inputs)
            self.evaluation_image_files = self.generation_images
            self.generation_images = [load_image(img) for img in self.generation_images]
        if self.reference_images is not None:
            assert len(self.reference_images) in [1, len(self.image_inputs)]
            if len(self.reference_images) == 1:
                self.reference_images = [self.reference_images[0]] * len(self.image_inputs)
            self.reference_images = [load_image(img) if img is not None else None for img in self.reference_images]
        else:
            self.reference_images = [None]
        if self.image_vprompt_masks is not None and self.image_inputs is not None:
            assert len(self.image_vprompt_masks) in [1, len(self.image_inputs)]
            if len(self.image_vprompt_masks) == 1:
                self.image_vprompt_masks = [self.image_vprompt_masks[0]] * len(self.image_inputs)
            self.image_vprompt_masks_files = self.image_vprompt_masks
            self.image_vprompt_masks = [
                (
                    torch.stack([load_image(img, mode="L", to_tensor=True) for img in vprompt_mask])
                    if vprompt_mask[0] is not None
                    else None
                )
                for vprompt_mask in self.image_vprompt_masks
            ]
        else:
            self.image_vprompt_masks = [None] * len(self.image_inputs) if self.image_inputs is not None else [None]
            self.image_vprompt_masks_files = (
                [None] * len(self.image_inputs) if self.image_inputs is not None else [None]
            )
        if self.image_postprocess_fns is not None:
            assert len(self.image_postprocess_fns) in [1, len(self.image_inputs)]
            if len(self.image_postprocess_fns) == 1:
                self.image_postprocess_fns = [self.image_postprocess_fns[0]] * len(self.image_inputs)
        else:
            self.image_postprocess_fns = [None] * len(self.image_inputs) if self.image_inputs is not None else [None]
        if prompt_template is None:
            instruction = "{input}"
        else:
            if isinstance(prompt_template, str):  # for resume
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get("INSTRUCTION", "{input}")
            if system != "":
                system = prompt_template.get("SYSTEM", "{system}\n").format(system=system)
            stop_words += prompt_template.get("STOP_WORDS", [])
        if stop_word is not None:
            # TODO: deprecation, v0.3.0
            warnings.warn(
                ("The `stop_word` argument is deprecated and will be removed " "in v0.3.0, use `stop_words` instead."),
                DeprecationWarning,
            )
            stop_words.append(stop_word)
        self.instruction = instruction
        self.system = system
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = None
        self.extra_image_processor = None
        self.visualizer = None
        if image_processor is not None:
            self.image_processor = BUILDER.build(image_processor)
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        if visualizer is not None:
            self.visualizer = BUILDER.build(visualizer)

        # default generation config
        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1,
            top_p=None,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            ),
        )
        default_generation_kwargs.update(generation_kwargs)
        self.gen_config = GenerationConfig(**default_generation_kwargs)

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(StopWordStoppingCriteria(self.tokenizer, word))

        self.is_first_run = True

        self.seg_token_idx = -1
        self.cls_token_idx = -1
        self.pstart_token_idx = -1
        self.pend_token_idx = -1
        if special_tokens is not None:
            assert all(token in DEFAULT_SPECIAL_TOKENS for token in special_tokens)
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

            if DEFAULT_SEG_TOKEN in special_tokens:
                self.seg_token_idx = self.tokenizer(DEFAULT_SEG_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_CLS_TOKEN in special_tokens:
                self.cls_token_idx = self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_PSTART_TOKEN in special_tokens:
                self.pstart_token_idx = self.tokenizer(DEFAULT_PSTART_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_PEND_TOKEN in special_tokens:
                self.pend_token_idx = self.tokenizer(DEFAULT_PEND_TOKEN, add_special_tokens=False)["input_ids"][0]

    @master_only
    def _save_eval_output(self, runner, eval_outputs, data_names=None):
        save_path = osp.join(runner.log_dir, "vis_data", f"eval_outputs_iter_{runner.iter}.txt")
        mkdir_or_exist(osp.dirname(save_path))
        with open(save_path, "a", encoding="utf-8") as f:
            for i, output in enumerate(eval_outputs):
                suffix = f"{data_names[i]}" if data_names is not None else ""
                f.write(f"Eval output {suffix}:\n{output}\n\n")

    def _decode_phrases_ids(self, phrases_ids):
        phrases = []
        for phrase_id in phrases_ids:
            if (phrase_id < 0).any():
                phrase = ""
            else:
                phrase = self.tokenizer.decode(phrase_id).strip()
            phrases.append(phrase)
        return phrases

    @staticmethod
    def _resolve_sample_identity(sample_file, task_name=None):
        image_stem = osp.splitext(osp.basename(sample_file))[0]
        if task_name is not None:
            return task_name, f"{image_stem}_{task_name}"
        return image_stem, image_stem

    def _eval_images(self, runner, model, device, max_new_tokens=None, save_eval_output=False):
        if save_eval_output:
            eval_outputs = []
            data_names = []

        for idx, (
            sample_image,
            sample_input,
            sample_image_file,
            sample_vprompt_mask,
            sample_vprompt_mask_file,
            postprocess_fn,
        ) in enumerate(
            zip(
                self.generation_images,
                self.image_inputs,
                self.evaluation_image_files,
                self.image_vprompt_masks,
                self.image_vprompt_masks_files,
                self.image_postprocess_fns,
            )
        ):
            task_name = self.image_task_names[idx] if self.image_task_names is not None else None
            data_name, output_suffix = self._resolve_sample_identity(sample_image_file, task_name)
            image = sample_image
            if self.expand2square:
                image = expand2square(
                    sample_image,
                    tuple(int(x * 255) for x in self.image_processor.image_mean),
                )
            output = self.image_processor.preprocess(image, return_tensors="pt")
            pixel_values = output["pixel_values"] if output["pixel_values"].ndim == 4 else output["pixel_values"]
            pixel_values = pixel_values.to(device)
            image_grid_thw = output.get("image_grid_thw", None)
            image_grid_thw = image_grid_thw.to(device) if image_grid_thw is not None else None

            sample_input = (self.image_token or f"{DEFAULT_IMAGE_TOKEN}\n") + sample_input
            input = (self.system + self.instruction).format(input=sample_input, round=1, **runner.cfg)

            if image_grid_thw is not None and self.use_placeholder:
                merge_length = getattr(self.image_processor, "merge_size", 1) ** 2
                index = 0
                while DEFAULT_IMAGE_TOKEN in input:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    input = input.replace(DEFAULT_IMAGE_TOKEN, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                input = input.replace("<|placeholder|>", DEFAULT_PLACEHOLDER_TOKEN)

            input_ids = []
            pattern = f"({'|'.join(re.escape(token) for token in TOKEN2INDEX.keys())})"
            chunks = [chunk for chunk in re.split(pattern, input) if chunk.strip() != ""]
            input_ids = [
                (
                    [TOKEN2INDEX[chunk]]
                    if chunk in TOKEN2INDEX
                    else self.tokenizer.encode(chunk, add_special_tokens=(idx == 0))
                )
                for idx, chunk in enumerate(chunks)
            ]
            input_ids = [id for sublist in input_ids if isinstance(sublist, list) for id in sublist]
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
            cond_ids = get_cond_ids(
                input_ids,
                self.pstart_token_idx,
                self.pend_token_idx,
                self.cls_token_idx,
                model.cond_type if hasattr(model, "cond_type") else "phrase",
            )

            data_dict = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "input_ids": input_ids,
                "cond_ids": cond_ids,
            }

            data_samples = DataSample()
            data_samples.image_files = [sample_image_file]
            data_samples.image_sizes = [(sample_image.size[1], sample_image.size[0])]

            extra_pixel_values = None
            vprompt_masks = None

            if hasattr(model, "segmentor") and self.extra_image_processor is not None:
                extra_output = self.extra_image_processor.preprocess(
                    sample_image, vprompt_masks=sample_vprompt_mask, return_tensors="pt"
                )
                extra_pixel_values = extra_output["pixel_values"].to(device)
                vprompt_masks = extra_output["vprompt_masks"] if "vprompt_masks" in extra_output else None

                data_samples.scaled_sizes = extra_output["scaled_sizes"].tolist()

                if vprompt_masks is not None:
                    data_samples.class_labels = [
                        torch.tensor([i for i in range(len(vprompt_masks))], dtype=torch.int64)
                    ]
                    data_samples.sampled_labels = [[i for i in range(len(vprompt_masks))]]
                    data_samples.contiguous_labels = [[i for i in range(len(vprompt_masks))]]
                    vprompt_masks = [vprompt_masks.to(device)]

                data_dict["extra_pixel_values"] = extra_pixel_values
                data_dict["vprompt_masks"] = vprompt_masks

            if vprompt_masks is not None:
                mask_suffix = "+".join(
                    [
                        osp.splitext(osp.basename(vprompt_mask_file))[0]
                        for vprompt_mask_file in sample_vprompt_mask_file
                    ]
                )
                output_suffix = f"{output_suffix}+{mask_suffix}"

            input_phrases = []
            output_phrases = []
            if hasattr(self, "pstart_token_idx") and hasattr(self, "pend_token_idx"):
                input_phrases_ids = get_phrases_ids(input_ids, self.pstart_token_idx, self.pend_token_idx)
                input_phrases = self._decode_phrases_ids(input_phrases_ids)

            metadata = MetadataCatalog.get(f"{data_name}") if data_name in MetadataCatalog.list() else None
            if postprocess_fn is not None:
                model.postprocess_fn = postprocess_fn

            with torch.no_grad():
                try:
                    mlm_outputs, seg_outputs = model(
                        data_dict,
                        data_samples,
                        mode="predict",
                        metadata=metadata,
                        generation_config=self.gen_config,
                        stopping_criteria=self.stop_criteria,
                        do_postprocess=True,
                    )
                except Exception as e:
                    print_log(f"Error in {data_name} prediction: {e}\n{traceback.format_exc()}", logger="current")
                    continue
            output_ids = mlm_outputs.sequences
            generation_output = self.tokenizer.decode(output_ids[0]).strip()

            if hasattr(self, "pstart_token_idx") and hasattr(self, "pend_token_idx"):
                output_phrases_ids = get_phrases_ids(
                    output_ids[0],
                    self.pstart_token_idx,
                    self.pend_token_idx,
                )
                output_phrases = self._decode_phrases_ids(output_phrases_ids)
            phrases = output_phrases or input_phrases

            input = re.sub(f"({re.escape(DEFAULT_PLACEHOLDER_TOKEN)}\\s*)+", DEFAULT_IMAGE_TOKEN, input)
            runner.logger.info(f"Sample output of {data_name}:\n" f"{input + generation_output}\n")
            if seg_outputs is not None and save_eval_output and self.visualizer is not None:
                save_path = osp.join(
                    runner.log_dir,
                    "vis_data",
                    f"eval_outputs_{runner.iter}_{output_suffix}.png",
                )
                mkdir_or_exist(osp.dirname(save_path))
                if data_name in MetadataCatalog.list():
                    self.visualizer.metadata = metadata
                try:
                    self.visualizer.draw_predictions(
                        sample_image,
                        data_name=data_name,
                        output_file=save_path,
                        phrases=phrases,
                        **(seg_outputs[0]),
                    )
                except Exception as e:
                    print_log(f"Error in {data_name} visualization: {e}\n{traceback.format_exc()}", logger="current")
                    continue
            if save_eval_output:
                eval_outputs.append(f"{input + generation_output}\n")
                data_names.append(data_name)

        if save_eval_output:
            self._save_eval_output(runner, eval_outputs, data_names)

    def _eval_language(self, runner, model, device, max_new_tokens=None, save_eval_output=False):
        if save_eval_output:
            eval_outputs = []

        for sample_input in self.image_inputs:
            inputs = (self.system + self.instruction).format(input=sample_input, round=1, **runner.cfg)
            input_ids = self.tokenizer.encode(inputs, return_tensors="pt")
            input_ids = input_ids.to(device)
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=self.gen_config,
                stopping_criteria=self.stop_criteria,
            )
            generation_output = self.tokenizer.decode(generation_output[0]).strip()
            runner.logger.info(f"Sample output:\n{generation_output}\n")
            if save_eval_output:
                eval_outputs.append(f"{generation_output}\n")

        if save_eval_output:
            self._save_eval_output(runner, eval_outputs)

    @master_only
    def _generate_samples(self, runner, max_new_tokens=None, save_eval_output=False):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        if self.is_first_run:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            model.to(device)
            self.is_first_run = False

        is_checkpointing = (
            model.llm.is_gradient_checkpointing if model.llm is not None else model.vlm.is_gradient_checkpointing
        )
        use_cache = model.llm.config.use_cache if model.llm is not None else model.vlm.config.text_config.use_cache

        # Cast to inference mode
        model.activation_checkpointing_disable()
        if model.llm is not None:
            model.llm.config.use_cache = True
        elif model.vlm is not None:
            model.vlm.config.text_config.use_cache = True
        model.eval()
        if self.generation_images is not None:
            self._eval_images(runner, model, device, max_new_tokens, save_eval_output)
        if self.generation_images is None:
            self._eval_language(runner, model, device, max_new_tokens, save_eval_output)

        # Cast to training mode
        if is_checkpointing:
            model.activation_checkpointing_enable()
        if model.llm is not None:
            model.llm.config.use_cache = use_cache
        elif model.vlm is not None:
            model.vlm.config.text_config.use_cache = use_cache
        model.train()

    def before_train(self, runner):
        runner.logger.info("before_train in GenerationChatHook.")
        self._generate_samples(runner, max_new_tokens=50)

    def _is_save_checkpoint(self, runner):
        hooks = runner.hooks
        checkpoint_hook = None
        for hook in hooks:
            if type(hook).__name__ == "CheckpointHook":
                checkpoint_hook = hook
                break
        if checkpoint_hook is None or checkpoint_hook.by_epoch:
            return False

        if checkpoint_hook.every_n_train_iters(runner, checkpoint_hook.interval, checkpoint_hook.save_begin) or (
            checkpoint_hook.save_last and checkpoint_hook.is_last_train_iter(runner)
        ):
            return True

        return False

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        if self.every_n_iters is None:
            return

        save_eval_output = self._is_save_checkpoint(runner)

        do_chat = save_eval_output or self.every_n_train_iters(runner, self.every_n_iters)
        if not do_chat:
            return

        runner.logger.info("after_train_iter in GenerationChatHook.")
        self._generate_samples(runner, save_eval_output=save_eval_output)

    def after_train(self, runner):
        runner.logger.info("after_train in GenerationChatHook.")
        self._generate_samples(runner)

    def after_val(self, runner) -> None:
        if self.every_n_iters is not None:
            return
        runner.logger.info("after_val in GenerationChatHook.")
        self._generate_samples(runner)


def get_phrases_ids(input_ids, pstart_token_idx, pend_token_idx):
    if input_ids.ndim == 2:
        assert input_ids.shape[0] == 1
        input_ids = input_ids[0, :]

    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases_ids = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrases_ids.append(input_ids[ps + 1 : pe - 1])
    return phrases_ids


def get_cond_ids(input_ids, pstart_token_idx, pend_token_idx, cls_token_idx, cond_type="phrase"):
    cond_ids = torch.full(input_ids.shape, -1, device=input_ids.device)
    pstart_idx = [i for i, x in enumerate(input_ids[0, :]) if x == pstart_token_idx]
    pend_idx = [i for i, x in enumerate(input_ids[0, :]) if x == pend_token_idx]
    cls_idx = [i for i, x in enumerate(input_ids[0, :]) if x == cls_token_idx]

    if len(pstart_idx) == 0 and len(pend_idx) == 0 and len(cls_idx) == 0:
        return None

    if cond_type in ["phrase", "all"]:
        for i, (ps, pe) in enumerate(zip(pstart_idx, pend_idx)):
            cond_ids[:, ps : pe + 1] = i
    if cond_type in ["cls", "all"]:
        for i, ci in enumerate(cls_idx):
            cond_ids[:, ci] = i

    return cond_ids
