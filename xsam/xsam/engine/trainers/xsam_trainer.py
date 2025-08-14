import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    _is_peft_model,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
from xtuner.dataset.samplers import LengthGroupedSampler

from ...dataset.samplers import SourceGroupedSampler
from ..utils.peft import (
    get_mm_adapter_state_maybe_zero_3,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)


class XSamTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            return LengthGroupedSampler(
                self.train_dataset,
                per_device_batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
                length_property="modality_length",
            )
        elif self.args.group_by_data_source:
            return SourceGroupedSampler(
                self.train_dataset,
                per_device_batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
                length_property="source_length",
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            specical_lr_mapper = {}
            if self.args.visual_encoder_lr is not None:
                specical_lr_mapper["mm_conv_encoder"] = self.args.visual_encoder_lr
            if self.args.segmentor_encoder_lr is not None:
                specical_lr_mapper["mm_seg_encoder.seg_encoder.vision_encoder"] = self.args.segmentor_encoder_lr
            if self.args.visual_projector_lr is not None:
                specical_lr_mapper["visual_projector"] = self.args.visual_projector_lr
                specical_lr_mapper["seg_projector"] = self.args.visual_projector_lr
            if self.args.segmentor_decoder_lr is not None:
                specical_lr_mapper["mm_seg_decoder.seg_decoder.model.mask_decoder"] = self.args.segmentor_decoder_lr
                specical_lr_mapper["mm_seg_decoder.seg_decoder.logit_scale"] = self.args.segmentor_decoder_lr

            if len(specical_lr_mapper) > 0:
                special_lr_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if any(module_keyword in name for module_keyword in specical_lr_mapper)
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in specical_lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (n in decay_parameters and n in module_parameters and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (n not in decay_parameters and n in module_parameters and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logging.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logging.debug(f"bitsandbytes: will optimize {module} in fp32")
                logging.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # for lora, addtional save non-lora parameters
        if self.args.lora_enable:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                self.model.save_pretrained(output_dir, state_dict=state_dict)
                torch.save(
                    non_lora_state_dict,
                    os.path.join(output_dir, "non_lora_trainables.bin"),
                )

        # for adapter, only save adapter
        if not getattr(self.args, "unfreeze_backbone", False) and not getattr(
            self.args, "unfreeze_mm_seg_decoder", False
        ):
            if getattr(self.args, "unfreeze_mm_projector", False):
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                # Only save Adapter
                keys_to_match = ["visual_projector", "seg_projector"]
                if getattr(self.args, "use_im_start_end", False):
                    keys_to_match.extend(["embed_tokens", "embed_in"])

                # self.model.state_dict().items() instead of self.model.named_parameters()
                # to save batch_norm buffer
                weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.state_dict().items(), keys_to_match)

                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    self.model.config.save_pretrained(output_dir)
                    torch.save(
                        weight_to_save,
                        os.path.join(output_dir, "mm_projector.bin"),
                    )

            if getattr(self.args, "unfreeze_mm_conv_encoder", False):
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                # save mm_conv_encoder
                keys_to_match = ["mm_conv_encoder"]
                if getattr(self.args, "use_im_start_end", False):
                    keys_to_match.extend(["embed_tokens", "embed_in"])

                # self.model.state_dict().items() instead of self.model.named_parameters()
                # to save batch_norm buffer
                weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.state_dict().items(), keys_to_match)

                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    self.model.config.save_pretrained(output_dir)
                    torch.save(weight_to_save, os.path.join(output_dir, "mm_conv_encoder.bin"))
        else:
            super(XSamTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if not getattr(self.args, "unfreeze_backbone", False) and not getattr(
            self.args, "unfreeze_mm_seg_decoder", False
        ):
            pass
        else:
            super(XSamTrainer, self)._save(output_dir, state_dict)

    def update_cached_loss_dict(self, loss_dict):
        if not hasattr(self, "cached_loss_dict"):
            self.cached_loss_dict = defaultdict(list)
            # log_cnt records how many times the loss is logged
            self.log_cnt = 0

        for key, value in loss_dict.items():
            value = value.item() if isinstance(value, torch.Tensor) else value
            if key not in self.cached_loss_dict:
                self.cached_loss_dict[key].append(0.0)
            else:
                self.cached_loss_dict[key].append(value)

            if len(self.cached_loss_dict[key]) > self.state.logging_interval * self.args.gradient_accumulation_steps:
                self.cached_loss_dict[key].pop(0)

        self.log_cnt += 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs, return_dict=True)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # We print different losses
            if isinstance(outputs, dict):
                loss_dict = outputs.get("loss_dict", {})

                self.update_cached_loss_dict(loss_dict)

            if (
                len(self.cached_loss_dict) > 0
                and self.state.global_step > 0
                and self.state.global_step % self.state.logging_interval == 0
                and self.log_cnt % self.args.gradient_accumulation_steps == 0
            ):
                # we log the mean of the self.state.logging_interval steps
                self.log({k: round(np.mean(v), 4) for k, v in self.cached_loss_dict.items()})
                self.log_cnt = 0

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
