import logging
import os.path as osp
import re
import time
import traceback
from typing import Dict, List, Optional, Sequence, Union

import torch
from mmengine.logging import HistoryBuffer
from mmengine.runner.amp import autocast
from mmengine.runner.loops import (
    BaseLoop,
    IterBasedTrainLoop,
    _InfiniteDataloaderIterator,
    _parse_losses,
    _update_losses,
)
from torch.utils.data import DataLoader

from xsam.evaluation.evaluators import BaseEvaluator
from xsam.registry import BUILDER
from xsam.utils.config import setup_model_config
from xsam.utils.constants import DEFAULT_SEG_TOKEN
from xsam.utils.logging import print_log


def _get_gcg_phrases(input_ids, tokenizer, pstart_token_idx, pend_token_idx):
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


def _get_gcg_caption(llm_generation_output):
    if DEFAULT_SEG_TOKEN not in llm_generation_output:
        return ""

    parts = re.split(
        r"(?<=[.!?])(?:\s+|$)|(?:\n+)|(?<=\s)(?=[A-Z])",
        llm_generation_output,
    )
    sents = [part.strip() for part in parts if part.strip() and DEFAULT_SEG_TOKEN not in part]
    caption = " ".join(sents)
    caption = re.sub(r"<.*?>", "", caption)
    caption = " ".join(caption.split()).strip("'").strip('"').strip()
    return caption


# Refer to https://github.com/open-mmlab/mmengine/pull/1548
class X2SamInfiniteDataloaderIterator(_InfiniteDataloaderIterator):
    def skip_iter(self, iters: int) -> None:
        if iters <= 0:
            return

        # first try to skip iters by sampler, if failed, then skip iters by iterator
        sampler = next(
            (
                candidate
                for candidate in (
                    getattr(self._dataloader, "sampler", None),
                    getattr(getattr(self._dataloader, "batch_sampler", None), "sampler", None),
                )
                if hasattr(candidate, "set_epoch")
            ),
            None,
        )
        if sampler is not None:
            num_iters_per_epoch = len(self._dataloader)
            epoch, step = divmod(iters, num_iters_per_epoch)
            can_reset_by_epoch = True
            try:
                sampler.set_epoch(epoch, step)
            except TypeError:
                if step == 0:
                    sampler.set_epoch(epoch)
                else:
                    can_reset_by_epoch = False
            if can_reset_by_epoch:
                self._epoch = epoch
                self._iterator = iter(self._dataloader)
                return

        for _ in range(iters):
            self._next_data(skip_loading=True)

    def __next__(self) -> Sequence[dict]:
        return self._next_data()

    def _next_data(self, skip_loading=False) -> Sequence[dict]:
        data = None
        try:
            if skip_loading:
                self._iterator._next_index()
            else:
                data = next(self._iterator)
        except StopIteration:
            print_log(
                "Reach the end of the dataloader, it will be "
                "restarted and continue to iterate. It is "
                "recommended to use "
                "`mmengine.dataset.InfiniteSampler` to enable the "
                "dataloader to iterate infinitely.",
                logger="current",
                level=logging.WARNING,
            )
            self._epoch += 1
            if hasattr(self._dataloader, "sampler") and hasattr(self._dataloader.sampler, "set_epoch"):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, "batch_sampler") and hasattr(
                self._dataloader.batch_sampler.sampler, "set_epoch"
            ):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(30)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
        return data


class X2SamIterBasedTrainLoop(IterBasedTrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader_iterator = X2SamInfiniteDataloaderIterator(self.dataloader)

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook("before_train")
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook("before_train_epoch")
        if self._iter > 0:
            resume_skip_iters = 0
            if hasattr(self.runner.optim_wrapper, "pop_resume_skip_iters"):
                resume_skip_iters = self.runner.optim_wrapper.pop_resume_skip_iters()
            total_skip_iters = self._iter + resume_skip_iters
            print_log(
                f"Advance dataloader {total_skip_iters} steps to skip data that has already been trained",
                logger="current",
                level=logging.WARNING,
            )
            self.dataloader_iterator.skip_iter(total_skip_iters)

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (
                self.runner.val_loop is not None
                and self._iter >= self.val_begin
                and (self._iter % self.val_interval == 0 or self._iter == self._max_iters)
            ):
                self.runner.val_loop.run()

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")
        return self.runner.model


class TrainLoop(X2SamIterBasedTrainLoop):
    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_iters: Optional[int] = None,
        max_epochs: Union[int, float] = None,
        **kwargs,
    ) -> None:
        if max_iters is None and max_epochs is None:
            raise RuntimeError("Please specify the `max_iters` or " "`max_epochs` in `train_cfg`.")
        elif max_iters is not None and max_epochs is not None:
            raise RuntimeError("Only one of `max_iters` or `max_epochs` can " "exist in `train_cfg`.")
        else:
            if max_iters is not None:
                iters = int(max_iters)
                assert iters == max_iters, "`max_iters` should be a integer " f"number, but get {max_iters}"
            elif max_epochs is not None:
                if isinstance(dataloader, dict):
                    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
                    dataloader = runner.build_dataloader(dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
                iters = max_epochs * len(dataloader)
            else:
                raise NotImplementedError

        print_log(f"Training max_iters: {iters}.", logger="current")
        super().__init__(runner=runner, dataloader=dataloader, max_iters=iters, **kwargs)

    @staticmethod
    def _get_batch_modality(data_batch: Dict) -> str:
        data_samples = data_batch.get("data_samples", None)
        task_names = getattr(data_samples, "task_names", None)
        if task_names:
            modalities = {task_name.split("_", 1)[0] for task_name in task_names if task_name}
            if len(modalities) != 1:
                raise RuntimeError(f"Mixed task names in the same batch are not supported: {task_names}")
            modality = next(iter(modalities))
            if modality in {"img", "vid"}:
                return modality

        data_dict = data_batch.get("data_dict", {})
        if "pixel_values_videos" in data_dict:
            return "vid"
        return "img"

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        if hasattr(self.runner.optim_wrapper, "set_current_modality"):
            modality = self._get_batch_modality(data_batch)
            self.runner.optim_wrapper.set_current_modality(modality)
        super().run_iter(data_batch)


class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[BaseEvaluator, Dict, List],
        fp16: bool = False,
    ) -> None:
        super().__init__(runner, dataloader)

        self.evaluator = self._build_evaluators(evaluator)
        self.dataset = self._collect_datasets(self.dataloader.dataset)
        self._evaluator_by_data_name = {
            evaluator.data_name: evaluator for evaluator in self.evaluator if getattr(evaluator, "data_name", None)
        }
        self._dataset_by_data_name = {
            dataset.data_name: dataset for dataset in self.dataset if getattr(dataset, "data_name", None)
        }
        self._datasets_by_task_name = self._group_datasets_by_task_name(self.dataset)
        self._prepare_evaluators()

        if hasattr(self.dataloader.dataset, "metainfo"):
            self.runner.visualizer.dataset_meta = self.dataloader.dataset.metainfo
        elif len(self.dataset) == 1 and hasattr(self.dataset[0], "metainfo"):
            self.runner.visualizer.dataset_meta = self.dataset[0].metainfo
        else:
            print_log(
                f"Dataset {self.dataloader.dataset.__class__.__name__} has no "
                "metainfo. ``dataset_meta`` in evaluator, metric and "
                "visualizer will be None.",
                logger="current",
                level=logging.WARNING,
            )
        self.fp16 = fp16
        self.val_loss: Dict[str, HistoryBuffer] = dict()

        # NOTE: at val-loop build time the model is not built/wrapped yet
        # (``_FlexibleRunner`` builds it via ``strategy.prepare`` *after*
        # constructing the loops), so ``runner.model`` is still the model
        # config dict. Defer generation-config setup to the first ``run``.
        self.stop_criteria = None
        self.gen_config = None
        self._model_config_ready = False

    @staticmethod
    def _build_evaluators(evaluator: Union[BaseEvaluator, Dict, List]) -> List[BaseEvaluator]:
        if isinstance(evaluator, list):
            val_evaluators = evaluator
        else:
            val_evaluators = [evaluator]

        return [BUILDER.build(item) if isinstance(item, dict) else item for item in val_evaluators]

    @staticmethod
    def _collect_datasets(dataset) -> List:
        return list(getattr(dataset, "datasets", [dataset]))

    @staticmethod
    def _group_datasets_by_task_name(datasets: List) -> Dict[str, List]:
        datasets_by_task_name: Dict[str, List] = {}
        for dataset in datasets:
            task_name = getattr(dataset, "task_name", None)
            if task_name is None:
                continue
            datasets_by_task_name.setdefault(task_name, []).append(dataset)
        return datasets_by_task_name

    def _prepare_evaluators(self) -> None:
        for evaluator in self.evaluator:
            data_name = getattr(evaluator, "data_name", None)
            dataset = self._dataset_by_data_name.get(data_name)
            if dataset is not None and hasattr(dataset, "metadata"):
                evaluator.metadata = dataset.metadata
            work_dir = getattr(self.runner, "work_dir", None)
            if data_name is not None and work_dir is not None and getattr(evaluator, "output_dir", None) is None:
                evaluator.output_dir = osp.join(work_dir, "pred_data", data_name)
            # Online val/test always evaluates in-memory predictions, never cached files.
            if hasattr(evaluator, "support_loading"):
                evaluator.support_loading = False

    @staticmethod
    def _get_unique_metainfo_value(data_batch: Sequence[dict], key: str):
        data_samples = data_batch.get("data_samples", None)
        metainfo = getattr(data_samples, "metainfo", {}) if data_samples is not None else {}
        values = metainfo.get(key, None)
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            values = [values]
        unique_values = [value for value in dict.fromkeys(values) if value is not None]
        if len(unique_values) > 1:
            raise RuntimeError(f"Validation batch contains multiple {key}: {unique_values}")
        return unique_values[0] if unique_values else None

    def _get_dataset(self, data_name: Optional[str], task_name: Optional[str]):
        if data_name is not None and data_name in self._dataset_by_data_name:
            return self._dataset_by_data_name[data_name]

        if task_name is not None and task_name in self._datasets_by_task_name:
            datasets = self._datasets_by_task_name[task_name]
            if len(datasets) == 1:
                return datasets[0]
            raise RuntimeError(
                f"Cannot route validation batch for task {task_name!r}; "
                f"multiple datasets are configured: {[dataset.data_name for dataset in datasets]}"
            )

        if len(self.dataset) == 1:
            return self.dataset[0]
        raise RuntimeError(f"Cannot route validation batch with data_name={data_name!r}, task_name={task_name!r}")

    def _get_evaluator(self, data_name: Optional[str], task_name: Optional[str]):
        if data_name is not None and data_name in self._evaluator_by_data_name:
            return self._evaluator_by_data_name[data_name]

        dataset = self._get_dataset(data_name, task_name)
        dataset_data_name = getattr(dataset, "data_name", None)
        if dataset_data_name in self._evaluator_by_data_name:
            return self._evaluator_by_data_name[dataset_data_name]

        if len(self.evaluator) == 1:
            return self.evaluator[0]
        raise RuntimeError(f"Cannot find evaluator for data_name={data_name!r}, task_name={task_name!r}")

    @staticmethod
    def _extract_eval_inputs(data_batch: Sequence[dict]):
        data_samples = data_batch.get("data_samples", None)
        metainfo = getattr(data_samples, "metainfo", {}) if data_samples is not None else {}
        input_infos = metainfo.get("image_infos", None)
        if input_infos is None:
            input_infos = metainfo.get("video_infos", None)
        if input_infos is None:
            raise RuntimeError("Validation batch does not contain image_infos or video_infos for evaluator inputs.")
        return input_infos

    @staticmethod
    def _extract_predictions(outputs):
        if isinstance(outputs, tuple):
            return outputs[-1]
        return outputs

    @staticmethod
    def _attach_gcg_outputs(outputs, predictions, task_name, model):
        """Decode and merge ``gcg_phrases``/``gcg_caption`` into seg predictions.

        For GCG tasks the model runs in ``predict`` mode and returns
        ``(mlm_outputs, seg_outputs)``; the generated phrases/caption must be
        decoded from the LLM sequences and attached to each seg prediction so
        the GCG evaluator can consume them (mirrors ``tools/eval.py``).
        """
        if task_name is None or "gcg" not in task_name:
            return
        if not isinstance(outputs, tuple) or predictions is None:
            return
        mlm_outputs = outputs[0]
        if mlm_outputs is None or not hasattr(mlm_outputs, "sequences"):
            return

        tokenizer = model.tokenizer
        generations = tokenizer.batch_decode(mlm_outputs.sequences)
        gcg_phrases = [
            _get_gcg_phrases(output_ids, tokenizer, model.pstart_token_idx, model.pend_token_idx)
            for output_ids in mlm_outputs.sequences
        ]
        gcg_captions = [_get_gcg_caption(generation) for generation in generations]
        for i, seg_output in enumerate(predictions):
            if isinstance(seg_output, list):
                for _seg_output in seg_output:
                    _seg_output.update({"gcg_phrases": gcg_phrases[i], "gcg_caption": gcg_captions[i]})
            elif isinstance(seg_output, dict):
                seg_output.update({"gcg_phrases": gcg_phrases[i], "gcg_caption": gcg_captions[i]})

    @staticmethod
    def _get_postprocess_model(model):
        return getattr(model, "module", model)

    @staticmethod
    def _get_forward_model(model):
        # MMDeepSpeedEngineWrapper is a plain (non-nn.Module) wrapper that holds
        # the DeepSpeed engine in ``.model``; DDP/FSDP wrappers hold the real
        # model in ``.module``. Avoid relying solely on ``hasattr(model, "model")``
        # because the underlying model's ``__getattr__`` may forward unknown
        # attributes (e.g. to ``llm``) and falsely report a ``model`` attribute.
        if not isinstance(model, torch.nn.Module) and hasattr(model, "model"):
            return model.model
        return getattr(model, "module", model)

    def _ensure_model_config(self):
        if self._model_config_ready:
            return
        model = self._get_forward_model(self.runner.model)
        self.stop_criteria, self.gen_config = setup_model_config(model, self.runner.cfg)
        self._model_config_ready = True

    def _build_forward_kwargs(self, mode, metadata):
        kwargs = dict(mode=mode, metadata=metadata, do_postprocess=True)
        if mode == "predict":
            kwargs.update(generation_config=self.gen_config, stopping_criteria=self.stop_criteria)
        return kwargs

    def run(self) -> dict:
        """Launch validation."""
        self._ensure_model_config()
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()

        # clear val loss
        self.val_loss.clear()
        for evaluator in self.evaluator:
            evaluator.reset()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = {}
        for evaluator in self.evaluator:
            if eval_metrics := evaluator.evaluate():
                metrics.update(eval_metrics)

        if self.val_loss:
            loss_dict = _parse_losses(self.val_loss, "val")
            metrics.update(loss_dict)

        self.runner.call_hook("after_val_epoch", metrics=metrics)
        self.runner.call_hook("after_val")
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)
        data_name = self._get_unique_metainfo_value(data_batch, "data_names")
        task_name = self._get_unique_metainfo_value(data_batch, "task_names")
        dataset = self._get_dataset(data_name, task_name)
        evaluator = self._get_evaluator(data_name, task_name)
        forward_model = self._get_forward_model(self.runner.model)
        postprocess_model = self._get_postprocess_model(self.runner.model)
        original_postprocess_fn = getattr(postprocess_model, "postprocess_fn", None)
        mode = "tensor" if getattr(dataset, "output_ids_with_output", True) else "predict"
        metadata = getattr(dataset, "metadata", None)

        try:
            if hasattr(dataset, "postprocess_fn"):
                postprocess_model.postprocess_fn = dataset.postprocess_fn
            # outputs should be sequence of BaseDataElement
            with autocast(enabled=self.fp16):
                data_batch = forward_model.data_preprocessor(data_batch, False)
                if hasattr(self.runner.model, "_cast_inputs_half"):
                    data_batch = self.runner.model._cast_inputs_half(data_batch)
                outputs = forward_model(**data_batch, **self._build_forward_kwargs(mode, metadata))
        except Exception as e:
            print_log(
                f"Skip validation batch {idx}: {e}\n{traceback.format_exc()}",
                logger="current",
                level=logging.WARNING,
            )
            self.runner.call_hook("after_val_iter", batch_idx=idx, data_batch=data_batch, outputs=None)
            return
        finally:
            if hasattr(postprocess_model, "postprocess_fn"):
                postprocess_model.postprocess_fn = original_postprocess_fn

        outputs, self.val_loss = _update_losses(outputs, self.val_loss)
        predictions = self._extract_predictions(outputs)
        if predictions is None:
            print_log(
                f"Skip validation batch {idx}: model did not return predictions.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            self._attach_gcg_outputs(outputs, predictions, task_name, forward_model)
            evaluator.process(self._extract_eval_inputs(data_batch), predictions)
        self.runner.call_hook("after_val_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs)


class TestLoop(ValLoop):
    """Loop for test.

    The inference path is identical to :class:`ValLoop` (same dataset/evaluator
    routing, generation-config setup and ``predict``/``tensor`` mode handling);
    only the lifecycle hook names differ.
    """

    def run(self) -> dict:
        """Launch test."""
        self._ensure_model_config()
        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        # clear loss buffer
        self.val_loss.clear()
        for evaluator in self.evaluator:
            evaluator.reset()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = {}
        for evaluator in self.evaluator:
            if eval_metrics := evaluator.evaluate():
                metrics.update(eval_metrics)

        if self.val_loss:
            loss_dict = _parse_losses(self.val_loss, "test")
            metrics.update(loss_dict)

        self.runner.call_hook("after_test_epoch", metrics=metrics)
        self.runner.call_hook("after_test")
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch (inference identical to ``ValLoop``)."""
        self.runner.call_hook("before_test_iter", batch_idx=idx, data_batch=data_batch)
        data_name = self._get_unique_metainfo_value(data_batch, "data_names")
        task_name = self._get_unique_metainfo_value(data_batch, "task_names")
        dataset = self._get_dataset(data_name, task_name)
        evaluator = self._get_evaluator(data_name, task_name)
        forward_model = self._get_forward_model(self.runner.model)
        postprocess_model = self._get_postprocess_model(self.runner.model)
        original_postprocess_fn = getattr(postprocess_model, "postprocess_fn", None)
        mode = "tensor" if getattr(dataset, "output_ids_with_output", True) else "predict"
        metadata = getattr(dataset, "metadata", None)

        try:
            if hasattr(dataset, "postprocess_fn"):
                postprocess_model.postprocess_fn = dataset.postprocess_fn
            # outputs should be sequence of BaseDataElement
            with autocast(enabled=self.fp16):
                data_batch = forward_model.data_preprocessor(data_batch, False)
                if hasattr(self.runner.model, "_cast_inputs_half"):
                    data_batch = self.runner.model._cast_inputs_half(data_batch)
                outputs = forward_model(**data_batch, **self._build_forward_kwargs(mode, metadata))
        except Exception as e:
            print_log(
                f"Skip test batch {idx}: {e}\n{traceback.format_exc()}",
                logger="current",
                level=logging.WARNING,
            )
            self.runner.call_hook("after_test_iter", batch_idx=idx, data_batch=data_batch, outputs=None)
            return
        finally:
            if hasattr(postprocess_model, "postprocess_fn"):
                postprocess_model.postprocess_fn = original_postprocess_fn

        outputs, self.val_loss = _update_losses(outputs, self.val_loss)
        predictions = self._extract_predictions(outputs)
        if predictions is None:
            print_log(
                f"Skip test batch {idx}: model did not return predictions.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            self._attach_gcg_outputs(outputs, predictions, task_name, forward_model)
            evaluator.process(self._extract_eval_inputs(data_batch), predictions)
        self.runner.call_hook("after_test_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs)
