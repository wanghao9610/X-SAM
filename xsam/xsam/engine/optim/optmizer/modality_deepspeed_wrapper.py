import logging
from typing import Optional

import torch
from mmengine._strategy.deepspeed import DeepSpeedOptimWrapper
from mmengine.registry import OPTIM_WRAPPERS

from xsam.utils.logging import print_log


@OPTIM_WRAPPERS.register_module()
class ModalityAwareDeepSpeedOptimWrapper(DeepSpeedOptimWrapper):
    def __init__(
        self,
        optimizer,
        image_accumulative_counts: int = 1,
        video_accumulative_counts: int = 1,
    ) -> None:
        super().__init__(optimizer)
        self.image_accumulative_counts = image_accumulative_counts
        self.video_accumulative_counts = video_accumulative_counts
        self.current_modality: Optional[str] = None
        self.inner_count = 0
        self.resume_skip_iters = 0

    def _reset_incomplete_window(self) -> None:
        if hasattr(self.optimizer, "zero_grad"):
            self.optimizer.zero_grad()
        self.current_modality = None
        self.inner_count = 0

    def set_current_modality(self, modality: str) -> None:
        if modality not in {"img", "vid"}:
            raise ValueError(f"Unsupported modality `{modality}`.")
        if self.inner_count > 0 and self.current_modality not in {None, modality}:
            print_log(
                f"Modality changed mid-accumulation window "
                f"({self.current_modality} -> {modality}, "
                f"{self.inner_count} micro-batch(es) accumulated). "
                "Dropping the partial window and starting fresh.",
                logger="current",
                level=logging.WARNING,
            )
            self._reset_incomplete_window()
        self.current_modality = modality

    def _get_accumulative_counts(self) -> int:
        if self.current_modality is None:
            raise RuntimeError(
                "Current batch modality is not set. Please call " "`set_current_modality()` before `update_params()`."
            )
        if self.current_modality == "img":
            return self.image_accumulative_counts
        return self.video_accumulative_counts

    def pop_resume_skip_iters(self) -> int:
        resume_skip_iters = self.resume_skip_iters
        self.resume_skip_iters = 0
        return resume_skip_iters

    def update_params(self, loss: torch.Tensor) -> None:  # type: ignore[override]
        accumulative_counts = self._get_accumulative_counts()
        if accumulative_counts < 1:
            raise ValueError(f"accumulative_counts should be positive, but got {accumulative_counts}.")

        self.backward(loss / accumulative_counts)
        self.inner_count += 1
        if self.inner_count >= accumulative_counts:
            self.step()
            self.inner_count = 0
            self.current_modality = None

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict["image_accumulative_counts"] = self.image_accumulative_counts
        state_dict["video_accumulative_counts"] = self.video_accumulative_counts
        state_dict["current_modality"] = self.current_modality
        state_dict["inner_count"] = self.inner_count
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        self.image_accumulative_counts = state_dict.pop("image_accumulative_counts", self.image_accumulative_counts)
        self.video_accumulative_counts = state_dict.pop("video_accumulative_counts", self.video_accumulative_counts)
        # If the checkpoint was saved mid-window, drop the unfinished
        # accumulation window on resume because its partial grads cannot be
        # reconstructed reliably.
        current_modality = state_dict.pop("current_modality", None)
        if current_modality not in {None, "img", "vid"}:
            raise ValueError(f"Invalid modality state `{current_modality}` in optimizer checkpoint.")
        inner_count = state_dict.pop("inner_count", 0)
        if inner_count < 0:
            raise ValueError(f"Invalid inner_count `{inner_count}` in optimizer checkpoint.")
        self.resume_skip_iters = 0
        if inner_count > 0:
            if current_modality is None:
                raise ValueError("Incomplete optimizer state must include the current modality.")
            accumulative_counts = (
                self.image_accumulative_counts if current_modality == "img" else self.video_accumulative_counts
            )
            self.resume_skip_iters = max(0, accumulative_counts - inner_count)
            print_log(
                "Checkpoint was saved in the middle of a gradient accumulation "
                f"window ({current_modality}, {inner_count} micro-batches). "
                "Drop the partial window and skip the remaining micro-batches "
                f"of that window ({self.resume_skip_iters}) on resume.",
                logger="current",
                level=logging.WARNING,
            )
        self.current_modality = None
        self.inner_count = 0
        super().load_state_dict(state_dict)
