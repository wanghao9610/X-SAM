import logging
import time
from typing import Dict, Optional, Sequence, Union

from mmengine.runner import IterBasedTrainLoop
from mmengine.runner.loops import IterBasedTrainLoop, _InfiniteDataloaderIterator
from torch.utils.data import DataLoader

from xsam.utils.logging import print_log


# Refer to https://github.com/open-mmlab/mmengine/pull/1548
class XSamInfiniteDataloaderIterator(_InfiniteDataloaderIterator):
    def skip_iter(self, iter: int) -> None:
        for _ in range(iter):
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


class XSamIterBasedTrainLoop(IterBasedTrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader_iterator = XSamInfiniteDataloaderIterator(self.dataloader)

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook("before_train")
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook("before_train_epoch")
        if self._iter > 0:
            print_log(
                f"Advance dataloader {self._iter} steps to skip data " "that has already been trained",
                logger="current",
                level=logging.WARNING,
            )
            self.dataloader_iterator.skip_iter(self._iter)

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


class TrainLoop(XSamIterBasedTrainLoop):
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
