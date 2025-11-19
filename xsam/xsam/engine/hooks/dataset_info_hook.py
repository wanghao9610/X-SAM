from mmengine.hooks import Hook
from tabulate import tabulate
from torch.utils.data import ConcatDataset as TorchConcatDataset
from xtuner.registry import BUILDER

from ...utils.constants import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PSTART_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
    INDEX2TOKEN,
)
from ..utils.util import split_list


class DatasetInfoHook(Hook):
    def __init__(self, tokenizer=None, special_tokens=None, is_intern_repo_dataset=False):
        self.tokenizer = BUILDER.build(tokenizer) if tokenizer is not None else None
        if special_tokens is not None:
            self._add_special_tokens(special_tokens)
        self.is_intern_repo_dataset = is_intern_repo_dataset

    def _add_special_tokens(self, special_tokens):
        assert all(token in DEFAULT_SPECIAL_TOKENS for token in special_tokens)
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.seg_token_idx = -1
        self.cls_token_idx = -1
        self.pstart_token_idx = -1
        self.pend_token_idx = -1

        if DEFAULT_SEG_TOKEN in special_tokens:
            self.seg_token_idx = self.tokenizer(DEFAULT_SEG_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_CLS_TOKEN in special_tokens:
            self.cls_token_idx = self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_PSTART_TOKEN in special_tokens:
            self.pstart_token_idx = self.tokenizer(DEFAULT_PSTART_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_PEND_TOKEN in special_tokens:
            self.pend_token_idx = self.tokenizer(DEFAULT_PEND_TOKEN, add_special_tokens=False)["input_ids"][0]

    def log(self, runner, dataset, mode="train"):
        def _log(input_ids, log_prefix=""):
            if self.is_intern_repo_dataset:
                input_ids = [abs(x) for x in input_ids]
            # Try to split list to be compatible with IMAGE token
            input_ids = split_list(input_ids, INDEX2TOKEN.keys())
            text = log_prefix
            for ids in input_ids:
                if len(ids) == 1 and ids[0] in INDEX2TOKEN:
                    text += INDEX2TOKEN[ids[0]]
                else:
                    text += self.tokenizer.decode(ids)
            runner.logger.info(text)

        if isinstance(dataset, TorchConcatDataset):
            dataset = dataset.datasets
        else:
            dataset = [dataset]

        headers = ["#", "Dataset", "Task", "# Repeats", "# Data", "# Samples"]
        data_table = tabulate(
            [
                *[
                    [i, ds.data_name, ds.task_name, f"{ds.repeats:.2f}", f"{ds.data_length:,}", f"{len(ds):,}"]
                    for i, ds in enumerate(dataset)
                ],
                ["=" * int(len(header) * scale) for header, scale in zip(headers, [5.0, 2.5, 2.0, 1.4, 1.4, 1.4])],
                [
                    "Total",
                    len(dataset),
                    len(set(ds.task_name for ds in dataset)),
                    f"{sum(ds.repeats for ds in dataset):.2f}",
                    f"{sum(ds.data_length for ds in dataset):,}",
                    f"{sum(len(ds) for ds in dataset):,}",
                ],
            ],
            headers=headers,
            tablefmt="outline",
            colalign=("center", "center", "center", "center", "right", "right"),
        )
        runner.logger.info(f"Dataset summary:\n{data_table}")
        for ds in dataset:
            if self.tokenizer is None:
                continue

            runner.logger.info(f"{mode} of {ds.data_name} example:")
            if "chosen_ids" in ds[0]:
                _log(ds[0]["chosen_ids"], log_prefix="chosen: ")
                _log(ds[0]["rejected_ids"], log_prefix="rejected: ")
            else:
                _log(ds[0]["input_ids"])

    def before_train(self, runner) -> None:
        do_train = runner.train_loop is not None
        do_eval = runner.val_loop is not None
        if do_train:
            train_dataset = runner.train_dataloader.dataset
            self.log(runner, train_dataset, mode="train")
        if do_eval:
            eval_dataset = runner.val_dataloader.dataset
            self.log(runner, eval_dataset, mode="eval")

    def before_val(self, runner) -> None:
        eval_dataset = runner.val_dataloader.dataset
        self.log(runner, eval_dataset, mode="eval")

    def before_test(self, runner) -> None:
        test_dataset = runner.test_dataloader.dataset
        self.log(runner, test_dataset, mode="test")
