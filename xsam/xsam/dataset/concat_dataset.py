import math

from torch.utils.data import ConcatDataset as TorchConcatDataset

from xsam.registry import BUILDER


class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets, oversample_ratio=0.0, oversample_mult=1.0, max_sample=None):
        self.oversample_ratio = oversample_ratio
        self.oversample_mult = oversample_mult
        self.max_sample = max_sample
        datasets_instance = []
        for cfg in datasets:
            dataset = BUILDER.build(cfg)
            self._truncate_dataset(dataset)
            datasets_instance.append(dataset)

        dataset_repeats = self.get_dataset_repeats(datasets_instance)
        for dataset, repeat in zip(datasets_instance, dataset_repeats):
            dataset.repeats = repeat * self.oversample_mult

        super().__init__(datasets=datasets_instance)

    def _truncate_dataset(self, dataset):
        if self.max_sample is None:
            return

        max_sample = int(self.max_sample)
        if max_sample <= 0:
            return

        if hasattr(dataset, "data"):
            dataset.data = dataset.data[:max_sample]
        if hasattr(dataset, "data_length"):
            dataset.data_length = min(dataset.data_length, max_sample)

    def get_dataset_repeats(self, datasets):
        dataset_lens = [dataset.data_length for dataset in datasets]
        total_len = sum(dataset_lens)
        if total_len == 0:
            return [1.0 for _ in datasets]
        dataset_freqs = [l / total_len for l in dataset_lens]
        # r_d = max(1, sqrt(t/f_d))
        dataset_repeats = [
            max(1.0, math.sqrt(self.oversample_ratio / freq)) if freq > 0 else 1.0 for freq in dataset_freqs
        ]
        return dataset_repeats

    def __repr__(self):
        main_str = "Dataset as a concatenation of multiple datasets. \n"
        main_str += f"Oversample ratio: {self.oversample_ratio}\n"
        main_str += f"Oversample multiplier: {self.oversample_mult}\n"
        main_str += ",\n".join([f"{repr(dataset)}" for dataset in self.datasets])
        return main_str
