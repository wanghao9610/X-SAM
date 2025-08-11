import math

from torch.utils.data import ConcatDataset as TorchConcatDataset
from xtuner.registry import BUILDER


class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets, oversample_ratio=0.0):
        self.oversample_ratio = oversample_ratio
        datasets_instance = []
        for cfg in datasets:
            datasets_instance.append(BUILDER.build(cfg))

        if self.oversample_ratio > 0.0:
            dataset_repeats = self.get_dataset_repeats(datasets_instance)
            for dataset, repeat in zip(datasets_instance, dataset_repeats):
                dataset.repeats = repeat

        super().__init__(datasets=datasets_instance)

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
        main_str += ",\n".join([f"{repr(dataset)}" for dataset in self.datasets])
        return main_str
