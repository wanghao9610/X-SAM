from typing import Iterator, List

from torch.utils.data.sampler import BatchSampler


class CustomBatchSampler(BatchSampler):
    # Implement for variant batch size support
    def __iter__(self) -> Iterator[List[int]]:
        # `SourceGroupedSampler` already yields batches (List[int]).
        for batch in self.sampler:
            yield batch

    def __len__(self) -> int:
        return len(self.sampler)
