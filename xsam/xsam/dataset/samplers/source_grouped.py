import math
import random
from typing import Iterator, Optional, Sized

import numpy as np
import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Sampler

from xsam.utils.logging import print_log


def get_source_grouped_indices(lengths, group_batch_size, seed=None):
    """
    Group indices by their source and create batches from the same source.

    Args:
        lengths: List of lengths for each source
        group_batch_size: Size of each group batch
        seed: Random number generator for reproducibility

    Returns:
        List of indices grouped by source and shuffled
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    assert all(length != 0 for length in lengths), "Should not have zero length."

    # Create indices for each source
    start_inds = [0] + np.cumsum(lengths).tolist()[:-1]
    all_source_indices = []

    for source, length in enumerate(lengths):
        indices = list(range(start_inds[source], start_inds[source] + length))
        # Shuffle indices within each source
        random.shuffle(indices)
        all_source_indices.append(indices)

    # Create megabatches by cycling through sources
    megabatches = []
    source_pointers = [0] * len(lengths)  # Track current position in each source

    while True:
        # Find sources that have enough remaining samples
        available_sources = []
        for i, (pointer, length) in enumerate(zip(source_pointers, lengths)):
            if pointer + group_batch_size <= length:
                available_sources.append(i)

        if not available_sources:
            break

        # Randomly select a source
        source_idx = random.choice(available_sources)

        # Extract batch from selected source
        start_pos = source_pointers[source_idx]
        end_pos = start_pos + group_batch_size
        megabatch = all_source_indices[source_idx][start_pos:end_pos]
        source_pointers[source_idx] = end_pos

        if len(megabatch) == group_batch_size:
            megabatches.append(megabatch)

    for source_idx, source_pointer in enumerate(source_pointers):
        remaining_samples = all_source_indices[source_idx][source_pointer:]
        neighbor_samples = all_source_indices[source_idx][source_pointer - group_batch_size : source_pointer]
        if len(remaining_samples) > 0:
            megabatches.append(
                remaining_samples[:group_batch_size] + neighbor_samples[: group_batch_size - len(remaining_samples)]
            )

    # Shuffle the megabatches
    random.shuffle(megabatches)

    return [idx for batch in megabatches for idx in batch]


class SourceGroupedSampler(Sampler):
    def __init__(
        self,
        dataset: Sized,
        per_device_batch_size: int,
        length_property="source_length",
        mega_batch_mult: Optional[int] = None,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        print_log("SourceGroupedSampler is used.", logger="current")
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.step = 0  # Added step for checkpoint resuming
        self.round_up = round_up

        if self.round_up:
            num_iters = math.ceil(len(self.dataset) / world_size / per_device_batch_size)
            self.num_samples = num_iters * per_device_batch_size
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

        total_batch_size = per_device_batch_size * self.world_size
        if mega_batch_mult is None:
            # Default for mega_batch_mult: 16 or the number to get 4
            # megabatches, whichever is smaller.
            mega_batch_mult = min(len(self.dataset) // (total_batch_size * 4), 16)
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1
        self.group_batch_size = mega_batch_mult * total_batch_size

        if isinstance(self.dataset, TorchConcatDataset):
            length = []
            for sub_dataset in self.dataset.datasets:
                length.append(getattr(sub_dataset, length_property))
            self.length = length
        else:
            self.length = [getattr(self.dataset, length_property)]
        assert isinstance(self.length, (list, tuple))

        self.total_batch_size = total_batch_size
        print_log(
            f"SourceGroupedSampler construction is complete, " f"and the selected attribute is {length_property}.",
            logger="current",
        )

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        seed = self.seed + self.epoch

        indices = get_source_grouped_indices(
            lengths=self.length,
            group_batch_size=self.group_batch_size,
            seed=seed,
        )

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[: self.total_size]
        # subsample
        assert len(indices) == self.total_size, f"Indices length {len(indices)} != total_size {self.total_size}"
        indices = indices[self.rank : self.total_size : self.world_size]

        assert (
            len(indices) == self.num_samples
        ), f"Final indices length {len(indices)} != num_samples {self.num_samples}"

        # Support for checkpoint resuming by skipping already processed samples
        return iter(indices[self.step :])

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples - self.step

    def set_epoch(self, epoch: int, step: int = 0) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            step (int): Step number for checkpoint resuming.
        """
        self.epoch = epoch
        self.step = step
