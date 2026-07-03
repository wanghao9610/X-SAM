import math
import random
from typing import Dict, Iterator, List, Optional, Sized

import numpy as np
import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Sampler

from xsam.utils.logging import print_log


def get_source_grouped_indices(
    lengths,
    sub_lengths,
    group_batch_sizes,
    batch_mults,
    round_up=True,
    seed=None,
):
    """
    Group indices by their source and create batches from the same source.

    Args:
        lengths: List of lengths for each source
        sub_lengths: List of sub lengths for each sample in the source
        group_batch_sizes: Base size of each source group batch
        batch_mults: List of mults for the batch size for each source
        round_up: Whether to round up the number of samples to the nearest multiple of the group batch size
        seed: Random number generator for reproducibility

    Returns:
        List of `(source_idx, indices)` grouped by source and shuffled
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    assert all(length != 0 for length in lengths), "Should not have zero length."
    assert len(lengths) == len(batch_mults), "lengths and batch_mults must align."
    assert len(lengths) == len(group_batch_sizes), "lengths and group_batch_sizes must align."

    # Create indices for each source
    start_inds = [0] + np.cumsum(lengths).tolist()[:-1]
    all_source_indices = []

    for source, length in enumerate(lengths):
        indices = list(range(start_inds[source], start_inds[source] + length))
        random.shuffle(indices)
        # Sort indices within each source by their sub_lengths
        source_sub_lengths = sub_lengths[source]
        if isinstance(source_sub_lengths, (list, tuple)):
            assert len(source_sub_lengths) == length, f"Sub length {len(source_sub_lengths)} != length {length}"
            # Create (index, length) pairs and sort by length
            indexed_lengths = [(idx, source_sub_lengths[idx - start_inds[source]]) for idx in indices]
            indexed_lengths.sort(key=lambda x: x[1])
            # Extract sorted indices
            indices = [idx for idx, _ in indexed_lengths]
        all_source_indices.append(indices)

    # Create mega_batches by cycling through sources
    mega_batches = []
    source_pointers = [0] * len(lengths)  # Track current position in each source

    while True:
        # Find sources that have enough remaining samples
        available_sources = []
        for i, (pointer, length, group_batch_size, batch_mult) in enumerate(
            zip(source_pointers, lengths, group_batch_sizes, batch_mults)
        ):
            if pointer + group_batch_size * batch_mult <= length:
                available_sources.append(i)

        if not available_sources:
            break

        # Randomly select a source
        source_idx = random.choice(available_sources)

        # Extract batch from selected source
        start_pos = source_pointers[source_idx]
        group_batch_size = group_batch_sizes[source_idx]
        batch_mult = batch_mults[source_idx]
        end_pos = start_pos + group_batch_size * batch_mult
        mega_batch = all_source_indices[source_idx][start_pos:end_pos]
        source_pointers[source_idx] = end_pos

        if len(mega_batch) == group_batch_size * batch_mult:
            mega_batches.append((source_idx, mega_batch))

    for source_idx, source_pointer in enumerate(source_pointers):
        remaining_samples = all_source_indices[source_idx][source_pointer:]
        group_batch_size = group_batch_sizes[source_idx]
        batch_mult = batch_mults[source_idx]
        target_size = group_batch_size * batch_mult
        neighbor_samples = all_source_indices[source_idx][max(0, source_pointer - target_size) : source_pointer]
        if len(remaining_samples) > 0 and round_up:
            batch = remaining_samples[:target_size]
            if len(batch) < target_size:
                pad_needed = target_size - len(batch)
                # Use neighbor samples if available, otherwise cycle remaining samples
                padding_pool = neighbor_samples if len(neighbor_samples) > 0 else remaining_samples
                padding_pool = padding_pool * math.ceil(pad_needed / max(1, len(padding_pool)))
                batch = batch + padding_pool[:pad_needed]
            mega_batches.append((source_idx, batch))

    # Shuffle the mega_batches
    random.shuffle(mega_batches)

    return mega_batches


class SourceGroupedSampler(Sampler):
    @staticmethod
    def _get_source_modality(task_name: Optional[str]) -> Optional[str]:
        if not task_name:
            return None
        if task_name.startswith("img_"):
            return "img"
        if task_name.startswith("vid_"):
            return "vid"
        return None

    def __init__(
        self,
        dataset: Sized,
        per_device_batch_size: int,
        accumulative_counts: int = 1,
        modality_accumulative_counts: Optional[Dict[str, int]] = None,
        length_property="source_length",
        sub_length_property=None,
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
        self._num_batches = None  # Cache for number of batches
        self.round_up = round_up
        self.per_device_batch_size = per_device_batch_size
        self.accumulative_counts = accumulative_counts
        self.modality_accumulative_counts = modality_accumulative_counts or {}

        if isinstance(self.dataset, TorchConcatDataset):
            lengths = []
            sub_lengths = []
            batch_mults = []
            source_accumulative_counts = []
            for sub_dataset in self.dataset.datasets:
                lengths.append(getattr(sub_dataset, length_property))
                sub_lengths.append(
                    getattr(sub_dataset, sub_length_property, None) if sub_length_property is not None else None
                )
                batch_mults.append(getattr(sub_dataset, "batch_mult", 1))
                source_modality = self._get_source_modality(getattr(sub_dataset, "task_name", None))
                source_accumulative_counts.append(
                    self.modality_accumulative_counts.get(source_modality, accumulative_counts)
                )
            self.lengths = lengths
            self.sub_lengths = sub_lengths
            self.batch_mults = batch_mults
            self.source_accumulative_counts = source_accumulative_counts
        else:
            self.lengths = [getattr(self.dataset, length_property)]
            self.sub_lengths = [
                getattr(self.dataset, sub_length_property, None) if sub_length_property is not None else None
            ]
            self.batch_mults = [getattr(self.dataset, "batch_mult", 1)]
            source_modality = self._get_source_modality(getattr(self.dataset, "task_name", None))
            self.source_accumulative_counts = [
                self.modality_accumulative_counts.get(source_modality, accumulative_counts)
            ]
        assert isinstance(self.lengths, (list, tuple))
        assert isinstance(self.sub_lengths, (list, tuple))
        assert isinstance(self.batch_mults, (list, tuple))
        assert isinstance(self.source_accumulative_counts, (list, tuple))

        if mega_batch_mult is None:
            max_total_batch_size = per_device_batch_size * max(self.source_accumulative_counts) * self.world_size
            # Default for mega_batch_mult: 16 or the number to get 4
            # mega_batches, whichever is smaller.
            mega_batch_mult = min(len(self.dataset) // (max_total_batch_size * 4), 16)
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1

        self.mega_batch_mult = mega_batch_mult
        self.group_batch_sizes = [
            mega_batch_mult * per_device_batch_size * source_accumulative_count * self.world_size
            for source_accumulative_count in self.source_accumulative_counts
        ]

        if self.round_up:
            num_group_iters = [
                math.ceil(length / (group_batch_size * batch_mult))
                for length, group_batch_size, batch_mult in zip(self.lengths, self.group_batch_sizes, self.batch_mults)
            ]
            self.total_size = sum(
                [
                    num_group_iter * group_batch_size * batch_mult
                    for num_group_iter, group_batch_size, batch_mult in zip(
                        num_group_iters, self.group_batch_sizes, self.batch_mults
                    )
                ]
            )
            self.num_samples = self.total_size // self.world_size
        else:
            num_group_iters = [
                length // (group_batch_size * batch_mult)
                for length, group_batch_size, batch_mult in zip(self.lengths, self.group_batch_sizes, self.batch_mults)
            ]
            self.total_size = sum(
                [
                    num_group_iter * group_batch_size * batch_mult
                    for num_group_iter, group_batch_size, batch_mult in zip(
                        num_group_iters, self.group_batch_sizes, self.batch_mults
                    )
                ]
            )
            self.num_samples = self.total_size // self.world_size

        self.total_batch_size = per_device_batch_size * self.world_size
        print_log(
            f"SourceGroupedSampler construction is complete, " f"and the selected attribute is {length_property}.",
            logger="current",
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate the indices."""
        seed = self.seed + self.epoch

        grouped_indices = get_source_grouped_indices(
            lengths=self.lengths,
            sub_lengths=self.sub_lengths,
            group_batch_sizes=self.group_batch_sizes,
            batch_mults=self.batch_mults,
            round_up=self.round_up,
            seed=seed,
        )
        assert (
            sum(len(indices) for _, indices in grouped_indices) == self.total_size
        ), f"Grouped indices length {sum(len(indices) for _, indices in grouped_indices)} != total size {self.total_size}"

        # subsample
        batch_indices = []
        for source_idx, mega_batch_indices in grouped_indices:
            # per_device_batch_size * source_accumulative_counts * world_size * mega_batch_mult * batch_mult
            source_accumulative_count = self.source_accumulative_counts[source_idx]
            mult_batch_size = len(mega_batch_indices) // (self.mega_batch_mult * source_accumulative_count)
            batch_splits = [
                mega_batch_indices[i * mult_batch_size : (i + 1) * mult_batch_size]
                for i in range(len(mega_batch_indices) // mult_batch_size)
            ]
            for batch_split in batch_splits:
                batch_split = batch_split[self.rank :: self.world_size]
                batch_indices.append(batch_split)

        assert (
            sum(len(indices) for indices in batch_indices) == self.num_samples
        ), f"Batch indices length {sum(len(indices) for indices in batch_indices)} != num samples {self.num_samples}"
        self._num_batches = len(batch_indices)

        # Support for checkpoint resuming by skipping already processed samples
        return iter(batch_indices[self.step :])

    def __len__(self) -> int:
        """The number of batches in this rank."""
        if self._num_batches is None:
            # If __iter__ hasn't been called yet, we need to compute it
            # This is inefficient but ensures correctness
            list(self.__iter__())
        return self._num_batches - self.step

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
        # Reset cached batch count when epoch or step changes
        self._num_batches = None
