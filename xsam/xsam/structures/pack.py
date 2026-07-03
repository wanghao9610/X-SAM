from itertools import chain

import numpy as np


class Packer:
    """Pack multiple pieces of data into one."""

    def __init__(self, chunk_size=2048, use_varlen_attn=False, drop_last=False):
        self.chunk_size = chunk_size
        self.residual = {"input_ids": [], "labels": []}
        self.use_varlen_attn = use_varlen_attn
        self.drop_last = drop_last
        if use_varlen_attn:
            self.residual_cumulative_len = [0]

    def get_cumulative_len(self, chunk_num):
        ptr_l = 0
        cumulative_len = []
        for chunk_idx in range(chunk_num):
            length_train = (chunk_idx + 1) * self.chunk_size
            ptr_r = np.searchsorted(self.residual_cumulative_len, length_train, side="left")
            if self.residual_cumulative_len[ptr_r] == length_train:
                cumulative_len_cur = self.residual_cumulative_len[ptr_l : ptr_r + 1]
                ptr_l = ptr_r + 1
            else:
                cumulative_len_cur = self.residual_cumulative_len[ptr_l:ptr_r] + [length_train]
                ptr_l = ptr_r
            cumulative_len_cur = [num - chunk_idx * self.chunk_size for num in cumulative_len_cur]
            if cumulative_len_cur[0] != 0:
                cumulative_len_cur = [0] + cumulative_len_cur

            cumulative_len.append(cumulative_len_cur)

        self.residual_cumulative_len = [num - length_train for num in self.residual_cumulative_len[ptr_l:]]
        if len(self.residual_cumulative_len) == 0:
            self.residual_cumulative_len = [0]
        elif self.residual_cumulative_len[0] != 0:
            self.residual_cumulative_len = [0] + self.residual_cumulative_len

        return cumulative_len

    def get_position_ids(self, cumulative_len):
        position_ids = []
        for cumulative_len_cur in cumulative_len:
            index_cur = []
            for i in range(len(cumulative_len_cur) - 1):
                index_cur.extend(list(range(cumulative_len_cur[i + 1] - cumulative_len_cur[i])))  # noqa: W504
            position_ids.append(index_cur)
        return position_ids

    def __call__(self, batch):
        concatenated_samples = {k: v + list(chain(*batch[k])) for k, v in self.residual.items()}

        if self.use_varlen_attn:
            for input_id in batch["input_ids"]:
                self.residual_cumulative_len.append(self.residual_cumulative_len[-1] + len(input_id))

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)  # noqa: W504
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {k: v[(chunk_num * self.chunk_size) :] for k, v in concatenated_samples.items()}

            if self.use_varlen_attn:
                cumulative_len = self.get_cumulative_len(chunk_num)
                result["cumulative_len"] = cumulative_len
                result["position_ids"] = self.get_position_ids(cumulative_len)
        else:
            if self.drop_last:
                result = {k: [] for k, v in concatenated_samples.items()}
            else:
                result = {k: [v] for k, v in concatenated_samples.items()}

            self.residual = {k: [] for k in concatenated_samples.keys()}

            if self.use_varlen_attn:
                result["cumulative_len"] = [] if self.drop_last else [self.residual_cumulative_len]
                result["position_ids"] = (
                    [] if self.drop_last else self.get_position_ids([self.residual_cumulative_len])
                )
                self.residual_cumulative_len = [0]

        return result
