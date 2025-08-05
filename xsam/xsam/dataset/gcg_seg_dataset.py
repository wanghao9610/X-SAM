# Copyright (c) OpenMMLab. All rights reserved.
from ..utils.constants import DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN
from .base_dataset import BaseDataset

SPECIAL_TOKENS = [DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN]


class GCGSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="gcgseg",
        cap_data_path=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            task_name=task_name,
            cap_data_path=cap_data_path,
            **kwargs,
        )
        raise NotImplementedError("GCGSegDataset is not released yet. We will update it soon.")
