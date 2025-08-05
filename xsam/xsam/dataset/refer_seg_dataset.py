# Copyright (c) OpenMMLab. All rights reserved.


from ..utils.constants import DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN
from .base_dataset import BaseDataset

SPECIAL_TOKENS = [DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN]


class ReferSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="refseg",
        dataset=None,
        data_root=None,
        data_split=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_path=None,
            dataset=dataset,
            task_name=task_name,
            data_root=data_root,
            data_split=data_split,
            **kwargs,
        )
        raise NotImplementedError("ReferSegDataset is not released yet. We will update it soon.")
