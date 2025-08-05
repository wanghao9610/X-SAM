# Copyright (c) OpenMMLab. All rights reserved.


from ..utils.constants import DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN
from .base_dataset import BaseDataset

SPECIAL_TOKENS = [DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN]


class ReasonSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="reaseg",
        data_root=None,
        explain_path=None,
        explain_ratio=0.5,
        query_type="all",
        **kwargs,
    ):
        super().__init__(
            *args,
            data_path=None,
            task_name=task_name,
            data_root=data_root,
            explain_path=explain_path,
            explain_ratio=explain_ratio,
            query_type=query_type,
            **kwargs,
        )
        raise NotImplementedError("ReasonSegDataset is not released yet. We will update it soon.")
