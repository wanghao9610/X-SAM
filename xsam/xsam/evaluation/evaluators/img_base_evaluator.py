import os
from typing import Optional

import torch

from xsam.dataset.utils.catalog import MetadataCatalog

from .base_evaluator import BaseEvaluator


class ImgBaseEvaluator(BaseEvaluator):
    def __init__(
        self,
        data_name: str = None,
        evaluation_metrics: list[str] = None,
        output_dir: Optional[str] = None,
        distributed: bool = True,
        support_loading: bool = True,
    ):
        """
        Args:
            metadata: metadata of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._data_name = data_name
        self._distributed = distributed
        self._metadata = MetadataCatalog.get(data_name)
        self._evaluation_metrics = evaluation_metrics
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._support_loading = support_loading

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
        self._dataset_name = self.data_name

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @property
    def data_name(self):
        return self._data_name

    @property
    def support_loading(self):
        return self._support_loading

    @support_loading.setter
    def support_loading(self, value):
        self._support_loading = value

    def reset(self):
        self._predictions = []
