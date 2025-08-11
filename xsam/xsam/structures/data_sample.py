from typing import List

import torch
from mmengine.structures import BaseDataElement


class DataSample(BaseDataElement):
    """A data structure that contains the information of a sample.

    Args:
        mask_labels (Tensor): The ground truth masks of the sample.
        class_labels (Tensor): The ground truth labels of the sample.
    """

    @property
    def mask_labels(self) -> List[torch.Tensor]:
        return self._mask_labels

    @mask_labels.setter
    def mask_labels(self, value: List[torch.Tensor]):
        self.set_field(value, "_mask_labels", dtype=list)

    @property
    def class_labels(self) -> List[torch.Tensor]:
        return self._class_labels

    @class_labels.setter
    def class_labels(self, value: List[torch.Tensor]):
        self.set_field(value, "_class_labels", dtype=list)
