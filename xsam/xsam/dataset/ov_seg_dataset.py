# Copyright (c) OpenMMLab. All rights reserved.


from .generic_seg_dataset import GenericSegDataset


class OVSegDataset(GenericSegDataset):
    def __init__(self, *args, label_file=None, label_shift=0, **kwargs):
        super().__init__(*args, label_file=label_file, label_shift=label_shift, **kwargs)

        raise NotImplementedError("OVSegDataset is not released yet. We will update it soon.")
