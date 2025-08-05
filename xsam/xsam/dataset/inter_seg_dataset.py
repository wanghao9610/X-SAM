from .vgd_seg_dataset import VGDSegDataset


class InterSegDataset(VGDSegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    raise NotImplementedError("InterSegDataset is not released yet. We will update it soon.")
