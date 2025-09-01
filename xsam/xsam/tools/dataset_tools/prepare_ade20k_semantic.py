# Copied and adapted from https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_ade20k_sem_seg.py
import os
import os.path as osp

import numpy as np
import tqdm
from PIL import Image


def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    dataset_dir = osp.join(os.getenv("root_dir", "."), "datas")
    for name in ["training", "validation"]:
        annotation_dir = osp.join(dataset_dir, "ade20k/annotations", name)
        output_dir = osp.join(dataset_dir, "ade20k/annotations_detectron2", name)
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = osp.join(output_dir, file.name)
            convert(file, output_file)
