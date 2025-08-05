import json
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def load_image(image_file, threshold=128, mode="RGB", to_numpy=False, to_tensor=False):
    assert mode in ["RGB", "L"]
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert(mode)
    else:
        image = Image.open(image_file).convert(mode)

    if mode == "L" and threshold is not None:
        image = image.point(lambda x: 1 if x > threshold else 0, mode="L")
    if to_numpy:
        image = np.array(image)
    if to_tensor:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = torch.from_numpy(np.ascontiguousarray(image.copy()))
    return image
