from enum import Enum
from typing import List, Tuple, Union

import numpy as np
from mmengine.utils import is_str

try:
    import seaborn as sns
except ImportError:
    sns = None


class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """

    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color: Union[Color, str, tuple, int, np.ndarray]) -> tuple:
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value  # type: ignore
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f"Invalid type for color: {type(color)}")


def _get_adaptive_scales(areas: np.ndarray, min_area: int = 800, max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) // (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def get_palette(palette: Union[List[tuple], str, tuple], num_classes: int) -> List[Tuple[int]]:
    """Get palette from various inputs.

    Args:
        palette (list[tuple] | str | tuple): palette inputs.
        num_classes (int): the number of classes.

    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)

    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == "random" or palette is None:
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    else:
        raise TypeError(f"Invalid type for palette: {type(palette)}")

    assert len(dataset_palette) >= num_classes, "The length of palette should not be less than `num_classes`."
    return dataset_palette


def jitter_color(color: tuple) -> tuple:
    """Randomly jitter the given color in order to better distinguish instances
    with the same class.

    Args:
        color (tuple): The RGB color tuple. Each value is between [0, 255].

    Returns:
        tuple: The jittered color tuple.
    """
    jitter = np.random.rand(3)
    jitter = (jitter / np.linalg.norm(jitter) - 0.5) * 0.5 * 255
    color = np.clip(jitter + color, 0, 255).astype(np.uint8)
    return tuple(color)


def random_color(seed):
    """Random a color according to the input seed."""
    if sns is None:
        raise RuntimeError(
            "motmetrics is not installed,\
                 please install it by: pip install seaborn"
        )
    np.random.seed(seed)
    colors = sns.color_palette()
    color = colors[np.random.choice(range(len(colors)))]
    color = tuple([int(255 * c) for c in color])
    return color
