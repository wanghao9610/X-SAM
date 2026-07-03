import numpy as np
from panopticapi.utils import rgb2id


class IdGenerator:
    """
    The class is designed to generate unique IDs that have meaningful RGB encoding.
    Given semantic category unique ID will be generated and its RGB encoding will
    have color close to the predefined semantic category color.
    The RGB encoding used is ID = R * 256 * G + 256 * 256 + B.
    Class constructor takes dictionary {id: category_info}, where all semantic
    class ids are presented and category_info record is a dict with fields
    'isthing' and 'color'
    """

    def __init__(self, categories, seed: int = 42):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        for category in self.categories.values():
            if category["isthing"] == 0:
                self.taken_colors.add(tuple(category["color"]))

    def reset_colors(self, seed: int | None = None):
        self.taken_colors = set([0, 0, 0])
        self._seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(self._seed)

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + self._rng.integers(low=-max_dist, high=max_dist + 1, size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category["isthing"] == 0:
            return category["color"]
        base_color_array = category["color"]
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                    self.taken_colors.add(color)
                    return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color
