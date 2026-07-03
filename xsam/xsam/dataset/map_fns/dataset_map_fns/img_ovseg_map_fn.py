from .img_genseg_map_fn import img_genseg_map_fn


def img_ovseg_map_fn(*args, **kwargs):
    return img_genseg_map_fn(*args, **kwargs)
