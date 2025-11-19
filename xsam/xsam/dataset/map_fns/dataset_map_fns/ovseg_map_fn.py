from .gen_seg_map_fn import genseg_map_fn


def ovseg_map_fn(*args, **kwargs):
    return genseg_map_fn(*args, **kwargs)
