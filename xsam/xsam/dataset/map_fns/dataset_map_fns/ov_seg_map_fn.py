from .generic_seg_map_fn import generic_seg_map_fn


def ov_seg_map_fn(*args, **kwargs):
    return generic_seg_map_fn(*args, **kwargs)
