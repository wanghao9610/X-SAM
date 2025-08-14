from .generic_seg_process_fn import generic_seg_postprocess_fn


def ov_seg_postprocess_fn(*args, **kwargs):
    return generic_seg_postprocess_fn(*args, **kwargs)
