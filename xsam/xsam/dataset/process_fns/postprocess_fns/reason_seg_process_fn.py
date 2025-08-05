from .refer_seg_process_fn import refer_seg_postprocess_fn


def reason_seg_postprocess_fn(*args, **kwargs):
    return refer_seg_postprocess_fn(*args, **kwargs)
