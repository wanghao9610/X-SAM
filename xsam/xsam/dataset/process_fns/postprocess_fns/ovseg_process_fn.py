from .genseg_process_fn import genseg_postprocess_fn


def ovseg_postprocess_fn(*args, **kwargs):
    return genseg_postprocess_fn(*args, **kwargs)
