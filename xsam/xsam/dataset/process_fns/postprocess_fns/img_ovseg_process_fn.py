from .img_genseg_process_fn import img_genseg_postprocess_fn


def img_ovseg_postprocess_fn(*args, **kwargs):
    return img_genseg_postprocess_fn(*args, **kwargs)
