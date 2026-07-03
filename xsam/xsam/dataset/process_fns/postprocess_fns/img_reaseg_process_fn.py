from .img_refseg_process_fn import img_refseg_postprocess_fn


def img_reaseg_postprocess_fn(*args, **kwargs):
    return img_refseg_postprocess_fn(*args, **kwargs)
