from functools import partial


def process_map_fn_factory(fn, **kwargs):
    return partial(fn, **kwargs)
