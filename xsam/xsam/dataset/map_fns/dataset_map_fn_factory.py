from functools import partial


def dataset_map_fn_factory(fn, **kwargs):
    return partial(fn, **kwargs)
