import torch
from mmengine.utils.misc import get_object_from_string


def set_obj_dtype(d):
    for key, value in d.items():
        if value in ["torch.float16", "torch.float32", "torch.bfloat16"]:
            d[key] = getattr(torch, value.split(".")[-1])


def try_build_module(cfg):
    builder = cfg["type"]
    if isinstance(builder, str):
        builder = get_object_from_string(builder)
    if builder is None:
        # support handling cfg with key 'type' can not be built, such as
        # {'rope_scaling': {'type': 'linear', 'factor': 2.0}}
        return cfg
    cfg.pop("type")
    module_built = builder(**cfg)
    return module_built


def traverse_dict(d):
    if isinstance(d, dict):
        set_obj_dtype(d)
        for key, value in d.items():
            if isinstance(value, dict):
                traverse_dict(value)
                if "type" in value:
                    module_built = try_build_module(value)
                    d[key] = module_built
    elif isinstance(d, list):
        for element in d:
            traverse_dict(element)


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
