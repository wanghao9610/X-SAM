# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from collections import defaultdict
from typing import Any, Dict, List

import torch.nn as nn
from mmengine.optim.optimizer.default_constructor import DefaultOptimWrapperConstructor
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS, OPTIMIZERS


def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Transform parameter groups into per-parameter structure.
    # Later items in `params` can overwrite parameters set in previous items.
    ret = defaultdict(dict)
    for item in params:
        assert "params" in item
        cur_params = {x: y for x, y in item.items() if x != "params"}
        for param in item["params"]:
            ret[param].update({"params": [param], **cur_params})
    return list(ret.values())


def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reorganize the parameter groups and merge duplicated groups.
    # The number of parameter groups needs to be as small as possible in order
    # to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
    # of using a parameter_group per single parameter, we reorganize the
    # parameter groups and merge duplicated groups. This approach speeds
    # up multi-tensor optimizer significantly.
    params = _expand_param_groups(params)
    groups = defaultdict(list)  # re-group all parameter groups by their hyperparams
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != "params")
        groups[cur_params].extend(item["params"])
    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur["params"] = param_values
        ret.append(cur)
    return ret


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class CustomOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __call__(self, model: nn.Module) -> OptimWrapper:
        if hasattr(model, "module"):
            model = model.module

        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        optim_wrapper_cfg.setdefault("type", "OptimWrapper")
        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_cls = self.optimizer_cfg["type"]
        # Optimizer like HybridAdam in colossalai requires the argument name
        # `model_params` rather than `params`. Here we get the first argument
        # name and fill it with the model parameters.
        if isinstance(optimizer_cls, str):
            with OPTIMIZERS.switch_scope_and_registry(None) as registry:
                optimizer_cls = registry.get(self.optimizer_cfg["type"])
        fisrt_arg_name = next(iter(inspect.signature(optimizer_cls).parameters))
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg[fisrt_arg_name] = model.parameters()
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        else:
            # set param-wise lr and weight decay recursively
            params: List = []
            self.add_params(params, model)
            # fix the bug that params groups are not efficient to use multi-tensor optimizer
            optimizer_cfg[fisrt_arg_name] = reduce_param_groups(params)
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper
