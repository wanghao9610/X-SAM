import torch
from mmengine.structures import BaseDataElement


def data_dict_to_device(input_dict, device="cuda", dtype=torch.bfloat16):
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = (
                v.to(device=device, dtype=dtype, non_blocking=True)
                if v.dtype == torch.float32 and v.dtype != dtype
                else v.to(device=device, non_blocking=True)
            )
        elif isinstance(v, list) and len(v) > 0:
            input_dict[k] = [
                (
                    (
                        ele.to(device=device, dtype=dtype, non_blocking=True)
                        if ele.dtype == torch.float32 and ele.dtype != dtype
                        else ele.to(device=device, non_blocking=True)
                    )
                    if isinstance(ele, torch.Tensor)
                    else ele
                )
                for ele in v
            ]
        elif isinstance(v, dict):
            input_dict[k] = data_dict_to_device(v, device=device, dtype=dtype)
    return input_dict


def data_sample_to_device(data_sample, device="cuda", dtype=torch.float32):
    for k, v in data_sample.items():
        if isinstance(v, BaseDataElement):
            data = {k: data_sample_to_device(v, device=device, dtype=dtype)}
            data_sample.set_data(data)
        elif isinstance(v, torch.Tensor):
            data = {
                k: (
                    v.to(device=device, dtype=dtype, non_blocking=True)
                    if v.dtype == torch.float32 and v.dtype != dtype
                    else v.to(device=device, non_blocking=True)
                )
            }
            data_sample.set_data(data)
        elif isinstance(v, list) and len(v) > 0:
            data = {
                k: [
                    (
                        (
                            ele.to(device=device, dtype=dtype, non_blocking=True)
                            if ele.dtype == torch.float32 and ele.dtype != dtype
                            else ele.to(device=device, non_blocking=True)
                        )
                        if isinstance(ele, torch.Tensor)
                        else ele
                    )
                    for ele in v
                ]
            }
            data_sample.set_data(data)
    return data_sample
