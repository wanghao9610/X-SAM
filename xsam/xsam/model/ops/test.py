# refer to https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/test.py
from __future__ import absolute_import, division, print_function

import torch

# from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch
from torch.autograd import gradcheck

from xsam.model.ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "npu")

N, M, D = 1, 4, 32
Lq, L, P = 32, 2, 4
shapes = torch.as_tensor([(8, 8), (4, 4)], dtype=torch.long).to(device)
level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])


torch.manual_seed(3)


@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    value = torch.rand(N, S, M, D).to(device) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).to(device)
    attention_weights = torch.rand(N, Lq, M, L, P).to(device) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = (
        ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double())
        .detach()
        .cpu()
    )
    print(
        value.double().dtype,
        shapes.dtype,
        level_start_index.dtype,
        sampling_locations.double().dtype,
        attention_weights.double().dtype,
    )
    output_cuda_or_npu = (
        MSDeformAttnFunction.apply(
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            attention_weights.double(),
            im2col_step,
        )
        .detach()
        .cpu()
    )
    fwdok = torch.allclose(output_cuda_or_npu, output_pytorch)
    max_abs_err = (output_cuda_or_npu - output_pytorch).abs().max()
    max_rel_err = ((output_cuda_or_npu - output_pytorch).abs() / output_pytorch.abs()).max()

    print(
        f"* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    )


@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    value = torch.rand(N, S, M, D).to(device) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).to(device)
    attention_weights = torch.rand(N, Lq, M, L, P).to(device) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights).detach().cpu()
    output_cuda_or_npu = (
        MSDeformAttnFunction.apply(
            value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step
        )
        .detach()
        .cpu()
    )
    fwdok = torch.allclose(output_cuda_or_npu, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda_or_npu - output_pytorch).abs().max()
    max_rel_err = ((output_cuda_or_npu - output_pytorch).abs() / output_pytorch.abs()).max()

    print(
        f"* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    )


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    value = torch.rand(N, S, M, channels).to(device) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).to(device)
    attention_weights = torch.rand(N, Lq, M, L, P).to(device) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    # NPU doesn't support double precision, skip strict gradcheck and verify backward runs
    if device.type == "npu":
        # Just verify backward pass runs without error
        output = func(
            value.float(),
            shapes,
            level_start_index,
            sampling_locations.float(),
            attention_weights.float(),
            im2col_step,
        )
        loss = output.sum()
        loss.backward()
        gradok = (
            (value.grad is not None) and (sampling_locations.grad is not None) and (attention_weights.grad is not None)
        )
        print(f"* {gradok} check_gradient_numerical(D={channels}) [NPU backward pass verified]")
    else:
        gradok = gradcheck(
            func,
            (
                value.double(),
                shapes,
                level_start_index,
                sampling_locations.double(),
                attention_weights.double(),
                im2col_step,
            ),
        )
        print(f"* {gradok} check_gradient_numerical(D={channels})")


if __name__ == "__main__":
    check_forward_equal_with_pytorch_double()
    check_forward_equal_with_pytorch_float()

    # NPU requires embed_dims (channels) to be in range [32, 256]
    for channels in [32, 64, 128, 256]:
        check_gradient_numerical(channels, True, True, True)
