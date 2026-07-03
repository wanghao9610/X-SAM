/*!
refer to https://github.com/open-mmlab/mmcv/blob/a8073c74bf83d62ec36a103f835faa4837fb6585/mmcv/ops/csrc/pytorch/npu/ms_deform_attn_npu.hpp
*/

#ifndef MS_DEFORM_ATTN_NPU_HPP_
#define MS_DEFORM_ATTN_NPU_HPP_

#include <torch/extension.h>
#include <vector>

#include "pytorch_npu_util.hpp"

using at::Tensor;

/**
 * @brief Forward pass for Multi-Scale Deformable Attention on NPU
 *
 * @param value Input feature tensor
 * @param value_spatial_shapes Spatial shapes of multi-scale features
 * @param value_level_start_index Start index for each feature level
 * @param sampling_locations Sampling locations for deformable attention
 * @param attention_weights Attention weights
 * @param im2col_step Step size for im2col operation
 * @return Output tensor
 */
at::Tensor ms_deform_attn_forward_npu(
    const Tensor &value,
    const Tensor &value_spatial_shapes,
    const Tensor &value_level_start_index,
    const Tensor &sampling_locations,
    const Tensor &attention_weights,
    const int im2col_step);

/**
 * @brief Backward pass for Multi-Scale Deformable Attention on NPU
 *
 * @param value Input feature tensor
 * @param spatial_shapes Spatial shapes of multi-scale features
 * @param level_start_index Start index for each feature level
 * @param sampling_loc Sampling locations for deformable attention
 * @param attn_weight Attention weights
 * @param grad_output Gradient of output
 * @param im2col_step Step size for im2col operation
 * @return Vector of gradients: [grad_value, grad_sampling_loc, grad_attn_weight]
 */
std::vector<at::Tensor> ms_deform_attn_backward_npu(
    const Tensor &value,
    const Tensor &spatial_shapes,
    const Tensor &level_start_index,
    const Tensor &sampling_loc,
    const Tensor &attn_weight,
    const Tensor &grad_output,
    const int im2col_step);

#endif  // MS_DEFORM_ATTN_NPU_HPP_
