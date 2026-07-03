/*!
refer to https://github.com/open-mmlab/mmcv/blob/a8073c74bf83d62ec36a103f835faa4837fb6585/mmcv/ops/csrc/pytorch/npu/ms_deform_attn_npu.cpp
*/
#include "ms_deform_attn_npu.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void check_support(const Tensor &value, const Tensor &attention_weights) {
  TORCH_CHECK(
      (value.scalar_type() == at::kFloat || value.scalar_type() == at::kHalf),
      "Dtype of value should be float32 or float16.");
  int64_t num_heads = value.size(2);
  int64_t embed_dims = value.size(3);
  int64_t num_points = attention_weights.size(4);
  TORCH_CHECK((num_heads >= 4 && num_heads <= 8),
              "num_heads should be in the range of [4, 8]");
  TORCH_CHECK((embed_dims >= 32 && embed_dims <= 256),
              "embed_dims should be in the range of [32, 256]");
  TORCH_CHECK((num_points >= 4 && num_points <= 8),
              "num_points should be in the range of [4, 8]");
}

Tensor ms_deform_attn_forward_npu(const Tensor &value,
                                  const Tensor &value_spatial_shapes,
                                  const Tensor &value_level_start_index,
                                  const Tensor &sampling_locations,
                                  const Tensor &attention_weights,
                                  const int im2col_step) {
  check_support(value, attention_weights);
  at::Tensor value_fp32 = value;
  at::Tensor value_spatial_shapes_int32 = value_spatial_shapes;
  at::Tensor value_level_start_index_int32 = value_level_start_index;
  at::Tensor sampling_locations_fp32 = sampling_locations;
  at::Tensor attention_weights_fp32 = attention_weights;
  if (value.scalar_type() != at::kFloat) {
    value_fp32 = value.to(at::kFloat);
  }
  if (value_spatial_shapes.scalar_type() != at::kInt) {
    value_spatial_shapes_int32 = value_spatial_shapes.to(at::kInt);
  }
  if (value_level_start_index.scalar_type() != at::kInt) {
    value_level_start_index_int32 = value_level_start_index.to(at::kInt);
  }
  if (sampling_locations.scalar_type() != at::kFloat) {
    sampling_locations_fp32 = sampling_locations.to(at::kFloat);
  }
  if (attention_weights.scalar_type() != at::kFloat) {
    attention_weights_fp32 = attention_weights.to(at::kFloat);
  }

  c10::SmallVector<int64_t, 3> output_size = {
      value.size(0), sampling_locations.size(1), value.size(2) * value.size(3)};
  at::Tensor output = at::zeros(output_size, value_fp32.options());

  EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnFunction, value_fp32,
               value_spatial_shapes_int32, value_level_start_index_int32,
               sampling_locations_fp32, attention_weights_fp32, output);

  at::Tensor real_output = output;
  if (value.scalar_type() != at::kFloat) {
    real_output = output.to(value.scalar_type());
  }
  return real_output;
}

std::vector<at::Tensor> ms_deform_attn_backward_npu(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output,
    const int im2col_step) {
  check_support(value, attn_weight);

  // Convert inputs to required types
  at::Tensor value_fp32 = value;
  at::Tensor spatial_shapes_int32 = spatial_shapes;
  at::Tensor level_start_index_int32 = level_start_index;
  at::Tensor sampling_loc_fp32 = sampling_loc;
  at::Tensor attn_weight_fp32 = attn_weight;
  at::Tensor grad_output_fp32 = grad_output;

  if (value.scalar_type() != at::kFloat) {
    value_fp32 = value.to(at::kFloat);
  }
  if (spatial_shapes.scalar_type() != at::kInt) {
    spatial_shapes_int32 = spatial_shapes.to(at::kInt);
  }
  if (level_start_index.scalar_type() != at::kInt) {
    level_start_index_int32 = level_start_index.to(at::kInt);
  }
  if (sampling_loc.scalar_type() != at::kFloat) {
    sampling_loc_fp32 = sampling_loc.to(at::kFloat);
  }
  if (attn_weight.scalar_type() != at::kFloat) {
    attn_weight_fp32 = attn_weight.to(at::kFloat);
  }
  if (grad_output.scalar_type() != at::kFloat) {
    grad_output_fp32 = grad_output.to(at::kFloat);
  }

  // Create output tensors
  at::Tensor grad_value = at::zeros_like(value_fp32);
  at::Tensor grad_sampling_loc = at::zeros_like(sampling_loc_fp32);
  at::Tensor grad_attn_weight = at::zeros_like(attn_weight_fp32);

  EXEC_NPU_CMD(aclnnMultiScaleDeformableAttentionGrad, value_fp32,
               spatial_shapes_int32, level_start_index_int32, sampling_loc_fp32,
               attn_weight_fp32, grad_output_fp32, grad_value, grad_sampling_loc,
               grad_attn_weight);

  // Convert back to original type if needed
  if (value.scalar_type() != at::kFloat) {
    grad_value = grad_value.to(value.scalar_type());
    grad_sampling_loc = grad_sampling_loc.to(sampling_loc.scalar_type());
    grad_attn_weight = grad_attn_weight.to(attn_weight.scalar_type());
  }

  return {grad_value, grad_sampling_loc, grad_attn_weight};
}
