// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/tokenization.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct VisibleDifferences {
  torch::Tensor values, visibility; // original [B,C,H,F], then packed [R,K,P*F]
};

inline VisibleDifferences visible_first_differences(const Input &input, const Config &c) {
  validate_input(input, c, true);
  const auto mask = input.observed.to(c.device);
  const auto data = input.data.to(c.device);
  const auto safe = torch::where(mask, data, torch::zeros_like(data));
  const auto H = c.history_length;
  TORCH_CHECK(H > 1, "[rpb-mae] visible differences require two original time positions");
  auto pair = torch::cat({torch::zeros_like(mask.narrow(2, 0, 1)),
      mask.narrow(2, 1, H - 1).logical_and(mask.narrow(2, 0, H - 1))}, 2);
  auto delta = torch::cat({torch::zeros_like(safe.narrow(2, 0, 1)),
      safe.narrow(2, 1, H - 1) - safe.narrow(2, 0, H - 1)}, 2);
  delta = torch::where(pair, delta, torch::zeros_like(delta));
  TORCH_CHECK(torch::isfinite(delta).all().item<bool>(), "[rpb-mae] visible first difference overflow");
  return {delta, pair};
}

inline VisibleDifferences pack_visible_differences(const VisibleDifferences &differences,
    const VisiblePatches &patches, const Config &c) {
  validate_config(c);
  const auto K = c.history_length / c.patch_length, PF = c.patch_length * c.input_width;
  TORCH_CHECK(differences.values.defined() && differences.values.dim() == 4 &&
      differences.values.size(1) == c.channel_count && differences.values.size(2) == c.history_length &&
      differences.values.size(3) == c.input_width && differences.values.device().type() == c.device.type() &&
      (!c.device.has_index() || differences.values.device().index() == c.device.index()) &&
      differences.values.scalar_type() == c.dtype && differences.visibility.defined() &&
      differences.visibility.device() == differences.values.device() && differences.visibility.scalar_type() == torch::kBool &&
      differences.visibility.sizes() == differences.values.sizes(), "[rpb-mae] difference input geometry mismatch");
  TORCH_CHECK(patches.positions.defined() && patches.row_indices.defined() && patches.valid.defined() &&
      patches.positions.scalar_type() == torch::kInt64 && patches.row_indices.scalar_type() == torch::kInt64 &&
      patches.valid.scalar_type() == torch::kBool && patches.positions.sizes() == patches.valid.sizes() &&
      patches.positions.dim() == 2 && patches.row_indices.dim() == 1 &&
      patches.positions.size(0) == patches.row_indices.numel() &&
      patches.positions.device() == differences.values.device() &&
      patches.row_indices.device() == differences.values.device() && patches.valid.device() == differences.values.device(),
      "[rpb-mae] difference packing must use original raw patch indices");
  if (patches.row_indices.numel() == 0)
    return {torch::zeros_like(patches.values), torch::zeros_like(patches.visibility)};
  const auto rows = differences.values.size(0) * c.channel_count;
  TORCH_CHECK(patches.row_indices.ge(0).all().item<bool>() && patches.row_indices.lt(rows).all().item<bool>() &&
      torch::where(patches.valid, patches.positions.ge(0).logical_and(patches.positions.lt(K)),
                   patches.positions.eq(-1)).all().item<bool>(), "[rpb-mae] invalid original difference packing indices");
  auto positions = patches.positions.clamp_min(0).unsqueeze(-1).expand({-1, -1, PF});
  auto values = differences.values.reshape({rows, K, PF}).index_select(0, patches.row_indices).gather(1, positions);
  auto visibility = differences.visibility.reshape({rows, K, PF}).index_select(0, patches.row_indices).gather(1, positions);
  visibility = visibility.logical_and(patches.valid.unsqueeze(-1));
  return {torch::where(visibility, values, torch::zeros_like(values)), visibility};
}

// No Linear constructor/reset: this last child consumes no initialization RNG.
struct VisibleDifferenceProjectionImpl : torch::nn::Module {
  VisibleDifferenceProjectionImpl(int64_t width, int64_t patch_features) {
    TORCH_CHECK(width > 0 && patch_features > 0, "[rpb-mae] difference projection dimensions must be positive");
    weight = register_parameter("weight", torch::zeros({width, 2 * patch_features}), true);
  }
  torch::Tensor forward(const torch::Tensor &input) {
    TORCH_CHECK(input.size(-1) == weight.size(1), "[rpb-mae] difference projection width mismatch");
    return torch::matmul(input, weight.transpose(0, 1));
  }
  torch::Tensor weight;
};
TORCH_MODULE(VisibleDifferenceProjection);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
