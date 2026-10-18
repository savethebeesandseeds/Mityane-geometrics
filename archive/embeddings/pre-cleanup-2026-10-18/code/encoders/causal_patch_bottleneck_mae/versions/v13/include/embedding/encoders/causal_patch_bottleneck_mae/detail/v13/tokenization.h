// SPDX-License-Identifier: MIT
// CPB-v13 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v13/types.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v13 {

struct VisiblePatches {
  torch::Tensor values, visibility; // active rows, packed patches, P*F
  torch::Tensor positions, valid;   // original patch indices and padding validity
  torch::Tensor row_indices, channel_indices;
  torch::Tensor observation_counts, patch_counts; // all [B,C]
};

// Engineering evidence from the actual guarded forward, never an alternate
// encoder path. No task targets or labels are present.
struct TemporalExecutionTrace {
  torch::Tensor patch_positions, patch_valid, patch_rows, patch_values, patch_visibility;
  torch::Tensor temporal_hidden, contextual_temporal_hidden, pool_weights, contextual_pool_weights;
  // Aggregate mixture weights sum over NONEMPTY local bins; mass=#nonempty.
  // They are not a single probability or the weights for any one export slice.
  torch::Tensor pool_bin_valid, contextual_pool_bin_valid; // bool [active_rows,4]
  torch::Tensor pool_local_weights,contextual_pool_local_weights; // [R,T], anchored local masses
  torch::Tensor pool_global_weights,contextual_pool_global_weights; // [R,T], one valid-history probability
  torch::Tensor pool_mixture_weights,contextual_pool_mixture_weights; // [R,4,T], actual per-export weights
  torch::Tensor pool_bin_mean,pool_bin_variance; // [R,4,W], live two-sum means and centered variances
  torch::Tensor pool_bin_regularized_root,pool_bin_dispersion; // raw root remains1/32 at absent bins
  torch::Tensor contextual_pool_bin_mean,contextual_pool_bin_variance;
  torch::Tensor contextual_pool_bin_regularized_root,contextual_pool_bin_dispersion;
  torch::Tensor pool_projected,contextual_pool_projected; // [R,32], actual post-empty slices; retained u is PRECLIP
  torch::Tensor content_values, content_counts, content_eligible;
  torch::Tensor spectral_values, spectral_counts, spectral_eligible;
  torch::Tensor change_values, change_valid, change_counts, conditioned_bank;
};

inline VisiblePatches tokenize_visible(const Input &input, const Config &c) {
  validate_input(input, c, true);
  const auto B = input.data.size(0), rows = B * c.channel_count;
  const auto K = c.history_length / c.patch_length, PF = c.patch_length * c.input_width;
  auto data = input.data.to(c.device);
  auto mask = input.observed.to(c.device);
  auto safe = torch::where(mask, data, torch::zeros_like(data));
  auto values = safe.reshape({rows, K, PF});
  auto visibility = mask.reshape({rows, K, PF});
  auto valid = visibility.any(-1);
  auto counts = valid.sum(-1);
  auto active = torch::nonzero(counts.gt(0)).reshape({-1});
  auto semantic = channel_indices(input.channel_ids, c, B, c.device).reshape({rows});
  VisiblePatches out;
  out.row_indices = active;
  out.channel_indices = semantic.index_select(0, active);
  out.observation_counts = mask.sum(std::vector<int64_t>{2, 3});
  out.patch_counts = counts.reshape({B, c.channel_count});
  if (active.numel() == 0) {
    out.values = torch::empty({0, 0, PF}, data.options());
    out.visibility = torch::empty({0, 0, PF}, mask.options());
    out.positions = torch::empty({0, 0}, active.options());
    out.valid = torch::empty({0, 0}, mask.options());
    return out;
  }
  const auto padded = counts.max().item<int64_t>();
  std::vector<torch::Tensor> packed_values, packed_visibility, positions, padding;
  for (int64_t i = 0; i < active.numel(); ++i) {
    const auto row = active[i].item<int64_t>();
    auto indices = torch::nonzero(valid[row]).reshape({-1});
    const auto n = indices.numel();
    packed_values.push_back(torch::cat({values[row].index_select(0, indices), torch::zeros({padded - n, PF}, data.options())}, 0));
    packed_visibility.push_back(torch::cat({visibility[row].index_select(0, indices), torch::zeros({padded - n, PF}, mask.options())}, 0));
    positions.push_back(torch::cat({indices, torch::full({padded - n}, -1, indices.options())}, 0));
    padding.push_back(torch::arange(padded, indices.options()).lt(n));
  }
  out.values = torch::stack(packed_values);
  out.visibility = torch::stack(packed_visibility);
  out.positions = torch::stack(positions);
  out.valid = torch::stack(padding);
  return out;
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v13
