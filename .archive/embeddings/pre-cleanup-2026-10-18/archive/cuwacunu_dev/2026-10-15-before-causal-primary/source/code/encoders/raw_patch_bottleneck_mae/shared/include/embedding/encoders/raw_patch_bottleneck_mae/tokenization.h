// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/types.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct VisiblePatches {
  torch::Tensor values, visibility; // active rows, packed patches, P*F
  torch::Tensor positions, valid;   // original patch indices and padding validity
  torch::Tensor row_indices, channel_indices;
  torch::Tensor observation_counts, patch_counts; // all [B,C]
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

} // namespace embedding::encoders::raw_patch_bottleneck_mae
