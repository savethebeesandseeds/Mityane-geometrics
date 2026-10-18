// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/types.h"
#include <numeric>
#include <random>

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline MaskPlan mask_from_hidden(const torch::Tensor &observed,
                                const torch::Tensor &hidden, const Config &c) {
  validate_config(c);
  TORCH_CHECK(observed.defined() && observed.dim() == 4 && observed.size(0) > 0 &&
                  observed.size(1) == c.channel_count && observed.size(2) == c.history_length &&
                  observed.size(3) == c.input_width && observed.scalar_type() == torch::kBool &&
                  hidden.defined() && hidden.scalar_type() == torch::kBool && hidden.sizes() == observed.sizes(),
              "[rpb-mae] training masks require configured BCHF bool tensors");
  const auto B = observed.size(0), K = c.history_length / c.patch_length;
  auto a = hidden.to(observed.device());
  auto patch_a = a.reshape({B, c.channel_count, K, c.patch_length * c.input_width});
  auto selected = patch_a.select(-1, 0);
  TORCH_CHECK(torch::equal(patch_a, selected.unsqueeze(-1).expand_as(patch_a)),
              "[rpb-mae] artificial masks must select complete patches");
  auto patch_o = observed.reshape({B, c.channel_count, K, -1}).any(-1);
  TORCH_CHECK(!selected.logical_and(patch_o.logical_not()).any().item<bool>(),
              "[rpb-mae] artificial mask selects an entirely unobserved patch");
  auto visible = observed.logical_and(a.logical_not());
  auto target = observed.logical_and(a);
  auto visible_patches = visible.reshape({B, c.channel_count, K, -1}).any(-1).sum(-1);
  auto target_patches = selected.sum(-1);
  auto eligible = patch_o.sum(-1).ge(3).logical_and(visible_patches.ge(2)).logical_and(target_patches.ge(1));
  TORCH_CHECK(!target_patches.gt(0).logical_and(eligible.logical_not()).any().item<bool>(),
              "[rpb-mae] training requires two visible and one observed target patches");
  return {a, visible, target, eligible};
}

inline MaskPlan make_training_mask(const torch::Tensor &observed, const Config &c,
                                  uint64_t seed) {
  validate_config(c);
  TORCH_CHECK(observed.defined() && observed.dim() == 4 && observed.size(0) > 0 &&
                  observed.scalar_type() == torch::kBool && observed.size(1) == c.channel_count &&
                  observed.size(2) == c.history_length && observed.size(3) == c.input_width,
              "[rpb-mae] observed mask must have the configured BCHF shape");
  const auto B = observed.size(0), K = c.history_length / c.patch_length;
  auto patch_o = observed.to(torch::kCPU).reshape({B * c.channel_count, K, -1}).any(-1).contiguous();
  auto selected = torch::zeros_like(patch_o);
  auto support = patch_o.accessor<bool, 2>();
  auto choices = selected.accessor<bool, 2>();
  std::mt19937_64 rng(seed);
  for (int64_t row = 0; row < B * c.channel_count; ++row) {
    std::vector<int64_t> positions;
    for (int64_t k = 0; k < K; ++k) if (support[row][k]) positions.push_back(k);
    const auto M = static_cast<int64_t>(positions.size());
    if (M < 3) continue;
    const auto count = std::clamp<int64_t>(static_cast<int64_t>(std::floor(c.mask_ratio * M + 0.5)), 1, M - 2);
    std::shuffle(positions.begin(), positions.end(), rng);
    for (int64_t j = 0; j < count; ++j) choices[row][positions[j]] = true;
  }
  auto hidden = selected.reshape({B, c.channel_count, K, 1, 1})
                    .expand({B, c.channel_count, K, c.patch_length, c.input_width})
                    .reshape_as(observed).to(observed.device());
  return mask_from_hidden(observed, hidden, c);
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
