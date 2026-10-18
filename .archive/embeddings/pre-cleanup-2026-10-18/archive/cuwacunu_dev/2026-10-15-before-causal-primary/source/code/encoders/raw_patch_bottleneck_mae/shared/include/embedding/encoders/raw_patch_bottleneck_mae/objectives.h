// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/types.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct HuberOutput {
  torch::Tensor loss, target_counts, eligible_channels, eligible_examples;
  int64_t eligible_channel_count{0}, eligible_example_count{0}, target_cell_count{0};
};

inline HuberOutput hierarchical_huber(const torch::Tensor &prediction,
                                      const torch::Tensor &target,
                                      const torch::Tensor &support,
                                      const torch::Tensor &eligible,
                                      double delta) {
  TORCH_CHECK(std::isfinite(delta) && delta > 0 && prediction.dim() == 4 &&
                  target.sizes() == prediction.sizes() && support.sizes() == prediction.sizes() &&
                  support.scalar_type() == torch::kBool && eligible.scalar_type() == torch::kBool &&
                  eligible.sizes() == prediction.sizes().slice(0, 2),
              "[rpb-mae] invalid hierarchical Huber arguments");
  TORCH_CHECK(torch::isfinite(prediction.masked_select(support)).all().item<bool>() &&
                  torch::isfinite(target.masked_select(support)).all().item<bool>(),
              "[rpb-mae] nonfinite prediction/target on fixed reconstruction support");
  auto counts = support.sum(std::vector<int64_t>{2, 3});
  TORCH_CHECK(torch::equal(counts.gt(0), eligible),
              "[rpb-mae] eligibility must match fixed target support");
  auto examples = eligible.any(1);
  HuberOutput out;
  out.target_counts = counts; out.eligible_channels = eligible; out.eligible_examples = examples;
  out.target_cell_count = counts.sum().item<int64_t>();
  out.eligible_channel_count = eligible.sum().item<int64_t>();
  out.eligible_example_count = examples.sum().item<int64_t>();
  if (out.eligible_example_count == 0) {
    out.loss = torch::zeros({}, prediction.options());
    return out;
  }
  // where before subtraction/square prevents absent NaN and large unused values leaking.
  auto safe_prediction = torch::where(support, prediction, torch::zeros_like(prediction));
  auto safe_target = torch::where(support, target, torch::zeros_like(target));
  auto error = safe_prediction - safe_target;
  auto absolute = error.abs();
  auto quadratic = error.clamp(-delta, delta).square() * 0.5;
  auto cell_loss = torch::where(absolute.le(delta), quadratic, delta * (absolute - 0.5 * delta));
  auto channel_loss = cell_loss.sum(std::vector<int64_t>{2, 3}) / counts.clamp_min(1).to(prediction.dtype());
  auto example_loss = torch::where(eligible, channel_loss, torch::zeros_like(channel_loss)).sum(1) /
                      eligible.sum(1).clamp_min(1).to(prediction.dtype());
  out.loss = example_loss.masked_select(examples).mean();
  TORCH_CHECK(torch::isfinite(out.loss).item<bool>(), "[rpb-mae] nonfinite reconstruction loss");
  return out;
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
