// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace embedding {

// Input dimensions and execution options shared by representation encoders.
// Supply the dimensions explicitly; no encoder architecture is implied.
struct input_shape_t {
  int64_t channel_count{0};
  int64_t history_length{0};
  int64_t input_width{0};
  torch::Dtype dtype{torch::kFloat32};
  torch::Device device{torch::kCPU};
};

struct input_t {
  torch::Tensor data{};         // [B,C,H,F]
  torch::Tensor feature_mask{}; // [B,C,H,F], bool; true means observed
};

struct vicreg_stability_loss_options_t {
  double invariance_weight{25.0};
  double variance_weight{25.0};
  double covariance_weight{1.0};
  double variance_floor{1.0};
  double eps{1e-4};
};

struct vicreg_stability_loss_result_t {
  torch::Tensor loss{};
  torch::Tensor invariance_loss{};
  torch::Tensor variance_loss{};
  torch::Tensor covariance_loss{};
  int64_t valid_rows{0};
  bool statistics_supported{false}; // variance/covariance need at least 2 rows
};

} // namespace embedding
