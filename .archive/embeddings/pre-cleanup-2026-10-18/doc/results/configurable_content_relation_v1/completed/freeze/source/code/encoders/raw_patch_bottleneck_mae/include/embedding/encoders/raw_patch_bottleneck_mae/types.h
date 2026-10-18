// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/config.h"
#include <algorithm>

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct Input {
  torch::Tensor data;        // [B,C,H,F]; raw float32/64 or normalized float32
  torch::Tensor observed;    // same shape, bool
  torch::Tensor channel_ids; // [C] or [B,C], int64; storage order is not identity
  torch::Tensor endpoints;   // optional for direct model calls; workflow archives require finite [B]
  double sampling_interval{1.0};
};

inline torch::Tensor channel_indices(const torch::Tensor &ids, const Config &c,
                                     int64_t batch, torch::Device device) {
  TORCH_CHECK(ids.defined() && ids.scalar_type() == torch::kInt64 &&
                  ((ids.dim() == 1 && ids.size(0) == c.channel_count) ||
                   (ids.dim() == 2 && ids.size(0) == batch && ids.size(1) == c.channel_count)),
              "[rpb-mae] channel_ids must be int64 [C] or [B,C]");
  auto rows = (ids.dim() == 1 ? ids.unsqueeze(0).expand({batch, c.channel_count}) : ids)
                  .to(torch::kCPU).contiguous();
  auto a = rows.accessor<int64_t, 2>();
  const auto known = resolved_channel_ids(c);
  std::vector<int64_t> indices;
  for (int64_t b = 0; b < batch; ++b) {
    std::set<int64_t> seen;
    for (int64_t j = 0; j < c.channel_count; ++j) {
      const auto it = std::find(known.begin(), known.end(), a[b][j]);
      TORCH_CHECK(it != known.end() && seen.insert(a[b][j]).second,
                  "[rpb-mae] unknown or duplicate semantic channel ID");
      indices.push_back(static_cast<int64_t>(it - known.begin()));
    }
  }
  return torch::tensor(indices, torch::TensorOptions().dtype(torch::kInt64))
      .reshape({batch, c.channel_count}).to(device);
}

inline void validate_input(const Input &input, const Config &c,
                           bool normalized = false) {
  validate_config(c);
  TORCH_CHECK(input.data.defined() && input.data.dim() == 4 && input.data.size(0) > 0 &&
                  input.data.size(1) == c.channel_count && input.data.size(2) == c.history_length &&
                  input.data.size(3) == c.input_width,
              "[rpb-mae] input must be nonempty BCHF with configured dimensions");
  TORCH_CHECK((input.data.scalar_type() == torch::kFloat32 ||
                   input.data.scalar_type() == torch::kFloat64) &&
                  (!normalized || input.data.scalar_type() == torch::kFloat32),
              "[rpb-mae] raw input must be float32/64; normalized input must be float32");
  TORCH_CHECK(input.observed.defined() && input.observed.scalar_type() == torch::kBool &&
                  input.observed.sizes() == input.data.sizes(),
              "[rpb-mae] observed must be bool with the input shape");
  auto observed = input.observed.to(input.data.device());
  TORCH_CHECK(torch::isfinite(input.data.masked_select(observed)).all().item<bool>(),
              "[rpb-mae] observed values must be finite");
  (void)channel_indices(input.channel_ids, c, input.data.size(0), torch::kCPU);
  TORCH_CHECK(std::isfinite(input.sampling_interval) && input.sampling_interval > 0 &&
                  std::abs(input.sampling_interval - c.sampling_interval) <=
                      1e-9 * std::max(input.sampling_interval, c.sampling_interval),
              "[rpb-mae] sampling interval must match the configured uniform schema");
  if (input.endpoints.defined()) {
    TORCH_CHECK(input.endpoints.dim() == 1 && input.endpoints.size(0) == input.data.size(0) &&
                    input.endpoints.is_floating_point() &&
                    torch::isfinite(input.endpoints).all().item<bool>(),
                "[rpb-mae] endpoints must be finite floating [B]");
  }
}

struct EncodeOutput {
  torch::Tensor z_local, z_global;
  // Global fields are equal-valid means in modes 0/1, semantic channel-summary
  // learned pools in mode 2, original-patch-state learned pools in mode 3.
  // In modes 1/2/3 the selected global is the sole decoder input. Mode 3 leaves
  // per-channel D summaries diagnostic-only; reconstruction does not train them.
  // Prospective mode2/input-source1 likewise retains per-channel D diagnostics
  // but pools learned temporal W summaries into the sole global D decoder input.
  // Undefined with channel_mixer_layers=0. Support remains observed-only.
  torch::Tensor z_contextual, z_contextual_global;
  torch::Tensor channel_valid_mask, sample_valid_mask;
  torch::Tensor visible_observation_counts, visible_patch_counts;
  torch::Tensor channel_ids; // expanded [B,C], matching local storage order
};

inline torch::Tensor compact_reconstruction_export(const EncodeOutput &output,
                                                   const Config &config) {
  const auto &selected = config.global_bottleneck_mode > 0 ?
      (config.channel_mixer_layers > 0 ? output.z_contextual_global : output.z_global) :
      (config.channel_mixer_layers > 0 ? output.z_contextual : output.z_local);
  TORCH_CHECK(selected.defined(),
              "[rpb-mae] selected compact reconstruction export is undefined");
  return selected;
}

struct MaskPlan {
  torch::Tensor hidden, visible, target;
  torch::Tensor eligible_channels;
};

struct ForwardOutput {
  EncodeOutput encoding;
  torch::Tensor reconstruction, loss;
  torch::Tensor eligible_channels, eligible_examples, target_counts;
  int64_t eligible_channel_count{0}, eligible_example_count{0}, target_cell_count{0};
};

} // namespace embedding::encoders::raw_patch_bottleneck_mae
