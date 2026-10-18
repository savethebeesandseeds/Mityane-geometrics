// SPDX-License-Identifier: MIT
#pragma once

#include <torch/torch.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct Config {
  int64_t channel_count{3}, history_length{32}, input_width{3}, patch_length{8};
  int64_t encoder_width{64}, export_width{32}, num_layers{3}, num_heads{4};
  int64_t channel_mixer_layers{0}; // 0 preserves the independent-channel encoder.
  // 0 keeps the original post-temporal mixer. 1 mixes aligned projected states
  // before temporal encoding and retains a separate independent local pass.
  int64_t channel_mixer_placement{0};
  // 0: per-channel decoder; 1: mean global; 2: learned channel-summary global;
  // 3: learned original-patch-state global, before channel-summary compression.
  int64_t global_bottleneck_mode{0};
  // Prospective closed route: 0 preserves D channel summaries; 1 uses the
  // existing learned temporally pooled W summaries before D diagnostics.
  int64_t global_pool_input_source{0};
  // 0 preserves raw patches. 1 adds a visible adjacent-difference projection.
  int64_t temporal_difference_input{0};
  int64_t feedforward_width{256}, decoder_hidden_width{128};
  double dropout{0.0}, layer_norm_epsilon{1e-5}, mask_ratio{0.25};
  double huber_delta{1.0}, scale_floor{1e-6}, sampling_interval{1.0};
  torch::Device device{torch::kCPU};
  torch::Dtype dtype{torch::kFloat32};
  std::vector<int64_t> channel_ids{};
};

inline std::vector<int64_t> resolved_channel_ids(const Config &config) {
  if (!config.channel_ids.empty()) return config.channel_ids;
  std::vector<int64_t> ids;
  for (int64_t i = 0; i < config.channel_count; ++i) ids.push_back(i);
  return ids;
}

inline void validate_config(const Config &c) {
  TORCH_CHECK(c.channel_count > 0 && c.history_length > 0 && c.input_width > 0 &&
                  c.patch_length > 0 && c.encoder_width > 0 && c.export_width > 0 &&
                  c.num_layers > 0 && c.num_heads > 0 && c.feedforward_width > 0 &&
                  c.decoder_hidden_width > 0,
              "[rpb-mae] dimensions must be positive");
  TORCH_CHECK(c.channel_mixer_layers >= 0,
              "[rpb-mae] channel_mixer_layers must be nonnegative");
  TORCH_CHECK(c.global_bottleneck_mode >= 0 && c.global_bottleneck_mode <= 3,
              "[rpb-mae] global_bottleneck_mode must be 0, 1, 2 or 3");
  TORCH_CHECK(c.global_pool_input_source == 0 || c.global_pool_input_source == 1,
              "[rpb-mae] global_pool_input_source must be 0 or 1");
  TORCH_CHECK(c.temporal_difference_input == 0 || c.temporal_difference_input == 1,
              "[rpb-mae] temporal_difference_input must be 0 or 1");
  TORCH_CHECK(c.temporal_difference_input == 0 ||
                  (c.channel_count == 3 && c.history_length == 32 && c.input_width == 3 &&
                   c.patch_length == 8 && c.encoder_width == 64 && c.export_width == 32 &&
                   c.channel_mixer_layers == 1 && c.channel_mixer_placement == 1 &&
                   c.global_bottleneck_mode == 2 && c.global_pool_input_source == 0 && c.dropout == 0),
              "[rpb-mae] visible differences require C3/H32/F3/P8/W64/D32/mixer1/early/mode2/source0/dropout0");
  TORCH_CHECK(c.global_pool_input_source == 0 ||
                  (c.global_bottleneck_mode == 2 && c.channel_mixer_layers == 1 &&
                   c.channel_mixer_placement == 1 && c.dropout == 0),
              "[rpb-mae] pooled-W global input requires mode2/mixer1/early/dropout0");
  if (c.global_pool_input_source == 1)
    TORCH_CHECK(c.encoder_width < std::numeric_limits<int64_t>::max() &&
                    c.channel_count <= std::numeric_limits<int64_t>::max() / (c.encoder_width + 1),
                "[rpb-mae] pooled-W global input width overflow");
  TORCH_CHECK(c.channel_mixer_placement == 0 || c.channel_mixer_placement == 1,
              "[rpb-mae] channel_mixer_placement must be 0 or 1");
  TORCH_CHECK(c.channel_mixer_placement == 0 ||
                  (c.global_bottleneck_mode == 2 && c.channel_mixer_layers == 1 && c.dropout == 0),
              "[rpb-mae] early mixer requires mode2/mixer1/dropout0");
  TORCH_CHECK(c.history_length % c.patch_length == 0,
              "[rpb-mae] history must be divisible by patch length");
  if (c.global_bottleneck_mode == 3) {
    const auto K = c.history_length / c.patch_length;
    TORCH_CHECK(c.encoder_width < std::numeric_limits<int64_t>::max() &&
                    c.channel_count <= std::numeric_limits<int64_t>::max() / K &&
                    c.channel_count * K <= std::numeric_limits<int64_t>::max() / (c.encoder_width + 1),
                "[rpb-mae] original-patch global pool input width overflow");
  }
  TORCH_CHECK(c.encoder_width % c.num_heads == 0,
              "[rpb-mae] encoder width must be divisible by heads");
  TORCH_CHECK(std::isfinite(c.dropout) && c.dropout >= 0 && c.dropout < 1 &&
                  std::isfinite(c.layer_norm_epsilon) && c.layer_norm_epsilon > 0 &&
                  std::isfinite(c.mask_ratio) && c.mask_ratio > 0 && c.mask_ratio < 1 &&
                  std::isfinite(c.huber_delta) && c.huber_delta > 0 &&
                  std::isfinite(c.scale_floor) && c.scale_floor > 0 &&
                  std::isfinite(c.sampling_interval) && c.sampling_interval > 0,
              "[rpb-mae] invalid/nonfinite numerical configuration");
  TORCH_CHECK(c.dtype == torch::kFloat32, "[rpb-mae] model dtype must be float32");
  const auto ids = resolved_channel_ids(c);
  TORCH_CHECK(static_cast<int64_t>(ids.size()) == c.channel_count &&
                  std::set<int64_t>(ids.begin(), ids.end()).size() == ids.size(),
              "[rpb-mae] channel IDs must be unique and match channel_count");
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
