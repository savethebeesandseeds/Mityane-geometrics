// SPDX-License-Identifier: MIT
// CPB-v13 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v13/temporal_pooling.h"

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v13/config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v13 {

inline constexpr const char *kConfigurableSpectralModelTag = "RPB-v20";
inline constexpr const char *kConfigurableSpectralArchitectureId =
    "early-configurable-shape-plus-fixed-two-band-complex-relations-v1";
inline constexpr const char *kConfigurableSpectralLayout =
    "learned-D-minus-four-per-semantic-pair;fixed-four-per-semantic-pair;"
    "canonical-all-channel-pairs;ReLow-ImLow-ReHigh-ImHigh;"
    "q1-3-low-q4-8-high;original-H;feature-pair-mean;pair-unit-L2-floor1e-12";

inline int64_t configurable_spectral_pair_count(const Config &c) {
  TORCH_CHECK(c.channel_count > 0 &&
                  c.channel_count <= std::numeric_limits<int64_t>::max() / c.channel_count,
              "[rpb-v20] channel pair count overflow or nonpositive channels");
  return c.channel_count * (c.channel_count - 1) / 2;
}

inline int64_t configurable_spectral_width(const Config &c) {
  const auto pairs = configurable_spectral_pair_count(c);
  TORCH_CHECK(pairs <= std::numeric_limits<int64_t>::max() / 4,
              "[rpb-v20] spectral width overflow");
  return pairs * 4;
}

inline int64_t configurable_spectral_shape_width(const Config &c) {
  const auto spectral = configurable_spectral_width(c);
  TORCH_CHECK(c.export_width > spectral,
              "[rpb-v20] embedding width must exceed four times the channel-pair count");
  return c.export_width - spectral;
}

// Internal patch padding belongs to the learned backbone only. Harmonics use
// the original history length and original time coordinates without padding.
inline Config configurable_spectral_backbone_config(const Config &logical) {
  TORCH_CHECK(logical.history_length > 0 && logical.patch_length > 0 &&
                  logical.history_length <= std::numeric_limits<int64_t>::max() -
                                                (logical.patch_length - 1),
              "[rpb-v20] padded history overflow or nonpositive dimensions");
  auto out = logical;
  out.history_length = ((logical.history_length + logical.patch_length - 1) /
                        logical.patch_length) * logical.patch_length;
  return out;
}

// Preserved input/scaler validation has a divisibility rule. Patch length does
// not enter its arithmetic, so this validation-only copy retains logical H.
inline Config configurable_spectral_input_config(const Config &logical) {
  auto out = logical;
  out.patch_length = 1;
  return out;
}

inline void validate_configurable_spectral_config(const Config &c) {
  TORCH_CHECK(c.device.is_cuda() && c.dtype == torch::kFloat32,
              "[rpb-v20] model execution requires actual CUDA float32");
  TORCH_CHECK(c.history_length >= 17,
              "[rpb-v20] history must contain at least 17 ticks for eight non-Nyquist harmonics");
  TORCH_CHECK(c.channel_mixer_layers == 1 && c.channel_mixer_placement == 1 &&
                  c.global_bottleneck_mode == 2 && c.global_pool_input_source == 0 &&
                  c.temporal_difference_input == 0 && c.dropout == 0,
              "[rpb-v20] requires early mixer1, global mode2/source0, raw patches and dropout0");
  const auto pairs = configurable_spectral_pair_count(c);
  (void)configurable_spectral_shape_width(c);
  TORCH_CHECK(c.input_width > 0 &&
                  c.input_width <= std::numeric_limits<int64_t>::max() / c.input_width / 4,
              "[rpb-v20] feature-pair width overflow or nonpositive features");
  const auto feature_pair_width = c.input_width * c.input_width * 4;
  TORCH_CHECK(pairs == 0 || pairs <= std::numeric_limits<int64_t>::max() / feature_pair_width,
              "[rpb-v20] spectral bank width overflow");
  TORCH_CHECK(c.patch_length > 0 &&
                  c.patch_length <= std::numeric_limits<int64_t>::max() / c.input_width / 2,
              "[rpb-v20] patch projection width overflow");
  TORCH_CHECK(c.export_width < std::numeric_limits<int64_t>::max() &&
                  c.channel_count <= std::numeric_limits<int64_t>::max() / (c.export_width + 1),
              "[rpb-v20] global pool width overflow");
  validate_temporal_pooling_config(configurable_spectral_backbone_config(c));
}

inline Config default_configurable_spectral_config(
    torch::Device device = torch::Device(torch::kCUDA, 0)) {
  Config c;
  c.input_width = 9;
  c.history_length = 128;
  c.export_width = 32;
  c.device = device;
  c.channel_mixer_layers = 1;
  c.channel_mixer_placement = 1;
  c.global_bottleneck_mode = 2;
  validate_configurable_spectral_config(c);
  return c;
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v13
