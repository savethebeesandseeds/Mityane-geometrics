// SPDX-License-Identifier: MIT
// CPB-v3 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/configurable_spectral_config.h"
#include <array>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3 {

inline constexpr const char *kContentOrderedRelationModelTag = "RPB-v22";
inline constexpr const char *kContentOrderedRelationArchitectureId =
    "configurable-fixed-spectral-plus-zero-start-content-ordered-lag-v1";
inline constexpr const char *kContentOrderedRelationLayout =
    "learned-base-shape-plus-zero-start-content-ordered-lag-projection;"
    "unchanged-fixed-four-per-semantic-pair;"
    "lag-bank-odd-even-no-additive-support;canonical-pair-feature-i-feature-j-lag;"
    "original-tick-lags1-2-4-8;four-endpoint-common-support;"
    "joint-endpoint-centering;min-common2;energy-floor1e-12";
inline constexpr std::array<int64_t, 4> kContentOrderedRelationLags{{1, 2, 4, 8}};
inline constexpr double kContentOrderedRelationEnergyFloor = 1e-12;
inline constexpr int64_t kContentOrderedRelationMinimumCommon = 2;

inline int64_t content_ordered_relation_moment_width(const Config &c) {
  const auto pairs = configurable_spectral_pair_count(c);
  TORCH_CHECK(c.input_width > 0 && c.input_width <=
                  std::numeric_limits<int64_t>::max() / c.input_width / 4,
              "[rpb-v22] feature/lag bank width overflow");
  const auto feature_lags = c.input_width * c.input_width * 4;
  TORCH_CHECK(!pairs || pairs <= std::numeric_limits<int64_t>::max() / feature_lags,
              "[rpb-v22] relation bank width overflow");
  return pairs * feature_lags;
}

inline int64_t content_ordered_relation_bank_width(const Config &c) {
  const auto moments = content_ordered_relation_moment_width(c);
  TORCH_CHECK(moments <= std::numeric_limits<int64_t>::max() / 2,
              "[rpb-v22] two-moment bank width overflow");
  return moments * 2;
}

inline void validate_content_ordered_relation_config(const Config &c) {
  validate_configurable_spectral_config(c);
  const auto width = content_ordered_relation_bank_width(c);
  TORCH_CHECK(!width || configurable_spectral_shape_width(c) <=
                  std::numeric_limits<int64_t>::max() / width,
              "[rpb-v22] learned projection parameter count overflow");
}

inline Config default_content_ordered_relation_config(
    torch::Device device = torch::Device(torch::kCUDA, 0)) {
  auto c = default_configurable_spectral_config(device);
  validate_content_ordered_relation_config(c);
  return c;
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3
