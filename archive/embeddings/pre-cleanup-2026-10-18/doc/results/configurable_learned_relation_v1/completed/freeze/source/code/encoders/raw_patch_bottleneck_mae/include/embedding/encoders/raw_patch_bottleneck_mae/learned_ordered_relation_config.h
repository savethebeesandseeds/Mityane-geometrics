// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_config.h"
#include <array>

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kLearnedOrderedRelationModelTag = "RPB-v21";
inline constexpr const char *kLearnedOrderedRelationArchitectureId =
    "configurable-fixed-spectral-plus-zero-start-learned-ordered-lag-v1";
inline constexpr const char *kLearnedOrderedRelationLayout =
    "learned-base-shape-plus-zero-start-ordered-lag-projection;"
    "unchanged-fixed-four-per-semantic-pair;"
    "lag-bank-odd-even-support;canonical-pair-feature-i-feature-j-lag;"
    "original-tick-lags1-2-4-8;four-endpoint-common-support;"
    "joint-endpoint-centering;min-common2;energy-floor1e-12";
inline constexpr std::array<int64_t, 4> kLearnedOrderedRelationLags{{1, 2, 4, 8}};
inline constexpr double kLearnedOrderedRelationEnergyFloor = 1e-12;
inline constexpr int64_t kLearnedOrderedRelationMinimumCommon = 2;

inline int64_t learned_ordered_relation_moment_width(const Config &c) {
  const auto pairs = configurable_spectral_pair_count(c);
  TORCH_CHECK(c.input_width > 0 && c.input_width <=
                  std::numeric_limits<int64_t>::max() / c.input_width / 4,
              "[rpb-v21] feature/lag bank width overflow");
  const auto feature_lags = c.input_width * c.input_width * 4;
  TORCH_CHECK(!pairs || pairs <= std::numeric_limits<int64_t>::max() / feature_lags,
              "[rpb-v21] relation bank width overflow");
  return pairs * feature_lags;
}

inline int64_t learned_ordered_relation_bank_width(const Config &c) {
  const auto moments = learned_ordered_relation_moment_width(c);
  TORCH_CHECK(moments <= std::numeric_limits<int64_t>::max() / 3,
              "[rpb-v21] three-moment bank width overflow");
  return moments * 3;
}

inline void validate_learned_ordered_relation_config(const Config &c) {
  validate_configurable_spectral_config(c);
  const auto width = learned_ordered_relation_bank_width(c);
  TORCH_CHECK(!width || configurable_spectral_shape_width(c) <=
                  std::numeric_limits<int64_t>::max() / width,
              "[rpb-v21] learned projection parameter count overflow");
}

inline Config default_learned_ordered_relation_config(
    torch::Device device = torch::Device(torch::kCUDA, 0)) {
  auto c = default_configurable_spectral_config(device);
  validate_learned_ordered_relation_config(c);
  return c;
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
