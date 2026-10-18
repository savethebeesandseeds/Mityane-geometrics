// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v13/content_ordered_relation_config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v13 {

inline constexpr const char *kConsecutiveChangeModelTag = "RPB-v23";
inline constexpr const char *kConsecutiveChangeArchitectureId =
    "v22-level-relation-plus-zero-start-consecutive-change-v1";
inline constexpr const char *kConsecutiveChangeLayout =
    "v22-learned-plus-zero-start-change-projection;unchanged-fixed-four-per-pair;"
    "canonical-channel-feature;global-mean-rms;four-original-time-quarter-mean-rms;"
    "quarter-moments-divided-by-global-rms;adjacent-visible-endpoints;"
    "minimum-global-and-quarter-count2;rms-floor1e-6;no-additive-support";
inline constexpr double kConsecutiveChangeRmsFloor = 1e-6;
inline constexpr int64_t kConsecutiveChangeMinimumCount = 2;
inline constexpr int64_t kConsecutiveChangeMoments = 10;

inline int64_t consecutive_change_bank_width(const Config &c) {
  validate_content_ordered_relation_config(c);
  TORCH_CHECK(c.channel_count <= std::numeric_limits<int64_t>::max() /
      c.input_width / kConsecutiveChangeMoments, "[rpb-v23] change bank width overflow");
  return c.channel_count * c.input_width * kConsecutiveChangeMoments;
}
inline void validate_consecutive_change_config(const Config &c) {
  const auto width = consecutive_change_bank_width(c);
  TORCH_CHECK(configurable_spectral_shape_width(c) <=
      std::numeric_limits<int64_t>::max() / width,
      "[rpb-v23] change projection width overflow");
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v13
