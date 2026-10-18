// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v10/consecutive_change_config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v10 {
inline constexpr const char *kSeparatedChangeModelTag="RPB-v24";
inline constexpr const char *kSeparatedChangeArchitectureId=
    "v22-native-prefix-plus-independent-change-bottleneck-linear-residual-v1";
inline constexpr const char *kSeparatedChangeLayout=
    "complete-v22-native-prefix;independent-bias-free-change-projection-suffix;"
    "unchanged-adjacent-visible-global-and-quarter-moments-v1;"
    "frozen-level-forward;canonical-channel-feature-time-linear-residual-decoder";
struct SeparatedChangeConfig { Config level;int64_t change_width{8}; };
inline void validate_separated_change_config(const SeparatedChangeConfig &c) {
  validate_consecutive_change_config(c.level);
  const auto limit=std::numeric_limits<int64_t>::max();
  const auto &p=c.level;
  TORCH_CHECK(c.change_width>0 && p.export_width<=limit-c.change_width &&
      c.change_width<=limit/consecutive_change_bank_width(p) &&
      p.channel_count<=limit/p.history_length/p.input_width &&
      c.change_width<=limit/(p.channel_count*p.history_length*p.input_width),
      "[rpb-v24] positive change dimension and safe native/projection/decoder sizes required");
}
inline int64_t separated_change_native_width(const SeparatedChangeConfig &c) {
  validate_separated_change_config(c);return c.level.export_width+c.change_width;
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v10
