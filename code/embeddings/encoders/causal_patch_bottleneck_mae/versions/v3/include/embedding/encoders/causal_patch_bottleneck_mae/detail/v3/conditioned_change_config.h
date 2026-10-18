// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/separated_change_config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3 {
using ConditionedChangeConfig=SeparatedChangeConfig;
inline constexpr const char *kConditionedChangeModelTag="RPB-v26";
inline constexpr const char *kConditionedChangeArchitectureId="content-protected32-separated-train-standardized-change-v1";
inline constexpr const char *kConditionedChangeLayout=
    "complete-v22-native-prefix;train-centered-standardized-v24-change-suffix;bias-free-waveform-residual";
inline constexpr const char *kConditionedChangeBankLayout=
    "canonical-channel-feature-moment;global-mean,rms;quarter0-mean,rms;quarter1-mean,rms;"
    "quarter2-mean,rms;quarter3-mean,rms;quarter-normalized-by-global-rms;visible-adjacent-min2-v1";
inline constexpr const char *kConditionedChangeNormalizationRecipe="valid-f32-bank-cpu-f64-count-sum-sumsq-population-center-std-min2-v1";
inline constexpr double kConditionedChangeScaleFloor=1e-6;
inline constexpr int64_t kConditionedChangeSupportCount=2;
inline void validate_conditioned_change_config(const ConditionedChangeConfig &c){validate_separated_change_config(c);}
inline int64_t conditioned_change_native_width(const ConditionedChangeConfig &c){return separated_change_native_width(c);}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3
