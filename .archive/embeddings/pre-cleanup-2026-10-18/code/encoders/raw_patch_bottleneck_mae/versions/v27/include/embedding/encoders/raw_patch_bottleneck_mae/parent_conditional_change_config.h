// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/conditioned_change_normalizer.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
using ParentConditionalChangeConfig=SeparatedChangeConfig;
inline constexpr const char *kParentConditionalChangeModelTag="RPB-v27";
inline constexpr const char *kParentConditionalChangeArchitectureId="content-protected32-parent-conditional-train-change-v1";
inline constexpr const char *kParentConditionalChangeLayout=
    "complete-v22-native-prefix;frozen-parent-conditional-standardized-bank-residual-suffix;bias-free-waveform-residual";
inline constexpr const char *kParentConditionalChangeBankLayout=
    "canonical-channel-feature-moment;global-mean,rms;quarter0-mean,rms;quarter1-mean,rms;"
    "quarter2-mean,rms;quarter3-mean,rms;quarter-normalized-by-global-rms;visible-adjacent-min2-v1";
inline constexpr const char *kParentConditionalChangeNormalizationRecipe="valid-f32-bank-cpu-f64-count-sum-sumsq-population-center-std-min2-v1";
inline constexpr double kParentConditionalChangeScaleFloor=1e-6;
inline constexpr int64_t kParentConditionalChangeSupportCount=2;
inline void validate_parent_conditional_change_config(const ParentConditionalChangeConfig &c){validate_separated_change_config(c);}
inline int64_t parent_conditional_change_native_width(const ParentConditionalChangeConfig &c){return separated_change_native_width(c);}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
