// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/separated_change_config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3 {
using RelativeChangeConfig=SeparatedChangeConfig;
inline constexpr const char *kRelativeChangeModelTag="RPB-v28";
inline constexpr const char *kRelativeChangeArchitectureId="content-protected32-raw-symmetric-relative-change-v1";
inline constexpr const char *kRelativeChangeLayout=
    "historical-raw-f32-scaled-exact-v22-prefix;raw-f64-relative-conditioned-change-suffix;bias-free-waveform-residual";
inline constexpr const char *kRelativeChangeBankFamily="raw-symmetric-relative-f64-rounded32-global-quarter-v1";
inline constexpr const char *kRelativeChangeRawContract="genuine-raw-f64;parent-cpu-raw-f32-before-scaler-v1";
inline constexpr const char *kRelativeChangeIncrementRecipe="cuda-f64-maxabs-scaled-symmetric-relative;u64-to-f32-to-f64;visible-adjacent-zero-safe-v1";
inline constexpr const char *kRelativeChangeSourcePrefix="RPB-v28;raw-symmetric-relative-f64-v1;";
inline constexpr const char *kRelativeChangeBankLayout=
    "canonical-channel-feature-moment;global-mean,rms;quarter0-mean,rms;quarter1-mean,rms;"
    "quarter2-mean,rms;quarter3-mean,rms;quarter-normalized-by-global-rms;visible-adjacent-min2-v1";
inline constexpr const char *kRelativeChangeNormalizationRecipe="valid-f32-bank-cpu-f64-count-sum-sumsq-population-center-std-min2-v1";
inline constexpr double kRelativeChangeScaleFloor=1e-6;
inline constexpr int64_t kRelativeChangeSupportCount=2;
inline void validate_relative_change_config(const RelativeChangeConfig &c){validate_separated_change_config(c);}
inline int64_t relative_change_native_width(const RelativeChangeConfig &c){return separated_change_native_width(c);}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3
