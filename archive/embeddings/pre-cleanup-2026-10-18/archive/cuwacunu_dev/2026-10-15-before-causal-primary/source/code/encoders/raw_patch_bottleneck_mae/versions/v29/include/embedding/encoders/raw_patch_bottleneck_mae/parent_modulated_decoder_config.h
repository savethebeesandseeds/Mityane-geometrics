// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/relative_change_config.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
using ParentModulatedDecoderConfig=RelativeChangeConfig;
inline constexpr const char *kParentModulatedDecoderModelTag="RPB-v29";
inline constexpr const char *kParentModulatedDecoderArchitectureId="content-protected32-raw-relative-parent-modulated-waveform-v1";
inline constexpr const char *kParentModulatedDecoderLayout=
    "historical-raw-f32-scaled-exact-v22-prefix;raw-f64-relative-conditioned-change-suffix;parent-modulated-bias-free-waveform-residual";
inline constexpr const char *kParentModulatedDecoderRecipe=
    "cuda-f32-residual-times-one-plus-detached-parent-linear-CF-gain;zero-biasfree-gain;canonical-CF-broadcast-H-before-semantic-gather-v1";
inline void validate_parent_modulated_decoder_config(const ParentModulatedDecoderConfig &c){validate_relative_change_config(c);}
inline int64_t parent_modulated_decoder_native_width(const ParentModulatedDecoderConfig &c){return relative_change_native_width(c);}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
