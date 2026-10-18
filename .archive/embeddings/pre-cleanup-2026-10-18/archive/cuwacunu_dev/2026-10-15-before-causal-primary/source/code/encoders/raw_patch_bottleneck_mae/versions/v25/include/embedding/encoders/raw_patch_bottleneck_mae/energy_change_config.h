// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_config.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
using EnergyChangeConfig=SeparatedChangeConfig;
inline constexpr const char *kEnergyChangeModelTag="RPB-v25";
inline constexpr const char *kEnergyChangeArchitectureId="content-protected32-separate-change-energy-aux-v1";
inline constexpr const char *kEnergyChangeLayout=
    "complete-v22-native-prefix;unchanged-v24-independent-change-suffix;"
    "canonical-visible-adjacent-global-and-quarter-moments-v1;training-only-bias-free-moment-decoder";
inline constexpr const char *kEnergyChangeMomentLayout=
    "canonical-channel-feature-moment;global-mean,rms;quarter0-mean,rms;quarter1-mean,rms;"
    "quarter2-mean,rms;quarter3-mean,rms;quarter-normalized-by-global-rms;visible-adjacent-min2-v1";
inline constexpr double kEnergyChangeMomentScaleFloor=1e-6;
inline constexpr double kEnergyChangeAuxiliaryWeight=.1;
inline constexpr double kEnergyChangeHuberDelta=1.;
inline constexpr double kEnergyChangeSignedWeight=.5;
inline constexpr double kEnergyChangeEnergyWeight=.5;
inline void validate_energy_change_config(const EnergyChangeConfig &c){validate_separated_change_config(c);}
inline int64_t energy_change_native_width(const EnergyChangeConfig &c){return separated_change_native_width(c);}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
