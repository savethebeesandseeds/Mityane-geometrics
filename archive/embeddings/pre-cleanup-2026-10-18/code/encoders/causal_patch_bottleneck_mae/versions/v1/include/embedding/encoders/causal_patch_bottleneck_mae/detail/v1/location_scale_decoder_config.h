// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v1/relative_change_config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v1 {
using LocationScaleDecoderConfig=RelativeChangeConfig;
inline constexpr const char *kLocationScaleDecoderModelTag="RPB-v30";
inline constexpr const char *kLocationScaleDecoderArchitectureId="content-protected32-raw-relative-gaussian-location-scale-v1";
inline constexpr const char *kLocationScaleDecoderLayout=
    "historical-raw-f32-scaled-exact-v22-prefix;raw-f64-relative-conditioned-change-suffix;unmodulated-mean-and-canonical-CF-variance";
inline constexpr const char *kLocationScaleDecoderRecipe=
    "cuda-f32-linear-change-CF-variance-plus-learned-bias;anchored-softplus-beta1-threshold20;eps-plus-coefficient-times-u-div-u0;broadcast-H-before-semantic-gather-v1";
inline constexpr const char *kLocationScaleDecoderLikelihood=
    "target-mean-variance-cast-cuda-f64-before-arithmetic;half-squared-error-over-variance-plus-log-variance;omit-log2pi;sanitize-nontarget-first;cell-channel-example-mean;negative-nll-valid-v1";
inline constexpr double kLocationScaleDecoderEpsilon=1e-6;
inline constexpr double kLocationScaleDecoderShift=0.5413232726357509;
inline constexpr double kLocationScaleDecoderSoftplusBeta=1.;
inline constexpr double kLocationScaleDecoderSoftplusThreshold=20.;
inline std::vector<float> location_scale_decoder_constants(){
  const float epsilon=static_cast<float>(kLocationScaleDecoderEpsilon);
  return{epsilon,static_cast<float>(kLocationScaleDecoderShift),1.f-epsilon};
}
inline void validate_location_scale_decoder_config(const LocationScaleDecoderConfig &c){validate_relative_change_config(c);}
inline int64_t location_scale_decoder_native_width(const LocationScaleDecoderConfig &c){return relative_change_native_width(c);}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v1
