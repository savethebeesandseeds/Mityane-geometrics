// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v4/relative_change_config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v4 {
using LocationScaleDecoderConfig=RelativeChangeConfig;
inline constexpr const char *kLocationScaleDecoderModelTag="CPB-v4";
inline constexpr const char *kLocationScaleDecoderArchitectureId="fresh-causal-learned32-relative4-gaussian-location-scale-variance-input-gradient-stop-v1";
inline constexpr const char *kLocationScaleDecoderLayout=
    "raw-f64-sanitized-f32-scaled-fresh-learned32;raw-f64-relative-conditioned-change4;live-causal-mean-and-input-gradient-stopped-canonical-CF-variance";
inline constexpr const char *kLocationScaleDecoderRecipe=
    "cuda-f32-linear-change-CF-variance-plus-learned-bias;anchored-softplus-beta1-threshold20;eps-plus-coefficient-times-u-div-u0;broadcast-H-before-semantic-gather-v1";
inline constexpr const char *kLocationScaleDecoderGradientPolicy="variance_input_gradient_stop-v1";
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
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v4
