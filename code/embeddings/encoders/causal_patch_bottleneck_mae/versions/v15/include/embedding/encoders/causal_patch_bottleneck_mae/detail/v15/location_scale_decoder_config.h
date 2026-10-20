// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v15/relative_change_config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v15 {
using LocationScaleDecoderConfig=RelativeChangeConfig;
inline constexpr const char *kLocationScaleDecoderModelTag="CPB-v15";
inline constexpr const char *kLocationScaleDecoderArchitectureId="fresh-causal-native36-v4-forward-huber1-mean-detached-error-variance-v1";
inline constexpr const char *kLocationScaleDecoderLayout=
    "raw-f64-sanitized-f32-scaled-fresh-learned32;raw-f64-relative-conditioned-change4;live-causal-mean-and-input-gradient-stopped-canonical-CF-variance";
inline constexpr const char *kLocationScaleDecoderRecipe=
    "cuda-f32-linear-change-CF-variance-plus-learned-bias;anchored-softplus-beta1-threshold20;eps-plus-coefficient-times-u-div-u0;broadcast-H-before-semantic-gather-v1";
inline constexpr const char *kLocationScaleDecoderObjectivePolicy="native36-huber1-mean-plus-detached-error-gaussian-variance-v1";
inline constexpr const char *kLocationScaleDecoderGradientPolicy="huber1-mean-live-parent32+mean4;gaussian-variance-detached-error-detached-mean4-input-head-only;shared-global-clip1-v1";
inline constexpr const char *kLocationScaleDecoderDirectGradientRoutes="mean->parent32+mean4;variance->variance-projection+bias-only;variance-input-mean4-detached;shared-global-clip1";
inline constexpr double kLocationScaleDecoderMeanHuberDelta=1.;
inline constexpr double kLocationScaleDecoderMeanWeight=1.;
inline constexpr double kLocationScaleDecoderVarianceWeight=1.;
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
inline void validate_location_scale_decoder_config(const LocationScaleDecoderConfig &c){
  validate_relative_change_config(c);
  TORCH_CHECK(c.level.export_width==32 && c.change_width==4 && c.level.huber_delta==kLocationScaleDecoderMeanHuberDelta,
      "[cpb-v15] fixed V4 native36 architecture and Huber delta1 required");
}
inline int64_t location_scale_decoder_native_width(const LocationScaleDecoderConfig &c){return relative_change_native_width(c);}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v15
