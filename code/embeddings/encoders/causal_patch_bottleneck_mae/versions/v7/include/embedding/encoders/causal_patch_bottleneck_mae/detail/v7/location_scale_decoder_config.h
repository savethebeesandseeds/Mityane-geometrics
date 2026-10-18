// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v7/relative_change_config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v7 {
struct LocationScaleDecoderConfig:RelativeChangeConfig {
  int64_t uncertainty_width{4};
  LocationScaleDecoderConfig()=default;
  LocationScaleDecoderConfig(const Config &level,int64_t mean_width=4,int64_t uncertainty=4):
      RelativeChangeConfig{level,mean_width},uncertainty_width(uncertainty){}
};
inline constexpr const char *kLocationScaleDecoderModelTag="CPB-v7";
inline constexpr const char *kLocationScaleDecoderArchitectureId="fresh-causal-parent32-mean4-uncertainty4-huber-mean-detached-error-gaussian-variance-v1";
inline constexpr const char *kLocationScaleDecoderLayout=
    "parent32[0:32];mean4[32:36];uncertainty4[36:40];independent-frozen-bank270-projections;mean-reads-parent32-and-mean4;variance-reads-live-uncertainty4-only";
inline constexpr const char *kLocationScaleDecoderRecipe=
    "cuda-f32-linear-independent-uncertainty-CF-variance-plus-learned-bias;anchored-softplus-beta1-threshold20;eps-plus-coefficient-times-u-div-u0;broadcast-H-before-semantic-gather-v1";
inline constexpr const char *kLocationScaleDecoderObjectivePolicy="huber1-mean-plus-detached-error-gaussian-variance-v1";
inline constexpr const char *kLocationScaleDecoderGradientPolicy="huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;disjoint-mean-uncertainty-clip1-v1";
inline constexpr const char *kLocationScaleDecoderDirectGradientRoutes="mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;disjoint-mean-uncertainty-clip1";
inline constexpr const char *kLocationScaleDecoderClipPolicy="disjoint-mean-uncertainty-norm2-clip1-v1";
inline constexpr double kLocationScaleDecoderClipNormType=2.;
inline constexpr double kLocationScaleDecoderMeanClipNorm=1.;
inline constexpr double kLocationScaleDecoderUncertaintyClipNorm=1.;
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
  validate_relative_change_config(c);const auto limit=std::numeric_limits<int64_t>::max();
  TORCH_CHECK(c.level.export_width==32 && c.change_width==4 && c.uncertainty_width==4 && c.level.huber_delta==kLocationScaleDecoderMeanHuberDelta,
      "[cpb-v7] explicit parent32/mean4/uncertainty4 native40 layout required");
  TORCH_CHECK(c.uncertainty_width>0 && relative_change_native_width(c)<=limit-c.uncertainty_width &&
      c.uncertainty_width<=limit/consecutive_change_bank_width(c.level),"[cpb-v7] positive safe independent uncertainty width required");
}
inline int64_t location_scale_decoder_native_width(const LocationScaleDecoderConfig &c){
  validate_location_scale_decoder_config(c);return relative_change_native_width(c)+c.uncertainty_width;
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v7
