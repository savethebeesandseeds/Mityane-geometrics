// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v12/config.h"
#include <limits>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v12 {
inline constexpr const char *kTemporalPoolingPolicy="original-position-four-bin-attention-mean4-uniform-centered-variance4-v1";
inline constexpr const char *kTemporalMomentPolicy="componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1";
inline constexpr const char *kTemporalExportLayout="per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1";
inline constexpr const char *kEmptyLocalBinPolicy="zero-whole-export-slice-after-bias-v1";
inline constexpr double kTemporalPoolingLocalMix=.5,kTemporalPoolingGlobalMix=.5;
inline constexpr int64_t kTemporalPoolingBins=4;
inline void validate_temporal_pooling_config(const Config &c){
  validate_config(c);
  TORCH_CHECK(c.export_width==32,"[cpb-v12] fixed four-mean/four-variance row layout requires export width32");
  TORCH_CHECK((c.history_length/c.patch_length)%kTemporalPoolingBins==0 &&
      c.export_width%(2*kTemporalPoolingBins)==0,
      "[cpb-v12] original patch count divisible by four and export width divisible by eight required");
  TORCH_CHECK(c.global_bottleneck_mode==2 && c.global_pool_input_source==0,
      "[cpb-v12] four-bin temporal summaries require compact-export global mode2/source0");
}
inline std::vector<int64_t> temporal_pooling_partition(const Config &c){
  validate_temporal_pooling_config(c);const auto K=c.history_length/c.patch_length,D=c.export_width;
  return{kTemporalPoolingBins,K,D,K/kTemporalPoolingBins,D/kTemporalPoolingBins};
}
inline std::vector<double> temporal_pooling_mix(){return{kTemporalPoolingLocalMix,kTemporalPoolingGlobalMix};}
inline std::vector<int64_t> temporal_moment_partition(){return{4,4};}
struct TemporalPoolOutput {
  torch::Tensor projected,weights,bin_valid;
  torch::Tensor local_weights,global_weights,mixture_weights;
  torch::Tensor bin_mean,bin_variance;
  torch::Tensor dispersion_weights,dispersion_mean;
};

// One owned kernel for BOTH actual ordinary/contextual paths. Learned attention
// means keep the parent two-sum expression. Only dispersion uses mask-only
// half-local/half-global uniform mass and its OWN two-sum center. Every slice
// describes the current cutoff endpoint, never an earlier bin endpoint.
inline TemporalPoolOutput pool_original_position_bins(const torch::Tensor &hidden,
    const torch::Tensor &scores,const torch::Tensor &valid,const torch::Tensor &positions,
    const torch::Tensor &projection_weight,const torch::Tensor &projection_bias,const Config &c){
  validate_temporal_pooling_config(c);const auto K=c.history_length/c.patch_length,D=c.export_width;
  TORCH_CHECK(hidden.defined()&&hidden.dim()==3&&hidden.size(2)==c.encoder_width&&hidden.size(1)>0&&
      hidden.device()==c.device&&hidden.scalar_type()==c.dtype&&
      scores.sizes()==hidden.sizes().slice(0,2)&&scores.scalar_type()==hidden.scalar_type()&&
      valid.scalar_type()==torch::kBool&&valid.sizes()==scores.sizes()&&
      positions.scalar_type()==torch::kInt64&&positions.sizes()==scores.sizes()&&
      scores.device()==hidden.device()&&valid.device()==hidden.device()&&positions.device()==hidden.device(),
      "[cpb-v12] pooling requires aligned R,T hidden/scores/support/original positions");
  TORCH_CHECK(projection_weight.sizes()==torch::IntArrayRef({D,c.encoder_width})&&
      projection_bias.sizes()==torch::IntArrayRef({D})&&
      projection_weight.device()==hidden.device()&&projection_bias.device()==hidden.device()&&
      projection_weight.scalar_type()==hidden.scalar_type()&&projection_bias.scalar_type()==hidden.scalar_type()&&
      torch::where(valid,positions.ge(0).logical_and(positions.lt(K)),positions.eq(-1)).all().item<bool>()&&
      torch::isfinite(scores.masked_select(valid)).all().item<bool>()&&
      torch::isfinite(hidden.masked_select(valid.unsqueeze(-1).expand_as(hidden))).all().item<bool>(),
      "[cpb-v12] finite supported hidden/scores, exact projection and original index bounds required");
  const auto safe_hidden=torch::where(valid.unsqueeze(-1),hidden,torch::zeros_like(hidden));
  const auto global_present=valid.any(1);
  const auto global_masked=scores.masked_fill(valid.logical_not(),-std::numeric_limits<double>::infinity());
  const auto global_fallback=torch::where(global_present.unsqueeze(1),global_masked,torch::zeros_like(global_masked));
  const auto global_weights=torch::where(valid,torch::softmax(global_fallback,1),torch::zeros_like(scores));
  const auto global_pooled=(safe_hidden*global_weights.unsqueeze(-1)).sum(1);
  // Counts depend only on retained support. clamp_min(1) is an empty-count
  // denominator guard, not an epsilon or a modification of nonempty counts.
  const auto global_count=valid.sum(1).clamp_min(1).to(hidden.scalar_type());
  const auto global_uniform=valid.to(hidden.scalar_type())/global_count.unsqueeze(1);
  const auto global_uniform_pooled=(safe_hidden*global_uniform.unsqueeze(-1)).sum(1);
  auto weights=torch::zeros_like(scores),local_weights=torch::zeros_like(scores);
  std::vector<torch::Tensor> slices,bins,mixtures,means,variances,dispersion_mixtures,dispersion_means;
  for(int64_t j=0;j<kTemporalPoolingBins;++j){
    const auto membership=valid.logical_and(positions.ge(j*(K/4))).logical_and(positions.lt((j+1)*(K/4)));
    const auto present=membership.any(1);
    const auto masked=scores.masked_fill(membership.logical_not(),-std::numeric_limits<double>::infinity());
    const auto finite_fallback=torch::where(present.unsqueeze(1),masked,torch::zeros_like(masked));
    const auto bin_weights=torch::where(membership,torch::softmax(finite_fallback,1),torch::zeros_like(scores));
    const auto local_pooled=(safe_hidden*bin_weights.unsqueeze(-1)).sum(1);
    const auto mixture=torch::where(present.unsqueeze(1),
        kTemporalPoolingLocalMix*bin_weights+kTemporalPoolingGlobalMix*global_weights,torch::zeros_like(scores));
    // Exact unchanged attention mean expression, with its own score derivatives.
    const auto pooled=kTemporalPoolingLocalMix*local_pooled+kTemporalPoolingGlobalMix*global_pooled;
    const auto local_count=membership.sum(1).clamp_min(1).to(hidden.scalar_type());
    const auto local_uniform=membership.to(hidden.scalar_type())/local_count.unsqueeze(1);
    const auto local_uniform_pooled=(safe_hidden*local_uniform.unsqueeze(-1)).sum(1);
    const auto dispersion_mixture=torch::where(present.unsqueeze(1),
        kTemporalPoolingLocalMix*local_uniform+kTemporalPoolingGlobalMix*global_uniform,torch::zeros_like(scores));
    const auto uniform_pooled=kTemporalPoolingLocalMix*local_uniform_pooled+
        kTemporalPoolingGlobalMix*global_uniform_pooled;
    // Sanitize before subtraction/square; hidden and its own uniform center are
    // live. No attention-mean center, detach, Bessel correction, sqrt or epsilon.
    const auto centered=safe_hidden-uniform_pooled.unsqueeze(1);
    const auto variance=(dispersion_mixture.unsqueeze(-1)*centered.square()).sum(1);
    const auto half=D/8,first=j*(D/4);
    const auto mean_projected=torch::matmul(pooled,projection_weight.narrow(0,first,half).transpose(0,1))+
        projection_bias.narrow(0,first,half);
    const auto variance_projected=torch::matmul(variance,projection_weight.narrow(0,first+half,half).transpose(0,1))+
        projection_bias.narrow(0,first+half,half);
    const auto projected=torch::cat({mean_projected,variance_projected},1);
    slices.push_back(torch::where(present.unsqueeze(1),projected,torch::zeros_like(projected)));
    means.push_back(torch::where(present.unsqueeze(1),pooled,torch::zeros_like(pooled)));
    variances.push_back(torch::where(present.unsqueeze(1),variance,torch::zeros_like(variance)));
    dispersion_mixtures.push_back(dispersion_mixture);
    dispersion_means.push_back(torch::where(present.unsqueeze(1),uniform_pooled,torch::zeros_like(uniform_pooled)));
    bins.push_back(present);mixtures.push_back(mixture);
    local_weights=local_weights+bin_weights;weights=weights+mixture;
  }
  // The existing aggregate remains a SUM of attention mixtures, not dispersion
  // mass. Each nonempty dispersion slice separately carries one unit of mass.
  return{torch::cat(slices,1),weights,torch::stack(bins,1),local_weights,global_weights,torch::stack(mixtures,1),
      torch::stack(means,1),torch::stack(variances,1),torch::stack(dispersion_mixtures,1),torch::stack(dispersion_means,1)};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v12
