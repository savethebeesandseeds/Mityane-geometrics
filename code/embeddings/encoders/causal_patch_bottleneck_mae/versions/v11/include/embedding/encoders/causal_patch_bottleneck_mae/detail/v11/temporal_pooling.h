// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v11/config.h"
#include <limits>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v11 {
inline constexpr const char *kTemporalPoolingPolicy="original-position-four-bin-half-mixture-mean4-centered-variance4-v1";
inline constexpr const char *kTemporalMomentPolicy="componentwise-centered-variance-two-sum-mean-live-attention-v1";
inline constexpr const char *kTemporalExportLayout="per-original-bin-eight-rows:first4-mean;last4-centered-variance-v1";
inline constexpr const char *kEmptyLocalBinPolicy="zero-whole-export-slice-after-bias-v1";
inline constexpr double kTemporalPoolingLocalMix=.5,kTemporalPoolingGlobalMix=.5;
inline constexpr int64_t kTemporalPoolingBins=4;
inline void validate_temporal_pooling_config(const Config &c){
  validate_config(c);
  TORCH_CHECK(c.export_width==32,"[cpb-v11] fixed four-mean/four-variance row layout requires export width32");
  TORCH_CHECK((c.history_length/c.patch_length)%kTemporalPoolingBins==0 &&
      c.export_width%(2*kTemporalPoolingBins)==0,
      "[cpb-v11] original patch count divisible by four and export width divisible by eight required");
  // The served family already requires this path. Do not silently retain an
  // alternative learned-global route that bypasses the new ordered summaries.
  TORCH_CHECK(c.global_bottleneck_mode==2 && c.global_pool_input_source==0,
      "[cpb-v11] four-bin temporal summaries require compact-export global mode2/source0");
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
};

// The only pooling kernel used by BOTH actual ordinary/contextual Model paths.
// No state, parameters, random draws or packed-rank binning are introduced.
// Every slice is for the CURRENT retained endpoint, never the bin's earlier
// endpoint. All valid states already passed cutoff cropping/causal attention;
// the shared global branch cannot admit a timestamp after the requested cutoff.
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
      "[cpb-v11] pooling requires aligned R,T hidden/scores/support/original positions");
  TORCH_CHECK(projection_weight.sizes()==torch::IntArrayRef({D,c.encoder_width})&&
      projection_bias.sizes()==torch::IntArrayRef({D})&&
      projection_weight.device()==hidden.device()&&projection_bias.device()==hidden.device()&&
      projection_weight.scalar_type()==hidden.scalar_type()&&projection_bias.scalar_type()==hidden.scalar_type()&&
      torch::where(valid,positions.ge(0).logical_and(positions.lt(K)),positions.eq(-1)).all().item<bool>()&&
      torch::isfinite(scores.masked_select(valid)).all().item<bool>()&&
      torch::isfinite(hidden.masked_select(valid.unsqueeze(-1).expand_as(hidden))).all().item<bool>(),
      "[cpb-v11] finite supported hidden/scores, exact projection and original index bounds required");
  const auto safe_hidden=torch::where(valid.unsqueeze(-1),hidden,torch::zeros_like(hidden));
  const auto global_present=valid.any(1);
  const auto global_masked=scores.masked_fill(valid.logical_not(),-std::numeric_limits<double>::infinity());
  const auto global_fallback=torch::where(global_present.unsqueeze(1),global_masked,torch::zeros_like(global_masked));
  const auto global_weights=torch::where(valid,torch::softmax(global_fallback,1),torch::zeros_like(scores));
  const auto global_pooled=(safe_hidden*global_weights.unsqueeze(-1)).sum(1);
  auto weights=torch::zeros_like(scores),local_weights=torch::zeros_like(scores);
  std::vector<torch::Tensor> slices,bins,mixtures,means,variances;
  for(int64_t j=0;j<kTemporalPoolingBins;++j){
    const auto membership=valid.logical_and(positions.ge(j*(K/4))).logical_and(positions.lt((j+1)*(K/4)));
    const auto present=membership.any(1);
    const auto masked=scores.masked_fill(membership.logical_not(),-std::numeric_limits<double>::infinity());
    // An empty bin must never evaluate all-minus-infinity softmax. Mask both
    // operands BEFORE arithmetic; unsupported NaN values cannot enter a sum.
    const auto finite_fallback=torch::where(present.unsqueeze(1),masked,torch::zeros_like(masked));
    const auto bin_weights=torch::where(membership,torch::softmax(finite_fallback,1),torch::zeros_like(scores));
    const auto local_pooled=(safe_hidden*bin_weights.unsqueeze(-1)).sum(1);
    const auto mixture=torch::where(present.unsqueeze(1),
        kTemporalPoolingLocalMix*bin_weights+kTemporalPoolingGlobalMix*global_weights,torch::zeros_like(scores));
    // Preserve the exact parent two-sum mean expression. All new learned
    // numeric values are freshly constructed; only parameter structure is shared.
    const auto pooled=kTemporalPoolingLocalMix*local_pooled+kTemporalPoolingGlobalMix*global_pooled;
    // Sanitize hidden BEFORE subtraction and square. No detached attention/mean,
    // square root, Bessel correction, cancellation-prone raw moments or epsilon.
    const auto centered=safe_hidden-pooled.unsqueeze(1);
    const auto variance=(mixture.unsqueeze(-1)*centered.square()).sum(1);
    const auto half=D/8,first=j*(D/4);
    const auto mean_projected=torch::matmul(pooled,projection_weight.narrow(0,first,half).transpose(0,1))+
        projection_bias.narrow(0,first,half);
    const auto variance_projected=torch::matmul(variance,projection_weight.narrow(0,first+half,half).transpose(0,1))+
        projection_bias.narrow(0,first+half,half);
    const auto projected=torch::cat({mean_projected,variance_projected},1);
    slices.push_back(torch::where(present.unsqueeze(1),projected,torch::zeros_like(projected)));
    means.push_back(torch::where(present.unsqueeze(1),pooled,torch::zeros_like(pooled)));
    variances.push_back(torch::where(present.unsqueeze(1),variance,torch::zeros_like(variance)));
    bins.push_back(present);mixtures.push_back(mixture);
    local_weights=local_weights+bin_weights;weights=weights+mixture;
  }
  // Aggregate weights are a diagnostic SUM, not one pooling probability:
  // projected slices use distinct mixtures and aggregate mass=#nonempty bins.
  return{torch::cat(slices,1),weights,torch::stack(bins,1),local_weights,global_weights,torch::stack(mixtures,1),
      torch::stack(means,1),torch::stack(variances,1)};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v11
