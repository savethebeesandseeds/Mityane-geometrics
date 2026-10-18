// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v14/config.h"
#include <limits>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v14 {
inline constexpr const char *kTemporalPoolingPolicy="original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1";
inline constexpr const char *kTemporalMomentPolicy="componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1";
inline constexpr const char *kTemporalExportLayout="per-trailing-window-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1";
inline constexpr const char *kTemporalDispersionTransformPolicy="sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1";
inline constexpr double kTemporalDispersionRegularizer=0.0009765625; // exactly 2^-10
inline constexpr double kTemporalDispersionRootOffset=0.03125; // exactly 1/32
inline constexpr const char *kEmptyLocalBinPolicy="zero-whole-export-slice-after-bias-v1";
inline constexpr double kTemporalPoolingLocalMix=.5,kTemporalPoolingGlobalMix=.5;
inline constexpr int64_t kTemporalPoolingBins=4;
inline constexpr int64_t kTemporalPoolingOriginalPositionBound=16;
inline std::vector<int64_t> temporal_pooling_trailing_lengths(){return{4,8,12,16};}
inline void validate_temporal_pooling_config(const Config &c){
  validate_config(c);
  TORCH_CHECK(c.export_width==32,"[cpb-v14] fixed four-mean/four-variance row layout requires export width32");
  TORCH_CHECK(c.history_length==128 && c.patch_length==8 &&
      c.history_length/c.patch_length==kTemporalPoolingOriginalPositionBound &&
      c.export_width%(2*kTemporalPoolingBins)==0,
      "[cpb-v14] exact H128/P8 fixed original-position bound16 and export width divisible by eight required");
  // The served family already requires this path. Do not silently retain an
  // alternative learned-global route that bypasses the new ordered summaries.
  TORCH_CHECK(c.global_bottleneck_mode==2 && c.global_pool_input_source==0,
      "[cpb-v14] nested trailing summaries require compact-export global mode2/source0");
}
inline std::vector<int64_t> temporal_pooling_geometry(const Config &c){
  validate_temporal_pooling_config(c);const auto K=c.history_length/c.patch_length,D=c.export_width;
  return{kTemporalPoolingBins,K,D,D/kTemporalPoolingBins};
}
inline std::vector<double> temporal_pooling_mix(){return{kTemporalPoolingLocalMix,kTemporalPoolingGlobalMix};}
inline std::vector<int64_t> temporal_moment_partition(){return{4,4};}
struct TemporalPoolOutput {
  torch::Tensor projected,weights,bin_valid;
  torch::Tensor local_weights,global_weights,mixture_weights;
  // local_weights is exact per-window [R,4,T], not the old disjoint SUM[R,T].
  // Both actual routes save this existing field with its new declared shape.
  torch::Tensor bin_mean,bin_variance,bin_regularized_root,bin_dispersion;
};

// The only pooling kernel used by BOTH actual ordinary/contextual Model paths.
// No state, parameters, random draws or packed-rank membership are introduced.
// Every slice is for the CURRENT retained endpoint, never an earlier window
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
      "[cpb-v14] pooling requires aligned R,T hidden/scores/support/original positions");
  TORCH_CHECK(projection_weight.sizes()==torch::IntArrayRef({D,c.encoder_width})&&
      projection_bias.sizes()==torch::IntArrayRef({D})&&
      projection_weight.device()==hidden.device()&&projection_bias.device()==hidden.device()&&
      projection_weight.scalar_type()==hidden.scalar_type()&&projection_bias.scalar_type()==hidden.scalar_type()&&
      torch::where(valid,positions.ge(0).logical_and(positions.lt(K)),positions.eq(-1)).all().item<bool>()&&
      torch::isfinite(scores.masked_select(valid)).all().item<bool>()&&
      torch::isfinite(hidden.masked_select(valid.unsqueeze(-1).expand_as(hidden))).all().item<bool>(),
      "[cpb-v14] finite supported hidden/scores, exact projection and original index bounds required");
  const auto safe_hidden=torch::where(valid.unsqueeze(-1),hidden,torch::zeros_like(hidden));
  const auto global_present=valid.any(1);
  const auto global_masked=scores.masked_fill(valid.logical_not(),-std::numeric_limits<double>::infinity());
  const auto global_fallback=torch::where(global_present.unsqueeze(1),global_masked,torch::zeros_like(global_masked));
  const auto global_weights=torch::where(valid,torch::softmax(global_fallback,1),torch::zeros_like(scores));
  const auto global_pooled=(safe_hidden*global_weights.unsqueeze(-1)).sum(1);
  auto weights=torch::zeros_like(scores);
  std::vector<torch::Tensor> slices,bins,mixtures,window_locals,means,variances,roots,dispersions;
  const auto trailing_lengths=temporal_pooling_trailing_lengths();
  for(int64_t j=0;j<kTemporalPoolingBins;++j){
    // Fixed edge16 follows the retained left-padded H128/P8 cutoff window.
    // Never infer thresholds from packed length, support count or last key.
    const auto membership=valid.logical_and(positions.ge(kTemporalPoolingOriginalPositionBound-trailing_lengths[j]))
        .logical_and(positions.lt(kTemporalPoolingOriginalPositionBound));
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
    // Preserve the live parent attention/two-sum center and raw variance graph.
    // The new F32 transform is explicitly add -> sqrt -> subtract, never a
    // rationalization, detach, hard clamp or differently rounded expression.
    const auto active=present.unsqueeze(1).logical_and(valid).unsqueeze(-1);
    const auto bin_hidden=torch::where(active,safe_hidden,torch::zeros_like(safe_hidden));
    const auto bin_center=torch::where(active,pooled.unsqueeze(1).expand_as(safe_hidden),torch::zeros_like(safe_hidden));
    const auto centered=bin_hidden-bin_center;
    const auto variance=(mixture.unsqueeze(-1)*centered.square()).sum(1);
    TORCH_CHECK(torch::isfinite(variance).all().item<bool>()&&variance.ge(0).all().item<bool>(),
        "[cpb-v14] finite nonnegative raw variance required before regularized sqrt");
    const auto regularized_root=torch::sqrt(variance+kTemporalDispersionRegularizer);
    const auto dispersion=regularized_root-kTemporalDispersionRootOffset;
    TORCH_CHECK(torch::isfinite(regularized_root).all().item<bool>()&&torch::isfinite(dispersion).all().item<bool>(),
        "[cpb-v14] finite root and dispersion required before export");
    const auto half=D/8,first=j*(D/4);
    const auto mean_projected=torch::matmul(pooled,projection_weight.narrow(0,first,half).transpose(0,1))+
        projection_bias.narrow(0,first,half);
    const auto variance_projected=torch::matmul(dispersion,projection_weight.narrow(0,first+half,half).transpose(0,1))+
        projection_bias.narrow(0,first+half,half);
    const auto projected=torch::cat({mean_projected,variance_projected},1);
    slices.push_back(torch::where(present.unsqueeze(1),projected,torch::zeros_like(projected)));
    means.push_back(torch::where(present.unsqueeze(1),pooled,torch::zeros_like(pooled)));
    variances.push_back(torch::where(present.unsqueeze(1),variance,torch::zeros_like(variance)));
    // Raw root is intentionally UNMASKED: absent-bin v=0 has root=1/32.
    // Dispersion is zero there, and the served WHOLE8 slice is zero after bias.
    roots.push_back(regularized_root);dispersions.push_back(dispersion);
    bins.push_back(present);mixtures.push_back(mixture);window_locals.push_back(bin_weights);
    weights=weights+mixture;
  }
  // Aggregate weights are a diagnostic SUM, not one pooling probability:
  // projected slices use distinct mixtures and aggregate mass=#nonempty bins.
  return{torch::cat(slices,1),weights,torch::stack(bins,1),torch::stack(window_locals,1),global_weights,torch::stack(mixtures,1),
      torch::stack(means,1),torch::stack(variances,1),torch::stack(roots,1),torch::stack(dispersions,1)};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v14
