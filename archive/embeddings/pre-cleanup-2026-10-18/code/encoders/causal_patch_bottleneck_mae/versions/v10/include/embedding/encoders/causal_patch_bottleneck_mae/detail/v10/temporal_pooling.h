// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v10/config.h"
#include <limits>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v10 {
inline constexpr const char *kTemporalPoolingPolicy="original-position-four-bin-shared-global-half-mixture-v1";
inline constexpr const char *kEmptyLocalBinPolicy="zero-whole-export-slice-after-bias-v1";
inline constexpr double kTemporalPoolingLocalMix=.5,kTemporalPoolingGlobalMix=.5;
inline constexpr int64_t kTemporalPoolingBins=4;
inline void validate_temporal_pooling_config(const Config &c){
  validate_config(c);
  TORCH_CHECK((c.history_length/c.patch_length)%kTemporalPoolingBins==0 &&
      c.export_width%kTemporalPoolingBins==0,
      "[cpb-v10] original patch count and export width must be divisible by four");
  // The served family already requires this path. Do not silently retain an
  // alternative learned-global route that bypasses the new ordered summaries.
  TORCH_CHECK(c.global_bottleneck_mode==2 && c.global_pool_input_source==0,
      "[cpb-v10] four-bin temporal summaries require compact-export global mode2/source0");
}
inline std::vector<int64_t> temporal_pooling_partition(const Config &c){
  validate_temporal_pooling_config(c);const auto K=c.history_length/c.patch_length,D=c.export_width;
  return{kTemporalPoolingBins,K,D,K/kTemporalPoolingBins,D/kTemporalPoolingBins};
}
inline std::vector<double> temporal_pooling_mix(){return{kTemporalPoolingLocalMix,kTemporalPoolingGlobalMix};}
struct TemporalPoolOutput {
  torch::Tensor projected,weights,bin_valid;
  torch::Tensor local_weights,global_weights,mixture_weights;
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
      "[cpb-v10] pooling requires aligned R,T hidden/scores/support/original positions");
  TORCH_CHECK(projection_weight.sizes()==torch::IntArrayRef({D,c.encoder_width})&&
      projection_bias.sizes()==torch::IntArrayRef({D})&&
      projection_weight.device()==hidden.device()&&projection_bias.device()==hidden.device()&&
      projection_weight.scalar_type()==hidden.scalar_type()&&projection_bias.scalar_type()==hidden.scalar_type()&&
      torch::where(valid,positions.ge(0).logical_and(positions.lt(K)),positions.eq(-1)).all().item<bool>()&&
      torch::isfinite(scores.masked_select(valid)).all().item<bool>()&&
      torch::isfinite(hidden.masked_select(valid.unsqueeze(-1).expand_as(hidden))).all().item<bool>(),
      "[cpb-v10] finite supported hidden/scores, exact projection and original index bounds required");
  const auto safe_hidden=torch::where(valid.unsqueeze(-1),hidden,torch::zeros_like(hidden));
  const auto global_present=valid.any(1);
  const auto global_masked=scores.masked_fill(valid.logical_not(),-std::numeric_limits<double>::infinity());
  const auto global_fallback=torch::where(global_present.unsqueeze(1),global_masked,torch::zeros_like(global_masked));
  const auto global_weights=torch::where(valid,torch::softmax(global_fallback,1),torch::zeros_like(scores));
  const auto global_pooled=(safe_hidden*global_weights.unsqueeze(-1)).sum(1);
  auto weights=torch::zeros_like(scores),local_weights=torch::zeros_like(scores);
  std::vector<torch::Tensor> slices,bins,mixtures;
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
    // Mix hidden vectors BEFORE the one application of this bin's existing
    // projection rows/bias. An absent LOCAL bin never borrows global support.
    const auto pooled=kTemporalPoolingLocalMix*local_pooled+kTemporalPoolingGlobalMix*global_pooled;
    const auto projected=torch::matmul(pooled,projection_weight.narrow(0,j*(D/4),D/4).transpose(0,1))+
        projection_bias.narrow(0,j*(D/4),D/4);
    slices.push_back(torch::where(present.unsqueeze(1),projected,torch::zeros_like(projected)));
    bins.push_back(present);mixtures.push_back(mixture);
    local_weights=local_weights+bin_weights;weights=weights+mixture;
  }
  // Aggregate weights are a diagnostic SUM, not one pooling probability:
  // projected slices use distinct mixtures and aggregate mass=#nonempty bins.
  return{torch::cat(slices,1),weights,torch::stack(bins,1),local_weights,global_weights,torch::stack(mixtures,1)};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v10
