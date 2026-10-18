// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v8/config.h"
#include <limits>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v8 {
inline constexpr const char *kTemporalPoolingPolicy="original-position-four-bin-disjoint-export-rows-v1";
inline constexpr int64_t kTemporalPoolingBins=4;
inline void validate_temporal_pooling_config(const Config &c){
  validate_config(c);
  TORCH_CHECK((c.history_length/c.patch_length)%kTemporalPoolingBins==0 &&
      c.export_width%kTemporalPoolingBins==0,
      "[cpb-v8] original patch count and export width must be divisible by four");
  // The served family already requires this path. Do not silently retain an
  // alternative learned-global route that bypasses the new ordered summaries.
  TORCH_CHECK(c.global_bottleneck_mode==2 && c.global_pool_input_source==0,
      "[cpb-v8] four-bin temporal summaries require compact-export global mode2/source0");
}
inline std::vector<int64_t> temporal_pooling_partition(const Config &c){
  validate_temporal_pooling_config(c);const auto K=c.history_length/c.patch_length,D=c.export_width;
  return{kTemporalPoolingBins,K,D,K/kTemporalPoolingBins,D/kTemporalPoolingBins};
}
struct TemporalPoolOutput {torch::Tensor projected,weights,bin_valid;};

// The only pooling kernel used by BOTH actual ordinary/contextual Model paths.
// No state, parameters, random draws or packed-rank binning are introduced.
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
      "[cpb-v8] pooling requires aligned R,T hidden/scores/support/original positions");
  TORCH_CHECK(projection_weight.sizes()==torch::IntArrayRef({D,c.encoder_width})&&
      projection_bias.sizes()==torch::IntArrayRef({D})&&
      projection_weight.device()==hidden.device()&&projection_bias.device()==hidden.device()&&
      projection_weight.scalar_type()==hidden.scalar_type()&&projection_bias.scalar_type()==hidden.scalar_type()&&
      torch::where(valid,positions.ge(0).logical_and(positions.lt(K)),positions.eq(-1)).all().item<bool>()&&
      torch::isfinite(scores.masked_select(valid)).all().item<bool>()&&
      torch::isfinite(hidden.masked_select(valid.unsqueeze(-1).expand_as(hidden))).all().item<bool>(),
      "[cpb-v8] finite supported hidden/scores, exact projection and original index bounds required");
  const auto safe_hidden=torch::where(valid.unsqueeze(-1),hidden,torch::zeros_like(hidden));
  auto weights=torch::zeros_like(scores);std::vector<torch::Tensor> slices,bins;
  for(int64_t j=0;j<kTemporalPoolingBins;++j){
    const auto membership=valid.logical_and(positions.ge(j*(K/4))).logical_and(positions.lt((j+1)*(K/4)));
    const auto present=membership.any(1);
    const auto masked=scores.masked_fill(membership.logical_not(),-std::numeric_limits<double>::infinity());
    // An empty bin must never evaluate all-minus-infinity softmax. Mask both
    // operands BEFORE arithmetic; unsupported NaN values cannot enter a sum.
    const auto finite_fallback=torch::where(present.unsqueeze(1),masked,torch::zeros_like(masked));
    const auto bin_weights=torch::where(membership,torch::softmax(finite_fallback,1),torch::zeros_like(scores));
    const auto pooled=(safe_hidden*bin_weights.unsqueeze(-1)).sum(1);
    const auto projected=torch::matmul(pooled,projection_weight.narrow(0,j*(D/4),D/4).transpose(0,1))+
        projection_bias.narrow(0,j*(D/4),D/4);
    slices.push_back(torch::where(present.unsqueeze(1),projected,torch::zeros_like(projected)));
    bins.push_back(present);weights=weights+bin_weights;
  }
  return{torch::cat(slices,1),weights,torch::stack(bins,1)};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v8
