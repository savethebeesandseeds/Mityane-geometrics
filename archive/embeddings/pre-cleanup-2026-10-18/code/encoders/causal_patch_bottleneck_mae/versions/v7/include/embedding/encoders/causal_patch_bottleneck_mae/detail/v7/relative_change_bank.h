// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v7/relative_change_config.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v7/conditioned_change_bank.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v7/configurable_spectral_model.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v7 {
using RelativeChangeTargets=ConditionedChangeTargets;
inline void validate_relative_change_raw(const Input &raw,const Config &c){
  validate_consecutive_change_config(c);validate_input(raw,configurable_spectral_input_config(c));
  TORCH_CHECK(raw.data.scalar_type()==torch::kFloat64,"[rpb-v28] genuine raw F64 observations required");
}
// Preserve historical loader rounding on CPU before the unchanged scaler.
inline Input relative_change_parent_input(const Input &raw,const FrozenScaler &scaler,const Config &c){
  validate_relative_change_raw(raw,c);
  const auto observed=raw.observed.to(torch::kCPU),data=raw.data.to(torch::kCPU,torch::kFloat64);
  const auto safe=torch::where(observed,data,torch::zeros_like(data)).to(torch::kFloat32);
  TORCH_CHECK(torch::isfinite(safe).all().item<bool>(),"[rpb-v28] observed raw F64 overflows historical F32 parent cast");
  return transform_configurable_spectral_input(scaler,{safe,observed,raw.channel_ids,raw.endpoints,raw.sampling_interval},c);
}
struct RelativeChangeBank {
  torch::Tensor values,valid,counts;
  torch::Tensor increments,adjacent_valid; // canonical B,C,H-1,F; rounded F32 and bool
};
inline RelativeChangeBank relative_change_bank(const Input &raw,const Config &c){
  validate_relative_change_raw(raw,c);const auto B=raw.data.size(0),C=c.channel_count,H=c.history_length,F=c.input_width;
  auto data=raw.data.to(c.device,torch::kFloat64);auto mask=raw.observed.to(c.device);
  const auto order=channel_indices(raw.channel_ids,c,B,c.device).argsort(int64_t{1});
  data=data.gather(1,order.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F}));mask=mask.gather(1,order.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F}));
  const auto safe=torch::where(mask,data,torch::zeros_like(data));
  const auto adjacent=mask.narrow(2,1,H-1).logical_and(mask.narrow(2,0,H-1));
  const auto a=safe.narrow(2,1,H-1),b=safe.narrow(2,0,H-1),m=torch::maximum(a.abs(),b.abs());
  const auto safe_m=torch::where(m.gt(0),m,torch::ones_like(m)),aa=a/safe_m,bb=b/safe_m,d=aa.abs()+bb.abs();
  const auto safe_d=torch::where(d.gt(0),d,torch::ones_like(d));
  const auto u64=torch::where(adjacent,2.*(aa-bb)/safe_d,torch::zeros_like(a));
  const auto u=u64.to(torch::kFloat32).to(torch::kFloat64),counts=adjacent.sum(2),divisor=counts.clamp_min(1).to(torch::kFloat64);
  const auto active=counts.ge(kConsecutiveChangeMinimumCount),mean=u.sum(2)/divisor,energy=u.square().sum(2)/divisor;
  const auto rms=energy.clamp_min(kConsecutiveChangeRmsFloor*kConsecutiveChangeRmsFloor).sqrt();
  std::vector<torch::Tensor> moments{torch::where(active,mean,torch::zeros_like(mean)),torch::where(active.logical_and(energy.gt(0)),rms,torch::zeros_like(rms))};
  std::vector<torch::Tensor> validity{active,active};
  for(int64_t q=0;q<4;++q){
    const auto begin=q*(H-1)/4,end=(q+1)*(H-1)/4;
    if(begin==end){moments.push_back(torch::zeros_like(mean));moments.push_back(torch::zeros_like(mean));validity.push_back(torch::zeros_like(active));validity.push_back(torch::zeros_like(active));continue;}
    const auto n=adjacent.narrow(2,begin,end-begin).sum(2),eligible=active.logical_and(n.ge(kConsecutiveChangeMinimumCount));
    const auto part=u.narrow(2,begin,end-begin),denominator=n.clamp_min(1).to(torch::kFloat64),part_mean=part.sum(2)/denominator,part_energy=part.square().sum(2)/denominator;
    const auto part_rms=torch::where(part_energy.gt(0),part_energy.clamp_min(kConsecutiveChangeRmsFloor*kConsecutiveChangeRmsFloor).sqrt(),torch::zeros_like(part_energy));
    moments.push_back(torch::where(eligible,part_mean/rms,torch::zeros_like(mean)));moments.push_back(torch::where(eligible,part_rms/rms,torch::zeros_like(mean)));
    validity.push_back(eligible);validity.push_back(eligible);
  }
  const auto values=torch::stack(moments,-1).flatten(1).to(torch::kFloat32),valid=torch::stack(validity,-1).flatten(1);
  TORCH_CHECK(torch::isfinite(values).all().item<bool>()&&torch::isfinite(u).all().item<bool>(),"[rpb-v28] finite relative increment/moment bank required");
  return{values.detach(),valid.detach(),counts.detach(),u.to(torch::kFloat32).detach(),adjacent.detach()};
}
inline RelativeChangeTargets relative_change_targets(const Input &raw,const Config &c){const auto bank=relative_change_bank(raw,c);return{bank.values,bank.valid};}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v7
