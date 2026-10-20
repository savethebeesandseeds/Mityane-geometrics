// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v15/conditioned_change_config.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v15/consecutive_change_bank.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v15 {
struct ConditionedChangeTargets{torch::Tensor values,valid;};
// Preserve the historical bank arithmetic and derive validity independently
// from actual observed endpoints. A valid constant (zero RMS) stays supported.
inline ConditionedChangeTargets conditioned_change_targets(const Input &visible,const Config &c){
  const auto bank=consecutive_change_bank(visible,c);
  const auto B=visible.data.size(0),C=c.channel_count,H=c.history_length,F=c.input_width;
  auto mask=visible.observed.to(c.device);
  const auto order=channel_indices(visible.channel_ids,c,B,c.device).argsort(int64_t{1});
  mask=mask.gather(1,order.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F}));
  const auto pairs=mask.narrow(2,1,H-1).logical_and(mask.narrow(2,0,H-1));
  const auto global=pairs.sum(2).ge(kConsecutiveChangeMinimumCount);
  std::vector<torch::Tensor> valid{global,global};
  for(int64_t q=0;q<4;++q){
    const auto begin=q*(H-1)/4,end=(q+1)*(H-1)/4;
    const auto part=begin==end?torch::zeros_like(global):
        global.logical_and(pairs.narrow(2,begin,end-begin).sum(2).ge(kConsecutiveChangeMinimumCount));
    valid.push_back(part);valid.push_back(part);
  }
  return{bank.values.detach(),torch::stack(valid,-1).flatten(1).detach()};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v15
