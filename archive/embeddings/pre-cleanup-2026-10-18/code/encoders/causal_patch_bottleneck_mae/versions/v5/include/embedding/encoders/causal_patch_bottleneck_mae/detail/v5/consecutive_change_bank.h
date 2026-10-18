// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v5/consecutive_change_config.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v5/types.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v5 {

struct ConsecutiveChangeBank {
  torch::Tensor values; // [B,C*F*10], model-device F32; counts never projected.
  torch::Tensor counts; // [B,C,F], actually observed adjacent pairs.
};

inline ConsecutiveChangeBank consecutive_change_bank(const Input &input, const Config &c) {
  validate_consecutive_change_config(c);
  validate_input(input, configurable_spectral_input_config(c), true);
  const auto B = input.data.size(0), C = c.channel_count, H = c.history_length;
  const auto F = c.input_width;
  auto data = input.data.to(c.device);
  auto mask = input.observed.to(c.device);
  const auto order = channel_indices(input.channel_ids, c, B, c.device).argsort(int64_t{1});
  data = data.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F}));
  mask = mask.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F}));
  // Sanitize storage before subtraction. Never connect observations across a gap.
  const auto safe = torch::where(mask,data,torch::zeros_like(data)).to(torch::kFloat64);
  const auto visible = mask.narrow(2,1,H-1).logical_and(mask.narrow(2,0,H-1));
  const auto delta = torch::where(visible,
      safe.narrow(2,1,H-1)-safe.narrow(2,0,H-1), torch::zeros_like(safe.narrow(2,1,H-1)));
  const auto counts = visible.sum(2);
  const auto divisor = counts.clamp_min(1).to(torch::kFloat64);
  const auto active = counts.ge(kConsecutiveChangeMinimumCount);
  const auto mean = delta.sum(2)/divisor;
  // Clamp before sqrt to keep even the inactive derivative finite at zero energy.
  const auto rms = (delta.square().sum(2)/divisor).clamp_min(
      kConsecutiveChangeRmsFloor*kConsecutiveChangeRmsFloor).sqrt();
  const auto actual_rms = torch::where(delta.square().sum(2).gt(0),rms,torch::zeros_like(rms));
  std::vector<torch::Tensor> moments{
      torch::where(active,mean,torch::zeros_like(mean)),
      torch::where(active,actual_rms,torch::zeros_like(rms))};
  for (int64_t q=0;q<4;++q) {
    // Partition by original pair endpoint time, independent of missingness.
    const auto begin=q*(H-1)/4, end=(q+1)*(H-1)/4;
    if (begin==end) {
      moments.push_back(torch::zeros_like(mean)); moments.push_back(torch::zeros_like(mean));
      continue;
    }
    const auto support=visible.narrow(2,begin,end-begin);
    const auto part=delta.narrow(2,begin,end-begin);
    const auto n=support.sum(2);
    const auto eligible=active.logical_and(n.ge(kConsecutiveChangeMinimumCount));
    const auto d=n.clamp_min(1).to(torch::kFloat64);
    const auto part_mean=part.sum(2)/d;
    const auto energy=part.square().sum(2)/d;
    const auto part_rms=torch::where(energy.gt(0),energy.clamp_min(
        kConsecutiveChangeRmsFloor*kConsecutiveChangeRmsFloor).sqrt(),torch::zeros_like(energy));
    moments.push_back(torch::where(eligible,part_mean/rms,torch::zeros_like(mean)));
    moments.push_back(torch::where(eligible,part_rms/rms,torch::zeros_like(mean)));
  }
  const auto values=torch::stack(moments,-1).flatten(1).to(c.dtype);
  TORCH_CHECK(values.sizes()==torch::IntArrayRef({B,consecutive_change_bank_width(c)}) &&
      torch::isfinite(values).all().item<bool>(), "[rpb-v23] finite change moments required");
  return {values,counts};
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v5
