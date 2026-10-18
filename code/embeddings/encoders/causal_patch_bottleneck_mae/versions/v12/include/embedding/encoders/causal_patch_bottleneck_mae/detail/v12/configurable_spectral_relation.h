// SPDX-License-Identifier: MIT
// CPB-v12 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v12/configurable_spectral_config.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v12/types.h"
#include <vector>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v12 {

inline constexpr int64_t kConfigurableSpectralBins = 8;
inline constexpr int64_t kConfigurableSpectralLowBins = 3;
inline constexpr double kConfigurableSpectralUnitNormFloor = 1e-12;

struct ConfigurableSpectralRelationBankOutput {
  torch::Tensor values;        // CUDA F32 [B, pairs * F * F * 4].
  torch::Tensor common_counts; // CUDA Long [B, pairs * F * F].
  torch::Tensor eligible;      // CUDA Bool, same shape as common_counts.
};

// This generalizes v19's arithmetic without selecting a winning frequency or
// interpolating missing ticks. Every feature pair has its own common mask.
// Semantic pairs follow configured channel order, regardless of storage order.
inline ConfigurableSpectralRelationBankOutput configurable_spectral_relation_bank(
    const Input &input, const Config &c) {
  validate_configurable_spectral_config(c);
  validate_input(input, configurable_spectral_input_config(c), true);
  const auto B = input.data.size(0), C = c.channel_count;
  const auto H = c.history_length, F = c.input_width;
  const auto pairs = configurable_spectral_pair_count(c);
  auto data = input.data.to(c.device);
  auto mask = input.observed.to(data.device());
  if (!pairs) {
    return {torch::zeros({B, 0}, data.options()),
            torch::zeros({B, 0}, data.options().dtype(torch::kInt64)),
            torch::zeros({B, 0}, data.options().dtype(torch::kBool))};
  }
  const auto order = channel_indices(input.channel_ids, c, B, data.device()).argsort(int64_t{1});
  data = data.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, C, H, F}));
  mask = mask.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, C, H, F}));
  const auto safe = torch::where(mask, data, torch::zeros_like(data)).to(torch::kFloat64);
  const auto time = torch::arange(H, safe.options()).unsqueeze(-1);
  const auto q = torch::arange(1, kConfigurableSpectralBins + 1, safe.options()).unsqueeze(0);
  const auto angle = time * q * (6.283185307179586476925286766559 / double(H));
  const auto cosine = angle.cos(), sine = angle.sin();
  std::vector<torch::Tensor> bank, counts, eligibility;
  for (int64_t i = 0; i < C; ++i) {
    for (int64_t j = i + 1; j < C; ++j) {
      const auto joint = mask.select(1, i).unsqueeze(-1)
                             .logical_and(mask.select(1, j).unsqueeze(-2))
                             .flatten(2).transpose(1, 2);
      const auto n = joint.sum(-1);
      const auto divisor = n.clamp_min(1).to(torch::kFloat64);
      const auto common = joint.to(torch::kFloat64);
      const auto left = safe.select(1, i).unsqueeze(-1).expand({B, H, F, F})
                            .flatten(2).transpose(1, 2);
      const auto right = safe.select(1, j).unsqueeze(-2).expand({B, H, F, F})
                             .flatten(2).transpose(1, 2);
      const auto a = torch::where(joint, left, torch::zeros_like(left));
      const auto b = torch::where(joint, right, torch::zeros_like(right));
      const auto a_mean = a.sum(-1) / divisor, b_mean = b.sum(-1) / divisor;
      const auto ac = torch::where(joint, a - a_mean.unsqueeze(-1), torch::zeros_like(a));
      const auto bc = torch::where(joint, b - b_mean.unsqueeze(-1), torch::zeros_like(b));
      const auto column_mask = joint.unsqueeze(-1);
      const auto column_divisor = divisor.unsqueeze(-1);
      const auto cos_mean = (common.unsqueeze(-1) * cosine).sum(2) / column_divisor;
      const auto sin_mean = (common.unsqueeze(-1) * sine).sum(2) / column_divisor;
      const auto cc = torch::where(column_mask, cosine - cos_mean.unsqueeze(2),
                                   torch::zeros_like(cos_mean.unsqueeze(2)));
      const auto ss = torch::where(column_mask, sine - sin_mean.unsqueeze(2),
                                   torch::zeros_like(sin_mean.unsqueeze(2)));
      const auto gcc = cc.square().sum(2) / column_divisor;
      const auto gss = ss.square().sum(2) / column_divisor;
      const auto gcs = (cc * ss).sum(2) / column_divisor;
      const auto determinant = gcc * gss - gcs.square();
      const auto trace = gcc + gss;
      const auto rank = n.ge(4).unsqueeze(-1).logical_and(trace.gt(1e-12))
                            .logical_and(determinant.gt(1e-8 * trace.square()));
      const auto inverse_denominator = determinant.clamp_min(1e-24);
      const auto ayc = (ac.unsqueeze(-1) * cc).sum(2) / column_divisor;
      const auto ays = (ac.unsqueeze(-1) * ss).sum(2) / column_divisor;
      const auto byc = (bc.unsqueeze(-1) * cc).sum(2) / column_divisor;
      const auto bys = (bc.unsqueeze(-1) * ss).sum(2) / column_divisor;
      const auto azc = torch::where(rank, (ayc * gss - ays * gcs) / inverse_denominator,
                                    torch::zeros_like(ayc));
      const auto azs = torch::where(rank, (ays * gcc - ayc * gcs) / inverse_denominator,
                                    torch::zeros_like(ays));
      const auto bzc = torch::where(rank, (byc * gss - bys * gcs) / inverse_denominator,
                                    torch::zeros_like(byc));
      const auto bzs = torch::where(rank, (bys * gcc - byc * gcs) / inverse_denominator,
                                    torch::zeros_like(bys));
      const auto ea = (azc.square() + azs.square()).sum(-1, true);
      const auto eb = (bzc.square() + bzs.square()).sum(-1, true);
      const auto active = rank.all(-1).logical_and(ea.squeeze(-1).gt(1e-12))
                             .logical_and(eb.squeeze(-1).gt(1e-12));
      const auto da = ea.clamp_min(1e-12).sqrt(), db = eb.clamp_min(1e-12).sqrt();
      const auto real = (azc / da) * (bzc / db) + (azs / da) * (bzs / db);
      const auto imag = (azs / da) * (bzc / db) - (azc / da) * (bzs / db);
      const auto four = torch::stack({real.narrow(-1, 0, kConfigurableSpectralLowBins).sum(-1),
          imag.narrow(-1, 0, kConfigurableSpectralLowBins).sum(-1),
          real.narrow(-1, kConfigurableSpectralLowBins,
                      kConfigurableSpectralBins-kConfigurableSpectralLowBins).sum(-1),
          imag.narrow(-1, kConfigurableSpectralLowBins,
                      kConfigurableSpectralBins-kConfigurableSpectralLowBins).sum(-1)}, -1);
      bank.push_back(torch::where(active.unsqueeze(-1), four, torch::zeros_like(four)).flatten(1));
      counts.push_back(n);
      eligibility.push_back(active);
    }
  }
  ConfigurableSpectralRelationBankOutput out{
      torch::cat(bank, 1).to(c.dtype), torch::cat(counts, 1), torch::cat(eligibility, 1)};
  TORCH_CHECK(out.values.sizes() == torch::IntArrayRef({B, pairs * F * F * 4}) &&
                  out.common_counts.sizes() == torch::IntArrayRef({B, pairs * F * F}) &&
                  out.eligible.sizes() == out.common_counts.sizes() &&
                  torch::isfinite(out.values).all().item<bool>(),
              "[rpb-v20] finite configured spectral bank required");
  return out;
}

inline torch::Tensor configurable_unit_spectral_relations(const torch::Tensor &values,
                                                         const Config &c) {
  const auto pairs = configurable_spectral_pair_count(c);
  TORCH_CHECK(values.defined() && values.is_cuda() && values.scalar_type() == torch::kFloat32 &&
                  values.dim() == 2 && values.size(0) > 0 && values.size(1) == pairs * 4 &&
                  torch::isfinite(values).all().item<bool>(),
              "[rpb-v20] finite configured CUDA spectral coordinates required");
  if (!pairs) return values;
  const auto groups = values.reshape({values.size(0), pairs, 4});
  const auto norm = groups.square().sum(-1, true).clamp_min(
      kConfigurableSpectralUnitNormFloor*kConfigurableSpectralUnitNormFloor).sqrt();
  return (groups / norm).reshape_as(values);
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v12
