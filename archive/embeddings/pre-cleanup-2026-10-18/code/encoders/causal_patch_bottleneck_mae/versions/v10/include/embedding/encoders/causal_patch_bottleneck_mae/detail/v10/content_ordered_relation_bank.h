// SPDX-License-Identifier: MIT
// CPB-v10 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v10/content_ordered_relation_config.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v10/types.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v10 {

struct ContentOrderedRelationBankOutput {
  torch::Tensor values;        // CUDA F32 [B,2*N]: odd/even content; no support inputs.
  torch::Tensor common_counts; // CUDA I64 [B,N], original common endpoint counts.
  torch::Tensor eligible;      // CUDA Bool [B,N], count and both energy guards.
};

// Four actually observed endpoints share one mask and two endpoint means.
// Counts and energy guard the same moments as v21; support never becomes a
// projected input. Missing storage is sanitized before arithmetic; no interpolation, packed-rank
// coordinates, scoring labels or hidden training targets enter this bank.
inline ContentOrderedRelationBankOutput content_ordered_relation_bank(
    const Input &input, const Config &c) {
  validate_content_ordered_relation_config(c);
  validate_input(input, configurable_spectral_input_config(c), true);
  const auto B = input.data.size(0), C = c.channel_count;
  const auto H = c.history_length, F = c.input_width;
  const auto N = content_ordered_relation_moment_width(c);
  auto data = input.data.to(c.device);
  auto mask = input.observed.to(c.device);
  if (!N) {
    return {torch::zeros({B, 0}, data.options()),
            torch::zeros({B, 0}, data.options().dtype(torch::kInt64)),
            torch::zeros({B, 0}, data.options().dtype(torch::kBool))};
  }
  const auto order = channel_indices(input.channel_ids, c, B, c.device).argsort(int64_t{1});
  data = data.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, C, H, F}));
  mask = mask.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, C, H, F}));
  const auto safe = torch::where(mask, data, torch::zeros_like(data)).to(torch::kFloat64);
  std::vector<torch::Tensor> odd, even, counts, eligibility;
  for (int64_t i = 0; i < C; ++i) {
    for (int64_t j = i + 1; j < C; ++j) {
      std::vector<torch::Tensor> pa, ps, pn, pe;
      for (const auto lag : kContentOrderedRelationLags) {
        // The reduction visits current endpoints s=lag..H-1; each pair is (s-lag,s).
        const auto T = H - lag;
        const auto mi = mask.select(1, i).narrow(1, 0, T).logical_and(
            mask.select(1, i).narrow(1, lag, T));
        const auto mj = mask.select(1, j).narrow(1, 0, T).logical_and(
            mask.select(1, j).narrow(1, lag, T));
        const auto common = mi.unsqueeze(-1).logical_and(mj.unsqueeze(-2));
        const auto n = common.sum(1);
        const auto divisor = n.clamp_min(1).to(torch::kFloat64);
        const auto ai_past = safe.select(1, i).narrow(1, 0, T).unsqueeze(-1);
        const auto ai_now = safe.select(1, i).narrow(1, lag, T).unsqueeze(-1);
        const auto bj_past = safe.select(1, j).narrow(1, 0, T).unsqueeze(-2);
        const auto bj_now = safe.select(1, j).narrow(1, lag, T).unsqueeze(-2);
        const auto a0 = torch::where(common, ai_past, torch::zeros_like(ai_past));
        const auto a1 = torch::where(common, ai_now, torch::zeros_like(ai_now));
        const auto b0 = torch::where(common, bj_past, torch::zeros_like(bj_past));
        const auto b1 = torch::where(common, bj_now, torch::zeros_like(bj_now));
        const auto am = (a0 + a1).sum(1) / (divisor * 2.);
        const auto bm = (b0 + b1).sum(1) / (divisor * 2.);
        const auto ac0 = torch::where(common, a0 - am.unsqueeze(1), torch::zeros_like(a0));
        const auto ac1 = torch::where(common, a1 - am.unsqueeze(1), torch::zeros_like(a1));
        const auto bc0 = torch::where(common, b0 - bm.unsqueeze(1), torch::zeros_like(b0));
        const auto bc1 = torch::where(common, b1 - bm.unsqueeze(1), torch::zeros_like(b1));
        const auto ea = (ac0.square() + ac1.square()).sum(1) / divisor;
        const auto eb = (bc0.square() + bc1.square()).sum(1) / divisor;
        // Clamp before sqrt, including the inactive branch, to keep derivatives finite.
        const auto denominator = (ea * eb).clamp_min(
            kContentOrderedRelationEnergyFloor * kContentOrderedRelationEnergyFloor).sqrt();
        const auto active = n.ge(kContentOrderedRelationMinimumCommon)
            .logical_and(ea.gt(kContentOrderedRelationEnergyFloor))
            .logical_and(eb.gt(kContentOrderedRelationEnergyFloor));
        const auto a = (ac0 * bc1 - ac1 * bc0).sum(1) / divisor / denominator;
        const auto s = (ac0 * bc0 + ac1 * bc1).sum(1) / divisor / denominator;
        pa.push_back(torch::where(active, a, torch::zeros_like(a)));
        ps.push_back(torch::where(active, s, torch::zeros_like(s)));
        pn.push_back(n);
        pe.push_back(active);
      }
      // Canonical pair, feature-i, feature-j, then lag order within each moment.
      odd.push_back(torch::stack(pa, -1).flatten(1));
      even.push_back(torch::stack(ps, -1).flatten(1));
      counts.push_back(torch::stack(pn, -1).flatten(1));
      eligibility.push_back(torch::stack(pe, -1).flatten(1));
    }
  }
  ContentOrderedRelationBankOutput out{
      torch::cat({torch::cat(odd, 1), torch::cat(even, 1)}, 1).to(c.dtype),
      torch::cat(counts, 1), torch::cat(eligibility, 1)};
  TORCH_CHECK(out.values.sizes() == torch::IntArrayRef({B, N * 2}) &&
                  out.common_counts.sizes() == torch::IntArrayRef({B, N}) &&
                  out.eligible.sizes() == out.common_counts.sizes() &&
                  torch::isfinite(out.values).all().item<bool>(),
              "[rpb-v22] finite configured ordered relation bank required");
  return out;
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v10
