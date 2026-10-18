// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/learned_ordered_relation_config.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/types.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct LearnedOrderedRelationBankOutput {
  torch::Tensor values;        // CUDA F32 [B,3*N]: odd, even, observed fractions.
  torch::Tensor common_counts; // CUDA I64 [B,N], original common endpoint counts.
  torch::Tensor eligible;      // CUDA Bool [B,N], count and both energy guards.
};

// Four actually observed endpoints share one mask and two endpoint means.
// Missing storage is sanitized before arithmetic; no interpolation, packed-rank
// coordinates, scoring labels or hidden training targets enter this bank.
inline LearnedOrderedRelationBankOutput learned_ordered_relation_bank(
    const Input &input, const Config &c) {
  validate_learned_ordered_relation_config(c);
  validate_input(input, configurable_spectral_input_config(c), true);
  const auto B = input.data.size(0), C = c.channel_count;
  const auto H = c.history_length, F = c.input_width;
  const auto N = learned_ordered_relation_moment_width(c);
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
  std::vector<torch::Tensor> odd, even, fractions, counts, eligibility;
  for (int64_t i = 0; i < C; ++i) {
    for (int64_t j = i + 1; j < C; ++j) {
      std::vector<torch::Tensor> pa, ps, pq, pn, pe;
      for (const auto lag : kLearnedOrderedRelationLags) {
        const auto T = H - lag;
        const auto mi = mask.select(1, i).narrow(1, 0, T).logical_and(
            mask.select(1, i).narrow(1, lag, T));
        const auto mj = mask.select(1, j).narrow(1, 0, T).logical_and(
            mask.select(1, j).narrow(1, lag, T));
        const auto common = mi.unsqueeze(-1).logical_and(mj.unsqueeze(-2));
        const auto n = common.sum(1);
        const auto divisor = n.clamp_min(1).to(torch::kFloat64);
        const auto ai0 = safe.select(1, i).narrow(1, 0, T).unsqueeze(-1);
        const auto ai1 = safe.select(1, i).narrow(1, lag, T).unsqueeze(-1);
        const auto bj0 = safe.select(1, j).narrow(1, 0, T).unsqueeze(-2);
        const auto bj1 = safe.select(1, j).narrow(1, lag, T).unsqueeze(-2);
        const auto a0 = torch::where(common, ai0, torch::zeros_like(ai0));
        const auto a1 = torch::where(common, ai1, torch::zeros_like(ai1));
        const auto b0 = torch::where(common, bj0, torch::zeros_like(bj0));
        const auto b1 = torch::where(common, bj1, torch::zeros_like(bj1));
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
            kLearnedOrderedRelationEnergyFloor * kLearnedOrderedRelationEnergyFloor).sqrt();
        const auto active = n.ge(kLearnedOrderedRelationMinimumCommon)
            .logical_and(ea.gt(kLearnedOrderedRelationEnergyFloor))
            .logical_and(eb.gt(kLearnedOrderedRelationEnergyFloor));
        const auto a = (ac0 * bc1 - ac1 * bc0).sum(1) / divisor / denominator;
        const auto s = (ac0 * bc0 + ac1 * bc1).sum(1) / divisor / denominator;
        pa.push_back(torch::where(active, a, torch::zeros_like(a)));
        ps.push_back(torch::where(active, s, torch::zeros_like(s)));
        pq.push_back(n.to(torch::kFloat64) / double(T));
        pn.push_back(n);
        pe.push_back(active);
      }
      // Canonical pair, feature-i, feature-j, then lag order within each moment.
      odd.push_back(torch::stack(pa, -1).flatten(1));
      even.push_back(torch::stack(ps, -1).flatten(1));
      fractions.push_back(torch::stack(pq, -1).flatten(1));
      counts.push_back(torch::stack(pn, -1).flatten(1));
      eligibility.push_back(torch::stack(pe, -1).flatten(1));
    }
  }
  LearnedOrderedRelationBankOutput out{
      torch::cat({torch::cat(odd, 1), torch::cat(even, 1), torch::cat(fractions, 1)}, 1).to(c.dtype),
      torch::cat(counts, 1), torch::cat(eligibility, 1)};
  TORCH_CHECK(out.values.sizes() == torch::IntArrayRef({B, N * 3}) &&
                  out.common_counts.sizes() == torch::IntArrayRef({B, N}) &&
                  out.eligible.sizes() == out.common_counts.sizes() &&
                  torch::isfinite(out.values).all().item<bool>(),
              "[rpb-v21] finite configured ordered relation bank required");
  return out;
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
