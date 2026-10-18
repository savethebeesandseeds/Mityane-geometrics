// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/temporal_relation_bank.h"
#include <array>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr int64_t kSpectralRelationBins = 8;
inline constexpr int64_t kSpectralRelationLowBins = 3;
inline constexpr int64_t kSpectralRelationMinimumCommonTicks = 4;
inline constexpr double kSpectralRelationBasisTraceFloor = 1e-12;
inline constexpr double kSpectralRelationScaledRankFloor = 1e-8;
inline constexpr double kSpectralRelationDeterminantFloor = 1e-24;
inline constexpr double kSpectralRelationEnergyFloor = 1e-12;
inline constexpr double kSpectralRelationUnitNormFloor = 1e-12;
inline constexpr const char *kSpectralRelationBankLayout =
    "semantic-pairs-01-02-12;feature-i-major-feature-j-major;ReLow-ImLow-ReHigh-ImHigh;"
    "q1-3-low-q4-8-high;fixed-H32;common-visible-time-intersection";

struct SpectralRelationBankOutput {
  torch::Tensor values;        // CUDA F32 [B,108]: 3 pairs x9 feature pairs x4 metrics.
  torch::Tensor common_counts; // CUDA Long [B,27], original shared visible tick counts.
  torch::Tensor eligible;      // CUDA Bool [B,27], same support for all four metrics.
};

// A fixed harmonic bank, not a frequency estimator: evaluate ALL q=1..8 at
// omega=2*pi*q/32, without a row-specific winner, hidden period or task input.
// For each channel/feature pair use its SAME original-time mask intersection.
// Center values AND sine/cosine columns on that mask. The analytic2x2 mask Gram
// correction removes their nonorthogonality; no interpolation fills any gap.
// Every frequency must pass the mask-only rank rule, and both signals must
// have positive eight-bin energy. Unsupported feature pairs are exactzero.
inline SpectralRelationBankOutput spectral_relation_bank(const Input &input, const Config &c) {
  validate_temporal_relation_config(c);
  TORCH_CHECK(c.device.is_cuda(), "[rpb-spectral-relations] actual CUDA required");
  validate_input(input, c, true);
  const auto B = input.data.size(0), H = c.history_length, F = c.input_width;
  auto data = input.data.to(c.device);
  auto mask = input.observed.to(data.device());
  const auto order = channel_indices(input.channel_ids, c, B, data.device()).argsort(int64_t{1});
  data = data.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, 3, H, F}));
  mask = mask.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, 3, H, F}));
  // Sanitization precedes subtraction, products and conversion. All arithmetic
  // below is CUDA double; only the resulting fixed bank is cast to nativeF32.
  const auto safe = torch::where(mask, data, torch::zeros_like(data)).to(torch::kFloat64);
  const auto options = safe.options();
  const auto time = torch::arange(H, options).unsqueeze(-1);
  const auto q = torch::arange(1, kSpectralRelationBins + 1, options).unsqueeze(0);
  const auto angle = time * q * (6.283185307179586476925286766559 / double(H));
  const auto cosine = angle.cos(), sine = angle.sin();
  constexpr std::array<std::array<int64_t, 2>, 3> pairs{{{0, 1}, {0, 2}, {1, 2}}};
  std::vector<torch::Tensor> bank, counts, eligibility;
  for (const auto &pair : pairs) {
    const auto joint = mask.select(1, pair[0]).unsqueeze(-1)
                           .logical_and(mask.select(1, pair[1]).unsqueeze(-2))
                           .flatten(2).transpose(1, 2); // B,9,H.
    const auto n = joint.sum(-1);
    const auto divisor = n.clamp_min(1).to(torch::kFloat64);
    const auto common = joint.to(torch::kFloat64);
    const auto left = safe.select(1, pair[0]).unsqueeze(-1).expand({B, H, F, F})
                          .flatten(2).transpose(1, 2);
    const auto right = safe.select(1, pair[1]).unsqueeze(-2).expand({B, H, F, F})
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
    // Reversal rotates the two basis columns. Trace and determinant are
    // rotation-invariant; a diagonal-product condition would not preserve
    // eligibility, and therefore spectral even/odd parity, near its cutoff.
    const auto rank = n.ge(kSpectralRelationMinimumCommonTicks).unsqueeze(-1)
                          .logical_and(trace.gt(kSpectralRelationBasisTraceFloor))
                          .logical_and(determinant.gt(kSpectralRelationScaledRankFloor * trace.square()));
    const auto inverse_denominator = determinant.clamp_min(kSpectralRelationDeterminantFloor);
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
    const auto active = rank.all(-1).logical_and(ea.squeeze(-1).gt(kSpectralRelationEnergyFloor))
                           .logical_and(eb.squeeze(-1).gt(kSpectralRelationEnergyFloor));
    const auto da = ea.clamp_min(kSpectralRelationEnergyFloor).sqrt();
    const auto db = eb.clamp_min(kSpectralRelationEnergyFloor).sqrt();
    // Z=beta_cos-i*beta_sin. conj(Za)*Zb has positive imaginary part for
    // channelb's positive phase shift, irrespective of a shared base phase.
    const auto real = (azc / da) * (bzc / db) + (azs / da) * (bzs / db);
    const auto imag = (azs / da) * (bzc / db) - (azc / da) * (bzs / db);
    const auto four = torch::stack({real.narrow(-1, 0, kSpectralRelationLowBins).sum(-1),
        imag.narrow(-1, 0, kSpectralRelationLowBins).sum(-1),
        real.narrow(-1, kSpectralRelationLowBins, kSpectralRelationBins-kSpectralRelationLowBins).sum(-1),
        imag.narrow(-1, kSpectralRelationLowBins, kSpectralRelationBins-kSpectralRelationLowBins).sum(-1)}, -1);
    bank.push_back(torch::where(active.unsqueeze(-1), four, torch::zeros_like(four)).flatten(1));
    counts.push_back(n); eligibility.push_back(active);
  }
  SpectralRelationBankOutput out{torch::cat(bank, 1).to(c.dtype), torch::cat(counts, 1), torch::cat(eligibility, 1)};
  TORCH_CHECK(out.values.sizes() == torch::IntArrayRef({B,108}) &&
                  out.common_counts.sizes() == torch::IntArrayRef({B,27}) &&
                  out.eligible.sizes() == torch::IntArrayRef({B,27}) &&
                  torch::isfinite(out.values).all().item<bool>(),
              "[rpb-spectral-relations] finite closed spectral bank required");
  return out;
}

// One joint norm for each pair's four coordinates retains relative band energy.
// Independent band norms would intentionally erase the component-balance cue.
inline torch::Tensor unit_spectral_relations(const torch::Tensor &four12) {
  TORCH_CHECK(four12.defined() && four12.is_cuda() && four12.scalar_type() == torch::kFloat32 &&
                  four12.dim() == 2 && four12.size(0) > 0 && four12.size(1) == 12 &&
                  torch::isfinite(four12).all().item<bool>(),
              "[rpb-spectral-relations] finite CUDA F32 spectral12 required");
  const auto groups = four12.reshape({four12.size(0),3,4});
  const auto norm = groups.square().sum(-1,true).clamp_min(
      kSpectralRelationUnitNormFloor*kSpectralRelationUnitNormFloor).sqrt();
  return (groups/norm).reshape({four12.size(0),12});
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
