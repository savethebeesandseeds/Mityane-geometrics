// SPDX-License-Identifier: MIT
#pragma once

#include <torch/torch.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::evaluation::two_component {

enum class InformationTask { SlowLagSign, ComponentBalance };

inline constexpr const char *kObservedInformationRule = "observed-two-band-affine-fit-v1";
inline constexpr double kInformationSlowFrequencyMin = 1.0 / 20.0;
inline constexpr double kInformationSlowFrequencyMax = 1.0 / 12.0;
inline constexpr double kInformationFastFrequencyMin = 1.0 / 7.0;
inline constexpr double kInformationFastFrequencyMax = 1.0 / 4.5;
inline constexpr int kInformationSlowGridIntervals = 16;
inline constexpr int kInformationFastGridIntervals = 24;
inline constexpr int kInformationRefinementRounds = 10;
inline constexpr int kInformationMaximumFrequencyPairs =
    (kInformationSlowGridIntervals + 1) * (kInformationFastGridIntervals + 1) +
    9 * kInformationRefinementRounds;
inline constexpr int kInformationMinimumFeatureTicks = 10;
inline constexpr int kInformationMinimumFeaturesPerChannel = 2;
inline constexpr int kInformationMinimumChannelUnionTicks = 16;
inline constexpr int kInformationMinimumChannelSpan = 24; // last minus first tick.
inline constexpr double kInformationScaledPivotFloor = 1e-8;
inline constexpr double kInformationEnergyFloor = 1e-12;
inline constexpr double kInformationMinimumComponentEnergyFraction = .05;
inline constexpr double kInformationMinimumPhaseCoherence = .95;
inline constexpr double kInformationMaximumNormalizedResidual = .0025;
inline constexpr double kInformationMinimumSlowPhaseMargin = .08; // radians.
inline constexpr double kInformationMaximumSlowPhaseMargin = .8;
inline constexpr double kInformationMinimumLogRatioMargin = .35;
inline constexpr double kInformationMaximumLogRatioMargin = 1.0;
inline constexpr double kInformationMaximumLogRatioSpread = .1;

struct InformationDecision {
  bool supported = false;
  int64_t predicted_label = 0; // meaningful only when supported.
  double slow_frequency = 0;
  double fast_frequency = 0;
  double normalized_residual = 0;
  double margin = 0; // signed slow phase, or mean log(slow/fast amplitude).
  // Selected fit only, in ORIGINAL observation units. Rows c0f0..2,c1f0..2;
  // columns constant, slow sine/cosine, fast sine/cosine. Ineligible rows zero.
  std::array<std::array<double, 5>, 6> coefficients{};
  std::array<bool, 6> eligible{};
  int64_t frequency_pairs_evaluated = 0;
  std::string reason;
};

// CPU F64/F32 [B,3,32,3] and CPU Bool of exactly the same shape. Only legal
// observations enter ordinary scalar arithmetic; hidden storage is ignored.
// Labels, source IDs, true frequencies/delays and clean waveforms are absent.
// Channels 0/1 fit shared frequencies with independent affine coefficients per
// feature. Channel 2 is irrelevant to this data-only information diagnostic.
// The bounded grid/refinement and every abstention threshold above are fixed
// before engineering outcomes. This procedure is not an encoder or head fit.
std::vector<InformationDecision> infer_information(
    const torch::Tensor &observed, const torch::Tensor &mask, InformationTask task);

} // namespace embedding::evaluation::two_component
