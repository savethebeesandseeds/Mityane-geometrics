// SPDX-License-Identifier: MIT
#pragma once
#include "embedding/shared/feature_harness.h"

namespace embedding::evaluation {
inline constexpr const char *kStructuredHardTimingProtocol = "structured-hard-timing-v1";
inline constexpr const char *kStructuredHardTimingDataset = "TEMPO-3";
inline constexpr int64_t kStructuredHardTimingComplexity = 4;
inline constexpr int64_t kStructuredHardTimingComplexityMaximum = 5;
inline constexpr uint64_t kStructuredHardTimingDeletionStream = 0x746d70332d64656cULL;

struct StructuredHardTimingDevelopment {
  uint64_t seed;
  ControlledDataset training, validation;
};
// C3/H32/F3; positive affine nuisances, bounded periods/delays, low noise,
// paired natural masks and short channel gaps. No TEST or hidden truth export.
StructuredHardTimingDevelopment make_structured_hard_timing_development(
    int64_t training_pairs, int64_t validation_pairs, uint64_t seed);

struct HardTimingSolvabilityResult {
  torch::Tensor predictions; // CPU int64[B], meaningful only where valid.
  torch::Tensor valid; // CPU bool[B], >=4 distinct centres and nonzero margin.
  torch::Tensor margins; // CPU float64[B], feature/spacing/centre means.
  torch::Tensor supported_time_positions; // CPU int64[B], distinct centres.
};
// Observed-only offset-invariant sign rule. No labels, true parameters,
// source IDs, fitted assets or encoder are arguments. This is not a head fit.
HardTimingSolvabilityResult structured_hard_timing_solvability(const Batch &observations);
} // namespace embedding::evaluation
