// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_harness.h"

namespace embedding::evaluation {

inline constexpr const char *kVariableDelayTimingProtocol = "variable-delay-timing-v1";

// Closed first harder recipe: C3/H32/F3, P~U[12,20), |delay|~U[.5,1.5),
// noise SD .005, natural missingness .10. No TEST is generated. The returned
// clean field aliases only a clone of legal observations, never hidden signals.
struct VariableDelayTimingDevelopment {
  uint64_t seed;
  ControlledDataset training, validation;
};
VariableDelayTimingDevelopment make_variable_delay_timing_development(
    int64_t training_pairs, int64_t validation_pairs, uint64_t seed);

struct TimingSolvabilityResult {
  torch::Tensor predictions; // CPU int64[B], meaningful only where valid.
  torch::Tensor valid; // CPU bool[B]; >=4 distinct supported times, nonzero margin.
  torch::Tensor margins; // CPU float64[B], time mean of within-time feature means.
  torch::Tensor supported_time_positions; // CPU int64[B].
};
// Analytic information check, not a fitted head. No labels, source IDs, true
// period/delay, clean signals, encoder or fitted assets enter this API.
TimingSolvabilityResult variable_delay_timing_solvability(const Batch &observations);

} // namespace embedding::evaluation
