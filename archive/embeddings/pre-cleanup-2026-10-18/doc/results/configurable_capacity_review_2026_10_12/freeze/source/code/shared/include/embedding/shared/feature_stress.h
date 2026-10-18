// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_evaluation.h"
#include <functional>
#include <map>
#include <string>

namespace embedding::evaluation {

struct StressPredictions {
  torch::Tensor valid, ridge, tiny; // CPU bool/int64 [B]; predictions are 0/1.
};

// Every transform/readout in predict was fitted on the ordinary training split.
// Unsupported ordinary fits retain their status and are never fitted under stress.
struct FrozenStressReadout {
  std::string status, reason;
  int64_t probe_dimensions{0};
  torch::Tensor native_valid, native_ridge, native_tiny;
  std::function<StressPredictions(const FeatureSurface &)> predict;
};

using StressReadouts =
    std::map<std::pair<std::string, DimensionTier>, FrozenStressReadout>;
using StressExtractor = std::function<FeatureMap(const Batch &)>;

// Fixed before any data generation. This is a development robustness diagnostic,
// not an acceptance card and not a training augmentation.
std::string fixed_readout_stress_card_json(const EvaluationCard &card);

// Creates a unique stress/ child of the existing per-seed/task directory.
// Only testing observations are corrupted; labels are used only after extraction.
// The extractor receives a fresh clone for every case, preserving saved masks.
std::string run_fixed_readout_stress(
    const std::string &directory, const EvaluationCard &card,
    const ControlledProtocol &protocol,
    const std::map<std::string, SurfaceDescription> &descriptions,
    const StressReadouts &readouts, const StressExtractor &extractor);

} // namespace embedding::evaluation
