// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_evaluation.h"
#include <cstdint>
#include <string>

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct EvaluationOptions {
  std::string config_path;
  std::string checkpoint_path;
  int64_t pretraining_updates{0};
  // Distinct registration names permit paired architecture variants in one card.
  std::string surface_prefix{"rpb"};
  bool require_channel_mixer{false};
};

// The callback sees only permitted training observations and declared metadata.
// Evaluation parsing, labels, held-out splits, probes and reporting stay outside
// the encoder adapter and ordinary synthetic/prepare/train/embed executable.
embedding::evaluation::FeatureProviderFactory
make_evaluation_provider(const EvaluationOptions &options);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
