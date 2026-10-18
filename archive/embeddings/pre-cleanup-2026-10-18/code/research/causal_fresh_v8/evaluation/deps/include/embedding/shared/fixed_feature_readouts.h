// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_harness.h"
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::evaluation {

struct FixedFeatureMethod {
  std::string name;
  FeatureSurface training, validation_intact, validation_deleted;
  // True only for caller-prepared TRAIN-normalized raw values or raw PCA
  // components. The caller persists the shared raw normalizer/PCA. The fixed
  // probe still fits its own TRAIN normalizer; this helper never instantiates
  // PCA or an encoder.
  bool inputs_train_prepared{false};
  // Optional caller-owned preprocessing failure (e.g. raw PCA rank bound).
  // Preserve declared-width surfaces/coverage; fit neither head in this case.
  std::string preparation_unsupported_reason;
};

struct FixedFeatureComparison {
  std::string reference, candidate;
};

struct FixedFeatureReadoutRun {
  std::string output_directory; // New exclusive leaf; parent may already exist.
  uint64_t master_seed{0};
  torch::Tensor training_labels, validation_labels; // CPU int64, binary [B].
  std::vector<std::string> training_source_ids, validation_source_ids;
  std::vector<FixedFeatureMethod> methods;
  // Optional explicit pair, scored as candidate minus reference. Both must
  // name distinct declared methods. Empty keeps the historical v7/v4 default.
  std::string comparison_reference, comparison_candidate;
  // Multiple explicit pairs reuse the same fitted heads and predictions.
  // Incompatible with the single-pair fields; empty preserves their behavior.
  std::vector<FixedFeatureComparison> comparisons;
};

// Fixed recipe: outer TRAIN FeatureNormalizer (unless already prepared), then
// RidgeProbe penalty1 and TinyProbe tanh16/100 Adam0.01. Repetitions2701/2802/
// 2903 use stream_seed(repetition,width), identical across equal-size methods.
// Fit once; save fit plus TRAIN/intact/deleted predictions; retained fit tensors
// remain exact. JSON retains unsupported fits/coverage and within-master source
// intervals. When named methods native_v4/native_v7 are both present, include
// common-valid v7-minus-v4 Ridge/Tiny paired intervals for each VALIDATION view.
// An explicit comparison uses the same math and candidate_minus_reference seed
// naming, without requiring encoder-specific names.
std::string run_fixed_feature_readouts(const FixedFeatureReadoutRun &run);

} // namespace embedding::evaluation
