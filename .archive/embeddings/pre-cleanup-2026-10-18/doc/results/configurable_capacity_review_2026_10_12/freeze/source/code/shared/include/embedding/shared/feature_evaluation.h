// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_harness.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace embedding::evaluation {

enum class SurfaceKind { global, channel_concatenation, control };
enum class DimensionTier { native, matched_global, matched_channels };

struct SurfaceDescription {
  SurfaceKind kind{SurfaceKind::control};
  std::string support_rule;
  std::vector<int64_t> channel_order; // Required for channel_concatenation.
};

using FeatureMap = std::map<std::string, FeatureSurface>;

// Fit callbacks never receive labels, hidden clean signals, validation or test
// observations. Frozen checkpoint adapters may ignore training observations.
struct ProviderFitInput {
  Batch training_observations;
  input_shape_t shape;
  uint64_t seed{0};
  std::vector<std::string> training_source_ids;
  std::vector<int64_t> channel_ids;
  std::string feature_units;
  std::string protocol_id;
  double sampling_interval{1.0};
  double endpoint{31.0};
};

struct FeatureProvider {
  std::function<FeatureMap(const Batch &)> extract;
  std::map<std::string, SurfaceDescription> surfaces;
  std::string provenance;
  std::map<std::string, std::string> audit_fields;
  std::function<void(const std::string &)> save_assets;
};
using FeatureProviderFactory = std::function<FeatureProvider(const ProviderFitInput &)>;

struct PairComparison {
  std::string id, left, right;
  DimensionTier tier{DimensionTier::matched_global};
};

struct EvaluationCard {
  std::string id{"controlled-pairs-v2"};
  int64_t version{2};
  std::string policy_version{"1.1"};
  std::string stage{"development"};
  input_shape_t shape{3, 32, 3, torch::kFloat64, torch::kCPU};
  std::vector<int64_t> channel_ids{0, 1, 2};
  std::string feature_units{"unitless,unitless,unitless"};
  double sampling_interval{1.0};
  int64_t matched_global_width{12}, matched_channel_width{36};
  int64_t train_pairs{32}, validation_pairs{16}, test_pairs{32};
  std::vector<uint64_t> seeds{101, 202, 303};
  std::vector<Task> tasks{Task::reversal, Task::level, Task::amplitude, Task::lag_sign};
  int64_t threads{1};
  // Frozen before generation/extraction; extra providers do not expand a pair.
  std::vector<PairComparison> comparisons;
};

struct EvaluationRun {
  EvaluationCard card;
  std::string output_directory;
  std::string source_fingerprint{"unrecorded"};
  std::string git_head{"unrecorded"}, git_dirty{"unrecorded"};
  // Additional test-only deletions use the ordinary frozen providers/readouts.
  bool stress_sweep{false};
};

void validate_evaluation_card(const EvaluationCard &card);
std::string evaluation_card_json(const EvaluationCard &card);
// Shared development driver. Confirmation/acceptance is deliberately unsupported
// until the card's thresholds, resource and robustness contracts are implemented.
void run_feature_evaluation(const EvaluationRun &run,
                            const std::vector<FeatureProviderFactory> &factories);

} // namespace embedding::evaluation
