// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_evaluation.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace embedding::evaluation {

struct ReconstructionLatent {
  torch::Tensor values; // Detached CPU floating [B,C,D], the exact served vector.
  torch::Tensor valid;  // CPU bool [B,C]; observed support, never inferred.
};

struct ReconstructionPrediction {
  torch::Tensor values; // CPU floating [B,C,H,F], raw input units.
  torch::Tensor valid;  // CPU bool [B,C]; observed-support export validity.
};

// A decoder receives only the supplied compact vectors and captured fixed
// metadata. Metadata-only prediction receives support, never observed values.
struct ReconstructionProvider {
  std::string name, export_semantics, provenance;
  std::function<ReconstructionLatent(const Batch &)> encode_visible;
  std::function<torch::Tensor(const torch::Tensor &)> decode;
  std::function<torch::Tensor(const torch::Tensor &)> metadata_predict;
  std::function<ReconstructionPrediction(const Batch &)> untrained_predict;
  std::map<std::string, std::string> audit_fields;
  std::function<void(const std::string &)> save_assets;
};
using ReconstructionProviderFactory =
    std::function<ReconstructionProvider(const ProviderFitInput &)>;

struct ReconstructionCard {
  std::string id{"controlled-reconstruction-v1"};
  int64_t version{1};
  std::string policy_version{"1.1"}, stage{"development"};
  input_shape_t shape{3, 32, 3, torch::kFloat64, torch::kCPU};
  std::vector<int64_t> channel_ids{0, 1, 2};
  std::string feature_units{"unitless,unitless,unitless"};
  double sampling_interval{1.0}, huber_delta{1.0};
  int64_t patch_length{8}, train_pairs{32}, validation_pairs{16}, test_pairs{64};
  std::vector<uint64_t> seeds{101, 202, 303};
  std::vector<Task> tasks{Task::reversal, Task::level, Task::amplitude, Task::lag_sign};
  int64_t threads{1}, bootstrap_replicates{1000};
  // Exact resolved adapter settings/budgets, captured before fitting.
  std::string provider_recipe;
};

struct ReconstructionRun {
  ReconstructionCard card;
  std::string output_directory;
  std::string source_fingerprint{"unrecorded"};
  std::string git_head{"unrecorded"}, git_dirty{"unrecorded"};
};

void validate_reconstruction_card(const ReconstructionCard &card);
std::string reconstruction_card_json(const ReconstructionCard &card);
// Fully observed controlled track, one complete target patch per channel;
// enumerate every patch. Shuffle complete source groups in disjoint swaps.
// Bootstrap whole exchange blocks, keeping donor/recipient and repeated masks
// together. Broader missingness and acceptance remain separate protocols.
void run_reconstruction_evaluation(const ReconstructionRun &run,
                                   const ReconstructionProviderFactory &factory);

} // namespace embedding::evaluation
