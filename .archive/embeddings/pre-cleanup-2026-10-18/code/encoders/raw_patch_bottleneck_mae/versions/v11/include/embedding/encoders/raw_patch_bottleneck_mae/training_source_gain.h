// SPDX-License-Identifier: MIT
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae {
enum class TrainingSourceGainRecipe { disabled, unit_control_v1, source_log2_v1 };
struct TrainingSourceGainOptions {
  TrainingSourceGainRecipe recipe{TrainingSourceGainRecipe::disabled};
};
inline constexpr const char *kMatchedTargetGainFitProtocol = "matched-target-gain-v1/lag_sign";
inline constexpr const char *kMatchedTargetGainFixtureFitProtocol = "matched-target-gain-engineering-v1/lag_sign";
inline constexpr const char *kMatchedTargetGainPolicy = "rpb-training-context-deletion-015-source-gain-v1";
inline constexpr const char *kMatchedTargetGainViewSuffix = ".gain-view.pt";
inline constexpr const char *kMatchedTargetGainViewArtifact = "rpb_matched_target_gain_training_view_v1";
inline constexpr const char *kMatchedTargetGainContinuationArtifact = "rpb_matched_target_gain_continuation_state_v1";

namespace training_source_gain {
inline constexpr uint64_t stream = 0x6761696e2d763131ULL;
inline constexpr double ln2 = 0x1.62e42fefa39efp-1;
inline constexpr const char *rng_policy = "lexical-distinct-source-rank;counter_seed(actual_training_seed,rank,0x6761696e2d763131)>>11;top53-times-0x1p-53;no-label-or-Torch-RNG";
inline constexpr const char *gain_policy = "exp((2.0*u-1.0)*0x1.62e42fefa39efp-1);static-source-paired-CPU-float64-before-original-scaler";
inline constexpr const char *target_policy = "gained-legal-input-and-original-Q-target-same-normalized-batch;original-O-A-Q-E-eligibility-unchanged";
inline constexpr const char *inference_policy = "ordinary-original-view;no-source-gain;no-training-context-deletion;frozen-original-TRAIN-scaler";
struct Manifest {
  TrainingSourceGainRecipe recipe{TrainingSourceGainRecipe::disabled};
  uint64_t actual_training_seed{0};
  std::vector<std::string> source_ids;
  torch::Tensor source_ranks, row_source_ranks, source_top53, source_uniforms, source_gains, row_gains;
  std::string identity;
};
void validate_options(TrainingSourceGainOptions);
bool enabled(TrainingSourceGainOptions);
bool gained(TrainingSourceGainOptions);
const char *recipe_name(TrainingSourceGainRecipe);
Manifest make_manifest(const std::vector<std::string> &, uint64_t, TrainingSourceGainOptions);
Input apply(const Input &selected_original, const torch::Tensor &row_indices, const Manifest &);
void write_manifest(torch::serialize::OutputArchive &, const Manifest &);
void verify_manifest(torch::serialize::InputArchive &, const Manifest &);
} // namespace training_source_gain
} // namespace embedding::encoders::raw_patch_bottleneck_mae
