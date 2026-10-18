// SPDX-License-Identifier: MIT
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kMatchedTargetGainProtocol = "matched-target-gain-v1";
inline constexpr const char *kMatchedTargetGainSnapshotArtifact = "rpb_matched_target_gain_cuda_snapshot_v1";
inline constexpr const char *kMatchedTargetGainSnapshotAuditFile = "matched-target-gain-snapshot-audit.pt";
enum class MatchedTargetGainPolicy { coordinate15_control, coordinate15_gain };
enum class MatchedTargetGainScope { quality, engineering };
struct MatchedTargetGainSnapshotOptions {
  std::string checkpoint_path;
  MatchedTargetGainPolicy policy;
  MatchedTargetGainScope scope{MatchedTargetGainScope::quality};
  int64_t expected_completed_updates{-1};
  std::string expected_core_source_fingerprint;
  std::string expected_training_producer_source_fingerprint;
};

// One fresh continuous AdamW/scaler session. Quality is exactly0/512; engineering
// is0/1/2/4. Control retains literal original .15 mathematics. Candidate gains
// legal CPU F64 input and Q targets together before the frozen original scaler.
// Tagged resume is deliberately unavailable; all callbacks preserve runtime RNG.
embedding::evaluation::CurveTrainerFactory make_matched_target_gain_trainer(
    const Settings &, MatchedTargetGainPolicy,
    MatchedTargetGainScope = MatchedTargetGainScope::quality);

// Independent frozen CUDA serving of ordinary original observations. Validates
// all six checkpoint companions; no gain/E, optimizer, CPU model or refitting.
embedding::evaluation::CurveSnapshot make_matched_target_gain_snapshot(
    const MatchedTargetGainSnapshotOptions &,
    const embedding::evaluation::ProviderFitInput &original_training);

// Same early architecture, complete point0 model/buffers/scaler/data/counter
// pairing. Original-view CUDA exports are separately proved by the driver/test.
std::map<std::string,std::string> audit_matched_target_gain_initialization(
    const std::string &control_point0,const std::string &candidate_point0,
    const embedding::evaluation::ProviderFitInput &original_training,
    MatchedTargetGainScope = MatchedTargetGainScope::quality);
} // namespace embedding::encoders::raw_patch_bottleneck_mae
