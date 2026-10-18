// SPDX-License-Identifier: MIT
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kPooledContextProtocol="pooled-context-v1";
inline constexpr const char *kPooledContextFitProtocol="pooled-context-v1/lag_sign";
inline constexpr const char *kPooledContextFixtureFitProtocol="pooled-context-engineering-v1/lag_sign";
inline constexpr const char *kPooledContextSnapshotArtifact="rpb_pooled_context_cuda_snapshot_v1";
inline constexpr const char *kPooledContextSnapshotAuditFile="pooled-context-snapshot-audit.pt";
enum class PooledContextRole { compact_control, pooled_width_candidate };
enum class PooledContextScope { quality, engineering };
struct PooledContextSnapshotOptions {
  std::string checkpoint_path;
  PooledContextRole role;
  int64_t expected_completed_updates{-1};
  PooledContextScope scope{PooledContextScope::quality};
  std::string expected_core_source_fingerprint,expected_training_producer_source_fingerprint;
};
// One fresh .15 continuous session. Candidate requires the SAME original TRAIN
// compact point0 and copies every shared named tensor before AdamW is created.
// Quality0/512; engineering0/1/2/4. No resume or inference augmentation.
embedding::evaluation::CurveTrainerFactory make_pooled_context_trainer(
    const Settings &,PooledContextRole,const std::string &control_point0_checkpoint_path="",
    PooledContextScope=PooledContextScope::quality);
embedding::evaluation::CurveSnapshot make_pooled_context_snapshot(
    const PooledContextSnapshotOptions &,const embedding::evaluation::ProviderFitInput &);
std::map<std::string,std::string> audit_pooled_context_initialization(
    const std::string &control_point0,const std::string &candidate_point0,
    const embedding::evaluation::ProviderFitInput &,PooledContextScope=PooledContextScope::quality);
} // namespace embedding::encoders::raw_patch_bottleneck_mae
