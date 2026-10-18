// SPDX-License-Identifier: MIT
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kVisibleDifferenceProtocol="visible-difference-v1";
inline constexpr const char *kVisibleDifferenceFitProtocol="visible-difference-v1/lag_sign";
inline constexpr const char *kVisibleDifferenceFixtureFitProtocol="visible-difference-engineering-v1/lag_sign";
inline constexpr const char *kVisibleDifferenceSnapshotArtifact="rpb_visible_difference_cuda_snapshot_v1";
inline constexpr const char *kVisibleDifferenceSnapshotAuditFile="visible-difference-snapshot-audit.pt";
enum class VisibleDifferenceScope { quality, engineering };
struct VisibleDifferenceSnapshotOptions {
  std::string checkpoint_path;
  int64_t expected_completed_updates{-1};
  VisibleDifferenceScope scope{VisibleDifferenceScope::quality};
  std::string expected_core_source_fingerprint,expected_training_producer_source_fingerprint;
};
// Assert exact original independent common initialization before AdamW; no
// parent state is copied and no optimizer is resumed. Six immutable point roles.
embedding::evaluation::CurveTrainerFactory make_visible_difference_trainer(
    const Settings &,const VisibleDifferenceOptions &,VisibleDifferenceScope=VisibleDifferenceScope::quality);
embedding::evaluation::CurveSnapshot make_visible_difference_snapshot(
    const VisibleDifferenceSnapshotOptions &,const embedding::evaluation::ProviderFitInput &);
} // namespace embedding::encoders::raw_patch_bottleneck_mae
