// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/early_mixer_adapter.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kStructuredHardTimingProtocol = "structured-hard-timing-comparison-v2";
inline constexpr const char *kStructuredHardTimingFitProtocol = "structured-hard-timing-comparison-v2/lag_sign";
inline constexpr const char *kStructuredHardTimingFixtureFitProtocol =
    "structured-hard-timing-comparison-engineering-v2/lag_sign";
inline constexpr const char *kStructuredHardTimingInformationRule = "observed-cross-feature-determinant-v2";
inline constexpr const char *kStructuredHardTimingSuffix = ".structured-timing.pt";
inline constexpr const char *kStructuredHardTimingArtifact = "rpb_structured_hard_timing_binding_v2";
inline constexpr const char *kStructuredHardTimingSnapshotArtifact = "rpb_structured_hard_timing_cuda_snapshot_v2";
inline constexpr const char *kStructuredHardTimingSnapshotFile = "structured-timing-snapshot-audit.pt";

enum class StructuredHardTimingScope { quality, engineering };

struct StructuredHardTimingSnapshotOptions {
  std::string checkpoint_path;
  int64_t expected_channel_mixer_placement{-1};
  int64_t expected_completed_updates{-1};
  StructuredHardTimingScope scope{StructuredHardTimingScope::quality};
  std::string expected_core_source_fingerprint;
  std::string expected_training_producer_source_fingerprint;
};

// NEW external cohort scope, unchanged historical implementation contract.
// The delegate receives an explicit copy with the reliability implementation
// namespace; its checkpoints and all old audit fields remain unchanged. A new
// fifth companion binds the external cohort, original fit and four parent bytes.
// Quality is fixed512/cap1024 and saves0/512; engineering is4/cap8 and saves0/2/4.
embedding::evaluation::CurveTrainerFactory make_structured_hard_timing_trainer(
    const Settings &, StructuredHardTimingScope = StructuredHardTimingScope::quality);

// Admit the structured timing companion BEFORE loading any model, then delegate the
// exact existing CUDA snapshot. No CPU forward, refit, optimizer or resume.
embedding::evaluation::CurveSnapshot make_structured_hard_timing_snapshot(
    const StructuredHardTimingSnapshotOptions &,
    const embedding::evaluation::ProviderFitInput &external_original_training);

std::map<std::string, std::string> audit_structured_hard_timing_initialization(
    const StructuredHardTimingSnapshotOptions &late_point0,
    const StructuredHardTimingSnapshotOptions &early_point0,
    const embedding::evaluation::ProviderFitInput &external_original_training);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
