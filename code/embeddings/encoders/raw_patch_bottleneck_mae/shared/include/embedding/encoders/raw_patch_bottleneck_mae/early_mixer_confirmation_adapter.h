// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/early_mixer_adapter.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kEarlyMixerConfirmationProtocol = "early-mixer-confirmation-v1";
inline constexpr const char *kEarlyMixerConfirmationFitProtocol = "early-mixer-confirmation-v1/lag_sign";
inline constexpr const char *kEarlyMixerConfirmationFixtureFitProtocol =
    "early-mixer-confirmation-engineering-v1/lag_sign";
inline constexpr const char *kEarlyMixerConfirmationSuffix = ".confirmation.pt";
inline constexpr const char *kEarlyMixerConfirmationArtifact = "rpb_early_mixer_confirmation_binding_v1";
inline constexpr const char *kEarlyMixerConfirmationSnapshotArtifact = "rpb_early_mixer_confirmation_cuda_snapshot_v1";
inline constexpr const char *kEarlyMixerConfirmationSnapshotFile = "confirmation-snapshot-audit.pt";

enum class EarlyMixerConfirmationScope { quality, engineering };

struct EarlyMixerConfirmationSnapshotOptions {
  std::string checkpoint_path;
  int64_t expected_channel_mixer_placement{-1};
  int64_t expected_completed_updates{-1};
  EarlyMixerConfirmationScope scope{EarlyMixerConfirmationScope::quality};
  std::string expected_core_source_fingerprint;
  std::string expected_training_producer_source_fingerprint;
};

// NEW external cohort scope, unchanged historical implementation contract.
// The delegate receives an explicit copy with the reliability implementation
// namespace; its checkpoints and all old audit fields remain unchanged. A new
// fifth companion binds the external cohort, original fit and four parent bytes.
// Quality is fixed512/cap1024 and saves0/512; engineering is4/cap8 and saves0/2/4.
embedding::evaluation::CurveTrainerFactory make_early_mixer_confirmation_trainer(
    const Settings &, EarlyMixerConfirmationScope = EarlyMixerConfirmationScope::quality);

// Admit the confirmation companion BEFORE loading any model, then delegate the
// exact existing CUDA snapshot. No CPU forward, refit, optimizer or resume.
embedding::evaluation::CurveSnapshot make_early_mixer_confirmation_snapshot(
    const EarlyMixerConfirmationSnapshotOptions &,
    const embedding::evaluation::ProviderFitInput &external_original_training);

std::map<std::string, std::string> audit_early_mixer_confirmation_initialization(
    const EarlyMixerConfirmationSnapshotOptions &late_point0,
    const EarlyMixerConfirmationSnapshotOptions &early_point0,
    const embedding::evaluation::ProviderFitInput &external_original_training);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
