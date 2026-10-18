// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h"
#include "embedding/shared/feature_evaluation.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kFrozenNativeFeatureProtocol =
    "rpb-frozen-native-feature-v1";
inline constexpr const char *kFrozenNativeFeatureArtifact =
    "rpb_frozen_native_feature_snapshot_v1";
inline constexpr const char *kFrozenNativeFeatureAuditFile =
    "frozen-native-feature-audit.pt";

struct FrozenNativeFeatureOptions {
  std::string parent_checkpoint_path;
  FrozenDecoderParentPolicy parent_policy{
      static_cast<FrozenDecoderParentPolicy>(-1)};
  // Explicit nonnegative original budget. The amplitude card restricts its
  // quality parents to ordinary point0/512 and coordinate15 point512.
  int64_t expected_parent_updates{-1};
  std::string expected_parent_core_source_fingerprint;
  std::string expected_training_producer_source_fingerprint;
};

// Admit the exact original timing TRAIN/scaler/checkpoint and producer scopes.
// The original metadata namespace is fresh-decoder-replication-v1/lag_sign;
// new transfer observations enter extract() only. No fitting, optimizer,
// reconstruction or checkpoint-writing callback is exposed. All encoder
// inference executes on CUDA and returns detached CPU Float32 native32 values.
embedding::evaluation::FeatureProvider make_frozen_native_feature_provider(
    const FrozenNativeFeatureOptions &options,
    const embedding::evaluation::ProviderFitInput &original_timing_training);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
