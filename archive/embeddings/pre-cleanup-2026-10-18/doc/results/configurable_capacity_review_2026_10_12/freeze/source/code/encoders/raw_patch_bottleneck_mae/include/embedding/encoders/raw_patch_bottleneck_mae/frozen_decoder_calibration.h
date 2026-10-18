// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/decoder_calibration.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kFreshDecoderProtocol = "fresh-decoder-replication-v1";
inline constexpr const char *kFreshDecoderArtifact = "rpb_fresh_frozen_decoder_calibration_v1";
enum class FrozenDecoderParentPolicy { ordinary_v4, coordinate15_v7 };

struct FrozenDecoderCalibrationOptions {
  std::string parent_checkpoint_path;
  // Deliberately no implicit parent policy. Unknown selectors reject before
  // RNG/thread changes or input access; the caller declares v4 or v7 explicitly.
  FrozenDecoderParentPolicy parent_policy{static_cast<FrozenDecoderParentPolicy>(-1)};
  int64_t expected_parent_updates{512};
  int64_t additional_updates{128};
};

// Fresh namespace only: fresh-decoder-replication-v1/lag_sign. Original encoder
// updates and policy remain intact. One fresh decoder-only AdamW trains on CUDA;
// all inference also executes on CUDA, returning CPU evidence to shared APIs.
DecoderCalibrationRun make_frozen_decoder_calibration(
    const FrozenDecoderCalibrationOptions &options,
    const embedding::evaluation::ProviderFitInput &training);

embedding::evaluation::CurveSnapshot make_frozen_decoder_calibration_snapshot(
    const std::string &artifact_path, const std::string &parent_checkpoint_path,
    FrozenDecoderParentPolicy policy,
    const embedding::evaluation::ProviderFitInput &training);

// Exact fresh0/0 checkpoint only. No calibration/optimizer/CPU encoder inference.
// Immutable CUDA serving and reconstruction support the single untrained control
// and full initialized-parameter/scaler pairing. Historical APIs still require
// positive parents, and neither artifact loader accepts the other's format.
embedding::evaluation::CurveSnapshot make_fresh_initial_snapshot(
    const std::string &parent_checkpoint_path, FrozenDecoderParentPolicy policy,
    const embedding::evaluation::ProviderFitInput &training);

// Internal fixed bindings shared by the historical wrapper and fresh API.
// Callers cannot substitute an arbitrary protocol, policy, objective or budget.
namespace frozen_decoder_detail {
enum class Binding { historical_v7, fresh_v4, fresh_v7 };
DecoderCalibrationRun make(const DecoderCalibrationOptions &options,
    const embedding::evaluation::ProviderFitInput &training,
    Binding binding, const std::string &producer_source);
embedding::evaluation::CurveSnapshot snapshot(const std::string &artifact_path,
    const std::string &parent_checkpoint_path,
    const embedding::evaluation::ProviderFitInput &training,
    Binding binding, const std::string &producer_source);
} // namespace frozen_decoder_detail

} // namespace embedding::encoders::raw_patch_bottleneck_mae
