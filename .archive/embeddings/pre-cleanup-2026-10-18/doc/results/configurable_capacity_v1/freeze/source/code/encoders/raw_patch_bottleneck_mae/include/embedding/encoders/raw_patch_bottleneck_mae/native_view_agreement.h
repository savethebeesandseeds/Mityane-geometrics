// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/learning_curve.h"
#include "embedding/shared/paired_pooling.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace native_view_agreement {
inline constexpr const char *policy_id = "rpb-training-native-view-agreement-015-v1";
inline constexpr const char *model_tag = "RPB-v9";
inline constexpr double agreement_weight = .05, variance_weight = .01;
inline constexpr double variance_target = .5, variance_epsilon = 1e-4, scale_floor = 1e-6;
inline constexpr const char *support_policy =
    "original-reconstruction-eligible;joint-sample-valid;equal-channel-valid-mask;exact-row-source-channel-endpoint-identity-v1";
inline constexpr const char *variance_policy =
    "first-supported-sampled-occurrence-per-distinct-source;population-variance;fewer-than-two-differentiable-zero-v1";
inline constexpr const char *scale_policy =
    "point0-CUDA-frozen-eval-full-O-native32;valid-TRAIN-rows;CPU-float64-population-std;floor1e-6;fixed-float32-loss-only-v1";

struct LossScale {
  torch::Tensor values; // CPU float32 [32], never a serving transform.
  torch::Tensor calibration_native, calibration_valid, floor_applied;
  std::vector<std::string> source_ids; // Exact TRAIN row order, including invalid rows.
  std::string identity;
};

struct RowIdentity {
  torch::Tensor row_indices, channel_ids, endpoints;
  std::vector<std::string> source_ids;
};

struct LossOutput {
  torch::Tensor agreement, variance, supported_rows, variance_rows;
  int64_t supported_count{0}, unique_source_count{0};
};

// Validates exact identities before support intersection; it never pairs rows
// by source alone. The ordinary native target is detached inside this function.
LossOutput view_loss(const EncodeOutput &ordinary, const EncodeOutput &student,
                     const RowIdentity &ordinary_identity, const RowIdentity &student_identity,
                     const torch::Tensor &original_eligible_examples,
                     const torch::Tensor &fixed_scale);

struct ForwardEvidence {
  ForwardOutput ordinary;
  EncodeOutput student;
  LossOutput auxiliary;
  torch::Tensor total_loss;
};

// The optional false branch is an admission-only exact ordinary-forward oracle.
// The production factory always enables the fixed auxiliary recipe.
ForwardEvidence training_forward(ModelImpl &model, const Input &normalized,
                                 const MaskPlan &original, const context_deletion::Plan &student_plan,
                                 const RowIdentity &identity, const torch::Tensor &fixed_scale,
                                 bool auxiliary_enabled = true);

LossScale calibrate_loss_scale(ModelImpl &model, const Input &normalized_full_observations,
                               const std::vector<std::string> &training_source_ids);
void validate_loss_scale(const LossScale &scale);
} // namespace native_view_agreement

// One fresh continuous CUDA trajectory; ordinary tagged reload/resume remains
// rejected. Original sampler/query/Torch streams, model and serving stay fixed.
embedding::evaluation::CurveTrainerFactory make_native_view_agreement_trainer(const Settings &settings);

// Validates this new training companion, then delegates unchanged ordinary
// CPU export/CUDA reconstruction to the existing retained snapshot loader.
embedding::evaluation::CurveSnapshot make_native_view_agreement_snapshot(
    const std::string &checkpoint_path, const embedding::evaluation::ProviderFitInput &training_only);

std::map<std::string, std::string> audit_native_view_agreement_initialization(
    const std::string &candidate_point_zero,
    const embedding::evaluation::RetainedPoolingCohort &reference,
    const embedding::evaluation::ProviderFitInput &training_only,
    int64_t expected_reference_updates = 512);
} // namespace embedding::encoders::raw_patch_bottleneck_mae
