// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_source_gain.h"
#include "embedding/shared/learning_curve.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kLearningCurveContinuationSuffix = ".continuation.pt";
inline constexpr const char *kLearningCurveContinuationArtifact =
    "rpb_early_mixer_curve_continuation_state_v1";
inline constexpr const char *kPooledContextContinuationArtifact =
    "rpb_pooled_context_continuation_state_v1";
inline constexpr const char *kVisibleDifferenceContinuationArtifact =
    "rpb_visible_difference_continuation_state_v1";
inline constexpr const char *kVisibleDifferenceBindingSuffix = ".visible-difference.pt";
inline constexpr const char *kVisibleDifferenceBindingArtifact =
    "rpb_visible_difference_training_binding_v1";
struct VisibleDifferenceOptions {
  bool enabled{false};
  // Original compact early point0; old writer and fit identities are explicit
  // and independent of the new trainer's enclosing source fingerprint.
  std::string parent_point0_checkpoint_path;
  std::string expected_parent_core_source_fingerprint;
  std::string expected_parent_training_producer_source_fingerprint;
  std::string expected_parent_fit_protocol;
};
struct PooledContextInitializationOptions {
  bool enabled{false};
  // Candidate only: immutable compact-control point0 in the SAME new TRAIN scope.
  // Empty for the compact control; never an ordinary checkpoint resume.
  std::string control_point0_checkpoint_path;
};

struct LearningCurveStateWitnessOptions {
  bool enabled{false};
};

// Captures resolved CUDA settings once. The label-free fit callback clones only
// permitted training observations and fits one frozen scaler. Completed budgets
// are absolute; optimizer state and attempted counter streams are continuous.
embedding::evaluation::CurveTrainerFactory
make_learning_curve_trainer(const Settings &settings);

// Fresh continuous TRAIN only. Enabled defaults to fixed RPB-v6 .30; explicit
// coordinate15_v1 selects RPB-v7 .15 on the same mode2/mixer1/native32 architecture.
// balanced30_v1 selects RPB-v8: absolute even attempts are ordinary forward,
// odd attempts use the original .30 deletion. Ineligible attempts abort this recipe.
// Checkpoints retain a policy tag; ordinary workflow resume must reject them.
// No augmented resume API is exposed in this bounded experiment.
// Placement1 is additionally bound to the new early-mixer timing protocol and
// carries RPB-v10 architecture provenance; its historical snapshot callback
// rejects placement1. Use make_early_mixer_trainer for CUDA-only snapshots.
embedding::evaluation::CurveTrainerFactory
make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options);

// Additive live-state witness for the separately bound early-mixer curve only.
// Old overloads keep this disabled and preserve their saved artifact schemas.
// No optimizer is constructed or reloaded when capturing the witness.
embedding::evaluation::CurveTrainerFactory make_learning_curve_trainer(
    const Settings &settings, ContextDeletionOptions options,
    LearningCurveStateWitnessOptions state_witness);

// Closed new TRAIN-view protocol only. Disabled delegates the literal old
// transform/archive path; unit_control_v1 preserves its mathematics while
// requesting new witnesses, and source_log2_v1 applies matched input/Q gains.
embedding::evaluation::CurveTrainerFactory make_learning_curve_trainer(
    const Settings &settings, ContextDeletionOptions options,
    LearningCurveStateWitnessOptions state_witness, TrainingSourceGainOptions gain);

// Closed pooled-context scope only. Existing overloads keep this disabled and
// reject source1. Common tensors are copied before AdamW and its initial witness.
embedding::evaluation::CurveTrainerFactory make_learning_curve_trainer(
    const Settings &, ContextDeletionOptions, LearningCurveStateWitnessOptions,
    TrainingSourceGainOptions, PooledContextInitializationOptions);

// Closed visible-difference protocol only. All historical overloads disable
// this option and reject temporal_difference_input1 before reading TRAIN.
embedding::evaluation::CurveTrainerFactory make_learning_curve_trainer(
    const Settings &, ContextDeletionOptions, LearningCurveStateWitnessOptions,
    TrainingSourceGainOptions, PooledContextInitializationOptions,
    VisibleDifferenceOptions);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
