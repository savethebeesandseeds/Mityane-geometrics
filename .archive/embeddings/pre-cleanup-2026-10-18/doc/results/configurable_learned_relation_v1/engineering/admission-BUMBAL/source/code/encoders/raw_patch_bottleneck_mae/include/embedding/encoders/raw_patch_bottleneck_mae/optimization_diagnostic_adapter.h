// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/learning_curve.h"
#include <cstdint>
#include <string>

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct OptimizationResumeOptions {
  std::string parent_checkpoint, training_archive, output_checkpoint;
  int64_t additional_updates{0};
};

struct OptimizationResumeResult {
  std::string checkpoint_path, audit_path, audit_json;
  int64_t attempted_before{0}, completed_before{0}, attempted_after{0}, completed_after{0};
  // Synchronized whole run_cli command: loading, normalization, updates,
  // logging and checkpoint writing. GPU-update-only duration is unmeasured.
  double synchronized_command_wall_seconds{0};
};

// Original exact TRAIN only; labels and held-out observations are not accepted.
// Requires mode3/mixer1/native32, explicit CUDA, a known parent producer audit
// and new output paths. Restores ordinary AdamW/counters through run_cli without
// changing its numerical loop. Writes output.audit.pt and output diagnostic JSON.
OptimizationResumeResult resume_optimization_diagnostic(
    const OptimizationResumeOptions &options,
    const embedding::evaluation::ProviderFitInput &training_only);

// Independently loaded native32 CPU serving and frozen CUDA reconstruction.
// No scaler/model fitting. The original TRAIN association is verified once;
// held-out callbacks capture metadata, model and scaler only.
embedding::evaluation::CurveSnapshot make_optimization_diagnostic_snapshot(
    const std::string &checkpoint_path,
    const embedding::evaluation::ProviderFitInput &training_only);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
