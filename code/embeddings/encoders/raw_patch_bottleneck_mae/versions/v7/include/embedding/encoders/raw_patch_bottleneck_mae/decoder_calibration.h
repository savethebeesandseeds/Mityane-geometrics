// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/learning_curve.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kDecoderCalibrationProtocol = "v7-decoder-calibration-v1";
inline constexpr const char *kDecoderCalibrationArtifact = "rpb_v7_decoder_calibration_v1";

struct DecoderCalibrationOptions {
  std::string parent_checkpoint_path;
  int64_t expected_parent_updates{512};
  int64_t additional_updates{128};
};

struct DecoderCalibrationLoss {
  int64_t absolute_attempt{0}, completed{0}, target_cells{0}, eligible_examples{0};
  double loss{0}, gradient_norm{0};
};

struct DecoderCalibrationProgress {
  int64_t original_encoder_attempted{0}, original_encoder_completed{0};
  int64_t attempted{0}, completed{0}, sampled_rows{0}, next_absolute_attempt{0};
  int64_t decoder_parameter_count{0}, frozen_parameter_count{0};
  double training_seconds{0}; // Synchronized loop; includes CPU evidence copies.
  bool encoder_parameters_exact{false}, encoder_buffers_exact{false}, scaler_exact{false};
  bool native_exports_exact{false}, zero_encoder_gradients{false};
  bool last_input_cuda{false}, last_latent_cuda{false}, last_loss_cuda{false};
  bool finite_decoder_gradients{false}, decoder_weights_changed{false};
  std::vector<DecoderCalibrationLoss> losses; // Every successful update.
};

struct DecoderCalibrationRun {
  // Absolute additional decoder updates, with one continuous fresh AdamW state.
  std::function<DecoderCalibrationProgress(int64_t)> train_to;
  std::function<void(const std::string &)> save;
  std::function<embedding::evaluation::CurveSnapshot(const std::string &)> snapshot;
  std::map<std::string, std::string> audit_fields;
};

// Label-free TRAIN only, CUDA only. No encoder/scaler fitting and no ordinary
// checkpoint resume. Parent must be an exact unskipped RPB-v7 positive point.
DecoderCalibrationRun make_decoder_calibration(
    const DecoderCalibrationOptions &options,
    const embedding::evaluation::ProviderFitInput &training);

// Compose only validated decoder tensors with an explicitly supplied immutable
// parent; all model inference uses an independent frozen CUDA copy. Feature
// tensors returned to the shared interface are CPU tensors; no heads are fitted.
embedding::evaluation::CurveSnapshot make_decoder_calibration_snapshot(
    const std::string &artifact_path, const std::string &parent_checkpoint_path,
    const embedding::evaluation::ProviderFitInput &training);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
