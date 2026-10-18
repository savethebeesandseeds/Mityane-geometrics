// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_evaluation.h"
#include <functional>
#include <string>
#include <vector>

namespace embedding::evaluation {

struct CurveLossPoint {
  int64_t attempted{0}, completed{0}, target_cells{0};
  double loss{0}, gradient_norm{0};
};

struct CurveProgress {
  int64_t attempted{0}, completed{0}, parameter_count{0}, cuda_parameter_count{0};
  int64_t sampled_rows{0};
  double training_seconds{0}; // CUDA synchronized; cumulative training only.
  std::string training_device, preprocessing_id, training_dataset_id;
  bool last_input_cuda{false}, last_loss_cuda{false}, finite_gradients{false};
  bool weights_changed{false}; // Relative to exact point-zero initialization.
  std::vector<CurveLossPoint> losses;
};

struct CurveReconstruction {
  // CPU tensors in units standardized by the frozen TRAINING scaler.
  torch::Tensor prediction, target; // Floating [B,C,H,F].
  torch::Tensor eligible; // Bool [B,C]; fixed target support eligibility.
};

struct CurveSnapshot {
  FeatureProvider features; // Exact saved checkpoint, CPU feature surfaces.
  // Receives legal observed values and a complete-patch artificial mask only.
  // Never receives clean hidden signals, labels or another split for fitting.
  std::function<CurveReconstruction(const Batch &, const torch::Tensor &)> reconstruct;
};

struct CurveTrainer {
  // Monotonic ABSOLUTE completed update budget; optimizer/counters persist.
  std::function<CurveProgress(int64_t)> train_to;
  std::function<void(const std::string &)> save_checkpoint;
  // Load an immutable copy of an exact ordinary resumable checkpoint.
  std::function<CurveSnapshot(const std::string &)> snapshot;
  std::map<std::string, std::string> audit_fields;
};
using CurveTrainerFactory = std::function<CurveTrainer(const ProviderFitInput &)>;

struct NamedCurveFactory {
  std::string name, recipe;
  CurveTrainerFactory factory;
};

struct LearningCurveRun {
  std::string output_directory;
  std::string source_fingerprint{"unrecorded"}, git_head{"unrecorded"}, git_dirty{"unrecorded"};
  input_shape_t shape{3, 32, 3, torch::kFloat64, torch::kCPU};
  std::vector<int64_t> channel_ids{0, 1, 2};
  std::string feature_units{"unitless,unitless,unitless"};
  double sampling_interval{1.0}, huber_delta{1.0};
  int64_t patch_length{8}, train_pairs{32}, validation_pairs{64}, test_pairs{64};
  int64_t matched_global_width{12}, matched_channel_width{36}, threads{1};
  std::vector<uint64_t> seeds{901, 1002, 1103};
  std::vector<int64_t> milestones{0, 128, 512, 2048};
  // One common nonzero budget maximizes mean validation PCA36 ridge accuracy
  // over every registered architecture and seed. Exact ties choose the smaller.
  // No testing observations are generated until selection.json is persisted.
};

void run_learning_curve(const LearningCurveRun &run,
                        const std::vector<NamedCurveFactory> &factories);

} // namespace embedding::evaluation
