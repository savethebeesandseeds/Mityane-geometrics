// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <array>
#include <string>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae {

// All paths are explicit TRAIN-only roles bound to a frozen SHA manifest by
// the caller. This adapter performs no split discovery, fitting or updates.
struct TrainingObjectiveDiagnosticInstance {
  std::string input_id, model_tag;
  uint64_t master_seed{0};
  int64_t budget{512};
  std::string expected_policy_id, training_namespace;
  std::string orchestration_source_id, training_producer_source_id, core_writer_source_id;
  std::string controlled_training_path, checkpoint_path, checkpoint_audit_path;
  std::string scaler_path, training_raw_path, native_training_path, training_reconstruction_path;
  std::array<std::string, 3> ridge_fit_paths, saved_training_prediction_paths;
};

struct TrainingObjectiveDiagnosticRun {
  std::string output_directory;
  std::string source_fingerprint;
  std::vector<TrainingObjectiveDiagnosticInstance> instances;
};

// Strict training-objective-diagnostic-v1: five masters, v4/v7/v8 at512,
// native32, four common-patch trials and absolute attempts512..515. Writes only
// a new output directory and returns its report JSON. CUDA is mandatory.
std::string run_training_objective_diagnostic(const TrainingObjectiveDiagnosticRun &run);

namespace objective_diagnostic {

struct PairDifferenceLoss {
  torch::Tensor loss, pair_losses, pair_target_counts, pair_valid, example_valid;
  int64_t supported_examples{0}, supported_pairs{0}, target_pair_cells{0};
};
// Original Q only. Pair means -> supported-pair means per example -> supported
// example mean. An unsupported auxiliary is differentiable zero, never a fit.
PairDifferenceLoss pair_difference_huber(const torch::Tensor &prediction,
    const torch::Tensor &target, const torch::Tensor &query, double delta);

struct FrozenRidgeHead {
  torch::Tensor outer_mean, outer_scale, ridge_mean, ridge_scale, weights, intercept;
};
// Final Ridge-standardized coordinates. At u=0 the decoder receives exactly z,
// with no inverse-normalization round trip. The saved two maps remain frozen.
torch::Tensor latent_from_zero_perturbation(const torch::Tensor &z,
    const torch::Tensor &u, const FrozenRidgeHead &head);
torch::Tensor signed_ridge_margin(const torch::Tensor &z, const torch::Tensor &labels,
    const FrozenRidgeHead &head);

struct GradientComparison {
  bool supported{false};
  double base_norm{0}, auxiliary_norm{0}, margin_norm{0}, combined_norm{0};
  double base_margin_dot{0}, combined_margin_dot{0}, combined_base_dot{0};
  double base_descent_margin_cosine{0}, combined_descent_margin_cosine{0};
  double combined_descent_base_directional_derivative{0};
};
GradientComparison compare_gradients(const torch::Tensor &base,
    const torch::Tensor &auxiliary, const torch::Tensor &margin);

// Narrow numerical admission surface for the coordinated CUDA test. Returns
// raw CPU audit tensors; parameter values/buffers and caller RNG are unchanged.
struct BankEvidence {
  torch::Tensor frozen_z, graph_z, prediction, graph_prediction, target, query, visible;
  torch::Tensor latent_base_gradient, latent_auxiliary_gradient, latent_margin_gradient;
  torch::Tensor encoder_base_gradient, encoder_auxiliary_gradient, encoder_margin_gradient;
  torch::Tensor parameter_offsets;
  std::vector<std::string> parameter_names, parameter_groups;
  PairDifferenceLoss auxiliary;
  GradientComparison comparison;
  double base_loss{0}, auxiliary_loss{0}, graph_export_max_error{0};
  int64_t cuda_parameter_count{0};
};
BankEvidence inspect_bank(Model &model, const Input &normalized,
    const MaskPlan &original, const torch::Tensor &visible,
    const torch::Tensor &labels, const FrozenRidgeHead &head);

} // namespace objective_diagnostic
} // namespace embedding::encoders::raw_patch_bottleneck_mae
