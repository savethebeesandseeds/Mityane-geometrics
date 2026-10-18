// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/data.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::evaluation {

// Tensor-only CPU contracts. No encoder, classifier construction, PCA,
// optimizer or autodiff is part of this saved-evidence analysis.
struct SavedFeatureFit {
  torch::Tensor feature_mean, feature_scale;
  torch::Tensor ridge_mean, ridge_scale, ridge_weights, ridge_intercept;
  torch::Tensor tiny_mean, tiny_scale, tiny_w1, tiny_b1, tiny_w2, tiny_b2;
};
struct SavedFeaturePrediction {
  torch::Tensor valid, ridge, tiny, probe_input;
  torch::Tensor ridge_logits, tiny_hidden_preactivation, tiny_logits;
};
struct SavedFeatureAnalysis {
  torch::Tensor features, valid, labels;
  std::vector<std::string> source_ids;
  SavedFeatureFit fit;
  SavedFeaturePrediction saved;
};
struct SavedFeatureGeometry {
  torch::Tensor coordinate_mean, population_sd, row_norm, centered_row_norm;
  double pooled_label_mean_distance{0};
};
struct SavedFeatureReplay {
  torch::Tensor outer_input, ridge_input, tiny_input;
  torch::Tensor ridge_logits, tiny_hidden_preactivation, tiny_logits;
  torch::Tensor ridge_signed_margin, tiny_signed_margin;
  torch::Tensor pair_row0, pair_row1, pair_valid;
  std::vector<std::string> pair_source_ids;
  torch::Tensor served_pair_distance, outer_pair_distance, ridge_pair_distance;
  torch::Tensor served_pair_midpoint_norm, outer_pair_midpoint_norm, ridge_pair_midpoint_norm;
  SavedFeatureGeometry served_geometry, outer_geometry, ridge_geometry;
  torch::Tensor ridge_pair_order_margin, tiny_pair_order_margin;
  std::string json;
};

// Saved classes are checked against THEIR saved logits exactly. Independently
// replayed arithmetic uses the predeclared absolute+relative tolerance; it
// never rewrites a saved class near a tie. Statistics use valid TRAIN rows,
// population SD, constant/near-zero threshold1e-12 and linear(n-1)p quantiles
// p={0,.05,.25,.5,.75,.95,1}. Zero norms are retained, never divided/repaired.
SavedFeatureReplay replay_saved_feature_analysis(const SavedFeatureAnalysis &input,
                                                double absolute_tolerance = 2e-9,
                                                double relative_tolerance = 2e-9);

struct SavedTrainingTraceRow {
  int64_t attempted{0}, completed{0}, target_cells{0};
  double loss{0}, gradient_norm{0};
};
struct SavedTrainingTrace {
  int64_t attempted{0}, completed{0}, sampled_rows{0}, batch_size{8};
  std::vector<SavedTrainingTraceRow> rows;
};
// Requires every unskipped absolute update exactly once. Blocks describe
// sampled reconstruction loss/gradient-norm traces, not fixed-query MAE.
std::string saved_training_trace_json(const SavedTrainingTrace &trace,
                                     int64_t block_width = 128);

struct SavedFeatureReliabilityInput {
  std::string id, display_tag;
  uint64_t master_seed{0};
  std::string controlled_training_path, native_training_path, encoder_progress_path;
  std::array<std::string, 3> fit_paths, training_prediction_paths;
};
struct SavedFeatureReliabilityComparison {
  std::string id, candidate_id, comparator_id;
};
struct SavedFeatureReliabilityRun {
  std::string output_directory, source_fingerprint, card_sha256;
  int64_t training_rows{256}, native_width{32}, expected_updates{512}, batch_size{8};
  int64_t expected_parameter_count{225805};
  input_shape_t shape{3, 32, 3, torch::kFloat64, torch::kCPU};
  std::vector<SavedFeatureReliabilityInput> inputs;
  std::vector<SavedFeatureReliabilityComparison> comparisons;
};

// Caller owns complete role/path/SHA admission BEFORE calling. This routine
// loads only the explicitly declared TRAIN paths and writes a new directory.
// It verifies legal hidden storage, source/label association, saved TRAIN map
// statistics, all three fixed heads and complete trace counts; persisted CPU
// arrays allow independent saved-arithmetic replay. No held-out input exists.
std::string run_saved_feature_reliability(const SavedFeatureReliabilityRun &run);

} // namespace embedding::evaluation
