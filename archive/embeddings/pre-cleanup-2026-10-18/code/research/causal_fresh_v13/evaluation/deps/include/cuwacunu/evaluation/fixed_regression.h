// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <string>
#include <vector>

namespace cuwacunu::evaluation {

struct TargetSpec { std::string name, unit; };
struct TargetValues {
  torch::Tensor values; // CPU float64 [N,T], scoring information only.
  torch::Tensor valid;  // CPU bool [N,T].
};
struct TargetRows {
  TargetValues targets;
  torch::Tensor row_ids, group_ids; // CPU int64 [N], exact chronological association.
};
struct FrozenFeatures {
  torch::Tensor values; // CPU float32/64 [N,D], no encoder or compression here.
  torch::Tensor valid;  // CPU bool [N].
  torch::Tensor row_ids; // CPU int64 [N], exact order of corresponding TargetRows.
};
struct RegressionMethod {
  std::string name, training_path, validation_path;
  FrozenFeatures training, validation;
};
struct FixedRegressionRun {
  std::string output_directory, card_sha256, input_manifest_sha256;
  std::string dataset_id{"MKT-HIST-1"};
  int64_t designed_complexity_level{4};
  TargetRows training, validation;
  std::vector<TargetSpec> target_specs;
  std::vector<RegressionMethod> methods;
};
struct RegressionScore {
  int64_t total{0}, target_eligible{0}, valid{0};
  double coverage{0}, mae{0}, rmse{0}, r2{0};
  bool supported{false}, r2_supported{false};
};

// Pure all-past targets from legal raw values/masks [N,C,H,9], registry order.
// Volatility/trend/activity require >=8 eligible values in EACH channel.
// Correlation requires >=2 channel pairs, each with >=8 common adjacent returns
// and population variances >=1e-12. Missing storage cannot influence labels.
TargetValues past_window_targets(const torch::Tensor &values, const torch::Tensor &observed);
std::vector<TargetSpec> past_window_target_specs();
RegressionScore regression_score(const torch::Tensor &prediction,
    const torch::Tensor &target, const torch::Tensor &target_valid,
    const torch::Tensor &prediction_valid);
void save_features(const std::string &path, const FrozenFeatures &features);
FrozenFeatures load_features(const std::string &path);
void save_targets(const std::string &path, const TargetRows &targets);
TargetRows load_targets(const std::string &path);

// Fixed TRAIN-only CPU readouts; no encoder execution. Ridge penalty1 (primal/
// dual solve), tanh16, Adam0.01,100 full-batch masked-MSE updates. Both have T
// outputs. Features use population TRAIN std floor1e-8; targets use shared
// per-target TRAIN mean/std floor1e-8, then predictions invert to original units.
// Repetitions2701/2802/2903 use the historical stream_seed(rep, input_width).
// Deterministic ridge is fit once/method, reused for the three neural fits.
// Outputs are exclusive and persist fitted tensors, predictions and report.
std::string run_fixed_regression(const FixedRegressionRun &run);

} // namespace cuwacunu::evaluation
