// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/evaluation/latent_dynamics/v1/baselines.h"

namespace cuwacunu::evaluation::latent_dynamics::v1 {

struct ObservableOrigins {
  torch::Tensor row_ids, roles, decision_ms; // CPU int64 [N]; Role values.
};
struct ObservableTargets {
  std::string identity; // Label formulas, units, source/split/support identity.
  ObservableOrigins origins;
  std::vector<std::string> names, units;
  torch::Tensor values; // CPU float64 [N,L], unchanged observable units.
  torch::Tensor valid; // CPU bool [N,L]; zero invalid storage is not an observation.
  torch::Tensor available_ms, support_begin_ms, support_end_ms; // int64 [N,L].
};
struct FeatureArm {
  std::string name, identity, information_contract;
  torch::Tensor values; // CPU float64 [N,P], declared e/z/history/forecast features.
  torch::Tensor valid; // CPU bool [N].
  torch::Tensor origin_row_ids, available_ms, support_begin_ms, support_end_ms; // int64 [N].
  int64_t artifact_fit_cutoff_ms{}; // Latest fit cutoff among all feature artifacts.
};
struct UtilityRecipe {
  int64_t fit_cutoff_ms{}, train_begin_ms{}, train_end_ms{};
  double ridge_penalty{1}, std_floor{1e-8}, denominator_epsilon{1e-12};
};
struct ObservableHead {
  int64_t training_rows{};
  torch::Tensor input_mean, input_scale, weights;
  double target_mean{}, target_scale{1};
};
struct FittedFeatureArm {
  std::string name, identity, information_contract;
  int64_t input_width{}, artifact_fit_cutoff_ms{};
  std::vector<ObservableHead> heads;
};
struct RiskReadoutBundle {
  std::string target_identity;
  std::vector<std::string> names, units;
  UtilityRecipe recipe;
  std::vector<FittedFeatureArm> arms;
  std::vector<int64_t> common_training_rows;
};
struct RiskPrediction {
  std::string arm_name, feature_identity, target_identity;
  int64_t fit_cutoff_ms{};
  torch::Tensor values, valid, reasons; // float64/bool/int64 [N,L], input-side only.
  ObservableOrigins origins;
};
enum UtilityReason : int64_t {
  utility_invalid_input = 1LL << 0, utility_input_unavailable = 1LL << 1,
  utility_fit_after_decision = 1LL << 2, utility_unfitted_observable = 1LL << 3
};

// Each observable uses exactly the same mature TRAIN rows for every arm.
// Different input widths are declared; all arms use one fixed ridge recipe.
RiskReadoutBundle fit_risk_readouts(const ObservableTargets&, const std::vector<FeatureArm>&,
                                   const UtilityRecipe&);
// This API receives no future target values, masks or maturity metadata.
std::vector<RiskPrediction> predict_risk_readouts(const ObservableOrigins&,
    const std::vector<FeatureArm>&, const RiskReadoutBundle&);
struct UtilityScore {
  std::string arm_name, observable, unit;
  int64_t total{}, eligible_targets{}, valid_inputs{}, common_scored{};
  double coverage{}, mae{}, mse{}, reference_normalized_mse{};
  double train_mean_skill{std::numeric_limits<double>::quiet_NaN()};
  bool supported{false}, skill_supported{false};
};
// Every arm/observable is scored on the exact same intersection of issued
// predictions and mature target eligibility, never arm-specific favorable rows.
std::vector<UtilityScore> score_risk_readouts(const ObservableTargets&,
    const std::vector<RiskPrediction>&, const RiskReadoutBundle&, Role scoring_role,
    int64_t scoring_cutoff_ms = std::numeric_limits<int64_t>::max());
void save_risk_predictions(const std::string&, const std::vector<RiskPrediction>&);
void save_risk_readouts(const std::string&, const RiskReadoutBundle&);
std::vector<RiskPrediction> load_risk_predictions(const std::string&);
RiskReadoutBundle load_risk_readouts(const std::string&);

} // namespace cuwacunu::evaluation::latent_dynamics::v1
