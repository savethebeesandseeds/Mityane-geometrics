// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../LICENSE.
#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cuwacunu::evaluation::latent_dynamics::v1 {

enum class Role : int64_t { train = 0, validation = 1, test = 2 };
struct ArrayRow {
  int64_t row_id{}, endpoint_ms{}, available_ms{}, support_begin_ms{}, support_end_ms{};
  Role role{Role::train};
  bool complete_support{true}, stale{false};
  double observed_fraction{1};
};
struct ArrayBank {
  torch::Tensor values; // CPU float32/64 [N,D], native coordinates; no encoder calls.
  torch::Tensor valid;  // CPU bool [N]; invalid storage is never used.
  std::vector<ArrayRow> rows;
  std::string coordinate_id; // Frozen embedding or frozen target-G_bar tuple identity.
  int64_t interval_ms{}, window_ticks{};
};
// One association per origin/horizon. history_row_ids remain on the candle grid,
// oldest first. Supply the largest history; K1 takes its final row, K4 its suffix.
// Missing associations are retained using unknown row IDs and explicit flags.
struct Association {
  int64_t pair_id{}, origin_row_id{}, target_row_id{}, horizon_ticks{}, decision_ms{};
  std::vector<int64_t> history_row_ids;
  bool input_eligible{true}, target_eligible{true};
  uint64_t ledger_input_reasons{}, ledger_target_reasons{};
  double sample_weight{1}, actual_overlap_fraction{};
  std::string dependence_group;
};
enum Reason : int64_t {
  ledger_input_invalid = 1LL << 0, missing_input = 1LL << 1,
  invalid_input = 1LL << 2, input_not_available = 1LL << 3,
  history_grid_mismatch = 1LL << 4, role_mismatch = 1LL << 5,
  fit_not_available = 1LL << 6, no_fitted_horizon = 1LL << 7,
  ledger_target_invalid = 1LL << 8, missing_target = 1LL << 9,
  invalid_target = 1LL << 10, target_not_mature = 1LL << 11,
  target_grid_mismatch = 1LL << 12, outside_train_support = 1LL << 13,
  incomplete_support = 1LL << 14
};
struct BaselineRecipe {
  int64_t fit_cutoff_ms{}, train_begin_ms{}, train_end_ms{};
  int64_t history_size{4}, spacing_ticks{1};
  double ridge_penalty{1}, std_floor{1e-8};
  bool require_complete_support{true};
};
struct HorizonModel {
  int64_t horizon_ticks{}, training_rows{};
  double training_weight{};
  torch::Tensor target_mean, input_mean, input_scale, ridge_weight; // CPU float64.
};
struct BaselineModel {
  std::string coordinate_id;
  BaselineRecipe recipe;
  int64_t width{}, interval_ms{}, window_ticks{};
  std::vector<HorizonModel> horizons;
  int64_t submitted_rows{}, excluded_rows{};
};
struct PredictionSet {
  std::string coordinate_id, method;
  int64_t fit_cutoff_ms{}, history_size{}, spacing_ticks{1};
  torch::Tensor values; // CPU float64 [pairs,D]; invalid storage is zero, not observed.
  torch::Tensor valid, reasons; // bool [pairs], int64 [pairs], INPUT-side only.
  std::vector<Association> associations; // Full eligible and invalid association ledger.
};

void validate_bank(const ArrayBank& bank);
BaselineModel fit_baselines(const ArrayBank&, const std::vector<Association>&,
                            const BaselineRecipe&);
// Target validity/masks/values are never consulted to issue a prediction.
// Persistence uses the origin from this same frozen TARGET coordinate bank.
std::vector<PredictionSet> predict_baselines(const ArrayBank&,
    const std::vector<Association>&, const BaselineModel&);

struct MetricSpec {
  torch::Tensor reference_scale; // CPU positive finite float64 [D], frozen TRAIN reference.
  torch::Tensor coordinate_weight; // CPU nonnegative float64 [D], nonzero total.
  double denominator_epsilon{1e-12};
};
struct Score {
  int64_t horizon_ticks{}, total{}, input_valid{}, target_valid{}, scored{};
  double scoring_weight{}, coverage{};
  double normalized_mse{}, native_mae{}, persistence_normalized_mse{};
  double skill{std::numeric_limits<double>::quiet_NaN()};
  bool supported{false}, skill_supported{false};
};
struct StratifiedScore { std::string support, overlap; Score score; };
struct Evaluation {
  std::vector<Score> horizons;
  std::vector<StratifiedScore> strata; // Exact support/overlap counts, no independence claim.
  torch::Tensor target_valid, target_reasons; // bool/int64 [pairs], target-side only.
};
Evaluation score_predictions(const ArrayBank&, const PredictionSet&, const MetricSpec&,
    int64_t scoring_cutoff_ms = std::numeric_limits<int64_t>::max());

struct Diversity {
  int64_t valid_rows{}, width{}, covariance_rank_bound{};
  bool supported{false};
  torch::Tensor native_std, reference_std;
  double effective_rank{}, largest_factor_fraction{};
};
Diversity diversity(const ArrayBank&, const torch::Tensor& frozen_reference_scale);
struct Retention {
  int64_t valid_rows{};
  bool supported{false};
  torch::Tensor mae, mse, r2, r2_valid; // Per unchanged observable coordinate.
};
// Saved fixed-probe outputs only. This function fits nothing and supplies no
// advancement thresholds; all rows/observables retain their declared identities.
Retention retention(const torch::Tensor& prediction, const torch::Tensor& target,
    const torch::Tensor& valid, double denominator_epsilon = 1e-12);

// Typed archives are CPU-only and immutable (exclusive hard-link publication).
// Prediction archives preserve ALL associations, history IDs, ledger reasons,
// dependence groups, target validity flags and sample weights, including invalids.
void save_predictions(const std::string&, const PredictionSet&);
PredictionSet load_predictions(const std::string&);
void save_model(const std::string&, const BaselineModel&);
BaselineModel load_model(const std::string&);

} // namespace cuwacunu::evaluation::latent_dynamics::v1
