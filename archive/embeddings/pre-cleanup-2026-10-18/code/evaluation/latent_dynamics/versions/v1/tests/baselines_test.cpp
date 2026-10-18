// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include "cuwacunu/evaluation/latent_dynamics/v1/baselines.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace ld = cuwacunu::evaluation::latent_dynamics::v1;
namespace {
int checks = 0;
void check(bool value, const char* message) {
  ++checks; if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F&& f, const char* message) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  check(rejected, message);
}
ld::ArrayBank fixture() {
  ld::ArrayBank b; b.coordinate_id = "synthetic-frozen-target-v1";
  b.interval_ms = 10; b.window_ticks = 4;
  std::vector<double> values;
  for (const auto start : {0, 100, 200}) {
    for (int64_t i = start; i < start + 40; ++i) {
      ld::ArrayRow r;
      r.row_id = i; r.endpoint_ms = i * 10; r.available_ms = i * 10 + 2;
      r.support_begin_ms = (i - 3) * 10; r.support_end_ms = i * 10;
      r.role = start == 0 ? ld::Role::train : start == 100 ? ld::Role::validation : ld::Role::test;
      b.rows.push_back(r);
      values.push_back(0.1 * i + std::sin(0.3 * i));
      values.push_back(0.01 * i * i - std::cos(0.1 * i));
    }
  }
  b.values = torch::tensor(values, torch::kFloat64).reshape({120, 2});
  b.valid = torch::ones({120}, torch::kBool); return b;
}
std::vector<ld::Association> pairs(int64_t start, int64_t end) {
  std::vector<ld::Association> out;
  for (auto i = start; i < end; ++i) {
    for (const auto h : {1, 4, 8}) {
      ld::Association p;
      p.pair_id = i * 10 + h; p.origin_row_id = i; p.target_row_id = i + h;
      p.horizon_ticks = h; p.decision_ms = i * 10 + 3;
      p.history_row_ids = {i - 3, i - 2, i - 1, i};
      p.actual_overlap_fraction = h >= 4 ? 0 : 0.75;
      p.dependence_group = "synthetic-block-" + std::to_string(i / 8);
      out.push_back(p);
    }
  }
  return out;
}
ld::MetricSpec metric() {
  return {torch::tensor({1.0, 2.0}, torch::kFloat64),
          torch::tensor({1.0, 3.0}, torch::kFloat64), 1e-12};
}
} // namespace
int main(int argc, char** argv) {
  try {
    check(argc == 2, "supply exclusive temporary archive directory");
    const std::filesystem::path temporary(argv[1]);
    check(std::filesystem::create_directory(temporary), "temporary directory already exists");
    torch::set_num_threads(1);
    auto bank = fixture();
    auto train = pairs(6, 31), validation = pairs(106, 131), test = pairs(206, 231);
    ld::BaselineRecipe recipe; recipe.fit_cutoff_ms = 400; recipe.train_begin_ms = 0;
    recipe.train_end_ms = 390; recipe.history_size = 4; recipe.ridge_penalty = 1;
    auto all = train; all.insert(all.end(), validation.begin(), validation.end());
    all.insert(all.end(), test.begin(), test.end());
    const auto model = ld::fit_baselines(bank, all, recipe);
    check(model.horizons.size() == 3 && model.excluded_rows == 150, "held-out roles entered TRAIN fit");
    for (const auto& h : model.horizons) check(h.training_rows == 25, "wrong fitting count");
    auto changed = bank; changed.values = bank.values.clone();
    changed.values.slice(0, 40).fill_(1e10);
    const auto changed_model = ld::fit_baselines(changed, all, recipe);
    for (size_t i = 0; i < model.horizons.size(); ++i) {
      const auto& a = model.horizons[i]; const auto& b = changed_model.horizons[i];
      check(torch::equal(a.target_mean, b.target_mean) && torch::equal(a.input_mean, b.input_mean) &&
            torch::equal(a.input_scale, b.input_scale) && torch::equal(a.ridge_weight, b.ridge_weight),
            "held-out values altered normalizer or fit");
    }
    // A target may have its close key inside TRAIN yet mature after fitting.
    auto late = bank; late.rows[14].available_ms = 401;
    auto late_pairs = std::vector<ld::Association>{train[2]}; // origin6+h8 ->14.
    const auto late_model = ld::fit_baselines(late, late_pairs, recipe);
    check(late_model.horizons.empty() && late_model.excluded_rows == 1, "late target entered fit");
    // Full raw-window support must be inside TRAIN, even if endpoint IDs differ.
    auto crossing = bank; crossing.rows[3].support_begin_ms = -1;
    const auto crossing_model = ld::fit_baselines(crossing, {train[0]}, recipe);
    check(crossing_model.horizons.empty(), "cross-role raw support entered fit");
    const auto predictions = ld::predict_baselines(bank, validation, model);
    check(predictions.size() == 3 && predictions[2].method == "ridge-k4", "missing core baseline");
    for (const auto& p : predictions) check(p.valid.all().item<bool>(), "valid validation forecast rejected");
    const auto persistence_score = ld::score_predictions(bank, predictions[1], metric());
    for (const auto& s : persistence_score.horizons) {
      check(s.supported && s.skill_supported && s.scored == 25 && s.skill == 0,
            "persistence must have exactly zero same-row skill");
    }
    // Target-side missingness must not change an already issued forecast.
    auto future_missing = bank; future_missing.valid = bank.valid.clone();
    future_missing.values = bank.values.clone(); future_missing.valid[54] = false;
    future_missing.values[54].fill_(std::numeric_limits<double>::quiet_NaN());
    auto one_pair = std::vector<ld::Association>{validation[2]};
    auto before = ld::predict_baselines(bank, one_pair, model);
    auto after = ld::predict_baselines(future_missing, one_pair, model);
    for (size_t i = 0; i < before.size(); ++i) {
      check(torch::equal(before[i].values, after[i].values) && torch::equal(before[i].valid, after[i].valid),
            "future validity or values influenced forecast");
      const auto unavailable = ld::score_predictions(future_missing, after[i], metric());
      check(!unavailable.horizons[0].supported && unavailable.horizons[0].target_valid == 0,
            "missing target treated as observed zero");
    }
    const auto not_mature = ld::score_predictions(bank, before[2], metric(), 1100);
    check(!not_mature.horizons[0].supported &&
          (not_mature.target_reasons[0].item<int64_t>() & ld::target_not_mature), "unmatured target scored");
    auto invalid = one_pair; invalid[0].pair_id = 99999; invalid[0].history_row_ids[0] = -100;
    invalid[0].ledger_input_reasons = 0xf0; invalid[0].ledger_target_reasons = 0xe0;
    auto invalid_prediction = ld::predict_baselines(bank, invalid, model)[2];
    check(!invalid_prediction.valid[0].item<bool>() && invalid_prediction.associations.size() == 1 &&
          (invalid_prediction.reasons[0].item<int64_t>() & ld::missing_input), "invalid association lost");
    // Targets can be in another role or at a wrong horizon; neither is scored.
    auto wrong = one_pair; wrong[0].target_row_id = 206;
    auto wrong_prediction = before[2]; wrong_prediction.associations = wrong;
    const auto wrong_score = ld::score_predictions(bank, wrong_prediction, metric());
    check((wrong_score.target_reasons[0].item<int64_t>() & ld::role_mismatch) &&
          (wrong_score.target_reasons[0].item<int64_t>() & ld::target_grid_mismatch), "wrong target association accepted");
    auto k1 = recipe; k1.history_size = 1;
    const auto model1 = ld::fit_baselines(bank, train, k1);
    const auto predictions1 = ld::predict_baselines(bank, invalid, model1);
    check(predictions1[2].valid[0].item<bool>() && predictions1[2].method == "ridge-k1",
          "K1 must use the final origin only");
    // Enforce origin availability and real candle spacing; never compress gaps.
    auto delayed = bank; delayed.rows[46].available_ms = 1064;
    check(!ld::predict_baselines(delayed, one_pair, model)[2].valid[0].item<bool>(), "unavailable origin predicted");
    auto gap = one_pair; gap[0].history_row_ids[1] = 101;
    check(!ld::predict_baselines(bank, gap, model)[2].valid[0].item<bool>(), "history gap silently compressed");
    auto old_origin = one_pair; old_origin[0].decision_ms = 399;
    check(!ld::predict_baselines(bank, old_origin, model)[2].valid[0].item<bool>(), "fit after decision used");
    // Near-zero same-row persistence denominators produce N/A, not infinite skill.
    auto constant = bank; constant.values = torch::ones_like(bank.values);
    const auto cm = ld::fit_baselines(constant, train, recipe);
    const auto cp = ld::predict_baselines(constant, validation, cm);
    for (const auto& s : ld::score_predictions(constant, cp[2], metric()).horizons)
      check(s.supported && !s.skill_supported && std::isnan(s.skill), "zero-denominator skill not N/A");
    // Reference-normalized arithmetic and pair weights are checked analytically.
    auto manual = before[2]; manual.values = bank.values[54].to(torch::kFloat64).unsqueeze(0) +
        torch::tensor({2.0, 4.0}, torch::kFloat64);
    const auto analytic = ld::score_predictions(bank, manual, metric());
    check(std::abs(analytic.horizons[0].normalized_mse - 4.0) < 1e-12 &&
          std::abs(analytic.horizons[0].native_mae - 3.5) < 1e-12, "weighted fixed-reference metric incorrect");
    auto weighted = manual;
    weighted.associations.push_back(weighted.associations.front());
    weighted.associations.back().pair_id += 100000; weighted.associations.back().sample_weight = 3;
    weighted.values = torch::cat({manual.values, bank.values[54].to(torch::kFloat64).unsqueeze(0) +
        torch::tensor({4.0, 8.0}, torch::kFloat64)}, 0);
    weighted.valid = torch::ones({2}, torch::kBool); weighted.reasons = torch::zeros({2}, torch::kInt64);
    const auto weighted_score = ld::score_predictions(bank, weighted, metric());
    check(std::abs(weighted_score.horizons[0].normalized_mse - 13.0) < 1e-12 &&
          std::abs(weighted_score.horizons[0].native_mae - 6.125) < 1e-12, "pair weights lost in metric");
    auto weighted_fit_pairs = std::vector<ld::Association>{train[0], train[3]};
    weighted_fit_pairs[1].sample_weight = 3;
    const auto weighted_fit = ld::fit_baselines(bank, weighted_fit_pairs, recipe);
    check(torch::allclose(weighted_fit.horizons[0].target_mean,
          (bank.values[7] + 3 * bank.values[8]) / 4, 0, 1e-14), "TRAIN target mean ignores pair weights");
    auto partial = bank; partial.rows[54].complete_support = false; partial.rows[54].observed_fraction = 0.5;
    const auto partial_score = ld::score_predictions(partial, before[2], metric());
    check(partial_score.strata[0].support == "partial" && partial_score.strata[0].overlap == "none",
          "support/overlap strata lost");
    const auto div = ld::diversity(bank, torch::ones({2}, torch::kFloat64));
    check(div.supported && div.valid_rows == 120 && div.covariance_rank_bound == 2 &&
          div.effective_rank >= 1 && div.effective_rank <= 2.000001, "diversity diagnostic incorrect");
    const auto collapsed = ld::diversity(constant, torch::ones({2}, torch::kFloat64));
    check(collapsed.native_std.eq(0).all().item<bool>() && collapsed.effective_rank == 0,
          "collapsed bank hidden by normalization");
    const auto retain = ld::retention(bank.values, bank.values, bank.valid);
    check(retain.supported && retain.mse.eq(0).all().item<bool>() && retain.r2.eq(1).all().item<bool>(),
          "saved-probe retention metric incorrect");
    // Typed immutable publication includes invalids, all associations and strings.
    const auto prediction_path = (temporary / "predictions.pt").string();
    ld::save_predictions(prediction_path, invalid_prediction);
    const auto loaded = ld::load_predictions(prediction_path);
    check(torch::equal(loaded.valid, invalid_prediction.valid) && torch::equal(loaded.reasons, invalid_prediction.reasons) &&
          loaded.associations[0].history_row_ids == invalid[0].history_row_ids &&
          loaded.associations[0].ledger_input_reasons == 0xf0 &&
          loaded.associations[0].ledger_target_reasons == 0xe0 &&
          loaded.associations[0].dependence_group == invalid[0].dependence_group,
          "prediction archive lost invalid association metadata");
    rejects([&] { ld::save_predictions(prediction_path, invalid_prediction); }, "archive overwritten");
    const auto model_path = (temporary / "model.pt").string(); ld::save_model(model_path, model);
    const auto loaded_model = ld::load_model(model_path);
    check(torch::equal(ld::predict_baselines(bank, validation, loaded_model)[2].values, predictions[2].values),
          "typed model roundtrip changed forecast");
    rejects([&] { ld::load_model(prediction_path); }, "prediction accepted as fitted model");
    auto mismatched = bank; mismatched.coordinate_id = "later-updated-target-head";
    rejects([&] { ld::score_predictions(mismatched, before[2], metric()); }, "mixed target coordinates scored");
    rejects([&] { ld::predict_baselines(mismatched, validation, model); }, "mixed model coordinates predicted");
    std::cout << "latent-dynamics-v1 synthetic baseline checks=" << checks
              << "; zero encoder calls, zero market quality fits\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
