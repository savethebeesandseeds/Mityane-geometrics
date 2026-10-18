// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
#include "cuwacunu/evaluation/latent_dynamics/v1/risk_readouts.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
namespace ld = cuwacunu::evaluation::latent_dynamics::v1;
namespace {
int checks = 0;
void check(bool yes, const char* why) { ++checks; if (!yes) throw std::runtime_error(why); }
template<class F> void rejects(F&& f, const char* why) {
  bool rejected = false; try { f(); } catch (const std::exception&) { rejected = true; } check(rejected, why);
}
}
int main(int argc, char** argv) {
  try {
    check(argc == 2, "supply exclusive temporary directory");
    const std::filesystem::path path(argv[1]); check(std::filesystem::create_directory(path), "directory already exists");
    torch::set_num_threads(1);
    constexpr int64_t n = 48, l = 2;
    ld::ObservableTargets t; t.identity = "synthetic-observables-formulas-v1";
    t.names = {"dispersion", "downside"}; t.units = {"log-return", "log-return"};
    t.origins.row_ids = torch::arange(n, torch::kInt64);
    t.origins.roles = torch::zeros({n}, torch::kInt64); t.origins.roles.slice(0, 24).fill_(1);
    t.origins.decision_ms = t.origins.row_ids * 10 + 5;
    const auto tick = t.origins.row_ids.to(torch::kFloat64);
    const auto x = torch::stack({tick, torch::sin(tick * 0.3)}, 1);
    t.values = torch::stack({0.5 * tick + 2 * torch::sin(tick * 0.3), -tick + 0.3 * torch::sin(tick * 0.3)}, 1);
    t.valid = torch::ones({n, l}, torch::kBool);
    t.support_begin_ms = (t.origins.row_ids * 10 + 10).unsqueeze(1).repeat({1, l});
    t.support_end_ms = (t.origins.row_ids * 10 + 20).unsqueeze(1).repeat({1, l});
    t.available_ms = t.support_end_ms + 2; t.available_ms[20][1] = 500;
    ld::FeatureArm a; a.name = "e-history"; a.identity = "frozen-encoder-e-k4";
    a.information_contract = "K4 causal history, stride1, one origin";
    a.values = x; a.valid = torch::ones({n}, torch::kBool);
    a.origin_row_ids = t.origins.row_ids; a.available_ms = t.origins.row_ids * 10 + 2;
    a.support_begin_ms = t.origins.row_ids * 10 - 30; a.support_end_ms = t.origins.row_ids * 10;
    a.artifact_fit_cutoff_ms = 200;
    auto b = a; b.name = "z-plus-forecast"; b.identity = "frozen-state-dynamics-k4";
    b.information_contract = "same K4 causal history plus frozen deterministic origin forecast";
    b.values = torch::cat({x, x * 2}, 1); b.valid = a.valid.clone(); b.valid[5] = false;
    std::vector<ld::FeatureArm> arms{a, b};
    ld::UtilityRecipe r; r.fit_cutoff_ms = 240; r.train_begin_ms = 0; r.train_end_ms = 230;
    const auto model = ld::fit_risk_readouts(t, arms, r);
    check(model.common_training_rows == std::vector<int64_t>({18, 17}), "common TRAIN masks/maturity/support incorrect");
    for (const auto& arm : model.arms) {
      check(arm.heads[0].training_rows == 18 && arm.heads[1].training_rows == 17, "arms fit different target populations");
    }
    auto altered = t; altered.values = t.values.clone(); altered.values.slice(0, 24).fill_(1e9);
    auto changed_arms = arms;
    for (auto& arm : changed_arms) { arm.values = arm.values.clone(); arm.values.slice(0, 24).fill_(1e8); }
    const auto changed = ld::fit_risk_readouts(altered, changed_arms, r);
    for (size_t k = 0; k < model.arms.size(); ++k) for (int64_t j = 0; j < l; ++j) {
      const auto& old = model.arms[k].heads[j]; const auto& now = changed.arms[k].heads[j];
      check(torch::equal(old.input_mean, now.input_mean) && torch::equal(old.input_scale, now.input_scale) &&
            torch::equal(old.weights, now.weights) && old.target_mean == now.target_mean && old.target_scale == now.target_scale,
            "heldout observations changed fitted head or reference");
    }
    const auto predictions = ld::predict_risk_readouts(t.origins, arms, model);
    check(!predictions[0].valid.slice(0, 0, 24).any().item<bool>() &&
          predictions[0].valid.slice(0, 24).all().item<bool>(), "readout fit used before decision");
    const auto scores = ld::score_risk_readouts(t, predictions, model, ld::Role::validation);
    check(scores.size() == 4, "missing observable/arm comparisons");
    for (const auto& s : scores) check(s.supported && s.common_scored == 24 && s.coverage == 1 &&
        std::isfinite(s.mse) && s.skill_supported && s.train_mean_skill > 0.9,
        "OOS synthetic linear readout failed to improve over the TRAIN-mean control");
    // Reject one origin for only one arm: all reported arms use the same remainder.
    auto missing = predictions; missing[1].valid = missing[1].valid.clone();
    missing[1].valid[30][0] = false;
    const auto matched = ld::score_risk_readouts(t, missing, model, ld::Role::validation);
    check(matched[0].common_scored == 23 && matched[2].common_scored == 23 &&
          matched[1].common_scored == 24 && matched[3].common_scored == 24, "favorable arm-specific rows scored");
    const auto maturity = ld::score_risk_readouts(t, predictions, model, ld::Role::validation, 282);
    for (const auto& s : maturity) check(s.common_scored == 3, "future labels scored before availability");
    auto reversed = arms; reversed[0].origin_row_ids = arms[0].origin_row_ids.flip({0});
    rejects([&] { ld::fit_risk_readouts(t, reversed, r); }, "different origin order accepted");
    auto future = arms; future[0].artifact_fit_cutoff_ms = 241;
    rejects([&] { ld::fit_risk_readouts(t, future, r); }, "feature artifact fitted after head fit accepted");
    auto unavailable = arms; unavailable[0].available_ms = arms[0].available_ms.clone();
    unavailable[0].available_ms[30] = 9999;
    const auto no_input = ld::predict_risk_readouts(t.origins, unavailable, model);
    check(!no_input[0].valid[30].any().item<bool>() && no_input[1].valid[30].all().item<bool>(),
          "unavailable features influenced forecast");
    rejects([&] { ld::score_risk_readouts(t, predictions, model, ld::Role::train); }, "optimistic TRAIN utility score allowed");
    auto identity = t; identity.identity = "different-label-formula";
    rejects([&] { ld::score_risk_readouts(identity, predictions, model, ld::Role::validation); }, "label formula identity mixed");
    auto no_labels = t; no_labels.valid = torch::zeros_like(t.valid);
    const auto empty_model = ld::fit_risk_readouts(no_labels, arms, r);
    const auto empty = ld::predict_risk_readouts(t.origins, arms, empty_model);
    check(!empty[0].valid.any().item<bool>(), "unfitted observable treated as valid zero");
    ld::save_risk_predictions((path / "predictions.pt").string(), no_input);
    ld::save_risk_readouts((path / "readouts.pt").string(), model);
    const auto loaded = ld::load_risk_readouts((path / "readouts.pt").string());
    const auto loaded_predictions = ld::predict_risk_readouts(t.origins, arms, loaded);
    check(torch::equal(loaded_predictions[0].values, predictions[0].values) &&
          loaded.common_training_rows == model.common_training_rows, "typed risk readout roundtrip changed predictions");
    const auto loaded_archive = ld::load_risk_predictions((path / "predictions.pt").string());
    check(torch::equal(loaded_archive[0].values, no_input[0].values) &&
          torch::equal(loaded_archive[0].valid, no_input[0].valid) &&
          torch::equal(loaded_archive[0].reasons, no_input[0].reasons), "risk prediction association/mask roundtrip failed");
    rejects([&] { ld::load_risk_readouts((path / "predictions.pt").string()); }, "predictions accepted as fitted risk heads");
    rejects([&] { ld::save_risk_predictions((path / "predictions.pt").string(), predictions); }, "risk predictions overwritten");
    rejects([&] { ld::save_risk_readouts((path / "readouts.pt").string(), model); }, "risk fitted heads overwritten");
    std::cout << "latent-risk-readouts-v1 synthetic checks=" << checks
              << "; no encoder calls, no market utility claim\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
