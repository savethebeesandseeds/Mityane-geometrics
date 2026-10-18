// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include "cuwacunu/config/document.h"
#include "cuwacunu/latent_market/artifacts.h"
#include "cuwacunu/latent_market/arrays.h"
#include "cuwacunu/latent_market/ledger_io.h"
#include "cuwacunu/evaluation/latent_dynamics/v1/risk_readouts.h"
#include <iostream>
#include <map>
#include <tuple>

namespace lm = cuwacunu::latent_market::v1;
namespace ev = cuwacunu::evaluation::latent_dynamics::v1;
namespace fs = std::filesystem;
namespace {
std::string text(torch::serialize::InputArchive& a, const std::string& key) {
  torch::IValue value; a.read(key, value); TORCH_CHECK(value.isString(), "Invalid state bank identity: ", key);
  return value.toStringRef();
}
torch::Tensor array(torch::serialize::InputArchive& a, const std::string& key) {
  torch::Tensor value; a.read(key, value, true); return value;
}
void shape(const torch::Tensor& t, torch::ScalarType type, torch::IntArrayRef size, const char* name) {
  TORCH_CHECK(t.defined() && t.device().is_cpu() && t.scalar_type() == type && t.sizes() == size,
              "Invalid saved-array field: ", name);
}
void check_forecast(const ev::PredictionSet& p, const std::vector<ev::Association>& a,
                    const std::string& coordinates, int64_t fit, int64_t spacing) {
  TORCH_CHECK(p.coordinate_id == coordinates && p.associations.size() == a.size() &&
      p.fit_cutoff_ms == fit && p.history_size == 4 && p.spacing_ticks == spacing, "Forecast identity/recipe mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    const auto& x = p.associations[i]; const auto& y = a[i];
    TORCH_CHECK(x.pair_id == y.pair_id && x.origin_row_id == y.origin_row_id && x.target_row_id == y.target_row_id &&
        x.history_row_ids == y.history_row_ids && x.horizon_ticks == y.horizon_ticks && x.decision_ms == y.decision_ms &&
        x.input_eligible == y.input_eligible && x.target_eligible == y.target_eligible &&
        x.ledger_input_reasons == y.ledger_input_reasons && x.ledger_target_reasons == y.ledger_target_reasons &&
        x.dependence_group == y.dependence_group, "Forecast association differs from admitted ledger");
  }
}
} // namespace

int main(int argc, char** argv) try {
  TORCH_CHECK(argc == 3, "usage: latent_market_v1_utility UTILITY.conf NEW_OUTPUT_LEAF");
  torch::set_num_threads(1); torch::set_num_interop_threads(1); torch::NoGradGuard guard;
  const fs::path settings = argv[1], output = argv[2]; const auto settings_hash = lm::sha256_file(settings);
  const auto doc = cuwacunu::config::Document::read(settings); doc.reject_unknown_blocks({"FUTURE_RISK_UTILITY_V1"});
  const auto& b = doc.block("FUTURE_RISK_UTILITY_V1");
  b.reject_unknown_keys({"CARD","CARD_SHA256","SOURCE_CLOSURE","SOURCE_CLOSURE_SHA256",
      "TIMELINE","TIMELINE_SHA256","PAIRS","PAIRS_SHA256","FUTURE_RISK","FUTURE_RISK_SHA256",
      "STATE_BANK","STATE_BANK_SHA256","STATE_PREDICTIONS","STATE_PREDICTIONS_SHA256",
      "E_PREDICTIONS","E_PREDICTIONS_SHA256","E_MODEL","E_MODEL_SHA256","FIT_CUTOFF_MS",
      "SCORING_CUTOFF_MS","REPRESENTATION_FIT_CUTOFF_MS","RIDGE_PENALTY","STD_FLOOR"});
  std::vector<lm::ArtifactPin> admitted;
  const auto pin = [&](const std::string& key) {
    const auto path = doc.resolve_path(b.string(key)); const auto hash = b.string(key + "_SHA256");
    lm::verify_sha256(path, hash); admitted.push_back({path, hash}); return path;
  };
  const auto card = pin("CARD"), closure = pin("SOURCE_CLOSURE");
  const auto source_pins = lm::read_pins(closure); lm::require_pinned(source_pins, __FILE__);
  const auto timeline_path = pin("TIMELINE"), pairs_path = pin("PAIRS"), risk_path = pin("FUTURE_RISK");
  const auto state_path = pin("STATE_BANK"), state_prediction_path = pin("STATE_PREDICTIONS");
  const auto e_prediction_path = pin("E_PREDICTIONS"), e_model_path = pin("E_MODEL");
  const auto timeline = lm::load_timeline(timeline_path);
  const auto risk = lm::load_risk_targets(risk_path);
  const auto pairs = lm::load_pairs(pairs_path, timeline);
  TORCH_CHECK(pairs.spec.history_steps == 4 && !pairs.rows.empty(), "Initial common-utility recipe requires K4 origins");
  TORCH_CHECK(risk.timeline_id == timeline.identity.artifact_id && risk.source_hash == timeline.identity.source_hash &&
      risk.interval_ms == timeline.interval_ms && risk.horizons_ticks == pairs.spec.future_label_ticks,
      "Future-risk/timeline/source/horizon identity mismatch");
  for (const auto& row : timeline.rows) TORCH_CHECK(row.role != lm::Role::test, "TEST timeline requires a separate confirmation protocol");
  for (const auto& row : risk.rows) TORCH_CHECK(row.role != lm::Role::test, "TEST labels are not admitted by this development runner");
  ev::UtilityRecipe recipe; recipe.fit_cutoff_ms = b.integer("FIT_CUTOFF_MS");
  recipe.ridge_penalty = b.number("RIDGE_PENALTY"); recipe.std_floor = b.number("STD_FLOOR");
  const auto representation_fit = b.integer("REPRESENTATION_FIT_CUTOFF_MS"), scoring_cutoff = b.integer("SCORING_CUTOFF_MS");
  bool have_train = false;
  for (const auto& r : pairs.roles) {
    TORCH_CHECK(r.role != lm::Role::test, "TEST role is not opened by this runner");
    if (r.role == lm::Role::train) { recipe.train_begin_ms = r.begin_ms; recipe.train_end_ms = r.end_ms; have_train = true; }
    else TORCH_CHECK(r.begin_ms > recipe.fit_cutoff_ms, "Utility fit must precede held-out origins");
  }
  TORCH_CHECK(have_train && timeline.identity.encoder_fit_cutoff_ms <= representation_fit &&
      representation_fit <= recipe.fit_cutoff_ms && recipe.fit_cutoff_ms <= scoring_cutoff, "Invalid artifact/head/scoring chronology");
  const auto native = lm::array_bank(timeline);
  const auto associations = lm::associations(timeline, pairs);
  const auto e_prediction = ev::load_predictions(e_prediction_path.string());
  const auto state_prediction = ev::load_predictions(state_prediction_path.string());
  const auto e_model = ev::load_model(e_model_path.string());
  TORCH_CHECK(e_model.coordinate_id == native.coordinate_id && e_model.recipe.history_size == 4 &&
      e_model.recipe.spacing_ticks == pairs.spec.spacing_ticks && e_model.recipe.fit_cutoff_ms == representation_fit &&
      e_model.width == static_cast<int64_t>(timeline.embedding_width) && e_model.interval_ms == timeline.interval_ms &&
      e_model.window_ticks == timeline.window_ticks, "Direct-e fitted model identity/geometry mismatch");
  check_forecast(e_prediction, associations, native.coordinate_id, representation_fit, pairs.spec.spacing_ticks);
  TORCH_CHECK(e_prediction.method == "ridge-k4", "Direct-e forecast must be the predeclared K4 ridge control");
  torch::serialize::InputArchive state; state.load_from(state_path.string(), torch::kCPU);
  TORCH_CHECK(text(state, "artifact_kind") == "latent_market_state_bank_v1" &&
      text(state, "timeline_sha256") == b.string("TIMELINE_SHA256") &&
      text(state, "pair_ledger_sha256") == b.string("PAIRS_SHA256"), "State bank/timeline/pair binding mismatch");
  const auto coordinates = text(state, "target_coordinate_id"), checkpoint_hash = text(state, "checkpoint_sha256");
  TORCH_CHECK(coordinates == lm::target_state_coordinate_id(timeline.identity.artifact_id, checkpoint_hash),
              "State target coordinate identity mismatch");
  check_forecast(state_prediction, associations, coordinates, representation_fit, pairs.spec.spacing_ticks);
  TORCH_CHECK(state_prediction.method.rfind("state-", 0) == 0, "State forecasts must come from the admitted frozen dynamics arm");
  const auto online = array(state, "online"), target = array(state, "target"), state_valid = array(state, "valid");
  const auto endpoints = array(state, "endpoint_ms"), fit = array(state, "fit_cutoff_ms");
  const auto fitted_forecasts = array(state, "fitting_forecasts"), fitted_valid = array(state, "fitting_forecast_valid");
  const auto n = static_cast<int64_t>(pairs.rows.size()), q = static_cast<int64_t>(pairs.spec.horizons_ticks.size());
  TORCH_CHECK(online.dim() == 2 && online.size(1) > 0, "State bank geometry missing"); const auto d = online.size(1);
  const auto rows = static_cast<int64_t>(timeline.rows.size()), e = static_cast<int64_t>(timeline.embedding_width);
  shape(e_prediction.values, torch::kFloat64, {n*q, e}, "direct-e issued forecasts");
  shape(state_prediction.values, torch::kFloat64, {n*q, d}, "state issued forecasts");
  shape(online, torch::kFloat64, {rows, d}, "online"); shape(target, torch::kFloat64, {rows, d}, "target");
  shape(state_valid, torch::kBool, {rows}, "state valid"); shape(endpoints, torch::kInt64, {rows}, "state endpoints");
  shape(fit, torch::kInt64, {1}, "representation fit cutoff");
  shape(fitted_forecasts, torch::kFloat64, {n, q, d}, "retrospective fitting forecasts");
  shape(fitted_valid, torch::kBool, {n}, "retrospective fitting forecast validity");
  TORCH_CHECK(fit[0].item<int64_t>() == representation_fit && torch::equal(state_valid, native.valid) &&
      torch::isfinite(online).all().item<bool>() && torch::isfinite(target).all().item<bool>() &&
      torch::isfinite(fitted_forecasts).all().item<bool>(), "State bank values/masks/fit mismatch");
  for (int64_t i = 0; i < rows; ++i) TORCH_CHECK(endpoints[i].item<int64_t>() == timeline.rows[i].endpoint_ms, "State row order mismatch");

  ev::ObservableTargets targets; targets.identity = b.string("FUTURE_RISK_SHA256") + ":" + risk.formula_id;
  targets.origins.row_ids = torch::arange(1, n + 1, torch::kInt64);
  targets.origins.roles = torch::zeros({n}, torch::kInt64); targets.origins.decision_ms = torch::zeros({n}, torch::kInt64);
  using LabelKey = std::pair<int64_t, int64_t>;
  std::map<LabelKey, int64_t> labels;
  for (const auto channel : timeline.identity.channel_ids) for (const auto horizon : risk.horizons_ticks) {
    const auto j = static_cast<int64_t>(labels.size()) * 2; labels[{channel, horizon}] = j;
    for (const auto& name : {std::string("dispersion"), std::string("downside")}) {
      targets.names.push_back(name + ":asset=" + std::to_string(channel) + ":h=" + std::to_string(horizon));
      targets.units.push_back(name == "dispersion" ? risk.dispersion_units : risk.downside_units);
    }
  }
  const auto l = static_cast<int64_t>(targets.names.size()); TORCH_CHECK(l > 0, "No declared risk observables");
  targets.values = torch::zeros({n, l}, torch::kFloat64); targets.valid = torch::zeros({n, l}, torch::kBool);
  targets.available_ms = torch::full({n, l}, std::numeric_limits<int64_t>::max(), torch::kInt64);
  targets.support_begin_ms = torch::zeros({n, l}, torch::kInt64); targets.support_end_ms = torch::zeros({n, l}, torch::kInt64);
  std::map<std::string, int64_t> origin_index;
  for (int64_t i = 0; i < n; ++i) {
    const auto& p = pairs.rows[i]; TORCH_CHECK(p.role == lm::Role::train || p.role == lm::Role::validation, "Unadmitted origin role");
    TORCH_CHECK(origin_index.emplace(p.pair_id, i).second, "Duplicate origin identity");
    targets.origins.roles[i] = static_cast<int64_t>(lm::metric_role(p.role)); targets.origins.decision_ms[i] = p.decision_ms;
    for (const auto& [key, j] : labels) for (int64_t k = 0; k < 2; ++k) {
      targets.support_begin_ms[i][j+k] = p.endpoint_ms;
      targets.support_end_ms[i][j+k] = lm::shift_timestamp(p.endpoint_ms, key.second, timeline.interval_ms);
    }
  }
  std::set<std::pair<int64_t, int64_t>> assigned;
  for (const auto& r : risk.rows) {
    const auto found = origin_index.find(r.pair_id); const auto label = labels.find({r.channel_id, r.horizon_ticks});
    TORCH_CHECK(found != origin_index.end() && label != labels.end(), "Risk row has an undeclared origin/asset/horizon");
    const auto i = found->second, j = label->second; const auto& p = pairs.rows[i];
    TORCH_CHECK(assigned.insert({i, j}).second && r.endpoint_ms == p.endpoint_ms && r.decision_ms == p.decision_ms &&
        r.role == p.role, "Risk origin association differs from pair ledger");
    targets.values[i][j] = r.dispersion_1m; targets.values[i][j+1] = r.downside;
    for (int64_t k = 0; k < 2; ++k) {
      targets.valid[i][j+k] = r.eligible;
      targets.available_ms[i][j+k] = r.available_ms.value_or(std::numeric_limits<int64_t>::max());
    }
  }
  TORCH_CHECK(assigned.size() == static_cast<size_t>(n * l / 2), "Missing risk associations must remain explicit ledger rows");

  std::vector<ev::FeatureArm> arms;
  const std::vector<std::string> names{"current_e","history_e","current_z","history_z",
      "e_history+forecast_e","e_history+forecast_z","z_history+forecast_z"};
  const std::vector<int64_t> widths{e, 4*e, d, 4*d, 4*e+q*e, 4*e+q*d, 4*d+q*d};
  for (size_t k = 0; k < names.size(); ++k) {
    ev::FeatureArm a; a.name = names[k];
    a.identity = b.string("TIMELINE_SHA256") + ":" + b.string("STATE_BANK_SHA256") + ":" +
        b.string("E_MODEL_SHA256") + ":" + b.string("E_PREDICTIONS_SHA256") + ":" +
        b.string("STATE_PREDICTIONS_SHA256") + ":" + names[k];
    a.information_contract = "frozen Encoder embeddings / State Representation Network history / Target State Representation Network forecasts;K4;s=" + std::to_string(pairs.spec.spacing_ticks) +
        ";all declared horizons;TRAIN retrospective input-only feature construction;VALIDATION issued forecasts";
    a.artifact_fit_cutoff_ms = (k <= 1) ? timeline.identity.encoder_fit_cutoff_ms : representation_fit;
    if (k == 4) a.artifact_fit_cutoff_ms = representation_fit;
    a.values = torch::zeros({n, widths[k]}, torch::kFloat64); a.valid = torch::zeros({n}, torch::kBool);
    a.origin_row_ids = targets.origins.row_ids; a.available_ms = torch::zeros({n}, torch::kInt64);
    a.support_begin_ms = torch::zeros({n}, torch::kInt64); a.support_end_ms = torch::zeros({n}, torch::kInt64);
    arms.push_back(std::move(a));
  }
  for (int64_t i = 0; i < n; ++i) {
    const auto& p = pairs.rows[i]; const auto current = p.history.back().row_index;
    const auto current_begin = current ? timeline.rows[*current].support.timestamps_ms.front() : p.endpoint_ms;
    bool history_valid = p.input_reasons == lm::no_issue; int64_t begin = p.endpoint_ms, available = p.endpoint_ms;
    auto eh = torch::zeros({4, e}, torch::kFloat64), zh = torch::zeros({4, d}, torch::kFloat64);
    for (int64_t k = 0; k < 4; ++k) {
      if (!p.history[k].row_index) { history_valid = false; continue; }
      const auto index = *p.history[k].row_index; eh[k].copy_(native.values[index]); zh[k].copy_(online[index]);
      begin = std::min(begin, timeline.rows[index].support.timestamps_ms.front()); available = std::max(available, timeline.rows[index].available_ms);
    }
    auto ef = torch::zeros({q, e}, torch::kFloat64), zf = torch::zeros({q, d}, torch::kFloat64);
    bool e_valid = history_valid, z_valid = history_valid;
    for (int64_t j = 0; j < q; ++j) {
      const auto offset = i*q+j;
      if (p.role == lm::Role::train) {
        const auto model = std::find_if(e_model.horizons.begin(), e_model.horizons.end(),
            [&](const auto& h) { return h.horizon_ticks == pairs.spec.horizons_ticks[j]; });
        if (model == e_model.horizons.end()) e_valid = false;
        else ef[j].copy_(((eh.reshape({-1})-model->input_mean)/model->input_scale).matmul(model->ridge_weight)+model->target_mean);
        if (!fitted_valid[i].item<bool>()) z_valid = false; else zf[j].copy_(fitted_forecasts[i][j]);
      } else {
        if (!e_prediction.valid[offset].item<bool>()) e_valid = false; else ef[j].copy_(e_prediction.values[offset]);
        if (!state_prediction.valid[offset].item<bool>()) z_valid = false; else zf[j].copy_(state_prediction.values[offset]);
      }
    }
    const bool current_valid = current && native.valid[*current].item<bool>() && p.history.back().reasons == lm::no_issue;
    if (current) { arms[0].values[i].copy_(native.values[*current]); arms[2].values[i].copy_(online[*current]); }
    arms[1].values[i].copy_(eh.reshape({-1})); arms[3].values[i].copy_(zh.reshape({-1}));
    arms[4].values[i].copy_(torch::cat({eh.reshape({-1}),ef.reshape({-1})}));
    arms[5].values[i].copy_(torch::cat({eh.reshape({-1}),zf.reshape({-1})}));
    arms[6].values[i].copy_(torch::cat({zh.reshape({-1}),zf.reshape({-1})}));
    for (size_t k = 0; k < arms.size(); ++k) {
      arms[k].valid[i] = k == 0 || k == 2 ? current_valid : k == 4 ? e_valid : k >= 5 ? z_valid : history_valid;
      arms[k].support_begin_ms[i] = k == 0 || k == 2 ? current_begin : begin;
      arms[k].support_end_ms[i] = p.endpoint_ms;
      arms[k].available_ms[i] = k == 0 || k == 2 ? (current ? timeline.rows[*current].available_ms : p.endpoint_ms) : available;
    }
  }
  const auto model = ev::fit_risk_readouts(targets, arms, recipe);
  const auto predictions = ev::predict_risk_readouts(targets.origins, arms, model);
  const auto scores = ev::score_risk_readouts(targets, predictions, model, ev::Role::validation, scoring_cutoff);
  lm::new_leaf(output); ev::save_risk_readouts((output/"readouts.pt").string(), model);
  ev::save_risk_predictions((output/"predictions.pt").string(), predictions);
  std::ofstream report(output/"scores.csv"); report<<std::setprecision(17)
      <<"arm,observable,unit,role,total,target_eligible,input_valid,common_scored,coverage,mae,mse,reference_normalized_mse,train_mean_skill\n";
  for (const auto& s : scores) {
    report<<std::quoted(s.arm_name)<<','<<std::quoted(s.observable)<<','<<std::quoted(s.unit)<<",VALIDATION,"
      <<s.total<<','<<s.eligible_targets<<','<<s.valid_inputs<<','<<s.common_scored<<','<<s.coverage<<',';
    if (s.supported) report<<s.mae<<','<<s.mse<<','<<s.reference_normalized_mse; else report<<"N/A,N/A,N/A";
    report<<','; if (s.skill_supported) report<<s.train_mean_skill; else report<<"N/A"; report<<'\n';
  }
  std::ofstream receipt(output/"utility.txt"); receipt<<"latent-market-future-risk-utility-v1\nsettings_sha256 "<<settings_hash
      <<"\ncard_sha256 "<<b.string("CARD_SHA256")<<"\nsource_closure_sha256 "<<b.string("SOURCE_CLOSURE_SHA256")
      <<"\nbinary_sha256 "<<lm::sha256_file("/proc/self/exe")<<"\ntarget_identity "<<targets.identity
      <<"\nrepresentation_fit_cutoff_ms "<<representation_fit<<"\nreadout_fit_cutoff_ms "<<recipe.fit_cutoff_ms
      <<"\nscoring_cutoff_ms "<<scoring_cutoff<<"\nencoder_calls 0\nnetwork_updates 0\nTEST_opened false\n";
  for (size_t j = 0; j < model.names.size(); ++j) receipt<<"common_TRAIN_rows "<<std::quoted(model.names[j])<<' '<<model.common_training_rows[j]<<'\n';
  for (const auto& p : admitted) receipt<<"input_pin "<<p.hash<<' '<<std::quoted(p.path.string())<<'\n';
  receipt<<"TRAIN_forecasts retrospective frozen-model origin-only feature construction; never issued OOS predictions or surprise\n"
      <<"VALIDATION_forecasts issued artifacts; target masks stay target-side\n"
      <<"limitations conventional observable-history controls, equal-capacity controls, forecast placebos, historical retention guards and temporal-block uncertainty remain required by prospective comparisons\n"
      <<"quality_gate no automatic advancement; descriptive common-target utility only; no portfolio claim\n";
  lm::verify_sha256(settings, settings_hash); lm::verify_pins(admitted); lm::verify_pins(source_pins);
  TORCH_CHECK(report.good() && receipt.good(), "Utility artifact write failed");
  std::cout<<"Saved matched future-risk readouts and issued VALIDATION predictions; no automatic quality claim.\n";
  return 0;
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
