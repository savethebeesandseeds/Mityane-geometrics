// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
#include "cuwacunu/evaluation/latent_dynamics/v1/risk_readouts.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace cuwacunu::evaluation::latent_dynamics::v1 {
namespace {
void require(bool yes, const std::string& why) {
  if (!yes) throw std::invalid_argument("latent risk readouts v1: " + why);
}
void tensor(const torch::Tensor& x, torch::ScalarType type, torch::IntArrayRef shape,
            const char* what) {
  require(x.defined() && x.device().is_cpu() && x.scalar_type() == type && x.sizes() == shape, what);
}
int64_t origins(const ObservableOrigins& o) {
  require(o.row_ids.defined() && o.row_ids.dim() == 1, "origin IDs required");
  const auto n = o.row_ids.size(0);
  tensor(o.row_ids, torch::kInt64, {n}, "invalid origin IDs");
  tensor(o.roles, torch::kInt64, {n}, "invalid roles");
  tensor(o.decision_ms, torch::kInt64, {n}, "invalid decisions");
  std::set<int64_t> ids;
  for (int64_t i = 0; i < n; ++i) {
    require(ids.insert(o.row_ids[i].item<int64_t>()).second, "duplicate origin ID");
    const auto role = o.roles[i].item<int64_t>();
    require(role >= static_cast<int64_t>(Role::train) && role <= static_cast<int64_t>(Role::test), "unknown role");
  }
  return n;
}
void targets(const ObservableTargets& t) {
  const auto n = origins(t.origins), l = static_cast<int64_t>(t.names.size());
  require(!t.identity.empty() && l > 0 && t.units.size() == t.names.size(), "target identity/specs required");
  require(std::set<std::string>(t.names.begin(), t.names.end()).size() == t.names.size(), "duplicate observable names");
  for (size_t i = 0; i < t.names.size(); ++i) require(!t.names[i].empty() && !t.units[i].empty(), "empty name/unit");
  tensor(t.values, torch::kFloat64, {n, l}, "invalid observable values");
  tensor(t.valid, torch::kBool, {n, l}, "invalid observable masks");
  tensor(t.available_ms, torch::kInt64, {n, l}, "invalid target availability");
  tensor(t.support_begin_ms, torch::kInt64, {n, l}, "invalid target support begin");
  tensor(t.support_end_ms, torch::kInt64, {n, l}, "invalid target support end");
  require(torch::isfinite(t.values.index({t.valid})).all().item<bool>(), "nonfinite valid observable");
  require(t.support_begin_ms.le(t.support_end_ms).all().item<bool>() &&
          t.support_end_ms.le(t.available_ms).all().item<bool>(), "noncausal target availability");
}
void features(const ObservableOrigins& o, const std::vector<FeatureArm>& arms) {
  const auto n = origins(o);
  require(!arms.empty(), "at least one feature arm required"); std::set<std::string> names;
  for (const auto& a : arms) {
    require(!a.name.empty() && names.insert(a.name).second && !a.identity.empty() && !a.information_contract.empty(),
            "unique arm name, artifact identity and information contract required");
    require(a.values.defined() && a.values.dim() == 2 && a.values.size(1) > 0, "invalid feature shape");
    tensor(a.values, torch::kFloat64, {n, a.values.size(1)}, "invalid feature array");
    tensor(a.valid, torch::kBool, {n}, "invalid feature validity");
    tensor(a.origin_row_ids, torch::kInt64, {n}, "invalid feature origin IDs");
    require(torch::equal(a.origin_row_ids, o.row_ids), "feature origin IDs/order differ from common targets");
    tensor(a.available_ms, torch::kInt64, {n}, "invalid feature availability");
    tensor(a.support_begin_ms, torch::kInt64, {n}, "invalid feature support begin");
    tensor(a.support_end_ms, torch::kInt64, {n}, "invalid feature support end");
    require(a.support_begin_ms.le(a.support_end_ms).all().item<bool>() &&
            a.support_end_ms.le(a.available_ms).all().item<bool>(), "noncausal feature support");
    require(torch::isfinite(a.values.index({a.valid})).all().item<bool>(), "nonfinite valid feature row");
  }
}
void recipe(const UtilityRecipe& r) {
  require(r.train_begin_ms <= r.train_end_ms && std::isfinite(r.ridge_penalty) && r.ridge_penalty > 0 &&
          std::isfinite(r.std_floor) && r.std_floor > 0 && std::isfinite(r.denominator_epsilon) &&
          r.denominator_epsilon >= 0, "invalid readout recipe");
}
torch::Tensor input_eligible(const ObservableOrigins& o, const FeatureArm& a) {
  return a.valid & a.available_ms.le(o.decision_ms) & a.support_end_ms.le(o.decision_ms);
}
void string(torch::serialize::OutputArchive& a, const std::string& key, const std::string& value) {
  a.write(key, torch::tensor(std::vector<uint8_t>(value.begin(), value.end()), torch::kUInt8), true);
}
torch::Tensor read(torch::serialize::InputArchive& a, const std::string& key) {
  torch::Tensor value; a.read(key, value, true); return value;
}
std::string string(torch::serialize::InputArchive& a, const std::string& key) {
  auto value = read(a, key);
  require(value.scalar_type() == torch::kUInt8 && value.dim() == 1, "invalid archived string");
  value = value.contiguous();
  return std::string(reinterpret_cast<const char*>(value.data_ptr<uint8_t>()), value.numel());
}
void publish(const std::string& path, torch::serialize::OutputArchive& a) {
  static std::atomic<uint64_t> serial{0};
  require(!std::filesystem::exists(path), "archive already exists");
  const auto temp = path + ".stage-risk-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
      "-" + std::to_string(serial++);
  try { a.save_to(temp); std::filesystem::create_hard_link(temp, path); std::filesystem::remove(temp); }
  catch (...) { std::filesystem::remove(temp); throw; }
}
} // namespace

RiskReadoutBundle fit_risk_readouts(const ObservableTargets& t, const std::vector<FeatureArm>& arms,
                                   const UtilityRecipe& r) {
  torch::NoGradGuard no_grad; targets(t); features(t.origins, arms); recipe(r);
  RiskReadoutBundle out; out.target_identity = t.identity; out.names = t.names; out.units = t.units; out.recipe = r;
  auto common_input = t.origins.roles.eq(static_cast<int64_t>(Role::train)) & t.origins.decision_ms.le(r.fit_cutoff_ms);
  for (const auto& a : arms) {
    require(a.artifact_fit_cutoff_ms <= r.fit_cutoff_ms, "feature artifact fitted after readout fit");
    common_input = common_input & input_eligible(t.origins, a) & a.support_begin_ms.ge(r.train_begin_ms) &
        a.support_end_ms.le(r.train_end_ms);
    FittedFeatureArm fitted;
    fitted.name = a.name; fitted.identity = a.identity; fitted.information_contract = a.information_contract;
    fitted.input_width = a.values.size(1); fitted.artifact_fit_cutoff_ms = a.artifact_fit_cutoff_ms;
    out.arms.push_back(std::move(fitted));
  }
  for (int64_t j = 0; j < static_cast<int64_t>(t.names.size()); ++j) {
    const auto selected = common_input & t.valid.select(1, j) & t.available_ms.select(1, j).le(r.fit_cutoff_ms) &
        t.support_begin_ms.select(1, j).ge(r.train_begin_ms) & t.support_end_ms.select(1, j).le(r.train_end_ms);
    const auto count = selected.sum().item<int64_t>(); out.common_training_rows.push_back(count);
    for (size_t k = 0; k < arms.size(); ++k) {
      ObservableHead head; head.training_rows = count;
      const auto width = arms[k].values.size(1);
      head.input_mean = torch::zeros({width}, torch::kFloat64);
      head.input_scale = torch::ones({width}, torch::kFloat64); head.weights = torch::zeros({width}, torch::kFloat64);
      if (count) {
        const auto x = arms[k].values.index({selected}), y = t.values.select(1, j).index({selected});
        head.input_mean = x.mean(0);
        head.input_scale = (x - head.input_mean).square().mean(0).sqrt().clamp_min(r.std_floor);
        head.target_mean = y.mean().item<double>();
        head.target_scale = std::max((y - head.target_mean).square().mean().sqrt().item<double>(), r.std_floor);
        const auto a = (x - head.input_mean) / head.input_scale, target = y - head.target_mean;
        if (width > count) head.weights = a.t().matmul(at::linalg_solve(a.matmul(a.t()) +
            r.ridge_penalty * torch::eye(count, a.options()), target.unsqueeze(1))).squeeze(1);
        else head.weights = at::linalg_solve(a.t().matmul(a) + r.ridge_penalty *
            torch::eye(width, a.options()), a.t().matmul(target.unsqueeze(1))).squeeze(1);
        require(torch::isfinite(head.weights).all().item<bool>(), "nonfinite risk ridge fit");
      }
      out.arms[k].heads.push_back(std::move(head));
    }
  }
  return out;
}

std::vector<RiskPrediction> predict_risk_readouts(const ObservableOrigins& o,
    const std::vector<FeatureArm>& arms, const RiskReadoutBundle& model) {
  torch::NoGradGuard no_grad; features(o, arms); recipe(model.recipe);
  require(arms.size() == model.arms.size(), "feature arm count mismatch");
  const auto n = origins(o), l = static_cast<int64_t>(model.names.size());
  std::vector<RiskPrediction> out;
  for (size_t k = 0; k < arms.size(); ++k) {
    const auto& a = arms[k]; const auto& m = model.arms[k];
    require(a.name == m.name && a.identity == m.identity && a.information_contract == m.information_contract &&
        a.values.size(1) == m.input_width && a.artifact_fit_cutoff_ms == m.artifact_fit_cutoff_ms &&
        static_cast<int64_t>(m.heads.size()) == l, "readout/feature identity or geometry mismatch");
    RiskPrediction p; p.arm_name = a.name; p.feature_identity = a.identity;
    p.target_identity = model.target_identity; p.fit_cutoff_ms = model.recipe.fit_cutoff_ms;
    p.origins = o; p.values = torch::zeros({n, l}, torch::kFloat64);
    p.valid = torch::zeros({n, l}, torch::kBool); p.reasons = torch::zeros({n, l}, torch::kInt64);
    for (int64_t i = 0; i < n; ++i) {
      int64_t why = a.valid[i].item<bool>() ? int64_t{0} : static_cast<int64_t>(utility_invalid_input);
      if (a.available_ms[i].item<int64_t>() > o.decision_ms[i].item<int64_t>() ||
          a.support_end_ms[i].item<int64_t>() > o.decision_ms[i].item<int64_t>()) why |= utility_input_unavailable;
      if (std::max(p.fit_cutoff_ms, a.artifact_fit_cutoff_ms) > o.decision_ms[i].item<int64_t>())
        why |= utility_fit_after_decision;
      for (int64_t j = 0; j < l; ++j) {
        auto reasons = why; const auto& h = m.heads[j];
        if (!h.training_rows) reasons |= utility_unfitted_observable;
        p.reasons[i][j] = reasons;
        if (reasons) continue;
        p.values[i][j] = ((a.values[i] - h.input_mean) / h.input_scale).dot(h.weights) + h.target_mean;
        p.valid[i][j] = true;
      }
    }
    require(torch::isfinite(p.values.index({p.valid})).all().item<bool>(), "nonfinite issued risk prediction");
    out.push_back(std::move(p));
  }
  return out;
}

std::vector<UtilityScore> score_risk_readouts(const ObservableTargets& t,
    const std::vector<RiskPrediction>& predictions, const RiskReadoutBundle& model,
    Role scoring_role, int64_t scoring_cutoff) {
  torch::NoGradGuard no_grad; targets(t); recipe(model.recipe);
  require(scoring_role == Role::validation || scoring_role == Role::test,
          "risk utility scoring requires out-of-sample role");
  require(t.identity == model.target_identity && t.names == model.names && t.units == model.units &&
          predictions.size() == model.arms.size() && !predictions.empty(), "common target/readout mismatch");
  const auto n = t.origins.row_ids.size(0), l = static_cast<int64_t>(t.names.size());
  const auto role = t.origins.roles.eq(static_cast<int64_t>(scoring_role));
  auto common = t.valid & t.available_ms.le(scoring_cutoff) & role.unsqueeze(1);
  for (size_t k = 0; k < predictions.size(); ++k) {
    const auto& p = predictions[k]; const auto& a = model.arms[k];
    require(p.target_identity == t.identity && p.arm_name == a.name && p.feature_identity == a.identity &&
            p.fit_cutoff_ms == model.recipe.fit_cutoff_ms && torch::equal(p.origins.row_ids, t.origins.row_ids) &&
            torch::equal(p.origins.roles, t.origins.roles) && torch::equal(p.origins.decision_ms, t.origins.decision_ms),
            "prediction origin, feature, target or fit identity mismatch");
    tensor(p.values, torch::kFloat64, {n, l}, "invalid risk prediction values");
    tensor(p.valid, torch::kBool, {n, l}, "invalid risk prediction validity");
    tensor(p.reasons, torch::kInt64, {n, l}, "invalid risk prediction reasons");
    require(torch::isfinite(p.values.index({p.valid})).all().item<bool>() &&
            p.reasons.index({p.valid}).eq(0).all().item<bool>(), "invalid issued prediction");
    common = common & p.valid & t.origins.decision_ms.ge(std::max(p.fit_cutoff_ms, a.artifact_fit_cutoff_ms)).unsqueeze(1);
  }
  std::vector<UtilityScore> out;
  for (size_t k = 0; k < predictions.size(); ++k) {
    for (int64_t j = 0; j < l; ++j) {
      UtilityScore s; s.arm_name = predictions[k].arm_name; s.observable = t.names[j]; s.unit = t.units[j];
      s.total = role.sum().item<int64_t>();
      s.eligible_targets = (role & t.valid.select(1, j) & t.available_ms.select(1, j).le(scoring_cutoff)).sum().item<int64_t>();
      s.valid_inputs = (role & predictions[k].valid.select(1, j)).sum().item<int64_t>();
      const auto selected = common.select(1, j); s.common_scored = selected.sum().item<int64_t>();
      s.coverage = s.eligible_targets ? static_cast<double>(s.common_scored) / s.eligible_targets : 0;
      s.supported = s.common_scored > 0;
      if (s.supported) {
        const auto y = t.values.select(1, j).index({selected});
        const auto error = predictions[k].values.select(1, j).index({selected}) - y;
        const auto& h = model.arms[k].heads[j];
        s.mae = error.abs().mean().item<double>(); s.mse = error.square().mean().item<double>();
        s.reference_normalized_mse = s.mse / (h.target_scale * h.target_scale);
        const auto denominator = (y - h.target_mean).square().mean().item<double>();
        s.skill_supported = denominator > model.recipe.denominator_epsilon;
        if (s.skill_supported) s.train_mean_skill = 1 - s.mse / denominator;
      }
      out.push_back(std::move(s));
    }
  }
  return out;
}

void save_risk_predictions(const std::string& path, const std::vector<RiskPrediction>& predictions) {
  require(!predictions.empty(), "cannot archive empty arm list");
  torch::serialize::OutputArchive archive;
  string(archive, "archive_type", "latent-risk-predictions-v1");
  archive.write("arm_count", torch::tensor({static_cast<int64_t>(predictions.size())}, torch::kInt64), true);
  for (size_t i = 0; i < predictions.size(); ++i) {
    const auto& p = predictions[i]; const auto n = origins(p.origins);
    require(p.values.defined() && p.values.dim() == 2, "invalid archived risk predictions");
    const auto l = p.values.size(1);
    tensor(p.values, torch::kFloat64, {n, l}, "invalid prediction values");
    tensor(p.valid, torch::kBool, {n, l}, "invalid prediction validity");
    tensor(p.reasons, torch::kInt64, {n, l}, "invalid prediction reasons");
    require(torch::isfinite(p.values.index({p.valid})).all().item<bool>(), "nonfinite archived prediction");
    torch::serialize::OutputArchive a;
    string(a, "arm_name", p.arm_name); string(a, "feature_identity", p.feature_identity);
    string(a, "target_identity", p.target_identity);
    a.write("fit_cutoff_ms", torch::tensor({p.fit_cutoff_ms}, torch::kInt64), true);
    a.write("values", torch::where(p.valid, p.values, torch::zeros_like(p.values)), true);
    a.write("valid", p.valid, true); a.write("reasons", p.reasons, true);
    a.write("origin_row_ids", p.origins.row_ids, true); a.write("roles", p.origins.roles, true);
    a.write("decision_ms", p.origins.decision_ms, true); archive.write("arm_" + std::to_string(i), a);
  }
  publish(path, archive);
}
void save_risk_readouts(const std::string& path, const RiskReadoutBundle& model) {
  recipe(model.recipe); require(!model.target_identity.empty(), "target identity required");
  torch::serialize::OutputArchive archive; string(archive, "archive_type", "latent-risk-readouts-v1");
  string(archive, "target_identity", model.target_identity);
  archive.write("recipe_times", torch::tensor({model.recipe.fit_cutoff_ms, model.recipe.train_begin_ms,
      model.recipe.train_end_ms}, torch::kInt64), true);
  archive.write("recipe_settings", torch::tensor({model.recipe.ridge_penalty, model.recipe.std_floor,
      model.recipe.denominator_epsilon}, torch::kFloat64), true);
  archive.write("counts", torch::tensor({static_cast<int64_t>(model.arms.size()),
      static_cast<int64_t>(model.names.size())}, torch::kInt64), true);
  archive.write("common_training_rows", torch::tensor(model.common_training_rows, torch::kInt64), true);
  for (size_t j = 0; j < model.names.size(); ++j) {
    string(archive, "name_" + std::to_string(j), model.names[j]); string(archive, "unit_" + std::to_string(j), model.units[j]);
  }
  for (size_t k = 0; k < model.arms.size(); ++k) {
    const auto& a = model.arms[k]; torch::serialize::OutputArchive arm;
    string(arm, "name", a.name); string(arm, "identity", a.identity); string(arm, "information_contract", a.information_contract);
    arm.write("metadata", torch::tensor({a.input_width, a.artifact_fit_cutoff_ms}, torch::kInt64), true);
    for (size_t j = 0; j < a.heads.size(); ++j) {
      const auto& h = a.heads[j]; torch::serialize::OutputArchive head;
      head.write("training_rows", torch::tensor({h.training_rows}, torch::kInt64), true);
      head.write("target_reference", torch::tensor({h.target_mean, h.target_scale}, torch::kFloat64), true);
      head.write("input_mean", h.input_mean, true); head.write("input_scale", h.input_scale, true);
      head.write("weights", h.weights, true); arm.write("head_" + std::to_string(j), head);
    }
    archive.write("arm_" + std::to_string(k), arm);
  }
  publish(path, archive);
}

std::vector<RiskPrediction> load_risk_predictions(const std::string& path) {
  torch::serialize::InputArchive archive; archive.load_from(path, torch::kCPU);
  require(string(archive, "archive_type") == "latent-risk-predictions-v1", "wrong risk prediction archive type");
  const auto counts = read(archive, "arm_count"); tensor(counts, torch::kInt64, {1}, "invalid archive arm count");
  const auto count = counts[0].item<int64_t>(); require(count > 0, "empty archived arms");
  std::vector<RiskPrediction> out;
  std::set<std::string> names;
  for (int64_t i = 0; i < count; ++i) {
    torch::serialize::InputArchive a; archive.read("arm_" + std::to_string(i), a);
    RiskPrediction p; p.arm_name = string(a, "arm_name"); p.feature_identity = string(a, "feature_identity");
    p.target_identity = string(a, "target_identity");
    require(!p.arm_name.empty() && names.insert(p.arm_name).second && !p.feature_identity.empty() &&
            !p.target_identity.empty(), "invalid archived arm identities");
    const auto cutoff = read(a, "fit_cutoff_ms"); tensor(cutoff, torch::kInt64, {1}, "invalid archived fit cutoff");
    p.fit_cutoff_ms = cutoff[0].item<int64_t>();
    p.origins.row_ids = read(a, "origin_row_ids"); p.origins.roles = read(a, "roles"); p.origins.decision_ms = read(a, "decision_ms");
    const auto n = origins(p.origins);
    p.values = read(a, "values"); p.valid = read(a, "valid"); p.reasons = read(a, "reasons");
    require(p.values.dim() == 2 && p.values.size(1) > 0, "invalid archived prediction geometry");
    const auto l = p.values.size(1);
    tensor(p.values, torch::kFloat64, {n, l}, "invalid archived risk values");
    tensor(p.valid, torch::kBool, {n, l}, "invalid archived risk validity");
    tensor(p.reasons, torch::kInt64, {n, l}, "invalid archived risk reasons");
    require(torch::isfinite(p.values.index({p.valid})).all().item<bool>() &&
            p.reasons.index({p.valid}).eq(0).all().item<bool>(), "invalid archived issued prediction");
    out.push_back(std::move(p));
  }
  return out;
}
RiskReadoutBundle load_risk_readouts(const std::string& path) {
  torch::serialize::InputArchive archive; archive.load_from(path, torch::kCPU);
  require(string(archive, "archive_type") == "latent-risk-readouts-v1", "wrong risk readout archive type");
  RiskReadoutBundle model; model.target_identity = string(archive, "target_identity");
  require(!model.target_identity.empty(), "empty archived target identity");
  const auto times = read(archive, "recipe_times"), settings = read(archive, "recipe_settings");
  const auto counts = read(archive, "counts");
  tensor(times, torch::kInt64, {3}, "invalid readout times"); tensor(settings, torch::kFloat64, {3}, "invalid readout settings");
  tensor(counts, torch::kInt64, {2}, "invalid readout counts");
  auto& r = model.recipe; r.fit_cutoff_ms = times[0].item<int64_t>(); r.train_begin_ms = times[1].item<int64_t>();
  r.train_end_ms = times[2].item<int64_t>(); r.ridge_penalty = settings[0].item<double>();
  r.std_floor = settings[1].item<double>(); r.denominator_epsilon = settings[2].item<double>(); recipe(r);
  const auto arms = counts[0].item<int64_t>(), labels = counts[1].item<int64_t>();
  require(arms > 0 && labels > 0, "empty archived risk model");
  const auto common = read(archive, "common_training_rows"); tensor(common, torch::kInt64, {labels}, "invalid TRAIN counts");
  for (int64_t j = 0; j < labels; ++j) {
    model.names.push_back(string(archive, "name_" + std::to_string(j)));
    model.units.push_back(string(archive, "unit_" + std::to_string(j)));
    model.common_training_rows.push_back(common[j].item<int64_t>());
    require(!model.names.back().empty() && !model.units.back().empty() && model.common_training_rows.back() >= 0,
            "invalid archived observable spec");
  }
  std::set<std::string> names;
  for (int64_t k = 0; k < arms; ++k) {
    torch::serialize::InputArchive arm; archive.read("arm_" + std::to_string(k), arm);
    FittedFeatureArm a; a.name = string(arm, "name"); a.identity = string(arm, "identity");
    a.information_contract = string(arm, "information_contract");
    require(!a.name.empty() && names.insert(a.name).second && !a.identity.empty() &&
            !a.information_contract.empty(), "invalid archived feature identity");
    const auto metadata = read(arm, "metadata"); tensor(metadata, torch::kInt64, {2}, "invalid feature metadata");
    a.input_width = metadata[0].item<int64_t>(); a.artifact_fit_cutoff_ms = metadata[1].item<int64_t>();
    require(a.input_width > 0 && a.artifact_fit_cutoff_ms <= r.fit_cutoff_ms, "invalid feature geometry/cutoff");
    for (int64_t j = 0; j < labels; ++j) {
      torch::serialize::InputArchive head; arm.read("head_" + std::to_string(j), head);
      const auto count = read(head, "training_rows"), reference = read(head, "target_reference");
      tensor(count, torch::kInt64, {1}, "invalid head TRAIN count"); tensor(reference, torch::kFloat64, {2}, "invalid target reference");
      ObservableHead h; h.training_rows = count[0].item<int64_t>(); h.target_mean = reference[0].item<double>();
      h.target_scale = reference[1].item<double>(); h.input_mean = read(head, "input_mean");
      h.input_scale = read(head, "input_scale"); h.weights = read(head, "weights");
      require(h.training_rows == model.common_training_rows[j] && std::isfinite(h.target_mean) &&
              std::isfinite(h.target_scale) && h.target_scale > 0, "invalid head population/reference");
      for (const auto& value : {h.input_mean, h.input_scale, h.weights}) {
        tensor(value, torch::kFloat64, {a.input_width}, "invalid head tensor shape");
        require(torch::isfinite(value).all().item<bool>(), "nonfinite head tensor");
      }
      require(h.input_scale.gt(0).all().item<bool>(), "nonpositive input scale");
      if (!model.arms.empty()) {
        const auto& first = model.arms[0].heads[j];
        require(h.target_mean == first.target_mean && h.target_scale == first.target_scale,
                "arm target reference mismatch");
      }
      a.heads.push_back(std::move(h));
    }
    model.arms.push_back(std::move(a));
  }
  return model;
}
} // namespace cuwacunu::evaluation::latent_dynamics::v1
