// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
#include "cuwacunu/evaluation/latent_dynamics/v1/baselines.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace cuwacunu::evaluation::latent_dynamics::v1 {
namespace {
using Index = std::unordered_map<int64_t, int64_t>;
void need(bool ok, const std::string& message) {
  if (!ok) throw std::invalid_argument("latent dynamics v1: " + message);
}
bool cpu(const torch::Tensor& x) { return x.defined() && x.device().is_cpu(); }
bool finite(double x) { return std::isfinite(x); }
Index index(const ArrayBank& bank) {
  Index out;
  for (size_t i = 0; i < bank.rows.size(); ++i) out.emplace(bank.rows[i].row_id, i);
  return out;
}
int64_t find(const Index& idx, int64_t id) {
  const auto it = idx.find(id);
  return it == idx.end() ? -1 : it->second;
}
void validate_pairs(const std::vector<Association>& pairs) {
  std::set<int64_t> ids;
  for (const auto& p : pairs) {
    need(ids.insert(p.pair_id).second, "duplicate pair ID");
    need(p.horizon_ticks > 0, "horizon must be positive");
    need(finite(p.sample_weight) && p.sample_weight > 0, "invalid pair weight");
    need(finite(p.actual_overlap_fraction) && p.actual_overlap_fraction >= 0 &&
         p.actual_overlap_fraction <= 1, "invalid actual overlap fraction");
  }
}
bool exact_offset(int64_t from, int64_t to, int64_t ticks, int64_t interval) {
  if (ticks < 0 || ticks > std::numeric_limits<int64_t>::max() / interval) return false;
  const auto delta = ticks * interval;
  return from <= std::numeric_limits<int64_t>::max() - delta && from + delta == to;
}
std::vector<int64_t> history(const Association& p, int64_t count, const Index& idx) {
  std::vector<int64_t> out;
  if (static_cast<int64_t>(p.history_row_ids.size()) < count) return out;
  const auto begin = p.history_row_ids.end() - count;
  for (auto it = begin; it != p.history_row_ids.end(); ++it) out.push_back(find(idx, *it));
  return out;
}
int64_t input_reasons(const ArrayBank& b, const Association& p, const Index& idx,
                      int64_t count, int64_t spacing, bool complete, int64_t fit_cutoff) {
  int64_t why = p.input_eligible ? int64_t{0} : static_cast<int64_t>(ledger_input_invalid);
  const auto origin = find(idx, p.origin_row_id);
  const auto hist = history(p, count, idx);
  if (origin < 0 || static_cast<int64_t>(hist.size()) != count) return why | missing_input;
  const auto& o = b.rows[origin];
  if (fit_cutoff > p.decision_ms) why |= fit_not_available;
  if (p.history_row_ids.back() != p.origin_row_id) why |= history_grid_mismatch;
  for (int64_t k = 0; k < count; ++k) {
    const auto i = hist[k];
    if (i < 0) { why |= missing_input; continue; }
    const auto& r = b.rows[i];
    if (!b.valid[i].item<bool>()) why |= invalid_input;
    if (r.available_ms > p.decision_ms) why |= input_not_available;
    if (r.role != o.role) why |= role_mismatch;
    if (count - 1 - k > std::numeric_limits<int64_t>::max() / spacing ||
        !exact_offset(r.endpoint_ms, o.endpoint_ms, (count - 1 - k) * spacing,
                      b.interval_ms)) why |= history_grid_mismatch;
    if (complete && (!r.complete_support || r.stale)) why |= incomplete_support;
  }
  return why;
}
int64_t target_reasons(const ArrayBank& b, const Association& p, const Index& idx,
                       int64_t cutoff, bool complete) {
  int64_t why = p.target_eligible ? int64_t{0} : static_cast<int64_t>(ledger_target_invalid);
  const auto origin = find(idx, p.origin_row_id), target = find(idx, p.target_row_id);
  if (target < 0 || origin < 0) return why | missing_target;
  const auto& o = b.rows[origin];
  const auto& t = b.rows[target];
  if (!b.valid[target].item<bool>()) why |= invalid_target;
  if (t.available_ms > cutoff) why |= target_not_mature;
  if (t.role != o.role) why |= role_mismatch;
  if (!exact_offset(o.endpoint_ms, t.endpoint_ms, p.horizon_ticks, b.interval_ms))
    why |= target_grid_mismatch;
  if (complete && (!t.complete_support || t.stale)) why |= incomplete_support;
  return why;
}
torch::Tensor flattened(const ArrayBank& bank, const Association& pair,
                        int64_t count, const Index& idx) {
  return bank.values.index_select(0, torch::tensor(history(pair, count, idx), torch::kInt64))
      .to(torch::kFloat64).reshape({-1});
}
void validate_recipe(const BaselineRecipe& r) {
  need(r.history_size == 1 || r.history_size == 4, "initial baseline history must be K1 or K4");
  need(r.spacing_ticks > 0, "spacing must be positive");
  need(r.train_begin_ms <= r.train_end_ms, "invalid TRAIN range");
  need(finite(r.ridge_penalty) && r.ridge_penalty > 0, "ridge penalty must be positive");
  need(finite(r.std_floor) && r.std_floor > 0, "std floor must be positive");
}
void validate_prediction(const PredictionSet& p) {
  need(!p.coordinate_id.empty() && !p.method.empty(), "missing prediction identity");
  need(p.history_size > 0 && p.spacing_ticks > 0, "invalid prediction history");
  validate_pairs(p.associations);
  const auto n = static_cast<int64_t>(p.associations.size());
  need(cpu(p.values) && p.values.scalar_type() == torch::kFloat64 && p.values.dim() == 2 &&
       p.values.size(0) == n && p.values.size(1) > 0, "invalid prediction array");
  need(cpu(p.valid) && p.valid.scalar_type() == torch::kBool && p.valid.sizes() == torch::IntArrayRef({n}),
       "invalid prediction validity");
  need(cpu(p.reasons) && p.reasons.scalar_type() == torch::kInt64 && p.reasons.sizes() == torch::IntArrayRef({n}),
       "invalid prediction reasons");
  need(torch::isfinite(p.values.index({p.valid})).all().item<bool>(), "nonfinite valid prediction");
  need(p.reasons.index({p.valid}).eq(0).all().item<bool>(), "valid prediction with invalid reason");
}
struct Accumulator {
  Score score;
  double squared{}, absolute{}, persistence{};
  void finish(double epsilon) {
    score.coverage = score.target_valid ? static_cast<double>(score.scored) / score.target_valid : 0;
    score.supported = score.scoring_weight > 0;
    if (!score.supported) return;
    score.normalized_mse = squared / score.scoring_weight;
    score.native_mae = absolute / score.scoring_weight;
    score.persistence_normalized_mse = persistence / score.scoring_weight;
    score.skill_supported = score.persistence_normalized_mse > epsilon;
    if (score.skill_supported) score.skill = 1 - squared / persistence;
  }
};
torch::Tensor doubles(const std::vector<double>& x) { return torch::tensor(x, torch::kFloat64); }
torch::Tensor integers(const std::vector<int64_t>& x) { return torch::tensor(x, torch::kInt64); }
void write_string(torch::serialize::OutputArchive& a, const std::string& key, const std::string& s) {
  std::vector<uint8_t> bytes(s.begin(), s.end());
  a.write(key, torch::tensor(bytes, torch::kUInt8), true);
}
std::string read_string(torch::serialize::InputArchive& a, const std::string& key) {
  torch::Tensor t; a.read(key, t, true);
  need(cpu(t) && t.scalar_type() == torch::kUInt8 && t.dim() == 1, "invalid archived string");
  t = t.contiguous();
  return std::string(reinterpret_cast<const char*>(t.data_ptr<uint8_t>()), t.numel());
}
torch::Tensor read_tensor(torch::serialize::InputArchive& a, const std::string& key) {
  torch::Tensor t; a.read(key, t, true); return t;
}
void publish(const std::string& path, torch::serialize::OutputArchive& archive) {
  static std::atomic<uint64_t> sequence{0};
  need(!std::filesystem::exists(path), "archive already exists: " + path);
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto temporary = path + ".stage-" + std::to_string(stamp) + "-" + std::to_string(sequence++);
  try {
    archive.save_to(temporary);
    std::filesystem::create_hard_link(temporary, path); // Atomic no-replace publication.
    std::filesystem::remove(temporary);
  } catch (...) { std::filesystem::remove(temporary); throw; }
}
void check_archive(torch::serialize::InputArchive& a, const std::string& kind) {
  need(read_string(a, "archive_type") == kind, "wrong archive type/version");
}
} // namespace

void validate_bank(const ArrayBank& b) {
  const auto n = static_cast<int64_t>(b.rows.size());
  need(!b.coordinate_id.empty(), "coordinate identity is required");
  need(b.interval_ms > 0 && b.window_ticks > 0, "invalid bank geometry");
  need(cpu(b.values) && (b.values.scalar_type() == torch::kFloat64 || b.values.scalar_type() == torch::kFloat32) &&
       b.values.dim() == 2 && b.values.size(0) == n && b.values.size(1) > 0, "invalid bank values");
  need(cpu(b.valid) && b.valid.scalar_type() == torch::kBool && b.valid.sizes() == torch::IntArrayRef({n}),
       "invalid bank mask");
  need(torch::isfinite(b.values.index({b.valid})).all().item<bool>(), "nonfinite valid bank row");
  std::set<int64_t> ids, endpoints;
  for (const auto& r : b.rows) {
    need(ids.insert(r.row_id).second, "duplicate bank row ID");
    need(endpoints.insert(r.endpoint_ms).second, "duplicate bank endpoint");
    need(r.role == Role::train || r.role == Role::validation || r.role == Role::test, "unknown role");
    need(r.support_begin_ms <= r.support_end_ms && r.support_end_ms <= r.endpoint_ms &&
         r.endpoint_ms <= r.available_ms, "noncausal row support/availability");
    need(finite(r.observed_fraction) && r.observed_fraction >= 0 && r.observed_fraction <= 1,
         "invalid observed fraction");
  }
}

BaselineModel fit_baselines(const ArrayBank& b, const std::vector<Association>& pairs,
                            const BaselineRecipe& recipe) {
  torch::NoGradGuard no_grad;
  validate_bank(b); validate_pairs(pairs); validate_recipe(recipe);
  const auto idx = index(b);
  BaselineModel out;
  out.coordinate_id = b.coordinate_id; out.recipe = recipe; out.width = b.values.size(1);
  out.interval_ms = b.interval_ms; out.window_ticks = b.window_ticks;
  out.submitted_rows = pairs.size();
  std::map<int64_t, std::vector<const Association*>> selected;
  for (const auto& p : pairs) {
    auto why = input_reasons(b, p, idx, recipe.history_size, recipe.spacing_ticks,
                             recipe.require_complete_support, recipe.fit_cutoff_ms);
    // A fit is made after these TRAIN decisions; model availability is checked
    // against future forecast decisions only, never against fitting origins.
    why &= ~fit_not_available;
    why |= target_reasons(b, p, idx, recipe.fit_cutoff_ms, recipe.require_complete_support);
    if (p.decision_ms > recipe.fit_cutoff_ms) why |= input_not_available;
    const auto target = find(idx, p.target_row_id);
    auto hist = history(p, recipe.history_size, idx); hist.push_back(target);
    for (const auto i : hist) {
      if (i < 0) continue;
      const auto& r = b.rows[i];
      if (r.role != Role::train) why |= role_mismatch;
      if (r.support_begin_ms < recipe.train_begin_ms || r.endpoint_ms > recipe.train_end_ms ||
          r.support_end_ms > recipe.train_end_ms ||
          r.available_ms > recipe.fit_cutoff_ms) why |= outside_train_support;
    }
    if (why) ++out.excluded_rows;
    else selected[p.horizon_ticks].push_back(&p);
  }
  for (const auto& [horizon, rows] : selected) {
    std::vector<torch::Tensor> xx, yy;
    std::vector<double> ww;
    for (const auto* p : rows) {
      xx.push_back(flattened(b, *p, recipe.history_size, idx));
      yy.push_back(b.values[find(idx, p->target_row_id)].to(torch::kFloat64));
      ww.push_back(p->sample_weight);
    }
    const auto x = torch::stack(xx), y = torch::stack(yy), w = doubles(ww).unsqueeze(1);
    HorizonModel model; model.horizon_ticks = horizon; model.training_rows = rows.size();
    model.training_weight = w.sum().item<double>();
    model.input_mean = (x * w).sum(0) / model.training_weight;
    model.input_scale = torch::sqrt(((x - model.input_mean).square() * w).sum(0) /
                                   model.training_weight).clamp_min(recipe.std_floor);
    model.target_mean = (y * w).sum(0) / model.training_weight;
    const auto a = ((x - model.input_mean) / model.input_scale) * torch::sqrt(w);
    const auto target = (y - model.target_mean) * torch::sqrt(w);
    // Identical weighted ridge objective in primal/dual form; intercept unpenalized.
    if (a.size(1) > a.size(0)) {
      model.ridge_weight = a.t().matmul(at::linalg_solve(a.matmul(a.t()) +
          recipe.ridge_penalty * torch::eye(a.size(0), a.options()), target));
    } else {
      model.ridge_weight = at::linalg_solve(a.t().matmul(a) +
          recipe.ridge_penalty * torch::eye(a.size(1), a.options()), a.t().matmul(target));
    }
    need(torch::isfinite(model.ridge_weight).all().item<bool>(), "nonfinite ridge fit");
    out.horizons.push_back(std::move(model));
  }
  return out;
}

std::vector<PredictionSet> predict_baselines(const ArrayBank& b,
    const std::vector<Association>& pairs, const BaselineModel& model) {
  torch::NoGradGuard no_grad;
  validate_bank(b); validate_pairs(pairs); validate_recipe(model.recipe);
  need(b.coordinate_id == model.coordinate_id && b.values.size(1) == model.width &&
       b.interval_ms == model.interval_ms && b.window_ticks == model.window_ticks,
       "baseline/bank coordinate or geometry mismatch");
  const auto idx = index(b);
  const auto n = static_cast<int64_t>(pairs.size());
  std::vector<PredictionSet> out;
  for (const auto& name : {"mean", "persistence", "ridge-k"}) {
    PredictionSet set;
    set.coordinate_id = model.coordinate_id;
    set.method = name == std::string("ridge-k") ? std::string(name) + std::to_string(model.recipe.history_size) : name;
    set.fit_cutoff_ms = model.recipe.fit_cutoff_ms;
    set.history_size = model.recipe.history_size; set.spacing_ticks = model.recipe.spacing_ticks;
    set.values = torch::zeros({n, model.width}, torch::kFloat64);
    set.valid = torch::zeros({n}, torch::kBool); set.reasons = torch::zeros({n}, torch::kInt64);
    set.associations = pairs;
    for (int64_t i = 0; i < n; ++i) {
      const auto& p = pairs[i];
      auto why = input_reasons(b, p, idx, model.recipe.history_size, model.recipe.spacing_ticks,
                               model.recipe.require_complete_support, model.recipe.fit_cutoff_ms);
      const auto it = std::find_if(model.horizons.begin(), model.horizons.end(),
          [&](const auto& h) { return h.horizon_ticks == p.horizon_ticks; });
      if (it == model.horizons.end() && set.method != "persistence") why |= no_fitted_horizon;
      set.reasons[i] = why;
      if (why) continue;
      torch::Tensor prediction;
      if (set.method == "persistence") prediction = b.values[find(idx, p.origin_row_id)].to(torch::kFloat64);
      else if (set.method == "mean") prediction = it->target_mean;
      else prediction = ((flattened(b, p, model.recipe.history_size, idx) - it->input_mean) /
                          it->input_scale).matmul(it->ridge_weight) + it->target_mean;
      need(torch::isfinite(prediction).all().item<bool>(), "nonfinite baseline prediction");
      set.values[i].copy_(prediction); set.valid[i] = true;
    }
    out.push_back(std::move(set));
  }
  return out;
}

Evaluation score_predictions(const ArrayBank& b, const PredictionSet& p,
    const MetricSpec& metric, int64_t scoring_cutoff) {
  torch::NoGradGuard no_grad;
  validate_bank(b); validate_prediction(p);
  need(p.coordinate_id == b.coordinate_id && p.values.size(1) == b.values.size(1),
       "prediction/target coordinate mismatch");
  const auto d = b.values.size(1), n = p.values.size(0);
  const auto& scale = metric.reference_scale;
  const auto& cw = metric.coordinate_weight;
  need(cpu(scale) && scale.scalar_type() == torch::kFloat64 && scale.sizes() == torch::IntArrayRef({d}) &&
       torch::isfinite(scale).all().item<bool>() && scale.gt(0).all().item<bool>(), "invalid frozen reference scale");
  need(cpu(cw) && cw.scalar_type() == torch::kFloat64 && cw.sizes() == torch::IntArrayRef({d}) &&
       torch::isfinite(cw).all().item<bool>() && cw.ge(0).all().item<bool>() && cw.sum().item<double>() > 0,
       "invalid coordinate weights");
  need(finite(metric.denominator_epsilon) && metric.denominator_epsilon >= 0, "invalid denominator threshold");
  const auto coordinate_weight = cw / cw.sum();
  const auto idx = index(b);
  Evaluation result;
  result.target_valid = torch::zeros({n}, torch::kBool);
  result.target_reasons = torch::zeros({n}, torch::kInt64);
  std::map<int64_t, Accumulator> horizon;
  std::map<std::tuple<int64_t, std::string, std::string>, Accumulator> strata;
  for (int64_t i = 0; i < n; ++i) {
    const auto& pair = p.associations[i];
    const auto why = target_reasons(b, pair, idx, scoring_cutoff, false);
    result.target_valid[i] = why == 0; result.target_reasons[i] = why;
    const auto origin = find(idx, pair.origin_row_id), target = find(idx, pair.target_row_id);
    std::string support = "missing";
    if (origin >= 0 && target >= 0) {
      bool stale = b.rows[target].stale, complete = b.rows[target].complete_support;
      for (const auto j : history(pair, p.history_size, idx)) {
        if (j < 0) { complete = false; continue; }
        stale = stale || b.rows[j].stale; complete = complete && b.rows[j].complete_support;
      }
      support = stale ? "stale" : complete ? "complete-current" : "partial";
    }
    const std::string overlap = pair.actual_overlap_fraction == 0 ? "none" :
        pair.actual_overlap_fraction == 1 ? "full" : "partial";
    auto& a = horizon[pair.horizon_ticks];
    auto& s = strata[{pair.horizon_ticks, support, overlap}];
    const bool valid_input = p.valid[i].item<bool>() && input_reasons(b, pair, idx,
        p.history_size, p.spacing_ticks, false, p.fit_cutoff_ms) == 0;
    for (auto* destination : {&a, &s}) {
      destination->score.horizon_ticks = pair.horizon_ticks; ++destination->score.total;
      destination->score.input_valid += valid_input; destination->score.target_valid += why == 0;
    }
    if (why || !valid_input) continue;
    const auto target_value = b.values[target].to(torch::kFloat64);
    const auto error = p.values[i] - target_value;
    const auto persist_error = b.values[origin].to(torch::kFloat64) - target_value;
    const double squared = ((error / scale).square() * coordinate_weight).sum().item<double>();
    const double absolute = (error.abs() * coordinate_weight).sum().item<double>();
    const double persistence = ((persist_error / scale).square() * coordinate_weight).sum().item<double>();
    for (auto* destination : {&a, &s}) {
      ++destination->score.scored; destination->score.scoring_weight += pair.sample_weight;
      destination->squared += pair.sample_weight * squared;
      destination->absolute += pair.sample_weight * absolute;
      destination->persistence += pair.sample_weight * persistence;
    }
  }
  for (auto& [h, a] : horizon) { (void)h; a.finish(metric.denominator_epsilon); result.horizons.push_back(a.score); }
  for (auto& [key, a] : strata) {
    a.finish(metric.denominator_epsilon);
    result.strata.push_back({std::get<1>(key), std::get<2>(key), a.score});
  }
  return result;
}

Diversity diversity(const ArrayBank& b, const torch::Tensor& reference) {
  torch::NoGradGuard no_grad; validate_bank(b);
  const auto d = b.values.size(1);
  need(cpu(reference) && reference.scalar_type() == torch::kFloat64 &&
       reference.sizes() == torch::IntArrayRef({d}) && torch::isfinite(reference).all().item<bool>() &&
       reference.gt(0).all().item<bool>(), "invalid diversity reference scale");
  Diversity out;
  out.width = d; out.valid_rows = b.valid.sum().item<int64_t>();
  out.covariance_rank_bound = std::min(d, std::max<int64_t>(0, out.valid_rows - 1));
  out.native_std = torch::zeros({d}, torch::kFloat64); out.reference_std = out.native_std.clone();
  if (out.valid_rows < 2) return out;
  const auto x = b.values.index({b.valid}).to(torch::kFloat64);
  const auto centered = x - x.mean(0);
  const auto cov = centered.t().matmul(centered) / static_cast<double>(out.valid_rows - 1);
  out.native_std = cov.diag().clamp_min(0).sqrt(); out.reference_std = out.native_std / reference;
  const auto normalized = cov / reference.unsqueeze(0) / reference.unsqueeze(1);
  const auto eig = at::linalg_eigvalsh(normalized).clamp_min(0);
  const double total = eig.sum().item<double>();
  out.supported = true;
  if (total > 0) {
    const auto fractions = eig / total;
    const auto positive = fractions.index({fractions.gt(0)});
    out.effective_rank = torch::exp(-(positive * torch::log(positive)).sum()).item<double>();
    out.largest_factor_fraction = fractions.max().item<double>();
  }
  return out;
}

Retention retention(const torch::Tensor& prediction, const torch::Tensor& target,
                    const torch::Tensor& valid, double epsilon) {
  torch::NoGradGuard no_grad;
  need(cpu(prediction) && cpu(target) && prediction.dim() == 2 && prediction.sizes() == target.sizes() &&
       (prediction.scalar_type() == torch::kFloat64 || prediction.scalar_type() == torch::kFloat32) &&
       (target.scalar_type() == torch::kFloat64 || target.scalar_type() == torch::kFloat32), "invalid retention arrays");
  need(cpu(valid) && valid.scalar_type() == torch::kBool && valid.sizes() == torch::IntArrayRef({target.size(0)}),
       "invalid retention validity");
  need(finite(epsilon) && epsilon >= 0, "invalid retention denominator threshold");
  const auto x = prediction.index({valid}).to(torch::kFloat64), y = target.index({valid}).to(torch::kFloat64);
  need(torch::isfinite(x).all().item<bool>() && torch::isfinite(y).all().item<bool>(), "nonfinite valid retention array");
  Retention out; out.valid_rows = x.size(0);
  const auto d = target.size(1);
  out.mae = torch::zeros({d}, torch::kFloat64); out.mse = out.mae.clone();
  out.r2 = torch::full({d}, std::numeric_limits<double>::quiet_NaN(), torch::kFloat64);
  out.r2_valid = torch::zeros({d}, torch::kBool);
  if (!out.valid_rows) return out;
  out.supported = true; const auto error = x - y;
  out.mae = error.abs().mean(0); out.mse = error.square().mean(0);
  const auto variance = (y - y.mean(0)).square().mean(0);
  out.r2_valid = variance.gt(epsilon);
  out.r2.index_put_({out.r2_valid}, 1 - out.mse.index({out.r2_valid}) / variance.index({out.r2_valid}));
  return out;
}

void save_predictions(const std::string& path, const PredictionSet& p) {
  validate_prediction(p);
  torch::serialize::OutputArchive a;
  write_string(a, "archive_type", "latent-dynamics-predictions-v1");
  write_string(a, "coordinate_id", p.coordinate_id); write_string(a, "method", p.method);
  a.write("recipe", integers({p.fit_cutoff_ms, p.history_size, p.spacing_ticks}), true);
  a.write("values", torch::where(p.valid.unsqueeze(1), p.values, torch::zeros_like(p.values)), true);
  a.write("valid", p.valid, true); a.write("reasons", p.reasons, true);
  std::vector<int64_t> metadata, history_ids, offsets{0}; std::vector<double> weights;
  for (size_t i = 0; i < p.associations.size(); ++i) {
    const auto& r = p.associations[i];
    for (auto v : {r.pair_id, r.origin_row_id, r.target_row_id, r.horizon_ticks, r.decision_ms,
         static_cast<int64_t>(r.input_eligible), static_cast<int64_t>(r.target_eligible),
         static_cast<int64_t>(r.ledger_input_reasons), static_cast<int64_t>(r.ledger_target_reasons)}) metadata.push_back(v);
    history_ids.insert(history_ids.end(), r.history_row_ids.begin(), r.history_row_ids.end());
    offsets.push_back(history_ids.size()); weights.push_back(r.sample_weight); weights.push_back(r.actual_overlap_fraction);
    write_string(a, "dependence_group_" + std::to_string(i), r.dependence_group);
  }
  a.write("associations", integers(metadata).reshape({static_cast<int64_t>(p.associations.size()), 9}), true);
  a.write("weights_overlap", doubles(weights).reshape({static_cast<int64_t>(p.associations.size()), 2}), true);
  a.write("history_ids", integers(history_ids), true); a.write("history_offsets", integers(offsets), true);
  publish(path, a);
}
PredictionSet load_predictions(const std::string& path) {
  torch::serialize::InputArchive a; a.load_from(path, torch::kCPU);
  check_archive(a, "latent-dynamics-predictions-v1"); PredictionSet out;
  out.coordinate_id = read_string(a, "coordinate_id"); out.method = read_string(a, "method");
  const auto recipe = read_tensor(a, "recipe"); need(recipe.numel() == 3, "invalid archived prediction recipe");
  out.fit_cutoff_ms = recipe[0].item<int64_t>(); out.history_size = recipe[1].item<int64_t>();
  out.spacing_ticks = recipe[2].item<int64_t>();
  out.values = read_tensor(a, "values"); out.valid = read_tensor(a, "valid"); out.reasons = read_tensor(a, "reasons");
  const auto meta = read_tensor(a, "associations"), w = read_tensor(a, "weights_overlap");
  const auto ids = read_tensor(a, "history_ids"), offsets = read_tensor(a, "history_offsets");
  const auto n = out.values.size(0);
  need(meta.scalar_type() == torch::kInt64 && meta.sizes() == torch::IntArrayRef({n, 9}) &&
       w.scalar_type() == torch::kFloat64 && w.sizes() == torch::IntArrayRef({n, 2}) &&
       ids.scalar_type() == torch::kInt64 && ids.dim() == 1 && offsets.scalar_type() == torch::kInt64 &&
       offsets.sizes() == torch::IntArrayRef({n + 1}), "invalid archived associations");
  need(offsets[0].item<int64_t>() == 0 && offsets[n].item<int64_t>() == ids.numel(), "invalid history offsets");
  for (int64_t i = 0; i < n; ++i) {
    Association p; p.pair_id = meta[i][0].item<int64_t>(); p.origin_row_id = meta[i][1].item<int64_t>();
    p.target_row_id = meta[i][2].item<int64_t>(); p.horizon_ticks = meta[i][3].item<int64_t>();
    p.decision_ms = meta[i][4].item<int64_t>(); p.input_eligible = meta[i][5].item<int64_t>() != 0;
    p.target_eligible = meta[i][6].item<int64_t>() != 0;
    p.ledger_input_reasons = static_cast<uint64_t>(meta[i][7].item<int64_t>());
    p.ledger_target_reasons = static_cast<uint64_t>(meta[i][8].item<int64_t>());
    p.sample_weight = w[i][0].item<double>(); p.actual_overlap_fraction = w[i][1].item<double>();
    const auto begin = offsets[i].item<int64_t>(), end = offsets[i + 1].item<int64_t>();
    need(begin >= 0 && begin <= end && end <= ids.numel(), "invalid history offset");
    for (auto j = begin; j < end; ++j) p.history_row_ids.push_back(ids[j].item<int64_t>());
    p.dependence_group = read_string(a, "dependence_group_" + std::to_string(i));
    out.associations.push_back(std::move(p));
  }
  validate_prediction(out); return out;
}

void save_model(const std::string& path, const BaselineModel& m) {
  validate_recipe(m.recipe);
  torch::serialize::OutputArchive a;
  write_string(a, "archive_type", "latent-dynamics-baseline-model-v1"); write_string(a, "coordinate_id", m.coordinate_id);
  const auto& r = m.recipe;
  a.write("integers", integers({r.fit_cutoff_ms, r.train_begin_ms, r.train_end_ms, r.history_size,
      r.spacing_ticks, static_cast<int64_t>(r.require_complete_support), m.width, m.interval_ms,
      m.window_ticks, m.submitted_rows, m.excluded_rows, static_cast<int64_t>(m.horizons.size())}), true);
  a.write("settings", doubles({r.ridge_penalty, r.std_floor}), true);
  for (size_t i = 0; i < m.horizons.size(); ++i) {
    const auto& h = m.horizons[i]; torch::serialize::OutputArchive child;
    child.write("integers", integers({h.horizon_ticks, h.training_rows}), true);
    child.write("weight", doubles({h.training_weight}), true);
    child.write("target_mean", h.target_mean, true); child.write("input_mean", h.input_mean, true);
    child.write("input_scale", h.input_scale, true); child.write("ridge_weight", h.ridge_weight, true);
    a.write("horizon_" + std::to_string(i), child);
  }
  publish(path, a);
}
BaselineModel load_model(const std::string& path) {
  torch::serialize::InputArchive a; a.load_from(path, torch::kCPU);
  check_archive(a, "latent-dynamics-baseline-model-v1"); BaselineModel m;
  m.coordinate_id = read_string(a, "coordinate_id");
  const auto ints = read_tensor(a, "integers"), settings = read_tensor(a, "settings");
  need(ints.scalar_type() == torch::kInt64 && ints.numel() == 12 &&
       settings.scalar_type() == torch::kFloat64 && settings.numel() == 2, "invalid archived model settings");
  auto get = [&](int64_t i) { return ints[i].item<int64_t>(); };
  auto& r = m.recipe; r.fit_cutoff_ms = get(0); r.train_begin_ms = get(1); r.train_end_ms = get(2);
  r.history_size = get(3); r.spacing_ticks = get(4); r.require_complete_support = get(5) != 0;
  r.ridge_penalty = settings[0].item<double>(); r.std_floor = settings[1].item<double>();
  m.width = get(6); m.interval_ms = get(7); m.window_ticks = get(8);
  m.submitted_rows = get(9); m.excluded_rows = get(10);
  validate_recipe(r); need(m.width > 0 && m.interval_ms > 0 && m.window_ticks > 0 && get(11) >= 0,
                         "invalid archived model geometry");
  std::set<int64_t> horizons;
  for (int64_t i = 0; i < get(11); ++i) {
    torch::serialize::InputArchive child; a.read("horizon_" + std::to_string(i), child);
    const auto hi = read_tensor(child, "integers"), hw = read_tensor(child, "weight");
    need(hi.scalar_type() == torch::kInt64 && hi.numel() == 2 && hw.numel() == 1, "invalid archived horizon settings");
    HorizonModel h; h.horizon_ticks = hi[0].item<int64_t>(); h.training_rows = hi[1].item<int64_t>();
    h.training_weight = hw[0].item<double>(); h.target_mean = read_tensor(child, "target_mean");
    h.input_mean = read_tensor(child, "input_mean"); h.input_scale = read_tensor(child, "input_scale");
    h.ridge_weight = read_tensor(child, "ridge_weight");
    const auto input_width = m.width * r.history_size;
    need(h.horizon_ticks > 0 && horizons.insert(h.horizon_ticks).second && h.training_rows > 0 &&
         finite(h.training_weight) && h.training_weight > 0, "invalid archived horizon metadata");
    for (const auto& t : {h.target_mean, h.input_mean, h.input_scale, h.ridge_weight})
      need(cpu(t) && t.scalar_type() == torch::kFloat64 && torch::isfinite(t).all().item<bool>(), "invalid model tensor");
    need(h.target_mean.sizes() == torch::IntArrayRef({m.width}) &&
         h.input_mean.sizes() == torch::IntArrayRef({input_width}) &&
         h.input_scale.sizes() == torch::IntArrayRef({input_width}) && h.input_scale.gt(0).all().item<bool>() &&
         h.ridge_weight.sizes() == torch::IntArrayRef({input_width, m.width}), "invalid model tensor shape");
    m.horizons.push_back(std::move(h));
  }
  return m;
}
} // namespace cuwacunu::evaluation::latent_dynamics::v1
