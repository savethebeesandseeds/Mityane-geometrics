// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Additive saved-TRAIN-only arithmetic. No model forward or head fitting code.
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using T = torch::Tensor;
constexpr int64_t rows = 112, prefix_width = 32, change_width = 4, tasks = 4;
constexpr double prediction_tolerance = 1e-8, equality_tolerance = 1e-12,
                 kkt_tolerance = 1e-8;
constexpr std::array<int64_t, 3> repetitions{2701, 2802, 2903};
constexpr std::array<const char *, 4> task_names{
    "past_volatility", "past_trend", "past_activity", "past_relation"};
constexpr std::array<const char *, 4> task_units{
    "logreturn", "logreturn/tick", "log1p(base_volume)", "correlation"};
std::vector<std::string> opened_paths;

void require(bool condition, const std::string &why) {
  if (!condition) throw std::runtime_error("[saved TRAIN diagnosis] " + why);
}

std::string quote(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                         << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}

std::string number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string array(const std::vector<std::string> &parts) {
  std::string result = "[";
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) result += ',';
    result += parts[i];
  }
  return result + ']';
}

std::string object(const std::vector<std::pair<std::string, std::string>> &parts) {
  std::string result = "{";
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) result += ',';
    result += quote(parts[i].first) + ':' + parts[i].second;
  }
  return result + '}';
}

std::string tensor_json(const T &value) {
  const auto x = value.to(torch::kFloat64).contiguous();
  if (x.dim() == 0) return number(x.item<double>());
  std::vector<std::string> pieces;
  for (int64_t i = 0; i < x.size(0); ++i) pieces.push_back(tensor_json(x[i]));
  return array(pieces);
}

T read(torch::serialize::InputArchive &archive, const std::string &key) {
  T result;
  archive.read(key, result, true);
  require(result.defined() && result.device().is_cpu() && !result.requires_grad(),
          key + " must be detached CPU data");
  return result;
}

void open(torch::serialize::InputArchive &archive, const fs::path &path) {
  require(fs::is_regular_file(path), "missing input " + path.string());
  opened_paths.push_back(fs::relative(path, "/cuwacunu").generic_string());
  archive.load_from(path.string(), torch::kCPU);
}

void finite(const T &value, const std::string &name) {
  require(torch::isfinite(value).all().item<bool>(), "nonfinite input " + name);
}

void shape(const T &value, at::IntArrayRef dimensions, torch::ScalarType dtype,
           const std::string &name) {
  require(value.sizes() == dimensions && value.scalar_type() == dtype,
          "shape/type mismatch for " + name);
}

double maximum(const T &value) {
  return value.numel() ? value.abs().max().item<double>() : 0.;
}

double equality_error(const T &left, const T &right) {
  require(left.sizes() == right.sizes(), "equality shape mismatch");
  return maximum((left - right).abs() / (1. + right.abs()));
}

void equal_normalizer(const T &left, const T &right, const std::string &name) {
  require(equality_error(left, right) <= equality_tolerance,
          name + " differs beyond absolute/relative 1e-12");
}

T selected(const T &valid) { return valid.nonzero().reshape({-1}); }
double scalar_mean(const T &value) { return value.mean().item<double>(); }
double variance(const T &value) { return scalar_mean((value - value.mean()).square()); }
double rms(const T &value) { return std::sqrt(scalar_mean(value.square())); }
double l2_norm(const T &value) { return std::sqrt(value.square().sum().item<double>()); }

double correlation(const T &left, const T &right) {
  if (left.numel() < 2) return std::numeric_limits<double>::quiet_NaN();
  auto a = left - left.mean(), b = right - right.mean();
  const auto denominator = l2_norm(a) * l2_norm(b);
  return denominator > 0. ? a.dot(b).item<double>() / denominator
                         : std::numeric_limits<double>::quiet_NaN();
}

T covariance(const T &left, const T &right) {
  require(left.size(0) == right.size(0) && left.size(0) > 0, "covariance rows");
  auto a = left - left.mean(0), b = right - right.mean(0);
  return a.transpose(0, 1).matmul(b) / left.size(0);
}

T correlations(const T &left, const T &right) {
  auto result = covariance(left, right);
  auto a = (left - left.mean(0)).square().mean(0).sqrt();
  auto b = (right - right.mean(0)).square().mean(0).sqrt();
  auto denominator = a.unsqueeze(1) * b.unsqueeze(0);
  return torch::where(denominator.gt(0), result / denominator,
      torch::full_like(result, std::numeric_limits<double>::quiet_NaN()));
}

std::string summary_matrix(const T &value, bool off_diagonal) {
  auto mask = torch::isfinite(value);
  if (off_diagonal) mask = mask.logical_and(
      torch::eye(value.size(0), value.options().dtype(torch::kBool)).logical_not());
  auto picked = value.masked_select(mask);
  const auto missing = std::numeric_limits<double>::quiet_NaN();
  return object({{"count", std::to_string(picked.numel())},
      {"sum", number(picked.sum().item<double>())},
      {"sum_absolute", number(picked.abs().sum().item<double>())},
      {"mean", number(picked.numel() ? scalar_mean(picked) : missing)},
      {"mean_absolute", number(picked.numel() ? scalar_mean(picked.abs()) : missing)},
      {"max_absolute", number(picked.numel() ? maximum(picked) : missing)},
      {"sum_squared", number(picked.square().sum().item<double>())},
      {"absolute_correlation_at_least_0_9_count_descriptive_only",
       std::to_string(picked.abs().ge(.9).sum().item<int64_t>())}});
}

struct Features { T values, valid, row_ids; };
struct Targets { T values, valid, row_ids, group_ids; };
struct Fit { T feature_mean, feature_scale, target_mean, target_scale,
    weights, intercept, supported, target_counts, target_supported;
    int64_t feature_rows; };
struct Predictions { T ridge, neural, targets, valid, target_valid, row_ids, group_ids; };
struct Method {
  std::string label, name;
  Features features;
  std::array<Fit, 3> fits;
  std::array<Predictions, 3> predictions;
  T standardized;
};

Features load_features(const fs::path &path, int64_t width) {
  torch::serialize::InputArchive archive;
  open(archive, path);
  Features result{read(archive, "values"), read(archive, "valid"), read(archive, "row_ids")};
  require(result.values.scalar_type() == torch::kFloat32 ||
          result.values.scalar_type() == torch::kFloat64, "floating features required");
  require(result.values.sizes() == at::IntArrayRef({rows, width}), "feature geometry");
  result.values = result.values.to(torch::kFloat64);
  shape(result.valid, {rows}, torch::kBool, "feature valid");
  shape(result.row_ids, {rows}, torch::kInt64, "feature row IDs");
  finite(result.values, "all feature storage");
  return result;
}

Targets load_targets(const fs::path &path) {
  torch::serialize::InputArchive archive;
  open(archive, path);
  Targets result{read(archive, "targets"), read(archive, "valid"),
      read(archive, "row_ids"), read(archive, "group_ids")};
  shape(result.values, {rows, tasks}, torch::kFloat64, "targets");
  shape(result.valid, {rows, tasks}, torch::kBool, "target valid");
  shape(result.row_ids, {rows}, torch::kInt64, "target row IDs");
  shape(result.group_ids, {rows}, torch::kInt64, "target group IDs");
  finite(result.values, "all target storage");
  require(result.row_ids.slice(0, 1).gt(result.row_ids.slice(0, 0, rows - 1)).all().item<bool>(),
          "fixed halves require strictly chronological row IDs");
  return result;
}

Fit load_fit(const fs::path &path, int64_t width) {
  torch::serialize::InputArchive archive;
  open(archive, path);
  Fit result;
  result.feature_mean = read(archive, "feature_mean");
  result.feature_scale = read(archive, "feature_scale");
  result.target_mean = read(archive, "target_mean");
  result.target_scale = read(archive, "target_scale");
  result.weights = read(archive, "ridge_weights");
  result.intercept = read(archive, "ridge_intercept");
  result.supported = read(archive, "supported_targets");
  result.target_counts = read(archive, "target_training_counts");
  result.target_supported = read(archive, "target_training_supported");
  result.feature_rows = read(archive, "training_feature_rows").item<int64_t>();
  require(read(archive, "feature_training_supported").item<bool>(), "feature support");
  for (const auto &[name, value] : std::vector<std::pair<std::string, T>>{
           {"feature_mean", result.feature_mean}, {"feature_scale", result.feature_scale}}) {
    shape(value, {width}, torch::kFloat64, name);
    finite(value, name);
  }
  shape(result.weights, {width, tasks}, torch::kFloat64, "ridge weights");
  shape(result.intercept, {tasks}, torch::kFloat64, "ridge intercept");
  shape(result.target_mean, {tasks}, torch::kFloat64, "target mean");
  shape(result.target_scale, {tasks}, torch::kFloat64, "target scale");
  shape(result.supported, {tasks}, torch::kBool, "supported targets");
  shape(result.target_counts, {tasks}, torch::kInt64, "target counts");
  shape(result.target_supported, {tasks}, torch::kBool, "target support");
  finite(result.weights, "ridge weights");
  finite(result.intercept, "ridge intercept");
  finite(result.target_mean, "target mean");
  finite(result.target_scale, "target scale");
  require(result.feature_scale.gt(0).all().item<bool>() &&
          result.target_scale.gt(0).all().item<bool>(), "positive stored scales required");
  return result;
}

Predictions load_predictions(const fs::path &path) {
  torch::serialize::InputArchive archive;
  open(archive, path);
  Predictions result{read(archive, "ridge_predictions"), read(archive, "neural_predictions"),
      read(archive, "targets"), read(archive, "valid"), read(archive, "target_valid"),
      read(archive, "row_ids"), read(archive, "group_ids")};
  for (const auto &[name, value] : std::vector<std::pair<std::string, T>>{
           {"ridge", result.ridge}, {"neural", result.neural}, {"targets", result.targets}}) {
    shape(value, {rows, tasks}, torch::kFloat64, name);
    finite(value, name);
  }
  shape(result.valid, {rows, tasks}, torch::kBool, "prediction valid");
  shape(result.target_valid, {rows, tasks}, torch::kBool, "prediction target valid");
  shape(result.row_ids, {rows}, torch::kInt64, "prediction row IDs");
  shape(result.group_ids, {rows}, torch::kInt64, "prediction group IDs");
  return result;
}

Method load_method(const fs::path &root, const std::string &label,
                   const std::string &name, const fs::path &run,
                   const std::string &feature_name, int64_t width) {
  Method result;
  result.label = label;
  result.name = name;
  result.features = load_features(root / run / "features" / (feature_name + "-train.pt"), width);
  for (size_t index = 0; index < repetitions.size(); ++index) {
    const auto folder = root / run / "readouts/models" / name /
                        ("rep-" + std::to_string(repetitions[index]));
    result.fits[index] = load_fit(folder / "fit.pt", width);
    result.predictions[index] = load_predictions(folder / "predictions_train.pt");
  }
  return result;
}

std::string score(const T &prediction, const T &target, const T &valid,
                  const T &target_valid) {
  auto i = selected(valid.logical_and(target_valid));
  const auto n = prediction.numel(), count = i.numel();
  const double missing = std::numeric_limits<double>::quiet_NaN();
  double mae = missing, rmse_value = missing, r2 = missing;
  if (count) {
    auto p = prediction.index_select(0, i), y = target.index_select(0, i), error = p - y;
    mae = scalar_mean(error.abs());
    rmse_value = rms(error);
    auto sst = (y - y.mean()).square().sum().item<double>();
    if (count > 1 && sst > 0.) r2 = 1. - error.square().sum().item<double>() / sst;
  }
  return object({{"total", std::to_string(n)},
      {"target_eligible", std::to_string(target_valid.sum().item<int64_t>())},
      {"valid", std::to_string(count)}, {"coverage", number(double(count) / n)},
      {"mae", number(mae)}, {"rmse", number(rmse_value)}, {"r2", number(r2)}});
}

void validate(Method &method, const Targets &targets) {
  const auto &f = method.features;
  require(torch::equal(f.row_ids, targets.row_ids), "feature target association mismatch");
  require(f.valid.all().item<bool>() && targets.valid.all().item<bool>(),
          "declared full 112-row TRAIN feature/target support is required");
  auto support = selected(f.valid);
  require(support.numel() >= 2, "TRAIN feature support too small");
  auto eligible = f.values.index_select(0, support), feature_mean = eligible.mean(0);
  auto feature_scale = (eligible - feature_mean).square().mean(0).sqrt().clamp_min(1e-8);
  for (size_t rep = 0; rep < repetitions.size(); ++rep) {
    const auto &fit = method.fits[rep];
    const auto &p = method.predictions[rep];
    equal_normalizer(fit.feature_mean, feature_mean, "stored feature mean");
    equal_normalizer(fit.feature_scale, feature_scale, "stored feature scale");
    require(fit.feature_rows == support.numel(), "stored feature support count mismatch");
    require(torch::equal(p.row_ids, targets.row_ids) && torch::equal(p.group_ids, targets.group_ids) &&
            torch::equal(p.targets, targets.values) && torch::equal(p.target_valid, targets.valid),
            "saved TRAIN prediction association mismatch");
    require(torch::equal(p.valid, f.valid.unsqueeze(1).logical_and(targets.valid)
        .logical_and(fit.supported.unsqueeze(0))), "saved prediction validity mismatch");
    require(p.valid.all().item<bool>(), "declared full 112-row saved prediction support is required");
    require(torch::equal(fit.target_counts, targets.valid.sum(0)), "stored target counts mismatch");
    require(torch::equal(fit.target_supported, fit.target_counts.ge(2)), "stored target support mismatch");
    for (int64_t task = 0; task < tasks; ++task) {
      auto i = selected(targets.valid.select(1, task));
      require(i.numel() >= 2 && fit.supported[task].item<bool>(), "unsupported target");
      auto y = targets.values.select(1, task).index_select(0, i);
      equal_normalizer(fit.target_mean[task], y.mean(), "stored target mean");
      equal_normalizer(fit.target_scale[task], (y - y.mean()).square().mean().sqrt().clamp_min(1e-8),
                       "stored target scale");
    }
    if (rep) {
      require(torch::equal(fit.weights, method.fits[0].weights) &&
              torch::equal(fit.intercept, method.fits[0].intercept) &&
              torch::equal(p.ridge, method.predictions[0].ridge),
              "three saved deterministic ridge repetitions differ");
      equal_normalizer(fit.feature_mean, method.fits[0].feature_mean, "repetition feature mean");
      equal_normalizer(fit.feature_scale, method.fits[0].feature_scale, "repetition feature scale");
      equal_normalizer(fit.target_mean, method.fits[0].target_mean, "repetition target mean");
      equal_normalizer(fit.target_scale, method.fits[0].target_scale, "repetition target scale");
    }
  }
  method.standardized = torch::where(f.valid.unsqueeze(1),
      (f.values - method.fits[0].feature_mean) / method.fits[0].feature_scale,
      torch::zeros_like(f.values));
}

std::string score_task(const Method &method, const Targets &targets, int64_t task,
                       int64_t start, int64_t end) {
  const auto y = targets.values.slice(0, start, end).select(1, task);
  const auto tv = targets.valid.slice(0, start, end).select(1, task);
  const auto valid = method.predictions[0].valid.slice(0, start, end).select(1, task);
  std::vector<std::string> neural_scores;
  double mean_r2 = 0., mean_mae = 0., mean_rmse = 0.;
  auto i = selected(valid.logical_and(tv));
  for (size_t rep = 0; rep < repetitions.size(); ++rep) {
    auto p = method.predictions[rep].neural.slice(0, start, end).select(1, task);
    neural_scores.push_back(object({{"repetition", std::to_string(repetitions[rep])},
        {"score", score(p, y, valid, tv)}}));
    if (i.numel()) {
      auto yi = y.index_select(0, i), error = p.index_select(0, i) - yi;
      mean_mae += scalar_mean(error.abs()) / repetitions.size();
      mean_rmse += rms(error) / repetitions.size();
      auto denominator = (yi - yi.mean()).square().sum().item<double>();
      mean_r2 += (denominator > 0. && i.numel() > 1
          ? 1. - error.square().sum().item<double>() / denominator
          : std::numeric_limits<double>::quiet_NaN()) / repetitions.size();
    } else mean_r2 = mean_mae = mean_rmse = std::numeric_limits<double>::quiet_NaN();
  }
  return object({{"task", quote(task_names[task])}, {"unit", quote(task_units[task])},
      {"row_start_inclusive", std::to_string(start)}, {"row_end_exclusive", std::to_string(end)},
      {"ridge", score(method.predictions[0].ridge.slice(0, start, end).select(1, task), y, valid, tv)},
      {"neural_each", array(neural_scores)},
      {"neural_mean_repetition_scores", object({{"r2", number(mean_r2)},
          {"mae", number(mean_mae)}, {"rmse", number(mean_rmse)}})}});
}

std::string vector_stats(const T &value) {
  const auto missing = std::numeric_limits<double>::quiet_NaN();
  return object({{"mean", number(value.numel() ? scalar_mean(value) : missing)},
      {"rms", number(value.numel() ? rms(value) : missing)},
      {"population_std", number(value.numel() ? std::sqrt(variance(value)) : missing)},
      {"min", number(value.numel() ? value.min().item<double>() : missing)},
      {"max", number(value.numel() ? value.max().item<double>() : missing)}});
}

std::string contribution_context(const T &prefix, const T &change, const T &old_prediction,
    const T &new_prediction, const T &prefix_delta, const T &intercept_delta, const T &target,
    const T &valid, double target_scale, int64_t start, int64_t end) {
  auto i = selected(valid.slice(0, start, end));
  const auto pick = [&](const T &x) { return x.slice(0, start, end).index_select(0, i); };
  auto level = pick(prefix), suffix = pick(change), old = pick(old_prediction),
       now = pick(new_prediction), changed_level = pick(prefix_delta),
       intercept_shift = pick(intercept_delta), y = pick(target);
  if (!i.numel()) return object({{"rows", "0"}});
  auto combined = level + suffix;
  const auto denominator = variance(combined);
  auto delta = now - old, error = old - y;
  const double mse_delta = scalar_mean((now - y).square()) - scalar_mean(error.square());
  const double cross_term = 2. * scalar_mean(error * delta), delta_square = scalar_mean(delta.square());
  const double identity_relative_error = std::abs(mse_delta - cross_term - delta_square) /
      (1. + std::abs(mse_delta) + std::abs(cross_term) + delta_square);
  const double identity_target_standardized_error = std::abs(mse_delta - cross_term - delta_square) /
      (target_scale * target_scale);
  require(identity_target_standardized_error <= kkt_tolerance, "TRAIN MSE delta identity mismatch");
  auto blocks = torch::stack({old, changed_level, suffix, intercept_shift}, 1);
  return object({{"rows", std::to_string(i.numel())},
      {"prefix32_contribution", vector_stats(level)}, {"change4_contribution", vector_stats(suffix)},
      {"parent_prediction_physical", vector_stats(old)},
      {"prefix32_contribution_target_standardized", vector_stats(level / target_scale)},
      {"change4_contribution_target_standardized", vector_stats(suffix / target_scale)},
      {"prefix_change_target_standardized", vector_stats(changed_level / target_scale)},
      {"prediction_delta_target_standardized", vector_stats(delta / target_scale)},
      {"prefix_change_physical", vector_stats(changed_level)},
      {"intercept_shift_physical", vector_stats(intercept_shift)},
      {"prediction_delta_physical", vector_stats(delta)},
      {"prefix_change_correlation", number(correlation(level, suffix))},
      {"prefix32_target_correlation", number(correlation(level, y))},
      {"change4_target_correlation", number(correlation(suffix, y))},
      {"prefix32_parent_error_prediction_minus_target_correlation", number(correlation(level, error))},
      {"prefix_change_covariance", number(scalar_mean((level - level.mean()) * (suffix - suffix.mean())))},
      {"cancellation_ratio", number(denominator > 0. ? (variance(level) + variance(suffix)) / denominator
          : std::numeric_limits<double>::quiet_NaN())},
      {"prediction_target_correlation", number(correlation(now, y))},
      {"change4_parent_error_correlation", number(correlation(suffix, error))},
      {"prediction_delta_parent_error_correlation", number(correlation(delta, error))},
      {"mse_new_minus_parent_physical", number(mse_delta)},
      {"twice_mean_parent_error_times_delta_physical", number(cross_term)},
      {"mean_delta_squared_physical", number(delta_square)},
      {"mse_delta_identity_relative_error", number(identity_relative_error)},
      {"mse_delta_identity_target_standardized_abs_error", number(identity_target_standardized_error)},
      {"delta_decomposition_covariance_block_order",
       "[\"parent_prediction\",\"changed_prefix\",\"change4\",\"intercept_shift\"]"},
      {"delta_decomposition_covariance_physical", tensor_json(covariance(blocks, blocks))}});
}

std::string task_diagnosis(const Method &method, const Method &parent,
                           const Targets &targets, int64_t task) {
  const auto &fit = method.fits[0], &old_fit = parent.fits[0];
  auto mask = method.predictions[0].valid.select(1, task)
      .logical_and(parent.predictions[0].valid.select(1, task));
  auto i = selected(mask);
  require(i.numel() >= 2, "too few common task rows");
  auto x = method.standardized.index_select(0, i),
       level = x.slice(1, 0, prefix_width),
       old_x = parent.standardized.index_select(0, i),
       y = (targets.values.select(1, task).index_select(0, i) - fit.target_mean[task]) /
           fit.target_scale[task];
  auto weights = fit.weights.select(1, task),
       parent_weights = old_fit.weights.select(1, task),
       prefix_weights = weights.slice(0, 0, prefix_width),
       difference = prefix_weights - parent_weights;
  auto normalized_prediction = x.matmul(weights) + fit.intercept[task];
  auto parent_prediction = old_x.matmul(parent_weights) + old_fit.intercept[task];
  auto error = normalized_prediction - y, parent_error = parent_prediction - y;
  auto xc = x - x.mean(0), yc = y - y.mean();
  auto rhs = xc.transpose(0, 1).matmul(yc);
  auto lhs = xc.transpose(0, 1).matmul(xc.matmul(weights)) + weights;
  auto kkt = lhs - rhs;
  const double kkt_relative = l2_norm(kkt) / std::max({1., l2_norm(lhs), l2_norm(rhs)});
  require(kkt_relative <= kkt_tolerance, "stored ridge KKT residual exceeds declared tolerance");
  const double intercept_residual = std::abs(scalar_mean(error));
  require(intercept_residual <= kkt_tolerance * (1. + maximum(y)), "stored intercept stationarity");
  const double new_sse = error.square().sum().item<double>(),
      old_sse = parent_error.square().sum().item<double>(),
      penalty = weights.square().sum().item<double>(),
      old_penalty = parent_weights.square().sum().item<double>();
  auto normalized_delta = normalized_prediction - parent_prediction;
  const double normalized_mse_delta = scalar_mean(error.square()) - scalar_mean(parent_error.square()),
      normalized_cross_term = 2. * scalar_mean(parent_error * normalized_delta),
      normalized_delta_squared = scalar_mean(normalized_delta.square());
  const double normalized_mse_identity_error = std::abs(normalized_mse_delta -
      normalized_cross_term - normalized_delta_squared) /
      (1. + std::abs(normalized_mse_delta) + std::abs(normalized_cross_term) + normalized_delta_squared);
  require(normalized_mse_identity_error <= kkt_tolerance, "normalized TRAIN MSE delta identity mismatch");
  auto padded_weights = method.features.values.size(1) > prefix_width
      ? torch::cat({parent_weights, torch::zeros({change_width}, parent_weights.options())})
      : parent_weights;
  auto padded_prediction = x.matmul(padded_weights) + old_fit.intercept[task];
  const double padded_error = maximum(padded_prediction - parent_prediction);
  require(padded_error <= prediction_tolerance,
          "padded parent prediction mismatch");
  T redistribution = torch::zeros_like(difference);
  T redistribution_left = torch::zeros_like(difference),
    redistribution_cross = torch::zeros_like(difference);
  T suffix_weight = torch::zeros({change_width}, weights.options());
  T suffix_full = torch::zeros({rows}, weights.options());
  if (method.features.values.size(1) > prefix_width) {
    auto change = x.slice(1, prefix_width);
    auto lc = level - level.mean(0), cc = change - change.mean(0);
    suffix_weight = weights.slice(0, prefix_width);
    redistribution_left = lc.transpose(0, 1).matmul(lc.matmul(difference)) + difference;
    redistribution_cross = lc.transpose(0, 1).matmul(cc.matmul(suffix_weight));
    redistribution = redistribution_left + redistribution_cross;
    suffix_full = method.standardized.slice(1, prefix_width).matmul(suffix_weight) * fit.target_scale[task];
  }
  const double redistribution_relative = l2_norm(redistribution) /
      std::max({1., l2_norm(redistribution_left), l2_norm(redistribution_cross)});
  require(redistribution_relative <= kkt_tolerance, "prefix redistribution identity mismatch");
  auto prefix_full = method.standardized.slice(1, 0, prefix_width).matmul(prefix_weights) * fit.target_scale[task];
  auto prefix_delta = method.standardized.slice(1, 0, prefix_width).matmul(difference) * fit.target_scale[task];
  auto intercept_delta = torch::ones({rows}, weights.options()) *
      ((fit.intercept[task] - old_fit.intercept[task]) * fit.target_scale[task]);
  auto saved_parent = parent.predictions[0].ridge.select(1, task),
       saved_new = method.predictions[0].ridge.select(1, task);
  auto decomposition = saved_parent + prefix_delta + suffix_full + intercept_delta;
  const double decomposition_error = maximum((decomposition - saved_new) / fit.target_scale[task]);
  require(decomposition_error <= prediction_tolerance,
      "physical parent plus prefix delta plus change plus intercept identity mismatch");
  const auto context = [&](int64_t start, int64_t end) {
    return contribution_context(prefix_full, suffix_full, saved_parent, saved_new,
        prefix_delta, intercept_delta, targets.values.select(1, task), mask,
        fit.target_scale[task].item<double>(), start, end);
  };
  return object({{"task", quote(task_names[task])}, {"unit", quote(task_units[task])},
      {"common_train_rows", std::to_string(i.numel())},
      {"normalized_unregularized_mse", number(new_sse / i.numel())},
      {"parent_normalized_unregularized_mse", number(old_sse / i.numel())},
      {"normalized_sse", number(new_sse)}, {"ridge_weight_squared_norm", number(penalty)},
      {"ridge_objective_sse_plus_penalty", number(new_sse + penalty)},
      {"parent_padded_normalized_sse", number(old_sse)},
      {"parent_padded_weight_squared_norm", number(old_penalty)},
      {"parent_padded_ridge_objective_sse_plus_penalty", number(old_sse + old_penalty)},
      {"ridge_objective_improvement_vs_parent_padded", number(old_sse + old_penalty - new_sse - penalty)},
      {"normalized_mse_new_minus_parent", number(normalized_mse_delta)},
      {"normalized_twice_mean_parent_error_times_delta", number(normalized_cross_term)},
      {"normalized_mean_delta_squared", number(normalized_delta_squared)},
      {"normalized_mse_delta_identity_relative_error", number(normalized_mse_identity_error)},
      {"parent_padded_prediction_max_abs_normalized_error", number(padded_error)},
      {"kkt_max_abs", number(maximum(kkt))}, {"kkt_relative_l2", number(kkt_relative)},
      {"kkt_lhs_l2", number(l2_norm(lhs))}, {"kkt_rhs_l2", number(l2_norm(rhs))},
      {"intercept_stationarity_abs", number(intercept_residual)},
      {"prefix_redistribution_max_abs", number(maximum(redistribution))},
      {"prefix_redistribution_relative_l2", number(redistribution_relative)},
      {"prefix_redistribution_left_l2", number(l2_norm(redistribution_left))},
      {"prefix_redistribution_cross_l2", number(l2_norm(redistribution_cross))},
      {"physical_decomposition_max_abs_target_standardized_error", number(decomposition_error)},
      {"prefix_coefficient_difference_l2", number(l2_norm(difference))},
      {"parent_prefix_coefficient_l2", number(l2_norm(parent_weights))},
      {"new_prefix_coefficient_l2", number(l2_norm(prefix_weights))},
      {"prefix_coefficient_cosine", number(l2_norm(parent_weights) * l2_norm(prefix_weights) > 0.
          ? parent_weights.dot(prefix_weights).item<double>() / (l2_norm(parent_weights) * l2_norm(prefix_weights))
          : std::numeric_limits<double>::quiet_NaN())},
      {"change4_coefficient_l2", number(l2_norm(suffix_weight))},
      {"ridge_intercept_target_standardized", number(fit.intercept[task].item<double>())},
      {"ridge_intercept_physical_including_target_mean", number(
          (fit.intercept[task] * fit.target_scale[task] + fit.target_mean[task]).item<double>())},
      {"parent_ridge_intercept_target_standardized", number(old_fit.intercept[task].item<double>())},
      {"parent_ridge_intercept_physical_including_target_mean", number(
          (old_fit.intercept[task] * old_fit.target_scale[task] + old_fit.target_mean[task]).item<double>())},
      {"prefix_coefficient_difference", tensor_json(difference)},
      {"change4_coefficients", tensor_json(suffix_weight)},
      {"all_train_contributions", context(0, rows)},
      {"fixed_first_56_rows_contributions", context(0, rows / 2)},
      {"fixed_last_56_rows_contributions", context(rows / 2, rows)}});
}

std::string method_diagnosis(const Method &method, const Method &parent, const Targets &targets) {
  const auto &fit = method.fits[0];
  auto i = selected(method.features.valid), standardized = method.standardized.index_select(0, i);
  auto level = standardized.slice(1, 0, prefix_width);
  auto parent_residual = targets.values - parent.predictions[0].ridge;
  std::vector<std::string> coordinates;
  for (int64_t coordinate = 0; coordinate < method.features.values.size(1); ++coordinate) {
    std::vector<std::string> target_corr, residual_corr;
    for (int64_t task = 0; task < tasks; ++task) {
      auto common = selected(method.predictions[0].valid.select(1, task)
          .logical_and(parent.predictions[0].valid.select(1, task)));
      auto feature = method.standardized.select(1, coordinate).index_select(0, common);
      target_corr.push_back(number(correlation(feature, targets.values.select(1, task).index_select(0, common))));
      residual_corr.push_back(number(correlation(feature, parent_residual.select(1, task).index_select(0, common))));
    }
    coordinates.push_back(object({{"coordinate", std::to_string(coordinate)},
        {"block", quote(coordinate < prefix_width ? "prefix32" : "change4")},
        {"saved_feature_mean", number(fit.feature_mean[coordinate].item<double>())},
        {"saved_feature_scale", number(fit.feature_scale[coordinate].item<double>())},
        {"standardized_mean", number(scalar_mean(standardized.select(1, coordinate)))},
        {"standardized_population_std", number(std::sqrt(variance(standardized.select(1, coordinate))))},
        {"saved_ridge_weights_all_tasks", tensor_json(fit.weights[coordinate])},
        {"target_correlations", array(target_corr)},
        {"parent_residual_target_minus_prediction_correlations", array(residual_corr)}}));
  }
  std::vector<std::string> quality, first_quality, last_quality, detail;
  for (int64_t task = 0; task < tasks; ++task) {
    quality.push_back(score_task(method, targets, task, 0, rows));
    first_quality.push_back(score_task(method, targets, task, 0, rows / 2));
    last_quality.push_back(score_task(method, targets, task, rows / 2, rows));
    detail.push_back(task_diagnosis(method, parent, targets, task));
  }
  auto level_cov = covariance(level, level), level_corr = correlations(level, level);
  std::vector<std::pair<std::string, std::string>> fields{
      {"model", quote(method.label)}, {"saved_method", quote(method.name)},
      {"native_width", std::to_string(method.features.values.size(1))},
      {"saved_feature_valid_rows", std::to_string(i.numel())},
      {"ridge_three_repetitions_identical", "true"},
      {"all_train_quality", array(quality)},
      {"fixed_first_56_rows_quality", array(first_quality)},
      {"fixed_last_56_rows_quality", array(last_quality)},
      {"prefix32_covariance_off_diagonal_summary", summary_matrix(level_cov, true)},
      {"prefix32_correlation_off_diagonal_summary", summary_matrix(level_corr, true)},
      {"prefix32_covariance", tensor_json(level_cov)},
      {"prefix32_correlation", tensor_json(level_corr)},
      {"coordinates", array(coordinates)}, {"task_diagnoses", array(detail)}};
  if (method.features.values.size(1) > prefix_width) {
    auto change = standardized.slice(1, prefix_width),
         cross_cov = covariance(level, change), cross_corr = correlations(level, change),
         change_cov = covariance(change, change), change_corr = correlations(change, change);
    fields.push_back({"prefix32_change4_cross_covariance", tensor_json(cross_cov)});
    fields.push_back({"prefix32_change4_cross_covariance_summary", summary_matrix(cross_cov, false)});
    fields.push_back({"prefix32_change4_cross_correlation", tensor_json(cross_corr)});
    fields.push_back({"prefix32_change4_cross_correlation_summary", summary_matrix(cross_corr, false)});
    fields.push_back({"change4_covariance", tensor_json(change_cov)});
    fields.push_back({"change4_covariance_off_diagonal_summary", summary_matrix(change_cov, true)});
    fields.push_back({"change4_correlation", tensor_json(change_corr)});
    fields.push_back({"change4_correlation_off_diagonal_summary", summary_matrix(change_corr, true)});
  }
  return object(fields);
}

std::string method_witness(const Method &method) {
  const auto &fit = method.fits[0];
  std::vector<std::string> neural;
  for (const auto &predictions : method.predictions) neural.push_back(tensor_json(predictions.neural));
  return object({{"model", quote(method.label)}, {"saved_method", quote(method.name)},
      {"raw_features", tensor_json(method.features.values)},
      {"feature_valid", tensor_json(method.features.valid)},
      {"row_ids", tensor_json(method.features.row_ids)},
      {"feature_mean", tensor_json(fit.feature_mean)},
      {"feature_scale", tensor_json(fit.feature_scale)},
      {"standardized_features", tensor_json(method.standardized)},
      {"ridge_weights", tensor_json(fit.weights)}, {"ridge_intercept", tensor_json(fit.intercept)},
      {"target_mean", tensor_json(fit.target_mean)}, {"target_scale", tensor_json(fit.target_scale)},
      {"supported_targets", tensor_json(fit.supported)},
      {"prediction_valid", tensor_json(method.predictions[0].valid)},
      {"saved_ridge_predictions", tensor_json(method.predictions[0].ridge)},
      {"saved_neural_predictions_each", array(neural)}});
}

void write_new(const fs::path &path, const std::string &contents) {
  require(!fs::exists(path), "refusing to overwrite " + path.string());
  std::ofstream out(path);
  require(bool(out), "cannot create output " + path.string());
  out << contents << '\n';
  out.close();
  require(bool(out), "cannot save output " + path.string());
}
} // namespace

int main(int argc, char **argv) {
  try {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    torch::NoGradGuard no_grad;
    fs::path root = "/cuwacunu", output;
    require(argc == 2, "usage: market_train_volatility_diagnosis NEW_LEAF");
    output = argv[1];
    require(!output.empty() && !fs::exists(output), "new exclusive output leaf required");
    require(fs::is_directory(root) && fs::is_regular_file("/.dockerenv"), "managed container required");
    output = fs::weakly_canonical(output);
    require(output.string().starts_with("/cuwacunu/output/runs/market-train-volatility-diagnosis/"),
            "output must stay in the dedicated ignored diagnosis run tree");
    const auto started = std::chrono::steady_clock::now();
    const auto targets = load_targets(root / "output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/train-targets.pt");
    const auto parent_run = fs::path("output/runs/market-capacity-study/run-base-gryiAV");
    const auto candidate_run = fs::path("output/runs/market-separate-changes-native36/run-4SauBf");
    std::vector<Method> methods;
    methods.push_back(load_method(root, "RPB-v22.market-02", "base_trained", parent_run, "trained_v22", prefix_width));
    methods.push_back(load_method(root, "RPB-v24.market-01.alt-01.init", "initialized_v24_native36", candidate_run,
                                  "initialized_v24_native36", prefix_width + change_width));
    methods.push_back(load_method(root, "RPB-v24.market-01.alt-01", "trained_v24_native36", candidate_run,
                                  "trained_v24_native36", prefix_width + change_width));
    for (auto &method : methods) validate(method, targets);
    require(opened_paths.size() == 22 && std::set<std::string>(opened_paths.begin(), opened_paths.end()).size() == 22,
            "the fixed 22 input archives must each be opened exactly once");
    const auto &parent = methods[0];
    double reconstruction_max = 0.;
    for (const auto &method : methods) {
      const auto &fit = method.fits[0];
      require(torch::equal(method.features.valid, parent.features.valid), "method feature validity differs");
      require(torch::equal(method.features.values.slice(1, 0, prefix_width), parent.features.values),
              "saved parent prefix32 values are not bit-exact");
      equal_normalizer(fit.feature_mean.slice(0, 0, prefix_width), parent.fits[0].feature_mean, "prefix means");
      equal_normalizer(fit.feature_scale.slice(0, 0, prefix_width), parent.fits[0].feature_scale, "prefix scales");
      equal_normalizer(fit.target_mean, parent.fits[0].target_mean, "shared target means");
      equal_normalizer(fit.target_scale, parent.fits[0].target_scale, "shared target scales");
      auto reconstructed = method.standardized.matmul(fit.weights) + fit.intercept;
      for (size_t rep = 0; rep < repetitions.size(); ++rep) {
        auto expected = (method.predictions[rep].ridge - fit.target_mean) / fit.target_scale;
        const double error = maximum(reconstructed - expected);
        reconstruction_max = std::max(reconstruction_max, error);
        require(error <= prediction_tolerance, "saved ridge prediction reconstruction mismatch");
      }
    }
    const double loading_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::vector<std::string> details;
    for (const auto &method : methods) details.push_back(method_diagnosis(method, parent, targets));
    std::vector<std::string> inputs, names, units, witnesses;
    for (const auto &path : opened_paths) inputs.push_back(quote(path));
    for (const auto name : task_names) names.push_back(quote(name));
    for (const auto unit : task_units) units.push_back(quote(unit));
    for (const auto &method : methods) witnesses.push_back(method_witness(method));
    const auto witness = object({{"protocol", quote("market-train-volatility-diagnosis-v1")},
        {"view", quote("saved chronological TRAIN only")}, {"opened_input_paths", array(inputs)},
        {"boolean_tensors_json_encoding", quote("0=false, 1=true")},
        {"targets", object({{"values", tensor_json(targets.values)},
            {"valid", tensor_json(targets.valid)}, {"row_ids", tensor_json(targets.row_ids)},
            {"group_ids", tensor_json(targets.group_ids)}})},
        {"methods", array(witnesses)}});
    const auto report = object({{"protocol", quote("market-train-volatility-diagnosis-v1")},
        {"dataset_id", quote("MKT-HIST-2")}, {"view", quote("saved chronological TRAIN only")},
        {"designed_complexity_level", "4"}, {"complexity_scale_max", "5"},
        {"device", quote("cpu; archive-only diagnostic arithmetic in managed GPU-enabled container")},
        {"encoder_calls", "0"}, {"encoder_updates", "0"}, {"head_fits", "0"},
        {"normalizer_fits", "0"}, {"post_encoder_pca_fits", "0"},
        {"TRAIN_rows", std::to_string(rows)}, {"ridge_penalty", "1"},
        {"task_order", array(names)}, {"task_units", array(units)},
        {"normalizer_check_absolute_relative_tolerance", number(equality_tolerance)},
        {"prediction_reconstruction_target_standardized_absolute_tolerance", number(prediction_tolerance)},
        {"kkt_relative_tolerance", number(kkt_tolerance)},
        {"correlation_flag_threshold_descriptive_only", "0.9"},
        {"undefined_correlation_representation", quote("null; zero variance or fewer than two rows")},
        {"parent_prefix32_exact", "true"}, {"row_ids_validity_target_normalizers_verified", "true"},
        {"saved_ridge_reconstruction_max_abs_target_standardized_error", number(reconstruction_max)},
        {"opened_input_paths", array(inputs)}, {"methods", array(details)},
        {"load_and_verification_seconds", number(loading_seconds)},
        {"load_verify_and_diagnose_seconds_excluding_write", number(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count())}});
    require(fs::create_directories(output), "exclusive output leaf could not be created");
    write_new(output / "diagnosis.json", report);
    write_new(output / "numerical-witness.json", witness);
    write_new(output / "opened-inputs.json", array(inputs));
    std::cout << "TRAIN_DIAGNOSIS_REPORT=" << (output / "diagnosis.json").string() << '\n';
    std::cout << "TRAIN_ARCHIVES_OPENED=" << opened_paths.size() << '\n';
    std::cout << "ENCODER_CALLS=0 HEAD_FITS=0 ENCODER_UPDATES=0\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
