// SPDX-License-Identifier: MIT
#include "embedding/shared/feature_harness.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(std::string("[feature harness] ") + message);
}
torch::Tensor rows(const FeatureSurface &surface) {
  return surface.valid.nonzero().reshape({-1});
}
void labels_valid(const torch::Tensor &labels, int64_t count) {
  require(labels.defined() && labels.device().is_cpu() && labels.scalar_type() == torch::kInt64 &&
          labels.dim() == 1 && labels.size(0) == count &&
          labels.ge(0).logical_and(labels.le(1)).all().item<bool>(), "expected CPU binary labels [B]");
}
void batch_valid(const Batch &batch) {
  require(batch.data.defined() && batch.data.device().is_cpu() && batch.data.dim() == 4 &&
          batch.data.is_floating_point() && batch.data.size(0) > 0 && batch.data.size(1) > 0 &&
          batch.data.size(2) > 0 && batch.data.size(3) > 0, "expected CPU observations [B,C,H,F]");
  require(batch.feature_mask.defined() && batch.feature_mask.device().is_cpu() &&
          batch.feature_mask.scalar_type() == torch::kBool &&
          batch.feature_mask.sizes() == batch.data.sizes(), "observation mask mismatch");
  require(torch::isfinite(batch.data.masked_select(batch.feature_mask)).all().item<bool>(),
          "observed values must be finite");
}
ControlledDataset assemble(const std::vector<torch::Tensor> &clean,
    const std::vector<torch::Tensor> &masks, const std::vector<int64_t> &labels,
    const std::vector<std::string> &sources) {
  if (clean.empty()) {
    require(masks.empty() && labels.empty() && sources.empty(), "empty split has source metadata");
    return {};
  }
  auto values = torch::stack(clean);
  auto mask = torch::stack(masks);
  return {{values, torch::ones_like(mask)},
          {torch::where(mask, values, torch::zeros_like(values)), mask},
          torch::tensor(labels, torch::kInt64), sources};
}
void validate_split(const ControlledDataset &split, std::set<std::string> &all_sources) {
  batch_valid(split.observed); batch_valid(split.clean);
  labels_valid(split.labels, split.observed.data.size(0));
  require(split.source_ids.size() == size_t(split.labels.size(0)), "source ID count mismatch");
  std::map<std::string, std::vector<int64_t>> groups;
  for (int64_t i = 0; i < split.labels.size(0); ++i)
    groups[split.source_ids[i]].push_back(split.labels[i].item<int64_t>());
  for (const auto &[id, variants] : groups) {
    require(!id.empty() && variants.size() == 2 && variants[0] != variants[1],
            "each source requires both paired variants in one split");
    require(all_sources.insert(id).second, "source group overlaps train/validation/test");
  }
}
} // namespace

void validate_features(const FeatureSurface &surface) {
  require(surface.values.defined() && surface.values.device().is_cpu() &&
          surface.values.is_floating_point() && surface.values.dim() == 2 &&
          surface.values.size(0) > 0 && surface.values.size(1) > 0,
          "expected floating CPU feature values [B,D]");
  require(surface.valid.defined() && surface.valid.device().is_cpu() &&
          surface.valid.scalar_type() == torch::kBool && surface.valid.dim() == 1 &&
          surface.valid.size(0) == surface.values.size(0), "feature validity mismatch");
  require(torch::isfinite(surface.values.index_select(0, rows(surface))).all().item<bool>(),
          "valid feature rows must be finite");
}

RepresentationDiagnostics feature_diagnostics(const FeatureSurface &surface) {
  validate_features(surface);
  const auto selected = rows(surface);
  RepresentationDiagnostics out;
  out.valid_rows = selected.numel(); out.dimensions = surface.values.size(1);
  out.valid_fraction = double(out.valid_rows)/surface.values.size(0);
  out.per_dimension_std = torch::zeros({out.dimensions},torch::kFloat64);
  if(!out.valid_rows)return out;
  const auto x = surface.values.detach().to(torch::kFloat64).index_select(0,selected);
  const auto centered = x-x.mean(0);
  out.per_dimension_std = centered.pow(2).mean(0).sqrt();
  out.std_mean = out.per_dimension_std.mean().item<double>();
  out.std_min = out.per_dimension_std.min().item<double>();
  out.std_max = out.per_dimension_std.max().item<double>();
  const auto norms = x.pow(2).sum(1).sqrt();
  out.norm_mean=norms.mean().item<double>();out.norm_min=norms.min().item<double>();out.norm_max=norms.max().item<double>();
  if(out.valid_rows>=2){
    const auto covariance = out.valid_rows<out.dimensions
      ? centered.matmul(centered.transpose(0,1))/(out.valid_rows-1)
      : centered.transpose(0,1).matmul(centered)/(out.valid_rows-1);
    const auto spectrum=at::linalg_eigvalsh(covariance).clamp_min(0);
    const double total=spectrum.sum().item<double>();
    if(total>1e-20){
      const auto p=spectrum/total;
      out.covariance_effective_rank=torch::exp(-(p*p.clamp_min(1e-30).log()).sum()).item<double>();
    }
  }
  return out;
}

FeatureNormalizer::FeatureNormalizer(const FeatureSurface &training) {
  validate_features(training);
  const auto selected = rows(training);
  fitted_rows = selected.numel();
  require(fitted_rows >= 2, "feature fit requires two valid training rows");
  const auto x = training.values.to(torch::kFloat64).index_select(0, selected);
  mean = x.mean(0);
  scale = (x - mean).pow(2).mean(0).sqrt().clamp_min(1e-8);
  require(torch::isfinite(mean).all().item<bool>() && torch::isfinite(scale).all().item<bool>(),
          "feature fitted statistics overflow");
}
FeatureSurface FeatureNormalizer::transform(const FeatureSurface &surface) const {
  validate_features(surface);
  require(surface.values.size(1) == mean.numel(), "feature normalizer dimension mismatch");
  auto out = torch::zeros_like(surface.values, torch::kFloat64);
  const auto selected = rows(surface);
  out.index_copy_(0, selected, (surface.values.to(torch::kFloat64).index_select(0, selected) - mean) / scale);
  require(torch::isfinite(out).all().item<bool>(), "feature scaling overflow");
  return {out, surface.valid.clone(), surface.provenance};
}

TrainPca::TrainPca(const FeatureSurface &training, int64_t dimensions) {
  validate_features(training);
  const auto selected = rows(training);
  fitted_rows = selected.numel();
  require(dimensions > 0 && dimensions <= std::min(training.values.size(1), fitted_rows - 1),
          "PCA dimensions exceed valid centered training-row bound");
  const auto x = training.values.to(torch::kFloat64).index_select(0, selected);
  mean = x.mean(0);
  auto decomposition = at::linalg_svd(x - mean, false);
  singular_values = std::get<1>(decomposition);
  const double largest = singular_values.numel() ? singular_values[0].item<double>() : 0;
  const double tolerance = std::max(x.size(0), x.size(1)) *
      std::numeric_limits<double>::epsilon() * largest;
  numerical_rank = singular_values.gt(tolerance).sum().item<int64_t>();
  require(dimensions <= numerical_rank, "PCA dimensions exceed numerical training rank");
  components = std::get<2>(decomposition).narrow(0, 0, dimensions).transpose(0, 1).contiguous();
}
FeatureSurface TrainPca::transform(const FeatureSurface &surface) const {
  validate_features(surface);
  require(surface.values.size(1) == mean.numel(), "PCA input dimension mismatch");
  auto out = torch::zeros({surface.values.size(0), components.size(1)}, torch::kFloat64);
  const auto selected = rows(surface);
  out.index_copy_(0, selected,
      (surface.values.to(torch::kFloat64).index_select(0, selected) - mean).matmul(components));
  return {out, surface.valid.clone(), surface.provenance + "; train-fit PCA"};
}

RidgeProbe::RidgeProbe(const FeatureSurface &training, const torch::Tensor &labels,
                     double penalty) : normalizer(training) {
  require(std::isfinite(penalty) && penalty > 0, "ridge penalty must be finite positive");
  labels_valid(labels, training.values.size(0));
  const auto selected = rows(training);
  const auto y_labels = labels.index_select(0, selected);
  require(y_labels.eq(0).any().item<bool>() && y_labels.eq(1).any().item<bool>(),
          "probe training requires both classes");
  const auto x = normalizer.transform(training).values.index_select(0, selected);
  auto y = torch::zeros({selected.numel(), 2}, torch::kFloat64);
  y.scatter_(1, y_labels.unsqueeze(1), 1.0);
  intercept = y.mean(0);
  const auto target = y - intercept;
  if (x.size(1) > x.size(0)) {
    auto gram = x.matmul(x.transpose(0, 1)) + penalty * torch::eye(x.size(0), x.options());
    weights = x.transpose(0, 1).matmul(at::linalg_solve(gram, target));
  } else {
    auto gram = x.transpose(0, 1).matmul(x) + penalty * torch::eye(x.size(1), x.options());
    weights = at::linalg_solve(gram, x.transpose(0, 1).matmul(target));
  }
  require(torch::isfinite(weights).all().item<bool>(), "ridge fit is nonfinite");
}
torch::Tensor RidgeProbe::predict(const FeatureSurface &surface) const {
  return (normalizer.transform(surface).values.matmul(weights) + intercept).argmax(1);
}

TinyProbe::TinyProbe(const FeatureSurface &training, const torch::Tensor &labels,
    uint64_t seed, int64_t steps, int64_t hidden, double learning_rate) : normalizer(training) {
  labels_valid(labels, training.values.size(0));
  require(steps > 0 && hidden > 0 && std::isfinite(learning_rate) && learning_rate > 0,
          "invalid fixed nonlinear probe configuration");
  const auto selected = rows(training);
  const auto x = normalizer.transform(training).values.index_select(0, selected);
  const auto y = labels.index_select(0, selected);
  require(y.eq(0).any().item<bool>() && y.eq(1).any().item<bool>(), "probe training requires both classes");
  torch::manual_seed(seed);
  w1 = (torch::randn({x.size(1), hidden}, torch::kFloat64) * std::sqrt(2.0 / x.size(1))).set_requires_grad(true);
  b1 = torch::zeros({hidden}, torch::kFloat64).set_requires_grad(true);
  w2 = (torch::randn({hidden, 2}, torch::kFloat64) / std::sqrt(double(hidden))).set_requires_grad(true);
  b2 = torch::zeros({2}, torch::kFloat64).set_requires_grad(true);
  torch::optim::Adam optimizer({w1, b1, w2, b2}, torch::optim::AdamOptions(learning_rate));
  for (int64_t step = 0; step < steps; ++step) {
    optimizer.zero_grad();
    auto logits = torch::tanh(x.matmul(w1) + b1).matmul(w2) + b2;
    auto loss = torch::nn::functional::cross_entropy(logits, y);
    require(torch::isfinite(loss).item<bool>(), "nonlinear probe loss is nonfinite");
    loss.backward(); optimizer.step();
  }
  w1 = w1.detach(); b1 = b1.detach(); w2 = w2.detach(); b2 = b2.detach();
}
torch::Tensor TinyProbe::predict(const FeatureSurface &surface) const {
  torch::NoGradGuard guard;
  auto x = normalizer.transform(surface).values;
  return (torch::tanh(x.matmul(w1) + b1).matmul(w2) + b2).argmax(1);
}

Score score(const torch::Tensor &predictions, const torch::Tensor &labels,
            const torch::Tensor &valid) {
  labels_valid(labels, labels.size(0));
  labels_valid(predictions, labels.size(0));
  require(valid.device().is_cpu() && valid.scalar_type() == torch::kBool &&
          valid.dim() == 1 && valid.size(0) == labels.size(0), "score validity mismatch");
  Score out;
  out.total = labels.size(0); out.valid = valid.sum().item<int64_t>();
  out.correct = predictions.eq(labels).logical_and(valid).sum().item<int64_t>();
  out.coverage = double(out.valid) / out.total; out.supported = out.valid > 0;
  if (out.supported) out.accuracy = double(out.correct) / out.valid;
  return out;
}

GroupedInterval grouped_accuracy_interval(const torch::Tensor &predictions,
    const torch::Tensor &labels, const torch::Tensor &valid,
    const std::vector<std::string> &source_ids, uint64_t seed,
    int64_t replicates, const torch::Tensor &comparison_predictions) {
  const auto measured = score(predictions, labels, valid);
  require(source_ids.size() == size_t(labels.size(0)) && replicates >= 100,
          "invalid source-group uncertainty configuration");
  if (comparison_predictions.defined()) labels_valid(comparison_predictions, labels.size(0));
  std::map<std::string, std::pair<double,int64_t>> groups;
  for (int64_t row = 0; row < labels.size(0); ++row) {
    if (!valid[row].item<bool>()) continue;
    require(!source_ids[row].empty(), "uncertainty source ID is empty");
    const auto truth = labels[row].item<int64_t>();
    double correct = predictions[row].item<int64_t>() == truth ? 1.0 : 0.0;
    if (comparison_predictions.defined())
      correct -= comparison_predictions[row].item<int64_t>() == truth ? 1.0 : 0.0;
    auto &group = groups[source_ids[row]]; group.first += correct; ++group.second;
  }
  GroupedInterval out; out.source_groups = groups.size();
  if (!measured.supported) return out;
  std::vector<std::pair<double,int64_t>> observations;
  double correct = 0;
  for (const auto &[id, group] : groups) { (void)id; observations.push_back(group); correct += group.first; }
  out.estimate = correct / measured.valid;
  if (observations.size() < 2) return out;
  std::mt19937_64 rng(seed); std::uniform_int_distribution<size_t> draw(0, observations.size()-1);
  std::vector<double> values; values.reserve(replicates);
  for (int64_t replicate = 0; replicate < replicates; ++replicate) {
    double numerator = 0; int64_t denominator = 0;
    for (size_t sample = 0; sample < observations.size(); ++sample) {
      const auto &group = observations[draw(rng)]; numerator += group.first; denominator += group.second;
    }
    values.push_back(numerator/denominator);
  }
  std::sort(values.begin(),values.end());
  out.lower = values[size_t(0.025*(values.size()-1))];
  out.upper = values[size_t(0.975*(values.size()-1))]; out.supported = true;
  return out;
}

uint64_t stream_seed(uint64_t seed, uint64_t stream) {
  uint64_t value = seed + 0x9e3779b97f4a7c15ULL * stream;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
std::string task_name(Task task) {
  switch (task) {
  case Task::reversal: return "reversal";
  case Task::level: return "level";
  case Task::amplitude: return "amplitude";
  case Task::lag_sign: return "lag_sign";
  }
  throw std::runtime_error("unknown controlled task");
}

CoordinateDeletionView make_coordinate_deletion_view(const Batch &observations,
    const std::vector<std::string> &source_ids, Task task, uint64_t seed,
    double rate, const std::string &rng_namespace) {
  batch_valid(observations);
  require(source_ids.size() == static_cast<size_t>(observations.data.size(0)),
          "coordinate deletion source rows do not match observations");
  require(std::isfinite(rate) && rate >= 0 && rate <= 1 && !rng_namespace.empty(),
          "coordinate deletion requires a finite rate in [0,1] and a namespace");
  const auto prefix = rng_namespace + "/" + task_name(task) + "/coordinate/";
  std::map<std::string, torch::Tensor> requested;
  std::vector<torch::Tensor> rows;
  for (const auto &id : source_ids) {
    require(!id.empty(), "coordinate deletion source IDs must be nonempty");
    if (!requested.count(id)) {
      uint64_t hash = 14695981039346656037ULL;
      for (const unsigned char c : prefix + std::to_string(id.size()) + ":" + id) {
        hash ^= c; hash *= 1099511628211ULL;
      }
      std::mt19937_64 rng(stream_seed(seed, hash));
      auto erased = torch::zeros({observations.data.size(1), observations.data.size(2),
                                  observations.data.size(3)}, torch::kBool);
      auto flat = erased.reshape({-1}); auto values = flat.accessor<bool, 1>();
      for (int64_t i = 0; i < flat.numel(); ++i)
        values[i] = (rng() >> 11) * (1.0 / 9007199254740992.0) < rate;
      requested.emplace(id, erased);
    }
    rows.push_back(requested.at(id));
  }
  const auto erased = torch::stack(rows);
  const auto mask = observations.feature_mask.logical_and(erased.logical_not());
  return {{torch::where(mask, observations.data, torch::zeros_like(observations.data)).detach().clone(),
           mask.clone()}, erased};
}

namespace {
ControlledProtocol generate_controlled_protocol(Task task, const input_shape_t &shape,
    int64_t training_pairs, int64_t validation_pairs, int64_t testing_pairs,
    uint64_t seed, double missing_rate) {
  require(shape.channel_count > 0 && shape.history_length >= 8 && shape.input_width > 0,
          "controlled fixtures require C>0, H>=8, F>0");
  require(task != Task::lag_sign || shape.channel_count >= 2,
          "lag-sign fixtures require at least two channels");
  require(training_pairs >= 0 && validation_pairs >= 0 && testing_pairs >= 0 &&
          training_pairs <= std::numeric_limits<int64_t>::max() / 6 &&
          validation_pairs <= std::numeric_limits<int64_t>::max() / 6 &&
          testing_pairs <= std::numeric_limits<int64_t>::max() / 6 &&
          std::isfinite(missing_rate) && missing_rate >= 0 && missing_rate < 1,
          "invalid pair counts/missingness");
  const auto total = training_pairs + validation_pairs + testing_pairs;
  require(total > 0, "controlled generation requires at least one populated split");
  std::vector<int64_t> assignments(total);
  std::iota(assignments.begin(), assignments.end(), 0);
  std::mt19937_64 split_rng(stream_seed(seed, 1));
  std::shuffle(assignments.begin(), assignments.end(), split_rng);
  std::vector<int> partitions(total);
  for (int64_t i = 0; i < total; ++i)
    partitions[assignments[i]] = i < training_pairs ? 0 : i < training_pairs + validation_pairs ? 1 : 2;
  std::vector<torch::Tensor> values[3], masks[3];
  std::vector<int64_t> labels[3];
  std::vector<std::string> sources[3];
  constexpr double pi = 3.14159265358979323846;
  for (int64_t source = 0; source < total; ++source) {
    std::mt19937_64 signal_rng(stream_seed(stream_seed(seed, 2), source + 1));
    std::mt19937_64 mask_rng(stream_seed(stream_seed(seed, 3), source + 1));
    std::mt19937_64 order_rng(stream_seed(stream_seed(seed, 4), source + 1));
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> noise(0, 0.005);
    const double phase = uniform(signal_rng) * 2 * pi;
    const double amplitude = 0.8 + 0.4 * uniform(signal_rng);
    auto a = torch::zeros({shape.channel_count, shape.history_length, shape.input_width}, torch::kFloat64);
    auto b = torch::zeros_like(a);
    auto mask = torch::zeros_like(a, torch::kBool);
    auto xa = a.accessor<double, 3>(); auto xb = b.accessor<double, 3>();
    auto observed = mask.accessor<bool, 3>();
    for (int64_t c = 0; c < shape.channel_count; ++c) {
      const double channel_phase = c < 2 ? phase : uniform(signal_rng) * 2 * pi;
      for (int64_t h = 0; h < shape.history_length; ++h) {
        const double t = double(h) / (shape.history_length - 1);
        for (int64_t f = 0; f < shape.input_width; ++f) {
          const double coefficient = 1.0 + 0.1 * f;
          const double base = std::sin(2 * pi * h / 16.0 + channel_phase) * coefficient;
          const double nuisance = noise(signal_rng);
          if (task == Task::reversal) {
            xa[c][h][f] = amplitude * coefficient * (t + 0.35*t*t + 0.07*base) + nuisance;
          } else if (task == Task::level) {
            xa[c][h][f] = base - 2.0 + nuisance; xb[c][h][f] = base + 2.0 + nuisance;
          } else if (task == Task::amplitude) {
            xa[c][h][f] = 0.5 * base + nuisance; xb[c][h][f] = 2.0 * base + nuisance;
          } else {
            const double lag = c == 1 ? 2.0 : 0.0;
            xa[c][h][f] = std::sin(2*pi*(h-lag)/16.0 + channel_phase)*coefficient + nuisance;
            xb[c][h][f] = std::sin(2*pi*(h+lag)/16.0 + channel_phase)*coefficient + nuisance;
          }
          observed[c][h][f] = uniform(mask_rng) >= missing_rate;
        }
      }
    }
    if (task == Task::reversal) b = a.flip({1});
    const int partition = partitions[source];
    const bool swap = uniform(order_rng) > 0.5;
    const auto id = "seed-" + std::to_string(seed) + "/" + task_name(task) + "/source-" + std::to_string(source);
    for (int variant = 0; variant < 2; ++variant) {
      const int label = swap ? 1 - variant : variant;
      values[partition].push_back(label ? b : a);
      masks[partition].push_back(mask.clone()); labels[partition].push_back(label);
      sources[partition].push_back(id);
    }
  }
  ControlledProtocol out{task, seed, shape,
      assemble(values[0], masks[0], labels[0], sources[0]),
      assemble(values[1], masks[1], labels[1], sources[1]),
      assemble(values[2], masks[2], labels[2], sources[2])};
  std::set<std::string> all_sources;
  for (const auto *split : {&out.training, &out.validation, &out.testing})
    if (split->observed.data.defined()) validate_split(*split, all_sources);
  return out;
}
} // namespace

ControlledProtocol make_controlled_protocol(Task task, const input_shape_t &shape,
    int64_t training_pairs, int64_t validation_pairs, int64_t testing_pairs,
    uint64_t seed, double missing_rate) {
  require(training_pairs > 0 && validation_pairs > 0 && testing_pairs > 0,
          "invalid pair counts/missingness");
  return generate_controlled_protocol(task, shape, training_pairs, validation_pairs,
                                     testing_pairs, seed, missing_rate);
}

ControlledProtocol make_controlled_development_protocol(Task task, const input_shape_t &shape,
    int64_t training_pairs, int64_t validation_pairs, uint64_t seed, double missing_rate) {
  require(training_pairs > 0 && validation_pairs > 0,
          "development generation requires positive training and validation pairs");
  return generate_controlled_protocol(task, shape, training_pairs, validation_pairs,
                                     0, seed, missing_rate);
}

ControlledDataset make_controlled_test_dataset(Task task, const input_shape_t &shape,
    int64_t testing_pairs, uint64_t fresh_seed, double missing_rate) {
  require(testing_pairs > 0, "test-only generation requires positive testing pairs");
  return generate_controlled_protocol(task, shape, 0, 0, testing_pairs,
                                     fresh_seed, missing_rate).testing;
}

void validate_protocol(const ControlledProtocol &protocol) {
  std::set<std::string> all_sources;
  for (const auto *split : {&protocol.training, &protocol.validation, &protocol.testing})
    validate_split(*split, all_sources);
}

OracleResult raw_oracle(Task task, const Batch &observations) {
  batch_valid(observations);
  const auto x = observations.data.to(torch::kFloat64);
  const auto mask = observations.feature_mask;
  auto predictions = torch::zeros({x.size(0)}, torch::kInt64);
  auto valid = torch::zeros({x.size(0)}, torch::kBool);
  for (int64_t b = 0; b < x.size(0); ++b) {
    if (task == Task::lag_sign) {
      double positive = 0, negative = 0; int64_t npositive = 0, nnegative = 0;
      for (int64_t h = 2; h < x.size(2) - 2; ++h) {
        if (!mask[b][1][h][0].item<bool>()) continue;
        if (mask[b][0][h+2][0].item<bool>()) {
          positive += x[b][1][h][0].item<double>() * x[b][0][h+2][0].item<double>(); ++npositive;
        }
        if (mask[b][0][h-2][0].item<bool>()) {
          negative += x[b][1][h][0].item<double>() * x[b][0][h-2][0].item<double>(); ++nnegative;
        }
      }
      valid[b] = npositive >= 4 && nnegative >= 4;
      if (valid[b].item<bool>()) predictions[b] = positive/npositive > negative/nnegative;
    } else {
      auto selected = x[b].masked_select(mask[b]);
      if (task == Task::reversal) {
        auto t = torch::arange(x.size(2), torch::kFloat64).view({1, x.size(2), 1}).expand_as(x[b]);
        const auto observed_t = t.masked_select(mask[b]);
        valid[b] = selected.numel() >= 2 && observed_t.var(false).item<double>() > 0;
        if (valid[b].item<bool>())
          predictions[b] = ((selected-selected.mean())*(observed_t-observed_t.mean())).sum().item<double>() < 0;
      } else {
        valid[b] = selected.numel() > 0;
        if (selected.numel()) predictions[b] = task == Task::level
            ? selected.mean().item<double>() > 0 : selected.pow(2).mean().item<double>() > 0.8;
      }
    }
  }
  return {predictions, valid};
}

OracleResult raw_oracle(Task task, const ControlledDataset &dataset) {
  return raw_oracle(task, dataset.observed);
}

ObservationScaler::ObservationScaler(const Batch &training) {
  batch_valid(training);
  const auto mask = training.feature_mask;
  const auto x = torch::where(mask, training.data.to(torch::kFloat64), torch::zeros_like(training.data, torch::kFloat64));
  counts = mask.sum(std::vector<int64_t>{0, 2}).to(torch::kFloat64);
  require(counts.gt(0).all().item<bool>(), "an input coordinate has no fitted observations");
  mean = x.sum(std::vector<int64_t>{0, 2}) / counts;
  const auto centered = torch::where(mask, x - mean.unsqueeze(0).unsqueeze(2), torch::zeros_like(x));
  scale = (centered.pow(2).sum(std::vector<int64_t>{0, 2}) / counts).sqrt().clamp_min(1e-8);
  require(torch::isfinite(mean).all().item<bool>() && torch::isfinite(scale).all().item<bool>(),
          "input fitted statistics overflow");
}
Batch ObservationScaler::transform(const Batch &batch) const {
  batch_valid(batch);
  require(batch.data.size(1) == mean.size(0) && batch.data.size(3) == mean.size(1), "input scaler schema mismatch");
  const auto safe = torch::where(batch.feature_mask, batch.data.to(torch::kFloat64), torch::zeros_like(batch.data, torch::kFloat64));
  const auto scaled = (safe - mean.unsqueeze(0).unsqueeze(2)) / scale.unsqueeze(0).unsqueeze(2);
  require(torch::isfinite(scaled.masked_select(batch.feature_mask)).all().item<bool>(), "input scaling overflow");
  return {torch::where(batch.feature_mask, scaled, torch::zeros_like(scaled)), batch.feature_mask.clone()};
}

} // namespace embedding::evaluation
