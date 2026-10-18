// SPDX-License-Identifier: MIT
#include "embedding/shared/learning_curve.h"
#include <ATen/Context.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
namespace fs = std::filesystem;
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[learning curve] " + message);
}
bool safe_name(const std::string &name) {
  return !name.empty() && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}
std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
template <typename T> std::string numbers(const std::vector<T> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  return out.str() + ']';
}
std::string strings(const std::vector<std::string> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  return out.str() + ']';
}
std::string fields(const std::map<std::string, std::string> &values) {
  std::ostringstream out;
  out << '{';
  bool first = true;
  for (const auto &[key, value] : values) {
    if (!first) out << ',';
    first = false;
    out << quote(key) << ':' << quote(value);
  }
  return out.str() + '}';
}
void write_text(const fs::path &path, const std::string &text) {
  require(!fs::exists(path), "artifact exists: " + path.string());
  std::ofstream output(path);
  require(bool(output), "cannot create artifact: " + path.string());
  output << text;
  output.close();
  require(bool(output), "cannot save artifact: " + path.string());
}
void save_archive(const fs::path &path, torch::serialize::OutputArchive &output) {
  require(!fs::exists(path), "archive exists: " + path.string());
  archive::save_archive(path.string(), output);
}
uint64_t named_stream(const std::string &name) {
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char c : name) { hash ^= c; hash *= 1099511628211ULL; }
  return hash;
}
std::string checksum(const torch::Tensor &value) {
  const auto tensor = value.detach().to(torch::kCPU).contiguous();
  const auto *bytes = static_cast<const unsigned char *>(tensor.const_data_ptr());
  uint64_t hash = 14695981039346656037ULL;
  for (int64_t i = 0; i < tensor.numel() * tensor.element_size(); ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
Batch legal_clone(const Batch &batch) {
  return {torch::where(batch.feature_mask, batch.data.detach(), torch::zeros_like(batch.data)).clone(),
          batch.feature_mask.clone()};
}
std::string tier_name(DimensionTier tier) {
  if (tier == DimensionTier::native) return "native";
  if (tier == DimensionTier::matched_global) return "matched_global";
  if (tier == DimensionTier::matched_channels) return "matched_channels";
  throw std::runtime_error("[learning curve] unknown tier");
}
std::string score_json(const Score &score) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"total\":" << score.total << ",\"valid\":" << score.valid
      << ",\"correct\":" << score.correct << ",\"coverage\":" << score.coverage << ",\"accuracy\":";
  if (score.supported) out << score.accuracy; else out << "null";
  return out.str() + '}';
}
std::string interval_json(const GroupedInterval &interval) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"method\":\"source-group percentile bootstrap; within-run conditional on fitted checkpoint/readouts\","
      << "\"replicates\":1000,\"confidence\":0.95,\"source_groups\":" << interval.source_groups << ",\"estimate\":";
  if (interval.source_groups) out << interval.estimate; else out << "null";
  out << ",\"lower\":";
  if (interval.supported) out << interval.lower; else out << "null";
  out << ",\"upper\":";
  if (interval.supported) out << interval.upper; else out << "null";
  return out.str() + '}';
}
std::string population_json(const torch::Tensor &valid, const ControlledDataset &split) {
  std::set<std::string> all, selected, complete;
  std::map<std::string, std::pair<int64_t, int64_t>> groups;
  int64_t classes[2]{0, 0};
  for (int64_t row = 0; row < valid.size(0); ++row) {
    const auto &source = split.source_ids.at(row);
    all.insert(source); ++groups[source].first;
    if (valid[row].item<bool>()) {
      selected.insert(source); ++groups[source].second; ++classes[split.labels[row].item<int64_t>()];
    }
  }
  for (const auto &[id, count] : groups) if (count.first == count.second) complete.insert(id);
  std::ostringstream out;
  out << std::setprecision(17) << "{\"total_rows\":" << valid.size(0) << ",\"valid_rows\":" << classes[0] + classes[1]
      << ",\"class_valid_rows\":[" << classes[0] << ',' << classes[1] << "],\"total_source_groups\":" << all.size()
      << ",\"valid_source_groups\":" << selected.size() << ",\"complete_source_pairs\":" << complete.size()
      << ",\"coverage\":" << double(classes[0] + classes[1]) / valid.size(0) << '}';
  return out.str();
}
void validate_split(const ControlledDataset &split, const LearningCurveRun &run, int64_t pairs,
                    std::set<std::string> &all_sources) {
  require(split.observed.data.defined() && split.observed.data.device().is_cpu() &&
          split.observed.data.scalar_type() == torch::kFloat64 &&
          split.observed.data.sizes() == torch::IntArrayRef({2 * pairs, run.shape.channel_count, run.shape.history_length, run.shape.input_width}) &&
          split.observed.feature_mask.defined() && split.observed.feature_mask.device().is_cpu() &&
          split.observed.feature_mask.scalar_type() == torch::kBool &&
          split.observed.feature_mask.sizes() == split.observed.data.sizes() &&
          torch::isfinite(split.observed.data.masked_select(split.observed.feature_mask)).all().item<bool>() &&
          split.labels.defined() && split.labels.device().is_cpu() &&
          split.labels.scalar_type() == torch::kInt64 && split.labels.sizes() == torch::IntArrayRef({2 * pairs}) &&
          split.labels.ge(0).logical_and(split.labels.le(1)).all().item<bool>() &&
          split.source_ids.size() == size_t(2 * pairs), "controlled split shape/dtype/label contract");
  std::map<std::string, std::vector<int64_t>> groups;
  for (int64_t row = 0; row < 2 * pairs; ++row) groups[split.source_ids.at(row)].push_back(row);
  require(groups.size() == size_t(pairs), "controlled source pair count differs");
  for (const auto &[id, rows] : groups) {
    require(!id.empty() && rows.size() == 2 && split.labels[rows[0]].item<int64_t>() != split.labels[rows[1]].item<int64_t>() &&
            torch::equal(split.observed.feature_mask[rows[0]], split.observed.feature_mask[rows[1]]) &&
            all_sources.insert(id).second, "source groups overlap or paired labels/masks differ");
  }
}
void save_split(const fs::path &path, const ControlledDataset &split) {
  torch::serialize::OutputArchive output;
  output.write("observed", split.observed.data, true);
  output.write("feature_mask", split.observed.feature_mask, true);
  output.write("labels_scoring_only", split.labels, true);
  output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
  save_archive(path, output);
}
std::string split_manifest(const ControlledDataset &split, const std::string &name) {
  return "{\"split\":" + quote(name) + ",\"source_ids\":" + strings(split.source_ids) +
      ",\"observed_checksum_fnv1a64\":" + quote(checksum(split.observed.data)) +
      ",\"mask_checksum_fnv1a64\":" + quote(checksum(split.observed.feature_mask)) +
      ",\"labels_checksum_fnv1a64\":" + quote(checksum(split.labels)) + '}';
}
void validate_run(const LearningCurveRun &run, const std::vector<NamedCurveFactory> &factories) {
  EvaluationCard card;
  card.shape = run.shape; card.channel_ids = run.channel_ids; card.feature_units = run.feature_units;
  card.sampling_interval = run.sampling_interval; card.train_pairs = run.train_pairs;
  card.validation_pairs = run.validation_pairs; card.test_pairs = run.test_pairs;
  card.matched_global_width = run.matched_global_width; card.matched_channel_width = run.matched_channel_width;
  card.seeds = run.seeds; card.tasks = {Task::lag_sign}; card.threads = run.threads;
  validate_evaluation_card(card);
  require(run.patch_length > 0 && run.shape.history_length % run.patch_length == 0 &&
          run.shape.history_length / run.patch_length >= 3 && std::isfinite(run.huber_delta) && run.huber_delta > 0,
          "curve reconstruction requires at least3whole patches and finite positive Huber delta");
  require(run.milestones.size() >= 2 && run.milestones.front() == 0, "curve requires initialization plus nonzero milestones");
  for (size_t i = 1; i < run.milestones.size(); ++i)
    require(run.milestones[i] > run.milestones[i - 1], "milestones must be strictly increasing absolute budgets");
  require(factories.size() == 2, "common curve selection requires exactly two architectures");
  std::set<std::string> names;
  for (const auto &factory : factories)
    require(safe_name(factory.name) && names.insert(factory.name).second && !factory.recipe.empty() && bool(factory.factory),
            "curve factory requires unique safe name, recipe and callback");
}
std::string card_json(const LearningCurveRun &run, const std::vector<NamedCurveFactory> &factories) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"version\":1,\"protocol\":\"validation-selected-learning-curve-v1\","
      << "\"stage\":\"development\",\"policy_version\":\"1.1\",\"task\":\"lag_sign\","
      << "\"shape\":[" << run.shape.channel_count << ',' << run.shape.history_length << ',' << run.shape.input_width
      << "],\"dtype\":\"float64\",\"channel_ids\":" << numbers(run.channel_ids) << ",\"feature_units\":" << quote(run.feature_units)
      << ",\"sampling_interval\":" << run.sampling_interval << ",\"patch_length\":" << run.patch_length
      << ",\"huber_delta\":" << run.huber_delta << ",\"train_pairs\":" << run.train_pairs
      << ",\"validation_pairs\":" << run.validation_pairs << ",\"test_pairs\":" << run.test_pairs
      << ",\"matched_global_width\":" << run.matched_global_width << ",\"matched_channel_width\":" << run.matched_channel_width
      << ",\"milestones\":" << numbers(run.milestones) << ",\"seeds\":[";
  for (size_t i = 0; i < run.seeds.size(); ++i) { if (i) out << ','; out << quote(std::to_string(run.seeds[i])); }
  out << "],\"threads\":" << run.threads << ",\"missing_rate\":0.1,\"providers\":[";
  for (size_t i = 0; i < factories.size(); ++i) {
    if (i) out << ',';
    out << "{\"name\":" << quote(factories[i].name) << ",\"recipe\":" << quote(factories[i].recipe) << '}';
  }
  out << "],\"selection\":{\"metric\":\"validation matched-channel ridge accuracy\","
      << "\"weighting\":\"equal architecture and seed weights; within-seed two-architecture common validity\","
      << "\"candidates\":\"all nonzero milestones; any missing/unsupported fit or validation class excludes entire budget\","
      << "\"tie_rule\":\"smallest exact-tied budget\",\"initialization_is_diagnostic_only\":true},"
      << "\"readout_recipe\":{\"ridge_penalty\":1,\"tiny_secondary\":{\"activation\":\"tanh\",\"hidden\":16,\"updates\":100,\"optimizer\":\"Adam\",\"learning_rate\":0.01},"
      << "\"seed\":\"stream_seed(development_seed,FNV1a64(surface/tier));same named initialization across architectures and milestones\","
      << "\"pipeline\":\"valid-training outer normalizer;centered train-fit PCA for matched tiers;each probe own valid-training normalizer\"},"
      << "\"test_policy\":\"generate only after persisted selection; named fresh seed stream_seed(development_seed,0x6375727665746573); chosen exact checkpoint/readouts; no refit\","
      << "\"reconstruction_policy\":\"enumerate every original patch, same hidden patch across channels; observed target cells only; >=2 visible patch groups/channel; cell then channel then example equal means\","
      << "\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"limits\":[\"synthetic development only\",\"common updates do not match parameters or cost\","
      << "\"source-group intervals are within seed conditional on fitted checkpoint/readouts\",\"no across-seed uncertainty or consumer acceptance\"]}";
  return out.str();
}
using Key = std::pair<std::string, DimensionTier>;
struct Readout {
  FeatureNormalizer normalizer;
  std::optional<TrainPca> pca;
  RidgeProbe ridge;
  TinyProbe tiny;
  Readout(FeatureNormalizer normalization, std::optional<TrainPca> compression,
          RidgeProbe linear, TinyProbe nonlinear)
      : normalizer(std::move(normalization)), pca(std::move(compression)),
        ridge(std::move(linear)), tiny(std::move(nonlinear)) {}
  FeatureSurface transform(const FeatureSurface &input) const {
    auto out = normalizer.transform(input);
    if (pca) out = pca->transform(out);
    return out;
  }
};
struct FeatureResult {
  std::string status, reason;
  torch::Tensor training_valid, validation_valid, ridge_validation, tiny_validation;
  std::unique_ptr<Readout> readout;
};
struct TestPrimary { torch::Tensor valid, ridge, tiny; };
// TinyProbe and provider snapshot construction may reseed default generators.
// Measurement must not consume/reset a trainer's persistent CPU/CUDA RNG.
struct MeasurementRngIsolation {
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  MeasurementRngIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  ~MeasurementRngIsolation() noexcept {
    try {
      for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]);
    } catch (...) { std::terminate(); }
  }
};
struct Point {
  CurveProgress progress;
  CurveSnapshot snapshot;
  FeatureMap training, validation;
  std::map<Key, FeatureResult> results;
  std::string focal, global, validation_json;
  fs::path directory;
};
struct Architecture {
  std::string name;
  std::map<int64_t, Point> points;
};
struct SeedState {
  uint64_t seed;
  ControlledProtocol development;
  std::vector<Architecture> architectures;
  std::unique_ptr<RidgeProbe> mask_probe;
};
FeatureMap extract(const FeatureProvider &provider, const Batch &batch, const LearningCurveRun &run,
                   const FeatureMap *training = nullptr) {
  const auto out = provider.extract(legal_clone(batch));
  require(out.size() == 2 && provider.surfaces.size() == 2, "curve snapshot must export exactly one global and one channel concatenation");
  FeatureMap copied;
  int globals = 0, concatenations = 0;
  for (const auto &[name, surface] : out) {
    require(safe_name(name) && provider.surfaces.count(name), "unsafe or undeclared feature surface");
    const auto &description = provider.surfaces.at(name);
    require(!description.support_rule.empty(), "empty surface support");
    if (description.kind == SurfaceKind::global) {
      ++globals; require(description.channel_order.empty(), "global surface cannot carry channel order");
    } else {
      require(description.kind == SurfaceKind::channel_concatenation, "curve surfaces must be typed global/concatenation");
      ++concatenations;
      require(description.channel_order.size() == run.channel_ids.size() &&
              std::set<int64_t>(description.channel_order.begin(), description.channel_order.end()) ==
              std::set<int64_t>(run.channel_ids.begin(), run.channel_ids.end()) &&
              surface.values.size(1) % run.shape.channel_count == 0, "concatenation channel order/width mismatch");
    }
    validate_features(surface);
    require(surface.values.size(0) == batch.data.size(0), "feature extraction row mismatch");
    FeatureSurface copy{surface.values.detach().clone(), surface.valid.clone(), surface.provenance + "; " + provider.provenance};
    if (training) {
      require(training->count(name), "snapshot changed feature key");
      const auto &fitted = training->at(name);
      require(fitted.values.size(1) == copy.values.size(1) && fitted.values.scalar_type() == copy.values.scalar_type() &&
              fitted.provenance == copy.provenance, "snapshot changed feature shape/dtype/provenance across splits");
    }
    copied.emplace(name, std::move(copy));
  }
  require(globals == 1 && concatenations == 1, "curve requires unique global and concatenation surfaces");
  return copied;
}
void save_surface(const fs::path &path, const FeatureSurface &surface) {
  torch::serialize::OutputArchive output;
  output.write("features", surface.values, true); output.write("valid", surface.valid, true);
  output.write("provenance", archive::text_tensor(surface.provenance), true);
  save_archive(path, output);
}
void save_fit(const fs::path &path, const Readout &readout) {
  torch::serialize::OutputArchive output;
  output.write("feature_mean", readout.normalizer.mean, true); output.write("feature_scale", readout.normalizer.scale, true);
  output.write("fitted_rows", torch::tensor(readout.normalizer.fitted_rows), true);
  if (readout.pca) {
    output.write("pca_mean", readout.pca->mean, true); output.write("pca_components", readout.pca->components, true);
    output.write("pca_singular_values", readout.pca->singular_values, true);
    output.write("pca_numerical_rank", torch::tensor(readout.pca->numerical_rank), true);
  }
  output.write("ridge_mean", readout.ridge.normalizer.mean, true); output.write("ridge_scale", readout.ridge.normalizer.scale, true);
  output.write("ridge_weights", readout.ridge.weights, true); output.write("ridge_intercept", readout.ridge.intercept, true);
  output.write("tiny_mean", readout.tiny.normalizer.mean, true); output.write("tiny_scale", readout.tiny.normalizer.scale, true);
  output.write("tiny_w1", readout.tiny.w1, true); output.write("tiny_b1", readout.tiny.b1, true);
  output.write("tiny_w2", readout.tiny.w2, true); output.write("tiny_b2", readout.tiny.b2, true);
  save_archive(path, output);
}
void save_predictions(const fs::path &path, const torch::Tensor &ridge, const torch::Tensor &tiny,
                      const torch::Tensor &valid, const ControlledDataset &split) {
  torch::serialize::OutputArchive output;
  output.write("ridge", ridge, true); output.write("tiny_secondary", tiny, true); output.write("valid", valid, true);
  output.write("labels_scoring_only", split.labels, true);
  output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
  save_archive(path, output);
}
std::string fit_readouts(Point &point, const ControlledProtocol &protocol, const LearningCurveRun &run, uint64_t seed) {
  std::ostringstream out;
  out << '[';
  bool first = true;
  for (const auto &[name, train] : point.training) {
    const auto &validation = point.validation.at(name);
    const auto &description = point.snapshot.features.surfaces.at(name);
    const bool global = description.kind == SurfaceKind::global;
    if (global) point.global = name; else point.focal = name;
    save_surface(point.directory / (name + "-training.pt"), train);
    save_surface(point.directory / (name + "-validation.pt"), validation);
    const auto labels = protocol.training.labels.masked_select(train.valid);
    const bool fit_supported = labels.numel() >= 2 && labels.eq(0).any().item<bool>() && labels.eq(1).any().item<bool>();
    for (const auto tier : {DimensionTier::native, global ? DimensionTier::matched_global : DimensionTier::matched_channels}) {
      if (!first) out << ',';
      first = false;
      auto &result = point.results[{name, tier}];
      result.training_valid = train.valid.clone(); result.validation_valid = validation.valid.clone();
      out << "{\"surface\":" << quote(name) << ",\"tier\":" << quote(tier_name(tier))
          << ",\"kind\":" << quote(global?"global":"channel_concatenation")
          << ",\"channel_order\":" << numbers(description.channel_order)
          << ",\"support_rule\":" << quote(description.support_rule) << ",\"provenance\":" << quote(train.provenance)
          << ",\"native_dimensions\":" << train.values.size(1) << ",\"training_population\":" << population_json(train.valid, protocol.training)
          << ",\"validation_population\":" << population_json(validation.valid, protocol.validation);
      if (!fit_supported) {
        result.status = "unsupported_fit"; result.reason = "valid training rows require both classes and >=2rows";
      } else {
        FeatureNormalizer normalizer(train);
        auto fitted = normalizer.transform(train);
        std::optional<TrainPca> pca;
        if (tier != DimensionTier::native) {
          try { pca.emplace(fitted, global ? run.matched_global_width : run.matched_channel_width); }
          catch (const std::exception &error) { result.status = "unsupported_compression"; result.reason = error.what(); }
        }
        if (result.status.empty()) {
          if (pca) fitted = pca->transform(fitted);
          const auto probe_seed = stream_seed(seed, named_stream(name + "/" + tier_name(tier)));
          RidgeProbe ridge(fitted, protocol.training.labels);
          TinyProbe tiny(fitted, protocol.training.labels, probe_seed);
          result.readout = std::make_unique<Readout>(std::move(normalizer), std::move(pca), std::move(ridge), std::move(tiny));
          const auto transformed = result.readout->transform(validation);
          result.ridge_validation = result.readout->ridge.predict(transformed);
          result.tiny_validation = result.readout->tiny.predict(transformed);
          result.status = "measured";
          save_fit(point.directory / (name + "-" + tier_name(tier) + "-fit.pt"), *result.readout);
          save_predictions(point.directory / (name + "-" + tier_name(tier) + "-validation-predictions.pt"),
                           result.ridge_validation, result.tiny_validation, transformed.valid, protocol.validation);
          out << ",\"probe_dimensions\":" << fitted.values.size(1) << ",\"probe_seed\":" << quote(std::to_string(probe_seed));
          out << ",\"fit_artifact\":" << quote(name+"-"+tier_name(tier)+"-fit.pt");
          if (result.readout->pca) out << ",\"pca_numerical_rank\":" << result.readout->pca->numerical_rank;
          out << ",\"ridge_validation\":" << score_json(score(result.ridge_validation, protocol.validation.labels, transformed.valid))
              << ",\"tiny_secondary_validation\":" << score_json(score(result.tiny_validation, protocol.validation.labels, transformed.valid));
        }
      }
      out << ",\"status\":" << quote(result.status);
      if (!result.reason.empty()) out << ",\"reason\":" << quote(result.reason);
      out << '}';
    }
  }
  return out.str() + ']';
}
std::string reconstruction(const fs::path &path, const ControlledDataset &split,
                           const CurveSnapshot &snapshot, const LearningCurveRun &run) {
  const auto B = split.observed.data.size(0), C = run.shape.channel_count;
  const auto H = run.shape.history_length, F = run.shape.input_width, K = H / run.patch_length;
  std::vector<torch::Tensor> predictions, targets, queries, requested, visible_masks, eligibility;
  for (int64_t trial = 0; trial < K; ++trial) {
    auto artificial = torch::zeros_like(split.observed.feature_mask);
    artificial.narrow(2, trial * run.patch_length, run.patch_length).fill_(true);
    const auto requested_query = split.observed.feature_mask.logical_and(artificial);
    const auto visible = split.observed.feature_mask.logical_and(artificial.logical_not());
    const auto fixed_eligible = visible.reshape({B, C, K, -1}).any(-1).sum(-1).ge(2)
        .logical_and(requested_query.flatten(2).any(2));
    // A channel that cannot leave2observed patch groups has no reconstruction
    // trial. This prevents invalid whole-patch masks reaching a strict provider.
    artificial = artificial.logical_and(fixed_eligible.unsqueeze(-1).unsqueeze(-1));
    CurveReconstruction value;
    {
      torch::NoGradGuard no_grad;
      value = snapshot.reconstruct(legal_clone(split.observed), artificial.clone());
    }
    for (const auto &tensor : {value.prediction, value.target})
      require(tensor.defined() && tensor.device().is_cpu() && tensor.is_floating_point() &&
              !tensor.requires_grad() && tensor.sizes() == split.observed.data.sizes(),
              "reconstruction must return detached CPU standardized prediction/target[B,C,H,F]");
    require(value.eligible.defined() && value.eligible.device().is_cpu() && value.eligible.scalar_type() == torch::kBool &&
            value.eligible.sizes() == torch::IntArrayRef({B, C}) &&
            !value.eligible.logical_and(fixed_eligible.logical_not()).any().item<bool>(),
            "reconstruction eligibility exceeds fixed observed target/visible-patch support");
    const auto query = split.observed.feature_mask.logical_and(artificial)
        .logical_and(value.eligible.unsqueeze(-1).unsqueeze(-1));
    require(torch::isfinite(value.prediction.masked_select(query)).all().item<bool>() &&
            torch::isfinite(value.target.masked_select(query)).all().item<bool>(), "nonfinite standardized reconstruction on valid target");
    predictions.push_back(torch::where(query, value.prediction.to(torch::kFloat64), torch::zeros({B,C,H,F}, torch::kFloat64)));
    targets.push_back(torch::where(query, value.target.to(torch::kFloat64), torch::zeros({B,C,H,F}, torch::kFloat64)));
    queries.push_back(query); requested.push_back(requested_query);
    visible_masks.push_back(split.observed.feature_mask.logical_and(artificial.logical_not()));
    eligibility.push_back(value.eligible.detach().clone());
  }
  const auto prediction = torch::stack(predictions), target = torch::stack(targets);
  const auto query = torch::stack(queries), requested_query = torch::stack(requested);
  const auto error = prediction - target;
  require(torch::isfinite(error.masked_select(query)).all().item<bool>(), "standardized reconstruction difference overflow");
  const auto absolute = error.abs();
  const auto huber = torch::where(absolute.le(run.huber_delta), .5 * error.clamp(-run.huber_delta,run.huber_delta).square(),
                                  run.huber_delta * (absolute - .5 * run.huber_delta));
  const std::vector<int64_t> axes{0,3,4};
  const auto counts = query.sum(axes), channel_valid = counts.gt(0);
  const auto channel_mae = absolute.sum(axes) / counts.clamp_min(1).to(torch::kFloat64);
  const auto channel_huber = huber.sum(axes) / counts.clamp_min(1).to(torch::kFloat64);
  const auto example_valid = channel_valid.any(1);
  const auto channel_count = channel_valid.sum(1).clamp_min(1).to(torch::kFloat64);
  const auto example_mae = channel_mae.sum(1) / channel_count, example_huber = channel_huber.sum(1) / channel_count;
  require(torch::isfinite(example_mae).all().item<bool>() && torch::isfinite(example_huber).all().item<bool>(),
          "reconstruction hierarchical reduction overflow");
  const auto retained = example_valid.sum().item<int64_t>(), target_cells = counts.sum().item<int64_t>();
  const auto full_targets = requested_query.sum().item<int64_t>();
  double mae = 0, loss = 0;
  if (retained) {
    mae = example_mae.masked_select(example_valid).mean().item<double>();
    loss = example_huber.masked_select(example_valid).mean().item<double>();
    require(std::isfinite(mae) && std::isfinite(loss), "reconstruction summary overflow");
  }
  torch::serialize::OutputArchive output;
  output.write("standardized_prediction", prediction, true); output.write("standardized_target", target, true);
  output.write("target_mask", query, true); output.write("requested_observed_target_mask", requested_query, true);
  output.write("visible_mask", torch::stack(visible_masks), true); output.write("trial_channel_eligible", torch::stack(eligibility), true);
  output.write("channel_target_counts", counts, true); output.write("channel_valid", channel_valid, true);
  output.write("channel_standardized_mae", channel_mae, true); output.write("channel_standardized_huber", channel_huber, true);
  output.write("example_valid", example_valid, true); output.write("example_standardized_mae", example_mae, true);
  output.write("example_standardized_huber", example_huber, true);
  output.write("source_ids_json", archive::text_tensor(strings(split.source_ids)), true);
  save_archive(path, output);
  std::ostringstream out;
  out << std::setprecision(17) << "{\"status\":" << quote(retained ? "measured" : "unsupported_zero_support")
      << ",\"units\":\"frozen training-scaler standardized\",\"total_examples\":" << B << ",\"valid_examples\":" << retained
      << ",\"example_coverage\":" << double(retained)/B << ",\"valid_channels\":" << channel_valid.sum().item<int64_t>()
      << ",\"valid_target_cells\":" << target_cells << ",\"requested_observed_target_cells\":" << full_targets
      << ",\"target_coverage\":";
  if (full_targets) out << double(target_cells)/full_targets; else out << "null";
  out << ",\"standardized_mae\":";
  if (retained) out << mae; else out << "null";
  out << ",\"standardized_huber\":";
  if (retained) out << loss; else out << "null";
  out << ",\"query_checksum_fnv1a64\":" << quote(checksum(query))
      << ",\"artifact\":" << quote(path.filename().string()) << '}';
  return out.str();
}
void check_progress(const CurveProgress &progress, int64_t budget, const CurveProgress *previous) {
  require(progress.completed == budget && progress.attempted >= progress.completed && progress.parameter_count > 0 &&
          progress.cuda_parameter_count >= 0 && progress.cuda_parameter_count <= progress.parameter_count &&
          progress.sampled_rows >= 0 && std::isfinite(progress.training_seconds) && progress.training_seconds >= 0 &&
          !progress.training_device.empty() && !progress.preprocessing_id.empty() && !progress.training_dataset_id.empty(),
          "trainer counters/device/scaler/dataset/timing contract");
  if (previous)
    require(progress.attempted >= previous->attempted && progress.sampled_rows >= previous->sampled_rows &&
            progress.training_seconds >= previous->training_seconds && progress.parameter_count == previous->parameter_count &&
            progress.cuda_parameter_count == previous->cuda_parameter_count &&
            progress.training_device == previous->training_device && progress.preprocessing_id == previous->preprocessing_id &&
            progress.training_dataset_id == previous->training_dataset_id, "trainer reset its absolute counter/model/scaler state");
  else require(progress.completed == 0 && progress.attempted == 0, "point0 must be exact fresh initialization");
  if (previous) {
    require(progress.losses.size() >= previous->losses.size(), "trainer discarded its cumulative loss trace");
    for (size_t i = 0; i < previous->losses.size(); ++i) {
      const auto &a = previous->losses[i], &b = progress.losses[i];
      require(a.attempted == b.attempted && a.completed == b.completed && a.target_cells == b.target_cells &&
              a.loss == b.loss && a.gradient_norm == b.gradient_norm, "trainer changed an earlier counter/loss prefix");
    }
  }
  int64_t attempted = 0, completed = 0;
  for (const auto &loss : progress.losses) {
    require(loss.attempted > attempted && loss.completed > completed && loss.completed <= progress.completed &&
            loss.attempted <= progress.attempted && loss.target_cells > 0 && std::isfinite(loss.loss) &&
            std::isfinite(loss.gradient_norm) && loss.gradient_norm >= 0, "nonfinite/unordered loss trace");
    attempted = loss.attempted; completed = loss.completed;
  }
}
std::string progress_json(const CurveProgress &progress) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"attempted\":" << progress.attempted << ",\"completed\":" << progress.completed
      << ",\"parameter_count\":" << progress.parameter_count << ",\"cuda_parameter_count\":" << progress.cuda_parameter_count
      << ",\"sampled_rows\":" << progress.sampled_rows << ",\"cumulative_training_seconds\":" << progress.training_seconds
      << ",\"training_device\":" << quote(progress.training_device) << ",\"preprocessing_id\":" << quote(progress.preprocessing_id)
      << ",\"training_dataset_id\":" << quote(progress.training_dataset_id)
      << ",\"last_input_cuda\":" << (progress.last_input_cuda ? "true":"false")
      << ",\"last_loss_cuda\":" << (progress.last_loss_cuda ? "true":"false")
      << ",\"finite_gradients\":" << (progress.finite_gradients ? "true":"false")
      << ",\"weights_changed_from_initialization\":" << (progress.weights_changed ? "true":"false") << ",\"loss_trace\":[";
  for (size_t i = 0; i < progress.losses.size(); ++i) {
    if (i) out << ',';
    const auto &loss = progress.losses[i];
    out << "{\"attempted\":" << loss.attempted << ",\"completed\":" << loss.completed
        << ",\"loss\":" << loss.loss << ",\"gradient_norm\":" << loss.gradient_norm << ",\"target_cells\":" << loss.target_cells << '}';
  }
  return out.str() + "]}";
}
struct Candidate { int64_t budget; bool supported; double utility; std::string details; };
Candidate utility(int64_t budget, const std::vector<SeedState> &states) {
  Candidate candidate{budget,true,0,""};
  std::ostringstream out;
  out << '[';
  bool first = true;
  int64_t count = 0;
  for (const auto &state : states) {
    const auto &a = state.architectures[0].points.at(budget), &b = state.architectures[1].points.at(budget);
    const auto &left = a.results.at({a.focal,DimensionTier::matched_channels});
    const auto &right = b.results.at({b.focal,DimensionTier::matched_channels});
    const auto common = left.validation_valid.logical_and(right.validation_valid);
    const auto labels = state.development.validation.labels.masked_select(common);
    const bool supported = left.status == "measured" && right.status == "measured" && labels.numel() >= 2 &&
                           labels.eq(0).any().item<bool>() && labels.eq(1).any().item<bool>();
    candidate.supported = candidate.supported && supported;
    for (size_t architecture = 0; architecture < state.architectures.size(); ++architecture) {
      if (!first) out << ',';
      first = false;
      const auto &model = state.architectures[architecture];
      const auto &point = model.points.at(budget);
      const auto &result = point.results.at({point.focal,DimensionTier::matched_channels});
      out << "{\"seed\":" << quote(std::to_string(state.seed)) << ",\"architecture\":" << quote(model.name)
          << ",\"focal_fit_status\":" << quote(result.status) << ",\"common_population\":"
          << population_json(common,state.development.validation) << ",\"accuracy\":";
      if (supported) {
        const auto measured = score(result.ridge_validation,state.development.validation.labels,common);
        out << measured.accuracy; candidate.utility += measured.accuracy; ++count;
      } else out << "null";
      out << '}';
    }
  }
  if (candidate.supported) {
    require(count == int64_t(2*states.size()), "common budget silently dropped an architecture/seed");
    candidate.utility /= count;
    require(std::isfinite(candidate.utility), "selection utility nonfinite");
  }
  candidate.details = out.str() + ']';
  return candidate;
}
std::string evaluate_test(Point &point, const ControlledDataset &testing, const LearningCurveRun &run, uint64_t seed,
                          TestPrimary &primary) {
  const auto surfaces = extract(point.snapshot.features,testing.observed,run,&point.training);
  std::ostringstream out;
  out << '[';
  bool first = true;
  for (const auto &[key,result] : point.results) {
    if (!first) out << ',';
    first = false;
    const auto &surface = surfaces.at(key.first);
    if (key.second == DimensionTier::native) save_surface(point.directory/(key.first+"-selected-testing.pt"),surface);
    out << "{\"surface\":" << quote(key.first) << ",\"tier\":" << quote(tier_name(key.second))
        << ",\"status\":" << quote(result.status) << ",\"testing_population\":" << population_json(surface.valid,testing);
    if (result.readout) {
      const auto transformed = result.readout->transform(surface);
      const auto ridge = result.readout->ridge.predict(transformed), tiny = result.readout->tiny.predict(transformed);
      if (key.first == point.focal && key.second == DimensionTier::matched_channels)
        primary = {transformed.valid.clone(),ridge.clone(),tiny.clone()};
      save_predictions(point.directory/(key.first+"-"+tier_name(key.second)+"-selected-test-predictions.pt"),ridge,tiny,transformed.valid,testing);
      const auto stream = named_stream(key.first+"/"+tier_name(key.second)+"/selected-test");
      out << ",\"ridge_test\":" << score_json(score(ridge,testing.labels,transformed.valid))
          << ",\"ridge_grouped_interval\":" << interval_json(grouped_accuracy_interval(ridge,testing.labels,transformed.valid,testing.source_ids,stream_seed(seed,stream)))
          << ",\"tiny_secondary_test\":" << score_json(score(tiny,testing.labels,transformed.valid))
          << ",\"tiny_secondary_grouped_interval\":" << interval_json(grouped_accuracy_interval(tiny,testing.labels,transformed.valid,testing.source_ids,stream_seed(seed,stream+1)));
    } else out << ",\"reason\":" << quote(result.reason);
    out << '}';
  }
  return out.str()+']';
}
} // namespace

void run_learning_curve(const LearningCurveRun &run,const std::vector<NamedCurveFactory> &factories) {
  validate_run(run,factories);
  const fs::path output(run.output_directory);
  require(!output.empty() && !fs::exists(output),"output must be a new directory");
  fs::create_directories(output);
  write_text(output/"learning-curve-card.json",card_json(run,factories));
  torch::set_num_threads(run.threads);
  std::vector<SeedState> states;
  states.reserve(run.seeds.size());
  std::ostringstream validation_report;
  validation_report << "{\"version\":1,\"protocol\":\"validation-selected-learning-curve-v1\","
      << "\"stage\":\"development\",\"card\":\"learning-curve-card.json\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"points\":[";
  bool first_point = true;
  for (const auto seed : run.seeds) {
    SeedState state{seed,make_controlled_development_protocol(Task::lag_sign,run.shape,run.train_pairs,run.validation_pairs,seed),{},nullptr};
    require(!state.development.testing.observed.data.defined() && state.development.testing.source_ids.empty(),
            "development generator created testing data before selection");
    std::set<std::string> sources;
    validate_split(state.development.training,run,run.train_pairs,sources);
    validate_split(state.development.validation,run,run.validation_pairs,sources);
    const auto seed_directory = output/("seed-"+std::to_string(seed));
    fs::create_directory(seed_directory);
    save_split(seed_directory/"controlled-training.pt",state.development.training);
    save_split(seed_directory/"controlled-validation.pt",state.development.validation);
    write_text(seed_directory/"development-manifest.json","{\"version\":1,\"testing_generated\":false,\"training\":"+
        split_manifest(state.development.training,"training")+",\"validation\":"+split_manifest(state.development.validation,"validation")+'}');
    const FeatureSurface mask_training{state.development.training.observed.feature_mask.to(torch::kFloat64).flatten(1),
        torch::ones({2*run.train_pairs},torch::kBool),"observation masks only; paired label-independent source masks"};
    state.mask_probe=std::make_unique<RidgeProbe>(mask_training,state.development.training.labels);
    torch::serialize::OutputArchive mask_fit;
    mask_fit.write("mean",state.mask_probe->normalizer.mean,true); mask_fit.write("scale",state.mask_probe->normalizer.scale,true);
    mask_fit.write("ridge_weights",state.mask_probe->weights,true); mask_fit.write("ridge_intercept",state.mask_probe->intercept,true);
    save_archive(seed_directory/"mask-metadata-ridge-fit.pt",mask_fit);
    for (const auto &factory : factories) {
      const ProviderFitInput input{legal_clone(state.development.training.observed),run.shape,seed,
          state.development.training.source_ids,run.channel_ids,run.feature_units,"learning-curve-v1/lag_sign",
          run.sampling_interval,(run.shape.history_length-1)*run.sampling_interval};
      const auto trainer = factory.factory(input);
      require(bool(trainer.train_to) && bool(trainer.save_checkpoint) && bool(trainer.snapshot),"incomplete curve trainer callbacks");
      Architecture architecture{factory.name,{}};
      const auto directory = seed_directory/factory.name;
      fs::create_directory(directory);
      write_text(directory/"trainer-audit.json",fields(trainer.audit_fields));
      for (const auto budget : run.milestones) {
        Point point;
        point.directory = directory/("milestone-"+std::to_string(budget));
        fs::create_directory(point.directory);
        point.progress = trainer.train_to(budget);
        const MeasurementRngIsolation measurement_rng;
        const CurveProgress *previous = architecture.points.empty()?nullptr:&architecture.points.rbegin()->second.progress;
        check_progress(point.progress,budget,previous);
        const auto checkpoint = point.directory/"checkpoint.pt";
        trainer.save_checkpoint(checkpoint.string());
        require(fs::is_regular_file(checkpoint),"trainer did not save the exact resumable checkpoint");
        point.snapshot = trainer.snapshot(checkpoint.string());
        require(bool(point.snapshot.features.extract) && !point.snapshot.features.provenance.empty() &&
                bool(point.snapshot.reconstruct),"incomplete immutable checkpoint snapshot");
        point.training = extract(point.snapshot.features,state.development.training.observed,run);
        point.validation = extract(point.snapshot.features,state.development.validation.observed,run,&point.training);
        if (!architecture.points.empty()) {
          const auto &prior = architecture.points.rbegin()->second;
          for (const auto &[name,surface] : point.training) {
            require(prior.training.count(name),"checkpoint changed declared feature keys");
            const auto &old_surface=prior.training.at(name);
            const auto &old_description=prior.snapshot.features.surfaces.at(name);
            const auto &description=point.snapshot.features.surfaces.at(name);
            require(surface.values.size(1)==old_surface.values.size(1) && surface.values.scalar_type()==old_surface.values.scalar_type() &&
                    description.kind==old_description.kind && description.channel_order==old_description.channel_order &&
                    description.support_rule==old_description.support_rule,"checkpoint changed export geometry/semantics");
          }
        }
        const auto assets = point.directory/"provider-assets";
        fs::create_directory(assets);
        if (point.snapshot.features.save_assets) point.snapshot.features.save_assets(assets.string());
        write_text(assets/"provider-audit.json","{\"provenance\":"+quote(point.snapshot.features.provenance)+
            ",\"audit_fields\":"+fields(point.snapshot.features.audit_fields)+'}');
        const auto features = fit_readouts(point,state.development,run,seed);
        if(!state.architectures.empty()) {
          const auto &reference=state.architectures.front().points.begin()->second;
          require(point.focal==reference.focal && point.global==reference.global,
                  "architectures must share active surface keys for matched named probe initialization");
        }
        const auto train_loss = reconstruction(point.directory/"training-reconstruction.pt",state.development.training,point.snapshot,run);
        const auto val_loss = reconstruction(point.directory/"validation-reconstruction.pt",state.development.validation,point.snapshot,run);
        const auto relative = fs::relative(point.directory,output).generic_string();
        point.validation_json = "{\"seed\":"+quote(std::to_string(seed))+",\"architecture\":"+quote(factory.name)+
            ",\"milestone\":"+std::to_string(budget)+",\"directory\":"+quote(relative)+",\"checkpoint\":\"checkpoint.pt\","
            "\"progress\":"+progress_json(point.progress)+",\"training_reconstruction\":"+train_loss+
            ",\"validation_reconstruction\":"+val_loss+",\"features\":"+features+'}';
        write_text(point.directory/"point.json",point.validation_json);
        if (!first_point) validation_report << ',';
        first_point = false;
        validation_report << point.validation_json;
        const auto &focal = point.results.at({point.focal,DimensionTier::matched_channels});
        std::cout << "curve architecture=" << factory.name << " seed=" << seed << " milestone=" << budget
            << " attempted=" << point.progress.attempted << " completed=" << point.progress.completed
            << " training_seconds=" << point.progress.training_seconds << " validation_primary=";
        if (focal.status=="measured") std::cout << score(focal.ridge_validation,state.development.validation.labels,focal.validation_valid).accuracy;
        else std::cout << focal.status;
        std::cout << '\n' << std::flush;
        architecture.points.emplace(budget,std::move(point));
      }
      state.architectures.push_back(std::move(architecture));
    }
    states.push_back(std::move(state));
  }
  validation_report << "]}";
  write_text(output/"validation-report.json",validation_report.str());
  std::vector<Candidate> candidates;
  std::optional<int64_t> selected;
  double best = -std::numeric_limits<double>::infinity();
  for (const auto budget : run.milestones) {
    if (!budget) continue;
    auto candidate = utility(budget,states);
    if (candidate.supported && candidate.utility > best) { selected=budget; best=candidate.utility; }
    candidates.push_back(std::move(candidate));
  }
  require(selected.has_value(),"no nonzero common budget has all architecture/seed focal readouts and validation classes");
  std::ostringstream selection;
  selection << std::setprecision(17) << "{\"version\":1,\"stage\":\"development\",\"selection_frozen_before_testing\":true,"
      << "\"selected_budget\":" << *selected << ",\"utility\":" << best
      << ",\"metric\":\"mean common-valid validation matched-channel ridge accuracy; all architectures and seeds equal\","
      << "\"tie_rule\":\"smallest exact-tied nonzero budget\",\"candidates\":[";
  for (size_t i=0;i<candidates.size();++i) {
    if (i) selection << ',';
    const auto &candidate=candidates[i];
    selection << "{\"budget\":" << candidate.budget << ",\"status\":" << quote(candidate.supported?"measured":"unsupported_common_budget")
        << ",\"utility\":";
    if(candidate.supported) selection << candidate.utility; else selection << "null";
    selection << ",\"entries\":" << candidate.details << '}';
  }
  selection << "],\"validation_report\":\"validation-report.json\",\"source_fingerprint\":" << quote(run.source_fingerprint) << '}';
  write_text(output/"selection.json",selection.str());
  std::ostringstream report;
  report << "{\"version\":1,\"protocol\":\"validation-selected-learning-curve-v1\",\"stage\":\"development\","
      << "\"card\":\"learning-curve-card.json\",\"validation_report\":\"validation-report.json\",\"selection\":\"selection.json\","
      << "\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"selected_budget\":" << *selected << ",\"interpretation\":\"fresh synthetic development test; no across-seed CI or consumer acceptance\","
      << "\"testing_runs\":[";
  bool first_test=true;
  for(auto &state:states) {
    const MeasurementRngIsolation selected_measurement_rng;
    // Future updates must not mutate an earlier retained snapshot or fitted
    // readout. Verify its original validation witness before drawing test data.
    for(auto &architecture:state.architectures) {
      auto &point=architecture.points.at(*selected);
      const auto witness=extract(point.snapshot.features,state.development.validation.observed,run,&point.training);
      for(const auto &[name,surface]:witness) {
        const auto &original=point.validation.at(name);
        require(torch::equal(surface.valid,original.valid) &&
                torch::equal(surface.values.index_select(0,surface.valid.nonzero().flatten()),
                             original.values.index_select(0,original.valid.nonzero().flatten())),
                "later training mutated the selected checkpoint snapshot");
      }
      for(const auto &[key,result]:point.results) if(result.readout) {
        const auto transformed=result.readout->transform(point.validation.at(key.first));
        require(torch::equal(result.readout->ridge.predict(transformed),result.ridge_validation) &&
                torch::equal(result.readout->tiny.predict(transformed),result.tiny_validation),
                "selected fitted readout changed after its validation score");
      }
    }
    const auto test_seed=stream_seed(state.seed,0x6375727665746573ULL);
    const auto testing=make_controlled_test_dataset(Task::lag_sign,run.shape,run.test_pairs,test_seed);
    std::set<std::string> sources(state.development.training.source_ids.begin(),state.development.training.source_ids.end());
    sources.insert(state.development.validation.source_ids.begin(),state.development.validation.source_ids.end());
    validate_split(testing,run,run.test_pairs,sources);
    const auto directory=output/("seed-"+std::to_string(state.seed));
    save_split(directory/"controlled-selected-testing.pt",testing);
    write_text(directory/"selected-test-manifest.json","{\"fresh_test_seed\":"+quote(std::to_string(test_seed))+
        ",\"selection\":\"../selection.json\",\"testing\":"+split_manifest(testing,"testing")+'}');
    const auto oracle=raw_oracle(Task::lag_sign,testing.observed);
    const auto oracle_score=score(oracle.predictions,testing.labels,oracle.valid);
    require(oracle_score.supported && oracle_score.accuracy>=.95,"selected-test legal raw oracle failed task solvability");
    const FeatureSurface mask_testing{testing.observed.feature_mask.to(torch::kFloat64).flatten(1),
        torch::ones({2*run.test_pairs},torch::kBool),"observation masks only; paired label-independent source masks"};
    const auto mask_prediction=state.mask_probe->predict(mask_testing);
    torch::serialize::OutputArchive mask_predictions;
    mask_predictions.write("ridge",mask_prediction,true); mask_predictions.write("valid",mask_testing.valid,true);
    mask_predictions.write("labels_scoring_only",testing.labels,true);
    mask_predictions.write("source_ids_json",archive::text_tensor(strings(testing.source_ids)),true);
    save_archive(directory/"mask-metadata-selected-test-predictions.pt",mask_predictions);
    if(!first_test) report << ',';
    first_test=false;
    report << "{\"development_seed\":" << quote(std::to_string(state.seed)) << ",\"fresh_test_seed\":" << quote(std::to_string(test_seed))
        << ",\"source_manifest\":" << quote(directory.filename().string()+"/selected-test-manifest.json")
        << ",\"raw_oracle\":" << score_json(oracle_score) << ",\"mask_metadata_ridge\":" << score_json(score(mask_prediction,testing.labels,mask_testing.valid))
        << ",\"mask_metadata_fit\":\"mask-metadata-ridge-fit.pt\",\"architectures\":[";
    std::vector<TestPrimary> primaries;
    for(size_t index=0;index<state.architectures.size();++index) {
      auto &architecture=state.architectures[index];
      auto &point=architecture.points.at(*selected);
      TestPrimary primary;
      const auto test_features=evaluate_test(point,testing,run,state.seed,primary);
      require(primary.valid.defined(),"chosen focal readout missing");
      primaries.push_back(std::move(primary));
      const auto reconstruction_metrics=reconstruction(point.directory/"selected-test-reconstruction.pt",testing,point.snapshot,run);
      if(index) report << ',';
      report << "{\"name\":" << quote(architecture.name) << ",\"checkpoint\":" << quote(fs::relative(point.directory/"checkpoint.pt",output).generic_string())
          << ",\"fitted_readouts_reused\":true,\"features\":" << test_features << ",\"reconstruction\":" << reconstruction_metrics << '}';
    }
    const auto common=primaries[0].valid.logical_and(primaries[1].valid);
    report << "],\"pair\":{\"candidate\":" << quote(state.architectures[1].name) << ",\"comparator\":" << quote(state.architectures[0].name)
        << ",\"tier\":\"matched_channels\",\"common_population\":" << population_json(common,testing)
        << ",\"ridge_candidate\":" << score_json(score(primaries[1].ridge,testing.labels,common))
        << ",\"ridge_comparator\":" << score_json(score(primaries[0].ridge,testing.labels,common))
        << ",\"ridge_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(
            primaries[1].ridge,testing.labels,common,testing.source_ids,stream_seed(state.seed,named_stream("curve/test/pair/ridge")),1000,primaries[0].ridge))
        << ",\"tiny_secondary_candidate\":" << score_json(score(primaries[1].tiny,testing.labels,common))
        << ",\"tiny_secondary_comparator\":" << score_json(score(primaries[0].tiny,testing.labels,common))
        << ",\"tiny_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(
            primaries[1].tiny,testing.labels,common,testing.source_ids,stream_seed(state.seed,named_stream("curve/test/pair/tiny")),1000,primaries[0].tiny)) << "}}";
  }
  report << "]}";
  write_text(output/"report.json",report.str());
}
} // namespace embedding::evaluation
