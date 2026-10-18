// SPDX-License-Identifier: MIT
#include "embedding/shared/feature_evaluation.h"
#include "embedding/shared/feature_stress.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
namespace fs = std::filesystem;
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error("[feature evaluation] " + message);
}
std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}
bool safe_name(const std::string &name) {
  return !name.empty() && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}
std::string tier_name(DimensionTier tier) {
  switch (tier) {
  case DimensionTier::native: return "native";
  case DimensionTier::matched_global: return "matched_global";
  case DimensionTier::matched_channels: return "matched_channels";
  }
  throw std::runtime_error("[feature evaluation] unknown dimension tier");
}
std::string kind_name(SurfaceKind kind) {
  switch (kind) {
  case SurfaceKind::global: return "global";
  case SurfaceKind::channel_concatenation: return "channel_concatenation";
  case SurfaceKind::control: return "control";
  }
  throw std::runtime_error("[feature evaluation] unknown surface kind");
}
bool accepts(SurfaceKind kind, DimensionTier tier) {
  if (tier == DimensionTier::native || kind == SurfaceKind::control) return true;
  return (kind == SurfaceKind::global && tier == DimensionTier::matched_global) ||
         (kind == SurfaceKind::channel_concatenation && tier == DimensionTier::matched_channels);
}
template <typename T> std::string number_array(const std::vector<T> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  return out.str() + ']';
}
std::string strings_json(const std::vector<std::string> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  return out.str() + ']';
}
std::string fields_json(const std::map<std::string, std::string> &fields) {
  std::ostringstream out;
  out << '{';
  bool first = true;
  for (const auto &[key, value] : fields) {
    if (!first) out << ',';
    first = false;
    out << quote(key) << ':' << quote(value);
  }
  return out.str() + '}';
}
void write_text(const fs::path &path, const std::string &value) {
  require(!fs::exists(path), "artifact already exists: " + path.string());
  std::ofstream output(path);
  require(bool(output), "cannot create artifact: " + path.string());
  output << value;
  output.close();
  require(bool(output), "cannot save artifact: " + path.string());
}
uint64_t name_stream(const std::string &name) {
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char c : name) { hash ^= c; hash *= 1099511628211ULL; }
  return hash;
}
std::string checksum(const torch::Tensor &value) {
  const auto tensor = value.detach().to(torch::kCPU).contiguous();
  uint64_t hash = 14695981039346656037ULL;
  const auto *data = static_cast<const unsigned char *>(tensor.const_data_ptr());
  const auto bytes = tensor.numel() * tensor.element_size();
  for (int64_t i = 0; i < bytes; ++i) { hash ^= data[i]; hash *= 1099511628211ULL; }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
std::string score_json(const Score &value) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"total\":" << value.total << ",\"valid\":" << value.valid
      << ",\"invalid_or_abstained\":" << value.total - value.valid << ",\"correct\":" << value.correct
      << ",\"coverage\":" << value.coverage << ",\"accuracy\":";
  if (value.supported) out << value.accuracy; else out << "null";
  return out.str() + '}';
}
std::string interval_json(const GroupedInterval &value) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"method\":\"source-group percentile bootstrap, 1000 replicates, 95% interval\","
      << "\"source_groups\":" << value.source_groups << ",\"estimate\":";
  if (value.source_groups) out << value.estimate; else out << "null";
  out << ",\"lower\":";
  if (value.supported) out << value.lower; else out << "null";
  out << ",\"upper\":";
  if (value.supported) out << value.upper; else out << "null";
  return out.str() + '}';
}
std::string population_json(const torch::Tensor &valid, const ControlledDataset &split) {
  std::set<std::string> all, selected, complete;
  std::map<std::string, std::pair<int64_t, int64_t>> counts;
  int64_t classes[2]{0, 0};
  std::set<std::string> class_groups[2];
  for (int64_t row = 0; row < split.labels.size(0); ++row) {
    const auto &id = split.source_ids.at(row);
    all.insert(id);
    ++counts[id].first;
    if (valid[row].item<bool>()) {
      const auto label = split.labels[row].item<int64_t>();
      ++classes[label];
      ++counts[id].second;
      selected.insert(id);
      class_groups[label].insert(id);
    }
  }
  for (const auto &[id, count] : counts) if (count.first == count.second) complete.insert(id);
  const auto total = split.labels.size(0), retained = classes[0] + classes[1];
  std::ostringstream out;
  out << std::setprecision(12) << "{\"total\":" << total << ",\"valid\":" << retained
      << ",\"invalid_or_abstained\":" << total - retained << ",\"coverage\":" << double(retained) / total
      << ",\"class_valid_rows\":[" << classes[0] << ',' << classes[1] << "]"
      << ",\"class_source_groups\":[" << class_groups[0].size() << ',' << class_groups[1].size() << "]"
      << ",\"total_source_groups\":" << all.size() << ",\"valid_source_groups\":" << selected.size()
      << ",\"complete_source_pairs\":" << complete.size() << '}';
  return out.str();
}
std::string diagnostics_json(const FeatureSurface &surface) {
  const auto value = feature_diagnostics(surface);
  require(std::isfinite(value.std_mean) && std::isfinite(value.covariance_effective_rank), "nonfinite representation diagnostics");
  std::ostringstream out;
  out << std::setprecision(12) << "{\"valid_rows\":" << value.valid_rows << ",\"dimensions\":" << value.dimensions
      << ",\"std_mean\":" << value.std_mean << ",\"effective_rank\":";
  if (value.valid_rows >= 2) out << value.covariance_effective_rank; else out << "null";
  return out.str() + '}';
}
void save_surface(const fs::path &path, const FeatureSurface &surface) {
  require(!fs::exists(path), "surface artifact collides with an existing artifact: " + path.string());
  torch::serialize::OutputArchive output;
  output.write("features", surface.values, true);
  output.write("valid", surface.valid, true);
  output.write("provenance", archive::text_tensor(surface.provenance), true);
  archive::save_archive(path.string(), output);
}
void save_split(const fs::path &path, const ControlledDataset &split) {
  // Keep the generator's float64 observations; the legacy input archive casts to float32.
  torch::serialize::OutputArchive output;
  output.write("observed", split.observed.data, true);
  output.write("feature_mask", split.observed.feature_mask, true);
  output.write("clean_diagnostic_only", split.clean.data, true);
  output.write("labels_scoring_only", split.labels, true);
  output.write("source_ids_json", archive::text_tensor(strings_json(split.source_ids)), true);
  archive::save_archive(path.string(), output);
}
std::string manifest_json(const ControlledProtocol &protocol, const EvaluationCard &card) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"format_version\":2,\"generator\":\"paired-controls-v1\",\"card\":\"../evaluation-card.json\","
      << "\"task\":" << quote(task_name(protocol.task)) << ",\"seed\":" << quote(std::to_string(protocol.seed))
      << ",\"shape\":[" << card.shape.channel_count << ',' << card.shape.history_length << ',' << card.shape.input_width
      << "],\"observation_dtype\":\"float64\",\"channel_ids\":" << number_array(card.channel_ids)
      << ",\"feature_units\":" << quote(card.feature_units) << ",\"sampling_interval\":" << card.sampling_interval
      << ",\"measurement_noise_std\":0.005,\"missing_rate\":0.1,\"records\":[";
  bool first = true;
  for (const auto &[name, split] : std::vector<std::pair<std::string, const ControlledDataset *>>{
         {"training", &protocol.training}, {"validation", &protocol.validation}, {"testing", &protocol.testing}}) {
    for (int64_t row = 0; row < split->labels.size(0); ++row) {
      if (!first) out << ',';
      first = false;
      out << "{\"split\":" << quote(name) << ",\"source_id\":" << quote(split->source_ids.at(row))
          << ",\"label\":" << split->labels[row].item<int64_t>() << ",\"endpoint\":"
          << (card.shape.history_length - 1) * card.sampling_interval << ",\"raw_support\":[0,"
          << (card.shape.history_length - 1) * card.sampling_interval << "]}";
    }
  }
  out << "],\"archives\":{\"training\":\"controlled-training.pt\",\"validation\":\"controlled-validation.pt\",\"testing\":\"controlled-testing.pt\"},"
      << "\"checksums_fnv1a64\":{\"training_values\":" << quote(checksum(protocol.training.observed.data))
      << ",\"validation_values\":" << quote(checksum(protocol.validation.observed.data))
      << ",\"testing_values\":" << quote(checksum(protocol.testing.observed.data))
      << ",\"training_mask\":" << quote(checksum(protocol.training.observed.feature_mask))
      << ",\"validation_mask\":" << quote(checksum(protocol.validation.observed.feature_mask))
      << ",\"testing_mask\":" << quote(checksum(protocol.testing.observed.feature_mask)) << "},"
      << "\"seed_policy\":\"independent group split, signal, observation mask, pair order and named probe streams; provider initialization seed is declared\"}";
  return out.str();
}
Batch clone_batch(const Batch &batch) { return {batch.data.detach().clone(), batch.feature_mask.clone()}; }
FeatureMap control_features(const Batch &batch, const ObservationScaler &scaler) {
  const auto scaled = scaler.transform(batch);
  return {{"raw", {torch::cat({scaled.data.flatten(1), batch.feature_mask.to(torch::kFloat64).flatten(1)}, 1),
                    batch.feature_mask.flatten(2).any(2).all(1), "raw histories plus mask; training-only float64 per-channel/feature scaling"}},
          {"mask_metadata", {batch.feature_mask.to(torch::kFloat64).flatten(1), torch::ones({batch.data.size(0)}, torch::kBool),
                    "observation mask coordinates; fixed channel IDs, units, history and endpoint are label independent"}}};
}
void validate_description(const std::string &name, const SurfaceDescription &description, const EvaluationCard &card) {
  require(safe_name(name), "unsafe surface name: " + name);
  (void)kind_name(description.kind);
  require(!description.support_rule.empty(), "surface support rule is empty: " + name);
  if (description.kind == SurfaceKind::channel_concatenation) {
    const std::set<int64_t> actual(description.channel_order.begin(), description.channel_order.end());
    const std::set<int64_t> expected(card.channel_ids.begin(), card.channel_ids.end());
    require(description.channel_order.size() == card.channel_ids.size() && actual == expected,
            "concatenation channel order is not a permutation of card channel IDs: " + name);
  } else require(description.channel_order.empty(), "channel order is only valid for concatenated surfaces: " + name);
}
FeatureMap extract_checked(const FeatureProvider &provider, const Batch &batch, const EvaluationCard &card,
                           const FeatureMap *training = nullptr) {
  // A const Tensor is a mutable handle. Isolate the protocol from provider writes.
  auto extracted = provider.extract(clone_batch(batch));
  require(extracted.size() == provider.surfaces.size(), "provider changed its declared surface keys");
  for (auto &[name, surface] : extracted) {
    const auto found = provider.surfaces.find(name);
    require(found != provider.surfaces.end(), "undeclared provider surface: " + name);
    validate_description(name, found->second, card);
    validate_features(surface);
    require(surface.values.size(0) == batch.data.size(0), "provider row count mismatch: " + name);
    if (found->second.kind == SurfaceKind::channel_concatenation)
      require(surface.values.size(1) % card.shape.channel_count == 0, "concatenation width is not divisible by channel count: " + name);
    const auto complete_provenance = surface.provenance + "; " + provider.provenance;
    if (training) {
      const auto &fitted = training->at(name);
      require(surface.values.size(1) == fitted.values.size(1) && surface.values.scalar_type() == fitted.values.scalar_type(),
              "provider changed its feature dimension or dtype across splits: " + name);
      require(complete_provenance == fitted.provenance, "provider changed surface provenance across splits: " + name);
    }
    surface.values = surface.values.detach().clone();
    surface.valid = surface.valid.clone();
    surface.provenance = complete_provenance;
  }
  return extracted;
}
void merge(FeatureMap &target, FeatureMap source) {
  for (auto &[name, value] : source) require(target.emplace(name, std::move(value)).second, "duplicate surface: " + name);
}
torch::Tensor shuffled_labels(const torch::Tensor &labels, const torch::Tensor &valid, uint64_t seed) {
  const auto rows = valid.nonzero().flatten();
  std::vector<int64_t> order(rows.numel());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 random(seed);
  std::shuffle(order.begin(), order.end(), random);
  auto out = labels.clone();
  out.index_copy_(0, rows, labels.index_select(0, rows).index_select(0, torch::tensor(order, torch::kInt64)));
  return out;
}
void save_fit(const fs::path &path, const FeatureNormalizer &normalizer, const std::optional<TrainPca> &pca,
              const RidgeProbe &ridge, const TinyProbe &tiny, const RidgeProbe &shuffled) {
  require(!fs::exists(path), "fit artifact collides with an existing artifact: " + path.string());
  torch::serialize::OutputArchive output;
  output.write("feature_mean", normalizer.mean, true); output.write("feature_scale", normalizer.scale, true);
  output.write("fitted_rows", torch::tensor(normalizer.fitted_rows), true);
  if (pca) {
    output.write("pca_mean", pca->mean, true); output.write("pca_components", pca->components, true);
    output.write("pca_singular_values", pca->singular_values, true);
    output.write("pca_numerical_rank", torch::tensor(pca->numerical_rank), true);
  }
  output.write("ridge_mean", ridge.normalizer.mean, true); output.write("ridge_scale", ridge.normalizer.scale, true);
  output.write("ridge_weights", ridge.weights, true); output.write("ridge_intercept", ridge.intercept, true);
  output.write("tiny_mean", tiny.normalizer.mean, true); output.write("tiny_scale", tiny.normalizer.scale, true);
  output.write("tiny_w1", tiny.w1, true); output.write("tiny_b1", tiny.b1, true);
  output.write("tiny_w2", tiny.w2, true); output.write("tiny_b2", tiny.b2, true);
  output.write("shuffled_mean", shuffled.normalizer.mean, true); output.write("shuffled_scale", shuffled.normalizer.scale, true);
  output.write("shuffled_weights", shuffled.weights, true); output.write("shuffled_intercept", shuffled.intercept, true);
  archive::save_archive(path.string(), output);
}
struct FittedResult {
  std::string status, reason;
  torch::Tensor validation_valid, test_valid;
  torch::Tensor ridge_validation, ridge_test, tiny_validation, tiny_test;
};
using ResultKey = std::pair<std::string, DimensionTier>;
using Results = std::map<ResultKey, FittedResult>;
FeatureNormalizer frozen_copy(const FeatureNormalizer &fitted) {
  auto copy = fitted;
  copy.mean = fitted.mean.detach().clone(); copy.scale = fitted.scale.detach().clone();
  return copy;
}
TrainPca frozen_copy(const TrainPca &fitted) {
  auto copy = fitted;
  copy.mean = fitted.mean.detach().clone(); copy.components = fitted.components.detach().clone();
  copy.singular_values = fitted.singular_values.detach().clone();
  return copy;
}
RidgeProbe frozen_copy(const RidgeProbe &fitted) {
  auto copy = fitted; copy.normalizer = frozen_copy(fitted.normalizer);
  copy.weights = fitted.weights.detach().clone(); copy.intercept = fitted.intercept.detach().clone();
  return copy;
}
TinyProbe frozen_copy(const TinyProbe &fitted) {
  auto copy = fitted; copy.normalizer = frozen_copy(fitted.normalizer);
  copy.w1 = fitted.w1.detach().clone(); copy.b1 = fitted.b1.detach().clone();
  copy.w2 = fitted.w2.detach().clone(); copy.b2 = fitted.b2.detach().clone();
  return copy;
}
std::vector<torch::Tensor> tensor_snapshots(const std::vector<torch::Tensor> &values) {
  std::vector<torch::Tensor> snapshots;
  snapshots.reserve(values.size());
  for (const auto &value : values) snapshots.push_back(value.detach().clone());
  return snapshots;
}
void require_unchanged(const std::vector<torch::Tensor> &values,
                       const std::vector<torch::Tensor> &snapshots, const std::string &label) {
  require(values.size() == snapshots.size(), label + " snapshot size changed");
  for (size_t i = 0; i < values.size(); ++i)
    require(torch::equal(values[i], snapshots[i]), label + " fitted tensor changed during stress");
}
std::string measure_features(const fs::path &directory, const FeatureMap &training, const FeatureMap &validation,
    const FeatureMap &testing, const std::map<std::string, SurfaceDescription> &descriptions,
    const ControlledProtocol &protocol, const EvaluationCard &card, Results &results,
    StressReadouts *stress_readouts = nullptr) {
  std::ostringstream report;
  report << '[';
  bool first = true;
  for (const auto &[name, train] : training) {
    const auto &val = validation.at(name), &test = testing.at(name);
    const auto &description = descriptions.at(name);
    save_surface(directory / (name + "-training.pt"), train);
    save_surface(directory / (name + "-validation.pt"), val);
    save_surface(directory / (name + "-testing.pt"), test);
    const auto diagnostic = diagnostics_json(test);
    const auto valid_labels = protocol.training.labels.masked_select(train.valid);
    const bool supported_fit = valid_labels.numel() >= 2 && valid_labels.eq(0).any().item<bool>() && valid_labels.eq(1).any().item<bool>();
    std::optional<FeatureNormalizer> normalizer;
    std::optional<FeatureSurface> scaled;
    if (supported_fit) { normalizer.emplace(train); scaled.emplace(normalizer->transform(train)); }
    for (const auto tier : {DimensionTier::native, DimensionTier::matched_global, DimensionTier::matched_channels}) {
      if (!accepts(description.kind, tier)) continue;
      if (!first) report << ',';
      first = false;
      const auto tier_text = tier_name(tier);
      auto &result = results[{name, tier}];
      result.validation_valid = val.valid; result.test_valid = test.valid;
      FrozenStressReadout *stress = nullptr;
      if (stress_readouts) {
        stress = &(*stress_readouts)[{name, tier}];
        stress->native_valid = test.valid.detach().clone();
      }
      report << "{\"surface\":" << quote(name) << ",\"kind\":" << quote(kind_name(description.kind))
          << ",\"tier\":" << quote(tier_text) << ",\"native_dimensions\":" << train.values.size(1)
          << ",\"channel_order\":" << number_array(description.channel_order)
          << ",\"support_rule\":" << quote(description.support_rule) << ",\"provenance\":" << quote(train.provenance)
          << ",\"training_population\":" << population_json(train.valid, protocol.training)
          << ",\"validation_population\":" << population_json(val.valid, protocol.validation)
          << ",\"testing_population\":" << population_json(test.valid, protocol.testing)
          << ",\"native_representation_diagnostics\":" << diagnostic;
      if (!supported_fit) {
        result.status = "unsupported_fit";
        result.reason = "requires at least two valid training rows and both training classes";
        if (stress) { stress->status = result.status; stress->reason = result.reason; }
        report << ",\"status\":" << quote(result.status) << ",\"reason\":" << quote(result.reason) << '}';
        continue;
      }
      std::optional<TrainPca> pca;
      if (tier != DimensionTier::native) {
        const auto width = tier == DimensionTier::matched_global ? card.matched_global_width : card.matched_channel_width;
        // Matching always uses PCA, even when the requested width equals native width.
        try { pca.emplace(*scaled, width); }
        catch (const std::exception &error) {
          const std::string reason = error.what();
          if (reason.find("PCA dimensions exceed") == std::string::npos) throw;
          result.status = "unsupported_compression"; result.reason = reason;
          if (stress) { stress->status = result.status; stress->reason = result.reason; }
          report << ",\"status\":" << quote(result.status) << ",\"reason\":" << quote(result.reason) << '}';
          continue;
        }
      }
      auto fit = *scaled, heldout = normalizer->transform(test), valid = normalizer->transform(val);
      if (pca) { fit = pca->transform(fit); heldout = pca->transform(heldout); valid = pca->transform(valid); }
      const auto named = name + "/" + tier_text;
      const auto probe_seed = stream_seed(protocol.seed, name_stream("tiny/" + named));
      const auto shuffle_seed = stream_seed(protocol.seed, name_stream("shuffle/" + named));
      const RidgeProbe ridge(fit, protocol.training.labels);
      const TinyProbe tiny(fit, protocol.training.labels, probe_seed);
      const RidgeProbe shuffled(fit, shuffled_labels(protocol.training.labels, fit.valid, shuffle_seed));
      result.status = "measured";
      result.ridge_validation = ridge.predict(valid); result.ridge_test = ridge.predict(heldout);
      result.tiny_validation = tiny.predict(valid); result.tiny_test = tiny.predict(heldout);
      if (stress) {
        stress->status = result.status; stress->probe_dimensions = fit.values.size(1);
        stress->native_ridge = result.ridge_test.detach().clone();
        stress->native_tiny = result.tiny_test.detach().clone();
        auto normalization = frozen_copy(*normalizer);
        std::optional<TrainPca> compression;
        if (pca) compression.emplace(frozen_copy(*pca));
        auto ridge_probe = frozen_copy(ridge);
        auto tiny_probe = frozen_copy(tiny);
        std::vector<torch::Tensor> fit_tensors{normalization.mean, normalization.scale,
            ridge_probe.normalizer.mean, ridge_probe.normalizer.scale, ridge_probe.weights, ridge_probe.intercept,
            tiny_probe.normalizer.mean, tiny_probe.normalizer.scale, tiny_probe.w1, tiny_probe.b1,
            tiny_probe.w2, tiny_probe.b2};
        if (compression) {
          fit_tensors.push_back(compression->mean); fit_tensors.push_back(compression->components);
          fit_tensors.push_back(compression->singular_values);
        }
        const auto snapshots = tensor_snapshots(fit_tensors);
        // Preserve the ordinary outer scaling/PCA followed by each probe's own
        // train-fit scaling. Captures own detached copies and never fit again.
        stress->predict = [normalization = std::move(normalization), compression = std::move(compression),
                           ridge_probe = std::move(ridge_probe), tiny_probe = std::move(tiny_probe),
                           fit_tensors = std::move(fit_tensors), snapshots](
            const FeatureSurface &surface) {
          torch::NoGradGuard no_grad;
          require_unchanged(fit_tensors, snapshots, "frozen readout");
          auto transformed = normalization.transform(surface);
          if (compression) transformed = compression->transform(transformed);
          const auto predictions = StressPredictions{transformed.valid.detach().clone(),
              ridge_probe.predict(transformed).detach().clone(), tiny_probe.predict(transformed).detach().clone()};
          require_unchanged(fit_tensors, snapshots, "frozen readout");
          return predictions;
        };
        const auto intact = stress->predict(test);
        require(torch::equal(intact.valid, stress->native_valid) &&
                torch::equal(intact.ridge, stress->native_ridge) &&
                torch::equal(intact.tiny, stress->native_tiny),
                "frozen readout changed ordinary intact predictions or validity");
      }
      save_fit(directory / (name + "-" + tier_text + "-fit.pt"), *normalizer, pca, ridge, tiny, shuffled);
      report << ",\"status\":\"measured\",\"probe_dimensions\":" << fit.values.size(1)
          << ",\"valid_fitted_training_rows\":" << normalizer->fitted_rows
          << ",\"probe_seed\":" << quote(std::to_string(probe_seed)) << ",\"shuffled_label_seed\":" << quote(std::to_string(shuffle_seed));
      if (pca) report << ",\"pca_numerical_rank\":" << pca->numerical_rank;
      report << ",\"ridge_test\":" << score_json(score(result.ridge_test, protocol.testing.labels, heldout.valid))
          << ",\"ridge_validation\":" << score_json(score(result.ridge_validation, protocol.validation.labels, valid.valid))
          << ",\"tiny_secondary_test\":" << score_json(score(result.tiny_test, protocol.testing.labels, heldout.valid))
          << ",\"tiny_secondary_validation\":" << score_json(score(result.tiny_validation, protocol.validation.labels, valid.valid))
          << ",\"tiny_minus_ridge_paired_grouped_uncertainty\":" << interval_json(grouped_accuracy_interval(
             result.tiny_test, protocol.testing.labels, heldout.valid, protocol.testing.source_ids,
             stream_seed(protocol.seed, name_stream("secondary/" + named)), 1000, result.ridge_test))
          << ",\"shuffled_training_labels_test\":" << score_json(score(shuffled.predict(heldout), protocol.testing.labels, heldout.valid)) << '}';
    }
  }
  return report.str() + ']';
}
std::string pair_results_json(const EvaluationCard &card, const ControlledProtocol &protocol, const Results &results) {
  std::ostringstream out;
  out << '[';
  bool first = true;
  for (const auto &pair : card.comparisons) {
    if (!first) out << ',';
    first = false;
    const auto &left = results.at({pair.left, pair.tier}), &right = results.at({pair.right, pair.tier});
    const auto test_common = left.test_valid.logical_and(right.test_valid);
    const auto val_common = left.validation_valid.logical_and(right.validation_valid);
    out << "{\"comparison\":" << quote(pair.id) << ",\"candidate\":" << quote(pair.left) << ",\"comparator\":" << quote(pair.right)
        << ",\"tier\":" << quote(tier_name(pair.tier)) << ",\"population_rule\":\"candidate AND comparator only; unrelated providers excluded\","
        << "\"testing_common_population\":" << population_json(test_common, protocol.testing)
        << ",\"validation_common_population\":" << population_json(val_common, protocol.validation);
    if (left.status != "measured" || right.status != "measured") {
      out << ",\"status\":\"unsupported_pair\",\"candidate_status\":" << quote(left.status)
          << ",\"comparator_status\":" << quote(right.status) << '}';
      continue;
    }
    const auto interval_seed = stream_seed(protocol.seed, name_stream("pair/" + pair.id));
    out << ",\"status\":\"measured\",\"ridge_candidate_test\":" << score_json(score(left.ridge_test, protocol.testing.labels, test_common))
        << ",\"ridge_comparator_test\":" << score_json(score(right.ridge_test, protocol.testing.labels, test_common))
        << ",\"ridge_candidate_validation\":" << score_json(score(left.ridge_validation, protocol.validation.labels, val_common))
        << ",\"ridge_comparator_validation\":" << score_json(score(right.ridge_validation, protocol.validation.labels, val_common))
        << ",\"ridge_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(
           left.ridge_test, protocol.testing.labels, test_common, protocol.testing.source_ids, interval_seed, 1000, right.ridge_test))
        << ",\"tiny_secondary_candidate_test\":" << score_json(score(left.tiny_test, protocol.testing.labels, test_common))
        << ",\"tiny_secondary_comparator_test\":" << score_json(score(right.tiny_test, protocol.testing.labels, test_common))
        << ",\"tiny_secondary_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(
           left.tiny_test, protocol.testing.labels, test_common, protocol.testing.source_ids, interval_seed, 1000, right.tiny_test)) << '}';
  }
  return out.str() + ']';
}
} // namespace

void validate_evaluation_card(const EvaluationCard &card) {
  require(!card.id.empty(), "card identity is empty");
  require(card.version == 2 && card.policy_version == "1.1",
          "only implemented card version 2 and policy version 1.1 are supported");
  require(card.stage == "development", "only development cards are supported; confirmation policy gates are incomplete");
  require(card.shape.channel_count > 0 && card.shape.history_length >= 8 && card.shape.input_width > 0,
          "controlled card requires C>0, H>=8, F>0");
  require(card.shape.device.is_cpu() && card.shape.dtype == torch::kFloat64,
          "controlled card requires CPU float64 raw observations; providers own any conversion");
  require(card.channel_ids.size() == size_t(card.shape.channel_count) &&
          std::set<int64_t>(card.channel_ids.begin(), card.channel_ids.end()).size() == card.channel_ids.size(), "card channel IDs must be unique and complete");
  std::istringstream units(card.feature_units);
  std::string item;
  int64_t unit_count = 0;
  while (std::getline(units, item, ',')) { require(!item.empty(), "empty feature unit"); ++unit_count; }
  require(!card.feature_units.empty() && card.feature_units.back() != ',' && unit_count == card.shape.input_width, "feature units must have one entry per feature");
  require(std::isfinite(card.sampling_interval) && card.sampling_interval > 0 &&
          std::isfinite((card.shape.history_length - 1) * card.sampling_interval), "invalid sampling interval or endpoint");
  require(card.matched_global_width > 0 && card.matched_channel_width > 0 && card.train_pairs >= 2 &&
          card.validation_pairs > 0 && card.test_pairs > 0 && card.threads > 0, "invalid card dimensions or run budget");
  require(card.train_pairs <= std::numeric_limits<int64_t>::max() / 6 &&
          card.validation_pairs <= std::numeric_limits<int64_t>::max() / 6 &&
          card.test_pairs <= std::numeric_limits<int64_t>::max() / 6, "pair count overflow");
  require(!card.seeds.empty() && std::set<uint64_t>(card.seeds.begin(), card.seeds.end()).size() == card.seeds.size(), "card seeds must be nonempty and unique");
  require(!card.tasks.empty(), "card requires a task");
  std::set<Task> tasks;
  for (const auto task : card.tasks) {
    (void)task_name(task);
    require(tasks.insert(task).second, "duplicate card task");
    require(task != Task::lag_sign || card.shape.channel_count >= 2, "lag-sign task requires at least two channels");
  }
  std::set<std::string> ids;
  for (const auto &pair : card.comparisons) {
    require(safe_name(pair.id) && safe_name(pair.left) && safe_name(pair.right) && pair.left != pair.right,
            "invalid comparison identity or surface names");
    require(ids.insert(pair.id).second, "duplicate comparison ID");
    (void)tier_name(pair.tier);
  }
}
std::string evaluation_card_json(const EvaluationCard &card) {
  validate_evaluation_card(card);
  std::ostringstream out;
  out << std::setprecision(17) << "{\"id\":" << quote(card.id) << ",\"version\":" << card.version
      << ",\"policy_version\":" << quote(card.policy_version) << ",\"stage\":" << quote(card.stage)
      << ",\"shape\":[" << card.shape.channel_count << ',' << card.shape.history_length << ',' << card.shape.input_width
      << "],\"input_dtype\":" << quote(card.shape.dtype == torch::kFloat64 ? "float64" : "float32")
      << ",\"device\":\"cpu\",\"channel_ids\":" << number_array(card.channel_ids) << ",\"feature_units\":" << quote(card.feature_units)
      << ",\"sampling_interval\":" << card.sampling_interval << ",\"matched_global_width\":" << card.matched_global_width
      << ",\"matched_channel_width\":" << card.matched_channel_width << ",\"train_pairs\":" << card.train_pairs
      << ",\"validation_pairs\":" << card.validation_pairs << ",\"test_pairs\":" << card.test_pairs << ",\"threads\":" << card.threads
      << ",\"seeds\":[";
  for (size_t i = 0; i < card.seeds.size(); ++i) { if (i) out << ','; out << quote(std::to_string(card.seeds[i])); }
  out << "],\"tasks\":[";
  for (size_t i = 0; i < card.tasks.size(); ++i) { if (i) out << ','; out << quote(task_name(card.tasks[i])); }
  out << "],\"comparisons\":[";
  for (size_t i = 0; i < card.comparisons.size(); ++i) {
    if (i) out << ',';
    const auto &pair = card.comparisons[i];
    out << "{\"id\":" << quote(pair.id) << ",\"candidate\":" << quote(pair.left) << ",\"comparator\":" << quote(pair.right)
        << ",\"tier\":" << quote(tier_name(pair.tier)) << '}';
  }
  out << "],\"probe\":{\"ridge_penalty\":1,\"tiny_secondary\":{\"hidden\":16,\"activation\":\"tanh\",\"steps\":100,\"learning_rate\":0.01}},"
      << "\"acceptance\":{\"status\":\"incomplete_development_only\",\"missing_contracts\":[\"primary metric and thresholds\",\"abstention and minimum support\",\"resource cost, latency and memory\",\"robustness suite\",\"decoder scale/shuffle/zero interventions\",\"confirmation data and multiplicity\"]}}";
  return out.str();
}
void run_feature_evaluation(const EvaluationRun &run, const std::vector<FeatureProviderFactory> &factories) {
  validate_evaluation_card(run.card);
  const auto &card = run.card;
  const fs::path output(run.output_directory);
  require(!output.empty() && !fs::exists(output), "output must name a new artifact directory");
  for (const auto &factory : factories) require(bool(factory), "empty provider factory");
  fs::create_directories(output);
  // Freeze the complete card before generating observations or calling any provider.
  write_text(output / "evaluation-card.json", evaluation_card_json(card));
  if (run.stress_sweep) write_text(output / "stress-card.json", fixed_readout_stress_card_json(card));
  torch::set_num_threads(card.threads);
  std::ostringstream report;
  report << "{\"format_version\":2,\"protocol\":\"generic-controlled-pairs-v2\",\"evaluation_card\":\"evaluation-card.json\","
      << "\"policy_version\":" << quote(card.policy_version) << ",\"stage\":" << quote(card.stage)
      << ",\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"interpretation\":\"development evidence; no acceptance or historical performance claim\","
      << "\"fit_policy\":\"source groups split before paired observations; providers receive cloned training observations and metadata only; fitting/PCA/probes use valid training rows only\","
      << "\"pair_policy\":\"declared candidate/comparator intersections only; full surface coverage remains separately reported\","
      << "\"checksum_policy\":\"FNV-1a is a reproducibility checksum, not cryptographic provenance\","
      << "\"incomplete_policy\":[\"promotion thresholds and primary metrics\",\"cost latency memory\",\"robustness\",\"decoder interventions\",\"confirmation\"],\"runs\":[";
  bool first = true;
  std::ostringstream stress_report;
  bool first_stress = true;
  if (run.stress_sweep)
    stress_report << "{\"format_version\":1,\"protocol\":\"fixed-readout-stress-v1\","
        << "\"evaluation_card\":\"evaluation-card.json\",\"stress_card\":\"stress-card.json\","
        << "\"policy_version\":" << quote(card.policy_version) << ",\"stage\":" << quote(card.stage)
        << ",\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
        << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
        << ",\"interpretation\":\"test-only development stress diagnostics; ordinary train-fit models/scalers/PCA/readouts stay fixed; no acceptance claim\","
        << "\"runs\":[";
  for (const auto seed : card.seeds) for (const auto task : card.tasks) {
    const auto directory = output / ("seed-" + std::to_string(seed) + "-" + task_name(task));
    fs::create_directory(directory);
    const auto protocol = make_controlled_protocol(task, card.shape, card.train_pairs, card.validation_pairs, card.test_pairs, seed);
    validate_protocol(protocol);
    write_text(directory / "manifest.json", manifest_json(protocol, card));
    save_split(directory / "controlled-training.pt", protocol.training);
    save_split(directory / "controlled-validation.pt", protocol.validation);
    save_split(directory / "controlled-testing.pt", protocol.testing);
    const ObservationScaler scaler(protocol.training.observed);
    torch::serialize::OutputArchive scaler_archive;
    scaler_archive.write("mean", scaler.mean, true); scaler_archive.write("scale", scaler.scale, true);
    scaler_archive.write("counts", scaler.counts, true);
    archive::save_archive((directory / "raw-control-scaler.pt").string(), scaler_archive);
    auto training = control_features(protocol.training.observed, scaler);
    auto validation = control_features(protocol.validation.observed, scaler);
    auto testing = control_features(protocol.testing.observed, scaler);
    std::map<std::string, SurfaceDescription> descriptions{
        {"raw", {SurfaceKind::control, "every channel has at least one observed coordinate", {}}},
        {"mask_metadata", {SurfaceKind::control, "all examples; mask coordinates and fixed label-independent metadata", {}}}};
    std::ostringstream audits;
    audits << '[';
    std::vector<FeatureProvider> frozen_providers;
    if (run.stress_sweep) frozen_providers.reserve(factories.size());
    for (size_t index = 0; index < factories.size(); ++index) {
      const ProviderFitInput input{clone_batch(protocol.training.observed), card.shape, seed, protocol.training.source_ids,
          card.channel_ids, card.feature_units, card.id + "/v" + std::to_string(card.version) + "/" + task_name(task),
          card.sampling_interval, (card.shape.history_length - 1) * card.sampling_interval};
      const auto provider = factories[index](input);
      require(bool(provider.extract) && !provider.surfaces.empty() && !provider.provenance.empty(), "provider requires extraction, declared surfaces and provenance");
      for (const auto &[name, description] : provider.surfaces) {
        validate_description(name, description, card);
        require(descriptions.emplace(name, description).second, "duplicate provider surface: " + name);
      }
      const auto assets = directory / ("provider-" + std::to_string(index));
      fs::create_directory(assets);
      if (provider.save_assets) provider.save_assets(assets.string());
      const auto audit = "{\"provider_index\":" + std::to_string(index) + ",\"provenance\":" + quote(provider.provenance)
          + ",\"fit_seed\":" + quote(std::to_string(seed)) + ",\"fit_training_rows\":" + std::to_string(protocol.training.labels.size(0))
          + ",\"audit_fields\":" + fields_json(provider.audit_fields) + '}';
      write_text(assets / "provider-audit.json", audit);
      if (index) audits << ',';
      audits << audit;
      const auto provider_train = extract_checked(provider, protocol.training.observed, card);
      const auto provider_validation = extract_checked(provider, protocol.validation.observed, card, &provider_train);
      const auto provider_test = extract_checked(provider, protocol.testing.observed, card, &provider_train);
      merge(training, provider_train); merge(validation, provider_validation); merge(testing, provider_test);
      if (run.stress_sweep) frozen_providers.push_back(provider);
    }
    audits << ']';
    for (const auto &pair : card.comparisons) {
      require(descriptions.count(pair.left) && descriptions.count(pair.right), "comparison references an unavailable surface: " + pair.id);
      require(accepts(descriptions.at(pair.left).kind, pair.tier) && accepts(descriptions.at(pair.right).kind, pair.tier), "comparison tier conflicts with a typed surface: " + pair.id);
    }
    Results results;
    StressReadouts stress_readouts;
    const auto features = measure_features(directory, training, validation, testing, descriptions, protocol, card, results,
                                          run.stress_sweep ? &stress_readouts : nullptr);
    const auto oracle = raw_oracle(task, protocol.testing.observed);
    const auto oracle_score = score(oracle.predictions, protocol.testing.labels, oracle.valid);
    require(oracle_score.supported && oracle_score.accuracy >= .95, "legal-observation oracle did not establish controlled task solvability");
    if (run.stress_sweep) {
      const std::vector<torch::Tensor> scaler_tensors{scaler.mean, scaler.scale, scaler.counts};
      const auto scaler_snapshots = tensor_snapshots(scaler_tensors);
      const StressExtractor extractor = [&frozen_providers, &scaler, &card, &training](const Batch &batch) {
        auto extracted = control_features(clone_batch(batch), scaler);
        for (const auto &provider : frozen_providers)
          merge(extracted, extract_checked(provider, batch, card, &training));
        return extracted;
      };
      const auto stress_json = run_fixed_readout_stress(directory.string(), card, protocol, descriptions, stress_readouts, extractor);
      require_unchanged(scaler_tensors, scaler_snapshots, "raw control scaler");
      if (!first_stress) stress_report << ',';
      first_stress = false;
      const auto relative = directory.filename().string() + "/stress";
      stress_report << "{\"seed\":" << quote(std::to_string(seed)) << ",\"task\":" << quote(task_name(task))
          << ",\"artifact_directory\":" << quote(relative) << ",\"report_file\":" << quote(relative + "/report.json")
          << ",\"results\":" << stress_json << '}';
    }
    if (!first) report << ',';
    first = false;
    report << "{\"seed\":" << quote(std::to_string(seed)) << ",\"task\":" << quote(task_name(task))
        << ",\"artifact_directory\":" << quote(directory.filename().string()) << ",\"provider_audits\":" << audits.str()
        << ",\"raw_oracle_test\":" << score_json(oracle_score) << ",\"oracle_population\":" << population_json(oracle.valid, protocol.testing)
        << ",\"features\":" << features << ",\"pair_comparisons\":" << pair_results_json(card, protocol, results) << '}';
  }
  report << "]}";
  write_text(output / "report.json", report.str());
  if (run.stress_sweep) {
    stress_report << "]}";
    write_text(output / "stress-report.json", stress_report.str());
  }
}
} // namespace embedding::evaluation
