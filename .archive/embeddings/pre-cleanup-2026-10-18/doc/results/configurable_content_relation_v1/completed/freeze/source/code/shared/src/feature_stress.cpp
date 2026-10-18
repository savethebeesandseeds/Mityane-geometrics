// SPDX-License-Identifier: MIT
#include "embedding/shared/feature_stress.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace embedding::evaluation {
namespace {
namespace fs = std::filesystem;
using Key = std::pair<std::string, DimensionTier>;
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[fixed readout stress] " + message);
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                         << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
template <typename T> std::string array_json(const std::vector<T> &values) {
  std::ostringstream out; out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  return out.str() + ']';
}
std::string strings_json(const std::vector<std::string> &values) {
  std::ostringstream out; out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  return out.str() + ']';
}
std::string tier_name(DimensionTier tier) {
  if (tier == DimensionTier::native) return "native";
  if (tier == DimensionTier::matched_global) return "matched_global";
  if (tier == DimensionTier::matched_channels) return "matched_channels";
  throw std::runtime_error("[fixed readout stress] invalid dimension tier");
}
std::string kind_name(SurfaceKind kind) {
  if (kind == SurfaceKind::global) return "global";
  if (kind == SurfaceKind::channel_concatenation) return "channel_concatenation";
  if (kind == SurfaceKind::control) return "control";
  throw std::runtime_error("[fixed readout stress] invalid surface kind");
}
bool safe_name(const std::string &name) {
  return !name.empty() && name.find_first_not_of(
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}
uint64_t fnv(const std::string &text) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : text) { hash ^= c; hash *= 1099511628211ULL; }
  return hash;
}
uint64_t source_seed(const ControlledProtocol &protocol, const std::string &id,
                     const std::string &stream) {
  // Length-delimited source identity; neither labels nor paired variant enter.
  return stream_seed(protocol.seed, fnv("fixed-readout-stress-v1/" +
      task_name(protocol.task) + "/" + stream + "/" + std::to_string(id.size()) + ":" + id));
}
uint64_t ci_seed(const ControlledProtocol &protocol, const std::string &name) {
  return stream_seed(protocol.seed, fnv("fixed-readout-stress-ci-v1/" +
      task_name(protocol.task) + "/" + name));
}
std::string checksum(const torch::Tensor &value) {
  const auto cpu = value.detach().to(torch::kCPU).contiguous();
  uint64_t hash = 14695981039346656037ULL;
  const auto *bytes = static_cast<const unsigned char *>(cpu.const_data_ptr());
  const auto size = cpu.numel() * cpu.element_size();
  for (int64_t i = 0; i < size; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
  std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
void write_text(const fs::path &path, const std::string &text) {
  require(!fs::exists(path), "refusing to replace artifact: " + path.string());
  std::ofstream output(path); require(bool(output), "cannot create: " + path.string());
  output << text; output.close(); require(bool(output), "cannot save: " + path.string());
}
Batch clone_batch(const Batch &batch) {
  return {batch.data.detach().clone(), batch.feature_mask.clone()};
}
void valid_mask(const torch::Tensor &mask, int64_t rows, const std::string &label) {
  require(mask.defined() && mask.device().is_cpu() && mask.scalar_type() == torch::kBool &&
      mask.dim() == 1 && mask.size(0) == rows, label + " must be CPU bool [B]");
}
void predictions_valid(const StressPredictions &predictions, int64_t rows,
                       const torch::Tensor &surface_valid) {
  valid_mask(predictions.valid, rows, "prediction validity");
  require(!predictions.valid.logical_and(surface_valid.logical_not()).any().item<bool>(),
      "readout predicts outside its stressed feature support");
  for (const auto &p : {predictions.ridge, predictions.tiny})
    require(p.defined() && p.device().is_cpu() && p.scalar_type() == torch::kInt64 &&
        p.dim() == 1 && p.size(0) == rows && p.ge(0).logical_and(p.le(1)).all().item<bool>(),
        "frozen predictions must be CPU int64 binary [B], including invalid placeholders");
}
std::string score_json(const torch::Tensor &predictions, const ControlledDataset &split,
                       const torch::Tensor &valid) {
  const auto value = score(predictions, split.labels, valid);
  std::ostringstream out;
  out << std::setprecision(12) << "{\"total\":" << value.total << ",\"valid\":" << value.valid
      << ",\"invalid_or_abstained\":" << value.total - value.valid << ",\"correct\":" << value.correct
      << ",\"coverage\":" << value.coverage << ",\"conditional_accuracy\":";
  if (value.supported) out << value.accuracy; else out << "null";
  out << ",\"full_population_correct_rate\":" << double(value.correct) / value.total << '}';
  return out.str();
}
std::string population_json(const torch::Tensor &valid, const ControlledDataset &split) {
  valid_mask(valid, split.labels.size(0), "population validity");
  std::set<std::string> all, selected, classes[2], complete;
  std::map<std::string, std::pair<int64_t, int64_t>> rows;
  int64_t class_rows[2]{0, 0};
  for (int64_t row = 0; row < split.labels.size(0); ++row) {
    const auto &id = split.source_ids.at(row);
    all.insert(id); ++rows[id].first;
    if (valid[row].item<bool>()) {
      const auto label = split.labels[row].item<int64_t>();
      ++class_rows[label]; ++rows[id].second; selected.insert(id); classes[label].insert(id);
    }
  }
  for (const auto &[id, count] : rows) if (count.first == count.second) complete.insert(id);
  const auto count = class_rows[0] + class_rows[1], total = split.labels.size(0);
  std::ostringstream out; out << std::setprecision(12)
      << "{\"total\":" << total << ",\"valid\":" << count
      << ",\"coverage\":" << double(count) / total
      << ",\"class_valid_rows\":[" << class_rows[0] << ',' << class_rows[1]
      << "],\"class_source_groups\":[" << classes[0].size() << ',' << classes[1].size()
      << "],\"total_source_groups\":" << all.size() << ",\"valid_source_groups\":" << selected.size()
      << ",\"complete_source_pairs\":" << complete.size() << '}';
  return out.str();
}
std::string interval_json(const GroupedInterval &value) {
  // Conditional point scores remain available with one group; a CI does not.
  std::ostringstream out; out << std::setprecision(12)
      << "{\"method\":\"source-group percentile bootstrap\",\"replicates\":1000,"
         "\"confidence\":0.95,\"source_groups\":" << value.source_groups << ",\"estimate\":";
  if (value.supported) out << value.estimate; else out << "null";
  out << ",\"lower\":"; if (value.supported) out << value.lower; else out << "null";
  out << ",\"upper\":"; if (value.supported) out << value.upper; else out << "null";
  return out.str() + '}';
}
GroupedInterval full_difference(const StressPredictions &left, const StressPredictions &right,
    const ControlledDataset &split, bool tiny, uint64_t seed) {
  // Abstention is an incorrect full-population outcome; the binary-prediction
  // grouped helper forbids -1, so bootstrap explicit correctness differences.
  const auto &a = tiny ? left.tiny : left.ridge;
  const auto &b = tiny ? right.tiny : right.ridge;
  const auto differences = a.eq(split.labels).logical_and(left.valid).to(torch::kFloat64) -
      b.eq(split.labels).logical_and(right.valid).to(torch::kFloat64);
  std::map<std::string, std::pair<double, int64_t>> groups;
  double sum = 0;
  for (int64_t row = 0; row < split.labels.size(0); ++row) {
    const auto value = differences[row].item<double>();
    auto &group = groups[split.source_ids.at(row)]; group.first += value; ++group.second; sum += value;
  }
  GroupedInterval result; result.source_groups = groups.size();
  result.estimate = sum / split.labels.size(0);
  if (groups.size() < 2) return result;
  std::vector<std::pair<double, int64_t>> observations;
  for (const auto &[id, group] : groups) { (void)id; observations.push_back(group); }
  std::mt19937_64 rng(seed); std::uniform_int_distribution<size_t> draw(0, observations.size() - 1);
  std::vector<double> estimates; estimates.reserve(1000);
  for (int replicate = 0; replicate < 1000; ++replicate) {
    double numerator = 0; int64_t denominator = 0;
    for (size_t sample = 0; sample < observations.size(); ++sample) {
      const auto &group = observations[draw(rng)]; numerator += group.first; denominator += group.second;
    }
    estimates.push_back(numerator / denominator);
  }
  std::sort(estimates.begin(), estimates.end());
  result.lower = estimates[size_t(0.025 * (estimates.size() - 1))];
  result.upper = estimates[size_t(0.975 * (estimates.size() - 1))]; result.supported = true;
  return result;
}
struct Case {
  std::string id, family;
  double rate{0};
  int64_t channel_index{-1}, channel_id{0};
};
std::vector<Case> cases(const EvaluationCard &card) {
  std::vector<Case> result{{"intact", "intact", 0}};
  for (const auto &[id, rate] : std::vector<std::pair<std::string, double>>{
           {"010", .1}, {"030", .3}, {"060", .6}, {"090", .9}})
    result.push_back({"random_dropout_" + id, "coordinate_dropout", rate});
  for (const auto &[id, rate] : std::vector<std::pair<std::string, double>>{
           {"025", .25}, {"050", .5}, {"075", .75}})
    result.push_back({"contiguous_" + id, "contiguous_gap", rate});
  for (size_t i = 0; i < card.channel_ids.size(); ++i)
    result.push_back({"channel_id_" + std::to_string(card.channel_ids[i]) + "_absent",
        "channel_absence", 1.0 / card.channel_ids.size(), static_cast<int64_t>(i), card.channel_ids[i]});
  result.push_back({"all_absent", "all_absent", 1});
  return result;
}
std::string case_json(const Case &value) {
  std::ostringstream out; out << std::setprecision(17)
      << "{\"id\":" << quote(value.id) << ",\"family\":" << quote(value.family)
      << ",\"requested_rate\":" << value.rate;
  if (value.channel_index >= 0)
    out << ",\"physical_channel_index\":" << value.channel_index << ",\"semantic_channel_id\":" << value.channel_id;
  return out.str() + '}';
}
torch::Tensor source_erasure(const Case &value, const EvaluationCard &card,
                            const ControlledProtocol &protocol, const std::string &source) {
  auto erased = torch::zeros({card.shape.channel_count, card.shape.history_length,
                              card.shape.input_width}, torch::kBool);
  if (value.family == "contiguous_gap") {
    std::mt19937_64 rng(source_seed(protocol, source, "contiguous_anchor"));
    std::uniform_int_distribution<int64_t> draw(0, card.shape.history_length - 1);
    const auto anchor = draw(rng);
    const auto length = static_cast<int64_t>(std::ceil(value.rate * card.shape.history_length));
    const auto start = std::clamp<int64_t>(anchor - length / 2, 0, card.shape.history_length - length);
    erased.narrow(1, start, length).fill_(true);
  } else if (value.family == "channel_absence") erased.select(0, value.channel_index).fill_(true);
  else if (value.family == "all_absent") erased.fill_(true);
  return erased;
}
struct Corrupted { Batch batch; torch::Tensor erased; };
Corrupted corrupt(const Case &value, const EvaluationCard &card,
                  const ControlledProtocol &protocol) {
  const auto &base = protocol.testing.observed;
  if (value.family == "coordinate_dropout") {
    const auto view = make_coordinate_deletion_view(base, protocol.testing.source_ids,
        protocol.task, protocol.seed, value.rate);
    return {view.observations, view.requested_erasure};
  }
  std::map<std::string, torch::Tensor> requested;
  std::vector<torch::Tensor> rows;
  for (const auto &id : protocol.testing.source_ids) {
    if (!requested.count(id)) requested.emplace(id, source_erasure(value, card, protocol, id));
    rows.push_back(requested.at(id));
  }
  const auto erased = torch::stack(rows);
  const auto mask = base.feature_mask.logical_and(erased.logical_not());
  auto data = torch::where(mask, base.data, torch::zeros_like(base.data)).detach().clone();
  return {{data, mask.clone()}, erased};
}
std::string support_json(const Corrupted &value, const Batch &base,
                         const EvaluationCard &card) {
  const auto total = base.data.numel(), original = base.feature_mask.sum().item<int64_t>();
  const auto observed = value.batch.feature_mask.sum().item<int64_t>();
  const auto requested = value.erased.sum().item<int64_t>();
  const auto channel_counts = value.batch.feature_mask.sum(std::vector<int64_t>{2, 3});
  const auto channel_rows = channel_counts.gt(0).sum(0);
  std::vector<int64_t> observed_channel_rows, absent_channel_rows, coordinates_by_channel;
  for (int64_t channel = 0; channel < card.shape.channel_count; ++channel) {
    const auto supported = channel_rows[channel].item<int64_t>();
    observed_channel_rows.push_back(supported);
    absent_channel_rows.push_back(value.batch.data.size(0) - supported);
    coordinates_by_channel.push_back(channel_counts.select(1, channel).sum().item<int64_t>());
  }
  std::ostringstream out; out << std::setprecision(12)
      << "{\"total_coordinates\":" << total << ",\"base_observed_coordinates\":" << original
      << ",\"requested_erased_coordinates\":" << requested
      << ",\"actual_erased_observed_coordinates\":" << original - observed
      << ",\"retained_observed_coordinates\":" << observed
      << ",\"channel_ids\":" << array_json(card.channel_ids)
      << ",\"observed_channel_rows\":" << array_json(observed_channel_rows)
      << ",\"absent_channel_rows\":" << array_json(absent_channel_rows)
      << ",\"observed_coordinate_counts_by_channel\":" << array_json(coordinates_by_channel)
      << ",\"requested_grid_erasure_fraction\":" << double(requested) / total
      << ",\"observed_retention_fraction\":";
  if (original) out << double(observed) / original; else out << "null";
  return out.str() + '}';
}
void save_corrupted(const fs::path &directory, const Case &condition, const Corrupted &value,
                    const EvaluationCard &card, const ControlledProtocol &protocol) {
  const auto batch_file = condition.id + "-corrupted.pt", manifest_file = condition.id + "-manifest.json";
  torch::serialize::OutputArchive archive;
  archive.write("observed", value.batch.data, true);
  archive.write("feature_mask", value.batch.feature_mask, true);
  archive.write("base_feature_mask", protocol.testing.observed.feature_mask, true);
  archive.write("requested_erasure_mask", value.erased, true);
  archive.write("source_ids_json", embedding::archive::text_tensor(strings_json(protocol.testing.source_ids)), true);
  archive.write("channel_ids", torch::tensor(card.channel_ids, torch::kInt64), true);
  archive.write("case_json", embedding::archive::text_tensor(case_json(condition)), true);
  embedding::archive::save_archive((directory / batch_file).string(), archive);
  std::ostringstream manifest; manifest << "{\"case\":" << case_json(condition)
      << ",\"batch_file\":" << quote(batch_file) << ",\"source_ids\":" << strings_json(protocol.testing.source_ids)
      << ",\"channel_ids\":" << array_json(card.channel_ids)
      << ",\"checksum_policy\":\"FNV-1a64 raw tensor bytes;reproducibility only\","
         "\"checksums_fnv1a64\":{\"observed\":" << quote(checksum(value.batch.data))
      << ",\"feature_mask\":" << quote(checksum(value.batch.feature_mask))
      << ",\"base_feature_mask\":" << quote(checksum(protocol.testing.observed.feature_mask))
      << ",\"requested_erasure_mask\":" << quote(checksum(value.erased)) << "},\"support\":"
      << support_json(value, protocol.testing.observed, card) << ",\"rows\":[";
  for (int64_t row = 0; row < value.batch.data.size(0); ++row) {
    if (row) manifest << ',';
    std::vector<int64_t> per_channel;
    for (int64_t channel = 0; channel < card.shape.channel_count; ++channel)
      per_channel.push_back(value.batch.feature_mask[row][channel].sum().item<int64_t>());
    manifest << "{\"row\":" << row << ",\"source_id\":" << quote(protocol.testing.source_ids.at(row))
        << ",\"base_observed\":" << protocol.testing.observed.feature_mask[row].sum().item<int64_t>()
        << ",\"retained_observed\":" << value.batch.feature_mask[row].sum().item<int64_t>()
        << ",\"requested_erased\":" << value.erased[row].sum().item<int64_t>()
        << ",\"observed_coordinate_counts_by_channel\":" << array_json(per_channel) << '}';
  }
  manifest << "]}"; write_text(directory / manifest_file, manifest.str());
}
void save_predictions(const fs::path &path, const FeatureSurface &surface,
                      const StressPredictions &value, const FrozenStressReadout &native) {
  torch::serialize::OutputArchive archive;
  archive.write("native_features_under_stress", surface.values.detach(), true);
  archive.write("feature_valid", surface.valid, true);
  archive.write("prediction_valid", value.valid, true);
  archive.write("ridge", value.ridge, true); archive.write("tiny", value.tiny, true);
  archive.write("ordinary_prediction_valid", native.native_valid, true);
  archive.write("ordinary_ridge", native.native_ridge, true); archive.write("ordinary_tiny", native.native_tiny, true);
  embedding::archive::save_archive(path.string(), archive);
}
std::string native_effect_json(const StressPredictions &stressed, const FrozenStressReadout &native,
    const ControlledProtocol &protocol, const std::string &stream) {
  const auto common = stressed.valid.logical_and(native.native_valid);
  const StressPredictions base{native.native_valid, native.native_ridge, native.native_tiny};
  std::ostringstream out; out << "{\"population_rule\":\"ordinary AND stressed prediction-valid rows\","
      "\"common_population\":" << population_json(common, protocol.testing)
      << ",\"ridge_stressed\":" << score_json(stressed.ridge, protocol.testing, common)
      << ",\"ridge_ordinary\":" << score_json(native.native_ridge, protocol.testing, common)
      << ",\"ridge_stressed_minus_ordinary_grouped_interval\":" << interval_json(grouped_accuracy_interval(
          stressed.ridge, protocol.testing.labels, common, protocol.testing.source_ids,
          ci_seed(protocol, stream + "/ridge-common"), 1000, native.native_ridge))
      << ",\"tiny_stressed\":" << score_json(stressed.tiny, protocol.testing, common)
      << ",\"tiny_ordinary\":" << score_json(native.native_tiny, protocol.testing, common)
      << ",\"tiny_stressed_minus_ordinary_grouped_interval\":" << interval_json(grouped_accuracy_interval(
          stressed.tiny, protocol.testing.labels, common, protocol.testing.source_ids,
          ci_seed(protocol, stream + "/tiny-common"), 1000, native.native_tiny))
      << ",\"full_population_rule\":\"all test rows;abstentions contribute zero correctness\","
         "\"ridge_full_population_stressed\":" << score_json(stressed.ridge, protocol.testing, stressed.valid)
      << ",\"ridge_full_population_ordinary\":" << score_json(native.native_ridge, protocol.testing, native.native_valid)
      << ",\"tiny_full_population_stressed\":" << score_json(stressed.tiny, protocol.testing, stressed.valid)
      << ",\"tiny_full_population_ordinary\":" << score_json(native.native_tiny, protocol.testing, native.native_valid)
      << ",\"ridge_full_population_stressed_minus_ordinary_grouped_interval\":"
      << interval_json(full_difference(stressed, base, protocol.testing, false,
          ci_seed(protocol, stream + "/ridge-full")))
      << ",\"tiny_full_population_stressed_minus_ordinary_grouped_interval\":"
      << interval_json(full_difference(stressed, base, protocol.testing, true,
          ci_seed(protocol, stream + "/tiny-full"))) << '}';
  return out.str();
}
} // namespace

std::string fixed_readout_stress_card_json(const EvaluationCard &card) {
  require(card.shape.channel_count > 0 && card.shape.history_length > 0 && card.shape.input_width > 0 &&
      static_cast<int64_t>(card.channel_ids.size()) == card.shape.channel_count &&
      std::set<int64_t>(card.channel_ids.begin(), card.channel_ids.end()).size() == card.channel_ids.size(),
      "stress card requires positive dimensions and unique semantic channel IDs");
  std::ostringstream out; out << "{\"format_version\":1,\"id\":\"fixed-readout-stress-v1\","
      "\"base_evaluation_card_id\":" << quote(card.id) << ",\"base_evaluation_card_version\":" << card.version
      << ",\"stage\":\"development\",\"acceptance\":\"unsupported\","
         "\"shape\":[" << card.shape.channel_count << ',' << card.shape.history_length << ',' << card.shape.input_width
      << "],\"channel_ids\":" << array_json(card.channel_ids)
      << ",\"fit_policy\":\"ordinary training normalizer/PCA/ridge/tiny reused unchanged;no stress refit\","
         "\"population\":\"testing observations only;paired variants retain their source group\","
         "\"mask_policy\":\"new support=base support AND NOT requested erasure;hidden storage zero\","
         "\"random_policy\":\"per-source/task/seed local mt19937_64;FNV-1a length-delimited source ID;"
         "one C*H*F stream;high53bits/2^53 threshold;rates nested;no Torch RNG or labels\","
         "\"contiguous_policy\":\"same per-source/task/seed uniform anchor in [0,H-1];"
         "length=ceil(rate*H);start=clamp(anchor-floor(length/2),0,H-length);all C/F;rates nested\","
         "\"uncertainty\":{\"method\":\"source-group percentile bootstrap\",\"replicates\":1000,"
         "\"confidence\":0.95,\"streams\":\"stable case/surface/tier/pair/method names\","
         "\"zero_or_one_group\":\"null interval\"},\"cases\":[";
  bool first = true;
  for (const auto &value : cases(card)) {
    if (!first) out << ',';
    first = false;
    out << case_json(value);
  }
  return out.str() + "]}";
}

std::string run_fixed_readout_stress(const std::string &directory, const EvaluationCard &card,
    const ControlledProtocol &protocol, const std::map<std::string, SurfaceDescription> &descriptions,
    const StressReadouts &readouts, const StressExtractor &extractor) {
  (void)fixed_readout_stress_card_json(card);
  validate_protocol(protocol);
  const auto &split = protocol.testing;
  const auto rows = split.labels.size(0);
  require(protocol.shape.channel_count == card.shape.channel_count &&
      protocol.shape.history_length == card.shape.history_length && protocol.shape.input_width == card.shape.input_width &&
      split.observed.data.size(1) == card.shape.channel_count &&
      split.observed.data.size(2) == card.shape.history_length &&
      split.observed.data.size(3) == card.shape.input_width, "protocol/card geometry mismatch");
  require(bool(extractor) && !readouts.empty(), "stress requires an extractor and declared frozen readouts");
  const auto output = fs::path(directory) / "stress";
  require(fs::is_directory(directory) && !fs::exists(output), "stress destination must be a new child of an existing run");
  require(fs::create_directory(output), "cannot create stress artifact directory");
  for (const auto &[key, readout] : readouts) {
    require(safe_name(key.first) && descriptions.count(key.first), "undeclared/unsafe readout surface");
    valid_mask(readout.native_valid, rows, "ordinary prediction validity");
    if (readout.status == "measured")
      require(bool(readout.predict), "measured readout lacks its frozen prediction callback");
  }
  std::ostringstream report; report << "{\"format_version\":1,\"protocol\":\"fixed-readout-stress-v1\","
      "\"seed\":" << quote(std::to_string(protocol.seed)) << ",\"task\":" << quote(task_name(protocol.task))
      << ",\"artifact_directory\":\"stress\",\"cases\":[";
  bool first_case = true;
  for (const auto &condition : cases(card)) {
    const auto corrupted = corrupt(condition, card, protocol);
    save_corrupted(output, condition, corrupted, card, protocol);
    auto surfaces = extractor(clone_batch(corrupted.batch));
    for (const auto &[name, description] : descriptions) {
      (void)description; require(surfaces.count(name), "extractor omitted declared surface: " + name);
      validate_features(surfaces.at(name));
      require(surfaces.at(name).values.size(0) == rows, "stressed surface row count mismatch");
    }
    const auto oracle = raw_oracle(protocol.task, corrupted.batch);
    if (!first_case) report << ',';
    first_case = false;
    report << "{\"case\":" << case_json(condition)
        << ",\"corrupted_batch\":" << quote(condition.id + "-corrupted.pt")
        << ",\"manifest\":" << quote(condition.id + "-manifest.json")
        << ",\"support\":" << support_json(corrupted, split.observed, card)
        << ",\"legal_raw_oracle_descriptive\":" << score_json(oracle.predictions, split, oracle.valid)
        << ",\"oracle_population\":" << population_json(oracle.valid, split) << ",\"features\":[";
    std::map<Key, StressPredictions> predictions;
    bool first_readout = true;
    for (const auto &[key, readout] : readouts) {
      const auto &surface = surfaces.at(key.first); const auto &description = descriptions.at(key.first);
      if (!first_readout) report << ',';
      first_readout = false;
      report << "{\"surface\":" << quote(key.first) << ",\"tier\":" << quote(tier_name(key.second))
          << ",\"kind\":" << quote(kind_name(description.kind))
          << ",\"probe_dimensions\":" << readout.probe_dimensions
          << ",\"channel_order\":" << array_json(description.channel_order)
          << ",\"feature_population\":" << population_json(surface.valid, split)
          << ",\"ordinary_fit_status\":" << quote(readout.status);
      if (readout.status != "measured") {
        report << ",\"status\":" << quote(readout.status) << ",\"reason\":" << quote(readout.reason) << '}';
        continue;
      }
      const auto value = readout.predict(surface);
      predictions_valid(value, rows, surface.valid);
      predictions_valid({readout.native_valid, readout.native_ridge, readout.native_tiny}, rows,
                         torch::ones({rows}, torch::kBool));
      predictions.emplace(key, value);
      const auto artifact = condition.id + "-" + key.first + "-" + tier_name(key.second) + "-predictions.pt";
      save_predictions(output / artifact, surface, value, readout);
      const auto stream = condition.id + "/surface/" + key.first + "/" + tier_name(key.second);
      report << ",\"status\":\"measured\",\"prediction_artifact\":" << quote(artifact)
          << ",\"prediction_population\":" << population_json(value.valid, split)
          << ",\"ridge\":" << score_json(value.ridge, split, value.valid)
          << ",\"ridge_grouped_interval\":" << interval_json(grouped_accuracy_interval(
              value.ridge, split.labels, value.valid, split.source_ids, ci_seed(protocol, stream + "/ridge")))
          << ",\"tiny_secondary\":" << score_json(value.tiny, split, value.valid)
          << ",\"tiny_secondary_grouped_interval\":" << interval_json(grouped_accuracy_interval(
              value.tiny, split.labels, value.valid, split.source_ids, ci_seed(protocol, stream + "/tiny")))
          << ",\"stress_vs_ordinary\":" << native_effect_json(value, readout, protocol, stream) << '}';
    }
    report << "],\"pair_comparisons\":[";
    bool first_pair = true;
    for (const auto &pair : card.comparisons) {
      if (!first_pair) report << ',';
      first_pair = false;
      const Key left{pair.left, pair.tier}, right{pair.right, pair.tier};
      report << "{\"comparison\":" << quote(pair.id) << ",\"candidate\":" << quote(pair.left)
          << ",\"comparator\":" << quote(pair.right) << ",\"tier\":" << quote(tier_name(pair.tier));
      if (!predictions.count(left) || !predictions.count(right)) {
        report << ",\"status\":\"unsupported_pair\",\"candidate_status\":"
            << quote(readouts.count(left) ? readouts.at(left).status : "unregistered")
            << ",\"comparator_status\":" << quote(readouts.count(right) ? readouts.at(right).status : "unregistered") << '}';
        continue;
      }
      const auto &a = predictions.at(left), &b = predictions.at(right);
      const auto common = a.valid.logical_and(b.valid);
      const auto stream = condition.id + "/pair/" + pair.id;
      report << ",\"status\":\"measured\",\"population_rule\":\"candidate AND comparator stressed validity only\","
          "\"common_population\":" << population_json(common, split)
          << ",\"ridge_candidate\":" << score_json(a.ridge, split, common)
          << ",\"ridge_comparator\":" << score_json(b.ridge, split, common)
          << ",\"ridge_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(
              a.ridge, split.labels, common, split.source_ids, ci_seed(protocol, stream + "/ridge"), 1000, b.ridge))
          << ",\"tiny_secondary_candidate\":" << score_json(a.tiny, split, common)
          << ",\"tiny_secondary_comparator\":" << score_json(b.tiny, split, common)
          << ",\"tiny_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(
              a.tiny, split.labels, common, split.source_ids, ci_seed(protocol, stream + "/tiny"), 1000, b.tiny))
          << ",\"full_population_rule\":\"all test rows;abstentions contribute zero correctness\","
             "\"ridge_full_population_candidate\":" << score_json(a.ridge, split, a.valid)
          << ",\"ridge_full_population_comparator\":" << score_json(b.ridge, split, b.valid)
          << ",\"tiny_full_population_candidate\":" << score_json(a.tiny, split, a.valid)
          << ",\"tiny_full_population_comparator\":" << score_json(b.tiny, split, b.valid)
          << ",\"ridge_full_population_candidate_minus_comparator_grouped_interval\":"
          << interval_json(full_difference(a, b, split, false, ci_seed(protocol, stream + "/ridge-full")))
          << ",\"tiny_full_population_candidate_minus_comparator_grouped_interval\":"
          << interval_json(full_difference(a, b, split, true, ci_seed(protocol, stream + "/tiny-full"))) << '}';
    }
    report << "]}";
  }
  report << "]}"; write_text(output / "report.json", report.str());
  return report.str();
}

} // namespace embedding::evaluation
