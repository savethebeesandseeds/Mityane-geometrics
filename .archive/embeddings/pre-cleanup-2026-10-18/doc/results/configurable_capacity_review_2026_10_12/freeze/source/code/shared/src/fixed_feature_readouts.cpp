// SPDX-License-Identifier: MIT
#include "embedding/shared/fixed_feature_readouts.h"
#include <ATen/Context.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
namespace fs = std::filesystem;
constexpr std::array<uint64_t, 3> repetitions{2701, 2802, 2903};
constexpr int64_t bootstrap_replicates = 1000;
void require(bool value, const std::string &why) {
  if (!value) throw std::runtime_error("[fixed feature readouts] " + why);
}
bool safe(const std::string &name) {
  return !name.empty() && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
std::string strings(const std::vector<std::string> &values) {
  std::ostringstream out; out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  return out.str() + ']';
}
uint64_t named(const std::string &value) {
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char c : value) { hash ^= c; hash *= 1099511628211ULL; }
  return hash;
}
void write(const fs::path &path, const std::string &text) {
  require(!fs::exists(path), "output exists: " + path.string());
  std::ofstream out(path); require(bool(out), "cannot create output"); out << text << '\n'; out.close();
  require(bool(out), "cannot save output");
}
void save(const fs::path &path, torch::serialize::OutputArchive &out) {
  require(!fs::exists(path), "archive exists: " + path.string()); archive::save_archive(path.string(), out);
}
struct Isolation {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  Isolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
    at::set_num_threads(1);
  }
  ~Isolation() noexcept {
    try { for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]); at::set_num_threads(threads); }
    catch (...) { std::terminate(); }
  }
};
void labels(const torch::Tensor &value, int64_t rows) {
  require(value.defined() && value.device().is_cpu() && value.scalar_type() == torch::kInt64 &&
      value.dim() == 1 && value.size(0) == rows && rows > 0 && value.ge(0).logical_and(value.le(1)).all().item<bool>(),
      "binary CPU label row schema");
}
std::set<std::string> source_set(const std::vector<std::string> &ids, int64_t rows) {
  require(ids.size() == size_t(rows), "source row association");
  std::set<std::string> result;
  for (const auto &id : ids) { require(!id.empty(), "empty source ID"); result.insert(id); }
  return result;
}
FeatureSurface cloned(const FeatureSurface &input) {
  validate_features(input); require(!input.provenance.empty(), "feature provenance required");
  return {torch::where(input.valid.unsqueeze(1), input.values.detach(), torch::zeros_like(input.values)).clone(),
          input.valid.detach().clone(), input.provenance};
}
std::string population(const torch::Tensor &valid, const torch::Tensor &truth, const std::vector<std::string> &ids) {
  std::set<std::string> all, retained;
  int64_t classes[2]{0, 0};
  for (int64_t i = 0; i < truth.size(0); ++i) {
    all.insert(ids.at(i)); if (valid[i].item<bool>()) { retained.insert(ids.at(i)); ++classes[truth[i].item<int64_t>()]; }
  }
  std::ostringstream out; out << std::setprecision(17) << "{\"total_rows\":" << truth.size(0)
      << ",\"valid_rows\":" << classes[0]+classes[1] << ",\"class_valid_rows\":[" << classes[0] << ',' << classes[1]
      << "],\"total_source_groups\":" << all.size() << ",\"valid_source_groups\":" << retained.size()
      << ",\"coverage\":" << double(classes[0]+classes[1])/truth.size(0) << '}'; return out.str();
}
std::string score_json(const Score &value) {
  std::ostringstream out; out << std::setprecision(17) << "{\"total\":" << value.total << ",\"valid\":" << value.valid
      << ",\"correct\":" << value.correct << ",\"coverage\":" << value.coverage
      << ",\"full_population_correctness\":" << double(value.correct)/value.total << ",\"accuracy\":";
  if (value.supported) out << value.accuracy; else out << "null";
  return out.str() + '}';
}
std::string interval_json(const GroupedInterval &value) {
  std::ostringstream out; out << std::setprecision(17) << "{\"method\":\"source-group percentile bootstrap; within-master conditional on fitted readout\","
      << "\"replicates\":1000,\"confidence\":0.95,\"source_groups\":" << value.source_groups << ",\"estimate\":";
  if (value.source_groups) out << value.estimate; else out << "null";
  out << ",\"lower\":"; if (value.supported) out << value.lower; else out << "null";
  out << ",\"upper\":"; if (value.supported) out << value.upper; else out << "null";
  return out.str() + '}';
}
void save_surface(const fs::path &path, const FeatureSurface &surface,
                  const torch::Tensor &truth, const std::vector<std::string> &ids) {
  torch::serialize::OutputArchive out; out.write("features", surface.values, true); out.write("valid", surface.valid, true);
  out.write("labels_scoring_only", truth, true); out.write("source_ids_json", archive::text_tensor(strings(ids)), true);
  out.write("provenance", archive::text_tensor(surface.provenance), true); save(path, out);
}
struct Prediction {
  torch::Tensor ridge, tiny, valid, probe_input;
  // Saved arithmetic witnesses preserve tie decisions without requiring a
  // separately ordered CPU dot product to make an identical class decision.
  torch::Tensor ridge_logits, tiny_hidden_preactivation, tiny_logits;
};
void save_prediction(const fs::path &path, const Prediction &p, const torch::Tensor &truth,
                     const std::vector<std::string> &ids) {
  torch::serialize::OutputArchive out; out.write("ridge", p.ridge, true); out.write("tiny_secondary", p.tiny, true);
  out.write("valid", p.valid, true); out.write("probe_input_features", p.probe_input, true);
  out.write("ridge_logits", p.ridge_logits, true);
  out.write("tiny_hidden_preactivation", p.tiny_hidden_preactivation, true);
  out.write("tiny_logits", p.tiny_logits, true);
  out.write("labels_scoring_only", truth, true); out.write("source_ids_json", archive::text_tensor(strings(ids)), true); save(path, out);
}
struct Fit {
  std::optional<FeatureNormalizer> outer;
  RidgeProbe ridge;
  TinyProbe tiny;
  uint64_t seed;
  Fit(std::optional<FeatureNormalizer> normalization, const FeatureSurface &prepared,
      const torch::Tensor &truth, uint64_t actual_seed)
      : outer(std::move(normalization)), ridge(prepared, truth, 1), tiny(prepared, truth, actual_seed, 100, 16, .01), seed(actual_seed) {}
  Prediction predict(const FeatureSurface &surface) const {
    torch::NoGradGuard no_grad;
    const auto input = outer ? outer->transform(surface) : FeatureSurface{surface.values.to(torch::kFloat64), surface.valid.clone(), surface.provenance};
    const auto ridge_logits = ridge.normalizer.transform(input).values.matmul(ridge.weights) + ridge.intercept;
    const auto hidden = tiny.normalizer.transform(input).values.matmul(tiny.w1) + tiny.b1;
    const auto tiny_logits = torch::tanh(hidden).matmul(tiny.w2) + tiny.b2;
    require(torch::isfinite(ridge_logits).all().item<bool>() && torch::isfinite(hidden).all().item<bool>() &&
        torch::isfinite(tiny_logits).all().item<bool>(), "nonfinite prediction arithmetic");
    return {ridge_logits.argmax(1), tiny_logits.argmax(1), input.valid.clone(), input.values.detach().clone(),
        ridge_logits.detach().clone(), hidden.detach().clone(), tiny_logits.detach().clone()};
  }
};
std::vector<torch::Tensor> state(const Fit &fit) {
  std::vector<torch::Tensor> values{fit.ridge.normalizer.mean, fit.ridge.normalizer.scale, fit.ridge.weights, fit.ridge.intercept,
      fit.tiny.normalizer.mean, fit.tiny.normalizer.scale, fit.tiny.w1, fit.tiny.b1, fit.tiny.w2, fit.tiny.b2};
  if (fit.outer) { values.push_back(fit.outer->mean); values.push_back(fit.outer->scale); }
  for (auto &value : values) value = value.detach().clone();
  return values;
}
void same_state(const Fit &fit, const std::vector<torch::Tensor> &before) {
  const auto after = state(fit); require(after.size() == before.size(), "frozen fit schema changed");
  for (size_t i = 0; i < before.size(); ++i) require(torch::equal(after[i], before[i]), "TRAIN fitted tensor changed while scoring");
}
void save_fit(const fs::path &path, const Fit &fit, int64_t width, const std::vector<std::string> &ids) {
  torch::serialize::OutputArchive out;
  out.write("feature_mean", fit.outer ? fit.outer->mean : torch::zeros({width}, torch::kFloat64), true);
  out.write("feature_scale", fit.outer ? fit.outer->scale : torch::ones({width}, torch::kFloat64), true);
  out.write("outer_normalizer_applied", torch::tensor(bool(fit.outer), torch::kBool), true);
  out.write("outer_fitted_rows", torch::tensor(fit.outer ? fit.outer->fitted_rows : int64_t(0)), true);
  out.write("fitted_rows", torch::tensor(fit.ridge.normalizer.fitted_rows), true);
  out.write("ridge_mean", fit.ridge.normalizer.mean, true); out.write("ridge_scale", fit.ridge.normalizer.scale, true);
  out.write("ridge_weights", fit.ridge.weights, true); out.write("ridge_intercept", fit.ridge.intercept, true);
  out.write("tiny_mean", fit.tiny.normalizer.mean, true); out.write("tiny_scale", fit.tiny.normalizer.scale, true);
  out.write("tiny_w1", fit.tiny.w1, true); out.write("tiny_b1", fit.tiny.b1, true); out.write("tiny_w2", fit.tiny.w2, true); out.write("tiny_b2", fit.tiny.b2, true);
  out.write("actual_probe_seed_decimal", archive::text_tensor(std::to_string(fit.seed)), true);
  out.write("training_source_ids_json", archive::text_tensor(strings(ids)), true);
  out.write("ridge_penalty", torch::tensor(1.0, torch::kFloat64), true); out.write("tiny_hidden", torch::tensor(int64_t(16)), true);
  out.write("tiny_steps", torch::tensor(int64_t(100)), true); out.write("tiny_learning_rate", torch::tensor(.01, torch::kFloat64), true);
  save(path, out);
}
std::string prediction_json(const Prediction &p, const torch::Tensor &truth, const std::vector<std::string> &ids, uint64_t seed, bool intervals) {
  std::ostringstream out; out << "{\"population\":" << population(p.valid, truth, ids)
      << ",\"ridge\":" << score_json(score(p.ridge, truth, p.valid)) << ",\"tiny_secondary\":" << score_json(score(p.tiny, truth, p.valid));
  if (intervals) out << ",\"bootstrap_seed_decimal\":" << quote(std::to_string(seed))
      << ",\"ridge_grouped_interval\":" << interval_json(grouped_accuracy_interval(p.ridge, truth, p.valid, ids, seed, bootstrap_replicates))
      << ",\"tiny_grouped_interval\":" << interval_json(grouped_accuracy_interval(p.tiny, truth, p.valid, ids, seed, bootstrap_replicates));
  return out.str() + '}';
}
struct Result { bool measured{false}; std::array<Prediction, 3> predictions; };
} // namespace

std::string run_fixed_feature_readouts(const FixedFeatureReadoutRun &run) {
  require(!run.methods.empty(), "no named feature methods");
  require(run.training_labels.defined() && run.validation_labels.defined(), "label tensors are undefined");
  labels(run.training_labels, run.training_labels.size(0)); labels(run.validation_labels, run.validation_labels.size(0));
  const auto training_ids = source_set(run.training_source_ids, run.training_labels.size(0));
  const auto validation_ids = source_set(run.validation_source_ids, run.validation_labels.size(0));
  for (const auto &id : training_ids) require(!validation_ids.count(id), "TRAIN/VALIDATION source leakage");
  std::set<std::string> names;
  std::vector<FixedFeatureMethod> methods;
  for (const auto &method : run.methods) {
    require(safe(method.name) && names.insert(method.name).second, "unsafe/duplicate method name");
    require(!method.inputs_train_prepared || method.name == "pca_only" || method.name == "raw",
        "only caller-normalized raw values or raw PCA components may bypass the outer normalizer");
    auto training = cloned(method.training), intact = cloned(method.validation_intact), deleted = cloned(method.validation_deleted);
    require(training.values.size(0) == run.training_labels.size(0) && intact.values.size(0) == run.validation_labels.size(0) &&
        deleted.values.size(0) == intact.values.size(0) && training.values.size(1) == intact.values.size(1) &&
        training.values.size(1) == deleted.values.size(1), "method split shape mismatch");
    require(training.values.size(1) <= (std::numeric_limits<int64_t>::max()-50)/16,
        "fixed head parameter count overflow");
    methods.push_back({method.name, std::move(training), std::move(intact), std::move(deleted),
        method.inputs_train_prepared, method.preparation_unsupported_reason});
  }
  const bool explicit_pair = !run.comparison_reference.empty() || !run.comparison_candidate.empty();
  require(run.comparisons.empty() || !explicit_pair, "multiple comparisons cannot also declare a single pair");
  auto comparisons = run.comparisons;
  if (comparisons.empty()) comparisons.push_back({explicit_pair ? run.comparison_reference : "native_v4",
      explicit_pair ? run.comparison_candidate : "native_v7"});
  std::set<std::string> comparison_ids;
  if (explicit_pair || !run.comparisons.empty()) for (const auto &pair : comparisons) {
    require(safe(pair.reference) && safe(pair.candidate) && pair.reference != pair.candidate &&
        names.count(pair.reference) && names.count(pair.candidate), "explicit comparison requires two distinct declared methods");
    require(comparison_ids.insert(pair.candidate + "_minus_" + pair.reference).second, "duplicate explicit comparison ID");
  }
  const fs::path output(run.output_directory);
  require(!output.empty() && !fs::exists(output) && fs::create_directories(output), "new exclusive readout directory required");
  const Isolation isolation;
  const auto train_truth = run.training_labels.detach().clone(), val_truth = run.validation_labels.detach().clone();
  std::map<std::string, std::array<Result, 3>> results;
  std::ostringstream json; json << "{\"protocol\":\"fixed-feature-readouts-v1\",\"master_seed\":" << quote(std::to_string(run.master_seed))
      << ",\"recipe\":{\"ridge_penalty\":1,\"tiny_hidden\":16,\"tiny_steps\":100,\"tiny_learning_rate\":0.01,"
      << "\"probe_seed_policy\":\"stream_seed(repetition,width);same_equal_width_methods\",\"validation_fits\":0,\"encoder_calls\":0,\"pca_fits\":0},\"methods\":[";
  int64_t head_fits = 0, outer_fits = 0; bool first_method = true;
  for (const auto &method : methods) {
    const auto directory = output/method.name; require(fs::create_directory(directory), "method directory exists");
    save_surface(directory/"training-features.pt", method.training, train_truth, run.training_source_ids);
    save_surface(directory/"validation-intact-features.pt", method.validation_intact, val_truth, run.validation_source_ids);
    save_surface(directory/"validation-deleted-features.pt", method.validation_deleted, val_truth, run.validation_source_ids);
    const auto selected = train_truth.masked_select(method.training.valid);
    auto reason = method.preparation_unsupported_reason;
    if (reason.empty() && !(selected.numel() >= 2 && selected.eq(0).any().item<bool>() && selected.eq(1).any().item<bool>()))
      reason = "valid TRAIN fitting rows require at least 2 rows and both classes";
    std::optional<FeatureNormalizer> outer;
    if (reason.empty() && !method.inputs_train_prepared) { outer.emplace(method.training); ++outer_fits; }
    const auto fitted = outer ? outer->transform(method.training) : method.training;
    if (!first_method) json << ',';
    first_method = false;
    json << "{\"method\":" << quote(method.name) << ",\"size\":" << method.training.values.size(1)
        << ",\"inputs_train_prepared\":" << (method.inputs_train_prepared ? "true" : "false")
        << ",\"outer_train_normalizer_fits\":" << (outer ? 1 : 0)
        << ",\"status\":" << quote(reason.empty() ? "measured" : "unsupported_fit") << ",\"reason\":" << quote(reason)
        << ",\"training_population\":" << population(method.training.valid, train_truth, run.training_source_ids)
        << ",\"validation_intact_population\":" << population(method.validation_intact.valid, val_truth, run.validation_source_ids)
        << ",\"validation_deleted_population\":" << population(method.validation_deleted.valid, val_truth, run.validation_source_ids) << ",\"repetitions\":[";
    for (size_t r = 0; r < repetitions.size(); ++r) {
      if (r) json << ',';
      const auto id = "rep-"+std::to_string(repetitions[r]);
      const auto width = method.training.values.size(1); const auto seed = stream_seed(repetitions[r], uint64_t(width));
      json << "{\"id\":" << quote(id) << ",\"actual_probe_seed_decimal\":" << quote(std::to_string(seed))
          << ",\"status\":" << quote(reason.empty() ? "measured" : "unsupported_fit")
          << ",\"ridge_parameters\":" << 2*width+2 << ",\"neural_parameters\":" << 16*(width+3)+2;
      if (reason.empty()) {
        Fit fit(outer, fitted, train_truth, seed); ++head_fits;
        const auto witness = state(fit); const auto rep_directory = directory/id; require(fs::create_directory(rep_directory), "rep directory exists");
        save_fit(rep_directory/"fit.pt", fit, width, run.training_source_ids);
        auto &result = results[method.name][r]; result.measured = true;
        result.predictions = {fit.predict(method.training), fit.predict(method.validation_intact), fit.predict(method.validation_deleted)};
        same_state(fit, witness);
        const std::array<std::string, 3> views{"training", "validation-intact", "validation-deleted"};
        for (size_t view = 0; view < views.size(); ++view) {
          const auto &truth = view ? val_truth : train_truth; const auto &ids = view ? run.validation_source_ids : run.training_source_ids;
          save_prediction(rep_directory/(views[view]+"-predictions.pt"), result.predictions[view], truth, ids);
          json << ',' << quote(view == 0 ? "training" : view == 1 ? "validation_intact" : "validation_deleted") << ':'
              << prediction_json(result.predictions[view], truth, ids, stream_seed(run.master_seed, named(method.name+"/"+id+"/"+views[view])), view != 0);
        }
        const auto repeat = fit.predict(method.validation_intact); same_state(fit, witness);
        require(torch::equal(repeat.ridge, result.predictions[1].ridge) && torch::equal(repeat.tiny, result.predictions[1].tiny) &&
            torch::equal(repeat.valid, result.predictions[1].valid), "fixed readout changed during deleted view scoring");
        json << ",\"fit_artifact\":" << quote(method.name+"/"+id+"/fit.pt");
      }
      json << '}';
    }
    json << "]}";
    write(directory/"fit-counts.json", "{\"outer_train_normalizer_fits\":"+std::to_string(outer ? 1 : 0)+
        ",\"ridge_fits\":"+std::to_string(reason.empty() ? 3 : 0)+",\"tiny_fits\":"+std::to_string(reason.empty() ? 3 : 0)+",\"validation_fits\":0}");
  }
  json << "],\"pairs\":["; bool first_pair = true;
  for (const auto &pair : comparisons) {
    const auto &reference = pair.reference, &candidate = pair.candidate;
    const std::string pair_id = candidate + "_minus_" + reference;
    if (names.count(reference) && names.count(candidate)) for (size_t r = 0; r < repetitions.size(); ++r) for (size_t view : {size_t(1), size_t(2)}) {
      if (!first_pair) json << ',';
      first_pair = false;
      const auto id = "rep-"+std::to_string(repetitions[r]);
      const std::string view_name = view == 1 ? "validation_intact" : "validation_deleted";
      const auto &left = results[candidate][r], &right = results[reference][r];
      json << "{\"id\":" << quote(pair_id) << ",\"repetition\":" << quote(id) << ",\"view\":" << quote(view_name);
      if (!left.measured || !right.measured) json << ",\"status\":\"unsupported_fit\",\"reason\":\"one declared method has no TRAIN-fitted readout\"}";
      else {
        const auto &a = left.predictions[view], &b = right.predictions[view]; const auto common = a.valid.logical_and(b.valid);
        const auto seed = stream_seed(run.master_seed, named(pair_id+"/"+id+"/"+view_name));
        json << ",\"status\":" << quote(common.any().item<bool>() ? "measured" : "unsupported_zero_common")
            << ",\"common_population\":" << population(common, val_truth, run.validation_source_ids)
            << ",\"bootstrap_seed_decimal\":" << quote(std::to_string(seed))
            << ",\"ridge\":" << interval_json(grouped_accuracy_interval(a.ridge, val_truth, common, run.validation_source_ids, seed, bootstrap_replicates, b.ridge))
            << ",\"tiny_secondary\":" << interval_json(grouped_accuracy_interval(a.tiny, val_truth, common, run.validation_source_ids, seed, bootstrap_replicates, b.tiny)) << '}';
      }
    }
  }
  json << "],\"fit_counts\":{\"outer_train_normalizer_fits\":" << outer_fits << ",\"ridge_fits\":" << head_fits
      << ",\"tiny_fits\":" << head_fits << ",\"validation_fits\":0,\"encoder_calls\":0,\"pca_fits\":0}}";
  write(output/"report.json", json.str()); return json.str();
}
} // namespace embedding::evaluation
