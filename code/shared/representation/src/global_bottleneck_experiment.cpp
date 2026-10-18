// SPDX-License-Identifier: MIT
#include "embedding/shared/global_bottleneck_experiment.h"
#include "embedding/shared/feature_stress.h"
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
  if (!ok) throw std::runtime_error("[global bottleneck] " + message);
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
  throw std::runtime_error("[global bottleneck] unknown tier");
}
std::string score_json(const Score &score) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"total\":" << score.total << ",\"valid\":" << score.valid
      << ",\"correct\":" << score.correct << ",\"abstained\":" << score.total-score.valid
      << ",\"full_population_correctness\":" << (score.total?double(score.correct)/score.total:0)
      << ",\"coverage\":" << score.coverage << ",\"accuracy\":";
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
void validate_split(const ControlledDataset &split, const GlobalBottleneckRun &run, int64_t pairs,
                    std::set<std::string> &all_sources) {
  require(split.observed.data.defined() && split.observed.data.device().is_cpu() &&
          split.observed.data.scalar_type() == torch::kFloat64 &&
          split.observed.data.sizes() == torch::IntArrayRef({2 * pairs, run.card.shape.channel_count, run.card.shape.history_length, run.card.shape.input_width}) &&
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
void validate_run(const GlobalBottleneckRun &run, const std::vector<NamedCurveFactory> &factories) {
  validate_evaluation_card(run.card);
  require(std::find(run.card.tasks.begin(),run.card.tasks.end(),run.selection_task)!=run.card.tasks.end(),
          "selection task must belong to the frozen task list");
  require(run.patch_length > 0 && run.card.shape.history_length % run.patch_length == 0 &&
          run.card.shape.history_length / run.patch_length >= 3 && std::isfinite(run.huber_delta) && run.huber_delta > 0,
          "curve reconstruction requires at least3whole patches and finite positive Huber delta");
  require(run.milestones.size() >= 2 && run.milestones.front() == 0, "curve requires initialization plus nonzero milestones");
  for (size_t i = 1; i < run.milestones.size(); ++i)
    require(run.milestones[i] > run.milestones[i - 1], "milestones must be strictly increasing absolute budgets");
  require(run.fixed_test_budget >= 0 && (run.fixed_test_budget == 0 ||
          std::find(run.milestones.begin(),run.milestones.end(),run.fixed_test_budget)!=run.milestones.end()),
          "fixed testing budget must be zero or a declared positive milestone");
  require(run.training_reservoir_pairs >= 0 && run.training_reservoir_pairs <= std::numeric_limits<int64_t>::max()/6 &&
          (run.training_reservoir_pairs == 0 ||
          (run.training_reservoir_pairs >= run.card.train_pairs && run.validation_seed_stream > 0)),
          "training reservoir must cover the training pairs and requires an independent validation stream");
  const std::vector<std::string> names{"current_mixer","mean_global","learned_global"};
  require(factories.size()==names.size(),"global comparison requires exactly three declared variants");
  for (size_t i=0;i<names.size();++i)
    require(factories[i].name==names[i] && !factories[i].recipe.empty() && bool(factories[i].factory),
            "variants must be current_mixer, mean_global, learned_global in the declared order");
}
std::string card_json(const GlobalBottleneckRun &run, const std::vector<NamedCurveFactory> &factories) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"version\":1,\"protocol\":\"global-bottleneck-comparison-v1\","
      << "\"stage\":\"development\",\"policy_version\":\"1.1\",\"selection_task\":" << quote(task_name(run.selection_task)) << ','
      << "\"evaluation_card\":" << evaluation_card_json(run.card) << ','
      << "\"shape\":[" << run.card.shape.channel_count << ',' << run.card.shape.history_length << ',' << run.card.shape.input_width
      << "],\"dtype\":\"float64\",\"channel_ids\":" << numbers(run.card.channel_ids) << ",\"feature_units\":" << quote(run.card.feature_units)
      << ",\"sampling_interval\":" << run.card.sampling_interval << ",\"patch_length\":" << run.patch_length
      << ",\"huber_delta\":" << run.huber_delta << ",\"train_pairs\":" << run.card.train_pairs
      << ",\"validation_pairs\":" << run.card.validation_pairs << ",\"test_pairs\":" << run.card.test_pairs
      << ",\"matched_global_width\":" << run.card.matched_global_width << ",\"matched_channel_width\":" << run.card.matched_channel_width
      << ",\"milestones\":" << numbers(run.milestones)
      << ",\"fixed_test_budget\":" << run.fixed_test_budget
      << ",\"validation_seed_stream\":" << quote(std::to_string(run.validation_seed_stream))
      << ",\"training_reservoir_pairs\":" << run.training_reservoir_pairs << ",\"seeds\":[";
  for (size_t i = 0; i < run.card.seeds.size(); ++i) { if (i) out << ','; out << quote(std::to_string(run.card.seeds[i])); }
  out << "],\"threads\":" << run.card.threads << ",\"missing_rate\":0.1,\"stress_sweep\":" << (run.stress_sweep?"true":"false") << ",\"providers\":[";
  for (size_t i = 0; i < factories.size(); ++i) {
    if (i) out << ',';
    out << "{\"name\":" << quote(factories[i].name) << ",\"recipe\":" << quote(factories[i].recipe) << '}';
  }
  out << "],\"selection\":{\"metric\":\"validation matched-global ridge accuracy on selection task\","
      << "\"selection_policy\":" << quote(run.fixed_test_budget ? "declared fixed supported nonzero budget; no validation budget search" :
          "maximum mean validation primary; exact ties select lower nonzero budget") << ','
      << "\"weighting\":\"equal variant and master weights; within-master common validity across all three variants\","
      << "\"candidates\":\"all nonzero milestones; any missing/unsupported fit or validation class excludes entire budget\","
      << "\"tie_rule\":\"smallest exact-tied budget\",\"initialization_is_diagnostic_only\":true},"
      << "\"readout_recipe\":{\"ridge_penalty\":1,\"tiny_secondary\":{\"activation\":\"tanh\",\"hidden\":16,\"updates\":100,\"optimizer\":\"Adam\",\"learning_rate\":0.01},"
      << "\"seed\":\"stream_seed(master,FNV1a64(kind/tier));same named initialization across variants and milestones\","
      << "\"pipeline\":\"valid-training outer normalizer;centered train-fit PCA for matched tiers;each probe own valid-training normalizer\"},"
      << "\"fresh_test_stream\":" << quote(std::to_string(run.fresh_test_stream)) << ','
      << "\"validation_policy\":" << quote(run.validation_seed_stream ?
          "independent named stream_seed(master,validation_seed_stream); validation from development(1,validation_pairs); sentinel training discarded; cohort independent of training-pair count" :
          "ordinary jointly partitioned development cohort; validation partition depends on training-pair count") << ','
      << "\"training_policy\":" << quote(run.training_reservoir_pairs ?
          "development(training_reservoir_pairs,1,master) training split; first 2*train_pairs rows in ascending assembled source order; complete pairs; fixed reservoir gives nested training cohorts; sentinel validation discarded" :
          "ordinary development(train_pairs,validation_pairs,master) training split") << ','
      << "\"test_policy\":\"generate only after persisted selection; stream_seed(master,fresh_test_stream); task namespaces remain disjoint; chosen exact checkpoint/readouts; no refit\","
      << "\"reconstruction_policy\":\"enumerate every original patch, same hidden patch across channels; observed target cells only; >=2 visible patch groups/channel; cell then channel then example equal means\","
      << "\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"limits\":[\"synthetic development only\",\"common updates do not match parameters or cost\","
      << "\"source-group intervals are within seed conditional on fitted checkpoint/readouts\",\"no across-seed uncertainty or consumer acceptance\"]}";
  return out.str();
}
ControlledProtocol development_cohort(Task task,const GlobalBottleneckRun &run,uint64_t seed) {
  auto cohort=make_controlled_development_protocol(task,run.card.shape,
      run.training_reservoir_pairs?run.training_reservoir_pairs:run.card.train_pairs,
      run.training_reservoir_pairs?1:run.card.validation_pairs,seed);
  if(run.training_reservoir_pairs) {
    const auto rows=2*run.card.train_pairs;
    auto &training=cohort.training;
    training.clean.data=training.clean.data.slice(0,0,rows).clone();
    training.clean.feature_mask=training.clean.feature_mask.slice(0,0,rows).clone();
    training.observed.data=training.observed.data.slice(0,0,rows).clone();
    training.observed.feature_mask=training.observed.feature_mask.slice(0,0,rows).clone();
    training.labels=training.labels.slice(0,0,rows).clone();
    training.source_ids.resize(rows);
  }
  if(run.validation_seed_stream) {
    auto named_validation=make_controlled_development_protocol(task,run.card.shape,1,
        run.card.validation_pairs,stream_seed(seed,run.validation_seed_stream));
    require(!named_validation.testing.observed.data.defined() && named_validation.testing.source_ids.empty(),
            "named validation generation created testing observations");
    cohort.validation=std::move(named_validation.validation);
  }
  return cohort;
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
  Task task;
  ControlledProtocol development;
  std::vector<Architecture> architectures;
  std::unique_ptr<RidgeProbe> mask_probe;
  std::unique_ptr<TinyProbe> mask_tiny;
};

CurveSnapshot named_snapshot(CurveSnapshot snapshot,const std::string &variant) {
  const auto original=snapshot.features;
  require(original.surfaces.size()==2,"snapshot requires exactly two typed export surfaces");
  std::map<std::string,std::string> renamed;
  snapshot.features.surfaces.clear();
  for(const auto &[name,description]:original.surfaces) {
    require(description.kind==SurfaceKind::global || description.kind==SurfaceKind::channel_concatenation,
            "snapshot exports must be global and channel concatenation");
    const auto next=variant+(description.kind==SurfaceKind::global?"_global":"_channel_concatenation");
    require(snapshot.features.surfaces.emplace(next,description).second,"duplicate typed export kind");
    renamed.emplace(name,next);
  }
  snapshot.features.extract=[original,renamed](const Batch &batch) {
    const auto extracted=original.extract(legal_clone(batch));
    require(extracted.size()==renamed.size(),"snapshot extraction changed declared exports");
    FeatureMap out;
    for(const auto &[name,next]:renamed) out.emplace(next,extracted.at(name));
    return out;
  };
  return snapshot;
}
FeatureMap extract(const FeatureProvider &provider, const Batch &batch, const GlobalBottleneckRun &run,
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
      require(description.channel_order.size() == run.card.channel_ids.size() &&
              std::set<int64_t>(description.channel_order.begin(), description.channel_order.end()) ==
              std::set<int64_t>(run.card.channel_ids.begin(), run.card.channel_ids.end()) &&
              surface.values.size(1) % run.card.shape.channel_count == 0, "concatenation channel order/width mismatch");
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
std::string fit_readouts(Point &point, const ControlledProtocol &protocol, const GlobalBottleneckRun &run, uint64_t seed) {
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
          try { pca.emplace(fitted, global ? run.card.matched_global_width : run.card.matched_channel_width); }
          catch (const std::exception &error) { result.status = "unsupported_compression"; result.reason = error.what(); }
        }
        if (result.status.empty()) {
          if (pca) fitted = pca->transform(fitted);
          const auto probe_seed = stream_seed(seed, named_stream(std::string(global?"global":"channel_concatenation") + "/" + tier_name(tier)));
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
                           const CurveSnapshot &snapshot, const GlobalBottleneckRun &run) {
  const auto B = split.observed.data.size(0), C = run.card.shape.channel_count;
  const auto H = run.card.shape.history_length, F = run.card.shape.input_width, K = H / run.patch_length;
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
Candidate utility(int64_t budget, const std::vector<SeedState> &states,const GlobalBottleneckRun &run) {
  Candidate candidate{budget,true,0,""};
  std::ostringstream out;
  out << std::setprecision(17) << '[';
  bool first = true;
  int64_t count = 0;
  for (const auto &state : states) {
    if(state.task!=run.selection_task) continue;
    auto common=torch::ones_like(state.development.validation.labels,torch::kBool);
    bool supported=true;
    for(const auto &architecture:state.architectures) {
      const auto &point=architecture.points.at(budget);
      const auto &result=point.results.at({point.global,DimensionTier::matched_global});
      supported=supported && result.status=="measured";
      common=common.logical_and(result.validation_valid);
    }
    const auto labels = state.development.validation.labels.masked_select(common);
    supported=supported && labels.numel()>=2 && labels.eq(0).any().item<bool>() && labels.eq(1).any().item<bool>();
    candidate.supported = candidate.supported && supported;
    for (size_t architecture = 0; architecture < state.architectures.size(); ++architecture) {
      if (!first) out << ',';
      first = false;
      const auto &model = state.architectures[architecture];
      const auto &point = model.points.at(budget);
      const auto &result = point.results.at({point.global,DimensionTier::matched_global});
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
    require(count == int64_t(3*run.card.seeds.size()), "common budget silently dropped a variant/master");
    candidate.utility /= count;
    require(std::isfinite(candidate.utility), "selection utility nonfinite");
  }
  candidate.details = out.str() + ']';
  return candidate;
}
std::vector<torch::Tensor> fitted_state(const Readout &readout) {
  std::vector<torch::Tensor> out{readout.normalizer.mean,readout.normalizer.scale,
      readout.ridge.normalizer.mean,readout.ridge.normalizer.scale,readout.ridge.weights,readout.ridge.intercept,
      readout.tiny.normalizer.mean,readout.tiny.normalizer.scale,readout.tiny.w1,readout.tiny.b1,readout.tiny.w2,readout.tiny.b2};
  if(readout.pca) {
    out.push_back(readout.pca->mean); out.push_back(readout.pca->components); out.push_back(readout.pca->singular_values);
  }
  for(auto &value:out) value=value.detach().clone();
  return out;
}
std::string evaluate_test(Point &point, const ControlledDataset &testing, const GlobalBottleneckRun &run, uint64_t seed,
                         std::map<Key,StressPredictions> &predictions,StressReadouts &stress_readouts) {
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
      predictions.emplace(key,StressPredictions{transformed.valid.clone(),ridge.clone(),tiny.clone()});
      FrozenStressReadout frozen;
      frozen.status=result.status; frozen.probe_dimensions=transformed.values.size(1);
      frozen.native_valid=transformed.valid.clone(); frozen.native_ridge=ridge.clone(); frozen.native_tiny=tiny.clone();
      const auto *fixed=result.readout.get();
      const auto assets=fitted_state(*fixed);
      const StressPredictions ordinary{transformed.valid.clone(),ridge.clone(),tiny.clone()};
      frozen.predict=[fixed,assets,ordinary,first=true](const FeatureSurface &surface) mutable {
        const auto values=fixed->transform(surface);
        StressPredictions out{values.valid.clone(),fixed->ridge.predict(values),fixed->tiny.predict(values)};
        const auto after=fitted_state(*fixed);
        require(after.size()==assets.size(),"stress changed fitted readout structure");
        for(size_t i=0;i<assets.size();++i) require(torch::equal(after[i],assets[i]),"stress mutated frozen normalization/PCA/probe assets");
        if(first) {
          require(torch::equal(out.valid,ordinary.valid) && torch::equal(out.ridge,ordinary.ridge) && torch::equal(out.tiny,ordinary.tiny),
                  "intact stress predictions differ from ordinary selected-test predictions");
          first=false;
        }
        return out;
      };
      stress_readouts.emplace(key,std::move(frozen));
      save_predictions(point.directory/(key.first+"-"+tier_name(key.second)+"-selected-test-predictions.pt"),ridge,tiny,transformed.valid,testing);
      const auto stream = named_stream(key.first+"/"+tier_name(key.second)+"/selected-test");
      out << ",\"ridge_test\":" << score_json(score(ridge,testing.labels,transformed.valid))
          << ",\"ridge_grouped_interval\":" << interval_json(grouped_accuracy_interval(ridge,testing.labels,transformed.valid,testing.source_ids,stream_seed(seed,stream)))
          << ",\"tiny_secondary_test\":" << score_json(score(tiny,testing.labels,transformed.valid))
          << ",\"tiny_secondary_grouped_interval\":" << interval_json(grouped_accuracy_interval(tiny,testing.labels,transformed.valid,testing.source_ids,stream_seed(seed,stream+1)));
    } else {
      out << ",\"reason\":" << quote(result.reason);
      FrozenStressReadout frozen;
      frozen.status=result.status; frozen.reason=result.reason; frozen.native_valid=surface.valid.clone();
      stress_readouts.emplace(key,std::move(frozen));
    }
    out << '}';
  }
  return out.str()+']';
}

FeatureSurface mask_surface(const ControlledDataset &split) {
  return {split.observed.feature_mask.to(torch::kFloat64).flatten(1),
      torch::ones({split.labels.size(0)},torch::kBool),"observation masks only; source-paired class-independent missingness"};
}
std::string pair_json(const PairComparison &pair,const std::map<Key,StressPredictions> &predictions,
                      const ControlledDataset &testing,uint64_t seed) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"id\":" << quote(pair.id) << ",\"candidate\":" << quote(pair.left)
      << ",\"comparator\":" << quote(pair.right) << ",\"tier\":" << quote(tier_name(pair.tier));
  const auto left=predictions.find({pair.left,pair.tier}),right=predictions.find({pair.right,pair.tier});
  if(left==predictions.end() || right==predictions.end())
    return out.str()+",\"status\":\"unsupported_compared_fit\"}";
  const auto common=left->second.valid.logical_and(right->second.valid);
  const auto stream=stream_seed(seed,named_stream("global-bottleneck/pair/"+pair.id));
  out << ",\"status\":\"measured\",\"common_population\":" << population_json(common,testing)
      << ",\"ridge_candidate\":" << score_json(score(left->second.ridge,testing.labels,common))
      << ",\"ridge_comparator\":" << score_json(score(right->second.ridge,testing.labels,common))
      << ",\"ridge_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(
          left->second.ridge,testing.labels,common,testing.source_ids,stream,1000,right->second.ridge))
      << ",\"tiny_secondary_candidate\":" << score_json(score(left->second.tiny,testing.labels,common))
      << ",\"tiny_secondary_comparator\":" << score_json(score(right->second.tiny,testing.labels,common))
      << ",\"tiny_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(
          left->second.tiny,testing.labels,common,testing.source_ids,stream,1000,right->second.tiny));
  return out.str()+'}';
}
} // namespace

GlobalBottleneckRun::GlobalBottleneckRun() {
  card.id="global-bottleneck-comparison-v1"; card.version=2;
  card.seeds={1401,1502,1603}; card.train_pairs=32; card.validation_pairs=64; card.test_pairs=64;
  const std::vector<std::pair<std::string,std::string>> pairs{
      {"mean_global","current_mixer"},{"learned_global","current_mixer"},{"learned_global","mean_global"}};
  for(const auto &[left,right]:pairs)
    card.comparisons.push_back({left+"_vs_"+right+"_global_matched",left+"_global",right+"_global",DimensionTier::matched_global});
  for(const auto &[left,right]:pairs) {
    card.comparisons.push_back({left+"_vs_"+right+"_global_native",left+"_global",right+"_global",DimensionTier::native});
    card.comparisons.push_back({left+"_vs_"+right+"_channels_matched",left+"_channel_concatenation",right+"_channel_concatenation",DimensionTier::matched_channels});
    card.comparisons.push_back({left+"_vs_"+right+"_channels_native",left+"_channel_concatenation",right+"_channel_concatenation",DimensionTier::native});
  }
}

void run_global_bottleneck_experiment(const GlobalBottleneckRun &run,const std::vector<NamedCurveFactory> &factories) {
  validate_run(run,factories);
  const fs::path output(run.output_directory);
  require(!output.empty() && !fs::exists(output),"output must be a new directory");
  fs::create_directories(output);
  write_text(output/"global-bottleneck-card.json",card_json(run,factories));
  if(run.stress_sweep) write_text(output/"stress-card.json",fixed_readout_stress_card_json(run.card));
  torch::set_num_threads(run.card.threads);
  std::vector<SeedState> states;
  states.reserve(run.card.seeds.size()*run.card.tasks.size());
  std::set<std::string> source_universe;
  std::ostringstream validation_report;
  validation_report << "{\"version\":1,\"protocol\":\"global-bottleneck-comparison-v1\","
      << "\"stage\":\"development\",\"card\":\"global-bottleneck-card.json\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"points\":[";
  bool first_point=true;
  for(const auto seed:run.card.seeds) for(const auto task:run.card.tasks) {
    SeedState state{seed,task,development_cohort(task,run,seed),{},nullptr,nullptr};
    require(!state.development.testing.observed.data.defined() && state.development.testing.source_ids.empty(),
            "development generation created testing data before selection");
    validate_split(state.development.training,run,run.card.train_pairs,source_universe);
    validate_split(state.development.validation,run,run.card.validation_pairs,source_universe);
    const auto seed_directory=output/("seed-"+std::to_string(seed)+"-"+task_name(task));
    fs::create_directory(seed_directory);
    save_split(seed_directory/"controlled-training.pt",state.development.training);
    save_split(seed_directory/"controlled-validation.pt",state.development.validation);
    write_text(seed_directory/"development-manifest.json","{\"version\":1,\"testing_generated\":false,\"training_reservoir_pairs\":"+
        std::to_string(run.training_reservoir_pairs)+",\"validation_seed_stream\":"+
        quote(std::to_string(run.validation_seed_stream))+",\"validation_seed\":"+
        quote(std::to_string(run.validation_seed_stream?stream_seed(seed,run.validation_seed_stream):seed))+",\"training\":"+
        split_manifest(state.development.training,"training")+",\"validation\":"+split_manifest(state.development.validation,"validation")+'}');
    {
      const MeasurementRngIsolation rng;
      const auto mask_training=mask_surface(state.development.training);
      state.mask_probe=std::make_unique<RidgeProbe>(mask_training,state.development.training.labels);
      state.mask_tiny=std::make_unique<TinyProbe>(mask_training,state.development.training.labels,
          stream_seed(seed,named_stream("mask_metadata/native")));
      torch::serialize::OutputArchive mask_fit;
      mask_fit.write("ridge_mean",state.mask_probe->normalizer.mean,true); mask_fit.write("ridge_scale",state.mask_probe->normalizer.scale,true);
      mask_fit.write("ridge_weights",state.mask_probe->weights,true); mask_fit.write("ridge_intercept",state.mask_probe->intercept,true);
      mask_fit.write("tiny_mean",state.mask_tiny->normalizer.mean,true); mask_fit.write("tiny_scale",state.mask_tiny->normalizer.scale,true);
      mask_fit.write("tiny_w1",state.mask_tiny->w1,true); mask_fit.write("tiny_b1",state.mask_tiny->b1,true);
      mask_fit.write("tiny_w2",state.mask_tiny->w2,true); mask_fit.write("tiny_b2",state.mask_tiny->b2,true);
      save_archive(seed_directory/"mask-metadata-fit.pt",mask_fit);
    }
    for(const auto &factory:factories) {
      const ProviderFitInput input{legal_clone(state.development.training.observed),run.card.shape,seed,
          state.development.training.source_ids,run.card.channel_ids,run.card.feature_units,
          "global-bottleneck-comparison-v1/"+task_name(task),run.card.sampling_interval,
          (run.card.shape.history_length-1)*run.card.sampling_interval};
      const auto trainer=factory.factory(input);
      require(bool(trainer.train_to) && bool(trainer.save_checkpoint) && bool(trainer.snapshot),"incomplete trainer callbacks");
      Architecture architecture{factory.name,{}};
      const auto directory=seed_directory/factory.name;
      fs::create_directory(directory);
      write_text(directory/"trainer-audit.json",fields(trainer.audit_fields));
      for(const auto budget:run.milestones) {
        Point point;
        point.directory=directory/("milestone-"+std::to_string(budget));
        fs::create_directory(point.directory);
        point.progress=trainer.train_to(budget);
        const MeasurementRngIsolation measurement_rng;
        const CurveProgress *previous=architecture.points.empty()?nullptr:&architecture.points.rbegin()->second.progress;
        check_progress(point.progress,budget,previous);
        const auto checkpoint=point.directory/"checkpoint.pt";
        trainer.save_checkpoint(checkpoint.string());
        require(fs::is_regular_file(checkpoint),"trainer did not retain the ordinary checkpoint");
        point.snapshot=named_snapshot(trainer.snapshot(checkpoint.string()),factory.name);
        require(bool(point.snapshot.features.extract) && !point.snapshot.features.provenance.empty() &&
                bool(point.snapshot.reconstruct),"incomplete immutable checkpoint snapshot");
        point.training=extract(point.snapshot.features,state.development.training.observed,run);
        point.validation=extract(point.snapshot.features,state.development.validation.observed,run,&point.training);
        if(!architecture.points.empty()) {
          const auto &prior=architecture.points.rbegin()->second;
          for(const auto &[name,surface]:point.training) {
            const auto &old_surface=prior.training.at(name);
            const auto &before=prior.snapshot.features.surfaces.at(name),&after=point.snapshot.features.surfaces.at(name);
            require(surface.values.size(1)==old_surface.values.size(1) && surface.values.scalar_type()==old_surface.values.scalar_type() &&
                before.kind==after.kind && before.channel_order==after.channel_order && before.support_rule==after.support_rule,
                "checkpoint changed export geometry/semantics");
          }
        }
        const auto assets=point.directory/"provider-assets";
        fs::create_directory(assets);
        if(point.snapshot.features.save_assets) point.snapshot.features.save_assets(assets.string());
        write_text(assets/"provider-audit.json","{\"provenance\":"+quote(point.snapshot.features.provenance)+
            ",\"audit_fields\":"+fields(point.snapshot.features.audit_fields)+'}');
        const auto features=fit_readouts(point,state.development,run,seed);
        const auto train_loss=reconstruction(point.directory/"training-reconstruction.pt",state.development.training,point.snapshot,run);
        const auto val_loss=reconstruction(point.directory/"validation-reconstruction.pt",state.development.validation,point.snapshot,run);
        point.validation_json="{\"seed\":"+quote(std::to_string(seed))+",\"task\":"+quote(task_name(task))+
            ",\"architecture\":"+quote(factory.name)+",\"milestone\":"+std::to_string(budget)+
            ",\"directory\":"+quote(fs::relative(point.directory,output).generic_string())+",\"checkpoint\":\"checkpoint.pt\","
            "\"progress\":"+progress_json(point.progress)+",\"training_reconstruction\":"+train_loss+
            ",\"validation_reconstruction\":"+val_loss+",\"features\":"+features+'}';
        write_text(point.directory/"point.json",point.validation_json);
        if(!first_point) validation_report << ',';
        first_point=false;
        validation_report << point.validation_json;
        const auto &primary=point.results.at({point.global,DimensionTier::matched_global});
        std::cout << "global-bottleneck variant=" << factory.name << " master=" << seed << " task=" << task_name(task)
            << " milestone=" << budget << " attempted=" << point.progress.attempted
            << " completed=" << point.progress.completed << " training_seconds=" << point.progress.training_seconds
            << " validation_global_ridge=";
        if(primary.status=="measured") std::cout << score(primary.ridge_validation,state.development.validation.labels,primary.validation_valid).accuracy;
        else std::cout << primary.status;
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
  double best=-std::numeric_limits<double>::infinity();
  for(const auto budget:run.milestones) {
    if(!budget) continue;
    auto candidate=utility(budget,states,run);
    if(candidate.supported && (run.fixed_test_budget ? budget==run.fixed_test_budget : candidate.utility>best)) {
      selected=budget;best=candidate.utility;
    }
    candidates.push_back(std::move(candidate));
  }
  require(selected.has_value(),run.fixed_test_budget ?
          "declared fixed testing budget does not support every selection-task variant/master and both validation classes" :
          "no nonzero budget supports every selection-task variant/master and both validation classes");
  std::ostringstream selection;
  selection << std::setprecision(17) << "{\"version\":1,\"stage\":\"development\",\"selection_frozen_before_testing\":true,"
      << "\"selected_budget\":" << *selected << ",\"utility\":" << best
      << ",\"selection_policy\":" << quote(run.fixed_test_budget ? "declared fixed supported nonzero budget; no validation budget search" :
          "maximum mean validation primary; exact ties select lower nonzero budget")
      << ",\"fixed_test_budget\":" << run.fixed_test_budget
      << ",\"task\":" << quote(task_name(run.selection_task))
      << ",\"metric\":\"mean validation matched-global ridge accuracy; all three variants and masters equal; all-variant common validity\","
      << "\"tie_rule\":\"smallest exact-tied nonzero budget\",\"candidates\":[";
  for(size_t i=0;i<candidates.size();++i) {
    if(i) selection << ',';
    const auto &candidate=candidates[i];
    selection << "{\"budget\":" << candidate.budget << ",\"status\":" << quote(candidate.supported?"measured":"unsupported_common_budget")
        << ",\"utility\":";
    if(candidate.supported) selection << candidate.utility;else selection << "null";
    selection << ",\"entries\":" << candidate.details << '}';
  }
  selection << "],\"validation_report\":\"validation-report.json\",\"source_fingerprint\":" << quote(run.source_fingerprint) << '}';
  write_text(output/"selection.json",selection.str());
  std::ostringstream report,stress_report;
  report << "{\"version\":1,\"protocol\":\"global-bottleneck-comparison-v1\",\"stage\":\"development\","
      << "\"card\":\"global-bottleneck-card.json\",\"validation_report\":\"validation-report.json\",\"selection\":\"selection.json\","
      << "\"source_fingerprint_algorithm\":\"sha256-source-manifest-v1\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"selected_budget\":" << *selected << ",\"interpretation\":\"fresh synthetic development; no across-seed CI or acceptance\",\"testing_runs\":[";
  stress_report << "{\"protocol\":\"fixed-readout-stress-v1\",\"stage\":\"development\",\"card\":\"stress-card.json\",\"runs\":[";
  bool first_test=true;
  for(auto &state:states) {
    const MeasurementRngIsolation selected_measurement_rng;
    // Validate the retained selected witness before creating fresh observations.
    // A factory that captured a live training model violates the snapshot contract.
    for(auto &architecture:state.architectures) {
      auto &point=architecture.points.at(*selected);
      const auto witness=extract(point.snapshot.features,state.development.validation.observed,run,&point.training);
      for(const auto &[name,surface]:witness) {
        const auto &original=point.validation.at(name);
        const auto rows=surface.valid.nonzero().flatten();
        require(torch::equal(surface.valid,original.valid) &&
                torch::equal(surface.values.index_select(0,rows),original.values.index_select(0,rows)),
                "later training mutated the selected checkpoint snapshot");
      }
      for(const auto &[key,result]:point.results) if(result.readout) {
        const auto values=result.readout->transform(point.validation.at(key.first));
        require(torch::equal(result.readout->ridge.predict(values),result.ridge_validation) &&
                torch::equal(result.readout->tiny.predict(values),result.tiny_validation),
                "later measurement mutated the selected fitted readout");
      }
    }
    const auto test_seed=stream_seed(state.seed,run.fresh_test_stream);
    const auto testing=make_controlled_test_dataset(state.task,run.card.shape,run.card.test_pairs,test_seed);
    validate_split(testing,run,run.card.test_pairs,source_universe);
    const auto directory=output/("seed-"+std::to_string(state.seed)+"-"+task_name(state.task));
    save_split(directory/"controlled-selected-testing.pt",testing);
    write_text(directory/"selected-test-manifest.json","{\"fresh_test_seed\":"+quote(std::to_string(test_seed))+
        ",\"selection\":\"../selection.json\",\"testing\":"+split_manifest(testing,"testing")+'}');
    const auto oracle=raw_oracle(state.task,testing.observed);
    const auto oracle_score=score(oracle.predictions,testing.labels,oracle.valid);
    require(oracle_score.supported && oracle_score.accuracy>=.95,"selected-test legal raw oracle failed task solvability");
    const auto metadata=mask_surface(testing);
    const auto mask_ridge=state.mask_probe->predict(metadata),mask_tiny=state.mask_tiny->predict(metadata);
    save_predictions(directory/"mask-metadata-selected-test-predictions.pt",mask_ridge,mask_tiny,metadata.valid,testing);
    if(!first_test) {report << ',';stress_report << ',';}
    first_test=false;
    report << "{\"development_seed\":" << quote(std::to_string(state.seed)) << ",\"task\":" << quote(task_name(state.task))
        << ",\"fresh_test_seed\":" << quote(std::to_string(test_seed))
        << ",\"source_manifest\":" << quote(directory.filename().string()+"/selected-test-manifest.json")
        << ",\"raw_oracle\":" << score_json(oracle_score) << ",\"mask_metadata_ridge\":" << score_json(score(mask_ridge,testing.labels,metadata.valid))
        << ",\"mask_metadata_tiny_secondary\":" << score_json(score(mask_tiny,testing.labels,metadata.valid)) << ",\"architectures\":[";
    std::map<Key,StressPredictions> predictions;
    StressReadouts stress_readouts;
    std::map<std::string,SurfaceDescription> descriptions;
    for(size_t index=0;index<state.architectures.size();++index) {
      auto &architecture=state.architectures[index];
      auto &point=architecture.points.at(*selected);
      const auto features=evaluate_test(point,testing,run,state.seed,predictions,stress_readouts);
      descriptions.insert(point.snapshot.features.surfaces.begin(),point.snapshot.features.surfaces.end());
      const auto reconstructed=reconstruction(point.directory/"selected-test-reconstruction.pt",testing,point.snapshot,run);
      if(index) report << ',';
      report << "{\"name\":" << quote(architecture.name) << ",\"checkpoint\":"
          << quote(fs::relative(point.directory/"checkpoint.pt",output).generic_string())
          << ",\"fitted_readouts_reused\":true,\"features\":" << features << ",\"reconstruction\":" << reconstructed << '}';
    }
    report << "],\"pairs\":[";
    for(size_t i=0;i<run.card.comparisons.size();++i) {
      if(i) report << ',';
      report << pair_json(run.card.comparisons[i],predictions,testing,state.seed);
    }
    report << "]}";
    if(run.stress_sweep) {
      descriptions.emplace("mask_metadata",SurfaceDescription{SurfaceKind::control,"mask metadata; all rows including allmissing",{}});
      FrozenStressReadout mask_frozen;
      mask_frozen.status="measured";mask_frozen.probe_dimensions=metadata.values.size(1);
      mask_frozen.native_valid=metadata.valid.clone();mask_frozen.native_ridge=mask_ridge.clone();mask_frozen.native_tiny=mask_tiny.clone();
      const StressPredictions mask_ordinary{metadata.valid.clone(),mask_ridge.clone(),mask_tiny.clone()};
      mask_frozen.predict=[&state,mask_ordinary,first=true](const FeatureSurface &surface) mutable {
        StressPredictions out{surface.valid.clone(),state.mask_probe->predict(surface),state.mask_tiny->predict(surface)};
        if(first) {
          require(torch::equal(out.valid,mask_ordinary.valid) && torch::equal(out.ridge,mask_ordinary.ridge) && torch::equal(out.tiny,mask_ordinary.tiny),
                  "intact stress metadata predictions differ from ordinary selected-test predictions");
          first=false;
        }
        return out;
      };
      stress_readouts.emplace(Key{"mask_metadata",DimensionTier::native},std::move(mask_frozen));
      const auto extractor=[&state,selected,&run](const Batch &batch) {
        FeatureMap out;
        for(const auto &architecture:state.architectures) {
          const auto &point=architecture.points.at(*selected);
          const auto values=extract(point.snapshot.features,batch,run,&point.training);
          out.insert(values.begin(),values.end());
        }
        out.emplace("mask_metadata",FeatureSurface{batch.feature_mask.to(torch::kFloat64).flatten(1),
            torch::ones({batch.data.size(0)},torch::kBool),"observation masks only; source-paired class-independent missingness"});
        return out;
      };
      ControlledProtocol protocol{state.task,state.seed,run.card.shape,state.development.training,state.development.validation,testing};
      const auto stress=run_fixed_readout_stress(directory.string(),run.card,protocol,descriptions,stress_readouts,extractor);
      stress_report << "{\"seed\":" << quote(std::to_string(state.seed)) << ",\"task\":" << quote(task_name(state.task)) << ",\"report\":" << stress << '}';
    }
  }
  report << "]}";
  write_text(output/"report.json",report.str());
  if(run.stress_sweep) {stress_report << "]}";write_text(output/"stress-report.json",stress_report.str());}
}
} // namespace embedding::evaluation
