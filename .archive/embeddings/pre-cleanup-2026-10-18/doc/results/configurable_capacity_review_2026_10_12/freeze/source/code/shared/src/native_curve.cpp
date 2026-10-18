// SPDX-License-Identifier: MIT
#include "embedding/shared/native_curve.h"
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
#include <fcntl.h>
#include <unistd.h>

namespace embedding::evaluation {
namespace {
namespace fs=std::filesystem;
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[native curve] " + message);
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
std::string score_json(const Score &score) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"total\":" << score.total << ",\"valid\":" << score.valid
      << ",\"correct\":" << score.correct << ",\"abstained\":" << score.total-score.valid
      << ",\"full_population_correctness\":" << (score.total?double(score.correct)/score.total:0)
      << ",\"coverage\":" << score.coverage << ",\"accuracy\":";
  if (score.supported) out << score.accuracy; else out << "null";
  return out.str() + '}';
}
std::string interval_json(const GroupedInterval &interval,int64_t replicates) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"method\":\"source-group percentile bootstrap; within-run conditional on fitted checkpoint/readouts\","
      << "\"replicates\":" << replicates << ",\"confidence\":0.95,\"source_groups\":" << interval.source_groups << ",\"estimate\":";
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
void validate_split(const ControlledDataset &split, const NativeCurveRun &run, int64_t pairs,
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

std::string reconstruction(const fs::path &path, const ControlledDataset &split,
                           const CurveSnapshot &snapshot, const NativeCurveRun &run) {
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

struct MeasurementIsolation {
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  MeasurementIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for(size_t i=0;i<at::getNumGPUs();++i)generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for(const auto &generator:generators)states.push_back(generator.get_state().clone());
  }
  ~MeasurementIsolation() noexcept {
    try{for(size_t i=0;i<generators.size();++i)generators[i].set_state(states[i]);}catch(...){std::terminate();}
  }
};
struct ThreadsIsolation {
  int previous{at::get_num_threads()};
  ~ThreadsIsolation() noexcept {try{at::set_num_threads(previous);}catch(...){std::terminate();}}
};
void durable_selection(const fs::path &path,const std::string &value) {
  write_text(path,value);
  const int file=::open(path.c_str(),O_RDONLY);
  require(file>=0,"cannot open persisted selection for fsync");
  const int synced=::fsync(file),closed=::close(file);
  require(synced==0 && closed==0,"cannot durably flush persisted selection");
  const int directory=::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY);
  require(directory>=0,"cannot open selection directory for fsync");
  const int directory_synced=::fsync(directory),directory_closed=::close(directory);
  require(directory_synced==0 && directory_closed==0,"cannot durably flush selection directory");
}
struct Readout {
  FeatureNormalizer normalizer;
  std::optional<TrainPca> pca;
  RidgeProbe ridge;
  TinyProbe tiny;
  uint64_t seed;
  Readout(FeatureNormalizer outer,std::optional<TrainPca> projection,const FeatureSurface &training,
          const torch::Tensor &labels,const NativeCurveRun &run,uint64_t actual_seed)
      :normalizer(std::move(outer)),pca(std::move(projection)),ridge(training,labels,run.ridge_penalty),
       tiny(training,labels,actual_seed,run.tiny_steps,run.tiny_hidden,run.tiny_learning_rate),seed(actual_seed) {}
  FeatureSurface transform(const FeatureSurface &surface) const {
    auto out=normalizer.transform(surface);if(pca)out=pca->transform(out);validate_features(out);return out;
  }
  StressPredictions predict(const FeatureSurface &surface) const {
    torch::NoGradGuard no_grad;const auto out=transform(surface);
    const auto ridge_logits=ridge.normalizer.transform(out).values.matmul(ridge.weights)+ridge.intercept;
    const auto hidden=tiny.normalizer.transform(out).values.matmul(tiny.w1)+tiny.b1;
    require(torch::isfinite(ridge_logits).all().item<bool>() && torch::isfinite(hidden).all().item<bool>(),"probe arithmetic overflow");
    const auto tiny_logits=torch::tanh(hidden).matmul(tiny.w2)+tiny.b2;
    require(torch::isfinite(tiny_logits).all().item<bool>(),"neural probe logits overflow");
    return {out.valid.clone(),ridge_logits.argmax(1),tiny_logits.argmax(1)};
  }
};
std::vector<torch::Tensor> fitted_state(const Readout &readout) {
  std::vector<torch::Tensor> out{readout.normalizer.mean,readout.normalizer.scale,readout.ridge.normalizer.mean,readout.ridge.normalizer.scale,
      readout.ridge.weights,readout.ridge.intercept,readout.tiny.normalizer.mean,readout.tiny.normalizer.scale,
      readout.tiny.w1,readout.tiny.b1,readout.tiny.w2,readout.tiny.b2};
  if(readout.pca){out.push_back(readout.pca->mean);out.push_back(readout.pca->components);out.push_back(readout.pca->singular_values);}
  for(auto &tensor:out) tensor=tensor.detach().clone();
  return out;
}
void same_state(const Readout &readout,const std::vector<torch::Tensor> &expected) {
  const auto actual=fitted_state(readout);require(actual.size()==expected.size(),"frozen readout structure changed");
  for(size_t i=0;i<actual.size();++i)require(torch::equal(actual[i],expected[i]),"frozen TRAIN-fitted readout changed");
}
void save_fit(const fs::path &path,const Readout &readout) {
  torch::serialize::OutputArchive out;
  out.write("feature_mean",readout.normalizer.mean,true);out.write("feature_scale",readout.normalizer.scale,true);
  out.write("fitted_rows",torch::tensor(readout.normalizer.fitted_rows),true);
  if(readout.pca){out.write("pca_mean",readout.pca->mean,true);out.write("pca_components",readout.pca->components,true);
    out.write("pca_singular_values",readout.pca->singular_values,true);out.write("pca_numerical_rank",torch::tensor(readout.pca->numerical_rank),true);}
  out.write("ridge_mean",readout.ridge.normalizer.mean,true);out.write("ridge_scale",readout.ridge.normalizer.scale,true);
  out.write("ridge_weights",readout.ridge.weights,true);out.write("ridge_intercept",readout.ridge.intercept,true);
  out.write("tiny_mean",readout.tiny.normalizer.mean,true);out.write("tiny_scale",readout.tiny.normalizer.scale,true);
  out.write("tiny_w1",readout.tiny.w1,true);out.write("tiny_b1",readout.tiny.b1,true);out.write("tiny_w2",readout.tiny.w2,true);out.write("tiny_b2",readout.tiny.b2,true);
  out.write("actual_probe_seed_decimal",archive::text_tensor(std::to_string(readout.seed)),true);save_archive(path,out);
}
void save_surface(const fs::path &path,const FeatureSurface &surface,const ControlledDataset &split) {
  torch::serialize::OutputArchive out;out.write("features",surface.values,true);out.write("valid",surface.valid,true);
  out.write("provenance",archive::text_tensor(surface.provenance),true);out.write("labels_scoring_only",split.labels,true);
  out.write("source_ids_json",archive::text_tensor(strings(split.source_ids)),true);save_archive(path,out);
}
void save_predictions(const fs::path &path,const StressPredictions &prediction,const ControlledDataset &split) {
  torch::serialize::OutputArchive out;out.write("ridge",prediction.ridge,true);out.write("tiny_secondary",prediction.tiny,true);
  out.write("valid",prediction.valid,true);out.write("labels_scoring_only",split.labels,true);
  out.write("source_ids_json",archive::text_tensor(strings(split.source_ids)),true);save_archive(path,out);
}
FeatureSurface global_surface(const CurveSnapshot &snapshot,const Batch &batch,const NativeCurveRun &run) {
  require(bool(snapshot.features.extract) && !snapshot.features.provenance.empty(),"snapshot has no frozen extraction/provenance");
  std::optional<std::string> global;
  for(const auto &[name,description]:snapshot.features.surfaces)if(description.kind==SurfaceKind::global) {
    require(!global && safe_name(name) && description.channel_order.empty() && !description.support_rule.empty(),"snapshot requires one declared typed global export");global=name;
  }
  require(global.has_value(),"snapshot has no typed global export");
  auto all=snapshot.features.extract(legal_clone(batch));require(all.count(*global),"snapshot omitted global export");
  auto out=all.at(*global);validate_features(out);
  require(out.values.size(0)==batch.data.size(0) && out.values.size(1)==run.export_width &&
      !out.valid.logical_and(batch.feature_mask.flatten(1).any(1).logical_not()).any().item<bool>(),"native global geometry/support mismatch");
  return {out.values.detach().clone(),out.valid.clone(),out.provenance+"; "+snapshot.features.provenance};
}
struct Result {
  std::string status{"unsupported_fit"},reason;
  std::unique_ptr<Readout> readout;
  StressPredictions training,validation;
  std::vector<torch::Tensor> witness;
};
struct Point {
  fs::path directory;
  CurveProgress progress;
  CurveSnapshot snapshot;
  FeatureSurface training,validation;
  std::vector<Result> results;
};
struct Control {
  std::string name,label;
  int64_t width;
  FeatureSurface training,validation;
  std::vector<Result> results;
};
struct Cohort {
  uint64_t seed;
  Task task;
  ControlledProtocol development;
  fs::path directory;
  std::unique_ptr<ObservationScaler> raw_scaler;
  std::vector<torch::Tensor> scaler_witness;
  std::vector<Control> controls;
  std::map<int64_t,Point> points;
};
bool fit_supported(const FeatureSurface &surface,const torch::Tensor &labels) {
  const auto selected=labels.masked_select(surface.valid);
  return selected.numel()>=2 && selected.eq(0).any().item<bool>() && selected.eq(1).any().item<bool>();
}
FeatureSurface raw_surface(const ObservationScaler &scaler,const Batch &batch) {
  const auto prepared=scaler.transform(batch);
  return {torch::cat({prepared.data.flatten(1),prepared.feature_mask.to(torch::kFloat64).flatten(1)},1),prepared.feature_mask.flatten(1).any(1),
      "TRAIN-only float64 ObservationScaler per-channel/feature across rows/history; raw values plus original maskflags; no encoder"};
}
FeatureSurface mask_surface(const Batch &batch) {
  return {batch.feature_mask.to(torch::kFloat64).flatten(1),torch::ones({batch.data.size(0)},torch::kBool),"class-independent source-paired observation masks only"};
}
std::vector<Result> fit_readouts(const FeatureSurface &training,const FeatureSurface &validation,
                                const ControlledProtocol &protocol,const NativeCurveRun &run,int64_t width,
                                bool pca_only,const fs::path &directory) {
  std::vector<Result> results;
  std::optional<FeatureNormalizer> normalizer;
  std::optional<TrainPca> pca;
  FeatureSurface fitted;
  std::string reason;
  int64_t outer_fits=0,pca_attempts=0,pca_fits=0,head_fits=0;
  if(!fit_supported(training,protocol.training.labels)) reason="valid TRAIN rows require both classes and >=2 rows";
  else {
    normalizer.emplace(training);++outer_fits;
    fitted=normalizer->transform(training);
    if(pca_only) {
      try {++pca_attempts;pca.emplace(fitted,width);++pca_fits;fitted=pca->transform(fitted);validate_features(fitted);}
      catch(const std::runtime_error &error) {
        const std::string message=error.what();
        require(message.find("PCA dimensions exceed")!=std::string::npos,message);
        reason=message;
      }
    }
  }
  // Common TRAIN-only transform is fitted once, then immutable copies are used
  // by every declared head repetition. No milestone/test/stress control refit.
  for(const auto &rep:run.repetitions) {
    Result result;result.reason=reason;
    if(reason.empty()) {
      const auto actual_seed=stream_seed(rep.probe_seed,uint64_t(width));
      result.readout=std::make_unique<Readout>(*normalizer,pca,fitted,protocol.training.labels,run,actual_seed);
      ++head_fits;
      result.training=result.readout->predict(training);result.validation=result.readout->predict(validation);result.status="measured";
      result.witness=fitted_state(*result.readout);
      const auto rep_directory=directory/rep.id;require(fs::create_directory(rep_directory),"readout repetition directory exists");
      save_fit(rep_directory/"fit.pt",*result.readout);
      save_predictions(rep_directory/"training-predictions.pt",result.training,protocol.training);
      save_predictions(rep_directory/"validation-predictions.pt",result.validation,protocol.validation);
    }
    results.push_back(std::move(result));
  }
  write_text(directory/"fit-counts.json","{\"outer_train_normalizer_fits\":"+std::to_string(outer_fits)+
      ",\"standalone_pca_attempts\":"+std::to_string(pca_attempts)+",\"standalone_pca_fits\":"+std::to_string(pca_fits)+
      ",\"ridge_fits\":"+std::to_string(head_fits)+",\"tiny_fits\":"+std::to_string(head_fits)+
      ",\"validation_test_stress_fits\":0}");
  return results;
}
std::string results_json(const std::vector<Result> &results,const FeatureSurface &training,const FeatureSurface &validation,
                         const ControlledProtocol &protocol,const NativeCurveRun &run,int64_t width) {
  std::ostringstream out;out << '[';
  for(size_t i=0;i<results.size();++i) {
    if(i)out << ',';
    const auto &result=results[i];out << "{\"id\":" << quote(run.repetitions[i].id) << ",\"status\":" << quote(result.status)
        << ",\"reason\":" << quote(result.reason) << ",\"size\":" << width << ",\"actual_probe_seed\":" << quote(std::to_string(stream_seed(run.repetitions[i].probe_seed,uint64_t(width))))
        << ",\"ridge_parameters\":" << 2*width+2 << ",\"neural_parameters\":" << run.tiny_hidden*(width+3)+2
        << ",\"training_population\":" << population_json(training.valid,protocol.training) << ",\"validation_population\":" << population_json(validation.valid,protocol.validation);
    if(result.readout)out << ",\"ridge_training\":" << score_json(score(result.training.ridge,protocol.training.labels,result.training.valid))
        << ",\"tiny_training\":" << score_json(score(result.training.tiny,protocol.training.labels,result.training.valid))
        << ",\"ridge_validation\":" << score_json(score(result.validation.ridge,protocol.validation.labels,result.validation.valid))
        << ",\"tiny_secondary_validation\":" << score_json(score(result.validation.tiny,protocol.validation.labels,result.validation.valid));
    out << '}';
  }
  return out.str()+']';
}
void fit_controls(Cohort &cohort,const NativeCurveRun &run) {
  const MeasurementIsolation rng;
  int64_t observation_scaler_fits=0;
  cohort.raw_scaler=std::make_unique<ObservationScaler>(cohort.development.training.observed);++observation_scaler_fits;
  const auto &scaler=*cohort.raw_scaler;cohort.scaler_witness={scaler.mean.clone(),scaler.scale.clone(),scaler.counts.clone()};
  const auto root=cohort.directory/"controls";require(fs::create_directory(root),"control directory exists");
  torch::serialize::OutputArchive scalar;scalar.write("mean",scaler.mean,true);scalar.write("scale",scaler.scale,true);scalar.write("counts",scaler.counts,true);save_archive(root/"raw-scaler.pt",scalar);
  const auto training=raw_surface(scaler,cohort.development.training.observed),validation=raw_surface(scaler,cohort.development.validation.observed);
  cohort.controls.push_back({"raw","Raw data — no encoder",training.values.size(1),training,validation,{}});
  cohort.controls.push_back({"pca_only","PCA only — no encoder",run.export_width,training,validation,{}});
  cohort.controls.push_back({"mask_metadata","Mask metadata",run.card.shape.channel_count*run.card.shape.history_length*run.card.shape.input_width,
      mask_surface(cohort.development.training.observed),mask_surface(cohort.development.validation.observed),{}});
  std::ostringstream report;report << "{\"fit_policy\":\"once per cohort on TRAIN only; retained across milestones/test/stress\",\"observation_scaler_train_fits\":" << observation_scaler_fits << ",\"controls\":[";
  for(size_t i=0;i<cohort.controls.size();++i) {
    if(i)report << ',';
    auto &control=cohort.controls[i];const auto directory=root/control.name;require(fs::create_directory(directory),"control fit directory exists");
    save_surface(directory/"training-input.pt",control.training,cohort.development.training);save_surface(directory/"validation-input.pt",control.validation,cohort.development.validation);
    control.results=fit_readouts(control.training,control.validation,cohort.development,run,control.width,control.name=="pca_only",directory);
    report << "{\"method\":" << quote(control.name) << ",\"label\":" << quote(control.label) << ",\"size\":" << control.width
        << ",\"repetitions\":" << results_json(control.results,control.training,control.validation,cohort.development,run,control.width) << '}';
  }
  write_text(root/"controls.json",report.str()+"]}");
}
void verify_result(const Result &result,const FeatureSurface &validation) {
  if(!result.readout)return;
  same_state(*result.readout,result.witness);const auto actual=result.readout->predict(validation);
  require(torch::equal(actual.valid,result.validation.valid) && torch::equal(actual.ridge,result.validation.ridge) && torch::equal(actual.tiny,result.validation.tiny),
      "TRAIN-fitted readout changed after later measurements/training");
}
void verify_controls(const Cohort &cohort) {
  const auto &scaler=*cohort.raw_scaler;
  require(torch::equal(scaler.mean,cohort.scaler_witness[0]) && torch::equal(scaler.scale,cohort.scaler_witness[1]) && torch::equal(scaler.counts,cohort.scaler_witness[2]),"frozen raw scaler changed");
  for(const auto &control:cohort.controls)for(const auto &result:control.results)verify_result(result,control.validation);
}
void compare_reconstruction(const fs::path &original,const fs::path &witness) {
  torch::serialize::InputArchive left,right;left.load_from(original.string(),torch::kCPU);right.load_from(witness.string(),torch::kCPU);
  for(const auto &key:{"standardized_prediction","standardized_target","target_mask","requested_observed_target_mask","visible_mask","trial_channel_eligible"}) {
    torch::Tensor before,after;left.read(key,before,true);right.read(key,after,true);require(torch::equal(before,after),"later training changed exact checkpoint reconstruction witness");
  }
}
void validate_run(const NativeCurveRun &run,const NamedCurveFactory &factory) {
  const auto &card=run.card;const auto &shape=card.shape;
  if(run.development_only)require(card.test_pairs==0 && !run.stress_sweep,"development-only requires zero TEST pairs and disabled stress sweep");
  require(card.version==2 && card.policy_version=="1.2" && card.stage=="development" && safe_name(card.id),"new development policy/card contract");
  require(shape.channel_count>0 && shape.history_length>=8 && shape.input_width>0 && shape.dtype==torch::kFloat64 && shape.device.is_cpu(),"CPU float64 positive geometry required");
  require(shape.channel_count<=std::numeric_limits<int64_t>::max()/shape.history_length && shape.channel_count*shape.history_length<=std::numeric_limits<int64_t>::max()/shape.input_width/2,"geometry overflow");
  require(run.patch_length>0 && shape.history_length%run.patch_length==0 && shape.history_length/run.patch_length>=3,"complete patch geometry requires >=3 patches");
  require(card.channel_ids.size()==size_t(shape.channel_count) && std::set<int64_t>(card.channel_ids.begin(),card.channel_ids.end()).size()==card.channel_ids.size(),"channel ID geometry/uniqueness");
  require(!card.feature_units.empty() && size_t(1+std::count(card.feature_units.begin(),card.feature_units.end(),','))==size_t(shape.input_width),"feature units width");
  require(std::isfinite(card.sampling_interval) && card.sampling_interval>0 && std::isfinite((shape.history_length-1)*card.sampling_interval),"finite uniform sampling metadata");
  require(card.train_pairs>=2 && card.validation_pairs>0 && (run.development_only?card.test_pairs==0:card.test_pairs>0) && card.train_pairs<=std::numeric_limits<int64_t>::max()/2 && card.validation_pairs<=std::numeric_limits<int64_t>::max()/2 && card.test_pairs<=std::numeric_limits<int64_t>::max()/2,"split pair counts");
  require(!card.seeds.empty() && std::set<uint64_t>(card.seeds.begin(),card.seeds.end()).size()==card.seeds.size(),"master seeds empty/duplicate");
  require(!card.tasks.empty() && std::set<Task>(card.tasks.begin(),card.tasks.end()).size()==card.tasks.size() && (run.development_only || std::find(card.tasks.begin(),card.tasks.end(),run.selection_task)!=card.tasks.end()),"task list and selection task");
  for(const auto task:card.tasks) {task_name(task);require(task!=Task::lag_sign || shape.channel_count>=2,"lag task needs >=2 channels");}
  require(card.threads>0 && run.export_width>0 && !run.model_tag.empty() && (run.development_only || run.sanity_budget>0),"thread/export/tag/sanity settings");
  require(run.milestones.size()>=2 && run.milestones.front()==0 && std::is_sorted(run.milestones.begin(),run.milestones.end()) && std::adjacent_find(run.milestones.begin(),run.milestones.end())==run.milestones.end(),"ordered unique point0/positive milestones");
  for(const auto budget:run.milestones)require(budget>=0,"negative budget");
  require(std::isfinite(run.huber_delta) && run.huber_delta>0 && std::isfinite(run.missing_rate) && run.missing_rate>=0 && run.missing_rate<1,"reconstruction/missingness parameters");
  require(std::isfinite(run.ridge_penalty) && run.ridge_penalty>0 && run.tiny_hidden>0 && run.tiny_steps>0 && std::isfinite(run.tiny_learning_rate) && run.tiny_learning_rate>0 && run.bootstrap_replicates>=100,"readout recipe");
  const auto maximum_width=std::max(run.export_width,2*shape.channel_count*shape.history_length*shape.input_width);
  require(maximum_width<=std::numeric_limits<int64_t>::max()/2-3 && run.tiny_hidden<=(std::numeric_limits<int64_t>::max()-2)/(maximum_width+3),"readout parameter count overflow");
  require(!run.repetitions.empty(),"no head repetitions");std::set<std::string> repetitions;
  for(const auto &rep:run.repetitions)require(safe_name(rep.id) && repetitions.insert(rep.id).second,"unsafe/duplicate head repetition");
  require(safe_name(factory.name) && !factory.recipe.empty() && bool(factory.factory),"sole trainer registration incomplete");
  std::set<std::string> pairs;
  for(const auto &pair:card.comparisons)require(safe_name(pair.id) && pairs.insert(pair.id).second && pair.tier==DimensionTier::native && pair.left=="native" && (pair.right=="raw" || pair.right=="pca_only" || pair.right=="untrained_native"),"native-only explicit comparison contract");
}
const char *native_protocol(const NativeCurveRun &run) {
  return run.development_only?"native-development-v1":"native-curve-v1";
}
const char *native_card_filename(const NativeCurveRun &run) {
  return run.development_only?"native-development-card.json":"native-curve-card.json";
}
std::string card_json(const NativeCurveRun &run,const NamedCurveFactory &factory) {
  const auto &card=run.card;std::vector<std::string> tasks;for(const auto task:card.tasks)tasks.push_back(task_name(task));
  std::ostringstream out;out << std::setprecision(17) << "{\"version\":1,\"protocol\":" << quote(native_protocol(run)) << ",\"policy_version\":\"1.2\",\"stage\":\"development\","
      << "\"id\":" << quote(card.id) << ",\"source_fingerprint\":" << quote(run.source_fingerprint) << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"model_tag\":" << quote(run.model_tag) << ",\"trainer\":" << quote(factory.name) << ",\"trainer_recipe\":" << quote(factory.recipe)
      << ",\"shape\":{\"channels\":" << card.shape.channel_count << ",\"history\":" << card.shape.history_length << ",\"features\":" << card.shape.input_width << ",\"dtype\":\"float64\"},\"channel_ids\":" << numbers(card.channel_ids)
      << ",\"feature_units\":" << quote(card.feature_units) << ",\"sampling_interval\":" << card.sampling_interval << ",\"patch_length\":" << run.patch_length
      << ",\"train_pairs\":" << card.train_pairs << ",\"validation_pairs\":" << card.validation_pairs << ",\"test_pairs\":" << card.test_pairs
      << ",\"masters\":" << numbers(card.seeds) << ",\"tasks\":" << strings(tasks) << ",\"threads\":" << card.threads << ",\"missing_rate\":" << run.missing_rate;
  if(run.development_only) {
    out << ",\"development_only\":true,\"milestones\":" << numbers(run.milestones)
        << ",\"selection_performed\":false,\"testing_generated\":false,\"stress_evaluated\":false";
  } else {
    out << ",\"selection_task\":" << quote(task_name(run.selection_task)) << ",\"timing_milestones\":" << numbers(run.milestones) << ",\"sanity_milestones\":[0," << run.sanity_budget << ']'
      << ",\"selection_policy\":\"mean native global ridge VALIDATION accuracy over masters, first declared repetition (ridge deterministic); identical within-cohort checkpoint support; unsupported entire budget excluded; positive budgets only; exact ties choose smaller\","
      << "\"fresh_test_namespace\":\"native-curve-v1/fresh-testing\",\"fresh_test_stream\":" << quote(std::to_string(run.fresh_test_stream))
      << ",\"all_testing_after_durable_selection\":true";
  }
  out << ",\"export_width\":" << run.export_width << ",\"post_encoder_pca\":false,"
      << "\"raw_scaling\":\"once TRAIN observed float64 per channel/feature across history, scale floor1e-8; hidden values zero; flatten values+flags; support any observation\","
      << "\"control_transform_fit_policy\":" << quote(run.development_only?
          "once per cohort: TRAIN outer population standardization; standalone raw PCA then probe own TRAIN normalization; immutable across milestones":
          "once per cohort: TRAIN outer population standardization; standalone raw PCA then probe own TRAIN normalization; immutable across checkpoints/test/stress") << ','
      << "\"native_fit_policy\":\"once per checkpoint: exact native export, TRAIN outer standardization then each probe own TRAIN normalization; no PCA\","
      << "\"point0_control\":\"actual saved initial checkpoint from same continuous training path and TRAIN-fitted scaler\","
      << "\"head_seed_policy\":\"stream_seed(declared repetition seed, actual probe width); equal-width PCA-only/native/untrained native paired; raw width separately declared\","
      << "\"ridge_penalty\":" << run.ridge_penalty << ",\"tiny_hidden\":" << run.tiny_hidden << ",\"tiny_steps\":" << run.tiny_steps << ",\"tiny_learning_rate\":" << run.tiny_learning_rate
      << ",\"bootstrap_replicates\":" << run.bootstrap_replicates << ",\"uncertainty\":\"95% source-group paired percentile intervals conditional on one fitted encoder/head; no across-master or head-repetition CI\","
      << "\"reconstruction\":{\"query\":\"each complete original patch hidden once; observed cells only; >=2 visible observed patch groups\",\"units\":\"frozen TRAIN scaler standardized\",\"primary\":\"cell/channel/example hierarchical MAE\",\"secondary\":\"same hierarchy Huber\",\"huber_delta\":" << run.huber_delta << "},";
  if(!run.development_only)out << "\"stress_sweep\":" << (run.stress_sweep?"true":"false") << ',';
  out << "\"repetitions\":[";
  for(size_t i=0;i<run.repetitions.size();++i){if(i)out << ',';out << "{\"id\":" << quote(run.repetitions[i].id) << ",\"probe_seed\":" << quote(std::to_string(run.repetitions[i].probe_seed)) << '}';}
  out << "],\"comparisons\":[";
  for(size_t i=0;i<card.comparisons.size();++i){if(i)out << ',';const auto &pair=card.comparisons[i];out << "{\"id\":" << quote(pair.id) << ",\"candidate\":" << quote(pair.left) << ",\"comparator\":" << quote(pair.right) << ",\"tier\":\"native\"}";}
  return out.str()+"]}";
}
void same_surface(const FeatureSurface &actual,const FeatureSurface &expected,const std::string &message) {
  require(actual.provenance==expected.provenance && actual.values.scalar_type()==expected.values.scalar_type() && torch::equal(actual.valid,expected.valid),message);
  const auto rows=actual.valid.nonzero().flatten();require(torch::equal(actual.values.index_select(0,rows),expected.values.index_select(0,rows)),message);
}
void witness(Point &point,const Cohort &cohort,const NativeCurveRun &run) {
  const auto current=global_surface(point.snapshot,cohort.development.validation.observed,run);
  same_surface(current,point.validation,"later training changed retained native checkpoint features");
  for(const auto &result:point.results)verify_result(result,point.validation);
  const auto path=point.directory/"validation-reconstruction-witness.pt";reconstruction(path,cohort.development.validation,point.snapshot,run);
  compare_reconstruction(point.directory/"validation-reconstruction.pt",path);
  if(run.development_only) {
    same_surface(global_surface(point.snapshot,cohort.development.training.observed,run),point.training,"later training changed retained native TRAIN checkpoint features");
    const auto training_path=point.directory/"training-reconstruction-witness.pt";
    reconstruction(training_path,cohort.development.training,point.snapshot,run);
    compare_reconstruction(point.directory/"training-reconstruction.pt",training_path);
    write_text(point.directory/"witness-audit.json","{\"native_features_exact\":true,\"training_native_features_exact\":true,\"fitted_readouts_exact\":true,\"reconstruction_exact\":true,\"training_reconstruction_exact\":true,\"performed_after_all_training\":true,\"performed_before_development_complete\":true}");
  } else {
    write_text(point.directory/"witness-audit.json","{\"native_features_exact\":true,\"fitted_readouts_exact\":true,\"reconstruction_exact\":true,\"performed_after_all_training\":true,\"performed_before_all_testing\":true}");
  }
}
std::string paired_json(const PairComparison &pair,const std::map<std::string,StressPredictions> &predictions,const ControlledDataset &testing,const NativeCurveRun &run,uint64_t seed) {
  const auto left=predictions.find(pair.left),right=predictions.find(pair.right);std::ostringstream out;
  out << "{\"id\":" << quote(pair.id) << ",\"candidate\":" << quote(pair.left) << ",\"comparator\":" << quote(pair.right) << ",\"tier\":\"native\",\"status\":";
  if(left==predictions.end() || right==predictions.end())return out.str()+"\"unsupported_fit\"}";
  const auto common=left->second.valid.logical_and(right->second.valid);const auto stream=stream_seed(seed,named_stream(pair.id));
  out << quote(common.any().item<bool>()?"measured":"unsupported_zero_common_support") << ",\"common_population\":" << population_json(common,testing)
      << ",\"ridge_candidate\":" << score_json(score(left->second.ridge,testing.labels,common)) << ",\"ridge_comparator\":" << score_json(score(right->second.ridge,testing.labels,common))
      << ",\"ridge_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(left->second.ridge,testing.labels,common,testing.source_ids,stream,run.bootstrap_replicates,right->second.ridge),run.bootstrap_replicates)
      << ",\"tiny_secondary_candidate\":" << score_json(score(left->second.tiny,testing.labels,common)) << ",\"tiny_secondary_comparator\":" << score_json(score(right->second.tiny,testing.labels,common))
      << ",\"tiny_candidate_minus_comparator_grouped_interval\":" << interval_json(grouped_accuracy_interval(left->second.tiny,testing.labels,common,testing.source_ids,stream,run.bootstrap_replicates,right->second.tiny),run.bootstrap_replicates);
  return out.str()+'}';
}
struct TestMethod {std::string name,label;int64_t width;FeatureSurface surface;const std::vector<Result> *results;fs::path fit_directory;};
std::string test_cohort(Cohort &cohort,int64_t selected_budget,const NativeCurveRun &run,std::set<std::string> &source_universe,const fs::path &output,std::ostringstream &stress_report,bool &first_stress) {
  const MeasurementIsolation measurement;const auto budget=cohort.task==run.selection_task?selected_budget:run.sanity_budget;
  auto &selected=cohort.points.at(budget),&initial=cohort.points.at(0);const auto fresh_seed=stream_seed(cohort.seed,run.fresh_test_stream);
  const auto testing=make_controlled_test_dataset(cohort.task,run.card.shape,run.card.test_pairs,fresh_seed,run.missing_rate);
  validate_split(testing,run,run.card.test_pairs,source_universe);save_split(cohort.directory/"controlled-selected-testing.pt",testing);
  write_text(cohort.directory/"selected-test-manifest.json","{\"fresh_test_seed\":"+quote(std::to_string(fresh_seed))+",\"selection\":\"../selection.json\",\"checkpoint_budget\":"+std::to_string(budget)+",\"testing\":"+split_manifest(testing,"testing")+'}');
  const auto oracle=raw_oracle(cohort.task,testing.observed);const auto oracle_score=score(oracle.predictions,testing.labels,oracle.valid);
  require(oracle_score.supported && oracle_score.accuracy>=.95,"fresh testing legal raw oracle failed task solvability gate");
  const auto native=global_surface(selected.snapshot,testing.observed,run),untrained=global_surface(initial.snapshot,testing.observed,run);
  require(native.provenance==selected.training.provenance && untrained.provenance==initial.training.provenance,"TEST native lineage differs from retained checkpoint");
  std::vector<TestMethod> methods;
  for(const auto &control:cohort.controls)methods.push_back({control.name,control.label,control.width,control.name=="mask_metadata"?mask_surface(testing.observed):raw_surface(*cohort.raw_scaler,testing.observed),&control.results,cohort.directory/"controls"/control.name});
  methods.push_back({"native",run.model_tag,run.export_width,native,&selected.results,selected.directory});
  methods.push_back({"untrained_native",run.model_tag+" — untrained weights",run.export_width,untrained,&initial.results,initial.directory});
  for(const auto &method:methods)save_surface(cohort.directory/(method.name+"-selected-testing.pt"),method.surface,testing);
  const auto reconstructed=reconstruction(selected.directory/"selected-test-reconstruction.pt",testing,selected.snapshot,run);
  std::ostringstream out;out << "{\"development_seed\":" << quote(std::to_string(cohort.seed)) << ",\"task\":" << quote(task_name(cohort.task)) << ",\"fresh_test_seed\":" << quote(std::to_string(fresh_seed)) << ",\"checkpoint_budget\":" << budget
      << ",\"checkpoint\":" << quote(fs::relative(selected.directory/"checkpoint.pt",output).generic_string()) << ",\"fitted_readouts_reused\":true,\"source_manifest\":" << quote(fs::relative(cohort.directory/"selected-test-manifest.json",output).generic_string())
      << ",\"raw_oracle\":" << score_json(oracle_score) << ",\"progress\":" << progress_json(selected.progress) << ",\"reconstruction\":" << reconstructed << ",\"repetitions\":[";
  for(size_t rep=0;rep<run.repetitions.size();++rep) {
    if(rep)out << ',';
    const auto rep_directory=cohort.directory/("testing-"+run.repetitions[rep].id);require(fs::create_directory(rep_directory),"testing repetition output exists");
    std::map<std::string,StressPredictions> predictions;StressReadouts frozen;std::map<std::string,SurfaceDescription> descriptions;
    out << "{\"id\":" << quote(run.repetitions[rep].id) << ",\"methods\":[";
    for(size_t index=0;index<methods.size();++index) {
      if(index)out << ',';
      const auto &method=methods[index];const auto &result=method.results->at(rep);
      out << "{\"method\":" << quote(method.name) << ",\"label\":" << quote(method.label) << ",\"size\":" << method.width << ",\"status\":" << quote(result.status) << ",\"reason\":" << quote(result.reason)
          << ",\"actual_probe_seed\":" << quote(std::to_string(stream_seed(run.repetitions[rep].probe_seed,uint64_t(method.width)))) << ",\"fit_artifact\":" << quote(fs::relative(method.fit_directory/run.repetitions[rep].id/"fit.pt",output).generic_string())
          << ",\"ridge_parameters\":" << 2*method.width+2 << ",\"neural_parameters\":" << run.tiny_hidden*(method.width+3)+2 << ",\"population\":" << population_json(method.surface.valid,testing);
      FrozenStressReadout readout;readout.status=result.status;readout.reason=result.reason;readout.probe_dimensions=method.width;readout.native_valid=method.surface.valid.clone();
      if(result.readout) {
        same_state(*result.readout,result.witness);auto prediction=result.readout->predict(method.surface);
        save_predictions(rep_directory/(method.name+"-predictions.pt"),prediction,testing);predictions.emplace(method.name,prediction);
        const auto interval_seed=stream_seed(cohort.seed,named_stream(method.name+"/"+run.repetitions[rep].id));
        out << ",\"ridge\":" << score_json(score(prediction.ridge,testing.labels,prediction.valid)) << ",\"tiny_secondary\":" << score_json(score(prediction.tiny,testing.labels,prediction.valid))
            << ",\"ridge_grouped_interval\":" << interval_json(grouped_accuracy_interval(prediction.ridge,testing.labels,prediction.valid,testing.source_ids,interval_seed,run.bootstrap_replicates),run.bootstrap_replicates)
            << ",\"tiny_grouped_interval\":" << interval_json(grouped_accuracy_interval(prediction.tiny,testing.labels,prediction.valid,testing.source_ids,interval_seed,run.bootstrap_replicates),run.bootstrap_replicates);
        readout.native_valid=prediction.valid.clone();readout.native_ridge=prediction.ridge.clone();readout.native_tiny=prediction.tiny.clone();
        const auto *fit=result.readout.get();const auto witness_tensors=result.witness;
        readout.predict=[fit,witness_tensors,prediction,first=true](const FeatureSurface &surface) mutable {
          same_state(*fit,witness_tensors);auto actual=fit->predict(surface);same_state(*fit,witness_tensors);
          if(first) {require(torch::equal(actual.valid,prediction.valid) && torch::equal(actual.ridge,prediction.ridge) && torch::equal(actual.tiny,prediction.tiny),"intact stress predictions differ from exact ordinary frozen readout");first=false;}
          return actual;
        };
      }
      frozen.emplace(std::make_pair(method.name,DimensionTier::native),std::move(readout));
      descriptions.emplace(method.name,SurfaceDescription{method.name=="native" || method.name=="untrained_native"?SurfaceKind::global:SurfaceKind::control,method.name=="mask_metadata"?"all rows including allmissing; mask metadata only":"any observation; native provider may abstain on a subset",{}});
      out << '}';
    }
    out << "],\"pairs\":[";
    for(size_t i=0;i<run.card.comparisons.size();++i){if(i)out << ',';out << paired_json(run.card.comparisons[i],predictions,testing,run,stream_seed(cohort.seed,named_stream(run.repetitions[rep].id)));}
    out << "]}";
    if(run.stress_sweep) {
      const auto extractor=[&cohort,&selected,&initial,&run](const Batch &batch) {
        FeatureMap values{{"native",global_surface(selected.snapshot,batch,run)},{"untrained_native",global_surface(initial.snapshot,batch,run)}};
        values.emplace("raw",raw_surface(*cohort.raw_scaler,batch));values.emplace("pca_only",raw_surface(*cohort.raw_scaler,batch));values.emplace("mask_metadata",mask_surface(batch));return values;
      };
      const ControlledProtocol protocol{cohort.task,cohort.seed,run.card.shape,cohort.development.training,cohort.development.validation,testing};
      const auto stress=run_fixed_readout_stress(rep_directory.string(),run.card,protocol,descriptions,frozen,extractor);
      if(!first_stress)stress_report << ',';
      first_stress=false;stress_report << "{\"seed\":" << quote(std::to_string(cohort.seed)) << ",\"task\":" << quote(task_name(cohort.task)) << ",\"repetition\":" << quote(run.repetitions[rep].id) << ",\"report\":" << stress << '}';
    }
    verify_controls(cohort);for(const auto &result:selected.results)verify_result(result,selected.validation);for(const auto &result:initial.results)verify_result(result,initial.validation);
  }
  return out.str()+"]}";
}
} // namespace

NativeCurveRun::NativeCurveRun() {
  card.id="native-curve-v1";card.version=2;card.policy_version="1.2";card.stage="development";card.seeds={3101,3202,3303};card.train_pairs=128;card.validation_pairs=64;card.test_pairs=64;card.threads=1;
  card.comparisons={{"native_vs_pca_only","native","pca_only",DimensionTier::native},{"native_vs_raw","native","raw",DimensionTier::native},{"native_vs_untrained","native","untrained_native",DimensionTier::native}};
}
void run_native_curve(const NativeCurveRun &run,const NamedCurveFactory &factory) {
  validate_run(run,factory);const ThreadsIsolation threads;const MeasurementIsolation ambient_rng;torch::set_num_threads(run.card.threads);
  const fs::path output(run.output_directory);require(!output.empty() && fs::create_directories(output),"output must be exclusively claimed as a new directory");
  write_text(output/native_card_filename(run),card_json(run,factory));
  if(run.stress_sweep)write_text(output/"stress-card.json",fixed_readout_stress_card_json(run.card));
  std::vector<Cohort> cohorts;cohorts.reserve(run.card.seeds.size()*run.card.tasks.size());std::set<std::string> source_universe;
  std::ostringstream validation;validation << "{\"protocol\":" << quote(native_protocol(run)) << ",\"policy_version\":\"1.2\",\"stage\":\"development\",\"card\":" << quote(native_card_filename(run)) << ",\"source_fingerprint\":" << quote(run.source_fingerprint) << ",\"points\":[";
  bool first_point=true;
  for(const auto seed:run.card.seeds)for(const auto task:run.card.tasks) {
    Cohort cohort;cohort.seed=seed;cohort.task=task;
    cohort.development=make_controlled_development_protocol(task,run.card.shape,run.card.train_pairs,run.card.validation_pairs,seed,run.missing_rate);
    require(!cohort.development.testing.observed.data.defined() && cohort.development.testing.source_ids.empty(),"development generator produced testing observations");
    validate_split(cohort.development.training,run,run.card.train_pairs,source_universe);validate_split(cohort.development.validation,run,run.card.validation_pairs,source_universe);
    cohort.directory=output/("seed-"+std::to_string(seed)+"-"+task_name(task));require(fs::create_directory(cohort.directory),"cohort output exists");
    save_split(cohort.directory/"controlled-training.pt",cohort.development.training);save_split(cohort.directory/"controlled-validation.pt",cohort.development.validation);
    write_text(cohort.directory/"development-manifest.json","{\"testing_generated\":false,\"training\":"+split_manifest(cohort.development.training,"training")+",\"validation\":"+split_manifest(cohort.development.validation,"validation")+'}');
    fit_controls(cohort,run);
    const ProviderFitInput input{legal_clone(cohort.development.training.observed),run.card.shape,seed,cohort.development.training.source_ids,run.card.channel_ids,run.card.feature_units,std::string(native_protocol(run))+"/"+task_name(task),run.card.sampling_interval,(run.card.shape.history_length-1)*run.card.sampling_interval};
    const auto trainer=factory.factory(input);require(bool(trainer.train_to) && bool(trainer.save_checkpoint) && bool(trainer.snapshot),"continuous trainer callbacks incomplete");
    write_text(cohort.directory/"trainer-audit.json",fields(trainer.audit_fields));
    const auto budgets=run.development_only || task==run.selection_task?run.milestones:std::vector<int64_t>{0,run.sanity_budget};
    for(const auto budget:budgets) {
      Point point;point.directory=cohort.directory/("milestone-"+std::to_string(budget));require(fs::create_directory(point.directory),"checkpoint directory exists");
      point.progress=trainer.train_to(budget);const MeasurementIsolation measurement;
      check_progress(point.progress,budget,cohort.points.empty()?nullptr:&cohort.points.rbegin()->second.progress);
      if(budget>0)require(!point.progress.losses.empty() && point.progress.losses.back().completed==budget,"trainer trace omitted queried milestone endpoint");
      const auto checkpoint=point.directory/"checkpoint.pt";trainer.save_checkpoint(checkpoint.string());require(fs::is_regular_file(checkpoint),"ordinary resumable checkpoint missing");
      point.snapshot=trainer.snapshot(checkpoint.string());require(bool(point.snapshot.reconstruct),"checkpoint reconstruction callback missing");
      point.training=global_surface(point.snapshot,cohort.development.training.observed,run);point.validation=global_surface(point.snapshot,cohort.development.validation.observed,run);
      require(point.training.provenance==point.validation.provenance && point.training.values.scalar_type()==point.validation.values.scalar_type(),"TRAIN/VALIDATION native lineage/dtype differs");
      if(!cohort.points.empty()) {
        const auto &prior=cohort.points.begin()->second;
        require(torch::equal(point.validation.valid,prior.validation.valid),"native VALIDATION support changed across milestones");
        require(point.training.values.scalar_type()==prior.training.values.scalar_type(),"native dtype changed across milestones");
        std::vector<SurfaceDescription> before,after;
        for(const auto &[name,description]:prior.snapshot.features.surfaces)if(description.kind==SurfaceKind::global)before.push_back(description);
        for(const auto &[name,description]:point.snapshot.features.surfaces)if(description.kind==SurfaceKind::global)after.push_back(description);
        require(before.size()==1 && after.size()==1 && before[0].support_rule==after[0].support_rule,"native support semantics changed across milestones");
      }
      save_surface(point.directory/"native-training.pt",point.training,cohort.development.training);save_surface(point.directory/"native-validation.pt",point.validation,cohort.development.validation);
      const auto assets=point.directory/"provider-assets";require(fs::create_directory(assets),"provider assets exists");
      if(point.snapshot.features.save_assets)point.snapshot.features.save_assets(assets.string());
      write_text(assets/"provider-audit.json","{\"provenance\":"+quote(point.snapshot.features.provenance)+",\"audit_fields\":"+fields(point.snapshot.features.audit_fields)+'}');
      point.results=fit_readouts(point.training,point.validation,cohort.development,run,run.export_width,false,point.directory);
      const auto training_reconstruction=reconstruction(point.directory/"training-reconstruction.pt",cohort.development.training,point.snapshot,run);
      const auto validation_reconstruction=reconstruction(point.directory/"validation-reconstruction.pt",cohort.development.validation,point.snapshot,run);
      std::ostringstream entry;entry << "{\"seed\":" << quote(std::to_string(seed)) << ",\"task\":" << quote(task_name(task)) << ",\"milestone\":" << budget << ",\"model_tag\":" << quote(run.model_tag)
          << ",\"directory\":" << quote(fs::relative(point.directory,output).generic_string()) << ",\"progress\":" << progress_json(point.progress)
          << ",\"training_reconstruction\":" << training_reconstruction << ",\"validation_reconstruction\":" << validation_reconstruction
          << ",\"native_repetitions\":" << results_json(point.results,point.training,point.validation,cohort.development,run,run.export_width) << '}';
      write_text(point.directory/"point.json",entry.str());
      if(!first_point)validation << ',';
      first_point=false;validation << entry.str();
      std::cout << "native-curve master=" << seed << " task=" << task_name(task) << " milestone=" << budget << " completed=" << point.progress.completed << " training_seconds=" << point.progress.training_seconds << '\n' << std::flush;
      cohort.points.emplace(budget,std::move(point));verify_controls(cohort);
    }
    cohorts.push_back(std::move(cohort));
  }
  write_text(output/"validation-report.json",validation.str()+"]}");
  if(run.development_only) {
    size_t checked_points=0;
    for(auto &cohort:cohorts) {
      const MeasurementIsolation measurement;
      verify_controls(cohort);
      for(auto &[budget,point]:cohort.points) {
        (void)budget;
        witness(point,cohort,run);++checked_points;
        verify_controls(cohort);
      }
    }
    durable_selection(output/"development-complete.json","{\"protocol\":\"native-development-v1\",\"policy_version\":\"1.2\",\"stage\":\"development\",\"card\":\"native-development-card.json\",\"validation_report\":\"validation-report.json\",\"all_declared_milestones_complete\":true,\"all_retained_witnesses_exact\":true,\"controls_immutable\":true,\"selection_performed\":false,\"testing_generated\":false,\"stress_evaluated\":false,\"cohorts\":"+
        std::to_string(cohorts.size())+",\"checked_points\":"+std::to_string(checked_points)+",\"milestones\":"+numbers(run.milestones)+",\"durability\":\"close and fsync file plus parent directory after all retained witness checks\",\"source_fingerprint\":"+quote(run.source_fingerprint)+'}');
    return;
  }
  std::optional<int64_t> selected;double best=-std::numeric_limits<double>::infinity();
  std::ostringstream candidates;candidates << '[';bool first_candidate=true;
  for(const auto budget:run.milestones) {
    if(!budget)continue;
    bool supported=true;double total=0;int64_t count=0;std::ostringstream entries;entries << '[';bool first_entry=true;
    for(const auto &cohort:cohorts)if(cohort.task==run.selection_task) {
      const auto &point=cohort.points.at(budget);const auto &result=point.results.front();const bool both_classes=fit_supported(point.validation,cohort.development.validation.labels);
      const bool measured=bool(result.readout) && both_classes;const auto value=measured?score(result.validation.ridge,cohort.development.validation.labels,result.validation.valid):Score{};
      supported=supported && measured && value.supported;
      if(measured && value.supported){total+=value.accuracy;++count;}
      if(!first_entry)entries << ',';
      first_entry=false;entries << std::setprecision(17) << "{\"seed\":" << quote(std::to_string(cohort.seed)) << ",\"status\":" << quote(measured?"measured":"unsupported_fit_or_population") << ",\"population\":" << population_json(point.validation.valid,cohort.development.validation);
      if(measured)entries << ",\"ridge\":" << score_json(value);
      entries << '}';
    }
    supported=supported && count==int64_t(run.card.seeds.size());const auto utility=supported?total/count:0;
    if(supported && utility>best){selected=budget;best=utility;}
    if(!first_candidate)candidates << ',';
    first_candidate=false;candidates << std::setprecision(17) << "{\"budget\":" << budget << ",\"status\":" << quote(supported?"measured":"unsupported_common_budget") << ",\"utility\":";
    if(supported)candidates << utility;else candidates << "null";
    candidates << ",\"entries\":" << entries.str()+']' << '}';
  }
  require(selected.has_value(),"no positive budget has native ridge fits and both validation classes for every master");
  std::ostringstream selection;selection << std::setprecision(17) << "{\"protocol\":\"native-curve-v1\",\"policy_version\":\"1.2\",\"stage\":\"development\",\"selection_frozen_before_testing\":true,\"durability\":\"close and fsync file plus parent directory before any TEST generation\",\"selected_budget\":" << *selected << ",\"utility\":" << best
      << ",\"metric\":\"mean native global ridge VALIDATION accuracy over every master, first declared repetition; native support identical across milestones\",\"task\":" << quote(task_name(run.selection_task))
      << ",\"tie_rule\":\"smallest exact-tied positive budget\",\"sanity_budget\":" << run.sanity_budget << ",\"candidates\":" << candidates.str()+']' << ",\"source_fingerprint\":" << quote(run.source_fingerprint) << '}';
  durable_selection(output/"selection.json",selection.str());
  // All retained witness checks precede all TEST generation, including sanities.
  for(auto &cohort:cohorts) {
    const MeasurementIsolation measurement;verify_controls(cohort);
    witness(cohort.points.at(0),cohort,run);witness(cohort.points.at(cohort.task==run.selection_task?*selected:run.sanity_budget),cohort,run);
  }
  std::ostringstream report,stress_report;report << "{\"protocol\":\"native-curve-v1\",\"policy_version\":\"1.2\",\"stage\":\"development\",\"card\":\"native-curve-card.json\",\"validation_report\":\"validation-report.json\",\"selection\":\"selection.json\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty) << ",\"selected_budget\":" << *selected << ",\"interpretation\":\"fresh synthetic development; three head repetitions are not independent encoder runs; no across-master CI or consumer acceptance\",\"testing_runs\":[";
  stress_report << "{\"protocol\":\"fixed-readout-stress-v1\",\"policy_version\":\"1.2\",\"stage\":\"development\",\"card\":\"stress-card.json\",\"runs\":[";
  bool first_test=true,first_stress=true;
  for(auto &cohort:cohorts) {
    if(!first_test)report << ',';
    first_test=false;report << test_cohort(cohort,*selected,run,source_universe,output,stress_report,first_stress);
  }
  write_text(output/"report.json",report.str()+"]}");if(run.stress_sweep)write_text(output/"stress-report.json",stress_report.str()+"]}");
}
} // namespace embedding::evaluation
