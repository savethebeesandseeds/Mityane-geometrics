// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replay_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
constexpr const char *kind = "rpb_context_replay_continuation_audit_v1";
constexpr const char *counter_policy = "splitmix64-counter-rows-masks-torch-attempt-v1";
constexpr const char *sampling_policy = "with_replacement_counter_rows;sampled_rows_includes_no_update_attempts";
constexpr const char *timing_scope =
    "CUDA-synchronized cumulative existing training-loop wall time; includes CPU masks/transfers/bookkeeping; excludes replay audit, checkpoint IO and readouts; GPU-kernel-only time unmeasured";

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[RPB context replay] " + message);
}
std::string key(const std::string &path) {
  require(!path.empty(), "empty artifact path");
  return fs::weakly_canonical(fs::absolute(path)).string();
}
std::string decimal(double value) {
  std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
std::string fields_json(const std::map<std::string, std::string> &fields) {
  std::ostringstream out; out << '{'; bool first = true;
  for (const auto &[name, value] : fields) {
    if (!first) out << ',';
    first = false; out << quote(name) << ':' << quote(value);
  }
  return out.str() + '}';
}
std::string checksum(const std::string &path) {
  std::ifstream input(path, std::ios::binary); require(bool(input), "cannot read protected input: " + path);
  uint64_t hash = 14695981039346656037ULL; char buffer[65536];
  while (input) {
    input.read(buffer, sizeof(buffer));
    for (std::streamsize i = 0; i < input.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]); hash *= 1099511628211ULL;
    }
  }
  require(input.eof(), "failed reading protected input: " + path);
  std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << hash; return out.str();
}
void text(torch::serialize::OutputArchive &archive, const std::string &name, const std::string &value) {
  archive.write(name, embedding::archive::text_tensor(value), true);
}
std::string text(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true); return embedding::archive::tensor_text(value);
}
int64_t integer(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true);
  require(value.scalar_type() == torch::kInt64 && value.numel() == 1, "invalid typed counter: " + name);
  return value.item<int64_t>();
}
double floating(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true);
  require(value.scalar_type() == torch::kFloat64 && value.numel() == 1 &&
      torch::isfinite(value).all().item<bool>(), "invalid typed scalar: " + name);
  return value.item<double>();
}
struct RuntimeIsolation {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try {
      for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]);
      at::set_num_threads(threads);
    } catch (...) { std::terminate(); }
  }
};
std::string source_manifest(const std::vector<std::string> &ids) {
  std::string out;
  for (const auto &id : ids) out += std::to_string(id.size()) + ':' + id;
  return out;
}
struct PointAudit {
  std::map<std::string, std::string> fields;
  int64_t attempted{0}, completed{0}, sampled{0};
  std::array<int64_t, 3> context{0, 0, 0};
  torch::Tensor channel_order;
  double interval{0}, endpoint{0}, training_seconds{0};
};
PointAudit read_point_audit(const std::string &path) {
  torch::serialize::InputArchive archive; archive.load_from(path + ".audit.pt", torch::kCPU);
  require(text(archive, "artifact_kind") == "rpb_learning_curve_training_audit_v1", "unknown parent learning audit kind");
  PointAudit result;
  for (const std::string name : {"resolved_settings", "protocol_id", "feature_units", "initialization_seed",
      "actual_training_seed", "fit_source_manifest", "fit_source_manifest_id", "rng_policy", "sampling_policy",
      "optimizer_policy", "training_device", "training_producer_source_fingerprint", "core_writer_source_fingerprint",
      "training_dataset_id", "scaler_fit_dataset_id", "preprocessing_id", "model_tag", "training_policy_id",
      "context_deletion_ratio", "context_deletion_stream", "context_deletion_rng_policy",
      "context_deletion_repair_policy", "context_deletion_visibility_policy", "context_deletion_count_policy",
      "context_deletion_resume_policy"}) result.fields.emplace(name, text(archive, name));
  result.attempted = integer(archive, "attempted_steps"); result.completed = integer(archive, "completed_steps");
  result.sampled = integer(archive, "sampled_rows");
  result.context = {integer(archive, "context_requested_deleted_coordinates"),
      integer(archive, "context_actual_deleted_coordinates"), integer(archive, "context_restored_coordinates")};
  archive.read("channel_order", result.channel_order, true);
  require(result.channel_order.scalar_type() == torch::kInt64 && result.channel_order.dim() == 1,
      "invalid physical channel order witness");
  result.interval = floating(archive, "sampling_interval"); result.endpoint = floating(archive, "endpoint");
  result.training_seconds = floating(archive, "training_seconds");
  require(result.training_seconds >= 0 && result.attempted >= result.completed && result.completed >= 0 &&
      result.sampled >= 0 && result.context[0] >= 0 && result.context[1] >= 0 && result.context[2] >= 0 &&
      result.context[0] >= result.context[1] && result.context[0] - result.context[1] == result.context[2],
      "invalid cumulative parent counters");
  require(result.fields.at("model_tag") == "RPB-v6" && result.fields.at("training_policy_id") == context_deletion::policy_id &&
      result.fields.at("context_deletion_ratio") == "0.30" &&
      result.fields.at("context_deletion_stream") == "0x6374782d64726f70" &&
      result.fields.at("context_deletion_rng_policy") == context_deletion::rng_policy &&
      result.fields.at("context_deletion_repair_policy") == context_deletion::repair_policy &&
      result.fields.at("context_deletion_visibility_policy") == context_deletion::visibility_policy &&
      result.fields.at("context_deletion_count_policy") ==
          "cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only" &&
      result.fields.at("context_deletion_resume_policy") ==
          "fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API" &&
      floating(archive, "context_deletion_ratio_value") == context_deletion::ratio &&
      integer(archive, "context_deletion_stream_value") == static_cast<int64_t>(context_deletion::stream),
      "parent context policy differs from the fixed recipe");
  return result;
}
Input raw_input(const embedding::Batch &batch, const Config &config, const ev::ProviderFitInput &fit) {
  Input input{batch.data, batch.feature_mask, torch::tensor(fit.channel_ids, torch::kInt64),
      torch::full({batch.data.size(0)}, fit.endpoint, torch::kFloat64), fit.sampling_interval};
  validate_input(input, config); return input;
}
bool same_input(const Input &left, const Input &right) {
  return torch::equal(left.observed, right.observed) &&
      torch::equal(torch::where(left.observed, left.data, torch::zeros_like(left.data)),
                   torch::where(right.observed, right.data, torch::zeros_like(right.data))) &&
      torch::equal(left.channel_ids, right.channel_ids) && torch::equal(left.endpoints, right.endpoints) &&
      left.sampling_interval == right.sampling_interval && left.data.scalar_type() == right.data.scalar_type();
}
bool same_scaler(const FrozenScaler &left, const FrozenScaler &right) {
  return left.identity() == right.identity() && left.scale_floor == right.scale_floor &&
      torch::equal(left.mean, right.mean) && torch::equal(left.scale, right.scale) &&
      torch::equal(left.count, right.count) && torch::equal(left.channel_ids, right.channel_ids) &&
      torch::equal(left.floor_applied, right.floor_applied);
}
struct AdamState { int64_t step{0}; torch::Tensor first, second; };
using AdamStates = std::map<std::string, AdamState>;
AdamStates optimizer_states(Checkpoint &checkpoint, const std::string &path) {
  torch::optim::AdamW optimizer(checkpoint.model->parameters(),
      torch::optim::AdamWOptions(checkpoint.settings.learning_rate).weight_decay(checkpoint.settings.weight_decay));
  load_optimizer(path, optimizer, checkpoint.settings.model.device);
  AdamStates result;
  for (const auto &parameter : checkpoint.model->named_parameters()) {
    require(parameter.value().is_cuda(), "optimizer witness model is not CUDA");
    const auto found = optimizer.state().find(parameter.value().unsafeGetTensorImpl());
    if (found == optimizer.state().end()) continue;
    const auto *state = dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
    require(state && state->step() == checkpoint.completed_steps && state->exp_avg().is_cuda() &&
        state->exp_avg_sq().is_cuda() && state->exp_avg().sizes() == parameter.value().sizes() &&
        state->exp_avg_sq().sizes() == parameter.value().sizes() &&
        state->exp_avg().scalar_type() == parameter.value().scalar_type() &&
        state->exp_avg_sq().scalar_type() == parameter.value().scalar_type() &&
        torch::isfinite(state->exp_avg()).all().item<bool>() && torch::isfinite(state->exp_avg_sq()).all().item<bool>(),
        "invalid named AdamW CUDA step/device/moments: " + parameter.key());
    result.emplace(parameter.key(), AdamState{state->step(), state->exp_avg().detach().to(torch::kCPU).clone(),
        state->exp_avg_sq().detach().to(torch::kCPU).clone()});
  }
  require(result.size() == optimizer.state().size() && (checkpoint.completed_steps == 0 || !result.empty()),
      "unassociated or missing named AdamW state");
  return result;
}
void equal_model(const Model &left, const Model &right) {
  const auto a = left->named_parameters(), b = right->named_parameters();
  require(a.size() == b.size(), "replay named parameter set differs");
  for (const auto &parameter : a)
    require(b.contains(parameter.key()) && parameter.value().scalar_type() == b[parameter.key()].scalar_type() &&
        torch::equal(parameter.value(), b[parameter.key()]), "replay parameter differs: " + parameter.key());
  const auto x = left->named_buffers(), y = right->named_buffers();
  require(x.size() == y.size(), "replay named buffer set differs");
  for (const auto &buffer : x)
    require(y.contains(buffer.key()) && buffer.value().scalar_type() == y[buffer.key()].scalar_type() &&
        torch::equal(buffer.value(), y[buffer.key()]), "replay buffer differs: " + buffer.key());
}
void equal_optimizer(const AdamStates &left, const AdamStates &right) {
  require(left.size() == right.size(), "replay active optimizer state set differs");
  for (const auto &[name, state] : left) {
    const auto found = right.find(name);
    require(found != right.end() && state.step == found->second.step &&
        torch::equal(state.first, found->second.first) && torch::equal(state.second, found->second.second),
        "replay AdamW state differs: " + name);
  }
}
void write_model(torch::serialize::OutputArchive &archive, const std::string &name, const Model &model) {
  torch::serialize::OutputArchive out;
  const auto parameters = model->named_parameters(), buffers = model->named_buffers();
  out.write("parameter_count", torch::tensor(static_cast<int64_t>(parameters.size())), true);
  out.write("buffer_count", torch::tensor(static_cast<int64_t>(buffers.size())), true);
  int64_t i = 0;
  for (const auto &parameter : parameters) {
    torch::serialize::OutputArchive item; text(item, "parameter_name", parameter.key());
    item.write("value", parameter.value().detach().to(torch::kCPU).clone(), true);
    out.write("parameter_" + std::to_string(i++), item);
  }
  i = 0;
  for (const auto &buffer : buffers) {
    torch::serialize::OutputArchive item; text(item, "buffer_name", buffer.key());
    item.write("value", buffer.value().detach().to(torch::kCPU).clone(), true);
    out.write("buffer_" + std::to_string(i++), item);
  }
  archive.write(name, out);
}
void write_optimizer(torch::serialize::OutputArchive &archive, const std::string &name, const AdamStates &states) {
  torch::serialize::OutputArchive out;
  out.write("parameter_count", torch::tensor(static_cast<int64_t>(states.size())), true);
  int64_t i = 0;
  for (const auto &[parameter, state] : states) {
    torch::serialize::OutputArchive item; text(item, "parameter_name", parameter);
    item.write("step", torch::tensor(state.step), true);
    item.write("exp_avg", state.first, true); item.write("exp_avg_sq", state.second, true);
    out.write("parameter_" + std::to_string(i++), item);
  }
  archive.write(name, out);
}
void write_trace(torch::serialize::OutputArchive &archive, const ev::CurveProgress &progress) {
  auto counters = torch::empty({static_cast<int64_t>(progress.losses.size()), 3}, torch::kInt64);
  auto values = torch::empty({static_cast<int64_t>(progress.losses.size()), 2}, torch::kFloat64);
  auto a = counters.accessor<int64_t, 2>(); auto b = values.accessor<double, 2>();
  for (size_t i = 0; i < progress.losses.size(); ++i) {
    const auto &point = progress.losses[i];
    a[i][0] = point.attempted; a[i][1] = point.completed; a[i][2] = point.target_cells;
    b[i][0] = point.loss; b[i][1] = point.gradient_norm;
  }
  archive.write("loss_trace_counters", counters, true); archive.write("loss_trace_values", values, true);
}
struct ReplayState {
  ContextReplayOptions options;
  Settings settings;
  PointAudit parent_audit;
  ev::CurveTrainer live;
  ev::CurveProgress progress;
  bool passed{false}, failed{false};
  double replay_seconds{0}, audit_seconds{0};
  std::map<std::string, std::string> inputs, fields;
  std::map<std::string, std::map<std::string, std::string>> saved_fields;
};
double replay_cost(const ReplayState &state) {
  return state.passed ? state.replay_seconds : state.progress.training_seconds;
}
double continuation_cost(const ReplayState &state) {
  return state.passed ? std::max(0.0, state.progress.training_seconds - state.replay_seconds) : 0.0;
}
void inputs_unchanged(const ReplayState &state) {
  for (const auto &[path, expected] : state.inputs)
    require(checksum(path) == expected, "protected original input changed: " + path);
}
void new_outputs(const ReplayState &state, const std::string &path) {
  std::set<std::string> paths;
  for (const auto &output : {path, path + ".training-raw.pt", path + ".scaler.pt", path + ".audit.pt",
                           path + ".replay-audit.pt", path + ".replay-audit.json"}) {
    require(paths.insert(key(output)).second && !state.inputs.count(key(output)), "output aliases a protected input");
    require(!fs::exists(output) && !fs::is_symlink(fs::symlink_status(output)), "refusing an existing output: " + output);
    require(fs::is_directory(fs::path(output).parent_path()), "output parent directory must already exist");
  }
}
std::map<std::string, std::string> point_fields(const ReplayState &state, const std::string &path) {
  auto fields = state.fields;
  fields.emplace("checkpoint_path", key(path));
  fields.emplace("replay_gate_passed", state.passed ? "true" : "false");
  fields.emplace("replay_training_seconds", decimal(replay_cost(state)));
  fields.emplace("continuation_training_seconds", decimal(continuation_cost(state)));
  fields.emplace("total_fresh_training_seconds", decimal(state.progress.training_seconds));
  fields.emplace("gate_audit_wall_seconds", decimal(state.audit_seconds));
  return fields;
}
void write_sidecar(ReplayState &state, const std::string &path, Checkpoint &point, const PointAudit &audit,
                   const AdamStates &optimizer, Checkpoint *parent = nullptr, const AdamStates *parent_optimizer = nullptr) {
  auto fields = point_fields(state, path);
  fields.emplace("attempted_steps", std::to_string(point.attempted_steps));
  fields.emplace("completed_steps", std::to_string(point.completed_steps));
  fields.emplace("sampled_rows", std::to_string(audit.sampled));
  for (size_t i = 0; i < 3; ++i)
    fields.emplace(std::array<std::string, 3>{"context_requested_deleted_coordinates", "context_actual_deleted_coordinates",
        "context_restored_coordinates"}[i], std::to_string(audit.context[i]));
  torch::serialize::OutputArchive archive; text(archive, "artifact_kind", kind);
  for (const auto &[name, value] : fields) text(archive, name, value);
  archive.write("replay_gate_passed_value", torch::tensor(state.passed, torch::kBool), true);
  archive.write("attempted_steps_value", torch::tensor(point.attempted_steps), true);
  archive.write("completed_steps_value", torch::tensor(point.completed_steps), true);
  archive.write("sampled_rows_value", torch::tensor(audit.sampled), true);
  archive.write("context_counts", torch::tensor(std::vector<int64_t>(audit.context.begin(), audit.context.end()), torch::kInt64), true);
  archive.write("parent_attempted_steps", torch::tensor(state.parent_audit.attempted), true);
  archive.write("parent_completed_steps", torch::tensor(state.parent_audit.completed), true);
  archive.write("parent_sampled_rows", torch::tensor(state.parent_audit.sampled), true);
  archive.write("parent_context_counts", torch::tensor(std::vector<int64_t>(state.parent_audit.context.begin(), state.parent_audit.context.end()), torch::kInt64), true);
  archive.write("replay_training_seconds_value", torch::tensor(replay_cost(state), torch::kFloat64), true);
  archive.write("continuation_training_seconds_value", torch::tensor(continuation_cost(state), torch::kFloat64), true);
  archive.write("total_fresh_training_seconds_value", torch::tensor(state.progress.training_seconds, torch::kFloat64), true);
  archive.write("gate_audit_wall_seconds_value", torch::tensor(state.audit_seconds, torch::kFloat64), true);
  archive.write("original_parent_training_seconds", torch::tensor(state.parent_audit.training_seconds, torch::kFloat64), true);
  archive.write("channel_order", audit.channel_order, true);
  archive.write("sampling_interval", torch::tensor(audit.interval, torch::kFloat64), true);
  archive.write("endpoint", torch::tensor(audit.endpoint, torch::kFloat64), true);
  write_trace(archive, state.progress);
  write_model(archive, "model", point.model); write_optimizer(archive, "optimizer", optimizer);
  torch::serialize::OutputArchive scaler; point.scaler.save(scaler); archive.write("scaler", scaler);
  if (parent) {
    write_model(archive, "parent_model", parent->model);
    write_optimizer(archive, "parent_optimizer", *parent_optimizer);
    torch::serialize::OutputArchive original_scaler; parent->scaler.save(original_scaler);
    archive.write("parent_scaler", original_scaler);
  }
  embedding::archive::save_archive(path + ".replay-audit.pt", archive);
  std::ofstream json(path + ".replay-audit.json", std::ios::binary);
  require(bool(json), "cannot create replay JSON sidecar");
  json << "{\"artifact_kind\":" << quote(kind) << ",\"replay_gate_passed\":" << (state.passed ? "true" : "false")
       << ",\"attempted\":" << point.attempted_steps << ",\"completed\":" << point.completed_steps
       << ",\"sampled_rows\":" << audit.sampled << ",\"context_counts\":[" << audit.context[0] << ',' << audit.context[1] << ',' << audit.context[2]
       << "],\"replay_training_seconds\":" << decimal(replay_cost(state))
       << ",\"continuation_training_seconds\":" << decimal(continuation_cost(state))
       << ",\"total_fresh_training_seconds\":" << decimal(state.progress.training_seconds)
       << ",\"gate_audit_wall_seconds\":" << decimal(state.audit_seconds)
       << ",\"gpu_kernel_only_seconds\":null,\"loss_trace_points\":" << state.progress.losses.size()
       << ",\"fields\":" << fields_json(fields) << "}\n";
  json.close(); require(bool(json), "replay JSON sidecar write failed");
  state.saved_fields.emplace(key(path), std::move(fields));
}
void gate(ReplayState &state) {
  inputs_unchanged(state); new_outputs(state, state.options.replay_checkpoint);
  state.live.save_checkpoint(state.options.replay_checkpoint);
  const auto began = std::chrono::steady_clock::now();
  const RuntimeIsolation isolation;
  auto parent = load_checkpoint(state.options.audited_checkpoint, state.settings.model.device);
  auto replay = load_checkpoint(state.options.replay_checkpoint, state.settings.model.device);
  const auto audit = read_point_audit(state.options.replay_checkpoint);
  auto expected_settings = parent.settings; expected_settings.steps = state.settings.steps;
  require(settings_text(expected_settings) == settings_text(replay.settings) &&
      replay.completed_steps == state.options.expected_replay_updates &&
      replay.attempted_steps == parent.attempted_steps && replay.completed_steps == parent.completed_steps &&
      replay.training_policy_id == parent.training_policy_id && replay.dataset_id == parent.dataset_id &&
      replay.schema_id == parent.schema_id && replay.scaler_fit_dataset_id == parent.scaler_fit_dataset_id,
      "replay settings, parent identity, policy or absolute counters differ");
  require(audit.attempted == state.parent_audit.attempted && audit.completed == state.parent_audit.completed &&
      audit.sampled == state.parent_audit.sampled && audit.context == state.parent_audit.context &&
      torch::equal(audit.channel_order, state.parent_audit.channel_order) &&
      audit.interval == state.parent_audit.interval && audit.endpoint == state.parent_audit.endpoint,
      "replay sampled-row/context/physical-metadata witnesses differ");
  for (const auto &[name, value] : state.parent_audit.fields) {
    if (name == "resolved_settings" || name == "training_producer_source_fingerprint" ||
        name == "core_writer_source_fingerprint") continue;
    require(audit.fields.at(name) == value, "replay source/policy/counter metadata differs: " + name);
  }
  require(audit.fields.at("resolved_settings") == settings_text(replay.settings) &&
      audit.fields.at("core_writer_source_fingerprint") == replay.source_fingerprint &&
      audit.fields.at("training_producer_source_fingerprint") == state.live.audit_fields.at("training_producer_source_fingerprint"),
      "replay resolved settings or current writer/producer identity differs");
  equal_model(parent.model, replay.model);
  require(same_scaler(parent.scaler, replay.scaler), "replay frozen scaler buffers or identity differ");
  const auto original_optimizer = optimizer_states(parent, state.options.audited_checkpoint);
  const auto replay_optimizer = optimizer_states(replay, state.options.replay_checkpoint);
  equal_optimizer(original_optimizer, replay_optimizer);
  inputs_unchanged(state);
  state.replay_seconds = state.progress.training_seconds;
  state.audit_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  state.passed = true;
  state.fields.emplace("exact_gate_scope", "all-named-parameters/buffers;active-named-AdamW-steps/moments;complete-frozen-scaler;attempt/completed/sample/E-counts;source-order/seed/policy/physical-metadata");
  write_sidecar(state, state.options.replay_checkpoint, replay, audit, replay_optimizer, &parent, &original_optimizer);
}
} // namespace

ev::CurveTrainerFactory make_context_replay_trainer(const ContextReplayOptions &options) {
  require(options.expected_replay_updates > 0 && options.max_completed_updates > options.expected_replay_updates,
      "expected replay budget must be positive and below the continuation ceiling");
  require(torch::cuda::is_available(), "explicit CUDA is required; no CPU fallback");
  const RuntimeIsolation isolation;
  auto parent = load_checkpoint(options.audited_checkpoint, torch::kCPU);
  const auto audit = read_point_audit(options.audited_checkpoint);
  auto settings = parse_settings(audit.fields.at("resolved_settings"));
  auto cpu_settings = settings; cpu_settings.model.device = torch::kCPU;
  // Canonical settings text records "cuda" without an index. The producer's
  // device witness retains an explicit index when one was used originally.
  settings.model.device = torch::Device(audit.fields.at("training_device"));
  require(settings.model.channel_mixer_placement == 0 && parent.settings.model.channel_mixer_placement == 0,
      "historical context replay rejects early channel mixer placement");
  require(settings_text(cpu_settings) == settings_text(parent.settings) && settings.model.device.is_cuda() &&
      settings.model.global_bottleneck_mode == 2 && settings.model.channel_mixer_layers == 1 && settings.model.export_width == 32 &&
      parent.training_policy_id == context_deletion::policy_id && parent.completed_steps == options.expected_replay_updates &&
      parent.attempted_steps == audit.attempted && parent.completed_steps == audit.completed &&
      parent.attempted_steps <= std::numeric_limits<int64_t>::max() / settings.batch_size &&
      audit.sampled == parent.attempted_steps * settings.batch_size &&
      audit.fields.at("rng_policy") == counter_policy && audit.fields.at("sampling_policy") == sampling_policy &&
      audit.fields.at("optimizer_policy") == "one_continuous_AdamW_state;absolute_completed_update_budgets" &&
      audit.fields.at("training_device") == settings.model.device.str() &&
      audit.fields.at("actual_training_seed") == std::to_string(parent.settings.seed) &&
      audit.fields.at("core_writer_source_fingerprint") == parent.source_fingerprint &&
      !audit.fields.at("training_producer_source_fingerprint").empty() &&
      audit.fields.at("training_dataset_id") == parent.dataset_id &&
      audit.fields.at("scaler_fit_dataset_id") == parent.scaler_fit_dataset_id &&
      audit.fields.at("preprocessing_id") == parent.scaler.identity(),
      "parent checkpoint/audit is not the exact supported context training point");
  settings.steps = options.max_completed_updates;
  require(settings.attempt_limit >= settings.steps, "unchanged attempt ceiling cannot reach the requested completed budget");
  validate_settings(settings);
  auto frozen = std::make_shared<ReplayState>();
  frozen->options = options; frozen->settings = settings; frozen->parent_audit = audit;
  for (const auto &path : {options.audited_checkpoint, options.audited_checkpoint + ".audit.pt",
      options.audited_checkpoint + ".training-raw.pt", options.audited_checkpoint + ".scaler.pt", options.training_archive})
    frozen->inputs.emplace(key(path), checksum(path));
  new_outputs(*frozen, options.replay_checkpoint);
  const auto training = load_dataset(options.training_archive, parent.settings.model);
  const auto original = load_dataset(options.audited_checkpoint + ".training-raw.pt", parent.settings.model);
  const auto archived_scaler = load_scaler(options.audited_checkpoint + ".scaler.pt", parent.settings.model, parent.schema_id);
  require(training.dataset_id == parent.dataset_id && training.schema_id == parent.schema_id &&
      same_input(training.input, original.input) && training.feature_units == original.feature_units &&
      training.feature_units == audit.fields.at("feature_units") && same_scaler(parent.scaler, archived_scaler),
      "declared training archive/scaler companion differs from exact parent TRAIN");
  frozen->fields = {{"adapter_id", "rpb_context_replay_gate_v1"}, {"model_tag", "RPB-v6"},
      {"training_policy_id", context_deletion::policy_id}, {"audited_parent_checkpoint", key(options.audited_checkpoint)},
      {"replay_checkpoint", key(options.replay_checkpoint)}, {"training_archive", key(options.training_archive)},
      {"expected_replay_updates", std::to_string(options.expected_replay_updates)},
      {"max_completed_updates", std::to_string(options.max_completed_updates)},
      {"parent_training_producer_source_fingerprint", audit.fields.at("training_producer_source_fingerprint")},
      {"parent_core_writer_source_fingerprint", parent.source_fingerprint},
      {"replay_orchestrator_source_fingerprint", EVALUATION_SOURCE_ID},
      {"input_checksum_algorithm", "FNV1a64 byte-preservation witness; external frozen SHA256 inventory remains authoritative"},
      {"protected_input_checksums_fnv1a64", fields_json(frozen->inputs)},
      {"timing_scope", timing_scope}, {"gpu_kernel_only_seconds", "unmeasured"},
      {"gate_audit_wall_seconds_scope", "checkpoint loading and exact comparison; excludes CPU sidecar writing"},
      {"continuation_policy", "fresh-deterministic-replay;exact-parent-state-gate;then-same-live-AdamW;no-tagged-resume"},
      {"settings_change_policy", "only absolute steps ceiling; original model/optimizer/seed/attempt-limit/logging settings unchanged"},
      {"source_change_policy", "parent and current writer/producer IDs retained separately; exact numerical state equality required"}};
  const auto base_factory = make_learning_curve_trainer(settings, ContextDeletionOptions{true});
  return [frozen, base_factory, training](const ev::ProviderFitInput &fit) {
    require(fit.shape.dtype == torch::kFloat64 && fit.shape.device.is_cpu() &&
        fit.training_observations.data.defined() && fit.training_observations.data.device().is_cpu() &&
        fit.training_observations.data.scalar_type() == torch::kFloat64 &&
        fit.training_observations.data.dim() == 4 && fit.training_observations.feature_mask.defined() &&
        fit.training_observations.feature_mask.device().is_cpu() &&
        fit.training_source_ids.size() == static_cast<size_t>(fit.training_observations.data.size(0)) &&
        static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL) == frozen->settings.seed &&
        frozen->parent_audit.fields.at("initialization_seed") ==
            std::to_string(training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL)) &&
        frozen->parent_audit.fields.at("fit_source_manifest") == source_manifest(fit.training_source_ids) &&
        frozen->parent_audit.fields.at("protocol_id") == fit.protocol_id &&
        frozen->parent_audit.fields.at("feature_units") == fit.feature_units &&
        frozen->parent_audit.interval == fit.sampling_interval && frozen->parent_audit.endpoint == fit.endpoint &&
        torch::equal(frozen->parent_audit.channel_order, torch::tensor(fit.channel_ids, torch::kInt64)),
        "fit is not the original label-free TRAIN/source/seed/physical-metadata association");
    const auto raw = raw_input(fit.training_observations, frozen->settings.model, fit);
    require(same_input(raw, training.input), "fit values/support/order differ from exact original training archive");
    auto state = std::make_shared<ReplayState>(*frozen);
    inputs_unchanged(*state); new_outputs(*state, state->options.replay_checkpoint);
    state->live = base_factory(fit);
    require(state->live.audit_fields.at("preprocessing_id") == state->parent_audit.fields.at("preprocessing_id") &&
        state->live.audit_fields.at("training_dataset_id") == state->parent_audit.fields.at("training_dataset_id") &&
        state->live.audit_fields.at("fit_source_manifest_id") == state->parent_audit.fields.at("fit_source_manifest_id"),
        "fresh TRAIN fit/scaler/source identity differs from parent");
    state->fields.emplace("current_training_producer_source_fingerprint", state->live.audit_fields.at("training_producer_source_fingerprint"));
    state->fields.emplace("current_core_writer_source_fingerprint", state->live.audit_fields.at("core_writer_source_fingerprint"));
    state->fields.emplace("resolved_replay_settings", state->live.audit_fields.at("resolved_settings"));
    ev::CurveTrainer trainer; trainer.audit_fields = state->live.audit_fields;
    for (const auto &[name, value] : state->fields) trainer.audit_fields["context_replay_" + name] = value;
    trainer.train_to = [state](int64_t budget) {
      require(!state->failed, "a failed replay gate permanently blocks this trainer");
      require(budget >= state->progress.completed && budget <= state->options.max_completed_updates,
          "requested completed budget is nonmonotonic or exceeds the frozen ceiling");
      try {
        inputs_unchanged(*state);
        if (!state->passed && budget >= state->options.expected_replay_updates) {
          state->progress = state->live.train_to(state->options.expected_replay_updates);
          gate(*state);
        }
        if (budget != state->progress.completed) state->progress = state->live.train_to(budget);
        else if (budget == 0) state->progress = state->live.train_to(0);
        inputs_unchanged(*state);
        return state->progress;
      } catch (...) { state->failed = true; throw; }
    };
    trainer.save_checkpoint = [state](const std::string &path) {
      require(!state->failed, "a failed replay gate cannot save further points"); inputs_unchanged(*state);
      if (key(path) == key(state->options.replay_checkpoint)) {
        require(state->passed && state->progress.completed == state->options.expected_replay_updates,
            "reserved replay path is only the exact gate checkpoint");
        return;
      }
      new_outputs(*state, path); state->live.save_checkpoint(path);
      const RuntimeIsolation isolation;
      auto point = load_checkpoint(path, state->settings.model.device);
      const auto audit = read_point_audit(path); const auto optimizer = optimizer_states(point, path);
      write_sidecar(*state, path, point, audit, optimizer); inputs_unchanged(*state);
    };
    trainer.snapshot = [state](const std::string &path) {
      require(!state->failed, "a failed replay gate cannot serve further snapshots"); inputs_unchanged(*state);
      const auto saved = state->saved_fields.find(key(path));
      require(saved != state->saved_fields.end(), "snapshot requires this wrapper's saved replay-audited point");
      const RuntimeIsolation isolation;
      auto snapshot = state->live.snapshot(path); const auto fields = saved->second;
      for (const auto &[name, value] : fields) snapshot.features.audit_fields["context_replay_" + name] = value;
      snapshot.features.provenance += "; fresh context replay; exact-state gate=" + fields.at("replay_gate_passed") +
          "; audited_parent=" + fields.at("audited_parent_checkpoint");
      const auto original_save = snapshot.features.save_assets;
      snapshot.features.save_assets = [original_save, fields](const std::string &directory) {
        const auto destination = (fs::path(directory) / "context-replay-snapshot-audit.pt").string();
        require(!fs::exists(destination), "refusing existing replay snapshot audit");
        if (original_save) original_save(directory);
        torch::serialize::OutputArchive out; text(out, "artifact_kind", "rpb_context_replay_snapshot_audit_v1");
        for (const auto &[name, value] : fields) text(out, name, value);
        embedding::archive::save_archive(destination, out);
      };
      return snapshot;
    };
    return trainer;
  };
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
