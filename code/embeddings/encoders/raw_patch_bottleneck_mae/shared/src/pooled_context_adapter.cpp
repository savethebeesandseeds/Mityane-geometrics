// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/pooled_context_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

#ifndef POOLED_CONTEXT_ADAPTER_SOURCE_ID
#define POOLED_CONTEXT_ADAPTER_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using Named = std::map<std::string, torch::Tensor>;
constexpr const char *policy = "rpb-training-context-deletion-015-v1";
constexpr const char *counter_policy = "splitmix64-counter-rows-masks-torch-attempt-v1";
bool curve_scope(PooledContextScope) { return true; }
bool quality_scope(PooledContextScope scope) { return scope==PooledContextScope::quality; }
int64_t source(PooledContextRole role) {
  switch(role) { case PooledContextRole::compact_control:return 0; case PooledContextRole::pooled_width_candidate:return 1; }
  throw std::runtime_error("explicit pooled role required");
}
int64_t parameter_count(int64_t input_source) { return input_source==1 ? 231949 : 225805; }

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[rpb pooled context adapter] " + message);
}
const char *fit_protocol(PooledContextScope scope) {
  switch(scope) { case PooledContextScope::quality:return kPooledContextFitProtocol;
    case PooledContextScope::engineering:return kPooledContextFixtureFitProtocol; }
  throw std::runtime_error("explicit pooled scope required");
}
void budget_contract(PooledContextScope scope,int64_t updates) {
  (void)fit_protocol(scope);
  require(scope==PooledContextScope::quality ? updates==0 || updates==512 : updates==0 || updates==1 || updates==2 || updates==4,
      "budget outside the declared pooled scope");
}
const char *tag(int64_t input_source) { return input_source==1 ? "RPB-v12" : "RPB-v10"; }

bool source_id(const std::string &value) {
  return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
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
torch::Tensor cpu(const torch::Tensor &value) { return value.detach().to(torch::kCPU).contiguous().clone(); }
void text(torch::serialize::OutputArchive &a, const std::string &key, const std::string &value) {
  a.write(key, embedding::archive::text_tensor(value), true);
}
std::string text(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor value; a.read(key, value, true); return embedding::archive::tensor_text(value);
}
int64_t integer(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor value; a.read(key, value, true);
  require(value.device().is_cpu() && value.scalar_type() == torch::kInt64 && value.dim() == 0,
      "CPU int64 scalar required: " + key);
  return value.item<int64_t>();
}
std::string manifest(const std::vector<std::string> &ids) {
  std::string out; for (const auto &id : ids) out += std::to_string(id.size()) + ':' + id; return out;
}
std::string path_key(const std::string &path) {
  require(!path.empty() && !fs::is_symlink(fs::symlink_status(path)), "empty or symlink checkpoint path");
  return fs::weakly_canonical(fs::absolute(path)).string();
}
std::string file_id(const std::string &path) {
  require(fs::is_regular_file(path) && !fs::is_symlink(fs::symlink_status(path)) && fs::hard_link_count(path) == 1,
      "regular non-aliased parent required: " + path);
  std::ifstream in(path, std::ios::binary); require(bool(in), "cannot read parent bytes");
  uint64_t hash = 14695981039346656037ULL; std::array<char, 65536> data{};
  while (in) {
    in.read(data.data(), data.size());
    for (std::streamsize i = 0; i < in.gcount(); ++i) { hash ^= static_cast<unsigned char>(data[static_cast<size_t>(i)]); hash *= 1099511628211ULL; }
  }
  require(in.eof(), "parent read failed");
  std::ostringstream out; out << "fnv1a64-runtime-content-v1-" << std::hex << std::setw(16) << std::setfill('0') << hash; return out.str();
}
std::map<std::string, std::string> bindings(const std::string &path, PooledContextScope scope) {
  std::map<std::string, std::string> result;
  for (const auto &suffix : {std::string(), std::string(".audit.pt"), std::string(".scaler.pt"), std::string(".training-raw.pt")})
    result.emplace(suffix, file_id(path + suffix));
  if (curve_scope(scope)) result.emplace(kLearningCurveContinuationSuffix,
      file_id(path + kLearningCurveContinuationSuffix));
  return result;
}
Named parameters(const Model &model) {
  Named result; for (const auto &p : model->named_parameters()) result.emplace(p.key(), cpu(p.value())); return result;
}
Named buffers(const Model &model) {
  Named result; for (const auto &b : model->named_buffers()) result.emplace(b.key(), cpu(b.value())); return result;
}
bool same(const Named &left, const Named &right) {
  if (left.size() != right.size()) return false;
  for (const auto &[key, value] : left) {
    const auto found = right.find(key);
    if (found == right.end() || value.scalar_type() != found->second.scalar_type() ||
        value.sizes() != found->second.sizes() || !torch::equal(value, found->second)) return false;
  }
  return true;
}
void named(torch::serialize::OutputArchive &a, const std::string &key, const Named &values) {
  torch::serialize::OutputArchive group; group.write("count", torch::tensor(static_cast<int64_t>(values.size())), true);
  int64_t i = 0;
  for (const auto &[name, value] : values) {
    torch::serialize::OutputArchive child; text(child, "parameter_name", name); child.write("value", value, true);
    group.write("tensor_" + std::to_string(i++), child);
  }
  a.write(key, group);
}
void save_exclusive(const std::string &path, torch::serialize::OutputArchive &a) {
  require(!fs::exists(path) && !fs::is_symlink(fs::symlink_status(path)), "snapshot asset already exists");
  const int handle = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  require(handle >= 0, "cannot exclusively reserve snapshot asset"); ::close(handle);
  embedding::archive::save_archive(path, a);
}
Input raw_input(const embedding::Batch &batch, const Config &c, const ev::ProviderFitInput &fit) {
  Input result{batch.data, batch.feature_mask, torch::tensor(fit.channel_ids, torch::kInt64),
      torch::full({batch.data.size(0)}, fit.endpoint, torch::kFloat64), fit.sampling_interval};
  validate_input(result, c); return result;
}
void configuration(const Settings &settings, PooledContextScope scope) {
  validate_settings(settings); (void)fit_protocol(scope); const auto &c = settings.model;
  require(c.temporal_difference_input == 0, "historical pooled context factory rejects visible difference input");
  const bool ceiling=scope==PooledContextScope::quality ? settings.steps==512 && settings.attempt_limit==1024 : settings.steps==4;
  require(c.device.is_cuda() && c.channel_mixer_placement==1 && (c.global_pool_input_source==0 || c.global_pool_input_source==1) &&
      c.channel_count == 3 && c.history_length == 32 && c.input_width == 3 && c.patch_length == 8 &&
      c.encoder_width == 64 && c.export_width == 32 && c.num_layers == 3 && c.num_heads == 4 &&
      c.feedforward_width == 256 && c.decoder_hidden_width == 128 && c.channel_mixer_layers == 1 &&
      c.global_bottleneck_mode == 2 && c.dropout == 0 && c.huber_delta == 1 &&
      c.sampling_interval == 1 && resolved_channel_ids(c) == std::vector<int64_t>({0, 1, 2}) &&
      settings.batch_size == 8 && settings.threads == 1 && settings.learning_rate == .001 &&
      settings.weight_decay == .0001 && settings.gradient_clip_norm == 1 && settings.log_every == 1 &&
      ceiling,
      "fixed architecture, CUDA, optimizer, trace and scope ceiling required");
}
void fit_contract(const ev::ProviderFitInput &fit, PooledContextScope scope) {
  require(fit.protocol_id == fit_protocol(scope) && fit.shape.channel_count == 3 &&
      fit.shape.history_length == 32 && fit.shape.input_width == 3 && fit.shape.dtype == torch::kFloat64 &&
      fit.shape.device.is_cpu() && fit.channel_ids == std::vector<int64_t>({0, 1, 2}) &&
      fit.feature_units == "unitless,unitless,unitless" && fit.sampling_interval == 1 && fit.endpoint == 31 &&
      fit.training_observations.data.defined() && fit.training_observations.feature_mask.defined() &&
      fit.training_observations.data.device().is_cpu() && fit.training_observations.data.scalar_type() == torch::kFloat64 &&
      fit.training_observations.feature_mask.device().is_cpu() && fit.training_observations.feature_mask.scalar_type() == torch::kBool &&
      fit.training_observations.data.dim() == 4 && fit.training_observations.data.size(0) > 0 &&
      fit.training_source_ids.size() == static_cast<size_t>(fit.training_observations.data.size(0)),
      "exact label-free timing TRAIN namespace, precision, units and geometry required");
  for (const auto &id : fit.training_source_ids) require(!id.empty(), "empty TRAIN source identity");
  if (quality_scope(scope))
    require(fit.training_source_ids.size() == 256 && std::set<std::string>(fit.training_source_ids.begin(),
        fit.training_source_ids.end()).size() == 128, "quality requires256 TRAIN rows/128 source groups");
}
struct Parent {
  Checkpoint checkpoint;
  ev::ProviderFitInput fit;
  std::string path;
  Named initial_parameters, initial_buffers;
  std::string scaler_identity;
  std::map<std::string, std::string> content, audit;
  PooledContextScope scope{PooledContextScope::quality};
};

Named read_named(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::serialize::InputArchive group; archive.read(key, group);
  const auto count = integer(group, "count"); require(count >= 0, "negative named-state count");
  Named result;
  for (int64_t i = 0; i < count; ++i) {
    torch::serialize::InputArchive entry; group.read("tensor_" + std::to_string(i), entry);
    torch::Tensor value; entry.read("value", value, true);
    require(value.device().is_cpu() && torch::isfinite(value).all().item<bool>() &&
        result.emplace(text(entry, "parameter_name"), value).second, "invalid duplicate/non-CPU named state");
  }
  require(group.keys().size() == static_cast<size_t>(count + 1), "unexpected named-state entries");
  return result;
}

void verify_continuation(const std::shared_ptr<Parent> &parent) {
  if (!curve_scope(parent->scope)) return;
  const auto &cp = parent->checkpoint;
  torch::serialize::InputArchive a; a.load_from(parent->path + kLearningCurveContinuationSuffix, torch::kCPU);
  auto resolved = parse_settings(text(a, "resolved_settings"));
  resolved.model.device = cp.settings.model.device;
  require(text(a, "artifact_kind") == kPooledContextContinuationArtifact &&
      text(a, "protocol_id") == parent->fit.protocol_id &&
      settings_text(resolved) == settings_text(cp.settings) &&
      text(a, "fit_source_manifest") == manifest(parent->fit.training_source_ids) &&
      text(a, "checkpoint_path") == parent->path &&
      text(a, "core_writer_source_fingerprint") == cp.source_fingerprint &&
      text(a, "training_producer_source_fingerprint") == parent->audit.at("training_producer_source_fingerprint") &&
      text(a, "training_policy_id") == policy &&
      text(a, "state_capture_policy") ==
          "live_named_CUDA_parameter_state_to_CPU;no_model_forward;no_optimizer_reload_or_step" &&
      integer(a, "attempted_steps") == cp.attempted_steps && integer(a, "completed_steps") == cp.completed_steps &&
      integer(a, "sampled_rows") == cp.completed_steps * 8 &&
      integer(a, "channel_mixer_placement_value") == cp.settings.model.channel_mixer_placement,
      "curve continuation state identity/counter/source differs");
  require(integer(a,"global_pool_input_source_value")==cp.settings.model.global_pool_input_source &&
      integer(a,"shared_parameter_values")==219469 &&
      integer(a,"copied_parameter_values")== (cp.settings.model.global_pool_input_source ? 219469 : 0) &&
      integer(a,"inactive_projection_values")== (cp.settings.model.global_pool_input_source ? 2080 : 0),
      "typed pooled initialization values differ");
  require(text(a,"paired_initialization_json")==parent->audit.at("paired_initialization_json"),
      "paired initialization JSON differs between audit and continuation");
  const auto initial=read_named(a,"initial_model_parameters");
  require(initial.size()==parent->initial_parameters.size(),"complete paired initial parameters required");
  for (const auto &[name,value]:parent->initial_parameters)
    require(initial.count(name) && initial.at(name).scalar_type()==value.scalar_type() && initial.at(name).sizes()==value.sizes(),
        "initial/current parameter names, shapes and dtypes differ");
  require(same(read_named(a,"initial_model_buffers"),parent->initial_buffers),"initial/current buffers changed");
  if (cp.completed_steps==0) require(same(initial,parent->initial_parameters),"point0 initial/current values differ");
  if (cp.settings.model.global_pool_input_source==1)
  {
    torch::Tensor before; a.read("candidate_first_before_copy",before,true);
    require(before.device().is_cpu() && before.scalar_type()==torch::kFloat32 && before.sizes()==torch::IntArrayRef({64,195}) &&
        torch::equal(before,initial.at("global_pool_first.weight")),"wider first weight initialization changed during copy");
    for (const auto &name : {std::string("export_projection.weight"),std::string("export_projection.bias")})
      require(torch::equal(initial.at(name),parent->initial_parameters.at(name)),"diagnostic projection changed");
  }

  for (const auto &key : {"context_requested_deleted_coordinates", "context_actual_deleted_coordinates", "context_restored_coordinates"})
    require(std::to_string(integer(a, key)) == parent->audit.at(key), "live context-state count differs");
  require(same(read_named(a, "model_parameters"), parameters(cp.model)) &&
      same(read_named(a, "model_buffers"), buffers(cp.model)), "live named model state differs from saved checkpoint");
  torch::serialize::InputArchive scaler; a.read("scaler", scaler);
  require(FrozenScaler::load(scaler).identity() == cp.scaler.identity(), "live frozen scaler differs");
  torch::Tensor counters, values; a.read("loss_trace_counters", counters, true); a.read("loss_trace_values", values, true);
  require(counters.device().is_cpu() && counters.scalar_type() == torch::kInt64 &&
      counters.sizes() == torch::IntArrayRef({cp.completed_steps, 3}) &&
      values.device().is_cpu() && values.scalar_type() == torch::kFloat64 &&
      values.sizes() == torch::IntArrayRef({cp.completed_steps, 2}) && torch::isfinite(values).all().item<bool>(),
      "complete live trace shape/precision differs");
  for (int64_t i = 0; i < cp.completed_steps; ++i)
    require(counters[i][0].item<int64_t>() == i + 1 && counters[i][1].item<int64_t>() == i + 1 &&
        counters[i][2].item<int64_t>() > 0, "unskipped trace prefix differs");
  torch::serialize::InputArchive optimizer; a.read("optimizer_state", optimizer);
  const auto named_parameters = cp.model->named_parameters();
  const auto count = integer(optimizer, "parameter_count"), declared_active = integer(optimizer, "active_state_count");
  require(count == static_cast<int64_t>(named_parameters.size()) && declared_active >= 0 && declared_active <= count,
      "named AdamW association count differs");
  std::string previous; int64_t active = 0;
  for (int64_t i = 0; i < count; ++i) {
    torch::serialize::InputArchive entry; optimizer.read("parameter_" + std::to_string(i), entry);
    const auto name = text(entry, "parameter_name");
    require((i == 0 || name > previous) && named_parameters.contains(name), "AdamW lexical parameter association differs");
    previous = name; torch::Tensor shape, present; entry.read("parameter_shape", shape, true); entry.read("has_state", present, true);
    const auto parameter = named_parameters[name];
    require(shape.device().is_cpu() && shape.scalar_type() == torch::kInt64 &&
        torch::equal(shape, torch::tensor(parameter.sizes().vec(), torch::kInt64)) &&
        present.device().is_cpu() && present.scalar_type() == torch::kBool && present.dim() == 0,
        "AdamW association shape/active flag differs");
    if (cp.settings.model.global_pool_input_source==1 &&
        (name=="export_projection.weight" || name=="export_projection.bias"))
      require(!present.item<bool>(),"diagnostic projection must not acquire AdamW state");
    if (present.item<bool>()) {
      const auto step = integer(entry, "step"); torch::Tensor first, second;
      entry.read("exp_avg", first, true); entry.read("exp_avg_sq", second, true);
      require(step > 0 && step <= cp.completed_steps && first.device().is_cpu() && second.device().is_cpu() &&
          first.scalar_type() == parameter.scalar_type() && second.scalar_type() == parameter.scalar_type() &&
          first.sizes() == parameter.sizes() && second.sizes() == parameter.sizes() &&
          torch::isfinite(first).all().item<bool>() && torch::isfinite(second).all().item<bool>(),
          "CPU AdamW moment/step witness differs");
      ++active;
    }
  }
  require(active == declared_active && (cp.completed_steps != 0 || active == 0) &&
      optimizer.keys().size() == static_cast<size_t>(count + 2), "AdamW active named-state set differs");
}
std::shared_ptr<Parent> admit(const PooledContextSnapshotOptions &options, const ev::ProviderFitInput &fit) {
  budget_contract(options.scope, options.expected_completed_updates);
  require(source(options.role) == 0 || source(options.role) == 1,
      "explicit placement0 or placement1 required");
  require(source_id(options.expected_core_source_fingerprint) && source_id(options.expected_training_producer_source_fingerprint),
      "explicit parent writer and training producer fingerprints required");
  fit_contract(fit, options.scope); require(torch::cuda::is_available(), "CUDA mandatory; no CPU fallback");
  auto parent = std::make_shared<Parent>(); parent->path = path_key(options.checkpoint_path); parent->scope = options.scope;
  parent->content = bindings(parent->path, options.scope);
  parent->checkpoint = load_checkpoint(parent->path, torch::Device(torch::kCUDA, 0));
  auto &cp = parent->checkpoint; const auto &c = cp.settings.model; const auto updates = options.expected_completed_updates;
  configuration(cp.settings, options.scope);
  require(c.global_pool_input_source == source(options.role) &&
      cp.attempted_steps == updates && cp.completed_steps == updates && cp.training_policy_id == policy &&
      cp.source_fingerprint == options.expected_core_source_fingerprint &&
      cp.settings.seed == static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL),
      "declared placement, policy, seed, unskipped counters or writer scope differs");
  parent->fit = fit; parent->fit.training_observations.data = cpu(fit.training_observations.data);
  parent->fit.training_observations.feature_mask = cpu(fit.training_observations.feature_mask);
  const auto raw = describe_dataset(raw_input(parent->fit.training_observations, c, fit), c, fit.feature_units);
  const auto saved = load_dataset(parent->path + ".training-raw.pt", c);
  const auto scaler = load_scaler(parent->path + ".scaler.pt", c, cp.schema_id);
  require(raw.dataset_id == cp.dataset_id && raw.schema_id == cp.schema_id && cp.scaler_fit_dataset_id == cp.dataset_id &&
      saved.dataset_id == cp.dataset_id && saved.schema_id == cp.schema_id && saved.feature_units == fit.feature_units &&
      scaler.identity() == cp.scaler.identity(), "exact TRAIN/raw/scaler/schema companions differ");
  torch::serialize::InputArchive audit; audit.load_from(parent->path + ".audit.pt", torch::kCPU);
  require(text(audit, "artifact_kind") == "rpb_learning_curve_training_audit_v1" &&
      text(audit, "protocol_id") == fit.protocol_id && text(audit, "model_tag") == tag(c.global_pool_input_source) &&
      text(audit, "architecture_id") == architecture_id(c) &&
      text(audit, "channel_mixer_placement") == std::to_string(c.channel_mixer_placement) &&
      integer(audit, "channel_mixer_placement_value") == c.channel_mixer_placement &&
      text(audit, "output_semantics") == output_semantics(c) &&
      text(audit, "reconstruction_export_semantics") == reconstruction_output_semantics(c) &&
      text(audit, "fit_source_manifest") == manifest(fit.training_source_ids) && text(audit, "feature_units") == fit.feature_units &&
      text(audit, "actual_training_seed") == std::to_string(cp.settings.seed) &&
      text(audit, "initialization_seed") == std::to_string(training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL)) &&
      text(audit, "training_dataset_id") == cp.dataset_id && text(audit, "scaler_fit_dataset_id") == cp.dataset_id &&
      text(audit, "preprocessing_id") == cp.scaler.identity() && integer(audit, "attempted_steps") == updates &&
      integer(audit, "completed_steps") == updates && integer(audit, "sampled_rows") == updates * 8 &&
      text(audit, "model_weight_update_budget") == std::to_string(updates) &&
      torch::Device(text(audit, "training_device")).is_cuda() &&
      text(audit, "core_writer_source_fingerprint") == options.expected_core_source_fingerprint &&
      text(audit, "training_producer_source_fingerprint") == options.expected_training_producer_source_fingerprint &&
      text(audit, "rng_policy") == counter_policy &&
      text(audit, "sampling_policy") == "with_replacement_counter_rows;sampled_rows_includes_no_update_attempts" &&
      text(audit, "optimizer_policy") == "one_continuous_AdamW_state;absolute_completed_update_budgets",
      "typed architecture/source/counter/TRAIN audit differs");
  auto resolved = parse_settings(text(audit, "resolved_settings")); resolved.model.device = c.device;
  require(settings_text(resolved) == settings_text(cp.settings), "resolved saved settings differ");
  require(integer(audit,"global_pool_input_source_value")==c.global_pool_input_source &&
      text(audit,"global_pool_input_source")==std::to_string(c.global_pool_input_source) &&
      text(audit,"global_pool_input_semantics")==global_pool_input_semantics(c) &&
      integer(audit,"shared_parameter_values")==219469 &&
      integer(audit,"copied_parameter_values")== (c.global_pool_input_source ? 219469 : 0) &&
      integer(audit,"inactive_projection_values")== (c.global_pool_input_source ? 2080 : 0) &&
      text(audit,"pooled_shared_parameter_values")=="219469" &&
      text(audit,"pooled_copied_parameter_values")== (c.global_pool_input_source ? "219469" : "0") &&
      text(audit,"pooled_inactive_projection_values")== (c.global_pool_input_source ? "2080" : "0") &&
      text(audit,"pooled_loss_reachable_parameter_values")== (c.global_pool_input_source ? "229869" : "225805") &&
      text(audit,"pooled_nonshared_parameter_name")=="global_pool_first.weight" &&
      text(audit,"pooled_initialization_policy")== (c.global_pool_input_source ?
        "copy-all-shared-named-parameters-and-buffers-from-compact-point0-before-AdamW;except-global_pool_first.weight" :
        "compact-control-independent-initialization;no-copy"),"strict pooled initialization binding differs");
  if (c.global_pool_input_source==1) {
    const auto reference=text(audit,"pooled_reference_point0_path");
    require(!reference.empty() && path_key(reference)==reference,"canonical paired control point0 path required");
    for (const auto &[suffix,id]:bindings(reference,options.scope))
      require(text(audit,"pooled_reference_content_id"+suffix)==id,"paired control point0 bytes changed");
  } else require(text(audit,"pooled_reference_point0_path").empty(),"control cannot declare a copy reference");
  std::ostringstream paired;
  paired << "{\"input_source\":" << c.global_pool_input_source
      << ",\"copied_before_AdamW\":true,\"shared_parameter_values\":219469,\"copied_parameter_values\":"
      << (c.global_pool_input_source ? 219469 : 0) << ",\"inactive_projection_values\":" << (c.global_pool_input_source ? 2080 : 0)
      << ",\"nonshared_parameter_name\":\"global_pool_first.weight\",\"reference_point0_path\":"
      << std::quoted(text(audit,"pooled_reference_point0_path")) << "}";
  require(text(audit,"paired_initialization_json")==paired.str(),"typed paired initialization JSON declaration differs");
  torch::Tensor ids, interval, endpoint, ratio, seconds, changed, finite;
  audit.read("channel_order", ids, true); audit.read("sampling_interval", interval, true); audit.read("endpoint", endpoint, true);
  audit.read("context_deletion_ratio_value", ratio, true); audit.read("training_seconds", seconds, true);
  audit.read("weights_changed", changed, true); audit.read("finite_gradients", finite, true);
  require(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 &&
      torch::equal(ids, torch::tensor(fit.channel_ids, torch::kInt64)) &&
      interval.scalar_type() == torch::kFloat64 && interval.dim() == 0 && interval.item<double>() == 1 &&
      endpoint.scalar_type() == torch::kFloat64 && endpoint.dim() == 0 && endpoint.item<double>() == 31 &&
      ratio.scalar_type() == torch::kFloat64 && ratio.dim() == 0 && ratio.item<double>() == .15 &&
      integer(audit, "context_deletion_stream_value") == static_cast<int64_t>(context_deletion::stream) &&
      seconds.scalar_type() == torch::kFloat64 && seconds.dim() == 0 && std::isfinite(seconds.item<double>()) &&
      seconds.item<double>() >= 0 && changed.scalar_type() == torch::kBool && changed.dim() == 0 &&
      finite.scalar_type() == torch::kBool && finite.dim() == 0 &&
      (updates == 0 ? (!changed.item<bool>() && !finite.item<bool>() && seconds.item<double>() == 0) :
          (changed.item<bool>() && finite.item<bool>() && seconds.item<double>() > 0)),
      "typed IDs/time/context ratio or CUDA training witness differs");
  require(text(audit, "training_policy_id") == policy && text(audit, "context_deletion_ratio") == "0.15" &&
      text(audit, "context_deletion_stream") == "0x6374782d64726f70" &&
      text(audit, "context_deletion_rng_policy") == context_deletion::rng_policy &&
      text(audit, "context_deletion_repair_policy") == context_deletion::repair_policy &&
      text(audit, "context_deletion_visibility_policy") == context_deletion::visibility_policy &&
      text(audit, "context_deletion_count_policy") == "cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only" &&
      text(audit, "context_deletion_resume_policy") == "fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API",
      "unchanged coordinate15 policy companion required");
  const auto requested = integer(audit, "context_requested_deleted_coordinates");
  const auto actual = integer(audit, "context_actual_deleted_coordinates");
  const auto restored = integer(audit, "context_restored_coordinates");
  require(requested >= actual && actual >= 0 && restored == requested - actual && requested <= updates * 8 * 288 &&
      (updates != 0 || requested == 0), "cumulative context deletion counts differ");
  for (const auto &key : audit.keys()) require(key.rfind("context_deletion_schedule", 0) != 0 &&
      key != "context_ordinary_attempts" && key != "context_deletion_attempts", "balanced policy not permitted");
  int64_t count = 0;
  cp.model->eval();
  for (auto &p : cp.model->parameters()) {
    require(p.is_cuda() && torch::isfinite(p).all().item<bool>(), "finite CUDA parameters required");
    p.set_requires_grad(false); count += p.numel();
  }
  require(count == parameter_count(c.global_pool_input_source), "exact registered parameter values required");
  for (const auto &b : cp.model->buffers()) require(b.is_cuda() && torch::isfinite(b).all().item<bool>(), "finite CUDA buffers required");
  parent->initial_parameters = parameters(cp.model); parent->initial_buffers = buffers(cp.model);
  parent->scaler_identity = cp.scaler.identity();
  for (const auto &key : audit.keys()) {
    if (key == "artifact_kind") continue;
    torch::Tensor value; audit.read(key, value, true);
    if (value.scalar_type() == torch::kUInt8) parent->audit.emplace(key, embedding::archive::tensor_text(value));
  }
  parent->audit["context_requested_deleted_coordinates"] = std::to_string(requested);
  parent->audit["context_actual_deleted_coordinates"] = std::to_string(actual);
  parent->audit["context_restored_coordinates"] = std::to_string(restored);
  verify_continuation(parent);
  require(bindings(parent->path, parent->scope) == parent->content, "parent bytes changed during admission");
  return parent;
}
void immutable(const std::shared_ptr<Parent> &parent) {
  const auto &cp = parent->checkpoint;
  require(!cp.model->is_training() && same(parameters(cp.model), parent->initial_parameters) &&
      same(buffers(cp.model), parent->initial_buffers) && cp.scaler.identity() == parent->scaler_identity &&
      bindings(parent->path, parent->scope) == parent->content, "immutable checkpoint/model/buffers/scaler changed");
  for (const auto &p : cp.model->parameters()) require(!p.requires_grad() && p.is_cuda(), "frozen CUDA parameter flags changed");
}
} // namespace

ev::CurveSnapshot make_pooled_context_snapshot(const PooledContextSnapshotOptions &options, const ev::ProviderFitInput &fit) {
  // Declaration checks precede generator/thread state changes and file reads.
  budget_contract(options.scope, options.expected_completed_updates);
  require(source(options.role) == 0 || source(options.role) == 1,
      "explicit placement required");
  fit_contract(fit, options.scope);
  const RuntimeIsolation isolation; at::set_num_threads(1);
  const auto parent = admit(options, fit); auto metadata = parent->fit;
  metadata.training_observations = {}; metadata.training_source_ids.clear();
  const auto &cp = parent->checkpoint;
  ev::CurveSnapshot snapshot;
  snapshot.features.provenance = std::string(tag(source(options.role))) +
      "; immutable timing-protocol checkpoint; exact contextual-global32 CUDA serving; ordinary original-Q decoder";
  auto &fields = snapshot.features.audit_fields;
  fields = parent->audit;
  fields["protocol_id"] = kPooledContextProtocol;
  fields["original_training_protocol_id"] = fit.protocol_id;
  fields["parent_checkpoint_path"] = parent->path;
  fields["parent_writer_source_fingerprint"] = cp.source_fingerprint;
  fields["parent_training_producer_source_fingerprint"] = options.expected_training_producer_source_fingerprint;
  fields["snapshot_loader_source_fingerprint"] = POOLED_CONTEXT_ADAPTER_SOURCE_ID;
  fields["source_fingerprint_scope"] = "parent=core_writer_and_training_producer;loader=new_pooled_context_adapter";
  fields["original_encoder_attempted"] = std::to_string(cp.attempted_steps);
  fields["original_encoder_completed"] = std::to_string(cp.completed_steps);
  fields["training_schema_id"] = cp.schema_id; fields["training_dataset_id"] = cp.dataset_id;
  fields["preprocessing_id"] = cp.scaler.identity(); fields["parameter_count"] = std::to_string(parameter_count(cp.settings.model.global_pool_input_source));
  fields["encoder_updates"] = "0"; fields["decoder_updates"] = "0"; fields["head_refits"] = "0";
  fields["no_optimizer_created"] = "true"; fields["inference_device"] = cp.settings.model.device.str();
  fields["snapshot_policy"] = "independent_CUDA_model;eval_no_grad;frozen_timing_TRAIN_scaler;ordinary_query_no_context_erasure;no_refit";
  for (const auto &[suffix, content] : parent->content) fields["parent_content_id" + suffix] = content;
  snapshot.features.surfaces.emplace("curve_global", ev::SurfaceDescription{ev::SurfaceKind::global, "at least one observed channel", {}});
  snapshot.features.extract = [parent, metadata, provenance = snapshot.features.provenance](const embedding::Batch &batch) {
    const RuntimeIsolation guard; at::set_num_threads(1); torch::NoGradGuard no_grad; immutable(parent);
    auto &c = parent->checkpoint; const auto input = c.scaler.transform(raw_input(batch, c.settings.model, metadata), c.settings.model);
    const auto encoded = c.model->encode(input); const auto z = compact_reconstruction_export(encoded, c.settings.model);
    require(input.data.is_cuda() && z.is_cuda() && z.scalar_type() == torch::kFloat32 && z.dim() == 2 &&
        z.size(0) == batch.data.size(0) && z.size(1) == 32 && torch::isfinite(z).all().item<bool>(), "finite exact CUDA BD32 required");
    ev::FeatureMap result{{"curve_global", {cpu(z), cpu(encoded.sample_valid_mask), provenance}}};
    immutable(parent); return result;
  };
  snapshot.reconstruct = [parent, metadata](const embedding::Batch &batch, const torch::Tensor &hidden) {
    const RuntimeIsolation guard; at::set_num_threads(1); torch::NoGradGuard no_grad; immutable(parent);
    auto &c = parent->checkpoint; const auto input = c.scaler.transform(raw_input(batch, c.settings.model, metadata), c.settings.model);
    const auto output = c.model->forward(input, hidden);
    require(input.data.is_cuda() && output.reconstruction.is_cuda() && torch::isfinite(output.reconstruction).all().item<bool>(),
        "finite CUDA ordinary original-Q reconstruction required");
    ev::CurveReconstruction result{cpu(output.reconstruction), cpu(input.data), cpu(output.eligible_channels)};
    immutable(parent); return result;
  };
  snapshot.features.save_assets = [parent, fields](const std::string &directory) {
    const RuntimeIsolation guard; immutable(parent); torch::serialize::OutputArchive a;
    text(a, "artifact_kind", curve_scope(parent->scope) ? kPooledContextSnapshotArtifact : kPooledContextSnapshotArtifact);
    for (const auto &[key, value] : fields) text(a, key, value);
    a.write("global_pool_input_source_value",torch::tensor(parent->checkpoint.settings.model.global_pool_input_source),true);
    a.write("channel_mixer_placement_value", torch::tensor(parent->checkpoint.settings.model.channel_mixer_placement, torch::kInt64), true);
    a.write("original_encoder_attempted_value", torch::tensor(parent->checkpoint.attempted_steps), true);
    a.write("original_encoder_completed_value", torch::tensor(parent->checkpoint.completed_steps), true);
    named(a, "model_parameters", parent->initial_parameters); named(a, "model_buffers", parent->initial_buffers);
    torch::serialize::OutputArchive scaler; parent->checkpoint.scaler.save(scaler); a.write("scaler", scaler);
    save_exclusive((fs::path(directory) / kPooledContextSnapshotAuditFile).string(), a); immutable(parent);
  };
  immutable(parent); return snapshot;
}

ev::CurveTrainerFactory make_pooled_context_trainer(const Settings &settings,PooledContextRole role,const std::string &reference,PooledContextScope scope) {
  require(settings.model.global_pool_input_source==source(role) && (source(role) ? !reference.empty() : reference.empty()),
      "explicit pooled role/reference/config must agree");
  configuration(settings, scope);
  const RuntimeIsolation isolation;
  const auto base_factory = make_learning_curve_trainer(settings,
      ContextDeletionOptions{true, ContextDeletionRecipe::coordinate15_v1}, LearningCurveStateWitnessOptions{true},TrainingSourceGainOptions{},PooledContextInitializationOptions{true,reference});
  return [settings, role, scope, base_factory](const ev::ProviderFitInput &fit) {
    fit_contract(fit, scope); const RuntimeIsolation guard; at::set_num_threads(1);
    auto captured_fit = fit;
    captured_fit.training_observations.data = cpu(fit.training_observations.data);
    captured_fit.training_observations.feature_mask = cpu(fit.training_observations.feature_mask);
    auto base = base_factory(captured_fit); auto latest = std::make_shared<ev::CurveProgress>();
    auto saved = std::make_shared<std::map<std::string, std::pair<int64_t, std::map<std::string, std::string>>>>();
    auto failed = std::make_shared<bool>(false);
    const auto writer = base.audit_fields.at("core_writer_source_fingerprint");
    const auto producer = base.audit_fields.at("training_producer_source_fingerprint");
    ev::CurveTrainer result; result.audit_fields = base.audit_fields;
    result.audit_fields["snapshot_adapter_source_fingerprint"] = POOLED_CONTEXT_ADAPTER_SOURCE_ID;
    result.audit_fields["snapshot_policy"] = "new_pooled_protocol_bound_CUDA_only;historical_CPU_snapshot_not_called";
    result.train_to = [base, scope, latest, saved, failed](int64_t budget) {
      require(!*failed, "failed live trainer cannot continue"); budget_contract(scope, budget);
      const RuntimeIsolation isolate; at::set_num_threads(1);
      try {
        if (curve_scope(scope))
          for (const auto &[path, point] : *saved)
            require(bindings(path, scope) == point.second, "an earlier curve point's bytes changed");
        auto progress = base.train_to(budget);
        require(progress.completed == budget && progress.attempted == budget, "unskipped absolute prefix required");
        if (curve_scope(scope)) {
          require(progress.losses.size() == static_cast<size_t>(budget) &&
              progress.losses.size() >= latest->losses.size(), "complete cumulative curve trace required");
          for (size_t i = 0; i < latest->losses.size(); ++i) {
            const auto &before = latest->losses[i], &after = progress.losses[i];
            require(before.attempted == after.attempted && before.completed == after.completed &&
                before.target_cells == after.target_cells && before.loss == after.loss &&
                before.gradient_norm == after.gradient_norm, "previously delivered curve trace prefix changed");
          }
        }
        *latest = progress; return progress;
      } catch (...) { *failed = true; throw; }
    };
    result.save_checkpoint = [base, latest, saved, failed, scope](const std::string &path) {
      require(!*failed, "failed trainer cannot save"); budget_contract(scope, latest->completed);
      const RuntimeIsolation isolate; at::set_num_threads(1); base.save_checkpoint(path);
      require(saved->emplace(path_key(path), std::make_pair(latest->completed, bindings(path, scope))).second, "point already recorded");
    };
    result.snapshot = [fit = captured_fit, settings, role, scope, writer, producer, saved, failed](const std::string &path) {
      require(!*failed, "failed trainer cannot serve snapshots"); const auto found = saved->find(path_key(path));
      require(found != saved->end() && bindings(path, scope) == found->second.second, "snapshot is not this trainer's immutable saved point");
      PooledContextSnapshotOptions options{path,role,found->second.first,scope,writer,producer};
      return make_pooled_context_snapshot(options, fit);
    };
    return result;
  };
}

std::map<std::string,std::string> audit_pooled_context_initialization(const std::string &control_path,
    const std::string &candidate_path,const ev::ProviderFitInput &fit,PooledContextScope scope) {
  fit_contract(fit,scope); const RuntimeIsolation isolate; at::set_num_threads(1);
  auto options=[&](const std::string &path,PooledContextRole role) {
    torch::serialize::InputArchive a; a.load_from(path+".audit.pt",torch::kCPU);
    return PooledContextSnapshotOptions{path,role,0,scope,text(a,"core_writer_source_fingerprint"),text(a,"training_producer_source_fingerprint")};
  };
  const auto control=admit(options(control_path,PooledContextRole::compact_control),fit);
  const auto candidate=admit(options(candidate_path,PooledContextRole::pooled_width_candidate),fit);
  auto expected=candidate->checkpoint.settings; expected.model.global_pool_input_source=0;
  require(settings_text(expected)==settings_text(control->checkpoint.settings) && control->scaler_identity==candidate->scaler_identity &&
      control->checkpoint.dataset_id==candidate->checkpoint.dataset_id && control->checkpoint.schema_id==candidate->checkpoint.schema_id &&
      control->checkpoint.source_fingerprint==candidate->checkpoint.source_fingerprint &&
      control->audit.at("training_producer_source_fingerprint")==candidate->audit.at("training_producer_source_fingerprint") &&
      candidate->audit.at("pooled_reference_point0_path")==path_key(control_path) &&
      same(control->initial_buffers,candidate->initial_buffers),"paired original TRAIN/config/buffers/source differs");
  int64_t common=0;
  for (const auto &[name,value]:control->initial_parameters) {
    const auto &other=candidate->initial_parameters.at(name);
    if (name=="global_pool_first.weight") {
      require(value.sizes()==torch::IntArrayRef({64,99}) && other.sizes()==torch::IntArrayRef({64,195}),"declared wider input required"); continue;
    }
    require(value.scalar_type()==other.scalar_type() && value.sizes()==other.sizes() && torch::equal(value,other),"copied common point0 values differ");
    common+=value.numel();
  }
  require(common==219469,"exact219469 shared parameter values required");
  for (const auto &[suffix,id]:control->content)
    require(candidate->audit.at("pooled_reference_content_id"+suffix)==id,"paired reference bytes differ");
  immutable(control); immutable(candidate);
  return {{"common_parameters_exact","true"},{"common_buffers_exact","true"},{"scaler_exact","true"},
    {"training_dataset_exact","true"},{"counter_streams_exact","true"},{"shared_parameter_values","219469"},
    {"control_registered_parameter_values","225805"},{"candidate_registered_parameter_values","231949"},
    {"candidate_inactive_projection_values","2080"},{"candidate_loss_reachable_parameter_values","229869"},
    {"initial_features_equality_required","false"},{"nonshared_parameter_name","global_pool_first.weight"},
    {"copied_before_AdamW","true"},{"snapshot_loader_source_fingerprint",POOLED_CONTEXT_ADAPTER_SOURCE_ID}};
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
