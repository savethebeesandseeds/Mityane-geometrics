// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_native_feature_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
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
#include <memory>
#include <set>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

#ifndef FROZEN_DECODER_CALIBRATION_SOURCE_ID
#ifdef EVALUATION_SOURCE_ID
#define FROZEN_DECODER_CALIBRATION_SOURCE_ID EVALUATION_SOURCE_ID
#else
#define FROZEN_DECODER_CALIBRATION_SOURCE_ID "unrecorded"
#endif
#endif

#ifndef FROZEN_NATIVE_FEATURE_SOURCE_ID
#ifdef EVALUATION_SOURCE_ID
#define FROZEN_NATIVE_FEATURE_SOURCE_ID EVALUATION_SOURCE_ID
#else
#define FROZEN_NATIVE_FEATURE_SOURCE_ID "unrecorded"
#endif
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using Named = std::map<std::string, torch::Tensor>;
using Binding = frozen_decoder_detail::Binding;
struct Identity {
  const char *protocol, *artifact, *model_tag, *parent_policy, *fit_protocol;
  bool context;
};
const Identity &identity(Binding binding) {
  static constexpr Identity historical{kDecoderCalibrationProtocol, kDecoderCalibrationArtifact,
      "RPB-v7", "rpb-training-context-deletion-015-v1", "native-development-v1/lag_sign", true};
  static constexpr Identity fresh_v4{kFreshDecoderProtocol, kFreshDecoderArtifact,
      "RPB-v4", "", "fresh-decoder-replication-v1/lag_sign", false};
  static constexpr Identity fresh_v7{kFreshDecoderProtocol, kFreshDecoderArtifact,
      "RPB-v7", "rpb-training-context-deletion-015-v1", "fresh-decoder-replication-v1/lag_sign", true};
  switch (binding) {
    case Binding::historical_v7: return historical;
    case Binding::fresh_v4: return fresh_v4;
    case Binding::fresh_v7: return fresh_v7;
  }
  throw std::runtime_error("[rpb decoder calibration] unknown fixed protocol binding");
}
Binding fresh_binding(FrozenDecoderParentPolicy policy) {
  switch (policy) {
    case FrozenDecoderParentPolicy::ordinary_v4: return Binding::fresh_v4;
    case FrozenDecoderParentPolicy::coordinate15_v7: return Binding::fresh_v7;
  }
  throw std::runtime_error("[rpb decoder calibration] explicit ordinary_v4 or coordinate15_v7 parent policy required");
}
constexpr const char *counter_policy = "splitmix64-counter-rows-masks-torch-attempt-v1";

void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error("[rpb decoder calibration] " + message);
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
void write_text(torch::serialize::OutputArchive &a, const std::string &key, const std::string &value) {
  a.write(key, embedding::archive::text_tensor(value), true);
}
std::string text(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor value; a.read(key, value, true); return embedding::archive::tensor_text(value);
}
int64_t integer(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor value; a.read(key, value, true);
  require(value.device().is_cpu() && value.scalar_type() == torch::kInt64 && value.dim() == 0,
      "invalid typed counter: " + key);
  return value.item<int64_t>();
}
std::string source_manifest(const std::vector<std::string> &ids) {
  std::string out; for (const auto &id : ids) out += std::to_string(id.size()) + ':' + id; return out;
}
std::string path_key(const std::string &path) {
  require(!path.empty(), "empty artifact path"); return fs::weakly_canonical(fs::absolute(path)).string();
}
// Runtime content binding; the runner additionally owns the closed SHA256 manifest.
std::string file_identity(const std::string &path) {
  require(fs::is_regular_file(path) && !fs::is_symlink(fs::symlink_status(path)), "regular non-symlink parent required: " + path);
  std::ifstream input(path, std::ios::binary); require(bool(input), "cannot read bound parent: " + path);
  uint64_t hash = 14695981039346656037ULL; std::array<char, 65536> bytes{};
  while (input) { input.read(bytes.data(), bytes.size()); for (std::streamsize i = 0; i < input.gcount(); ++i) {
    hash ^= static_cast<unsigned char>(bytes[static_cast<size_t>(i)]); hash *= 1099511628211ULL;
  }}
  require(input.eof(), "parent content read failed");
  std::ostringstream out; out << "fnv1a64-runtime-content-v1-" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
std::map<std::string, std::string> parent_bindings(const std::string &path) {
  std::map<std::string, std::string> out;
  for (const auto &suffix : {std::string(), std::string(".audit.pt"), std::string(".scaler.pt"), std::string(".training-raw.pt")})
    out.emplace(suffix, file_identity(path + suffix));
  return out;
}
void verify_bindings(const std::string &path, const std::map<std::string, std::string> &bindings) {
  require(parent_bindings(path) == bindings, "immutable parent or companion bytes changed");
}
bool decoder_name(const std::string &name) {
  for (const std::string prefix : {"decoder_positions.", "decoder_channels.", "decoder_first.", "decoder_second."})
    if (name.rfind(prefix, 0) == 0) return true;
  return false;
}
Named parameters(const Model &model, bool decoder) {
  Named out; for (const auto &p : model->named_parameters())
    if (decoder_name(p.key()) == decoder) out.emplace(p.key(), cpu(p.value()));
  return out;
}
Named buffers(const Model &model) {
  Named out; for (const auto &p : model->named_buffers()) out.emplace(p.key(), cpu(p.value())); return out;
}
bool same(const Named &a, const Named &b) {
  if (a.size() != b.size()) return false;
  for (const auto &[name, value] : a) {
    const auto it = b.find(name); if (it == b.end() || value.scalar_type() != it->second.scalar_type() ||
        !torch::equal(value, it->second)) return false;
  }
  return true;
}
int64_t size(const Named &values) { int64_t result = 0; for (const auto &[name, v] : values) { (void)name; result += v.numel(); } return result; }
void freeze(Model &model) {
  model->eval(); for (auto &p : model->parameters()) p.set_requires_grad(false);
}
void write_named(torch::serialize::OutputArchive &a, const std::string &key, const Named &values) {
  torch::serialize::OutputArchive group; group.write("count", torch::tensor(static_cast<int64_t>(values.size())), true);
  int64_t index = 0;
  for (const auto &[name, value] : values) {
    torch::serialize::OutputArchive child; write_text(child, "parameter_name", name); child.write("value", value, true);
    group.write("tensor_" + std::to_string(index++), child);
  }
  a.write(key, group);
}
Named read_named(torch::serialize::InputArchive &a, const std::string &key) {
  torch::serialize::InputArchive group; a.read(key, group); const auto count = integer(group, "count");
  require(count >= 0 && count < 1000 && static_cast<int64_t>(group.keys().size()) == count + 1, "invalid named tensor group: " + key);
  Named out;
  for (int64_t i = 0; i < count; ++i) {
    torch::serialize::InputArchive child; group.read("tensor_" + std::to_string(i), child);
    torch::Tensor value; child.read("value", value, true);
    require(value.device().is_cpu() && torch::isfinite(value).all().item<bool>() &&
        out.emplace(text(child, "parameter_name"), value).second, "invalid/duplicate named tensor: " + key);
  }
  return out;
}
Input raw_input(const embedding::Batch &batch, const Config &c, const ev::ProviderFitInput &fit) {
  Input out{batch.data, batch.feature_mask, torch::tensor(fit.channel_ids, torch::kInt64),
      torch::full({batch.data.size(0)}, fit.endpoint, torch::kFloat64), fit.sampling_interval};
  validate_input(out, c); return out;
}
struct Parent {
  Checkpoint checkpoint;
  Dataset training;
  ev::ProviderFitInput fit;
  std::string path;
  std::map<std::string, std::string> bindings, audit;
};
std::shared_ptr<Parent> admit(const std::string &path, const ev::ProviderFitInput &fit,
                              int64_t updates, Binding binding, bool initial = false) {
  const auto &selected = identity(binding);
  require(torch::cuda::is_available(), "CUDA is mandatory");
  require((updates > 0 || (initial && updates == 0 && binding != Binding::historical_v7)) &&
      fit.protocol_id == selected.fit_protocol && fit.shape.dtype == torch::kFloat64 && fit.shape.device.is_cpu(),
      "exact parent budget and fixed protocol namespace float64 TRAIN required");
  auto out = std::make_shared<Parent>(); out->path = path_key(path); out->bindings = parent_bindings(out->path);
  out->checkpoint = load_checkpoint(out->path, torch::kCUDA);
  auto &cp = out->checkpoint; const auto &c = cp.settings.model;
  require(c.channel_mixer_placement == 0,
      "historical/fresh v4/v7 parent admission rejects early mixer placement");
  require(cp.training_policy_id == selected.parent_policy && cp.completed_steps == updates && cp.attempted_steps == updates,
      "exact declared unskipped parent policy/counters required");
  require(cp.settings.batch_size > 0 && updates <= std::numeric_limits<int64_t>::max() / cp.settings.batch_size,
      "parent sampled-row counter arithmetic overflow");
  require(c.channel_count == 3 && c.history_length == 32 && c.input_width == 3 && c.patch_length == 8 &&
      c.encoder_width == 64 && c.export_width == 32 && c.num_layers == 3 && c.num_heads == 4 && c.feedforward_width == 256 &&
      c.channel_mixer_layers == 1 && c.global_bottleneck_mode == 2 && c.decoder_hidden_width == 128 && c.dropout == 0 &&
      c.huber_delta == 1 && cp.settings.learning_rate == .001 && cp.settings.weight_decay == .0001 &&
      cp.settings.gradient_clip_norm == 1 && cp.settings.threads == 1,
      "fixed mode2/mixer1/native32 architecture, decoder optimizer and dropout0 required");
  require(fit.shape.channel_count == c.channel_count && fit.shape.history_length == c.history_length && fit.shape.input_width == c.input_width &&
      fit.training_source_ids.size() == static_cast<size_t>(fit.training_observations.data.size(0)) &&
      cp.settings.seed == static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL), "TRAIN geometry/source order/counter seed differs");
  out->fit = fit; out->fit.training_observations.data = cpu(fit.training_observations.data);
  out->fit.training_observations.feature_mask = cpu(fit.training_observations.feature_mask);
  out->training = describe_dataset(raw_input(out->fit.training_observations, c, fit), c, fit.feature_units);
  const auto saved_training = load_dataset(out->path + ".training-raw.pt", c);
  require(out->training.dataset_id == cp.dataset_id && out->training.schema_id == cp.schema_id &&
      cp.scaler_fit_dataset_id == cp.dataset_id && saved_training.dataset_id == cp.dataset_id &&
      saved_training.schema_id == cp.schema_id && saved_training.feature_units == fit.feature_units,
      "exact original legal TRAIN values/support/scaler fit/schema required");
  const auto scaler = load_scaler(out->path + ".scaler.pt", c, cp.schema_id);
  require(scaler.identity() == cp.scaler.identity(), "parent scaler companion differs");
  torch::serialize::InputArchive audit; audit.load_from(out->path + ".audit.pt", torch::kCPU);
  require(text(audit, "artifact_kind") == "rpb_learning_curve_training_audit_v1" &&
      text(audit, "protocol_id") == selected.fit_protocol && text(audit, "fit_source_manifest") == source_manifest(fit.training_source_ids) &&
      text(audit, "feature_units") == fit.feature_units && text(audit, "actual_training_seed") == std::to_string(cp.settings.seed) &&
      text(audit, "initialization_seed") == std::to_string(training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL)) &&
      text(audit, "rng_policy") == counter_policy && text(audit, "training_dataset_id") == cp.dataset_id &&
      text(audit, "scaler_fit_dataset_id") == cp.dataset_id && text(audit, "preprocessing_id") == cp.scaler.identity() &&
      integer(audit, "attempted_steps") == updates && integer(audit, "completed_steps") == updates &&
      integer(audit, "sampled_rows") == updates * cp.settings.batch_size && torch::Device(text(audit, "training_device")).is_cuda() &&
      text(audit, "core_writer_source_fingerprint") == cp.source_fingerprint,
      "parent source/seed/counter companion differs");
  torch::Tensor ids, interval, endpoint;
  audit.read("channel_order", ids, true); audit.read("sampling_interval", interval, true); audit.read("endpoint", endpoint, true);
  require(torch::equal(ids, torch::tensor(fit.channel_ids, torch::kInt64)) &&
      interval.item<double>() == fit.sampling_interval && endpoint.item<double>() == fit.endpoint,
      "parent typed semantic IDs/time differs");
  if (selected.context) {
    require(text(audit, "model_tag") == selected.model_tag && text(audit, "training_policy_id") == selected.parent_policy &&
        text(audit, "context_deletion_ratio") == "0.15" && text(audit, "context_deletion_stream") == "0x6374782d64726f70" &&
        text(audit, "context_deletion_rng_policy") == context_deletion::rng_policy &&
        text(audit, "context_deletion_repair_policy") == context_deletion::repair_policy &&
        text(audit, "context_deletion_visibility_policy") == context_deletion::visibility_policy &&
        text(audit, "context_deletion_count_policy") == "cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only" &&
        text(audit, "context_deletion_resume_policy") == "fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API",
        "parent coordinate15 policy companion differs");
    torch::Tensor ratio; audit.read("context_deletion_ratio_value", ratio, true);
    require(ratio.scalar_type() == torch::kFloat64 && ratio.numel() == 1 && ratio.item<double>() == .15 &&
        integer(audit, "context_deletion_stream_value") == static_cast<int64_t>(context_deletion::stream),
        "parent typed context ratio/stream differs");
    const auto requested = integer(audit, "context_requested_deleted_coordinates"), actual = integer(audit, "context_actual_deleted_coordinates"),
        restored = integer(audit, "context_restored_coordinates");
    require(requested >= actual && actual >= 0 && restored == requested - actual &&
        (!initial || (requested == 0 && actual == 0 && restored == 0)), "parent context count identity differs");
  } else {
    // An ordinary parent cannot carry a forged context declaration. Historical
    // ordinary audit envelopes intentionally omit both tag/policy text keys.
    for (const auto &key : audit.keys())
      require(key != "model_tag" && key != "training_policy_id" && key.rfind("context_", 0) != 0,
          "ordinary v4 parent has context metadata");
  }
  auto resolved = parse_settings(text(audit, "resolved_settings")); resolved.model.device = cp.settings.model.device;
  require(settings_text(resolved) == settings_text(cp.settings), "parent declared numerical settings differ");
  if (initial) {
    torch::Tensor changed, finite, seconds;
    audit.read("weights_changed", changed, true); audit.read("finite_gradients", finite, true); audit.read("training_seconds", seconds, true);
    require(changed.scalar_type() == torch::kBool && changed.dim() == 0 && !changed.item<bool>() &&
        finite.scalar_type() == torch::kBool && finite.dim() == 0 && !finite.item<bool>() &&
        seconds.scalar_type() == torch::kFloat64 && seconds.dim() == 0 && seconds.item<double>() == 0 &&
        text(audit, "model_weight_update_budget") == "0", "fresh initialization has a training/update witness");
  }
  for (const std::string key : {"training_producer_source_fingerprint", "core_writer_source_fingerprint", "initialization_seed",
      "actual_training_seed", "fit_source_manifest_id", "rng_policy", "sampling_policy", "optimizer_policy"})
    out->audit.emplace(key, text(audit, key));
  freeze(cp.model); verify_bindings(out->path, out->bindings); return out;
}
Named native_witness(const std::shared_ptr<Parent> &parent) {
  torch::NoGradGuard no_grad;
  const auto normalized = parent->checkpoint.scaler.transform(parent->training.input, parent->checkpoint.settings.model);
  const auto output = parent->checkpoint.model->encode(normalized);
  require(normalized.data.is_cuda() && output.z_contextual_global.is_cuda(), "native witness must execute on CUDA");
  return {{"global", cpu(output.z_contextual_global)}, {"contextual", cpu(output.z_contextual)},
      {"local", cpu(output.z_local)}, {"channel_valid", cpu(output.channel_valid_mask)}, {"sample_valid", cpu(output.sample_valid_mask)}};
}
ev::FeatureMap cuda_native_features(const std::shared_ptr<Parent> &parent,
    const ev::ProviderFitInput &metadata, const embedding::Batch &batch,
    const std::string &provenance) {
  const RuntimeIsolation callback_isolation; at::set_num_threads(1);
  torch::NoGradGuard no_grad;
  auto &checkpoint = parent->checkpoint;
  const auto input = checkpoint.scaler.transform(
      raw_input(batch, checkpoint.settings.model, metadata), checkpoint.settings.model);
  const auto output = checkpoint.model->encode(input);
  require(input.data.is_cuda() && output.z_contextual_global.is_cuda(),
      "native inference must execute CUDA");
  return ev::FeatureMap{{"curve_global", {cpu(output.z_contextual_global), cpu(output.sample_valid_mask), provenance}},
      {"curve_channel_concatenation", {cpu(output.z_contextual.flatten(1)), cpu(output.channel_valid_mask.all(1)), provenance}}};
}
void save_exclusive(const std::string &path, torch::serialize::OutputArchive &archive) {
  require(!fs::exists(path) && !fs::is_symlink(fs::symlink_status(path)), "refusing artifact replacement: " + path);
  const int handle = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  require(handle >= 0, "cannot exclusively reserve calibration artifact: " + path); ::close(handle);
  // Only our newly reserved file may be replaced by the shared atomic writer.
  embedding::archive::save_archive(path, archive);
}
struct State {
  DecoderCalibrationOptions options;
  Binding binding;
  std::string producer_source;
  std::shared_ptr<Parent> parent;
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::vector<torch::Tensor> decoder_parameters;
  Named initial_decoder, frozen_parameters, frozen_buffers, native_before;
  FrozenScaler scaler_before;
  DecoderCalibrationProgress progress;
  std::vector<torch::Tensor> indices, predictions, targets, queries, visible, eligible, latent, hidden;
  std::map<std::string, std::pair<int64_t, int64_t>> saved;
  std::map<std::string, std::string> saved_content;
  std::map<std::string, std::string> audit;
  bool failed{false};
};
DecoderCalibrationProgress report(const std::shared_ptr<State> &state) {
  require(!state->failed, "calibration previously aborted");
  auto out = state->progress; const auto &model = state->parent->checkpoint.model;
  for (const auto &p : model->parameters()) require(torch::isfinite(p).all().item<bool>(), "nonfinite calibrated model weights");
  out.encoder_parameters_exact = same(state->frozen_parameters, parameters(model, false));
  out.encoder_buffers_exact = same(state->frozen_buffers, buffers(model));
  out.scaler_exact = state->scaler_before.identity() == state->parent->checkpoint.scaler.identity();
  out.native_exports_exact = same(state->native_before, native_witness(state->parent));
  out.zero_encoder_gradients = true;
  for (const auto &p : model->named_parameters()) if (!decoder_name(p.key()))
    out.zero_encoder_gradients &= !p.value().grad().defined() || p.value().grad().eq(0).all().item<bool>();
  out.decoder_weights_changed = !same(state->initial_decoder, parameters(model, true));
  require(out.encoder_parameters_exact && out.encoder_buffers_exact && out.scaler_exact && out.native_exports_exact &&
      out.zero_encoder_gradients, "frozen encoder/scaler/native invariant violated");
  return out;
}
torch::Tensor stack(const std::vector<torch::Tensor> &values, const std::vector<int64_t> &empty, torch::ScalarType type) {
  return values.empty() ? torch::empty(empty, torch::TensorOptions().dtype(type)) : torch::stack(values);
}
void write_optimizer(torch::serialize::OutputArchive &a, const std::shared_ptr<State> &state) {
  torch::serialize::OutputArchive group; const auto named = state->parent->checkpoint.model->named_parameters();
  group.write("count", torch::tensor(static_cast<int64_t>(state->initial_decoder.size())), true); int64_t i = 0;
  for (const auto &[name, value] : state->initial_decoder) {
    (void)value; torch::serialize::OutputArchive child; write_text(child, "parameter_name", name);
    const auto found = state->optimizer->state().find(named[name].unsafeGetTensorImpl());
    if (state->progress.completed == 0) {
      require(found == state->optimizer->state().end(), "stage zero AdamW must be empty");
      child.write("step", torch::tensor(int64_t{0}), true);
      child.write("exp_avg", torch::zeros_like(cpu(named[name])), true);
      child.write("exp_avg_sq", torch::zeros_like(cpu(named[name])), true);
    } else {
      require(found != state->optimizer->state().end(), "decoder AdamW state missing");
      const auto *moment = dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
      require(moment && moment->step() == state->progress.completed && moment->exp_avg().is_cuda() &&
          moment->exp_avg_sq().is_cuda(), "fresh decoder CUDA AdamW step/device differs");
      child.write("step", torch::tensor(moment->step()), true);
      child.write("exp_avg", cpu(moment->exp_avg()), true); child.write("exp_avg_sq", cpu(moment->exp_avg_sq()), true);
    }
    group.write("tensor_" + std::to_string(i++), child);
  }
  a.write("decoder_optimizer", group);
}
void save(const std::shared_ptr<State> &state, const std::string &path) {
  const RuntimeIsolation isolation; at::set_num_threads(1); require(!state->failed, "calibration aborted; cannot save");
  verify_bindings(state->parent->path, state->parent->bindings); const auto progress = report(state);
  const auto key = path_key(path);
  for (const auto &[suffix, identity] : state->parent->bindings) {
    (void)identity; require(key != path_key(state->parent->path + suffix), "calibration cannot alias immutable parent");
  }
  torch::serialize::OutputArchive a; write_text(a, "artifact_kind", identity(state->binding).artifact);
  for (const auto &[name, value] : state->audit) write_text(a, name, value);
  write_text(a, "parent_checkpoint_path", state->parent->path);
  write_text(a, "parent_checkpoint_content_id", state->parent->bindings.at(""));
  write_text(a, "parent_audit_content_id", state->parent->bindings.at(".audit.pt"));
  write_text(a, "parent_scaler_content_id", state->parent->bindings.at(".scaler.pt"));
  write_text(a, "parent_raw_training_content_id", state->parent->bindings.at(".training-raw.pt"));
  for (const auto &[name, value] : std::map<std::string, int64_t>{{"original_encoder_attempted", progress.original_encoder_attempted},
      {"original_encoder_completed", progress.original_encoder_completed}, {"decoder_attempted", progress.attempted},
      {"decoder_completed", progress.completed}, {"sampled_rows", progress.sampled_rows}, {"next_absolute_attempt", progress.next_absolute_attempt},
      {"decoder_parameter_count", progress.decoder_parameter_count}, {"frozen_parameter_count", progress.frozen_parameter_count}})
    a.write(name, torch::tensor(value), true);
  a.write("decoder_training_seconds", torch::tensor(progress.training_seconds, torch::kFloat64), true);
  a.write("channel_order", torch::tensor(state->parent->fit.channel_ids, torch::kInt64), true);
  a.write("sampling_interval", torch::tensor(state->parent->fit.sampling_interval, torch::kFloat64), true);
  a.write("endpoint", torch::tensor(state->parent->fit.endpoint, torch::kFloat64), true);
  for (const auto &[name, value] : std::map<std::string, bool>{{"encoder_parameters_exact", progress.encoder_parameters_exact},
      {"encoder_buffers_exact", progress.encoder_buffers_exact}, {"scaler_exact", progress.scaler_exact},
      {"native_exports_exact", progress.native_exports_exact}, {"zero_encoder_gradients", progress.zero_encoder_gradients},
      {"last_input_cuda", progress.last_input_cuda}, {"last_latent_cuda", progress.last_latent_cuda}, {"last_loss_cuda", progress.last_loss_cuda},
      {"finite_decoder_gradients", progress.finite_decoder_gradients}, {"decoder_weights_changed", progress.decoder_weights_changed}})
    a.write(name, torch::tensor(value, torch::kBool), true);
  write_named(a, "decoder_before", state->initial_decoder); write_named(a, "decoder_after", parameters(state->parent->checkpoint.model, true));
  write_named(a, "encoder_before", state->frozen_parameters); write_named(a, "encoder_after", parameters(state->parent->checkpoint.model, false));
  write_named(a, "buffers_before", state->frozen_buffers); write_named(a, "buffers_after", buffers(state->parent->checkpoint.model));
  write_named(a, "native_before", state->native_before); write_named(a, "native_after", native_witness(state->parent));
  torch::serialize::OutputArchive sb, sa; state->scaler_before.save(sb); state->parent->checkpoint.scaler.save(sa);
  a.write("scaler_before", sb); a.write("scaler_after", sa); write_optimizer(a, state);
  const auto B = state->parent->checkpoint.settings.batch_size;
  a.write("sampled_indices", stack(state->indices, {0, B}, torch::kInt64), true);
  a.write("prediction", stack(state->predictions, {0, B, 3, 32, 3}, torch::kFloat32), true);
  a.write("normalized_target", stack(state->targets, {0, B, 3, 32, 3}, torch::kFloat32), true);
  a.write("target_support", stack(state->queries, {0, B, 3, 32, 3}, torch::kBool), true);
  a.write("visible_support", stack(state->visible, {0, B, 3, 32, 3}, torch::kBool), true);
  a.write("artificial_hidden", stack(state->hidden, {0, B, 3, 32, 3}, torch::kBool), true);
  a.write("eligible_channels", stack(state->eligible, {0, B, 3}, torch::kBool), true);
  a.write("detached_native32", stack(state->latent, {0, B, 32}, torch::kFloat32), true);
  std::vector<double> losses; std::vector<int64_t> attempts;
  for (const auto &p : progress.losses) {
    attempts.push_back(p.absolute_attempt); losses.insert(losses.end(), {p.loss, p.gradient_norm,
        static_cast<double>(p.target_cells), static_cast<double>(p.eligible_examples)});
  }
  a.write("absolute_attempts", torch::tensor(attempts, torch::kInt64), true);
  a.write("loss_trace", torch::tensor(losses, torch::kFloat64).reshape({progress.completed, 4}), true);
  write_text(a, "loss_trace_columns", "original_hierarchical_huber,preclip_decoder_gradient_norm,target_cells,eligible_examples");
  save_exclusive(path, a); state->saved.emplace(key, std::make_pair(progress.attempted, progress.completed));
  state->saved_content.emplace(key, file_identity(path));
  verify_bindings(state->parent->path, state->parent->bindings);
}
ev::CurveSnapshot snapshot_from_parent(const std::shared_ptr<Parent> &parent, const ev::ProviderFitInput &fit,
    const std::map<std::string, std::string> &fields, const std::string &provenance) {
  ev::CurveSnapshot snapshot; snapshot.features.audit_fields = fields;
  snapshot.features.provenance = provenance;
  snapshot.features.surfaces.emplace("curve_global", ev::SurfaceDescription{ev::SurfaceKind::global, "at least one observed channel", {}});
  snapshot.features.surfaces.emplace("curve_channel_concatenation", ev::SurfaceDescription{ev::SurfaceKind::channel_concatenation,
      "every declared channel observed-valid; contextual vectors in declared physical order", fit.channel_ids});
  auto metadata = fit; metadata.training_observations = {}; metadata.training_source_ids.clear();
  snapshot.features.extract = [parent, metadata, provenance = snapshot.features.provenance](const embedding::Batch &batch) {
    return cuda_native_features(parent, metadata, batch, provenance);
  };
  snapshot.reconstruct = [parent, metadata](const embedding::Batch &batch, const torch::Tensor &hidden) {
    const RuntimeIsolation callback_isolation; at::set_num_threads(1); torch::NoGradGuard no_grad;
    auto &checkpoint = parent->checkpoint;
    const auto input = checkpoint.scaler.transform(raw_input(batch, checkpoint.settings.model, metadata), checkpoint.settings.model);
    const auto output = checkpoint.model->forward(input, hidden);
    require(output.reconstruction.is_cuda() && torch::isfinite(output.reconstruction).all().item<bool>(), "finite CUDA query reconstruction required");
    return ev::CurveReconstruction{cpu(output.reconstruction), cpu(input.data), cpu(output.eligible_channels)};
  };
  snapshot.features.save_assets = [fields](const std::string &directory) {
    const RuntimeIsolation callback_isolation; torch::serialize::OutputArchive archive;
    write_text(archive, "artifact_kind", fields.at("protocol_id") == kFreshDecoderProtocol ?
        "rpb_fresh_frozen_decoder_calibration_snapshot_audit_v1" : "rpb_decoder_calibration_snapshot_audit_v1");
    for (const auto &[name, value] : fields) write_text(archive, name, value);
    save_exclusive((fs::path(directory) / "decoder-calibration-snapshot-audit.pt").string(), archive);
  };
  verify_bindings(parent->path, parent->bindings); return snapshot;
}
} // namespace

DecoderCalibrationRun frozen_decoder_detail::make(const DecoderCalibrationOptions &options,
    const ev::ProviderFitInput &fit, Binding binding, const std::string &producer_source) {
  const auto &selected = identity(binding);
  require(options.expected_parent_updates > 0 && options.additional_updates > 0, "positive bounded parent/stage updates required");
  const RuntimeIsolation isolation; at::set_num_threads(1);
  auto state = std::make_shared<State>(); state->options = options; state->binding = binding; state->producer_source = producer_source;
  state->parent = admit(options.parent_checkpoint_path, fit, options.expected_parent_updates, binding);
  auto &cp = state->parent->checkpoint;
  state->initial_decoder = parameters(cp.model, true); state->frozen_parameters = parameters(cp.model, false);
  state->frozen_buffers = buffers(cp.model); state->scaler_before = cp.scaler;
  state->scaler_before.mean = cpu(cp.scaler.mean); state->scaler_before.scale = cpu(cp.scaler.scale);
  state->scaler_before.count = cpu(cp.scaler.count); state->scaler_before.channel_ids = cpu(cp.scaler.channel_ids);
  state->scaler_before.floor_applied = cpu(cp.scaler.floor_applied);
  state->native_before = native_witness(state->parent);
  require(size(state->initial_decoder) == 11528 && state->initial_decoder.size() == 6 && size(state->frozen_parameters) == 214277,
      "exact four decoder families / 225805 total parameters required");
  for (auto &p : cp.model->named_parameters()) {
    require(p.value().is_cuda(), "all model parameters must be CUDA");
    if (decoder_name(p.key())) { p.value().set_requires_grad(true); state->decoder_parameters.push_back(p.value()); }
  }
  state->optimizer = std::make_unique<torch::optim::AdamW>(state->decoder_parameters,
      torch::optim::AdamWOptions(.001).weight_decay(.0001));
  state->progress.original_encoder_attempted = cp.attempted_steps; state->progress.original_encoder_completed = cp.completed_steps;
  state->progress.next_absolute_attempt = cp.attempted_steps;
  state->progress.decoder_parameter_count = 11528; state->progress.frozen_parameter_count = 214277;
  state->audit = {{"protocol_id", selected.protocol}, {"model_tag", selected.model_tag}, {"training_policy_id", selected.parent_policy},
      {"calibration_producer_source_fingerprint", producer_source}, {"parent_writer_source_fingerprint", cp.source_fingerprint},
      {"parent_training_producer_source_fingerprint", state->parent->audit.at("training_producer_source_fingerprint")},
      {"source_fingerprint_scope", "parent=original_encoder_writer_and_producer;calibration=decoder_only_producer"},
      {"resolved_parent_settings", settings_text(cp.settings)}, {"training_dataset_id", cp.dataset_id}, {"training_schema_id", cp.schema_id},
      {"preprocessing_id", cp.scaler.identity()}, {"fit_source_manifest", source_manifest(fit.training_source_ids)},
      {"fit_protocol_id", fit.protocol_id}, {"feature_units", fit.feature_units}, {"actual_training_seed", std::to_string(cp.settings.seed)},
      {"initialization_seed", state->parent->audit.at("initialization_seed")}, {"counter_policy", counter_policy},
      {"decoder_optimizer_policy", "fresh_decoder_only_AdamW;lr=0.001;weight_decay=0.0001;clip=1;no_parent_moments"},
      {"encoder_policy", "frozen_eval_no_grad;exact_served_contextual_global32_detached;all_buffers_and_scaler_unchanged"},
      {"query_policy", "original_O_A_Q_eligibility_hierarchical_Huber_delta1;no_E"},
      {"resume_policy", "typed_decoder_asset_composed_with_immutable_parent;no_ordinary_checkpoint_or_resume"},
      {"cost_scope", "CUDA_synchronized_decoder_update_loop_including_CPU_evidence_capture;encoder_cost_retained;query_export_audit_excluded"},
      {"parent_binding_scope", "runtime_FNV_content_guard_plus_runner_closed_SHA256_role_manifest"},
      {"additional_update_budget", std::to_string(options.additional_updates)}, {"batch_size", std::to_string(cp.settings.batch_size)}};
  DecoderCalibrationRun run; run.audit_fields = state->audit;
  run.train_to = [state](int64_t completed) {
    const RuntimeIsolation callback_isolation; at::set_num_threads(1);
    require(!state->failed && completed >= state->progress.completed && completed <= state->options.additional_updates,
        "monotone bounded live decoder budget required");
    verify_bindings(state->parent->path, state->parent->bindings);
    if (completed == state->progress.completed) return report(state);
    auto &checkpoint = state->parent->checkpoint; const auto &settings = checkpoint.settings;
    torch::cuda::synchronize(); const auto start = std::chrono::steady_clock::now();
    try {
      while (state->progress.completed < completed) {
        const auto attempt = checkpoint.attempted_steps + state->progress.attempted;
        const auto indices = training_detail::sampled_indices(state->parent->training.input.data.size(0),
            settings.batch_size, settings.seed, attempt);
        const auto raw = training_detail::selected(state->parent->training.input, indices);
        const auto normalized = checkpoint.scaler.transform(raw, settings.model);
        const auto masks = make_training_mask(normalized.observed, settings.model,
            training_detail::counter_seed(settings.seed, attempt, 0x6d61736bULL));
        require(masks.eligible_channels.any(1).all().item<bool>(), "ineligible original query; abort without replacement");
        torch::manual_seed(training_detail::counter_seed(settings.seed, attempt, 0x746f726368ULL));
        Input visible{torch::where(masks.visible, normalized.data, torch::zeros_like(normalized.data)),
            masks.visible, normalized.channel_ids, normalized.endpoints, normalized.sampling_interval};
        torch::Tensor latent;
        { torch::NoGradGuard no_grad; latent = compact_reconstruction_export(checkpoint.model->encode(visible), settings.model).detach(); }
        state->optimizer->zero_grad(); const auto prediction = checkpoint.model->decode(latent, normalized.channel_ids);
        const auto losses = hierarchical_huber(prediction, normalized.data.detach(), masks.target, masks.eligible_channels, settings.model.huber_delta);
        require(normalized.data.is_cuda() && latent.is_cuda() && !latent.requires_grad() && losses.loss.is_cuda() &&
            torch::isfinite(prediction).all().item<bool>() && torch::isfinite(latent).all().item<bool>() &&
            torch::isfinite(losses.loss).item<bool>(), "finite CUDA detached-native decoder loss required");
        losses.loss.backward();
        for (const auto &p : state->decoder_parameters)
          require(p.grad().defined() && p.grad().is_cuda() && torch::isfinite(p.grad()).all().item<bool>(), "finite decoder-only CUDA gradients required");
        for (const auto &p : checkpoint.model->named_parameters()) if (!decoder_name(p.key()))
          require(!p.value().grad().defined(), "encoder unexpectedly received a gradient");
        const double norm = torch::nn::utils::clip_grad_norm_(state->decoder_parameters, 1.0);
        require(std::isfinite(norm), "finite decoder gradient norm required");
        state->optimizer->step();
        ++state->progress.attempted; ++state->progress.completed; state->progress.sampled_rows += settings.batch_size;
        state->progress.next_absolute_attempt = checkpoint.attempted_steps + state->progress.attempted;
        state->progress.last_input_cuda = true; state->progress.last_latent_cuda = true; state->progress.last_loss_cuda = true;
        state->progress.finite_decoder_gradients = true;
        state->progress.losses.push_back({attempt, state->progress.completed, losses.target_cell_count,
            losses.eligible_example_count, losses.loss.item<double>(), norm});
        state->indices.push_back(cpu(indices)); state->predictions.push_back(cpu(prediction)); state->targets.push_back(cpu(normalized.data));
        state->queries.push_back(cpu(masks.target)); state->visible.push_back(cpu(masks.visible)); state->hidden.push_back(cpu(masks.hidden));
        state->eligible.push_back(cpu(losses.eligible_channels)); state->latent.push_back(cpu(latent));
      }
      torch::cuda::synchronize();
      state->progress.training_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      verify_bindings(state->parent->path, state->parent->bindings); return report(state);
    } catch (...) { state->failed = true; throw; }
  };
  run.save = [state](const std::string &path) { save(state, path); };
  run.snapshot = [state](const std::string &path) {
    require(!state->failed && state->saved.count(path_key(path)) == 1, "snapshot requires this live run's saved artifact");
    require(file_identity(path) == state->saved_content.at(path_key(path)), "saved calibration artifact bytes changed");
    return frozen_decoder_detail::snapshot(path, state->parent->path, state->parent->fit, state->binding, state->producer_source);
  };
  return run;
}

ev::CurveSnapshot frozen_decoder_detail::snapshot(const std::string &path, const std::string &parent_path,
    const ev::ProviderFitInput &fit, Binding binding, const std::string &producer_source) {
  const auto &selected = identity(binding);
  const RuntimeIsolation isolation; at::set_num_threads(1);
  require(fs::is_regular_file(path) && !fs::is_symlink(fs::symlink_status(path)), "regular typed calibration artifact required");
  torch::serialize::InputArchive a; a.load_from(path, torch::kCPU);
  require(text(a, "artifact_kind") == selected.artifact && text(a, "protocol_id") == selected.protocol &&
      text(a, "model_tag") == selected.model_tag && text(a, "training_policy_id") == selected.parent_policy,
      "typed decoder calibration protocol/policy required");
  auto parent = admit(parent_path, fit, integer(a, "original_encoder_completed"), binding);
  const auto &cp = parent->checkpoint;
  require(text(a, "parent_checkpoint_path") == parent->path && text(a, "parent_checkpoint_content_id") == parent->bindings.at("") &&
      text(a, "parent_audit_content_id") == parent->bindings.at(".audit.pt") && text(a, "parent_scaler_content_id") == parent->bindings.at(".scaler.pt") &&
      text(a, "parent_raw_training_content_id") == parent->bindings.at(".training-raw.pt") &&
      text(a, "resolved_parent_settings") == settings_text(cp.settings) && text(a, "training_dataset_id") == cp.dataset_id &&
      text(a, "training_schema_id") == cp.schema_id && text(a, "preprocessing_id") == cp.scaler.identity() &&
      text(a, "fit_source_manifest") == source_manifest(fit.training_source_ids) && text(a, "feature_units") == fit.feature_units &&
      text(a, "actual_training_seed") == std::to_string(cp.settings.seed) && text(a, "counter_policy") == counter_policy &&
      text(a, "parent_writer_source_fingerprint") == cp.source_fingerprint &&
      text(a, "parent_training_producer_source_fingerprint") == parent->audit.at("training_producer_source_fingerprint"),
      "calibration immutable parent/TRAIN/source identity differs");
  torch::Tensor order, interval, endpoint;
  a.read("channel_order", order, true); a.read("sampling_interval", interval, true); a.read("endpoint", endpoint, true);
  require(torch::equal(order, torch::tensor(fit.channel_ids, torch::kInt64)) && interval.scalar_type() == torch::kFloat64 &&
      endpoint.scalar_type() == torch::kFloat64 && interval.dim() == 0 && endpoint.dim() == 0 &&
      interval.item<double>() == fit.sampling_interval && endpoint.item<double>() == fit.endpoint,
      "calibration typed semantic IDs/time differ");
  const auto completed = integer(a, "decoder_completed");
  require(completed >= 0 && completed <= std::stoll(text(a, "additional_update_budget")) && integer(a, "decoder_attempted") == completed &&
      integer(a, "original_encoder_attempted") == cp.attempted_steps && integer(a, "sampled_rows") == completed * cp.settings.batch_size &&
      integer(a, "next_absolute_attempt") == cp.attempted_steps + completed && integer(a, "decoder_parameter_count") == 11528 &&
      integer(a, "frozen_parameter_count") == 214277, "calibration counters or parameter scope differ");
  require(text(a, "decoder_optimizer_policy") == "fresh_decoder_only_AdamW;lr=0.001;weight_decay=0.0001;clip=1;no_parent_moments" &&
      text(a, "encoder_policy") == "frozen_eval_no_grad;exact_served_contextual_global32_detached;all_buffers_and_scaler_unchanged" &&
      text(a, "query_policy") == "original_O_A_Q_eligibility_hierarchical_Huber_delta1;no_E" &&
      text(a, "resume_policy") == "typed_decoder_asset_composed_with_immutable_parent;no_ordinary_checkpoint_or_resume" &&
      text(a, "fit_protocol_id") == fit.protocol_id && text(a, "batch_size") == std::to_string(cp.settings.batch_size),
      "calibration decoder/query/freeze/resume policy differs");
  require(same(read_named(a, "encoder_before"), parameters(cp.model, false)) &&
      same(read_named(a, "encoder_after"), parameters(cp.model, false)) &&
      same(read_named(a, "buffers_before"), buffers(cp.model)) && same(read_named(a, "buffers_after"), buffers(cp.model)) &&
      same(read_named(a, "decoder_before"), parameters(cp.model, true)), "calibration frozen/initial named tensors differ from parent");
  torch::serialize::InputArchive sb, sa; a.read("scaler_before", sb); a.read("scaler_after", sa);
  const auto before = FrozenScaler::load(sb), after = FrozenScaler::load(sa); before.validate(cp.settings.model); after.validate(cp.settings.model);
  require(before.identity() == cp.scaler.identity() && after.identity() == cp.scaler.identity(), "calibration scaler changed");
  const auto old_native = native_witness(parent);
  require(same(read_named(a, "native_before"), old_native) && same(read_named(a, "native_after"), old_native), "calibration CUDA native witness differs");
  const auto decoder = read_named(a, "decoder_after"), expected = parameters(cp.model, true);
  require(decoder.size() == expected.size(), "decoder-only tensor names differ");
  torch::serialize::InputArchive optimizer; a.read("decoder_optimizer", optimizer);
  require(integer(optimizer, "count") == static_cast<int64_t>(expected.size()), "decoder-only AdamW state count differs");
  int64_t index = 0;
  { torch::NoGradGuard no_grad; const auto named = cp.model->named_parameters();
    for (const auto &[name, value] : expected) {
      const auto it = decoder.find(name);
      require(it != decoder.end() && it->second.scalar_type() == torch::kFloat32 && it->second.sizes() == value.sizes(), "decoder-only tensor schema differs");
      torch::serialize::InputArchive child; optimizer.read("tensor_" + std::to_string(index++), child);
      torch::Tensor mean, variance; child.read("exp_avg", mean, true); child.read("exp_avg_sq", variance, true);
      require(text(child, "parameter_name") == name && integer(child, "step") == completed &&
          mean.scalar_type() == torch::kFloat32 && variance.scalar_type() == torch::kFloat32 && mean.sizes() == value.sizes() &&
          variance.sizes() == value.sizes() && torch::isfinite(mean).all().item<bool>() && torch::isfinite(variance).all().item<bool>() &&
          variance.ge(0).all().item<bool>(), "decoder-only AdamW witness malformed");
      named[name].copy_(it->second.to(torch::kCUDA));
    }
  }
  freeze(parent->checkpoint.model);
  std::map<std::string, std::string> fields{{"model_tag", selected.model_tag}, {"training_policy_id", selected.parent_policy},
      {"protocol_id", selected.protocol}, {"calibration_artifact_path", path_key(path)},
      {"parent_checkpoint_path", parent->path}, {"original_encoder_completed", std::to_string(cp.completed_steps)},
      {"decoder_completed", std::to_string(completed)}, {"encoder_parameters_exact", "true"}, {"encoder_buffers_exact", "true"},
      {"scaler_exact", "true"}, {"native_exports_exact", "true"},
      {"calibration_producer_source_fingerprint", text(a, "calibration_producer_source_fingerprint")},
      {"snapshot_loader_source_fingerprint", producer_source},
      {"parent_writer_source_fingerprint", cp.source_fingerprint},
      {"parent_training_producer_source_fingerprint", parent->audit.at("training_producer_source_fingerprint")},
      {"snapshot_policy", "all_inference_CUDA;typed_decoder_composed_with_frozen_parent;classification_reused_no_heads_or_PCA_fitted"}};
  return snapshot_from_parent(parent, fit, fields, std::string(selected.model_tag) +
      " exact retained native32 encoder; decoder-only calibration; classification reused, no representation update");
}

DecoderCalibrationRun make_frozen_decoder_calibration(const FrozenDecoderCalibrationOptions &options,
                                                     const ev::ProviderFitInput &fit) {
  const auto binding = fresh_binding(options.parent_policy);
  return frozen_decoder_detail::make({options.parent_checkpoint_path, options.expected_parent_updates, options.additional_updates},
      fit, binding, FROZEN_DECODER_CALIBRATION_SOURCE_ID);
}
ev::CurveSnapshot make_frozen_decoder_calibration_snapshot(const std::string &path, const std::string &parent_path,
    FrozenDecoderParentPolicy policy, const ev::ProviderFitInput &fit) {
  const auto binding = fresh_binding(policy);
  return frozen_decoder_detail::snapshot(path, parent_path, fit, binding, FROZEN_DECODER_CALIBRATION_SOURCE_ID);
}
ev::CurveSnapshot make_fresh_initial_snapshot(const std::string &path, FrozenDecoderParentPolicy policy,
                                             const ev::ProviderFitInput &fit) {
  const auto binding = fresh_binding(policy); const auto &selected = identity(binding);
  const RuntimeIsolation isolation; at::set_num_threads(1);
  const auto parent = admit(path, fit, 0, binding, true);
  const auto &cp = parent->checkpoint;
  for (const auto &p : cp.model->parameters())
    require(p.is_cuda() && torch::isfinite(p).all().item<bool>(), "fresh point0 requires all finite CUDA parameters");
  require(size(parameters(cp.model, true)) == 11528 && size(parameters(cp.model, false)) == 214277,
      "exact 225805-parameter fresh initialization required");
  std::map<std::string, std::string> fields{{"protocol_id", selected.protocol}, {"model_tag", selected.model_tag},
      {"training_policy_id", selected.parent_policy}, {"parent_checkpoint_path", parent->path},
      {"original_encoder_completed", "0"}, {"original_encoder_attempted", "0"}, {"decoder_completed", "0"},
      {"initialization_seed", parent->audit.at("initialization_seed")}, {"actual_training_seed", parent->audit.at("actual_training_seed")},
      {"parent_writer_source_fingerprint", cp.source_fingerprint},
      {"parent_training_producer_source_fingerprint", parent->audit.at("training_producer_source_fingerprint")},
      {"snapshot_loader_source_fingerprint", FROZEN_DECODER_CALIBRATION_SOURCE_ID},
      {"snapshot_policy", "fresh_exact_point0;all_inference_CUDA;no_calibration_optimizer_or_refit"}};
  return snapshot_from_parent(parent, fit, fields, std::string(selected.model_tag) +
      " exact fresh untrained native32 encoder; CUDA inference; no fit or calibration");
}

ev::FeatureProvider make_frozen_native_feature_provider(const FrozenNativeFeatureOptions &options,
    const ev::ProviderFitInput &fit) {
  const auto valid_source = [](const std::string &source) {
    return source.size() == 64 &&
        source.find_first_not_of("0123456789abcdef") == std::string::npos;
  };
  // Reject unspecified selectors before opening any checkpoint or changing RNG.
  const auto binding = fresh_binding(options.parent_policy);
  require(options.expected_parent_updates >= 0 && !options.parent_checkpoint_path.empty() &&
      valid_source(options.expected_parent_core_source_fingerprint) &&
      valid_source(options.expected_training_producer_source_fingerprint),
      "explicit original budget/checkpoint/core/training producer pins required");
  require(fit.training_observations.data.defined() && fit.training_observations.feature_mask.defined() &&
      fit.training_observations.data.device().is_cpu() && fit.training_observations.data.scalar_type() == torch::kFloat64 &&
      fit.training_observations.feature_mask.device().is_cpu() && fit.training_observations.feature_mask.scalar_type() == torch::kBool,
      "original timing TRAIN requires CPU Float64 observations and bool support");
  const RuntimeIsolation isolation; at::set_num_threads(1);
  const auto parent = admit(options.parent_checkpoint_path, fit,
      options.expected_parent_updates, binding, options.expected_parent_updates == 0);
  const auto &cp = parent->checkpoint;
  require(cp.source_fingerprint == options.expected_parent_core_source_fingerprint &&
      parent->audit.at("core_writer_source_fingerprint") == options.expected_parent_core_source_fingerprint &&
      parent->audit.at("training_producer_source_fingerprint") == options.expected_training_producer_source_fingerprint,
      "explicit original producer scopes differ from the admitted parent");
  int64_t parameter_count = 0;
  for (const auto &p : cp.model->parameters()) {
    require(p.is_cuda() && !p.requires_grad() && torch::isfinite(p).all().item<bool>(),
        "frozen encoder requires finite CUDA parameters without gradients");
    parameter_count += p.numel();
  }
  require(parameter_count == 225805, "exact 225805-parameter frozen architecture required");
  for (const auto &b : cp.model->buffers())
    require(b.is_cuda() && torch::isfinite(b).all().item<bool>(), "finite CUDA buffers required");
  const auto encoder_before = parameters(cp.model, false), decoder_before = parameters(cp.model, true);
  const auto buffers_before = buffers(cp.model);
  const auto scaler_before = cp.scaler.identity();
  const auto immutable = [parent, encoder_before, decoder_before, buffers_before, scaler_before]() {
    require(same(parameters(parent->checkpoint.model, false), encoder_before) &&
        same(parameters(parent->checkpoint.model, true), decoder_before) &&
        same(buffers(parent->checkpoint.model), buffers_before) &&
        parent->checkpoint.scaler.identity() == scaler_before,
        "frozen model parameters/buffers/scaler changed during extraction");
    verify_bindings(parent->path, parent->bindings);
  };
  const auto &selected = identity(binding);
  ev::FeatureProvider provider;
  provider.provenance = std::string(selected.model_tag) +
      "; original timing-protocol checkpoint; frozen native32 CUDA encode; no fitting or updates";
  provider.audit_fields = {{"protocol_id", kFrozenNativeFeatureProtocol},
      {"original_training_protocol_id", selected.fit_protocol}, {"model_tag", selected.model_tag},
      {"training_policy_id", selected.parent_policy}, {"parent_checkpoint_path", parent->path},
      {"parent_writer_source_fingerprint", cp.source_fingerprint},
      {"parent_training_producer_source_fingerprint", parent->audit.at("training_producer_source_fingerprint")},
      {"snapshot_loader_source_fingerprint", FROZEN_NATIVE_FEATURE_SOURCE_ID},
      {"source_fingerprint_scope", "parent=original_core_writer_and_training_producer;extractor=new_frozen_native_feature_loader"},
      {"original_encoder_attempted", std::to_string(cp.attempted_steps)},
      {"original_encoder_completed", std::to_string(cp.completed_steps)},
      {"encoder_updates", "0"}, {"decoder_updates", "0"}, {"head_refits", "0"},
      {"no_optimizer_created", "true"}, {"inference_device", cp.settings.model.device.str()},
      {"parameter_count", std::to_string(parameter_count)}, {"training_dataset_id", cp.dataset_id},
      {"training_schema_id", cp.schema_id}, {"preprocessing_id", cp.scaler.identity()},
      {"fit_source_manifest", source_manifest(fit.training_source_ids)},
      {"fit_source_manifest_id", parent->audit.at("fit_source_manifest_id")},
      {"initialization_seed", parent->audit.at("initialization_seed")},
      {"actual_training_seed", parent->audit.at("actual_training_seed")},
      {"feature_units", fit.feature_units}, {"snapshot_policy", "bare_parent_only;CUDA_eval_no_grad;original_timing_scaler;no_optimizer_no_decoder_state_no_refit"}};
  for (const auto &[suffix, content] : parent->bindings)
    provider.audit_fields["parent_content_id" + suffix] = content;
  provider.surfaces.emplace("curve_global", ev::SurfaceDescription{
      ev::SurfaceKind::global, "at least one observed channel", {}});
  auto metadata = fit; metadata.training_observations = {}; metadata.training_source_ids.clear();
  provider.extract = [parent, metadata, immutable, provenance = provider.provenance](const embedding::Batch &batch) {
    const RuntimeIsolation callback_isolation; at::set_num_threads(1);
    immutable();
    const auto all = cuda_native_features(parent, metadata, batch, provenance);
    const auto &surface = all.at("curve_global");
    require(surface.values.scalar_type() == torch::kFloat32 && surface.values.device().is_cpu() &&
        surface.valid.scalar_type() == torch::kBool && surface.valid.device().is_cpu() &&
        surface.values.sizes() == torch::IntArrayRef({batch.data.size(0), 32}) &&
        torch::isfinite(surface.values).all().item<bool>(), "finite CPU Float32 native32 CUDA evidence required");
    immutable();
    return ev::FeatureMap{{"curve_global", surface}};
  };
  provider.save_assets = [fields = provider.audit_fields, immutable](const std::string &directory) {
    const RuntimeIsolation callback_isolation; immutable();
    torch::serialize::OutputArchive audit; write_text(audit, "artifact_kind", kFrozenNativeFeatureArtifact);
    for (const auto &[name, value] : fields) write_text(audit, name, value);
    save_exclusive((fs::path(directory) / kFrozenNativeFeatureAuditFile).string(), audit);
    immutable();
  };
  immutable(); return provider;
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
