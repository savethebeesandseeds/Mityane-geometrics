// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/native_view_agreement.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/paired_pooling_adapter.h"
#include "embedding/shared/data.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>

#ifndef NATIVE_VIEW_AGREEMENT_SOURCE_ID
#ifdef EVALUATION_SOURCE_ID
#define NATIVE_VIEW_AGREEMENT_SOURCE_ID EVALUATION_SOURCE_ID
#else
#define NATIVE_VIEW_AGREEMENT_SOURCE_ID "unrecorded"
#endif
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace nv = native_view_agreement;
namespace fs = std::filesystem;
constexpr const char *fit_protocol = "native-development-v1/lag_sign";
constexpr const char *rng_policy = "splitmix64-counter-rows-masks-torch-attempt-v1";
constexpr const char *sampling_policy = "with_replacement_counter_rows;sampled_rows_includes_no_update_attempts";
constexpr const char *optimizer_policy = "one_continuous_AdamW_state;absolute_completed_update_budgets";
constexpr const char *resume_policy = "fresh-live-continuous-only;ordinary-tagged-resume-rejected;no-agreement-resume-API";

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[rpb native view agreement] " + message);
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
struct ModelModeIsolation {
  ModelImpl &model;
  bool training;
  std::vector<bool> gradients;
  explicit ModelModeIsolation(ModelImpl &m) : model(m), training(m.is_training()) {
    for (auto &p : model.parameters()) { gradients.push_back(p.requires_grad()); p.set_requires_grad(false); }
    model.eval();
  }
  ~ModelModeIsolation() noexcept {
    try {
      auto parameters = model.parameters();
      for (size_t i = 0; i < parameters.size(); ++i) parameters[i].set_requires_grad(gradients[i]);
      model.train(training);
    } catch (...) { std::terminate(); }
  }
};
std::string source_manifest(const std::vector<std::string> &ids) {
  std::string out;
  for (const auto &id : ids) out += std::to_string(id.size()) + ':' + id;
  return out;
}
std::string fingerprint(const std::string &prefix, const std::string &bytes) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : bytes) { hash ^= byte; hash *= 1099511628211ULL; }
  std::ostringstream out; out << prefix << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
std::string manifest_id(const std::vector<std::string> &ids) {
  return fingerprint("rpb-fit-source-manifest-fnv1a-v1-", source_manifest(ids));
}
std::string tensor_bytes(const torch::Tensor &value) {
  const auto cpu = value.detach().to(torch::kCPU).contiguous();
  return std::string(static_cast<const char *>(cpu.const_data_ptr()), cpu.nbytes());
}
std::string scale_identity(const nv::LossScale &scale) {
  return fingerprint("rpb-native-view-loss-scale-fnv1a-v1-", std::string(nv::scale_policy) +
      source_manifest(scale.source_ids) + tensor_bytes(scale.values) + tensor_bytes(scale.calibration_native) +
      tensor_bytes(scale.calibration_valid) + tensor_bytes(scale.floor_applied));
}
Input protocol_input(const embedding::Batch &batch, const Config &config, const ev::ProviderFitInput &fit) {
  Input out{batch.data, batch.feature_mask, torch::tensor(fit.channel_ids, torch::kInt64),
      torch::full({batch.data.size(0)}, fit.endpoint, torch::kFloat64), fit.sampling_interval};
  validate_input(out, config); return out;
}
void fixed_architecture(const Config &c) {
  require(c.channel_mixer_placement == 0,
      "historical native view agreement rejects early channel mixer placement");
  require(c.global_bottleneck_mode == 2 && c.channel_mixer_layers == 1 && c.channel_count == 3 &&
      c.history_length == 32 && c.input_width == 3 && c.patch_length == 8 && c.encoder_width == 64 &&
      c.export_width == 32 && c.num_layers == 3 && c.num_heads == 4 && c.feedforward_width == 256 &&
      c.decoder_hidden_width == 128, "unchanged 225805-parameter mode2/mixer1/native32 architecture required");
}
void validate_fit(const Settings &settings, const ev::ProviderFitInput &fit) {
  fixed_architecture(settings.model);
  require(fit.protocol_id == fit_protocol && fit.shape.channel_count == 3 && fit.shape.history_length == 32 &&
      fit.shape.input_width == 3 && fit.shape.dtype == torch::kFloat64 && fit.shape.device.is_cpu() &&
      fit.training_observations.data.defined() && fit.training_observations.data.device().is_cpu() &&
      fit.training_observations.data.scalar_type() == torch::kFloat64 &&
      fit.training_observations.feature_mask.defined() && fit.training_observations.feature_mask.device().is_cpu() &&
      !fit.feature_units.empty() && fit.training_source_ids.size() ==
          static_cast<size_t>(fit.training_observations.data.size(0)), "explicit label-free native-development TRAIN required");
  for (const auto &id : fit.training_source_ids) require(!id.empty(), "empty TRAIN source ID");
  const auto known = resolved_channel_ids(settings.model);
  require(std::set<int64_t>(known.begin(), known.end()) == std::set<int64_t>(fit.channel_ids.begin(), fit.channel_ids.end()) &&
      fit.channel_ids.size() == 3, "TRAIN semantic identities differ");
  (void)protocol_input(fit.training_observations, settings.model, fit);
}
torch::Tensor native(const EncodeOutput &encoding) {
  const auto &value = encoding.z_contextual_global;
  require(value.defined() && value.dim() == 2 && value.size(1) == 32 && value.scalar_type() == torch::kFloat32 &&
      torch::isfinite(value).all().item<bool>(), "finite float32 exact contextual global32 required");
  return value;
}
void validate_identity(const nv::RowIdentity &identity, int64_t B, int64_t C) {
  require(identity.row_indices.defined() && identity.row_indices.scalar_type() == torch::kInt64 &&
      identity.row_indices.dim() == 1 && identity.row_indices.size(0) == B && identity.row_indices.ge(0).all().item<bool>() &&
      identity.channel_ids.defined() && identity.channel_ids.scalar_type() == torch::kInt64 &&
      identity.channel_ids.sizes() == torch::IntArrayRef({B, C}) && identity.endpoints.defined() &&
      identity.endpoints.is_floating_point() && identity.endpoints.sizes() == torch::IntArrayRef({B}) &&
      torch::isfinite(identity.endpoints).all().item<bool>() && identity.source_ids.size() == static_cast<size_t>(B),
      "exact row/channel/endpoint/source identities required");
  for (const auto &id : identity.source_ids) require(!id.empty(), "empty sampled source identity");
}
void write_text(torch::serialize::OutputArchive &archive, const std::string &key, const std::string &value) {
  archive.write(key, embedding::archive::text_tensor(value), true);
}
std::string read_text(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::Tensor value; archive.read(key, value, true); return embedding::archive::tensor_text(value);
}
int64_t read_integer(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::Tensor value; archive.read(key, value, true);
  require(value.scalar_type() == torch::kInt64 && value.dim() == 0, "invalid scalar count " + key); return value.item<int64_t>();
}
double read_double(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::Tensor value; archive.read(key, value, true);
  require(value.scalar_type() == torch::kFloat64 && value.dim() == 0 && torch::isfinite(value).item<bool>(),
      "invalid scalar coefficient " + key); return value.item<double>();
}
void save_scale(torch::serialize::OutputArchive &archive, const nv::LossScale &scale) {
  archive.write("loss_scale", scale.values, true);
  archive.write("loss_scale_calibration_native", scale.calibration_native, true);
  archive.write("loss_scale_calibration_valid", scale.calibration_valid, true);
  archive.write("loss_scale_floor_applied", scale.floor_applied, true);
  write_text(archive, "loss_scale_calibration_source_manifest", source_manifest(scale.source_ids));
}
nv::LossScale load_scale(torch::serialize::InputArchive &archive, const std::vector<std::string> &ids) {
  nv::LossScale out; out.source_ids = ids; out.identity = read_text(archive, "loss_scale_identity");
  archive.read("loss_scale", out.values, true); archive.read("loss_scale_calibration_native", out.calibration_native, true);
  archive.read("loss_scale_calibration_valid", out.calibration_valid, true);
  archive.read("loss_scale_floor_applied", out.floor_applied, true);
  require(read_text(archive, "loss_scale_calibration_source_manifest") == source_manifest(ids), "calibration source order differs");
  nv::validate_loss_scale(out); return out;
}
} // namespace

namespace native_view_agreement {
void validate_loss_scale(const LossScale &scale) {
  require(scale.values.defined() && scale.values.device().is_cpu() && scale.values.scalar_type() == torch::kFloat32 &&
      !scale.values.requires_grad() && scale.values.sizes() == torch::IntArrayRef({32}) && torch::isfinite(scale.values).all().item<bool>() &&
      scale.values.gt(0).all().item<bool>() && scale.calibration_native.defined() &&
      scale.calibration_native.device().is_cpu() && scale.calibration_native.scalar_type() == torch::kFloat32 &&
      scale.calibration_native.dim() == 2 && scale.calibration_native.size(1) == 32 &&
      scale.calibration_valid.defined() && scale.calibration_valid.device().is_cpu() &&
      scale.calibration_valid.scalar_type() == torch::kBool &&
      scale.calibration_valid.sizes() == scale.calibration_native.sizes().slice(0, 1) &&
      scale.source_ids.size() == static_cast<size_t>(scale.calibration_native.size(0)) &&
      scale.floor_applied.defined() && scale.floor_applied.device().is_cpu() &&
      scale.floor_applied.scalar_type() == torch::kBool && scale.floor_applied.sizes() == torch::IntArrayRef({32}) &&
      scale.calibration_valid.any().item<bool>() && torch::isfinite(scale.calibration_native).all().item<bool>(),
      "invalid fixed loss-scale/calibration schema");
  for (const auto &id : scale.source_ids) require(!id.empty(), "empty calibration source identity");
  require(scale.calibration_native.index({scale.calibration_valid.logical_not()}).eq(0).all().item<bool>(),
      "invalid calibration rows must preserve exact zero exports");
  const auto valid = scale.calibration_valid.contiguous(); const auto bits = valid.accessor<bool, 1>();
  std::set<std::string> valid_sources;
  for (int64_t row = 0; row < valid.numel(); ++row) if (bits[row]) valid_sources.insert(scale.source_ids[row]);
  require(valid_sources.size() >= 2, "calibration requires at least two distinct valid TRAIN source groups");
  const auto rows = scale.calibration_native.index({scale.calibration_valid}).to(torch::kFloat64);
  const auto std = (rows - rows.mean(0)).square().mean(0).sqrt();
  require(torch::equal(scale.values, std.clamp_min(scale_floor).to(torch::kFloat32)) &&
      torch::equal(scale.floor_applied, std.lt(scale_floor)) && scale.identity == scale_identity(scale),
      "fixed loss scale arithmetic or content identity differs");
}

LossScale calibrate_loss_scale(ModelImpl &model, const Input &full, const std::vector<std::string> &ids) {
  const RuntimeIsolation isolation; const ModelModeIsolation mode(model); torch::NoGradGuard no_grad;
  fixed_architecture(model.config()); validate_input(full, model.config(), true);
  require(full.data.is_cuda() && ids.size() == static_cast<size_t>(full.data.size(0)), "CUDA full-O TRAIN calibration required");
  Input legal{torch::where(full.observed.to(full.data.device()), full.data, torch::zeros_like(full.data)),
      full.observed, full.channel_ids, full.endpoints, full.sampling_interval};
  const auto encoding = model.encode(legal);
  LossScale scale; scale.calibration_native = native(encoding).detach().to(torch::kCPU).clone();
  scale.calibration_valid = encoding.sample_valid_mask.detach().to(torch::kCPU).clone(); scale.source_ids = ids;
  require(scale.calibration_valid.any().item<bool>(), "point-zero calibration has no valid TRAIN rows");
  const auto rows = scale.calibration_native.index({scale.calibration_valid}).to(torch::kFloat64);
  const auto std = (rows - rows.mean(0)).square().mean(0).sqrt();
  scale.values = std.clamp_min(scale_floor).to(torch::kFloat32);
  scale.floor_applied = std.lt(scale_floor); scale.identity = scale_identity(scale); validate_loss_scale(scale); return scale;
}

LossOutput view_loss(const EncodeOutput &ordinary, const EncodeOutput &student,
                    const RowIdentity &a, const RowIdentity &b, const torch::Tensor &original_eligible,
                    const torch::Tensor &scale) {
  const auto zo = native(ordinary), zs = native(student); const auto B = zo.size(0);
  require(zo.sizes() == zs.sizes() && zo.device() == zs.device() &&
      ordinary.channel_valid_mask.defined() && ordinary.channel_valid_mask.dim() == 2 &&
      student.channel_valid_mask.sizes() == ordinary.channel_valid_mask.sizes(), "matching native view shapes required");
  const auto C = ordinary.channel_valid_mask.size(1); validate_identity(a, B, C); validate_identity(b, B, C);
  require(torch::equal(a.row_indices.to(torch::kCPU), b.row_indices.to(torch::kCPU)) &&
      torch::equal(a.channel_ids.to(torch::kCPU), b.channel_ids.to(torch::kCPU)) &&
      torch::equal(a.endpoints.to(torch::kCPU), b.endpoints.to(torch::kCPU)) && a.source_ids == b.source_ids &&
      torch::equal(ordinary.channel_ids.to(torch::kCPU), a.channel_ids.to(torch::kCPU)) &&
      torch::equal(student.channel_ids.to(torch::kCPU), b.channel_ids.to(torch::kCPU)), "same-row identity mismatch");
  require(original_eligible.scalar_type() == torch::kBool && original_eligible.sizes() == torch::IntArrayRef({B}) &&
      ordinary.sample_valid_mask.scalar_type() == torch::kBool && student.sample_valid_mask.scalar_type() == torch::kBool &&
      ordinary.sample_valid_mask.sizes() == original_eligible.sizes() && student.sample_valid_mask.sizes() == original_eligible.sizes() &&
      ordinary.channel_valid_mask.scalar_type() == torch::kBool && student.channel_valid_mask.scalar_type() == torch::kBool &&
      scale.defined() && !scale.requires_grad() && scale.scalar_type() == torch::kFloat32 && scale.sizes() == torch::IntArrayRef({32}) &&
      torch::isfinite(scale).all().item<bool>() && scale.gt(0).all().item<bool>(), "invalid original eligibility or fixed scale");
  LossOutput out;
  out.supported_rows = original_eligible.to(zs.device()).logical_and(ordinary.sample_valid_mask.to(zs.device()))
      .logical_and(student.sample_valid_mask.to(zs.device())).logical_and(
          ordinary.channel_valid_mask.to(zs.device()).eq(student.channel_valid_mask.to(zs.device())).all(1));
  out.supported_count = out.supported_rows.sum().item<int64_t>();
  auto u = zs / scale.to(zs.device());
  out.agreement = out.supported_count ? (u - zo.detach() / scale.to(zs.device()))
      .index({out.supported_rows}).square().mean() : zs.sum() * 0;
  const auto supported = out.supported_rows.to(torch::kCPU).contiguous(); auto bits = supported.accessor<bool, 1>();
  std::set<std::string> seen; std::vector<int64_t> rows;
  for (int64_t row = 0; row < B; ++row)
    if (bits[row] && seen.insert(a.source_ids[row]).second) rows.push_back(row);
  out.variance_rows = torch::tensor(rows, torch::kInt64).to(zs.device()); out.unique_source_count = rows.size();
  if (rows.size() < 2) out.variance = zs.sum() * 0;
  else {
    const auto distinct = u.index_select(0, out.variance_rows);
    const auto std = ((distinct - distinct.mean(0)).square().mean(0) + variance_epsilon).sqrt();
    out.variance = torch::relu(variance_target - std).mean();
  }
  require(torch::isfinite(out.agreement).item<bool>() && torch::isfinite(out.variance).item<bool>(), "nonfinite native view auxiliary");
  return out;
}

ForwardEvidence training_forward(ModelImpl &model, const Input &input, const MaskPlan &original,
                                const context_deletion::Plan &student_plan, const RowIdentity &identity,
                                const torch::Tensor &scale, bool enabled) {
  fixed_architecture(model.config()); validate_input(input, model.config(), true);
  const auto checked = mask_from_hidden(input.observed.to(input.data.device()), original.hidden, model.config());
  require(torch::equal(checked.visible, original.visible) && torch::equal(checked.target, original.target) &&
      torch::equal(checked.eligible_channels, original.eligible_channels), "original O/A/Q/eligibility changed");
  ForwardEvidence out;
  if (!enabled) { out.ordinary = model.forward(input, original.hidden); out.total_loss = out.ordinary.loss; return out; }
  require(student_plan.visible.scalar_type() == torch::kBool && student_plan.visible.sizes() == original.visible.sizes() &&
      student_plan.deleted.scalar_type() == torch::kBool && student_plan.deleted.sizes() == original.visible.sizes() &&
      torch::equal(student_plan.visible, original.visible.logical_and(student_plan.deleted.logical_not())) &&
      !student_plan.deleted.logical_and(original.visible.logical_not()).any().item<bool>(), "student can delete only original visible context");
  const auto groups = student_plan.visible.reshape({input.data.size(0), model.config().channel_count,
      model.config().history_length / model.config().patch_length, model.config().patch_length * model.config().input_width}).any(3).sum(2);
  require(groups.masked_select(original.eligible_channels).ge(2).all().item<bool>(), "student repair lost original eligible patch support");
  const auto zeroed = [&](const torch::Tensor &visible) {
    return Input{torch::where(visible, input.data, torch::zeros_like(input.data)), visible,
        input.channel_ids, input.endpoints, input.sampling_interval};
  };
  out.ordinary.encoding = model.encode(zeroed(original.visible));
  out.ordinary.reconstruction = model.decode(compact_reconstruction_export(out.ordinary.encoding, model.config()), input.channel_ids);
  const auto reconstruction = hierarchical_huber(out.ordinary.reconstruction, input.data.detach(),
      original.target, original.eligible_channels, model.config().huber_delta);
  out.ordinary.loss = reconstruction.loss; out.ordinary.target_counts = reconstruction.target_counts;
  out.ordinary.eligible_channels = reconstruction.eligible_channels; out.ordinary.eligible_examples = reconstruction.eligible_examples;
  out.ordinary.eligible_channel_count = reconstruction.eligible_channel_count;
  out.ordinary.eligible_example_count = reconstruction.eligible_example_count; out.ordinary.target_cell_count = reconstruction.target_cell_count;
  out.student = model.encode(zeroed(student_plan.visible));
  out.auxiliary = view_loss(out.ordinary.encoding, out.student, identity, identity, reconstruction.eligible_examples, scale);
  out.total_loss = reconstruction.loss + agreement_weight * out.auxiliary.agreement + variance_weight * out.auxiliary.variance;
  require(torch::isfinite(out.total_loss).item<bool>(), "nonfinite combined loss"); return out;
}
} // namespace native_view_agreement

namespace {
struct State {
  Settings settings;
  ev::ProviderFitInput fit;
  Dataset training;
  FrozenScaler scaler;
  nv::LossScale scale;
  Model model{nullptr};
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::vector<torch::Tensor> initial_parameters;
  ev::CurveProgress progress;
  ev::CurveLossPoint last_loss;
  double calibration_seconds{0};
  bool failed{false};
  std::array<int64_t, 3> context_counts{0, 0, 0};
  int64_t original_eligible_rows{0}, auxiliary_rows{0}, unique_source_rows{0};
  std::map<std::string, std::string> audit;
  std::map<std::string, std::pair<int64_t, int64_t>> saved;
  // All evidence is detached CPU data; no old graph is retained across updates.
  std::vector<torch::Tensor> components, sampled_indices, ordinary_native, student_native;
  std::vector<torch::Tensor> supported, variance_selected, original_eligible;
  std::vector<torch::Tensor> ordinary_channels, student_channels, prediction, target, query;
  std::vector<torch::Tensor> ordinary_sample_valid, student_sample_valid;
};
ev::CurveProgress progress_report(const std::shared_ptr<State> &state) {
  require(!state->failed, "training previously failed; further use is forbidden");
  if (state->progress.completed > 0 && (state->progress.losses.empty() ||
      state->progress.losses.back().completed != state->last_loss.completed)) state->progress.losses.push_back(state->last_loss);
  auto out = state->progress;
  const auto parameters = state->model->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    require(torch::isfinite(parameters[i]).all().item<bool>(), "nonfinite updated parameters");
    out.weights_changed |= !torch::equal(parameters[i].detach(), state->initial_parameters[i]);
  }
  return out;
}
void require_new(const std::string &path) {
  require(!path.empty() && !fs::exists(path) && !fs::is_symlink(fs::symlink_status(path)), "refusing existing/symlink point artifact: " + path);
}
std::string canonical(const std::string &path) { return fs::weakly_canonical(fs::absolute(path)).string(); }
torch::Tensor stack_or_empty(const std::vector<torch::Tensor> &rows, const std::vector<int64_t> &shape, torch::ScalarType type) {
  return rows.empty() ? torch::empty(shape, torch::TensorOptions().dtype(type)) : torch::stack(rows);
}
void save_point(const std::shared_ptr<State> &state, const std::string &path) {
  const RuntimeIsolation isolation;
  const auto report = progress_report(state);
  require(report.attempted == report.completed && report.sampled_rows == report.attempted * state->settings.batch_size,
      "agreement policy requires unskipped absolute attempts");
  for (const auto &output : {path, path + ".training-raw.pt", path + ".scaler.pt", path + ".audit.pt"}) require_new(output);
  Checkpoint checkpoint;
  checkpoint.settings = state->settings; checkpoint.model = state->model; checkpoint.scaler = state->scaler;
  checkpoint.attempted_steps = report.attempted; checkpoint.completed_steps = report.completed;
  checkpoint.schema_id = state->training.schema_id; checkpoint.dataset_id = state->training.dataset_id;
  checkpoint.scaler_fit_dataset_id = state->training.dataset_id; checkpoint.source_fingerprint = workflow_source_fingerprint();
  checkpoint.training_policy_id = nv::policy_id;
  save_checkpoint(path, checkpoint, *state->optimizer);
  save_dataset(path + ".training-raw.pt", state->training);
  save_scaler(path + ".scaler.pt", state->scaler, state->settings.model, state->training.schema_id, state->training.dataset_id);
  torch::serialize::OutputArchive audit;
  write_text(audit, "artifact_kind", "rpb_native_view_agreement_training_audit_v1");
  for (const auto &[key, value] : state->audit) write_text(audit, key, value);
  write_text(audit, "fit_source_manifest", source_manifest(state->fit.training_source_ids));
  audit.write("attempted_steps", torch::tensor(report.attempted), true);
  audit.write("completed_steps", torch::tensor(report.completed), true);
  audit.write("sampled_rows", torch::tensor(report.sampled_rows), true);
  audit.write("channel_order", torch::tensor(state->fit.channel_ids, torch::kInt64), true);
  audit.write("sampling_interval", torch::tensor(state->fit.sampling_interval, torch::kFloat64), true);
  audit.write("endpoint", torch::tensor(state->fit.endpoint, torch::kFloat64), true);
  audit.write("training_seconds", torch::tensor(report.training_seconds, torch::kFloat64), true);
  audit.write("calibration_seconds_value", torch::tensor(state->calibration_seconds, torch::kFloat64), true);
  audit.write("finite_gradients", torch::tensor(report.finite_gradients, torch::kBool), true);
  audit.write("weights_changed", torch::tensor(report.weights_changed, torch::kBool), true);
  audit.write("last_input_cuda", torch::tensor(report.last_input_cuda, torch::kBool), true);
  audit.write("last_loss_cuda", torch::tensor(report.last_loss_cuda, torch::kBool), true);
  write_text(audit, "model_weight_update_budget", std::to_string(report.completed));
  save_scale(audit, state->scale);
  for (const auto &[key, value] : std::map<std::string, double>{{"agreement_weight_value", nv::agreement_weight},
      {"variance_weight_value", nv::variance_weight}, {"variance_target_value", nv::variance_target},
      {"variance_epsilon_value", nv::variance_epsilon}, {"loss_scale_floor_value", nv::scale_floor},
      {"context_deletion_ratio_value", .15}}) audit.write(key, torch::tensor(value, torch::kFloat64), true);
  audit.write("context_deletion_stream_value", torch::tensor(static_cast<int64_t>(context_deletion::stream)), true);
  audit.write("context_requested_deleted_coordinates", torch::tensor(state->context_counts[0]), true);
  audit.write("context_actual_deleted_coordinates", torch::tensor(state->context_counts[1]), true);
  audit.write("context_restored_coordinates", torch::tensor(state->context_counts[2]), true);
  audit.write("original_eligible_row_occurrences", torch::tensor(state->original_eligible_rows), true);
  audit.write("auxiliary_supported_row_occurrences", torch::tensor(state->auxiliary_rows), true);
  audit.write("variance_unique_source_occurrences", torch::tensor(state->unique_source_rows), true);
  const auto B = state->settings.batch_size;
  write_text(audit, "component_columns", "attempted,completed,target_cells,original_eligible_rows,auxiliary_rows,unique_sources,reconstruction,agreement,variance,total,gradient_norm,requested_deleted,actual_deleted,restored");
  audit.write("loss_components", stack_or_empty(state->components, {0, 14}, torch::kFloat64), true);
  audit.write("sampled_indices", stack_or_empty(state->sampled_indices, {0, B}, torch::kInt64), true);
  audit.write("ordinary_native", stack_or_empty(state->ordinary_native, {0, B, 32}, torch::kFloat32), true);
  audit.write("student_native", stack_or_empty(state->student_native, {0, B, 32}, torch::kFloat32), true);
  audit.write("auxiliary_supported", stack_or_empty(state->supported, {0, B}, torch::kBool), true);
  audit.write("variance_selected", stack_or_empty(state->variance_selected, {0, B}, torch::kBool), true);
  audit.write("original_eligible", stack_or_empty(state->original_eligible, {0, B}, torch::kBool), true);
  audit.write("ordinary_channel_valid", stack_or_empty(state->ordinary_channels, {0, B, 3}, torch::kBool), true);
  audit.write("student_channel_valid", stack_or_empty(state->student_channels, {0, B, 3}, torch::kBool), true);
  audit.write("ordinary_sample_valid", stack_or_empty(state->ordinary_sample_valid, {0, B}, torch::kBool), true);
  audit.write("student_sample_valid", stack_or_empty(state->student_sample_valid, {0, B}, torch::kBool), true);
  audit.write("ordinary_prediction", stack_or_empty(state->prediction, {0, B, 3, 32, 3}, torch::kFloat32), true);
  audit.write("normalized_target", stack_or_empty(state->target, {0, B, 3, 32, 3}, torch::kFloat32), true);
  audit.write("query_support", stack_or_empty(state->query, {0, B, 3, 32, 3}, torch::kBool), true);
  std::vector<torch::Tensor> trace;
  for (const auto &p : report.losses) trace.push_back(torch::tensor(std::vector<double>{
      static_cast<double>(p.attempted), static_cast<double>(p.completed), static_cast<double>(p.target_cells), p.loss, p.gradient_norm}, torch::kFloat64));
  audit.write("progress_loss_trace", stack_or_empty(trace, {0, 5}, torch::kFloat64), true);
  embedding::archive::save_archive(path + ".audit.pt", audit);
  state->saved.emplace(canonical(path), std::make_pair(report.attempted, report.completed));
}
std::map<std::string, std::string> validate_companion(const std::string &path, const Checkpoint &checkpoint,
                                                    const ev::ProviderFitInput &fit) {
  validate_fit(checkpoint.settings, fit);
  const auto raw = protocol_input(fit.training_observations, checkpoint.settings.model, fit);
  const auto described = describe_dataset(raw, checkpoint.settings.model, fit.feature_units);
  require(checkpoint.training_policy_id == nv::policy_id && checkpoint.dataset_id == described.dataset_id &&
      checkpoint.schema_id == described.schema_id && checkpoint.scaler_fit_dataset_id == described.dataset_id &&
      checkpoint.settings.seed == static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL) &&
      checkpoint.attempted_steps == checkpoint.completed_steps, "new tagged checkpoint TRAIN/policy/seed/counters differ");
  require(checkpoint.attempted_steps >= 0 && checkpoint.attempted_steps <=
      std::numeric_limits<int64_t>::max() / checkpoint.settings.batch_size / 288, "saved coordinate/sample counter overflow");
  torch::serialize::InputArchive archive; archive.load_from(path + ".audit.pt", torch::kCPU);
  require(read_text(archive, "artifact_kind") == "rpb_native_view_agreement_training_audit_v1", "new producer audit kind differs");
  const std::map<std::string, std::string> fixed{{"model_tag", nv::model_tag}, {"training_policy_id", nv::policy_id},
      {"agreement_weight", "0.05"}, {"variance_weight", "0.01"}, {"variance_target", "0.5"},
      {"variance_epsilon", "0.0001"}, {"loss_scale_floor", "0.000001"}, {"loss_scale_policy", nv::scale_policy},
      {"auxiliary_support_policy", nv::support_policy}, {"variance_source_policy", nv::variance_policy},
      {"agreement_target_policy", "same-online-ordinary-native32-detached;ordinary-reconstruction-gradient-retained;no-EMA-or-projector"},
      {"reconstruction_policy", "ordinary-V_o-only-decoded;original-O-A-Q-eligibility-hierarchical-Huber-unchanged"},
      {"context_deletion_ratio", "0.15"}, {"context_deletion_stream", "0x6374782d64726f70"},
      {"context_deletion_rng_policy", context_deletion::rng_policy}, {"context_deletion_repair_policy", context_deletion::repair_policy},
      {"context_deletion_visibility_policy", context_deletion::visibility_policy},
      {"context_deletion_count_policy", "cumulative-requested/actual/restored;student-every-unskipped-absolute-attempt"},
      {"training_skip_policy", "abort-on-ineligible-original-query;no-replacement;no-schedule-shift"},
      {"training_resume_policy", resume_policy}, {"rng_policy", rng_policy}, {"sampling_policy", sampling_policy},
      {"optimizer_policy", optimizer_policy}, {"protocol_id", fit.protocol_id}, {"feature_units", fit.feature_units},
      {"fit_source_manifest_id", manifest_id(fit.training_source_ids)},
      {"initialization_seed", std::to_string(training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL))},
      {"actual_training_seed", std::to_string(checkpoint.settings.seed)}, {"training_dataset_id", described.dataset_id},
      {"scaler_fit_dataset_id", described.dataset_id}, {"preprocessing_id", checkpoint.scaler.identity()},
      {"core_writer_source_fingerprint", checkpoint.source_fingerprint}};
  auto fields = fixed;
  for (const auto &[key, value] : fixed) require(read_text(archive, key) == value, "new companion text differs: " + key);
  auto saved_settings = parse_settings(read_text(archive, "resolved_settings")); saved_settings.model.device = checkpoint.settings.model.device;
  require(settings_text(saved_settings) == settings_text(checkpoint.settings) &&
      read_text(archive, "fit_source_manifest") == source_manifest(fit.training_source_ids) &&
      torch::Device(read_text(archive, "training_device")).is_cuda() &&
      read_integer(archive, "attempted_steps") == checkpoint.attempted_steps &&
      read_integer(archive, "completed_steps") == checkpoint.completed_steps &&
      read_integer(archive, "sampled_rows") == checkpoint.attempted_steps * checkpoint.settings.batch_size,
      "new companion settings/source/order/counters differ");
  for (const auto &[key, value] : std::map<std::string, double>{{"agreement_weight_value", .05}, {"variance_weight_value", .01},
      {"variance_target_value", .5}, {"variance_epsilon_value", 1e-4}, {"loss_scale_floor_value", 1e-6}, {"context_deletion_ratio_value", .15}})
    require(read_double(archive, key) == value, "new companion coefficient differs: " + key);
  require(read_integer(archive, "context_deletion_stream_value") == static_cast<int64_t>(context_deletion::stream), "student deletion stream differs");
  torch::Tensor channel_order; archive.read("channel_order", channel_order, true);
  require(torch::equal(channel_order, torch::tensor(fit.channel_ids, torch::kInt64)) &&
      read_double(archive, "sampling_interval") == fit.sampling_interval && read_double(archive, "endpoint") == fit.endpoint,
      "new companion physical channel/time order differs");
  const auto scale = load_scale(archive, fit.training_source_ids); fields.emplace("loss_scale_identity", scale.identity);
  const double calibration_seconds = read_double(archive, "calibration_seconds_value");
  require(calibration_seconds >= 0 && std::stod(read_text(archive, "calibration_seconds")) == calibration_seconds,
      "calibration cost text/typed witness differs");
  for (const auto &key : {"calibration_seconds", "calibration_time_scope", "training_time_scope"}) fields.emplace(key, read_text(archive, key));
  require(fields.at("calibration_time_scope") == "CUDA-synchronized-point0-full-O-scaler-transform-forward-CPU-std-identity-validation;separate-from-update-loop" &&
      fields.at("training_time_scope") == "CUDA-synchronized-update-loop;includes-two-encoder-passes-and-CPU-component-evidence-capture;excludes-calibration-checkpoint-heads-serving",
      "cost scopes differ");
  fields.emplace("training_producer_source_fingerprint", read_text(archive, "training_producer_source_fingerprint"));
  require(!fields.at("training_producer_source_fingerprint").empty(), "missing training producer identity");
  int64_t counts[3];
  const std::array<std::string, 3> names{"context_requested_deleted_coordinates", "context_actual_deleted_coordinates", "context_restored_coordinates"};
  for (size_t i = 0; i < names.size(); ++i) {
    counts[i] = read_integer(archive, names[i]); require(counts[i] >= 0, "negative student deletion count");
    fields.emplace(names[i], std::to_string(counts[i]));
  }
  require(counts[0] >= counts[1] && counts[2] == counts[0] - counts[1], "student deletion count arithmetic differs");
  require(counts[0] <= checkpoint.attempted_steps * checkpoint.settings.batch_size * 288,
      "student deletion requests exceed sampled coordinate population");
  for (const auto &key : {"original_eligible_row_occurrences", "auxiliary_supported_row_occurrences", "variance_unique_source_occurrences"}) {
    const auto count = read_integer(archive, key); require(count >= 0, "negative auxiliary row count"); fields.emplace(key, std::to_string(count));
  }
  require(read_integer(archive, "variance_unique_source_occurrences") <= read_integer(archive, "auxiliary_supported_row_occurrences") &&
      read_integer(archive, "auxiliary_supported_row_occurrences") <= read_integer(archive, "original_eligible_row_occurrences") &&
      read_integer(archive, "original_eligible_row_occurrences") <= checkpoint.attempted_steps * checkpoint.settings.batch_size,
      "saved auxiliary/group support exceeds original eligible/sample rows");
  if (checkpoint.completed_steps == 0) {
    require(counts[0] == 0 && read_integer(archive, "original_eligible_row_occurrences") == 0 &&
        read_integer(archive, "auxiliary_supported_row_occurrences") == 0 && read_integer(archive, "variance_unique_source_occurrences") == 0,
        "point zero has consumed training views");
  }
  return fields;
}
} // namespace

ev::CurveSnapshot make_native_view_agreement_snapshot(const std::string &path, const ev::ProviderFitInput &fit) {
  const RuntimeIsolation isolation;
  const auto checkpoint = load_checkpoint(path, torch::kCPU);
  const auto fields = validate_companion(path, checkpoint, fit);
  auto snapshot = make_retained_curve_snapshot(path, fit);
  for (const auto &[key, value] : fields) snapshot.features.audit_fields[key] = value;
  snapshot.features.audit_fields["attempted_steps"] = std::to_string(checkpoint.attempted_steps);
  snapshot.features.audit_fields["completed_steps"] = std::to_string(checkpoint.completed_steps);
  snapshot.features.audit_fields["loss_scale_serving_policy"] = "loss-only;ordinary-native32-and-decoder-unchanged";
  snapshot.features.provenance += "; model_tag=RPB-v9; training_policy=" + std::string(nv::policy_id) +
      "; loss_scale=" + fields.at("loss_scale_identity") + "; loss-only calibration; ordinary inference unchanged";
  const auto extract = snapshot.features.extract;
  snapshot.features.extract = [extract, identity = fields.at("loss_scale_identity")](const embedding::Batch &batch) {
    const RuntimeIsolation callback_isolation;
    auto surfaces = extract(batch);
    for (auto &[key, surface] : surfaces) {
      (void)key;
      surface.provenance += "; model_tag=RPB-v9; training_policy=" + std::string(nv::policy_id) + "; loss_scale=" + identity + "; loss-only";
    }
    return surfaces;
  };
  const auto reconstruct = snapshot.reconstruct;
  snapshot.reconstruct = [reconstruct](const embedding::Batch &batch, const torch::Tensor &hidden) {
    const RuntimeIsolation callback_isolation; return reconstruct(batch, hidden);
  };
  snapshot.features.save_assets = [fields = snapshot.features.audit_fields](const std::string &directory) {
    const auto path = (fs::path(directory) / "curve-snapshot-audit.pt").string(); require_new(path);
    torch::serialize::OutputArchive archive; write_text(archive, "artifact_kind", "rpb_native_view_agreement_snapshot_audit_v1");
    for (const auto &[key, value] : fields) write_text(archive, key, value);
    embedding::archive::save_archive(path, archive);
  };
  return snapshot;
}

ev::CurveTrainerFactory make_native_view_agreement_trainer(const Settings &settings) {
  validate_settings(settings); fixed_architecture(settings.model);
  require(settings.model.device.is_cuda() && torch::cuda::is_available(), "explicit CUDA required; no CPU fallback");
  require(settings.steps <= std::numeric_limits<int64_t>::max() / settings.batch_size / 288,
      "frozen budget would overflow cumulative coordinate/sample counters");
  return [settings](const ev::ProviderFitInput &fit) {
    const RuntimeIsolation isolation; validate_fit(settings, fit); at::set_num_threads(settings.threads);
    auto state = std::make_shared<State>(); state->settings = settings; state->fit = fit;
    state->settings.seed = static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL);
    state->fit.training_observations.data = fit.training_observations.data.detach().clone();
    state->fit.training_observations.feature_mask = fit.training_observations.feature_mask.detach().clone();
    const auto raw = protocol_input(state->fit.training_observations, settings.model, state->fit);
    state->training = describe_dataset(raw, settings.model, fit.feature_units); state->scaler = fit_scaler(raw, settings.model);
    const auto init_seed = training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL);
    torch::manual_seed(init_seed); state->model = Model(settings.model); state->model->train();
    for (const auto &p : state->model->parameters()) {
      state->initial_parameters.push_back(p.detach().clone()); state->progress.parameter_count += p.numel();
      if (p.is_cuda()) state->progress.cuda_parameter_count += p.numel();
    }
    require(state->progress.parameter_count == 225805 && state->progress.cuda_parameter_count == 225805, "actual CUDA parameter count differs");
    torch::cuda::synchronize(settings.model.device.index()); const auto calibration_began = std::chrono::steady_clock::now();
    state->scale = nv::calibrate_loss_scale(*state->model, state->scaler.transform(raw, settings.model), fit.training_source_ids);
    torch::cuda::synchronize(settings.model.device.index());
    state->calibration_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - calibration_began).count();
    state->optimizer = std::make_unique<torch::optim::AdamW>(state->model->parameters(),
        torch::optim::AdamWOptions(settings.learning_rate).weight_decay(settings.weight_decay));
    state->progress.training_device = settings.model.device.str(); state->progress.preprocessing_id = state->scaler.identity();
    state->progress.training_dataset_id = state->training.dataset_id;
    state->audit = {{"provider_id", kEncoderId}, {"adapter_id", "rpb_native_view_agreement_cuda_v1"},
      {"model_tag", nv::model_tag}, {"training_policy_id", nv::policy_id}, {"protocol_id", fit.protocol_id},
      {"fit_policy", "permitted_training_observations_only;scaler_fit_once;point0_loss_scale_fit_once"},
      {"resolved_settings", settings_text(state->settings)}, {"feature_units", fit.feature_units},
      {"initialization_seed", std::to_string(init_seed)}, {"actual_training_seed", std::to_string(state->settings.seed)},
      {"rng_policy", rng_policy}, {"sampling_policy", sampling_policy}, {"optimizer_policy", optimizer_policy},
      {"training_device", state->progress.training_device}, {"parameter_count", "225805"}, {"cuda_parameter_count", "225805"},
      {"output_semantics", output_semantics(settings.model)}, {"reconstruction_export_semantics", reconstruction_output_semantics(settings.model)},
      {"training_dataset_id", state->training.dataset_id}, {"scaler_fit_dataset_id", state->training.dataset_id},
      {"preprocessing_id", state->scaler.identity()}, {"fit_source_manifest_id", manifest_id(fit.training_source_ids)},
      {"training_observation_rows", std::to_string(fit.training_source_ids.size())},
      {"training_source_groups", std::to_string(std::set<std::string>(fit.training_source_ids.begin(), fit.training_source_ids.end()).size())},
      {"core_writer_source_fingerprint", workflow_source_fingerprint()}, {"training_producer_source_fingerprint", NATIVE_VIEW_AGREEMENT_SOURCE_ID},
      {"source_fingerprint_scope", "ordinary_checkpoint_field=core_writer;new_companion_training_producer=native_view_agreement_source"},
      {"trace_policy", "first_completed_update;every_log_every_updates;queried_milestone_endpoints_persist_as_cumulative_prefix"},
      {"training_skip_policy", "abort-on-ineligible-original-query;no-replacement;no-schedule-shift"},
      {"agreement_weight", "0.05"}, {"variance_weight", "0.01"}, {"variance_target", "0.5"}, {"variance_epsilon", "0.0001"},
      {"loss_scale_floor", "0.000001"}, {"loss_scale_policy", nv::scale_policy}, {"loss_scale_identity", state->scale.identity},
      {"auxiliary_support_policy", nv::support_policy}, {"variance_source_policy", nv::variance_policy},
      {"agreement_target_policy", "same-online-ordinary-native32-detached;ordinary-reconstruction-gradient-retained;no-EMA-or-projector"},
      {"reconstruction_policy", "ordinary-V_o-only-decoded;original-O-A-Q-eligibility-hierarchical-Huber-unchanged"},
      {"context_deletion_ratio", "0.15"}, {"context_deletion_stream", "0x6374782d64726f70"},
      {"context_deletion_rng_policy", context_deletion::rng_policy}, {"context_deletion_repair_policy", context_deletion::repair_policy},
      {"context_deletion_visibility_policy", context_deletion::visibility_policy},
      {"context_deletion_count_policy", "cumulative-requested/actual/restored;student-every-unskipped-absolute-attempt"},
      {"training_resume_policy", resume_policy}};
    std::ostringstream calibration_time; calibration_time << std::setprecision(17) << state->calibration_seconds;
    state->audit.emplace("calibration_seconds", calibration_time.str());
    state->audit.emplace("calibration_time_scope", "CUDA-synchronized-point0-full-O-scaler-transform-forward-CPU-std-identity-validation;separate-from-update-loop");
    state->audit.emplace("training_time_scope", "CUDA-synchronized-update-loop;includes-two-encoder-passes-and-CPU-component-evidence-capture;excludes-calibration-checkpoint-heads-serving");
    ev::CurveTrainer trainer; trainer.audit_fields = state->audit;
    trainer.train_to = [state](int64_t budget) {
      const RuntimeIsolation isolation; at::set_num_threads(state->settings.threads);
      require(!state->failed, "training previously failed; continuation is forbidden");
      require(budget >= state->progress.completed && budget <= state->settings.steps, "budget is outside frozen monotonic trajectory");
      if (budget == state->progress.completed) return progress_report(state);
      state->model->train(); torch::cuda::synchronize(state->settings.model.device.index());
      const auto began = std::chrono::steady_clock::now();
      const auto finish = [&] { torch::cuda::synchronize(state->settings.model.device.index());
        state->progress.training_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count(); };
      try {
        while (state->progress.completed < budget && state->progress.attempted < state->settings.attempt_limit) {
          const auto attempt = state->progress.attempted;
          require(state->progress.sampled_rows <= std::numeric_limits<int64_t>::max() - state->settings.batch_size, "sample counter overflow");
          const auto indices = training_detail::sampled_indices(state->training.input.data.size(0), state->settings.batch_size, state->settings.seed, attempt);
          const auto batch = state->scaler.transform(training_detail::selected(state->training.input, indices), state->settings.model);
          const auto original = make_training_mask(batch.observed, state->settings.model,
              training_detail::counter_seed(state->settings.seed, attempt, 0x6d61736bULL));
          require(original.eligible_channels.any().item<bool>(), "agreement policy aborts ineligible original query before update");
          const auto student = context_deletion::make_plan(original, batch.channel_ids, state->settings.model,
              state->settings.seed, attempt, ContextDeletionRecipe::coordinate15_v1);
          nv::RowIdentity identity{indices, batch.channel_ids.dim() == 1 ? batch.channel_ids.unsqueeze(0).expand({batch.data.size(0), 3}) : batch.channel_ids,
              batch.endpoints, {}};
          const auto cpu_indices = indices.to(torch::kCPU).contiguous(); const auto rows = cpu_indices.accessor<int64_t, 1>();
          for (int64_t i = 0; i < cpu_indices.numel(); ++i) identity.source_ids.push_back(state->fit.training_source_ids[rows[i]]);
          torch::manual_seed(training_detail::counter_seed(state->settings.seed, attempt, 0x746f726368ULL));
          ++state->progress.attempted; state->progress.sampled_rows += state->settings.batch_size;
          state->optimizer->zero_grad(); const auto output = nv::training_forward(*state->model, batch, original, student, identity, state->scale.values);
          state->progress.last_input_cuda = batch.data.is_cuda(); state->progress.last_loss_cuda = output.total_loss.is_cuda();
          require(state->progress.last_input_cuda && state->progress.last_loss_cuda &&
              output.ordinary.loss.is_cuda() && output.auxiliary.agreement.is_cuda() && output.auxiliary.variance.is_cuda(), "actual inputs/losses must be CUDA");
          output.total_loss.backward();
          for (const auto &p : state->model->parameters())
            require(!p.grad().defined() || (p.grad().is_cuda() && torch::isfinite(p.grad()).all().item<bool>()), "nonfinite/non-CUDA gradient");
          const auto norm = torch::nn::utils::clip_grad_norm_(state->model->parameters(), state->settings.gradient_clip_norm > 0 ?
              state->settings.gradient_clip_norm : std::numeric_limits<double>::infinity(), 2.0, true);
          state->progress.finite_gradients = std::isfinite(norm); require(state->progress.finite_gradients, "nonfinite gradient norm");
          state->optimizer->step(); ++state->progress.completed;
          state->context_counts[0] += student.requested_count; state->context_counts[1] += student.actual_count;
          state->context_counts[2] += student.restored_count; state->original_eligible_rows += output.ordinary.eligible_example_count;
          state->auxiliary_rows += output.auxiliary.supported_count; state->unique_source_rows += output.auxiliary.unique_source_count;
          state->last_loss = {state->progress.attempted, state->progress.completed, output.ordinary.target_cell_count, output.total_loss.item<double>(), norm};
          if (state->progress.completed == 1 || state->progress.completed % state->settings.log_every == 0) state->progress.losses.push_back(state->last_loss);
          state->components.push_back(torch::tensor(std::vector<double>{static_cast<double>(state->progress.attempted), static_cast<double>(state->progress.completed),
              static_cast<double>(output.ordinary.target_cell_count), static_cast<double>(output.ordinary.eligible_example_count),
              static_cast<double>(output.auxiliary.supported_count), static_cast<double>(output.auxiliary.unique_source_count),
              output.ordinary.loss.item<double>(), output.auxiliary.agreement.item<double>(), output.auxiliary.variance.item<double>(),
              output.total_loss.item<double>(), norm, static_cast<double>(student.requested_count), static_cast<double>(student.actual_count),
              static_cast<double>(student.restored_count)}, torch::kFloat64));
          const auto cpu = [](const torch::Tensor &x) { return x.detach().to(torch::kCPU).clone(); };
          state->sampled_indices.push_back(cpu_indices.clone()); state->ordinary_native.push_back(cpu(native(output.ordinary.encoding)));
          state->student_native.push_back(cpu(native(output.student))); state->supported.push_back(cpu(output.auxiliary.supported_rows));
          auto selected = torch::zeros({state->settings.batch_size}, torch::kBool);
          selected.index_fill_(0, output.auxiliary.variance_rows.to(torch::kCPU), true); state->variance_selected.push_back(selected);
          state->original_eligible.push_back(cpu(output.ordinary.eligible_examples));
          state->ordinary_channels.push_back(cpu(output.ordinary.encoding.channel_valid_mask)); state->student_channels.push_back(cpu(output.student.channel_valid_mask));
          state->ordinary_sample_valid.push_back(cpu(output.ordinary.encoding.sample_valid_mask));
          state->student_sample_valid.push_back(cpu(output.student.sample_valid_mask));
          state->prediction.push_back(cpu(output.ordinary.reconstruction)); state->target.push_back(cpu(batch.data)); state->query.push_back(cpu(original.target));
        }
        require(state->progress.completed == budget, "absolute attempt ceiling exhausted");
      } catch (...) { state->failed = true; finish(); throw; }
      finish(); return progress_report(state);
    };
    trainer.save_checkpoint = [state](const std::string &path) { save_point(state, path); };
    trainer.snapshot = [state](const std::string &path) {
      require(!state->failed && state->saved.count(canonical(path)), "snapshot must be an exact point saved by this live trainer");
      const RuntimeIsolation isolation;
      const auto checkpoint = load_checkpoint(path, torch::kCPU);
      require(checkpoint.attempted_steps == state->saved.at(canonical(path)).first &&
          checkpoint.completed_steps == state->saved.at(canonical(path)).second && checkpoint.scaler.identity() == state->scaler.identity(),
          "immutable saved snapshot counters/scaler differ");
      return make_native_view_agreement_snapshot(path, state->fit);
    };
    return trainer;
  };
}

std::map<std::string, std::string> audit_native_view_agreement_initialization(
    const std::string &candidate_path, const ev::RetainedPoolingCohort &reference,
    const ev::ProviderFitInput &fit, int64_t expected_updates) {
  const RuntimeIsolation isolation;
  require(expected_updates > 0 && reference.master_seed == fit.seed, "reference master/positive budget differs");
  auto candidate = load_checkpoint(candidate_path, torch::kCPU);
  const auto initial = load_checkpoint(reference.reference_initial_checkpoint, torch::kCPU);
  const auto positive = load_checkpoint(reference.reference_checkpoint, torch::kCPU);
  validate_fit(candidate.settings, fit); at::set_num_threads(candidate.settings.threads);
  const auto raw = protocol_input(fit.training_observations, candidate.settings.model, fit);
  const auto declared = describe_dataset(raw, candidate.settings.model, fit.feature_units);
  const auto fitted_scaler = fit_scaler(raw, candidate.settings.model);
  const auto same_scaler = [](const FrozenScaler &a, const FrozenScaler &b) {
    return a.identity() == b.identity() && a.scale_floor == b.scale_floor && torch::equal(a.mean, b.mean) &&
        torch::equal(a.scale, b.scale) && torch::equal(a.count, b.count) && torch::equal(a.channel_ids, b.channel_ids) &&
        torch::equal(a.floor_applied, b.floor_applied);
  };
  for (const auto *checkpoint : std::array<const Checkpoint *, 3>{&candidate, &initial, &positive}) {
    validate_fit(checkpoint->settings, fit);
    require(checkpoint->dataset_id == declared.dataset_id && checkpoint->schema_id == declared.schema_id &&
        checkpoint->scaler_fit_dataset_id == declared.dataset_id && checkpoint->settings.seed ==
            static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL) && same_scaler(checkpoint->scaler, fitted_scaler),
        "initial/reference exact TRAIN/schema/scaler/seed differs");
  }
  require(candidate.training_policy_id == nv::policy_id && initial.training_policy_id.empty() && positive.training_policy_id.empty() &&
      candidate.attempted_steps == 0 && candidate.completed_steps == 0 && initial.attempted_steps == 0 && initial.completed_steps == 0 &&
      positive.attempted_steps == expected_updates && positive.completed_steps == expected_updates,
      "new point0 or ordinary unskipped positive reference policy/counters differ");
  require(settings_text(candidate.settings) == settings_text(initial.settings) &&
      settings_text(initial.settings) == settings_text(positive.settings), "initial/reference frozen settings differ");
  const auto a = initial.model->named_parameters(), b = candidate.model->named_parameters(), p = positive.model->named_parameters();
  require(a.size() == b.size() && a.size() == p.size(), "common named parameter set differs");
  int64_t count = 0; bool positive_changed = false;
  for (const auto &parameter : a) {
    require(b.contains(parameter.key()) && p.contains(parameter.key()) && parameter.value().scalar_type() == b[parameter.key()].scalar_type() &&
        parameter.value().scalar_type() == p[parameter.key()].scalar_type() && parameter.value().sizes() == p[parameter.key()].sizes() &&
        torch::equal(parameter.value(), b[parameter.key()]), "common initialized parameter differs: " + parameter.key());
    count += parameter.value().numel(); positive_changed |= !torch::equal(parameter.value(), p[parameter.key()]);
  }
  require(count == 225805 && positive_changed, "reference must be the complete trained 225805-parameter architecture");
  const auto x = initial.model->named_buffers(), y = candidate.model->named_buffers(), z = positive.model->named_buffers();
  require(x.size() == y.size() && x.size() == z.size(), "named buffer sets differ");
  for (const auto &buffer : x)
    require(y.contains(buffer.key()) && z.contains(buffer.key()) && torch::equal(buffer.value(), y[buffer.key()]) &&
        torch::equal(buffer.value(), z[buffer.key()]), "common/fixed initialized buffer differs: " + buffer.key());
  const auto candidate_fields = validate_companion(candidate_path, candidate, fit);
  const std::array<std::string, 8> shared_keys{"protocol_id", "feature_units", "fit_source_manifest_id", "initialization_seed",
      "actual_training_seed", "rng_policy", "sampling_policy", "optimizer_policy"};
  for (const auto &[path, checkpoint] : std::array<std::pair<std::string, const Checkpoint *>, 2>{{
      {reference.reference_initial_checkpoint, &initial}, {reference.reference_checkpoint, &positive}}}) {
    torch::serialize::InputArchive audit; audit.load_from(path + ".audit.pt", torch::kCPU);
    require(read_text(audit, "artifact_kind") == "rpb_learning_curve_training_audit_v1", "ordinary reference companion kind differs");
    for (const auto &key : shared_keys) require(read_text(audit, key) == candidate_fields.at(key), "original reference counter/source field differs: " + key);
    auto archived_settings = parse_settings(read_text(audit, "resolved_settings")); archived_settings.model.device = torch::kCPU;
    require(settings_text(archived_settings) == settings_text(checkpoint->settings) &&
        read_text(audit, "fit_source_manifest") == source_manifest(fit.training_source_ids) &&
        read_text(audit, "training_dataset_id") == declared.dataset_id && read_text(audit, "scaler_fit_dataset_id") == declared.dataset_id &&
        read_text(audit, "preprocessing_id") == fitted_scaler.identity() &&
        read_text(audit, "core_writer_source_fingerprint") == checkpoint->source_fingerprint &&
        !read_text(audit, "training_producer_source_fingerprint").empty() && torch::Device(read_text(audit, "training_device")).is_cuda() &&
        read_integer(audit, "attempted_steps") == checkpoint->attempted_steps && read_integer(audit, "completed_steps") == checkpoint->completed_steps &&
        read_integer(audit, "sampled_rows") == checkpoint->attempted_steps * checkpoint->settings.batch_size,
        "ordinary reference source/settings/device/counters differ");
    torch::Tensor ids; audit.read("channel_order", ids, true);
    require(torch::equal(ids, torch::tensor(fit.channel_ids, torch::kInt64)) && read_double(audit, "endpoint") == fit.endpoint &&
        read_double(audit, "sampling_interval") == fit.sampling_interval, "ordinary reference physical channel/time order differs");
    for (const auto &key : {"training_policy_id", "context_deletion_ratio", "context_requested_deleted_coordinates"}) {
      torch::Tensor value; require(!audit.try_read(key, value, true), "ordinary reference contains a context policy");
    }
    if (checkpoint->completed_steps > 0) {
      torch::Tensor finite, changed; audit.read("finite_gradients", finite, true); audit.read("weights_changed", changed, true);
      require(finite.scalar_type() == torch::kBool && changed.scalar_type() == torch::kBool && finite.item<bool>() && changed.item<bool>(),
          "ordinary reference lacks real finite-gradient/weight-change evidence");
    }
  }
  auto gpu = load_checkpoint(candidate_path, torch::kCUDA);
  const auto calibrated = nv::calibrate_loss_scale(*gpu.model, gpu.scaler.transform(raw, gpu.settings.model), fit.training_source_ids);
  torch::serialize::InputArchive audit; audit.load_from(candidate_path + ".audit.pt", torch::kCPU);
  const auto stored = load_scale(audit, fit.training_source_ids);
  require(calibrated.identity == stored.identity && torch::equal(calibrated.calibration_native, stored.calibration_native) &&
      torch::equal(calibrated.calibration_valid, stored.calibration_valid), "point0 CUDA full-O loss calibration is not exact");
  return {{"common_parameters_exact", "true"}, {"all_parameters_exact_including_global_pool", "true"},
      {"common_parameter_count", std::to_string(count)}, {"common_tensors", std::to_string(a.size())},
      {"all_buffers_exact", "true"}, {"scaler_exact", "true"}, {"training_dataset_exact", "true"},
      {"counter_streams_exact", "true"}, {"training_policy_id", nv::policy_id}, {"model_tag", nv::model_tag},
      {"training_policy_companion_exact", "true"}, {"loss_scale_calibration_exact", "true"}, {"loss_scale_identity", stored.identity},
      {"loss_scale_policy", nv::scale_policy}, {"auxiliary_support_policy", nv::support_policy}, {"variance_source_policy", nv::variance_policy},
      {"training_dataset_id", declared.dataset_id}, {"schema_id", declared.schema_id}, {"preprocessing_id", fitted_scaler.identity()},
      {"initialization_seed", candidate_fields.at("initialization_seed")}, {"training_seed", candidate_fields.at("actual_training_seed")},
      {"training_protocol_id", fit_protocol}, {"reference_completed_updates", std::to_string(expected_updates)},
      {"reference_role", "fresh_equal_budget_training_reference"}, {"input_byte_integrity_owner", "shared_engine_explicit_input_hash_guard"},
      {"scope", "complete mode2 point0/scaler/TRAIN/source/order/original counter streams plus exact CUDA loss-only calibration; no old selected-reference claim"}};
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
