// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/optimization_diagnostic_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
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

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
constexpr const char *diagnostic_kind = "rpb_optimization_diagnostic_continuation_audit_v1";
constexpr const char *counter_policy = "splitmix64-counter-rows-masks-torch-attempt-v1";
constexpr const char *sampling_policy = "with_replacement_counter_rows;sampled_rows_includes_no_update_attempts";

void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error("[RPB optimization diagnostic] " + message);
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (const unsigned char c : value) {
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
std::string decimal(double value) {
  std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
std::string source_manifest(const std::vector<std::string> &ids) {
  std::string out;
  for (const auto &id : ids) out += std::to_string(id.size()) + ':' + id;
  return out;
}
std::string key(const std::string &path) {
  require(!path.empty(), "empty artifact path");
  return fs::weakly_canonical(fs::absolute(path)).string();
}
void new_path(const std::string &path) {
  require(!fs::exists(path) && !fs::is_symlink(fs::symlink_status(path)), "refusing an existing output: " + path);
  require(fs::is_directory(fs::path(path).parent_path()), "output parent directory must already exist");
}
std::string file_checksum(const std::string &path) {
  std::ifstream input(path, std::ios::binary); require(bool(input), "cannot read input artifact");
  uint64_t hash = 14695981039346656037ULL; char buffer[65536];
  while (input) {
    input.read(buffer, sizeof(buffer));
    for (std::streamsize i = 0; i < input.gcount(); ++i) { hash ^= static_cast<unsigned char>(buffer[i]); hash *= 1099511628211ULL; }
  }
  require(input.eof(), "input artifact read failed");
  std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << hash; return out.str();
}
void text(torch::serialize::OutputArchive &archive, const std::string &name, const std::string &value) {
  archive.write(name, embedding::archive::text_tensor(value), true);
}
std::string text(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true); return embedding::archive::tensor_text(value);
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
    try { for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]); at::set_num_threads(threads); }
    catch (...) { std::terminate(); }
  }
};
void candidate(const Checkpoint &checkpoint) {
  const auto &config = checkpoint.settings.model;
  require(config.device.is_cuda() && config.global_bottleneck_mode == 3 &&
      config.channel_mixer_layers == 1 && config.export_width == 32,
      "diagnostic requires explicit CUDA RPB-v5 mode3/mixer1/native32");
}
Input raw_input(const embedding::Batch &batch, const Config &config, const ev::ProviderFitInput &fit) {
  Input input{batch.data, batch.feature_mask, torch::tensor(fit.channel_ids, torch::kInt64),
      torch::full({batch.data.size(0)}, fit.endpoint, torch::kFloat64), fit.sampling_interval};
  validate_input(input, config); return input;
}
void validate_training(const Checkpoint &checkpoint, const ev::ProviderFitInput &fit) {
  const auto &config = checkpoint.settings.model;
  require(fit.shape.device.is_cpu() && fit.shape.dtype == torch::kFloat64 &&
      fit.shape.channel_count == config.channel_count && fit.shape.history_length == config.history_length &&
      fit.shape.input_width == config.input_width && fit.training_observations.data.defined() &&
      fit.training_observations.data.dim() == 4 && fit.training_observations.feature_mask.defined() &&
      fit.training_observations.feature_mask.device().is_cpu() &&
      fit.training_observations.data.device().is_cpu() && fit.training_observations.data.scalar_type() == torch::kFloat64 &&
      fit.training_source_ids.size() == static_cast<size_t>(fit.training_observations.data.size(0)),
      "exact original CPU float64 TRAIN and ordered source IDs required");
  for (const auto &id : fit.training_source_ids) require(!id.empty(), "empty TRAIN source ID");
  require(checkpoint.settings.seed == static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL), "original counter seed mismatch");
  const auto described = describe_dataset(raw_input(fit.training_observations, config, fit), config, fit.feature_units);
  require(described.schema_id == checkpoint.schema_id && described.dataset_id == checkpoint.dataset_id &&
      checkpoint.scaler_fit_dataset_id == described.dataset_id, "original TRAIN/schema/scaler-fit identity mismatch");
}
std::map<std::string, std::string> load_audit(const std::string &path, const Checkpoint &checkpoint,
                                           const ev::ProviderFitInput &fit) {
  torch::serialize::InputArchive archive; archive.load_from(path + ".audit.pt", torch::kCPU);
  const auto kind = text(archive, "artifact_kind");
  require(kind == "rpb_learning_curve_training_audit_v1" || kind == diagnostic_kind, "unknown parent producer audit");
  std::map<std::string, std::string> fields;
  for (const std::string name : {"resolved_settings", "actual_training_seed", "initialization_seed", "fit_source_manifest",
      "rng_policy", "sampling_policy", "training_producer_source_fingerprint", "core_writer_source_fingerprint"})
    fields.emplace(name, text(archive, name));
  require(fields.at("resolved_settings") == settings_text(checkpoint.settings) &&
      fields.at("actual_training_seed") == std::to_string(checkpoint.settings.seed) &&
      fields.at("initialization_seed") == std::to_string(training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL)) &&
      fields.at("fit_source_manifest") == source_manifest(fit.training_source_ids) &&
      fields.at("rng_policy") == counter_policy && fields.at("sampling_policy") == sampling_policy &&
      !fields.at("training_producer_source_fingerprint").empty() &&
      fields.at("core_writer_source_fingerprint") == checkpoint.source_fingerprint,
      "parent source/settings/initialization/ordered counter-stream audit mismatch");
  torch::Tensor attempted, completed; archive.read("attempted_steps", attempted, true); archive.read("completed_steps", completed, true);
  require(attempted.scalar_type() == torch::kInt64 && attempted.numel() == 1 &&
      completed.scalar_type() == torch::kInt64 && completed.numel() == 1 &&
      attempted.item<int64_t>() == checkpoint.attempted_steps && completed.item<int64_t>() == checkpoint.completed_steps,
      "parent producer counters differ from ordinary checkpoint");
  fields.emplace("artifact_kind", kind); return fields;
}
bool same_scaler(const FrozenScaler &a, const FrozenScaler &b) {
  return a.identity() == b.identity() && torch::equal(a.mean, b.mean) && torch::equal(a.scale, b.scale) &&
      torch::equal(a.count, b.count) && torch::equal(a.channel_ids, b.channel_ids) &&
      torch::equal(a.floor_applied, b.floor_applied) && a.scale_floor == b.scale_floor;
}
struct AdamWitness {
  int64_t step;
  torch::Tensor exp_avg, exp_avg_sq;
};
using OptimizerWitness = std::map<std::string, AdamWitness>;
OptimizerWitness optimizer_witness(Checkpoint &checkpoint, const std::string &path) {
  torch::optim::AdamW optimizer(checkpoint.model->parameters(),
      torch::optim::AdamWOptions(checkpoint.settings.learning_rate).weight_decay(checkpoint.settings.weight_decay));
  load_optimizer(path, optimizer, checkpoint.settings.model.device);
  OptimizerWitness states;
  for (const auto &parameter : checkpoint.model->named_parameters()) {
    const auto state = optimizer.state().find(parameter.value().unsafeGetTensorImpl());
    require(parameter.value().is_cuda(), "loaded model parameter is not on CUDA");
    if (state == optimizer.state().end()) continue;
    const auto *adam = dynamic_cast<const torch::optim::AdamWParamState *>(state->second.get());
    require(adam && adam->step() == checkpoint.completed_steps && adam->exp_avg().is_cuda() && adam->exp_avg_sq().is_cuda() &&
        adam->exp_avg().sizes() == parameter.value().sizes() && adam->exp_avg_sq().sizes() == parameter.value().sizes() &&
        adam->exp_avg().scalar_type() == parameter.value().scalar_type() &&
        adam->exp_avg_sq().scalar_type() == parameter.value().scalar_type() &&
        torch::isfinite(adam->exp_avg()).all().item<bool>() && torch::isfinite(adam->exp_avg_sq()).all().item<bool>(),
        "ordinary AdamW step/device/moment continuity mismatch");
    states.emplace(parameter.key(), AdamWitness{adam->step(), adam->exp_avg().detach().to(torch::kCPU).clone(),
        adam->exp_avg_sq().detach().to(torch::kCPU).clone()});
  }
  require(!states.empty() && states.size() == optimizer.state().size(), "missing or unassociated ordinary AdamW state");
  return states;
}
void write_optimizer_witness(torch::serialize::OutputArchive &archive, const std::string &key,
                             const OptimizerWitness &states) {
  torch::serialize::OutputArchive witness;
  witness.write("parameter_count", torch::tensor(static_cast<int64_t>(states.size())), true);
  int64_t index = 0;
  for (const auto &[name, state] : states) {
    torch::serialize::OutputArchive parameter;
    text(parameter, "parameter_name", name);
    parameter.write("step", torch::tensor(state.step), true);
    parameter.write("exp_avg", state.exp_avg, true);
    parameter.write("exp_avg_sq", state.exp_avg_sq, true);
    witness.write("parameter_" + std::to_string(index++), parameter);
  }
  archive.write(key, witness);
}
void freeze(Checkpoint &checkpoint) {
  checkpoint.model->eval();
  for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(false);
}
} // namespace

OptimizationResumeResult resume_optimization_diagnostic(const OptimizationResumeOptions &options,
                                                        const ev::ProviderFitInput &fit) {
  require(torch::cuda::is_available(), "CUDA required; no CPU fallback");
  require(options.additional_updates > 0, "positive additional update budget required");
  const std::string producer = EVALUATION_SOURCE_ID;
  require(producer.size() == 64 && producer.find_first_not_of("0123456789abcdef") == std::string::npos,
      "compiled diagnostic producer fingerprint required");
  const std::vector<std::string> inputs{options.parent_checkpoint, options.parent_checkpoint + ".audit.pt", options.training_archive};
  const std::vector<std::string> outputs{options.output_checkpoint, options.output_checkpoint + ".audit.pt",
      options.output_checkpoint + ".optimization-diagnostic.json"};
  std::map<std::string, std::string> checksums;
  for (const auto &input : inputs) { require(fs::is_regular_file(input), "required parent/TRAIN artifact missing"); checksums.emplace(key(input), file_checksum(input)); }
  std::set<std::string> output_keys;
  for (const auto &output : outputs) {
    require(!checksums.count(key(output)) && output_keys.insert(key(output)).second, "output aliases a preserved parent/TRAIN artifact");
    new_path(output);
  }
  const RuntimeIsolation isolation;
  auto parent = load_checkpoint(options.parent_checkpoint, torch::kCUDA); candidate(parent); validate_training(parent, fit);
  const auto parent_audit = load_audit(options.parent_checkpoint, parent, fit);
  require(parent.completed_steps > 0 && parent.completed_steps <= std::numeric_limits<int64_t>::max() - options.additional_updates,
      "invalid original/continued completed budget");
  const auto training = load_dataset(options.training_archive, parent.settings.model);
  require(training.dataset_id == parent.dataset_id && training.schema_id == parent.schema_id &&
      training.feature_units == fit.feature_units, "resume raw archive is not the exact original TRAIN");
  const auto before_states = optimizer_witness(parent, options.parent_checkpoint);
  std::vector<std::string> args{"rpb", "train", "--resume", options.parent_checkpoint, "--input", options.training_archive,
      "--device", "cuda", "--steps", std::to_string(options.additional_updates), "--checkpoint", options.output_checkpoint,
      "--checkpoint-every", "0"};
  std::vector<char *> argv; for (auto &arg : args) argv.push_back(arg.data());
  torch::cuda::synchronize(parent.settings.model.device.index());
  const auto began = std::chrono::steady_clock::now();
  require(run_cli(static_cast<int>(argv.size()), argv.data()) == 0, "ordinary workflow resume failed");
  torch::cuda::synchronize(parent.settings.model.device.index());
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  auto continued = load_checkpoint(options.output_checkpoint, torch::kCUDA); candidate(continued); validate_training(continued, fit);
  auto expected_settings = parent.settings; expected_settings.steps = options.additional_updates; expected_settings.checkpoint_every = 0;
  require(settings_text(continued.settings) == settings_text(expected_settings) &&
      continued.completed_steps == parent.completed_steps + options.additional_updates &&
      continued.attempted_steps >= parent.attempted_steps + options.additional_updates &&
      continued.dataset_id == parent.dataset_id && continued.schema_id == parent.schema_id &&
      continued.scaler_fit_dataset_id == parent.scaler_fit_dataset_id && same_scaler(parent.scaler, continued.scaler),
      "ordinary continuation changed settings/counters/data/scaler association");
  const auto after_states = optimizer_witness(continued, options.output_checkpoint);
  require(before_states.size() == after_states.size(), "continuation changed optimizer parameter associations");
  for (const auto &[name, state] : before_states)
    require(after_states.count(name) && after_states.at(name).step == state.step + options.additional_updates,
        "AdamW state was reset or did not advance by the exact completed budget");
  const auto old = parent.model->named_parameters(); bool changed = false;
  for (const auto &parameter : continued.model->named_parameters()) {
    require(torch::isfinite(parameter.value()).all().item<bool>(), "continued weights are nonfinite");
    changed = changed || !torch::equal(parameter.value(), old[parameter.key()]);
  }
  require(changed && std::isfinite(seconds) && seconds > 0, "continuation made no weight update or invalid timing");
  for (const auto &[path, checksum] : checksums) require(file_checksum(path) == checksum, "preserved parent/TRAIN artifact changed");
  std::ostringstream argv_json; argv_json << '[';
  for (size_t i = 0; i < args.size(); ++i) { if (i) argv_json << ','; argv_json << quote(args[i]); }
  argv_json << ']';
  std::map<std::string, std::string> audit{
      {"artifact_kind", diagnostic_kind}, {"protocol_id", fit.protocol_id}, {"model_tag", "RPB-v5"},
      {"resolved_settings", settings_text(continued.settings)}, {"actual_training_seed", std::to_string(continued.settings.seed)},
      {"initialization_seed", parent_audit.at("initialization_seed")}, {"fit_source_manifest", source_manifest(fit.training_source_ids)},
      {"rng_policy", counter_policy}, {"sampling_policy", sampling_policy},
      {"training_producer_source_fingerprint", workflow_source_fingerprint()},
      {"core_writer_source_fingerprint", continued.source_fingerprint}, {"diagnostic_orchestrator_source_fingerprint", producer},
      {"parent_checkpoint", key(options.parent_checkpoint)}, {"parent_artifact_kind", parent_audit.at("artifact_kind")},
      {"parent_checkpoint_writer_source_fingerprint", parent.source_fingerprint},
      {"parent_training_producer_source_fingerprint", parent_audit.at("training_producer_source_fingerprint")},
      {"training_archive", key(options.training_archive)}, {"training_dataset_id", continued.dataset_id},
      {"scaler_fit_dataset_id", continued.scaler_fit_dataset_id}, {"preprocessing_id", continued.scaler.identity()},
      {"feature_units", fit.feature_units}, {"output_semantics", output_semantics(continued.settings.model)},
      {"reconstruction_export_semantics", reconstruction_output_semantics(continued.settings.model)},
      {"timing_scope", "CUDA-synchronized whole run_cli command; load/normalize/update/log/save; excludes adapter pre/post validation and readouts"},
      {"gpu_update_only_seconds", "unmeasured"}, {"synchronized_command_wall_seconds", decimal(seconds)},
      {"fit_policy", "original exact TRAIN observations only; scaler/model initialization never refitted"},
      {"optimizer_policy", "ordinary saved AdamW state; exact active parameter step increments; absolute attempt counters"},
      {"tensor_witness_policy", "CPU exact copies of named ordinary AdamW moments/steps and frozen scaler before/after; no state refit"},
      {"input_checksum_algorithm", "fnv1a64-file-bytes; external frozen manifest supplies SHA256"},
      {"parent_checkpoint_checksum_fnv1a64", checksums.at(key(options.parent_checkpoint))},
      {"parent_audit_checksum_fnv1a64", checksums.at(key(options.parent_checkpoint + ".audit.pt"))},
      {"training_archive_checksum_fnv1a64", checksums.at(key(options.training_archive))}};
  torch::serialize::OutputArchive archive;
  for (const auto &[name, value] : audit) text(archive, name, value);
  archive.write("attempted_steps", torch::tensor(continued.attempted_steps), true);
  archive.write("completed_steps", torch::tensor(continued.completed_steps), true);
  archive.write("parent_attempted_steps", torch::tensor(parent.attempted_steps), true);
  archive.write("parent_completed_steps", torch::tensor(parent.completed_steps), true);
  archive.write("additional_completed_updates", torch::tensor(options.additional_updates), true);
  archive.write("synchronized_command_wall_seconds_value", torch::tensor(seconds, torch::kFloat64), true);
  archive.write("weights_changed", torch::tensor(changed, torch::kBool), true);
  write_optimizer_witness(archive, "optimizer_before", before_states);
  write_optimizer_witness(archive, "optimizer_after", after_states);
  torch::serialize::OutputArchive scaler_before, scaler_after;
  parent.scaler.save(scaler_before); continued.scaler.save(scaler_after);
  archive.write("scaler_before", scaler_before); archive.write("scaler_after", scaler_after);
  // Wall-time is already encoded as text in audit: keep the typed scalar under a distinct key.
  const auto audit_path = options.output_checkpoint + ".audit.pt";
  embedding::archive::save_archive(audit_path, archive);
  std::ostringstream report; report << std::setprecision(17) << "{\"protocol\":\"optimization-diagnostic-v1\",\"status\":\"completed\",\"model_tag\":\"RPB-v5\","
      << "\"additional_updates\":" << options.additional_updates << ",\"attempted_before\":" << parent.attempted_steps
      << ",\"completed_before\":" << parent.completed_steps << ",\"attempted_after\":" << continued.attempted_steps
      << ",\"completed_after\":" << continued.completed_steps << ",\"synchronized_command_wall_seconds\":" << seconds
      << ",\"gpu_update_only_seconds\":null,\"optimizer_steps_and_devices_verified\":true,\"original_inputs_preserved\":true,\"weights_changed\":true,"
      << "\"labels_access\":false,\"validation_access_during_training\":false,\"test_access\":false,\"argv\":" << argv_json.str()
      << ",\"audit_fields\":" << fields_json(audit) << '}';
  std::ofstream json(outputs[2], std::ios::binary); require(bool(json), "cannot create continuation JSON");
  json << report.str() << '\n'; json.close(); require(bool(json), "cannot save continuation JSON");
  return {options.output_checkpoint, audit_path, report.str(), parent.attempted_steps, parent.completed_steps,
      continued.attempted_steps, continued.completed_steps, seconds};
}

ev::CurveSnapshot make_optimization_diagnostic_snapshot(const std::string &path, const ev::ProviderFitInput &fit) {
  require(torch::cuda::is_available(), "CUDA snapshot reconstruction required");
  const RuntimeIsolation isolation;
  auto checkpoint = std::make_shared<Checkpoint>(load_checkpoint(path, torch::kCUDA));
  candidate(*checkpoint); validate_training(*checkpoint, fit); const auto producer_audit = load_audit(path, *checkpoint, fit);
  freeze(*checkpoint);
  EvaluationOptions options; options.checkpoint_path = path; options.surface_prefix = "diagnostic_checkpoint";
  const auto provider = make_evaluation_provider(options)(fit);
  const std::string base = "diagnostic_checkpoint_trained_contextual";
  ev::CurveSnapshot snapshot; snapshot.features.provenance = provider.provenance + "; immutable exact mode3 checkpoint; no fit";
  snapshot.features.surfaces.emplace("curve_global", provider.surfaces.at(base + "_global"));
  snapshot.features.audit_fields = producer_audit;
  snapshot.features.audit_fields["checkpoint_path"] = key(path);
  snapshot.features.audit_fields["snapshot_loader_source_fingerprint"] = EVALUATION_SOURCE_ID;
  snapshot.features.audit_fields["snapshot_policy"] = "independent CPU checkpoint serving and frozen CUDA reconstruction; scaler unchanged";
  snapshot.features.audit_fields["completed_steps"] = std::to_string(checkpoint->completed_steps);
  snapshot.features.audit_fields["attempted_steps"] = std::to_string(checkpoint->attempted_steps);
  snapshot.features.extract = [provider, base](const embedding::Batch &batch) {
    const auto all = provider.extract(batch); const auto &surface = all.at(base + "_global");
    require(surface.values.device().is_cpu() && surface.values.size(1) == 32 && !surface.values.requires_grad(), "native32 CPU snapshot contract");
    return ev::FeatureMap{{"curve_global", {surface.values.detach().clone(), surface.valid.clone(), surface.provenance}}};
  };
  snapshot.features.save_assets = [fields = snapshot.features.audit_fields](const std::string &directory) {
    const auto destination = (fs::path(directory) / "optimization-snapshot-audit.pt").string(); new_path(destination);
    torch::serialize::OutputArchive archive; text(archive, "artifact_kind", "rpb_optimization_diagnostic_snapshot_audit_v1");
    for (const auto &[name, value] : fields) if (name != "artifact_kind") text(archive, name, value);
    embedding::archive::save_archive(destination, archive);
  };
  auto metadata = fit; metadata.training_observations = {}; metadata.training_source_ids.clear();
  snapshot.reconstruct = [checkpoint, metadata](const embedding::Batch &batch, const torch::Tensor &hidden) {
    torch::NoGradGuard no_grad;
    const auto normalized = checkpoint->scaler.transform(raw_input(batch, checkpoint->settings.model, metadata), checkpoint->settings.model);
    const auto out = checkpoint->model->forward(normalized, hidden);
    require(out.reconstruction.is_cuda() && torch::isfinite(out.reconstruction).all().item<bool>(), "invalid frozen CUDA reconstruction");
    return ev::CurveReconstruction{out.reconstruction.detach().to(torch::kCPU), normalized.data.detach().to(torch::kCPU), out.eligible_channels.to(torch::kCPU)};
  };
  return snapshot;
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
