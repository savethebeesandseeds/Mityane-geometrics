// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/optimization_diagnostic_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;

std::string audit_text(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::Tensor value; archive.read(key, value, true);
  return embedding::archive::tensor_text(value);
}
int64_t count(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::Tensor value; archive.read(key, value, true);
  check(value.scalar_type() == torch::kInt64 && value.numel() == 1, key + " typed counter");
  return value.item<int64_t>();
}
std::string bytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  check(bool(file), "cannot read preserved artifact");
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void absent_outputs(const std::string &path) {
  for (const auto suffix : {"", ".audit.pt", ".optimization-diagnostic.json"})
    check(!fs::exists(path + suffix), "invalid continuation created output " + path + suffix);
}
void same_parameters(const rpb::Model &actual, const rpb::Model &expected, const std::string &label) {
  const auto reference = expected->named_parameters();
  check(actual->named_parameters().size() == reference.size(), label + " parameter count");
  for (const auto &parameter : actual->named_parameters())
    close(parameter.value().to(torch::kCPU), reference[parameter.key()].to(torch::kCPU),
          label + "/" + parameter.key(), 0, 0);
}
struct AdamState { int64_t step; torch::Tensor first, second; };
using AdamStates = std::map<std::string, AdamState>;
AdamStates adam_states(rpb::Checkpoint &checkpoint, const std::string &path) {
  torch::optim::AdamW optimizer(checkpoint.model->parameters(),
      torch::optim::AdamWOptions(checkpoint.settings.learning_rate).weight_decay(checkpoint.settings.weight_decay));
  rpb::load_optimizer(path, optimizer, checkpoint.settings.model.device);
  AdamStates states;
  for (const auto &parameter : checkpoint.model->named_parameters()) {
    const auto found = optimizer.state().find(parameter.value().unsafeGetTensorImpl());
    if (found == optimizer.state().end()) continue;
    const auto *state = dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
    check(state && state->exp_avg().is_cuda() && state->exp_avg_sq().is_cuda(), "ordinary CUDA AdamW state");
    states.emplace(parameter.key(), AdamState{state->step(), state->exp_avg().to(torch::kCPU).clone(),
        state->exp_avg_sq().to(torch::kCPU).clone()});
  }
  check(!states.empty() && states.size() == optimizer.state().size(), "every saved optimizer state has a named CUDA parameter");
  return states;
}
void same_adam(const AdamStates &actual, const AdamStates &expected, int64_t step, const std::string &label) {
  check(actual.size() == expected.size(), label + " state count");
  for (const auto &[name, state] : actual) {
    check(expected.count(name) && state.step == step && expected.at(name).step == step, label + " steps");
    close(state.first, expected.at(name).first, label + "/first/" + name, 0, 0);
    close(state.second, expected.at(name).second, label + "/second/" + name, 0, 0);
  }
}
void check_witness(torch::serialize::InputArchive &audit, const char *key, const AdamStates &expected) {
  torch::serialize::InputArchive witness; audit.read(key, witness);
  check(count(witness, "parameter_count") == static_cast<int64_t>(expected.size()), "CPU optimizer witness count");
  std::set<std::string> names;
  for (int64_t index = 0; index < static_cast<int64_t>(expected.size()); ++index) {
    torch::serialize::InputArchive parameter; witness.read("parameter_" + std::to_string(index), parameter);
    const auto name = audit_text(parameter, "parameter_name");
    check(expected.count(name) && names.insert(name).second && count(parameter, "step") == expected.at(name).step,
        "CPU optimizer witness named step association");
    torch::Tensor first, second; parameter.read("exp_avg", first, true); parameter.read("exp_avg_sq", second, true);
    check(first.device().is_cpu() && second.device().is_cpu(), "witness contains CPU moments only");
    close(first, expected.at(name).first, "witness first exact", 0, 0);
    close(second, expected.at(name).second, "witness second exact", 0, 0);
  }
}
struct RuntimeWitness {
  int threads = at::get_num_threads();
  at::Generator cpu = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  at::Generator gpu = at::globalContext().defaultGenerator(at::Device(at::kCUDA, 0));
  torch::Tensor cpu_state = cpu.get_state().clone(), gpu_state = gpu.get_state().clone();
  void unchanged() const {
    check(threads == at::get_num_threads() && torch::equal(cpu_state, cpu.get_state()) &&
        torch::equal(gpu_state, gpu.get_state()), "diagnostic preserves caller RNG and threads");
  }
};
void freeze(rpb::Checkpoint &checkpoint) {
  checkpoint.model->eval();
  for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(false);
}

void diagnostic_contract() {
  auto c = config(); c.export_width = 32; c.channel_mixer_layers = 1; c.global_bottleneck_mode = 3;
  c.dropout = .1; c.device = torch::Device(torch::kCUDA, 0); c.sampling_interval = .5;
  auto settings = rpb::default_settings(); settings.model = c; settings.steps = 6;
  settings.seed = 101; settings.batch_size = 2; settings.attempt_limit = 32;
  settings.log_every = 3; settings.checkpoint_every = 25;
  auto raw = input(c, 6); const auto order = torch::tensor({1, 0}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order);
  raw.channel_ids = raw.channel_ids.index_select(0, order);
  raw.observed.select(0, 0).select(0, 0).select(0, 12).fill_(false);
  raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch heldout{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{heldout,
      {c.channel_count, c.history_length, c.input_width, torch::kFloat64, torch::kCPU},
      202, {"a", "a", "b", "b", "c", "c"}, {202, 101}, "volts,amperes",
      "optimization-diagnostic-contract-v1", c.sampling_interval, (c.history_length - 1) * c.sampling_interval};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-optimization-diagnostic-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  const auto factory = rpb::make_learning_curve_trainer(settings);
  auto uninterrupted = factory(fit); uninterrupted.train_to(2);
  const auto parent_path = (directory / "parent-2.pt").string(); uninterrupted.save_checkpoint(parent_path);
  const auto raw_path = parent_path + ".training-raw.pt";
  const std::map<std::string, std::string> preserved{{parent_path, bytes(parent_path)},
      {parent_path + ".audit.pt", bytes(parent_path + ".audit.pt")}, {raw_path, bytes(raw_path)}};
  const auto four_path = (directory / "continued-4.pt").string();
  rpb::OptimizationResumeOptions options{parent_path, raw_path, four_path, 2};

  auto wrong_fit = fit; wrong_fit.seed = 303;
  rejects([&] { rpb::resume_optimization_diagnostic(options, wrong_fit); }, "wrong original seed"); absent_outputs(four_path);
  wrong_fit = fit; wrong_fit.training_source_ids[0] = "foreign_source";
  rejects([&] { rpb::resume_optimization_diagnostic(options, wrong_fit); }, "wrong ordered original source association"); absent_outputs(four_path);
  wrong_fit = fit; wrong_fit.training_observations.data = heldout.data.clone(); wrong_fit.training_observations.data[0][0][0][0] += 2;
  rejects([&] { rpb::resume_optimization_diagnostic(options, wrong_fit); }, "wrong original observations"); absent_outputs(four_path);
  auto wrong_raw = raw; wrong_raw.data = raw.data.clone(); wrong_raw.data[0][0][0][0] += 2;
  const auto wrong_path = (directory / "wrong-raw.pt").string();
  rpb::save_dataset(wrong_path, rpb::describe_dataset(wrong_raw, c, fit.feature_units));
  auto wrong_options = options; wrong_options.training_archive = wrong_path;
  rejects([&] { rpb::resume_optimization_diagnostic(wrong_options, fit); }, "wrong versioned original TRAIN archive"); absent_outputs(four_path);
  wrong_options = options; wrong_options.output_checkpoint = parent_path;
  rejects([&] { rpb::resume_optimization_diagnostic(wrong_options, fit); }, "overwriting original checkpoint");
  wrong_options = options; wrong_options.output_checkpoint = raw_path;
  rejects([&] { rpb::resume_optimization_diagnostic(wrong_options, fit); }, "aliasing original raw archive");
  wrong_options = options; wrong_options.additional_updates = 0;
  rejects([&] { rpb::resume_optimization_diagnostic(wrong_options, fit); }, "zero additional budget"); absent_outputs(four_path);

  RuntimeWitness before_resume;
  const auto resumed = rpb::resume_optimization_diagnostic(options, fit); before_resume.unchanged();
  check(resumed.completed_before == 2 && resumed.attempted_before == 2 && resumed.completed_after == 4 &&
      resumed.attempted_after == 4 && resumed.synchronized_command_wall_seconds > 0 &&
      resumed.audit_json.find("\"gpu_update_only_seconds\":null") != std::string::npos &&
      resumed.audit_json.find("\"test_access\":false") != std::string::npos,
      "continued counters and honest command timing/access report");
  auto continued = rpb::load_checkpoint(four_path, c.device), parent = rpb::load_checkpoint(parent_path, c.device);
  check(continued.settings.seed == 202 && continued.settings.steps == 2 && continued.settings.checkpoint_every == 0 &&
      continued.settings.batch_size == settings.batch_size && continued.settings.model.global_bottleneck_mode == 3 &&
      continued.scaler.identity() == parent.scaler.identity() && continued.dataset_id == parent.dataset_id &&
      continued.scaler_fit_dataset_id == parent.dataset_id, "ordinary resume retains exact original recipe/scaler/data");
  for (const auto &parameter : continued.model->parameters()) check(parameter.requires_grad() && parameter.is_cuda(), "continued checkpoint is trainable on CUDA");
  const auto parent_adam = adam_states(parent, parent_path), continued_adam = adam_states(continued, four_path);
  uninterrupted.train_to(4); const auto direct_four = (directory / "direct-4.pt").string(); uninterrupted.save_checkpoint(direct_four);
  auto expected_four = rpb::load_checkpoint(direct_four, c.device);
  same_parameters(continued.model, expected_four.model, "ordinary uninterrupted-versus-resumed exact weights");
  same_adam(continued_adam, adam_states(expected_four, direct_four), 4, "ordinary uninterrupted-versus-resumed exact optimizer");

  torch::serialize::InputArchive audit; audit.load_from(resumed.audit_path, torch::kCPU);
  check(audit_text(audit, "artifact_kind") == "rpb_optimization_diagnostic_continuation_audit_v1" &&
      audit_text(audit, "resolved_settings") == rpb::settings_text(continued.settings) &&
      audit_text(audit, "actual_training_seed") == "202" &&
      audit_text(audit, "training_producer_source_fingerprint") == rpb::workflow_source_fingerprint() &&
      audit_text(audit, "core_writer_source_fingerprint") == continued.source_fingerprint &&
      audit_text(audit, "diagnostic_orchestrator_source_fingerprint").size() == 64 &&
      audit_text(audit, "parent_artifact_kind") == "rpb_learning_curve_training_audit_v1" &&
      audit_text(audit, "gpu_update_only_seconds") == "unmeasured" &&
      count(audit, "parent_completed_steps") == 2 && count(audit, "completed_steps") == 4,
      "source producer scopes, parent audit compatibility and typed absolute counters");
  torch::Tensor seconds; audit.read("synchronized_command_wall_seconds_value", seconds, true);
  check(seconds.scalar_type() == torch::kFloat64 && seconds.item<double>() == resumed.synchronized_command_wall_seconds,
      "typed synchronized wall-time has distinct archive key");
  check_witness(audit, "optimizer_before", parent_adam); check_witness(audit, "optimizer_after", continued_adam);
  torch::serialize::InputArchive scaler_before, scaler_after; audit.read("scaler_before", scaler_before); audit.read("scaler_after", scaler_after);
  const auto first_scaler = rpb::FrozenScaler::load(scaler_before), second_scaler = rpb::FrozenScaler::load(scaler_after);
  check(first_scaler.identity() == parent.scaler.identity() && second_scaler.identity() == parent.scaler.identity(),
      "CPU frozen scaler before/after witness is exact");

  RuntimeWitness before_snapshot;
  const auto snapshot = rpb::make_optimization_diagnostic_snapshot(four_path, fit); before_snapshot.unchanged();
  check(snapshot.features.surfaces.size() == 1 && snapshot.features.surfaces.at("curve_global").kind == ev::SurfaceKind::global,
      "only native32 exact trained checkpoint global surface");
  const auto features = snapshot.features.extract(heldout);
  check(features.size() == 1 && features.at("curve_global").values.size(1) == 32 &&
      features.at("curve_global").valid.all().item<bool>(), "native32 observed-valid feature extraction");
  auto cpu_checkpoint = rpb::load_checkpoint(four_path); cpu_checkpoint.model->eval();
  auto cpu_config = c; cpu_config.device = torch::kCPU;
  const auto saved_raw = rpb::load_dataset(raw_path, cpu_config);
  auto hidden = torch::zeros_like(heldout.feature_mask); hidden.narrow(2, 0, c.patch_length).fill_(true);
  const auto reconstruction = snapshot.reconstruct(heldout, hidden);
  {
    torch::NoGradGuard no_grad;
    const auto encoded = cpu_checkpoint.model->encode(cpu_checkpoint.scaler.transform(saved_raw.input, cpu_checkpoint.settings.model));
    close(features.at("curve_global").values, encoded.z_contextual_global, "CPU exact ordinary native32 export", 0, 0);
    close(features.at("curve_global").valid, encoded.sample_valid_mask, "CPU exact native support", 0, 0);
    freeze(continued); const auto normalized = continued.scaler.transform(saved_raw.input, continued.settings.model);
    const auto forward = continued.model->forward(normalized, hidden);
    close(reconstruction.prediction, forward.reconstruction.to(torch::kCPU), "frozen CUDA exact ordinary decoder32 route", 0, 0);
    close(reconstruction.target, normalized.data.to(torch::kCPU), "exact standardized targets", 0, 0);
    close(reconstruction.eligible, forward.eligible_channels.to(torch::kCPU), "exact query eligibility", 0, 0);
  }
  const embedding::Batch changed_hidden{torch::where(hidden, heldout.data + 100, heldout.data), heldout.feature_mask};
  const auto changed = snapshot.reconstruct(changed_hidden, hidden);
  close(changed.prediction, reconstruction.prediction, "hidden query values cannot enter exact export decoder", 0, 0);
  close(changed.eligible, reconstruction.eligible, "hidden target perturbation preserves legal support", 0, 0);
  check(!torch::equal(changed.target, reconstruction.target), "held-out standardized targets are scoring-only");
  embedding::Batch absent{torch::full_like(heldout.data, std::numeric_limits<double>::quiet_NaN()), torch::zeros_like(heldout.feature_mask)};
  const auto absent_surface = snapshot.features.extract(absent).at("curve_global");
  check(!absent_surface.valid.any().item<bool>(), "allmissing does not invent global support");
  close(absent_surface.values, torch::zeros_like(absent_surface.values), "allmissing exact native zero", 0, 0);
  // An allmissing row has no observed patch that can be a legal query. The
  // ordinary mask contract must keep rejecting queries on unobserved patches.
  rejects([&] { snapshot.reconstruct(absent, hidden); }, "querying an entirely unobserved patch");
  check(!snapshot.reconstruct(absent, torch::zeros_like(hidden)).eligible.any().item<bool>(),
      "allmissing no reconstruction eligibility");
  const auto assets = directory / "snapshot-assets"; fs::create_directory(assets); snapshot.features.save_assets(assets.string());
  torch::serialize::InputArchive snapshot_audit; snapshot_audit.load_from((assets / "optimization-snapshot-audit.pt").string(), torch::kCPU);
  check(audit_text(snapshot_audit, "artifact_kind") == "rpb_optimization_diagnostic_snapshot_audit_v1" &&
      audit_text(snapshot_audit, "completed_steps") == "4", "immutable snapshot source audit");
  rejects([&] { snapshot.features.save_assets(assets.string()); }, "snapshot asset overwrite");
  rejects([&] { rpb::resume_optimization_diagnostic(options, fit); }, "continued output overwrite");

  const auto six_path = (directory / "continued-6.pt").string();
  const auto second = rpb::resume_optimization_diagnostic({four_path, raw_path, six_path, 2}, fit);
  check(second.completed_before == 4 && second.completed_after == 6, "new diagnostic parent audit supports second ordinary continuation");
  uninterrupted.train_to(6); const auto direct_six = (directory / "direct-6.pt").string(); uninterrupted.save_checkpoint(direct_six);
  auto actual_six = rpb::load_checkpoint(six_path, c.device), expected_six = rpb::load_checkpoint(direct_six, c.device);
  same_parameters(actual_six.model, expected_six.model, "second segment exact uninterrupted CUDA weights");
  same_adam(adam_states(actual_six, six_path), adam_states(expected_six, direct_six), 6, "second segment exact optimizer state");
  close(snapshot.features.extract(heldout).at("curve_global").values, features.at("curve_global").values,
      "earlier CPU feature snapshot immutable after later continuation", 0, 0);
  const auto retained = snapshot.reconstruct(heldout, hidden);
  close(retained.prediction, reconstruction.prediction, "earlier CUDA reconstruction immutable after later continuation", 0, 0);
  close(retained.eligible, reconstruction.eligible, "earlier snapshot support immutable", 0, 0);
  for (const auto &[path, original] : preserved) check(bytes(path) == original, "original input artifact bytes preserved");
  std::cout << "RPB CUDA optimization diagnostic tests passed; artifacts=" << directory << '\n';
}
} // namespace

int main() {
  try { torch::set_num_threads(1); check(torch::cuda::is_available(), "optimization diagnostic tests require CUDA"); diagnostic_contract(); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
