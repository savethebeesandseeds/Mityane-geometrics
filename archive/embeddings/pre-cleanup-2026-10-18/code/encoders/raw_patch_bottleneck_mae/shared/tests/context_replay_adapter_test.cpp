// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replay_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;

std::string text(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true); return embedding::archive::tensor_text(value);
}
int64_t integer(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true);
  check(value.scalar_type() == torch::kInt64 && value.numel() == 1, "typed replay counter");
  return value.item<int64_t>();
}
double floating(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true);
  check(value.scalar_type() == torch::kFloat64 && value.numel() == 1 && torch::isfinite(value).all().item<bool>(), "typed replay time");
  return value.item<double>();
}
std::string file_bytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary); check(bool(input), "fixture file readable");
  std::ostringstream out; out << input.rdbuf(); return out.str();
}
void same_model(const rpb::Model &a, const rpb::Model &b) {
  const auto x = a->named_parameters(), y = b->named_parameters();
  check(x.size() == y.size(), "exact model named parameter set");
  for (const auto &parameter : x)
    check(y.contains(parameter.key()) && torch::equal(parameter.value(), y[parameter.key()]), "exact model value: " + parameter.key());
  const auto u = a->named_buffers(), v = b->named_buffers();
  check(u.size() == v.size(), "exact model buffer set");
  for (const auto &buffer : u)
    check(v.contains(buffer.key()) && torch::equal(buffer.value(), v[buffer.key()]), "exact buffer value: " + buffer.key());
}
void same_optimizer(rpb::Checkpoint &a, const std::string &ap, rpb::Checkpoint &b, const std::string &bp) {
  torch::optim::AdamW x(a.model->parameters(), torch::optim::AdamWOptions(a.settings.learning_rate).weight_decay(a.settings.weight_decay));
  torch::optim::AdamW y(b.model->parameters(), torch::optim::AdamWOptions(b.settings.learning_rate).weight_decay(b.settings.weight_decay));
  rpb::load_optimizer(ap, x, a.settings.model.device); rpb::load_optimizer(bp, y, b.settings.model.device);
  check(x.state().size() == y.state().size() && !x.state().empty(), "exact active optimizer state count");
  const auto yp = b.model->named_parameters();
  for (const auto &parameter : a.model->named_parameters()) {
    const auto i = x.state().find(parameter.value().unsafeGetTensorImpl());
    const auto j = y.state().find(yp[parameter.key()].unsafeGetTensorImpl());
    check((i == x.state().end()) == (j == y.state().end()), "same named active state support");
    if (i == x.state().end()) continue;
    const auto *u = dynamic_cast<const torch::optim::AdamWParamState *>(i->second.get());
    const auto *v = dynamic_cast<const torch::optim::AdamWParamState *>(j->second.get());
    check(u && v && u->step() == v->step() && u->step() == a.completed_steps &&
        u->exp_avg().is_cuda() && u->exp_avg_sq().is_cuda() &&
        torch::equal(u->exp_avg(), v->exp_avg()) && torch::equal(u->exp_avg_sq(), v->exp_avg_sq()),
        "exact CUDA optimizer continuation by parameter name");
  }
}
void same_trace(const ev::CurveProgress &a, const ev::CurveProgress &b) {
  check(a.attempted == b.attempted && a.completed == b.completed && a.sampled_rows == b.sampled_rows &&
      a.preprocessing_id == b.preprocessing_id && a.training_dataset_id == b.training_dataset_id && a.losses.size() == b.losses.size(),
      "exact replay counters/scaler/source/full trace");
  for (size_t i = 0; i < a.losses.size(); ++i) {
    const auto &x = a.losses[i], &y = b.losses[i];
    check(x.attempted == y.attempted && x.completed == y.completed && x.target_cells == y.target_cells &&
        x.loss == y.loss && x.gradient_norm == y.gradient_norm, "exact full replay numerical trace");
  }
}
void freeze(rpb::Checkpoint &checkpoint) {
  checkpoint.model->eval(); for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(false);
}
void copied_companions(const std::string &parent, const std::string &copy) {
  for (const std::string suffix : {".training-raw.pt", ".scaler.pt", ".audit.pt"})
    fs::copy_file(parent + suffix, copy + suffix, fs::copy_options::none);
}
void changed_parent(const std::string &parent, const std::string &copy, const torch::Device &device,
                    bool change_moment, bool change_policy = false) {
  auto saved = rpb::load_checkpoint(parent, device);
  torch::optim::AdamW optimizer(saved.model->parameters(),
      torch::optim::AdamWOptions(saved.settings.learning_rate).weight_decay(saved.settings.weight_decay));
  rpb::load_optimizer(parent, optimizer, device);
  if (change_policy) saved.training_policy_id.clear();
  else if (change_moment) {
    auto *state = dynamic_cast<torch::optim::AdamWParamState *>(optimizer.state().begin()->second.get());
    check(state != nullptr, "Adam fixture state exists"); state->exp_avg().add_(.125);
  } else {
    torch::NoGradGuard no_grad; saved.model->named_parameters()["global_pool_second.weight"].add_(.125);
  }
  rpb::save_checkpoint(copy, saved, optimizer); copied_companions(parent, copy);
}
void state_witnesses(torch::serialize::InputArchive &archive) {
  torch::serialize::InputArchive model, parent_model, optimizer, parent_optimizer, scaler, parent_scaler;
  archive.read("model", model); archive.read("parent_model", parent_model);
  archive.read("optimizer", optimizer); archive.read("parent_optimizer", parent_optimizer);
  archive.read("scaler", scaler); archive.read("parent_scaler", parent_scaler);
  check(integer(model, "parameter_count") == integer(parent_model, "parameter_count") &&
      integer(model, "buffer_count") == integer(parent_model, "buffer_count"), "typed exact model witness counts");
  for (int64_t i = 0; i < integer(model, "parameter_count"); ++i) {
    torch::serialize::InputArchive a, b; torch::Tensor x, y;
    model.read("parameter_" + std::to_string(i), a); parent_model.read("parameter_" + std::to_string(i), b);
    a.read("value", x, true); b.read("value", y, true);
    check(x.device().is_cpu() && text(a, "parameter_name") == text(b, "parameter_name") && torch::equal(x, y), "portable exact model witness");
  }
  check(integer(optimizer, "parameter_count") == integer(parent_optimizer, "parameter_count") &&
      integer(optimizer, "parameter_count") > 0, "typed active Adam witness count");
  for (int64_t i = 0; i < integer(optimizer, "parameter_count"); ++i) {
    torch::serialize::InputArchive a, b; torch::Tensor x, y;
    optimizer.read("parameter_" + std::to_string(i), a); parent_optimizer.read("parameter_" + std::to_string(i), b);
    check(text(a, "parameter_name") == text(b, "parameter_name") && integer(a, "step") == 2 && integer(b, "step") == 2,
        "portable exact optimizer name/step witness");
    for (const std::string field : {"exp_avg", "exp_avg_sq"}) {
      a.read(field, x, true); b.read(field, y, true);
      check(x.device().is_cpu() && torch::equal(x, y), "portable exact moment witness");
    }
  }
  const auto a = rpb::FrozenScaler::load(scaler), b = rpb::FrozenScaler::load(parent_scaler);
  check(a.identity() == b.identity() && torch::equal(a.mean, b.mean) && torch::equal(a.scale, b.scale) &&
      torch::equal(a.count, b.count) && torch::equal(a.floor_applied, b.floor_applied) &&
      torch::equal(a.channel_ids, b.channel_ids) && a.scale_floor == b.scale_floor, "portable full exact scaler witness");
}

void cuda_replay_gate_contract() {
  auto c = config(); c.device = torch::Device(torch::kCUDA, 0); c.global_bottleneck_mode = 2;
  c.channel_mixer_layers = 1; c.export_width = 32; c.dropout = .1; c.sampling_interval = .5;
  auto settings = rpb::default_settings(); settings.model = c; settings.steps = 2;
  settings.batch_size = 2; settings.attempt_limit = 32; settings.log_every = 3;
  auto raw = input(c, 6); const auto order = torch::tensor({1, 0}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order);
  raw.channel_ids = raw.channel_ids.index_select(0, order);
  raw.observed[0][0][14][1] = false; raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch train{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{train, {c.channel_count, c.history_length, c.input_width, torch::kFloat64, torch::kCPU},
      202, {"a", "a", "b", "b", "c", "c"}, {202, 101}, "volts,amperes", "context-replay-contract-v1",
      c.sampling_interval, (c.history_length - 1) * c.sampling_interval};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-context-replay-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  auto original = rpb::make_learning_curve_trainer(settings, rpb::ContextDeletionOptions{true})(fit);
  const auto parent_progress = original.train_to(2);
  const auto parent = (directory / "audited-2.pt").string(); original.save_checkpoint(parent);
  const auto training = parent + ".training-raw.pt";
  std::map<std::string, std::string> protected_bytes;
  for (const std::string suffix : {"", ".audit.pt", ".training-raw.pt", ".scaler.pt"}) protected_bytes[parent + suffix] = file_bytes(parent + suffix);
  auto direct_settings = settings; direct_settings.steps = 6;
  auto direct = rpb::make_learning_curve_trainer(direct_settings, rpb::ContextDeletionOptions{true})(fit);
  const auto direct_two = direct.train_to(2); same_trace(parent_progress, direct_two);
  const auto direct_four = direct.train_to(4); const auto direct_four_path = (directory / "direct-4.pt").string();
  direct.save_checkpoint(direct_four_path);
  const auto replay_path = (directory / "replay-2.pt").string();
  rpb::ContextReplayOptions options{parent, training, replay_path, 2, 6};
  auto cpu_generator = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  auto gpu_generator = at::globalContext().defaultGenerator(at::Device(at::kCUDA, 0));
  const auto cpu_before = cpu_generator.get_state().clone(), gpu_before = gpu_generator.get_state().clone();
  auto factory = rpb::make_context_replay_trainer(options);
  check(torch::equal(cpu_before, cpu_generator.get_state()) && torch::equal(gpu_before, gpu_generator.get_state()),
      "factory archive/model inspection preserves caller RNG");
  auto replay = factory(fit); const auto zero = replay.train_to(0);
  check(zero.completed == 0 && zero.attempted == 0 && zero.sampled_rows == 0, "fresh replay starts at zero");
  const auto zero_path = (directory / "replay-0.pt").string(); replay.save_checkpoint(zero_path);
  const auto zero_snapshot = replay.snapshot(zero_path);
  const auto zero_values = zero_snapshot.features.extract(train).at("curve_global").values.clone();
  const auto replay_two = replay.train_to(2); same_trace(replay_two, direct_two);
  check(replay_two.cuda_parameter_count == replay_two.parameter_count && replay_two.last_input_cuda &&
      replay_two.last_loss_cuda && replay_two.finite_gradients && replay_two.weights_changed, "actual replay CUDA evidence");
  replay.save_checkpoint(replay_path); // Reserved exact gate save is idempotent.
  auto expected_two = rpb::load_checkpoint(parent, c.device), actual_two = rpb::load_checkpoint(replay_path, c.device);
  same_model(actual_two.model, expected_two.model); same_optimizer(actual_two, replay_path, expected_two, parent);
  check(actual_two.settings.steps == 6 && actual_two.settings.seed == 202 && actual_two.attempted_steps == 2 &&
      actual_two.completed_steps == 2 && actual_two.training_policy_id == rpb::context_deletion::policy_id &&
      actual_two.scaler.identity() == expected_two.scaler.identity(), "only continuation ceiling changes; exact parent policy/scaler/counters");
  torch::serialize::InputArchive gate; gate.load_from(replay_path + ".replay-audit.pt", torch::kCPU);
  torch::Tensor passed, contexts, parent_contexts, counters, values;
  gate.read("replay_gate_passed_value", passed, true);
  gate.read("context_counts", contexts, true); gate.read("parent_context_counts", parent_contexts, true);
  gate.read("loss_trace_counters", counters, true); gate.read("loss_trace_values", values, true);
  check(passed.item<bool>() && integer(gate, "attempted_steps_value") == 2 && integer(gate, "completed_steps_value") == 2 &&
      integer(gate, "sampled_rows_value") == 4 && torch::equal(contexts, parent_contexts) && contexts[1].item<int64_t>() > 0 &&
      counters.scalar_type() == torch::kInt64 && values.scalar_type() == torch::kFloat64 && counters.size(0) == 2 &&
      floating(gate, "replay_training_seconds_value") > 0 && floating(gate, "continuation_training_seconds_value") == 0 &&
      text(gate, "gpu_kernel_only_seconds") == "unmeasured", "typed exact gate counts/full trace/honest replay cost");
  state_witnesses(gate);
  check(file_bytes(replay_path + ".replay-audit.json").find("\"gpu_kernel_only_seconds\":null") != std::string::npos,
      "JSON distinguishes unmeasured GPU kernel duration");
  const auto snapshot = replay.snapshot(replay_path);
  const auto before = snapshot.features.extract(train).at("curve_global");
  check(before.values.size(1) == 32 && before.valid.all().item<bool>() &&
      snapshot.features.audit_fields.at("context_replay_replay_gate_passed") == "true", "exact frozen native32 gated snapshot");
  auto hidden = torch::zeros_like(raw.observed); hidden.narrow(2, 0, c.patch_length).fill_(true);
  const auto prediction = snapshot.reconstruct(train, hidden);
  {
    torch::NoGradGuard no_grad;
    auto cpu = rpb::load_checkpoint(replay_path); freeze(cpu);
    const auto encoded = cpu.model->encode(cpu.scaler.transform(raw, cpu.settings.model));
    close(before.values, encoded.z_contextual_global, "snapshot exact saved CPU global32", 0, 0);
    freeze(actual_two); const auto normalized = actual_two.scaler.transform(raw, c);
    const auto expected = actual_two.model->forward(normalized, hidden);
    close(prediction.prediction, expected.reconstruction.to(torch::kCPU), "snapshot exact saved GPU reconstruction", 0, 0);
    check(torch::equal(prediction.eligible, expected.eligible_channels.to(torch::kCPU)), "snapshot exact fixed-query support");
  }
  auto changed = train; changed.data = train.data.clone(); changed.data.masked_fill_(hidden, 987654.0);
  const auto changed_prediction = snapshot.reconstruct(changed, hidden);
  close(prediction.prediction, changed_prediction.prediction, "held-out target values never reach encoder/decoder", 0, 0);
  check(torch::equal(prediction.eligible, changed_prediction.eligible), "hidden target perturbation preserves legal support");
  const auto replay_four = replay.train_to(4); same_trace(replay_four, direct_four);
  const auto four_path = (directory / "replay-4.pt").string(); replay.save_checkpoint(four_path);
  auto saved_four = rpb::load_checkpoint(four_path, c.device), expected_four = rpb::load_checkpoint(direct_four_path, c.device);
  same_model(saved_four.model, expected_four.model); same_optimizer(saved_four, four_path, expected_four, direct_four_path);
  const auto direct_six = direct.train_to(6); const auto direct_six_path = (directory / "direct-6.pt").string(); direct.save_checkpoint(direct_six_path);
  const auto replay_six = replay.train_to(6); same_trace(replay_six, direct_six);
  const auto six_path = (directory / "replay-6.pt").string(); replay.save_checkpoint(six_path);
  auto saved_six = rpb::load_checkpoint(six_path, c.device), expected_six = rpb::load_checkpoint(direct_six_path, c.device);
  same_model(saved_six.model, expected_six.model); same_optimizer(saved_six, six_path, expected_six, direct_six_path);
  close(snapshot.features.extract(train).at("curve_global").values, before.values, "earlier native snapshot immutable after continuation", 0, 0);
  close(snapshot.reconstruct(train, hidden).prediction, prediction.prediction, "earlier GPU reconstruction snapshot immutable", 0, 0);
  close(zero_snapshot.features.extract(train).at("curve_global").values, zero_values, "initial snapshot immutable", 0, 0);
  check(zero_snapshot.features.audit_fields.at("context_replay_replay_gate_passed") == "false" &&
      snapshot.features.audit_fields.at("context_replay_completed_steps") == "2", "snapshot gate/count metadata is point-specific");
  torch::serialize::InputArchive six_audit; six_audit.load_from(six_path + ".replay-audit.pt", torch::kCPU);
  check(floating(six_audit, "replay_training_seconds_value") == replay_two.training_seconds &&
      floating(six_audit, "continuation_training_seconds_value") == replay_six.training_seconds - replay_two.training_seconds &&
      floating(six_audit, "total_fresh_training_seconds_value") == replay_six.training_seconds,
      "replay overhead and live continuation cost are not double-counted");
  for (const auto &[path, bytes] : protected_bytes) check(file_bytes(path) == bytes, "all original input bytes preserved");

  auto wrong_seed = fit; wrong_seed.seed = 303;
  rejects([&] { factory(wrong_seed); }, "wrong original seed before fitting");
  auto wrong_sources = fit; wrong_sources.training_source_ids[0] = "other";
  rejects([&] { factory(wrong_sources); }, "wrong ordered source association before fitting");
  auto wrong_data = fit; wrong_data.training_observations.data = fit.training_observations.data.clone(); wrong_data.training_observations.data[0][0][0][0] += 1;
  rejects([&] { factory(wrong_data); }, "wrong original observed TRAIN values");
  auto wrong_order = fit; wrong_order.channel_ids = {101, 202};
  rejects([&] { factory(wrong_order); }, "wrong physical semantic channel order");
  auto overwrite = options; overwrite.replay_checkpoint = parent;
  rejects([&] { rpb::make_context_replay_trainer(overwrite); }, "protected checkpoint overwrite");
  overwrite.replay_checkpoint = (directory / "fresh-check.pt").string(); overwrite.training_archive = four_path + ".training-raw.pt";
  // Equal legal TRAIN archives from a later point are allowed; changed values are not.
  auto changed_raw = rpb::load_dataset(training, c); changed_raw.input.data = changed_raw.input.data.clone(); changed_raw.input.data[0][0][0][0] += 1;
  changed_raw = rpb::describe_dataset(changed_raw.input, c, changed_raw.feature_units);
  const auto changed_training = (directory / "changed-train.pt").string(); rpb::save_dataset(changed_training, changed_raw);
  overwrite.training_archive = changed_training;
  rejects([&] { rpb::make_context_replay_trainer(overwrite); }, "wrong parent TRAIN archive");
  const auto wrong_policy = (directory / "wrong-policy.pt").string(); changed_parent(parent, wrong_policy, c.device, false, true);
  auto invalid = options; invalid.audited_checkpoint = wrong_policy; invalid.training_archive = wrong_policy + ".training-raw.pt";
  invalid.replay_checkpoint = (directory / "wrong-policy-replay.pt").string();
  rejects([&] { rpb::make_context_replay_trainer(invalid); }, "ordinary parent cannot enter context replay");
  for (const bool change_moment : {false, true}) {
    const auto corrupted = (directory / (change_moment ? "wrong-moment.pt" : "wrong-model.pt")).string();
    changed_parent(parent, corrupted, c.device, change_moment);
    auto mismatched = options; mismatched.audited_checkpoint = corrupted; mismatched.training_archive = corrupted + ".training-raw.pt";
    mismatched.replay_checkpoint = (directory / (change_moment ? "moment-replay.pt" : "model-replay.pt")).string();
    auto blocked = rpb::make_context_replay_trainer(mismatched)(fit);
    rejects([&] { blocked.train_to(4); }, "numerical state mismatch blocks update beyond exact replay gate");
    const auto stopped = rpb::load_checkpoint(mismatched.replay_checkpoint);
    check(stopped.completed_steps == 2 && stopped.attempted_steps == 2, "failed gate never performs continuation update");
    rejects([&] { blocked.train_to(6); }, "failed gate permanently blocks trainer");
    rejects([&] { blocked.save_checkpoint((directory / "must-not-save.pt").string()); }, "failed gate cannot save later point");
  }
  const auto protected_copy = (directory / "byte-protected.pt").string(); fs::copy_file(parent, protected_copy); copied_companions(parent, protected_copy);
  auto protected_options = options; protected_options.audited_checkpoint = protected_copy;
  protected_options.training_archive = protected_copy + ".training-raw.pt"; protected_options.replay_checkpoint = (directory / "byte-replay.pt").string();
  auto protected_trainer = rpb::make_context_replay_trainer(protected_options)(fit);
  { std::ofstream changed_file(protected_copy + ".audit.pt", std::ios::binary | std::ios::app); changed_file << 'x'; }
  rejects([&] { protected_trainer.train_to(4); }, "changed original input bytes block all updates");
  check(!fs::exists(protected_options.replay_checkpoint), "input tamper rejected before replay checkpoint creation");
  std::cout << "[context replay test] exact CUDA2->4->6/state gate, rejection and immutable serving passed; artifacts=" << directory << '\n';
}
} // namespace

int main() {
  try {
    check(torch::cuda::is_available(), "CUDA mandatory; no CPU fallback"); at::set_num_threads(1);
    cuda_replay_gate_contract(); return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
