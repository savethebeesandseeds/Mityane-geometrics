// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h"
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
#include <sstream>

#ifndef FROZEN_DECODER_CALIBRATION_SOURCE_ID
#ifdef EVALUATION_SOURCE_ID
#define FROZEN_DECODER_CALIBRATION_SOURCE_ID EVALUATION_SOURCE_ID
#else
#define FROZEN_DECODER_CALIBRATION_SOURCE_ID "unrecorded"
#endif
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;
using Policy = rpb::FrozenDecoderParentPolicy;

std::string bytes(const std::string &path) {
  std::ifstream stream(path, std::ios::binary); check(bool(stream), "fixture readable");
  std::ostringstream out; out << stream.rdbuf(); return out.str();
}
struct RuntimeWitness {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeWitness() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  void unchanged(const std::string &label) const {
    check(threads == at::get_num_threads(), label + " threads unchanged");
    for (size_t i = 0; i < generators.size(); ++i)
      check(torch::equal(states[i], generators[i].get_state()), label + " RNG unchanged");
  }
};
struct FixtureRuntimeIsolation {
  RuntimeWitness witness;
  ~FixtureRuntimeIsolation() noexcept {
    try {
      for (size_t i = 0; i < witness.generators.size(); ++i) witness.generators[i].set_state(witness.states[i]);
      at::set_num_threads(witness.threads);
    } catch (...) { std::terminate(); }
  }
};
template <typename Function> auto isolated_legacy_fixture(Function function) {
  // Existing continuous-trainer fixtures deliberately seed global generators.
  // Their historical contract does not promise ambient RNG isolation. Restore
  // only this test-owned setup; NEW decoder callbacks below run unwrapped.
  const FixtureRuntimeIsolation isolation;
  return function();
}
rpb::Checkpoint load_fixture(const std::string &path, const torch::Device &device) {
  return isolated_legacy_fixture([&] { return rpb::load_checkpoint(path, device); });
}
void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &label) {
  check(a.scalar_type() == b.scalar_type() && torch::equal(a, b), label);
}
torch::Tensor tensor(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor value; a.read(key, value, true); return value;
}
std::string text(torch::serialize::InputArchive &a, const std::string &key) {
  return embedding::archive::tensor_text(tensor(a, key));
}
void named_equal(torch::serialize::InputArchive &a, torch::serialize::InputArchive &b,
                 const std::string &key, const std::vector<std::string> &fields = {"value"}) {
  torch::serialize::InputArchive x, y; a.read(key, x); b.read(key, y);
  const auto count = tensor(x, "count").item<int64_t>();
  check(count == tensor(y, "count").item<int64_t>(), "named group count " + key);
  for (int64_t i = 0; i < count; ++i) {
    torch::serialize::InputArchive u, v; x.read("tensor_" + std::to_string(i), u); y.read("tensor_" + std::to_string(i), v);
    check(text(u, "parameter_name") == text(v, "parameter_name"), "named identity " + key);
    for (const auto &field : fields) exact(tensor(u, field), tensor(v, field), "named exact " + key + '/' + field);
  }
}
void assets_equal(const std::string &a_path, const std::string &b_path) {
  torch::serialize::InputArchive a, b; a.load_from(a_path, torch::kCPU); b.load_from(b_path, torch::kCPU);
  for (const std::string key : {"decoder_before", "decoder_after", "encoder_before", "encoder_after", "buffers_before", "buffers_after",
      "native_before", "native_after"}) named_equal(a, b, key);
  named_equal(a, b, "decoder_optimizer", {"step", "exp_avg", "exp_avg_sq"});
  for (const std::string key : {"sampled_indices", "absolute_attempts", "prediction", "normalized_target", "target_support", "visible_support",
      "artificial_hidden", "eligible_channels", "detached_native32", "loss_trace"}) exact(tensor(a, key), tensor(b, key), "exact numeric evidence " + key);
}
void same_query(const ev::CurveReconstruction &a, const ev::CurveReconstruction &b, const std::string &label) {
  exact(a.prediction, b.prediction, label + " predictions"); exact(a.target, b.target, label + " targets"); exact(a.eligible, b.eligible, label + " eligible");
}
void same_features(const ev::FeatureMap &a, const ev::FeatureMap &b, const std::string &label) {
  check(a.size() == b.size(), label + " feature keys");
  for (const auto &[name, feature] : a) {
    exact(feature.values, b.at(name).values, label + " native " + name);
    exact(feature.valid, b.at(name).valid, label + " support " + name);
  }
}
void same_models(const rpb::Model &a, const rpb::Model &b) {
  const auto x = a->named_parameters(), y = b->named_parameters();
  check(x.size() == y.size(), "all initialized parameter names match");
  int64_t count = 0;
  for (const auto &p : x) { exact(p.value(), y[p.key()], "exact common initialization including global pool " + p.key()); count += p.value().numel(); }
  check(count == 225805, "all 225805 initialized parameters paired");
  const auto u = a->named_buffers(), v = b->named_buffers(); check(u.size() == v.size(), "initialized buffer names match");
  for (const auto &p : u) exact(p.value(), v[p.key()], "initialized buffer " + p.key());
}
void assert_progress(const rpb::DecoderCalibrationProgress &p) {
  check(p.original_encoder_attempted == 4 && p.original_encoder_completed == 4 && p.attempted == 4 && p.completed == 4 &&
      p.next_absolute_attempt == 8 && p.sampled_rows == 8 && p.decoder_parameter_count == 11528 && p.frozen_parameter_count == 214277 &&
      p.encoder_parameters_exact && p.encoder_buffers_exact && p.scaler_exact && p.native_exports_exact && p.zero_encoder_gradients &&
      p.last_input_cuda && p.last_latent_cuda && p.last_loss_cuda && p.finite_decoder_gradients && p.decoder_weights_changed && p.training_seconds > 0,
      "actual CUDA decoder-only updates, original encoder counters and all exact frozen witnesses");
}
void ordinary_resume_rejects(const std::string &path, const fs::path &directory) {
  // This auxiliary historical CLI rejection is outside the new callback RNG
  // contract, while its rejection and no-output assertions remain unchanged.
  const FixtureRuntimeIsolation isolation;
  std::vector<std::string> args{"rpb", "train", "--resume", path, "--checkpoint", (directory / "forbidden.pt").string(),
      "--input", (directory / "never-read.pt").string(), "--steps", "1", "--device", "cuda"};
  std::vector<char *> argv; for (auto &arg : args) argv.push_back(arg.data());
  rejects([&] { rpb::run_cli(static_cast<int>(argv.size()), argv.data()); }, "ordinary workflow rejects fresh typed decoder artifact");
  check(!fs::exists(directory / "forbidden.pt") && !fs::exists(directory / "never-read.pt"), "typed rejection before TRAIN access/output");
}
void cuda_contract() {
  auto settings = rpb::default_settings(); auto &c = settings.model;
  c.device = torch::Device(torch::kCUDA, 0); c.global_bottleneck_mode = 2; c.channel_mixer_layers = 1;
  c.channel_ids = {101, 202, 303}; c.sampling_interval = .5; c.dropout = 0;
  settings.steps = 4; settings.batch_size = 2; settings.log_every = 1; settings.attempt_limit = 16;
  auto raw = input(c, 8); const auto order = torch::tensor({2, 0, 1}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order); raw.channel_ids = raw.channel_ids.index_select(0, order);
  raw.observed[0][0][15][1] = false; raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch legal{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{legal, {3, 32, 3, torch::kFloat64, torch::kCPU}, 9109,
      {"fresh-a", "fresh-a", "fresh-b", "fresh-b", "fresh-c", "fresh-c", "fresh-d", "fresh-d"},
      {303, 101, 202}, "volts,amperes,kelvin", "fresh-decoder-replication-v1/lag_sign", .5, raw.endpoints[0].item<double>()};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-fresh-decoder-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory), "exclusive generated CUDA admission directory");
  const RuntimeWitness runtime;
  auto v4 = isolated_legacy_fixture([&] { return rpb::make_learning_curve_trainer(settings)(fit); });
  auto v7 = isolated_legacy_fixture([&] {
    return rpb::make_learning_curve_trainer(settings, {true, rpb::ContextDeletionRecipe::coordinate15_v1})(fit);
  });
  const auto initial4 = (directory / "v4-initial.pt").string(), initial7 = (directory / "v7-initial.pt").string();
  v4.save_checkpoint(initial4); v7.save_checkpoint(initial7);
  const auto cp4 = load_fixture(initial4, c.device), cp7 = load_fixture(initial7, c.device);
  same_models(cp4.model, cp7.model); check(cp4.scaler.identity() == cp7.scaler.identity(), "initial scaler exact pairing");
  for (const auto &[left, right] : std::vector<std::pair<torch::Tensor, torch::Tensor>>{
      {cp4.scaler.mean, cp7.scaler.mean}, {cp4.scaler.scale, cp7.scaler.scale}, {cp4.scaler.count, cp7.scaler.count},
      {cp4.scaler.channel_ids, cp7.scaler.channel_ids}, {cp4.scaler.floor_applied, cp7.scaler.floor_applied}})
    exact(left, right, "all fitted scaler buffers exact pairing");
  check(cp4.scaler.scale_floor == cp7.scaler.scale_floor, "fitted scaler floor exact pairing");
  const RuntimeWitness initial_callbacks;
  const auto zero4 = rpb::make_fresh_initial_snapshot(initial4, Policy::ordinary_v4, fit);
  const auto zero7 = rpb::make_fresh_initial_snapshot(initial7, Policy::coordinate15_v7, fit);
  const auto initial_features = zero4.features.extract(legal);
  same_features(initial_features, zero7.features.extract(legal), "paired CUDA untrained native32");
  auto hidden = torch::zeros_like(raw.observed); hidden.slice(2, 0, 8).fill_(true);
  const auto initial_query = zero4.reconstruct(legal, hidden);
  same_query(initial_query, zero7.reconstruct(legal, hidden), "paired fresh0 CUDA queries");
  rejects([&] { rpb::make_frozen_decoder_calibration({initial4, Policy::ordinary_v4, 0, 4}, fit); }, "calibration cannot start from point0");
  rejects([&] { rpb::make_fresh_initial_snapshot(initial4, Policy::coordinate15_v7, fit); }, "point0 explicit policy mismatch");
  rejects([&] { rpb::make_fresh_initial_snapshot(initial7, Policy::ordinary_v4, fit); }, "point0 context cannot masquerade as ordinary");
  initial_callbacks.unchanged("new CUDA point0 callback lifecycle");
  isolated_legacy_fixture([&] { v4.train_to(4); v7.train_to(4); });
  const auto parent4 = (directory / "v4-parent4.pt").string(), parent7 = (directory / "v7-parent4.pt").string();
  v4.save_checkpoint(parent4); v7.save_checkpoint(parent7);
  same_features(initial_features, zero4.features.extract(legal), "point0 remains immutable after encoder training");
  same_query(initial_query, zero4.reconstruct(legal, hidden), "point0 query remains immutable after encoder training");
  rejects([&] { rpb::make_fresh_initial_snapshot(parent4, Policy::ordinary_v4, fit); }, "initial helper rejects positive checkpoint");
  std::map<std::string, std::string> protected_bytes;
  for (const auto &path : {initial4, initial7, parent4, parent7})
    for (const std::string suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt"}) protected_bytes.emplace(path + suffix, bytes(path + suffix));

  std::map<Policy, std::string> calibrated;
  for (const auto &[policy, path] : std::vector<std::pair<Policy, std::string>>{{Policy::ordinary_v4, parent4}, {Policy::coordinate15_v7, parent7}}) {
    const RuntimeWitness calibration_callbacks;
    const auto name = policy == Policy::ordinary_v4 ? "v4" : "v7";
    const rpb::FrozenDecoderCalibrationOptions options{path, policy, 4, 4};
    auto split = rpb::make_frozen_decoder_calibration(options, fit), direct = rpb::make_frozen_decoder_calibration(options, fit);
    const auto p0 = split.train_to(0); check(p0.completed == 0 && p0.training_seconds == 0 && p0.native_exports_exact && !p0.decoder_weights_changed,
        "fresh stage0 no optimization/cost");
    const auto s0 = (directory / (std::string(name) + "-decoder0.pt")).string(); split.save(s0); const auto snapshot0 = split.snapshot(s0);
    const auto before = snapshot0.features.extract(legal);
    const auto query0 = snapshot0.reconstruct(legal, hidden);
    auto parent = load_fixture(path, c.device); parent.model->eval(); for (auto &p : parent.model->parameters()) p.set_requires_grad(false);
    { torch::NoGradGuard no_grad;
      const auto normalized = parent.scaler.transform(raw, c); const auto expected = parent.model->forward(normalized, hidden);
      same_query(query0, {expected.reconstruction.cpu(), normalized.data.cpu(), expected.eligible_channels.cpu()}, "exact decoder0 parent CUDA parity");
    }
    const auto one = split.train_to(1); const auto s1 = (directory / (std::string(name) + "-decoder1.pt")).string(); split.save(s1);
    const auto snapshot1 = split.snapshot(s1); const auto query1 = snapshot1.reconstruct(legal, hidden);
    split.train_to(2); const auto four = split.train_to(4), uninterrupted = direct.train_to(4);
    assert_progress(four); assert_progress(uninterrupted);
    check(one.losses.front().loss == four.losses.front().loss && one.losses.front().gradient_norm == four.losses.front().gradient_norm,
        "every-update trace preserves queried prefix");
    const auto s4 = (directory / (std::string(name) + "-decoder4.pt")).string(), direct4 = (directory / (std::string(name) + "-direct4.pt")).string();
    split.save(s4); direct.save(direct4); calibrated[policy] = s4; assets_equal(s4, direct4);
    torch::serialize::InputArchive artifact; artifact.load_from(s4, torch::kCPU);
    check(text(artifact, "artifact_kind") == rpb::kFreshDecoderArtifact && text(artifact, "protocol_id") == rpb::kFreshDecoderProtocol &&
        text(artifact, "model_tag") == (policy == Policy::ordinary_v4 ? "RPB-v4" : "RPB-v7") &&
        text(artifact, "training_policy_id") == parent.training_policy_id, "fresh typed format retains each actual encoder identity");
    exact(tensor(artifact, "absolute_attempts"), torch::arange(4, 8, torch::kInt64), "same original absolute row/A/Torch attempts");
    for (int64_t i = 0; i < 4; ++i) {
      const auto indices = rpb::training_detail::sampled_indices(8, 2, fit.seed, 4 + i);
      exact(tensor(artifact, "sampled_indices")[i], indices, "exact row counter");
      const auto normalized = parent.scaler.transform(rpb::training_detail::selected(raw, indices), c);
      const auto masks = rpb::make_training_mask(normalized.observed, c, rpb::training_detail::counter_seed(fit.seed, 4 + i, 0x6d61736bULL));
      exact(tensor(artifact, "target_support")[i], masks.target.cpu(), "original Q unchanged for both decoder phases");
    }
    const auto final = split.snapshot(s4), reloaded = rpb::make_frozen_decoder_calibration_snapshot(s4, path, policy, fit);
    same_query(final.reconstruct(legal, hidden), reloaded.reconstruct(legal, hidden), "independent immutable typed reload");
    same_query(snapshot0.reconstruct(legal, hidden), query0, "decoder0 immutable after updates");
    same_query(snapshot1.reconstruct(legal, hidden), query1, "decoder1 immutable after updates");
    same_features(before, final.features.extract(legal), "CUDA native values/support exact after calibration");
    auto poisoned = legal; poisoned.data = legal.data.clone(); poisoned.data.masked_fill_(hidden, 1e6);
    exact(final.reconstruct(poisoned, hidden).prediction, final.reconstruct(legal, hidden).prediction, "hidden target storage cannot enter decoder signal");
    rejects([&] { split.save(s4); }, "fresh typed asset overwrite"); rejects([&] { split.save(path); }, "parent replacement");
    ordinary_resume_rejects(s4, directory);
    rejects([&] { rpb::make_frozen_decoder_calibration_snapshot(s4, path,
        policy == Policy::ordinary_v4 ? Policy::coordinate15_v7 : Policy::ordinary_v4, fit); }, "typed load requires explicit matching policy");
    calibration_callbacks.unchanged("new CUDA decoder callback lifecycle");
  }

  // A metadata namespace change cannot change any numerical mechanics. The old
  // API and its positive-v7 format are exercised against the shared fresh v7
  // route using a separate generated parent with exactly the same values/seed.
  auto historical_fit = fit; historical_fit.protocol_id = "native-development-v1/lag_sign";
  auto historical_trainer = isolated_legacy_fixture([&] {
    return rpb::make_learning_curve_trainer(settings, {true, rpb::ContextDeletionRecipe::coordinate15_v1})(historical_fit);
  });
  isolated_legacy_fixture([&] { historical_trainer.train_to(4); });
  const auto historical_parent = (directory / "historical-v7-parent4.pt").string();
  historical_trainer.save_checkpoint(historical_parent);
  auto historical = rpb::make_decoder_calibration({historical_parent, 4, 4}, historical_fit); historical.train_to(4);
  const auto old_asset = (directory / "historical-decoder4.pt").string(); historical.save(old_asset);
  assets_equal(old_asset, calibrated.at(Policy::coordinate15_v7));
  rejects([&] { rpb::make_decoder_calibration_snapshot(calibrated.at(Policy::coordinate15_v7), historical_parent, historical_fit); }, "historical loader rejects fresh artifact");
  rejects([&] { rpb::make_frozen_decoder_calibration_snapshot(old_asset, parent7, Policy::coordinate15_v7, fit); }, "fresh loader rejects historical artifact");
  rejects([&] { rpb::make_decoder_calibration({initial7, 0, 4}, historical_fit); }, "historical API still rejects initial parent");
  auto wrong = fit; wrong.protocol_id = historical_fit.protocol_id;
  rejects([&] { rpb::make_frozen_decoder_calibration({parent7, Policy::coordinate15_v7, 4, 4}, wrong); }, "historical namespace not fresh");
  wrong = fit; wrong.seed += 1;
  rejects([&] { rpb::make_frozen_decoder_calibration({parent4, Policy::ordinary_v4, 4, 4}, wrong); }, "wrong fresh counter seed");
  wrong = fit; std::swap(wrong.training_source_ids[0], wrong.training_source_ids[2]);
  rejects([&] { rpb::make_frozen_decoder_calibration({parent7, Policy::coordinate15_v7, 4, 4}, wrong); }, "wrong fresh source order");
  wrong = fit; wrong.training_observations.data = fit.training_observations.data.clone(); wrong.training_observations.data[0][0][0][0] += 1;
  rejects([&] { rpb::make_frozen_decoder_calibration({parent4, Policy::ordinary_v4, 4, 4}, wrong); }, "wrong fresh legal TRAIN");
  const RuntimeWitness invalid;
  rejects([&] { rpb::make_frozen_decoder_calibration({}, fit); }, "implicit/invalid policy forbidden");
  rejects([&] { rpb::make_fresh_initial_snapshot(initial4, static_cast<Policy>(99), fit); }, "invalid initial policy forbidden");
  invalid.unchanged("invalid selectors reject before RNG changes");
  runtime.unchanged("fresh initialization/calibration/snapshots/history preserve ambient runtime");
  for (const auto &[path, original] : protected_bytes) check(bytes(path) == original, "all generated immutable parent bytes preserved");
}
} // namespace

int main() {
  try {
    check(torch::cuda::is_available(), "actual CUDA mandatory"); at::set_num_threads(1); cuda_contract();
    std::cout << "FROZEN_DECODER_CALIBRATION_SOURCE_ID=" << FROZEN_DECODER_CALIBRATION_SOURCE_ID << '\n';
    std::cout << "Fresh decoder replication CUDA admission passed\n"; return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
