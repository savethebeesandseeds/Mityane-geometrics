// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/decoder_calibration.h"
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

#ifndef DECODER_CALIBRATION_SOURCE_ID
#define DECODER_CALIBRATION_SOURCE_ID "unrecorded"
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;

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
    check(threads == at::get_num_threads(), label + " threads restored");
    for (size_t i = 0; i < generators.size(); ++i)
      check(torch::equal(states[i], generators[i].get_state()), label + " RNG restored");
  }
};
void equal(const torch::Tensor &a, const torch::Tensor &b, const std::string &label) {
  check(a.scalar_type() == b.scalar_type() && torch::equal(a, b), label);
}
std::string text(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor v; a.read(key, v, true); return embedding::archive::tensor_text(v);
}
torch::Tensor tensor(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor v; a.read(key, v, true); return v;
}
void compare_named(torch::serialize::InputArchive &a, torch::serialize::InputArchive &b, const std::string &key,
                   const std::vector<std::string> &fields = {"value"}) {
  torch::serialize::InputArchive x, y; a.read(key, x); b.read(key, y);
  const auto count = tensor(x, "count").item<int64_t>();
  check(count == tensor(y, "count").item<int64_t>(), "named count " + key);
  for (int64_t i = 0; i < count; ++i) {
    torch::serialize::InputArchive u, v; x.read("tensor_" + std::to_string(i), u); y.read("tensor_" + std::to_string(i), v);
    check(text(u, "parameter_name") == text(v, "parameter_name"), "named identity " + key);
    for (const auto &field : fields) equal(tensor(u, field), tensor(v, field), "exact named tensor " + key + '/' + field);
  }
}
void same_reconstruction(const ev::CurveReconstruction &a, const ev::CurveReconstruction &b, const std::string &label) {
  equal(a.prediction, b.prediction, label + " prediction"); equal(a.target, b.target, label + " target"); equal(a.eligible, b.eligible, label + " support");
}
void prefix(const rpb::DecoderCalibrationProgress &a, const rpb::DecoderCalibrationProgress &b) {
  check(a.losses.size() <= b.losses.size(), "every-update prefix length");
  for (size_t i = 0; i < a.losses.size(); ++i) {
    const auto &x = a.losses[i], &y = b.losses[i];
    check(x.absolute_attempt == y.absolute_attempt && x.completed == y.completed && x.target_cells == y.target_cells &&
        x.eligible_examples == y.eligible_examples && x.loss == y.loss && x.gradient_norm == y.gradient_norm, "every-update exact prefix");
  }
}
void calibration_cuda() {
  auto settings = rpb::default_settings(); auto &c = settings.model;
  c.device = torch::Device(torch::kCUDA, 0); c.global_bottleneck_mode = 2; c.channel_mixer_layers = 1;
  c.channel_ids = {101, 202, 303}; c.dropout = 0; c.sampling_interval = .5;
  settings.steps = 4; settings.batch_size = 2; settings.log_every = 1; settings.attempt_limit = 32;
  auto raw = input(c, 8); const auto order = torch::tensor({2, 0, 1}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order); raw.channel_ids = raw.channel_ids.index_select(0, order);
  raw.observed[0][0][15][1] = false; raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch legal{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{legal, {3, 32, 3, torch::kFloat64, torch::kCPU}, 4404,
      {"a", "a", "b", "b", "c", "c", "d", "d"}, {303, 101, 202}, "volts,amperes,kelvin",
      "native-development-v1/lag_sign", .5, raw.endpoints[0].item<double>()};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-decoder-calibration-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory), "exclusive admission fixture directory");
  auto trainer = rpb::make_learning_curve_trainer(settings, {true, rpb::ContextDeletionRecipe::coordinate15_v1})(fit);
  const auto parent_progress = trainer.train_to(4);
  check(parent_progress.completed == 4 && parent_progress.attempted == 4 && parent_progress.last_input_cuda && parent_progress.finite_gradients,
      "actual unskipped CUDA v7 fixture parent");
  const auto parent = (directory / "parent-v7.pt").string(); trainer.save_checkpoint(parent);
  std::map<std::string, std::string> original_bytes;
  for (const std::string suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt"}) original_bytes.emplace(parent + suffix, bytes(parent + suffix));
  const rpb::DecoderCalibrationOptions options{parent, 4, 4};
  const RuntimeWitness factory_rng;
  auto split = rpb::make_decoder_calibration(options, fit), direct = rpb::make_decoder_calibration(options, fit);
  factory_rng.unchanged("calibration factory");
  const auto zero = split.train_to(0);
  check(zero.original_encoder_completed == 4 && zero.original_encoder_attempted == 4 && zero.completed == 0 && zero.next_absolute_attempt == 4 &&
      zero.decoder_parameter_count == 11528 && zero.frozen_parameter_count == 214277 && zero.encoder_parameters_exact &&
      zero.encoder_buffers_exact && zero.scaler_exact && zero.native_exports_exact && zero.zero_encoder_gradients && !zero.decoder_weights_changed,
      "exact frozen 214277 + fresh decoder 11528 stage zero");
  const auto stage0 = (directory / "decoder-stage0.pt").string(); split.save(stage0);
  const auto snapshot0 = split.snapshot(stage0);
  const auto initial_native = snapshot0.features.extract(legal);
  auto hidden = torch::zeros_like(raw.observed); hidden.slice(2, 0, c.patch_length).fill_(true);
  const auto initial_query = snapshot0.reconstruct(legal, hidden);
  auto checkpoint = rpb::load_checkpoint(parent, c.device); checkpoint.model->eval();
  for (auto &p : checkpoint.model->parameters()) p.set_requires_grad(false);
  { torch::NoGradGuard no_grad;
    const auto normalized = checkpoint.scaler.transform(raw, c); const auto expected = checkpoint.model->forward(normalized, hidden);
    same_reconstruction(initial_query, {expected.reconstruction.cpu(), normalized.data.cpu(), expected.eligible_channels.cpu()}, "exact frozen CUDA stage0 parent");
    const auto encoding = checkpoint.model->encode(normalized);
    equal(initial_native.at("curve_global").values, encoding.z_contextual_global.cpu(), "exact CUDA stage0 native32 parent");
  }
  const RuntimeWitness update_rng;
  const auto one = split.train_to(1); const auto stage1 = (directory / "decoder-stage1.pt").string(); split.save(stage1);
  const auto snapshot1 = split.snapshot(stage1); const auto query1 = snapshot1.reconstruct(legal, hidden);
  const auto two = split.train_to(2); const auto four = split.train_to(4); const auto direct4 = direct.train_to(4);
  update_rng.unchanged("CUDA updates, artifacts and snapshots"); prefix(one, two); prefix(two, four);
  check(four.original_encoder_completed == 4 && four.completed == 4 && four.attempted == 4 && four.sampled_rows == 8 && four.next_absolute_attempt == 8 &&
      four.last_input_cuda && four.last_latent_cuda && four.last_loss_cuda && four.finite_decoder_gradients && four.decoder_weights_changed &&
      four.encoder_parameters_exact && four.encoder_buffers_exact && four.scaler_exact && four.native_exports_exact && four.zero_encoder_gradients &&
      four.training_seconds > 0, "real CUDA decoder updates with exact frozen encoder/exports");
  prefix(four, direct4); check(four.losses.size() == direct4.losses.size(), "direct/split full trace equality");
  const auto stage4 = (directory / "decoder-stage4.pt").string(), direct_path = (directory / "decoder-direct4.pt").string();
  split.save(stage4); direct.save(direct_path);
  torch::serialize::InputArchive a, b; a.load_from(stage4, torch::kCPU); b.load_from(direct_path, torch::kCPU);
  check(text(a, "artifact_kind") == rpb::kDecoderCalibrationArtifact && tensor(a, "decoder_completed").item<int64_t>() == 4 &&
      tensor(a, "original_encoder_completed").item<int64_t>() == 4, "typed artifact keeps original and additional budgets distinct");
  for (const std::string key : {"decoder_after", "encoder_before", "encoder_after", "buffers_before", "buffers_after", "native_before", "native_after"}) compare_named(a, b, key);
  compare_named(a, b, "decoder_optimizer", {"step", "exp_avg", "exp_avg_sq"});
  for (const std::string key : {"sampled_indices", "absolute_attempts", "prediction", "normalized_target", "target_support", "visible_support",
      "artificial_hidden", "eligible_channels", "detached_native32", "loss_trace"}) equal(tensor(a, key), tensor(b, key), "exact direct/split evidence " + key);
  const auto attempts = tensor(a, "absolute_attempts");
  equal(attempts, torch::arange(4, 8, torch::kInt64), "original absolute counters continue 4..7 with fresh Adam steps1..4");
  for (int64_t i = 0; i < 4; ++i) {
    const auto indices = rpb::training_detail::sampled_indices(8, 2, static_cast<int64_t>(fit.seed), 4 + i);
    equal(tensor(a, "sampled_indices")[i], indices, "unchanged original row counter stream");
    const auto selected = rpb::training_detail::selected(raw, indices); const auto normalized = checkpoint.scaler.transform(selected, c);
    const auto masks = rpb::make_training_mask(normalized.observed, c, rpb::training_detail::counter_seed(fit.seed, 4 + i, 0x6d61736bULL));
    equal(tensor(a, "target_support")[i], masks.target.cpu(), "original Q unchanged, no context E");
    const auto losses = rpb::hierarchical_huber(tensor(a, "prediction")[i].to(c.device), tensor(a, "normalized_target")[i].to(c.device),
        tensor(a, "target_support")[i].to(c.device), tensor(a, "eligible_channels")[i].to(c.device), 1.0);
    check(losses.loss.item<double>() == four.losses[static_cast<size_t>(i)].loss,
        "saved original hierarchical Huber arithmetic");
  }
  same_reconstruction(snapshot0.reconstruct(legal, hidden), initial_query, "stage0 immutable after all updates");
  same_reconstruction(snapshot1.reconstruct(legal, hidden), query1, "stage1 immutable after all updates");
  const auto final_snapshot = split.snapshot(stage4);
  const auto reload = rpb::make_decoder_calibration_snapshot(stage4, parent, fit);
  same_reconstruction(final_snapshot.reconstruct(legal, hidden), reload.reconstruct(legal, hidden), "typed decoder reload CUDA parity");
  const auto final_native = final_snapshot.features.extract(legal);
  for (const auto &[name, feature] : initial_native) {
    equal(feature.values, final_native.at(name).values, "exact native values frozen " + name);
    equal(feature.valid, final_native.at(name).valid, "exact native support frozen " + name);
  }
  auto changed = legal; changed.data = legal.data.clone(); changed.data.masked_fill_(hidden, 1e6);
  equal(final_snapshot.reconstruct(changed, hidden).prediction, final_snapshot.reconstruct(legal, hidden).prediction, "hidden target values cannot reach served encoder/decoder");
  auto missing = legal; missing.data = torch::full_like(legal.data, std::numeric_limits<double>::quiet_NaN()); missing.feature_mask = torch::zeros_like(legal.feature_mask);
  const auto absent = final_snapshot.features.extract(missing).at("curve_global");
  check(!absent.valid.any().item<bool>() && absent.values.eq(0).all().item<bool>(), "no inferred all-absent native support");
  rejects([&] { split.save(stage4); }, "typed artifact overwrite");
  rejects([&] { split.save(parent); }, "typed artifact cannot replace immutable parent");
  rejects([&] { rpb::load_checkpoint(stage4, c.device); }, "typed decoder artifact cannot masquerade as ordinary checkpoint");
  std::vector<std::string> resume_args{"rpb", "train", "--resume", stage4, "--checkpoint", (directory / "forbidden-ordinary.pt").string(),
      "--input", (directory / "never-read-input.pt").string(), "--steps", "1", "--device", "cuda"};
  std::vector<char *> resume_argv; for (auto &arg : resume_args) resume_argv.push_back(arg.data());
  rejects([&] { rpb::run_cli(static_cast<int>(resume_argv.size()), resume_argv.data()); }, "ordinary CLI rejects typed decoder resume before TRAIN access");
  check(!fs::exists(directory / "never-read-input.pt") && !fs::exists(directory / "forbidden-ordinary.pt"), "rejected resume cannot write/read TRAIN artifacts");
  rejects([&] { split.train_to(5); }, "bounded decoder budget");
  auto wrong = fit; wrong.seed = 5505; rejects([&] { rpb::make_decoder_calibration(options, wrong); }, "wrong original seed");
  wrong = fit; std::swap(wrong.training_source_ids[0], wrong.training_source_ids[2]);
  rejects([&] { rpb::make_decoder_calibration(options, wrong); }, "wrong original source order");
  wrong = fit; wrong.feature_units = "changed"; rejects([&] { rpb::make_decoder_calibration(options, wrong); }, "wrong units");
  wrong = fit; wrong.training_observations.data = fit.training_observations.data.clone(); wrong.training_observations.data[0][0][0][0] += 1;
  rejects([&] { rpb::make_decoder_calibration(options, wrong); }, "wrong TRAIN observation");
  wrong = fit; wrong.training_observations.feature_mask = fit.training_observations.feature_mask.clone(); wrong.training_observations.feature_mask[1][0][0][0] = false;
  rejects([&] { rpb::make_decoder_calibration(options, wrong); }, "wrong original observation support");
  rejects([&] { rpb::make_decoder_calibration({parent, 3, 4}, fit); }, "wrong parent update budget");
  auto ordinary = rpb::make_learning_curve_trainer(settings)(fit); ordinary.train_to(4);
  const auto ordinary_path = (directory / "ordinary-parent.pt").string(); ordinary.save_checkpoint(ordinary_path);
  rejects([&] { rpb::make_decoder_calibration({ordinary_path, 4, 4}, fit); }, "ordinary parent policy");
  rejects([&] { rpb::make_decoder_calibration_snapshot(stage4, ordinary_path, fit); }, "wrong composed parent");
  const RuntimeWitness invalid_rng;
  rejects([&] { rpb::make_decoder_calibration({parent, 0, 4}, fit); }, "invalid parent budget before RNG mutation");
  invalid_rng.unchanged("invalid calibration arguments");
  for (const auto &[path, original] : original_bytes) check(bytes(path) == original, "immutable original v7 bytes " + path);
  check(!fs::exists(directory / "forbidden-ordinary.pt"), "no ordinary encoder replacement");
}
} // namespace

int main() {
  try {
    check(torch::cuda::is_available(), "actual CUDA mandatory for decoder calibration admission");
    at::set_num_threads(1); calibration_cuda();
    std::cout << "DECODER_CALIBRATION_SOURCE_ID=" << DECODER_CALIBRATION_SOURCE_ID << '\n';
    std::cout << "V7 decoder calibration CUDA admission passed\n"; return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
