// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/masking.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"
#include "rpb_test_support.h"
#include <torch/cuda.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;

void same_parameters(const rpb::Model &actual, const rpb::Model &expected,
                     const std::string &label) {
  const auto reference = expected->named_parameters();
  check(actual->named_parameters().size() == reference.size(), label + " parameter count");
  for (const auto &parameter : actual->named_parameters())
    close(parameter.value().to(torch::kCPU), reference[parameter.key()].to(torch::kCPU),
          label + "/" + parameter.key(), 0, 0);
}

std::string audit_text(torch::serialize::InputArchive &archive, const char *name) {
  torch::Tensor value; archive.read(name, value, true);
  return embedding::archive::tensor_text(value);
}

void resume_two_updates(rpb::Checkpoint &checkpoint, const rpb::Dataset &raw,
                        const std::string &path) {
  checkpoint.model->train();
  torch::optim::AdamW optimizer(checkpoint.model->parameters(),
      torch::optim::AdamWOptions(checkpoint.settings.learning_rate)
          .weight_decay(checkpoint.settings.weight_decay));
  rpb::load_optimizer(path, optimizer, checkpoint.settings.model.device);
  check(!optimizer.state().empty(), "saved CUDA AdamW state missing");
  for (const auto &[key, state] : optimizer.state()) {
    (void)key;
    const auto *adam = dynamic_cast<const torch::optim::AdamWParamState *>(state.get());
    check(adam && adam->exp_avg().is_cuda() && adam->exp_avg_sq().is_cuda(),
          "loaded AdamW moments must follow CUDA parameters");
  }
  while (checkpoint.completed_steps < 4) {
    const auto attempt = checkpoint.attempted_steps;
    const auto indices = rpb::training_detail::sampled_indices(raw.input.data.size(0),
        checkpoint.settings.batch_size, checkpoint.settings.seed, attempt);
    const auto batch = checkpoint.scaler.transform(
        rpb::training_detail::selected(raw.input, indices), checkpoint.settings.model);
    const auto mask = rpb::make_training_mask(batch.observed, checkpoint.settings.model,
        rpb::training_detail::counter_seed(checkpoint.settings.seed, attempt, 0x6d61736bULL));
    torch::manual_seed(rpb::training_detail::counter_seed(checkpoint.settings.seed, attempt, 0x746f726368ULL));
    ++checkpoint.attempted_steps;
    if (!mask.eligible_channels.any().item<bool>()) continue;
    optimizer.zero_grad();
    const auto output = checkpoint.model->forward(batch, mask.hidden);
    output.loss.backward();
    torch::nn::utils::clip_grad_norm_(checkpoint.model->parameters(),
        checkpoint.settings.gradient_clip_norm, 2.0, true);
    optimizer.step();
    ++checkpoint.completed_steps;
  }
}

void continuity_and_snapshots(int64_t mixer_layers, int64_t global_mode) {
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-curve-mixer-" + std::to_string(mixer_layers) + "-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  auto c = config(); c.channel_mixer_layers = mixer_layers;
  c.global_bottleneck_mode = global_mode; c.dropout = 0.1;
  c.device = torch::Device(torch::kCUDA, 0); c.sampling_interval = 0.5;
  auto settings = rpb::default_settings(); settings.model = c; settings.steps = 4;
  settings.batch_size = 2; settings.attempt_limit = 16; settings.log_every = 3;
  settings.seed = 101;
  auto raw = input(c, 6);
  const auto order = torch::tensor({1, 0}, torch::kInt64);
  raw.data = raw.data.index_select(1, order);
  raw.observed = raw.observed.index_select(1, order);
  raw.channel_ids = raw.channel_ids.index_select(0, order);
  raw.observed.select(0, 0).select(0, 0).select(0, 2).fill_(false);
  raw.data = raw.data.masked_fill(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch heldout{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{{raw.data, raw.observed},
      {c.channel_count, c.history_length, c.input_width, torch::kFloat64, torch::kCPU},
      202, {"a", "a", "b", "b", "c", "c"}, {202, 101},
      "volts,amperes", "learning-curve-adapter-contract-v1", c.sampling_interval,
      (c.history_length - 1) * c.sampling_interval};
  const auto factory = rpb::make_learning_curve_trainer(settings);
  auto trainer = factory(fit);
  check(trainer.audit_fields.at("actual_training_seed") == "202" &&
      trainer.audit_fields.at("training_device") == "cuda:0" &&
      trainer.audit_fields.at("optimizer_policy").find("continuous") != std::string::npos,
      "actual CUDA seed/session provenance");
  const auto zero = trainer.train_to(0);
  check(zero.completed == 0 && zero.attempted == 0 && zero.sampled_rows == 0 &&
      zero.parameter_count > 0 && zero.parameter_count == zero.cuda_parameter_count &&
      !zero.weights_changed && zero.training_seconds == 0 && zero.losses.empty(),
      "point zero exact initialization and CUDA residence");
  const auto zero_path = (directory / "point-0.pt").string();
  trainer.save_checkpoint(zero_path);
  const auto zero_checkpoint = rpb::load_checkpoint(zero_path);
  torch::manual_seed(rpb::training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL));
  auto cpu_config = c; cpu_config.device = torch::kCPU;
  const auto initialized = rpb::Model(cpu_config);
  same_parameters(zero_checkpoint.model, initialized, "exact common CPU-initialized point zero");
  const auto point_zero = trainer.snapshot(zero_path);
  check(point_zero.features.surfaces.size() == 2 &&
      point_zero.features.surfaces.at("curve_global").kind == ev::SurfaceKind::global &&
      point_zero.features.surfaces.at("curve_channel_concatenation").channel_order == fit.channel_ids,
      "uniform surfaces and semantic concatenation order");
  const auto zero_features = point_zero.features.extract(heldout);
  check(zero_features.size() == 2 && zero_features.count("curve_global") &&
      zero_features.count("curve_channel_concatenation"), "only exact saved point surfaces");

  // Caller mutations must not rewrite permitted fit rows, support or scaler.
  raw.data.fill_(999); raw.observed.fill_(false);
  const auto two = trainer.train_to(2);
  check(two.completed == 2 && two.attempted == 2 && two.sampled_rows == 4 &&
      two.last_input_cuda && two.last_loss_cuda && two.finite_gradients && two.weights_changed &&
      two.training_seconds > 0 && two.preprocessing_id == zero.preprocessing_id &&
      two.training_dataset_id == zero.training_dataset_id && !two.losses.empty(),
      "real CUDA updates with fixed scaler and absolute counters");
  for (const auto &point : two.losses)
    check(point.completed > 0 && point.attempted >= point.completed && point.target_cells > 0 &&
        std::isfinite(point.loss) && std::isfinite(point.gradient_norm), "finite sampled trace");
  const auto two_path = (directory / "point-2.pt").string();
  trainer.save_checkpoint(two_path);
  const auto point_two = trainer.snapshot(two_path);
  const auto two_features = point_two.features.extract(heldout);
  auto hidden = torch::zeros_like(heldout.feature_mask);
  hidden.narrow(2, 0, c.patch_length).fill_(true);
  const auto two_reconstruction = point_two.reconstruct(heldout, hidden);
  check(two_reconstruction.prediction.device().is_cpu() &&
      two_reconstruction.target.device().is_cpu() && !two_reconstruction.prediction.requires_grad() &&
      two_reconstruction.prediction.sizes() == heldout.data.sizes() &&
      two_reconstruction.eligible.scalar_type() == torch::kBool &&
      two_reconstruction.eligible.all().item<bool>(), "frozen standardized reconstruction contract");
  const embedding::Batch changed_target{
      torch::where(hidden, heldout.data + 100, heldout.data), heldout.feature_mask};
  const auto changed_reconstruction = point_two.reconstruct(changed_target, hidden);
  close(changed_reconstruction.prediction, two_reconstruction.prediction,
        "artificially hidden target values cannot enter reconstruction encoder", 0, 0);
  check(!torch::equal(changed_reconstruction.target, two_reconstruction.target),
        "changed targets remain scoring-only standardized targets");
  auto point_two_gpu = rpb::load_checkpoint(two_path, c.device); point_two_gpu.model->eval();
  auto point_two_cpu = rpb::load_checkpoint(two_path); point_two_cpu.model->eval();
  const auto saved_raw = rpb::load_dataset(two_path + ".training-raw.pt", cpu_config);
  check(saved_raw.feature_units == fit.feature_units && saved_raw.dataset_id == zero.training_dataset_id &&
      point_two_cpu.settings.seed == 202 && point_two_cpu.settings.steps == 4 &&
      point_two_cpu.attempted_steps == 2 && point_two_cpu.completed_steps == 2 &&
      point_two_cpu.scaler.identity() == zero.preprocessing_id &&
      point_two_cpu.scaler_fit_dataset_id == saved_raw.dataset_id,
      "ordinary resumable checkpoint settings/fit/scaler association");
  {
    torch::NoGradGuard no_grad;
    const auto normalized_gpu = point_two_gpu.scaler.transform(saved_raw.input, point_two_gpu.settings.model);
    close(two_reconstruction.target, normalized_gpu.data.to(torch::kCPU),
          "frozen training-scaled target values", 0, 0);
    const auto trainable_expected = point_two_gpu.model->forward(normalized_gpu, hidden);
    // CUDA serving is frozen. Match parameter flags as well as eval/no-grad:
    // LibTorch may choose numerically different CUDA paths for trainable weights.
    for (auto &parameter : point_two_gpu.model->parameters()) parameter.set_requires_grad(false);
    const auto expected = point_two_gpu.model->forward(normalized_gpu, hidden);
    const auto trainable_difference =
        (two_reconstruction.prediction - trainable_expected.reconstruction.to(torch::kCPU)).abs();
    const auto frozen_difference =
        (two_reconstruction.prediction - expected.reconstruction.to(torch::kCPU)).abs();
    std::cout << std::setprecision(17) << "CUDA reconstruction parity mixer=" << mixer_layers
        << " global_mode=" << global_mode
        << " trainable_max_abs=" << trainable_difference.max().item<double>()
        << " trainable_changed_coordinates=" << trainable_difference.ne(0).sum().item<int64_t>()
        << " frozen_max_abs=" << frozen_difference.max().item<double>()
        << " target_exact=true\n";
    close(two_reconstruction.eligible, expected.eligible_channels.to(torch::kCPU),
          "snapshot exact fixed-mask reconstruction eligibility", 0, 0);
    close(two_reconstruction.prediction, expected.reconstruction.to(torch::kCPU),
          "snapshot exact exported-bottleneck reconstruction", 0, 0);
    // The independent checkpoint below is also used to test ordinary training
    // resume. Restore trainability before constructing/loading its optimizer.
    for (auto &parameter : point_two_gpu.model->parameters()) parameter.set_requires_grad(true);
    const auto encoding = point_two_cpu.model->encode(
        point_two_cpu.scaler.transform(saved_raw.input, point_two_cpu.settings.model));
    close(two_features.at("curve_global").values,
          mixer_layers > 0 ? encoding.z_contextual_global : encoding.z_global,
          "CPU features match exact saved checkpoint global", 0, 0);
    close(two_features.at("curve_channel_concatenation").values,
          (mixer_layers > 0 ? encoding.z_contextual : encoding.z_local).flatten(1),
          "CPU features match exact selected channel branch", 0, 0);
    const auto reconstruction_export = rpb::compact_reconstruction_export(encoding, c);
    check(reconstruction_export.dim() == (global_mode == 0 ? 3 : 2),
          "reconstruction uses declared per-channel or sole global export");
  }
  torch::serialize::InputArchive audit;
  audit.load_from(two_path + ".audit.pt", torch::kCPU);
  check(audit_text(audit, "artifact_kind") == "rpb_learning_curve_training_audit_v1" &&
      audit_text(audit, "resolved_settings") == rpb::settings_text(
          rpb::load_checkpoint(two_path, c.device).settings) &&
      audit_text(audit, "actual_training_seed") == "202" &&
      audit_text(audit, "core_writer_source_fingerprint") == rpb::workflow_source_fingerprint() &&
      !audit_text(audit, "training_producer_source_fingerprint").empty(),
      "companion producer/source scopes and canonical CUDA settings");
  const auto assets = directory / "snapshot-assets"; fs::create_directory(assets);
  point_two.features.save_assets(assets.string());
  torch::serialize::InputArchive snapshot_audit;
  snapshot_audit.load_from((assets / "curve-snapshot-audit.pt").string(), torch::kCPU);
  check(audit_text(snapshot_audit, "artifact_kind") == "rpb_learning_curve_snapshot_audit_v1" &&
      audit_text(snapshot_audit, "checkpoint_path") == two_path &&
      audit_text(snapshot_audit, "completed_steps") == "2",
      "snapshot provenance points to exact ordinary checkpoint");
  rejects([&] { trainer.save_checkpoint(two_path); }, "overwriting saved point");
  rejects([&] { trainer.snapshot((directory / "foreign.pt").string()); }, "foreign snapshot path");
  rejects([&] { trainer.train_to(1); }, "decreasing completed budget");
  rejects([&] { trainer.train_to(5); }, "beyond frozen session budget");

  const auto four = trainer.train_to(4);
  check(four.attempted == 4 && four.completed == 4 && four.sampled_rows == 8 &&
      four.training_seconds >= two.training_seconds && four.preprocessing_id == zero.preprocessing_id,
      "continuous second training segment");
  check(two.losses.size() == 2 && two.losses.back().completed == 2 &&
      four.losses.size() == 4 && four.losses.back().completed == 4,
      "nonperiodic queried milestone endpoints persist in the sampled trace");
  for (size_t i = 0; i < two.losses.size(); ++i) {
    const auto &before = two.losses[i], &after_point = four.losses[i];
    check(before.attempted == after_point.attempted && before.completed == after_point.completed &&
        before.target_cells == after_point.target_cells && before.loss == after_point.loss &&
        before.gradient_norm == after_point.gradient_norm,
        "previous milestone trace must remain an exact cumulative positional prefix");
  }
  const auto four_path = (directory / "point-4.pt").string();
  trainer.save_checkpoint(four_path);
  const auto four_checkpoint = rpb::load_checkpoint(four_path);
  for (const auto &[name, surface] : point_two.features.extract(heldout))
    close(surface.values, two_features.at(name).values, "earlier CPU snapshot remains immutable", 0, 0);
  close(point_two.reconstruct(heldout, hidden).prediction, two_reconstruction.prediction,
        "earlier CUDA reconstruction remains immutable", 0, 0);
  for (const auto &[name, surface] : point_zero.features.extract(heldout))
    close(surface.values, zero_features.at(name).values, "point zero remains immutable", 0, 0);
  const auto after = trainer.train_to(4);
  check(after.attempted == four.attempted && after.training_seconds == four.training_seconds,
      "same absolute budget performs no additional updates");
  embedding::Batch absent{torch::full_like(heldout.data, std::numeric_limits<double>::quiet_NaN()),
      torch::zeros_like(heldout.feature_mask)};
  for (const auto &[name, surface] : point_two.features.extract(absent)) {
    check(!surface.valid.any().item<bool>(), name + " no invented support");
    close(surface.values, torch::zeros_like(surface.values), name + " absent exact zero", 0, 0);
  }
  const auto no_query = point_two.reconstruct(absent, torch::zeros_like(hidden));
  check(!no_query.eligible.any().item<bool>(), "no invented reconstruction eligibility");

  fit.training_observations = heldout;
  auto direct = factory(fit);
  direct.train_to(4);
  const auto direct_path = (directory / "direct-4.pt").string(); direct.save_checkpoint(direct_path);
  same_parameters(four_checkpoint.model, rpb::load_checkpoint(direct_path).model,
                  "continuous split budgets preserve exact stochastic update stream");
  // Ordinary checkpoint moments/counters also support exact CUDA continuation.
  resume_two_updates(point_two_gpu, saved_raw, two_path);
  same_parameters(point_two_gpu.model, four_checkpoint.model, "ordinary GPU optimizer resume parity");
  check(point_two_gpu.attempted_steps == four.attempted, "resumed absolute attempt stream");

  auto invalid_fit = fit; invalid_fit.feature_units = "wrong_number_of_units";
  rejects([&] { factory(invalid_fit); }, "invalid fit units");
  invalid_fit = fit; invalid_fit.channel_ids = {101, 303};
  rejects([&] { factory(invalid_fit); }, "semantic schema mismatch");
  invalid_fit = fit; invalid_fit.shape.dtype = torch::kFloat32;
  rejects([&] { factory(invalid_fit); }, "declared source precision mismatch");
  std::cout << "RPB CUDA learning-curve continuity tests passed; artifacts=" << directory << '\n';
}

void exhausted_attempts_preserve_recovery() {
  auto c = config(); c.device = torch::Device(torch::kCUDA, 0);
  auto settings = rpb::default_settings(); settings.model = c; settings.steps = 1;
  settings.attempt_limit = 3; settings.batch_size = 2;
  auto raw = input(c, 4);
  // Scaler remains fit-able but no channel has the three required patch groups.
  raw.observed.narrow(2, 2 * c.patch_length, c.history_length - 2 * c.patch_length).fill_(false);
  ev::ProviderFitInput fit{{raw.data, raw.observed},
      {c.channel_count, c.history_length, c.input_width, torch::kFloat64, torch::kCPU},
      303, {"a", "a", "b", "b"}, {101, 202}, "volts,amperes",
      "learning-curve-attempt-test-v1", 1, static_cast<double>(c.history_length - 1)};
  auto trainer = rpb::make_learning_curve_trainer(settings)(fit);
  rejects([&] { trainer.train_to(1); }, "no-update attempt exhaustion");
  const auto after = trainer.train_to(0);
  check(after.completed == 0 && after.attempted == 3 && after.sampled_rows == 6 &&
      !after.weights_changed && !after.last_loss_cuda && after.losses.empty(),
      "no-update attempts advance without optimizer updates");
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-curve-recovery-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  const auto path = (directory / "recovery.pt").string();
  trainer.save_checkpoint(path);
  const auto checkpoint = rpb::load_checkpoint(path);
  check(checkpoint.attempted_steps == 3 && checkpoint.completed_steps == 0,
      "no-update recovery checkpoint retains attempted counter");
}
} // namespace

int main() {
  try {
    torch::set_num_threads(1);
    rejects([] { rpb::make_learning_curve_trainer(rpb::default_settings()); },
            "implicit CPU fallback");
    check(torch::cuda::is_available(), "learning-curve adapter test requires CUDA");
    for (const int64_t mixer_layers : {0, 1})
      for (const int64_t global_mode : {0, 1, 2, 3})
        continuity_and_snapshots(mixer_layers, global_mode);
    exhausted_attempts_preserve_recovery();
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
