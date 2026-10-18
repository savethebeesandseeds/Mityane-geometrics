// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ctx = rpb::context_deletion;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;

std::string audit_text(torch::serialize::InputArchive &archive, const char *name) {
  torch::Tensor value; archive.read(name, value, true); return embedding::archive::tensor_text(value);
}
int64_t count(torch::serialize::InputArchive &archive, const char *name) {
  torch::Tensor value; archive.read(name, value, true);
  check(value.scalar_type() == torch::kInt64 && value.numel() == 1, std::string(name) + " typed count");
  return value.item<int64_t>();
}
void same_parameters(const rpb::Model &actual, const rpb::Model &expected, const std::string &label) {
  const auto reference = expected->named_parameters();
  check(actual->named_parameters().size() == reference.size(), label + " parameter count");
  for (const auto &parameter : actual->named_parameters())
    close(parameter.value().to(torch::kCPU), reference[parameter.key()].to(torch::kCPU), label + "/" + parameter.key(), 0, 0);
}
void same_optimizer(rpb::Checkpoint &actual, const std::string &actual_path,
                    rpb::Checkpoint &expected, const std::string &expected_path) {
  torch::optim::AdamW a(actual.model->parameters(), torch::optim::AdamWOptions(actual.settings.learning_rate).weight_decay(actual.settings.weight_decay));
  torch::optim::AdamW b(expected.model->parameters(), torch::optim::AdamWOptions(expected.settings.learning_rate).weight_decay(expected.settings.weight_decay));
  rpb::load_optimizer(actual_path, a, actual.settings.model.device); rpb::load_optimizer(expected_path, b, expected.settings.model.device);
  check(a.state().size() == b.state().size() && !a.state().empty(), "exact saved optimizer state count");
  const auto expected_parameters = expected.model->named_parameters();
  for (const auto &parameter : actual.model->named_parameters()) {
    const auto ai = a.state().find(parameter.value().unsafeGetTensorImpl());
    const auto bi = b.state().find(expected_parameters[parameter.key()].unsafeGetTensorImpl());
    check((ai == a.state().end()) == (bi == b.state().end()), "exact named optimizer associations");
    if (ai == a.state().end()) continue;
    const auto *as = dynamic_cast<const torch::optim::AdamWParamState *>(ai->second.get());
    const auto *bs = dynamic_cast<const torch::optim::AdamWParamState *>(bi->second.get());
    check(as && bs && as->step() == bs->step() && as->step() == actual.completed_steps &&
        as->exp_avg().is_cuda() && as->exp_avg_sq().is_cuda(), "exact AdamW CUDA steps");
    close(as->exp_avg(), bs->exp_avg(), "exact first moments", 0, 0);
    close(as->exp_avg_sq(), bs->exp_avg_sq(), "exact second moments", 0, 0);
  }
}
void same_trace(const ev::CurveProgress &a, const ev::CurveProgress &b) {
  check(a.attempted == b.attempted && a.completed == b.completed && a.sampled_rows == b.sampled_rows &&
      a.preprocessing_id == b.preprocessing_id && a.training_dataset_id == b.training_dataset_id && a.losses.size() == b.losses.size(),
      "disabled overload exact counters/scaler/trace size");
  for (size_t i = 0; i < a.losses.size(); ++i) {
    const auto &x = a.losses[i], &y = b.losses[i];
    check(x.attempted == y.attempted && x.completed == y.completed && x.target_cells == y.target_cells &&
        x.loss == y.loss && x.gradient_norm == y.gradient_norm, "disabled overload exact numerical loss trace");
  }
}
void freeze(rpb::Checkpoint &checkpoint) {
  checkpoint.model->eval(); for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(false);
}
std::array<int64_t, 3> expected_counts(const rpb::Dataset &raw, const rpb::Settings &settings, int64_t attempts) {
  std::array<int64_t, 3> counts{0, 0, 0};
  for (int64_t attempt = 0; attempt < attempts; ++attempt) {
    const auto indices = rpb::training_detail::sampled_indices(raw.input.data.size(0), settings.batch_size, settings.seed, attempt);
    const auto batch = rpb::training_detail::selected(raw.input, indices);
    const auto original = rpb::make_training_mask(batch.observed, settings.model,
        rpb::training_detail::counter_seed(settings.seed, attempt, 0x6d61736bULL));
    if (!original.eligible_channels.any().item<bool>()) continue;
    const auto context = ctx::make_plan(original, batch.channel_ids, settings.model, settings.seed, attempt);
    counts[0] += context.requested_count; counts[1] += context.actual_count; counts[2] += context.restored_count;
  }
  return counts;
}

void cuda_fresh_recipe_and_disabled_parity() {
  auto c = config(); c.device = torch::Device(torch::kCUDA, 0); c.global_bottleneck_mode = 2;
  c.channel_mixer_layers = 1; c.export_width = 32; c.dropout = .1; c.sampling_interval = .5;
  auto settings = rpb::default_settings(); settings.model = c; settings.steps = 4; settings.batch_size = 2;
  settings.seed = 101; settings.attempt_limit = 16; settings.log_every = 3;
  auto raw = input(c, 6); const auto order = torch::tensor({1, 0}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order); raw.channel_ids = raw.channel_ids.index_select(0, order);
  raw.observed[0][0][14][1] = false; raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch heldout{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{heldout, {c.channel_count, c.history_length, c.input_width, torch::kFloat64, torch::kCPU},
      202, {"a", "a", "b", "b", "c", "c"}, {202, 101}, "volts,amperes", "context-deletion-adapter-contract-v1",
      c.sampling_interval, (c.history_length - 1) * c.sampling_interval};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-context-deletion-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  auto legacy = rpb::make_learning_curve_trainer(settings)(fit);
  auto disabled = rpb::make_learning_curve_trainer(settings, rpb::ContextDeletionOptions{})(fit);
  auto enabled = rpb::make_learning_curve_trainer(settings, rpb::ContextDeletionOptions{true})(fit);
  check(legacy.audit_fields == disabled.audit_fields && !legacy.audit_fields.count("training_policy_id") &&
      !legacy.audit_fields.count("model_tag") && enabled.audit_fields.at("model_tag") == "RPB-v6" &&
      enabled.audit_fields.at("training_policy_id") == ctx::policy_id &&
      enabled.audit_fields.at("context_deletion_rng_policy") == ctx::rng_policy &&
      enabled.audit_fields.at("context_deletion_repair_policy") == ctx::repair_policy,
      "enabled-only fixed policy audit; disabled metadata exact");
  const auto legacy_zero = (directory / "legacy-0.pt").string(), disabled_zero = (directory / "disabled-0.pt").string();
  const auto enabled_zero = (directory / "context-0.pt").string();
  legacy.save_checkpoint(legacy_zero); disabled.save_checkpoint(disabled_zero); enabled.save_checkpoint(enabled_zero);
  const auto initial = rpb::load_checkpoint(legacy_zero), explicit_initial = rpb::load_checkpoint(disabled_zero);
  const auto context_initial = rpb::load_checkpoint(enabled_zero);
  same_parameters(initial.model, explicit_initial.model, "default overload exact initialization");
  same_parameters(context_initial.model, initial.model, "context policy changes no initialized parameter including global pool");
  check(context_initial.scaler.identity() == initial.scaler.identity() && context_initial.dataset_id == initial.dataset_id &&
      initial.training_policy_id.empty() && context_initial.training_policy_id == ctx::policy_id,
      "context policy keeps original TRAIN/scaler identity with explicit checkpoint tag");
  torch::serialize::InputArchive ordinary_archive, ordinary_audit; torch::Tensor absent;
  ordinary_archive.load_from(legacy_zero, torch::kCPU); ordinary_audit.load_from(legacy_zero + ".audit.pt", torch::kCPU);
  check(!ordinary_archive.try_read("training_policy_id", absent, true) &&
      !ordinary_audit.try_read("context_requested_deleted_coordinates", absent, true) &&
      !ordinary_audit.try_read("context_deletion_ratio", absent, true), "default checkpoint/audit has no optional policy fields");
  auto cpu_generator = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  auto gpu_generator = at::globalContext().defaultGenerator(at::Device(at::kCUDA, 0));
  const auto legacy_four = legacy.train_to(4);
  const auto legacy_cpu_rng = cpu_generator.get_state().clone(), legacy_gpu_rng = gpu_generator.get_state().clone();
  const auto disabled_four = disabled.train_to(4);
  same_trace(legacy_four, disabled_four);
  check(torch::equal(legacy_cpu_rng, cpu_generator.get_state()) && torch::equal(legacy_gpu_rng, gpu_generator.get_state()),
      "disabled overload consumes exact historical CPU/CUDA training RNG");
  const auto legacy_four_path = (directory / "legacy-4.pt").string(), disabled_four_path = (directory / "disabled-4.pt").string();
  legacy.save_checkpoint(legacy_four_path); disabled.save_checkpoint(disabled_four_path);
  auto legacy_saved = rpb::load_checkpoint(legacy_four_path, c.device), disabled_saved = rpb::load_checkpoint(disabled_four_path, c.device);
  same_parameters(legacy_saved.model, disabled_saved.model, "disabled overload exact update weights");
  same_optimizer(legacy_saved, legacy_four_path, disabled_saved, disabled_four_path);
  // A separate ordinary workflow loop is a direct numerical reference for the
  // default path, rather than comparing two wrappers of the same new branch.
  const auto ordinary_four_path = (directory / "ordinary-4.pt").string();
  std::vector<std::string> args{"rpb", "train", "--resume", legacy_zero, "--input", legacy_zero + ".training-raw.pt",
      "--device", "cuda", "--steps", "4", "--checkpoint", ordinary_four_path, "--checkpoint-every", "0"};
  std::vector<char *> argv; for (auto &arg : args) argv.push_back(arg.data());
  check(rpb::run_cli(static_cast<int>(argv.size()), argv.data()) == 0, "ordinary historical training reference");
  auto ordinary = rpb::load_checkpoint(ordinary_four_path, c.device);
  same_parameters(legacy_saved.model, ordinary.model, "disabled path exact ordinary workflow weights");
  same_optimizer(legacy_saved, legacy_four_path, ordinary, ordinary_four_path);

  const auto two = enabled.train_to(2);
  check(two.attempted == 2 && two.completed == 2 && two.last_input_cuda && two.last_loss_cuda &&
      two.finite_gradients && two.weights_changed && two.cuda_parameter_count == two.parameter_count &&
      two.parameter_count == legacy_four.parameter_count && two.preprocessing_id == legacy_four.preprocessing_id,
      "actual augmented CUDA input/loss/finite gradients/weight updates with same model/scaler");
  const auto two_path = (directory / "context-2.pt").string(); enabled.save_checkpoint(two_path);
  auto c_cpu = c; c_cpu.device = torch::kCPU; const auto saved_raw = rpb::load_dataset(two_path + ".training-raw.pt", c_cpu);
  auto two_checkpoint = rpb::load_checkpoint(two_path, c.device);
  const auto counts_two = expected_counts(saved_raw, two_checkpoint.settings, 2);
  torch::serialize::InputArchive audit; audit.load_from(two_path + ".audit.pt", torch::kCPU);
  check(count(audit, "context_requested_deleted_coordinates") == counts_two[0] &&
      count(audit, "context_actual_deleted_coordinates") == counts_two[1] &&
      count(audit, "context_restored_coordinates") == counts_two[2] && counts_two[1] > 0 &&
      audit_text(audit, "training_policy_id") == ctx::policy_id && audit_text(audit, "context_deletion_visibility_policy") == ctx::visibility_policy,
      "cumulative exact coordinate counts and unchanged-query policy companion");
  torch::Tensor ratio, stream; audit.read("context_deletion_ratio_value", ratio, true); audit.read("context_deletion_stream_value", stream, true);
  check(ratio.scalar_type() == torch::kFloat64 && ratio.item<double>() == .30 &&
      stream.item<int64_t>() == static_cast<int64_t>(ctx::stream), "fixed typed ratio and independent stream");
  for (const auto &point : two.losses) {
    const auto indices = rpb::training_detail::sampled_indices(saved_raw.input.data.size(0), settings.batch_size, 202, point.attempted - 1);
    const auto source = rpb::training_detail::selected(saved_raw.input, indices);
    const auto original = rpb::make_training_mask(source.observed, c,
        rpb::training_detail::counter_seed(202, point.attempted - 1, 0x6d61736bULL));
    check(point.target_cells == original.target.sum().item<int64_t>(), "augmented trace keeps exact original target count");
  }
  bool pool_changed = false, decoder_changed = false;
  const auto initialized = context_initial.model->named_parameters();
  for (const auto &parameter : two_checkpoint.model->named_parameters()) {
    const bool changed = !torch::equal(parameter.value().to(torch::kCPU), initialized[parameter.key()]);
    if (parameter.key().rfind("global_pool_", 0) == 0) pool_changed |= changed;
    if (parameter.key().rfind("decoder_", 0) == 0) decoder_changed |= changed;
  }
  check(pool_changed && decoder_changed, "augmented CUDA loss changes global32 pool and decoder weights");
  const auto snapshot = enabled.snapshot(two_path); const auto features = snapshot.features.extract(heldout);
  check(features.at("curve_global").values.size(1) == 32 && features.at("curve_global").valid.all().item<bool>() &&
      features.at("curve_global").provenance.find("RPB-v6") != std::string::npos &&
      snapshot.features.audit_fields.at("context_actual_deleted_coordinates") == std::to_string(counts_two[1]),
      "native32 ordinary serving retains trained-policy provenance");
  auto hidden = torch::zeros_like(heldout.feature_mask); hidden.narrow(2, 0, c.patch_length).fill_(true);
  const auto reconstruction = snapshot.reconstruct(heldout, hidden);
  {
    torch::NoGradGuard no_grad; auto cpu_checkpoint = rpb::load_checkpoint(two_path); cpu_checkpoint.model->eval();
    const auto encoded = cpu_checkpoint.model->encode(cpu_checkpoint.scaler.transform(saved_raw.input, cpu_checkpoint.settings.model));
    close(features.at("curve_global").values, encoded.z_contextual_global, "exact ordinary CPU native32 serving with no added deletion", 0, 0);
    freeze(two_checkpoint); const auto normalized = two_checkpoint.scaler.transform(saved_raw.input, two_checkpoint.settings.model);
    const auto expected = two_checkpoint.model->forward(normalized, hidden);
    close(reconstruction.prediction, expected.reconstruction.to(torch::kCPU), "exact frozen ordinary CUDA reconstruction with no added deletion", 0, 0);
    close(reconstruction.eligible, expected.eligible_channels.to(torch::kCPU), "ordinary held-out query eligibility", 0, 0);
  }
  const embedding::Batch changed_query{torch::where(hidden, heldout.data + 100, heldout.data), heldout.feature_mask};
  close(snapshot.reconstruct(changed_query, hidden).prediction, reconstruction.prediction, "ordinary frozen query-hidden isolation", 0, 0);
  const auto four = enabled.train_to(4); const auto four_path = (directory / "context-4.pt").string(); enabled.save_checkpoint(four_path);
  check(four.completed == 4 && four.attempted == 4 && four.preprocessing_id == two.preprocessing_id && four.losses.size() >= two.losses.size(),
      "one continuous augmented optimizer and immutable scaler");
  for (size_t i = 0; i < two.losses.size(); ++i)
    check(two.losses[i].loss == four.losses[i].loss && two.losses[i].completed == four.losses[i].completed,
        "augmented trace retains exact earlier milestone prefix");
  const auto zero_after_training = enabled.snapshot(enabled_zero);
  check(zero_after_training.features.audit_fields.at("context_actual_deleted_coordinates") == "0", "old snapshot audit counters are point-specific");
  close(snapshot.features.extract(heldout).at("curve_global").values, features.at("curve_global").values, "earlier native snapshot immutable", 0, 0);
  close(snapshot.reconstruct(heldout, hidden).prediction, reconstruction.prediction, "earlier frozen GPU snapshot immutable", 0, 0);
  auto direct = rpb::make_learning_curve_trainer(settings, rpb::ContextDeletionOptions{true})(fit);
  direct.train_to(4); const auto direct_path = (directory / "context-direct-4.pt").string(); direct.save_checkpoint(direct_path);
  auto split_saved = rpb::load_checkpoint(four_path, c.device), direct_saved = rpb::load_checkpoint(direct_path, c.device);
  same_parameters(split_saved.model, direct_saved.model, "split/direct augmented exact CUDA training");
  same_optimizer(split_saved, four_path, direct_saved, direct_path);
  const auto counts_four = expected_counts(saved_raw, split_saved.settings, 4);
  torch::serialize::InputArchive four_audit; four_audit.load_from(four_path + ".audit.pt", torch::kCPU);
  check(count(four_audit, "context_requested_deleted_coordinates") == counts_four[0] &&
      count(four_audit, "context_actual_deleted_coordinates") == counts_four[1] &&
      count(four_audit, "context_restored_coordinates") == counts_four[2], "cumulative deletion counts continue without reset");
  const auto forbidden_path = (directory / "forbidden-ordinary-resume.pt").string();
  args = {"rpb", "train", "--resume", four_path, "--input", (directory / "missing-input.pt").string(),
      "--device", "cuda", "--steps", "1", "--checkpoint", forbidden_path}; argv.clear(); for (auto &arg : args) argv.push_back(arg.data());
  bool rejected_policy = false;
  try { rpb::run_cli(static_cast<int>(argv.size()), argv.data()); }
  catch (const std::exception &error) { rejected_policy = std::string(error.what()).find("ordinary train cannot resume this saved training policy") != std::string::npos; }
  check(rejected_policy && !fs::exists(forbidden_path), "ordinary resume rejects policy before reading TRAIN or updating");
  auto invalid = settings; invalid.model.global_bottleneck_mode = 3;
  rejects([&] { rpb::make_learning_curve_trainer(invalid, rpb::ContextDeletionOptions{true}); }, "context architecture drift");
  invalid = settings; invalid.model.channel_mixer_layers = 0;
  rejects([&] { rpb::make_learning_curve_trainer(invalid, rpb::ContextDeletionOptions{true}); }, "context without mixer");
  std::cout << "RPB context-deletion CUDA/default-parity tests passed; artifacts=" << directory << '\n';
}
} // namespace

int main() {
  try { torch::set_num_threads(1); check(torch::cuda::is_available(), "context-deletion adapter tests require CUDA"); cuda_fresh_recipe_and_disabled_parity(); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
