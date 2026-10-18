// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replication_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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

std::string file_bytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary); check(bool(input), "original fixture readable");
  std::ostringstream bytes; bytes << input.rdbuf(); return bytes.str();
}
void copy_audit(const std::string &path, const std::string &copy) {
  fs::copy_file(path + ".audit.pt", copy + ".audit.pt", fs::copy_options::none);
}
void changed_checkpoint(const std::string &path, const std::string &copy, const torch::Device &device,
                         const std::function<void(rpb::Checkpoint &)> &change) {
  auto checkpoint = rpb::load_checkpoint(path, device);
  torch::optim::AdamW optimizer(checkpoint.model->parameters(),
      torch::optim::AdamWOptions(checkpoint.settings.learning_rate).weight_decay(checkpoint.settings.weight_decay));
  rpb::load_optimizer(path, optimizer, device); change(checkpoint);
  rpb::save_checkpoint(copy, checkpoint, optimizer); copy_audit(path, copy);
}
void changed_audit(const std::string &path, const std::string &copy,
                    const std::string &field, const torch::Tensor &replacement) {
  fs::copy_file(path, copy, fs::copy_options::none);
  torch::serialize::InputArchive old; old.load_from(path + ".audit.pt", torch::kCPU);
  torch::serialize::OutputArchive changed; bool found = false;
  for (const auto &name : old.keys()) {
    torch::Tensor value; old.read(name, value, true);
    if (name == field) { value = replacement; found = true; }
    changed.write(name, value, true);
  }
  check(found, "altered audit field exists"); embedding::archive::save_archive(copy + ".audit.pt", changed);
}
void freeze(rpb::Checkpoint &checkpoint) {
  checkpoint.model->eval(); for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(false);
}

void fresh_cuda_replication_initialization_contract() {
  auto settings = rpb::default_settings(); auto &c = settings.model;
  c.device = torch::Device(torch::kCUDA, 0); c.global_bottleneck_mode = 2; c.channel_mixer_layers = 1;
  c.channel_ids = {101, 202, 303}; c.dropout = .1; c.sampling_interval = .5;
  settings.steps = 4; settings.batch_size = 2; settings.log_every = 3; settings.attempt_limit = 32;
  auto raw = input(c, 6); const auto order = torch::tensor({2, 0, 1}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order);
  raw.channel_ids = raw.channel_ids.index_select(0, order);
  raw.observed[0][0][14][1] = false; raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch legal{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{legal, {3, 32, 3, torch::kFloat64, torch::kCPU}, 4101,
      {"fresh-a", "fresh-a", "fresh-b", "fresh-b", "fresh-c", "fresh-c"}, {303, 101, 202},
      "volts,amperes,kelvin", "native-development-v1/lag_sign", c.sampling_interval,
      (c.history_length - 1) * c.sampling_interval};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-context-replication-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  auto baseline = rpb::make_learning_curve_trainer(settings)(fit);
  auto candidate = rpb::make_learning_curve_trainer(settings, rpb::ContextDeletionOptions{true})(fit);
  const auto base0 = (directory / "fresh-v4-0.pt").string(), candidate0 = (directory / "fresh-v6-0.pt").string();
  baseline.save_checkpoint(base0); candidate.save_checkpoint(candidate0);
  const auto baseline_initial = baseline.snapshot(base0), candidate_initial = candidate.snapshot(candidate0);
  const auto initial_features = baseline_initial.features.extract(legal).at("curve_global");
  const auto candidate_features = candidate_initial.features.extract(legal).at("curve_global");
  close(initial_features.values, candidate_features.values, "entire initialized global32 surface exact", 0, 0);
  check(initial_features.values.size(1) == 32 && torch::equal(initial_features.valid, candidate_features.valid) &&
      initial_features.valid.all().item<bool>(), "same observed native32 support");
  auto hidden = torch::zeros_like(raw.observed); hidden.narrow(2, 0, c.patch_length).fill_(true);
  const auto initial_prediction = baseline_initial.reconstruct(legal, hidden);
  close(initial_prediction.prediction, candidate_initial.reconstruct(legal, hidden).prediction,
      "entire initialized CUDA32 decoder route exact", 0, 0);

  const auto base_two = baseline.train_to(2); const auto base2 = (directory / "fresh-v4-2.pt").string();
  baseline.save_checkpoint(base2);
  ev::RetainedPoolingCohort reference; reference.master_seed = fit.seed;
  reference.lineage = "fresh equal-budget replication fixture";
  reference.reference_initial_checkpoint = base0; reference.reference_checkpoint = base2;
  std::map<std::string, std::string> originals;
  for (const auto &path : {base0, base2, candidate0})
    for (const std::string suffix : {"", ".audit.pt"}) originals[path + suffix] = file_bytes(path + suffix);
  auto cpu_rng = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  auto gpu_rng = at::globalContext().defaultGenerator(at::Device(at::kCUDA, 0));
  const auto cpu_before = cpu_rng.get_state().clone(), gpu_before = gpu_rng.get_state().clone();
  const auto threads = at::get_num_threads();
  const auto audited = rpb::audit_context_replication_initialization(candidate0, reference, fit, 2);
  check(rpb::audit_context_replication_initialization(candidate0, reference, fit,
      rpb::ContextDeletionRecipe::coordinate30_v1, 2) == audited, "legacy audit is exact explicit30");
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, reference, fit,
      rpb::ContextDeletionRecipe::coordinate15_v1, 2); }, "explicit15 gate rejects old30 checkpoint");
  check(torch::equal(cpu_before, cpu_rng.get_state()) && torch::equal(gpu_before, gpu_rng.get_state()) &&
      threads == at::get_num_threads(), "successful inspection preserves CPU/CUDA RNG and threads");
  check(audited.at("common_parameter_count") == "225805" && audited.at("all_parameters_exact_including_global_pool") == "true" &&
      audited.at("all_buffers_exact") == "true" && audited.at("scaler_exact") == "true" &&
      audited.at("counter_streams_exact") == "true" && audited.at("training_protocol_id") == fit.protocol_id &&
      audited.at("reference_completed_updates") == "2" && audited.at("reference_role") == "fresh_equal_budget_training_reference" &&
      audited.at("training_policy_companion_exact") == "true", "fresh full architecture/scaler/stream/policy proof");
  check(base_two.attempted == 2 && base_two.completed == 2 && base_two.sampled_rows == 4 &&
      base_two.parameter_count == 225805 && base_two.cuda_parameter_count == 225805 && base_two.last_input_cuda &&
      base_two.last_loss_cuda && base_two.finite_gradients && base_two.weights_changed, "actual uninterrupted fresh v4 CUDA budget");
  const auto candidate_two = candidate.train_to(2); const auto candidate2 = (directory / "fresh-v6-2.pt").string();
  candidate.save_checkpoint(candidate2);
  check(candidate_two.attempted == 2 && candidate_two.completed == 2 && candidate_two.sampled_rows == 4 &&
      candidate_two.parameter_count == base_two.parameter_count && candidate_two.cuda_parameter_count == 225805 &&
      candidate_two.last_input_cuda && candidate_two.last_loss_cuda && candidate_two.finite_gradients &&
      candidate_two.weights_changed && candidate_two.preprocessing_id == base_two.preprocessing_id &&
      candidate_two.losses.size() == base_two.losses.size(), "actual fresh v6 CUDA counters and unchanged frozen scaler");
  for (size_t i = 0; i < base_two.losses.size(); ++i)
    check(base_two.losses[i].target_cells == candidate_two.losses[i].target_cells &&
        base_two.losses[i].attempted == candidate_two.losses[i].attempted,
        "fresh v4/v6 original patch-query targets and absolute attempts match");
  const auto candidate_snapshot = candidate.snapshot(candidate2);
  const auto frozen_values = candidate_snapshot.features.extract(legal).at("curve_global").values.clone();
  const auto frozen_prediction = candidate_snapshot.reconstruct(legal, hidden);
  {
    torch::NoGradGuard no_grad;
    auto saved = rpb::load_checkpoint(candidate2, c.device); freeze(saved);
    const auto normalized = saved.scaler.transform(raw, saved.settings.model);
    const auto served = saved.model->forward(normalized, hidden);
    close(frozen_prediction.prediction, served.reconstruction.to(torch::kCPU), "saved CUDA global32 route matches snapshot", 0, 0);
  }
  auto hidden_change = legal; hidden_change.data = legal.data.clone(); hidden_change.data.masked_fill_(hidden, 998877.0);
  close(candidate_snapshot.reconstruct(hidden_change, hidden).prediction, frozen_prediction.prediction,
      "query-hidden values cannot enter frozen reconstruction", 0, 0);
  baseline.train_to(4); candidate.train_to(4);
  close(candidate_snapshot.features.extract(legal).at("curve_global").values, frozen_values, "earlier candidate surface remains immutable", 0, 0);
  close(candidate_snapshot.reconstruct(legal, hidden).prediction, frozen_prediction.prediction, "earlier GPU snapshot remains immutable", 0, 0);
  close(candidate_initial.features.extract(legal).at("curve_global").values, initial_features.values, "point-zero candidate remains immutable", 0, 0);
  const auto audited_again = rpb::audit_context_replication_initialization(candidate0, reference, fit, 2);
  check(audited_again == audited, "saved equal-budget audit remains stable after live continuation");

  rejects([&] { rpb::audit_context_replication_initialization(candidate0, reference, fit); }, "default512 budget rejects tiny fixture2");
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, reference, fit, 0); }, "nonpositive declared reference budget");
  rejects([&] { rpb::audit_context_replication_initialization(candidate2, reference, fit, 2); }, "candidate must be point zero");
  auto wrong_reference = reference; wrong_reference.master_seed = 999;
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, wrong_reference, fit, 2); }, "wrong fresh master association");
  wrong_reference = reference; wrong_reference.reference_checkpoint = base0;
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, wrong_reference, fit, 2); }, "untrained reference cannot claim positive budget");
  auto wrong = fit; wrong.protocol_id = "native-curve-v1/lag_sign";
  const auto rejection_cpu = cpu_rng.get_state().clone(), rejection_gpu = gpu_rng.get_state().clone();
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, reference, wrong, 2); }, "historical namespace is not fresh replication");
  check(torch::equal(rejection_cpu, cpu_rng.get_state()) && torch::equal(rejection_gpu, gpu_rng.get_state()) &&
      threads == at::get_num_threads(), "rejected inspection also preserves runtime state");
  wrong = fit; wrong.seed += 1; wrong_reference = reference; wrong_reference.master_seed = wrong.seed;
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, wrong_reference, wrong, 2); }, "wrong original seed");
  wrong = fit; wrong.training_source_ids[0] = "other";
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, reference, wrong, 2); }, "wrong ordered source IDs");
  wrong = fit; wrong.channel_ids = {101, 202, 303};
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, reference, wrong, 2); }, "wrong physical semantic channel order");
  wrong = fit; wrong.training_observations.data = fit.training_observations.data.clone(); wrong.training_observations.data[0][0][0][0] += 1;
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, reference, wrong, 2); }, "wrong legal TRAIN value");
  wrong = fit; wrong.feature_units = "other,other,other";
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, reference, wrong, 2); }, "wrong declared units");

  const auto different_weight = (directory / "changed-initial-weight.pt").string();
  changed_checkpoint(candidate0, different_weight, c.device, [](rpb::Checkpoint &saved) {
    torch::NoGradGuard no_grad; saved.model->named_parameters()["global_pool_second.weight"].add_(.25);
  });
  rejects([&] { rpb::audit_context_replication_initialization(different_weight, reference, fit, 2); }, "changed point-zero global pool parameter");
  const auto different_scaler = (directory / "changed-initial-scaler.pt").string();
  changed_checkpoint(candidate0, different_scaler, c.device, [](rpb::Checkpoint &saved) { saved.scaler.mean = saved.scaler.mean + .25; });
  rejects([&] { rpb::audit_context_replication_initialization(different_scaler, reference, fit, 2); }, "changed frozen training scaler");
  const auto different_settings = (directory / "changed-settings.pt").string();
  changed_checkpoint(candidate0, different_settings, c.device, [](rpb::Checkpoint &saved) { saved.settings.learning_rate *= 2; });
  rejects([&] { rpb::audit_context_replication_initialization(different_settings, reference, fit, 2); }, "different optimizer recipe");
  const auto no_policy = (directory / "missing-policy.pt").string();
  changed_checkpoint(candidate0, no_policy, c.device, [](rpb::Checkpoint &saved) { saved.training_policy_id.clear(); });
  rejects([&] { rpb::audit_context_replication_initialization(no_policy, reference, fit, 2); }, "missing candidate training tag");
  const auto nonzero_counts = (directory / "nonzero-context-count.pt").string();
  changed_audit(candidate0, nonzero_counts, "context_actual_deleted_coordinates", torch::tensor(int64_t{1}));
  rejects([&] { rpb::audit_context_replication_initialization(nonzero_counts, reference, fit, 2); }, "consumed context at point zero");
  const auto different_ratio = (directory / "wrong-context-ratio.pt").string();
  changed_audit(candidate0, different_ratio, "context_deletion_ratio_value", torch::tensor(.2, torch::kFloat64));
  rejects([&] { rpb::audit_context_replication_initialization(different_ratio, reference, fit, 2); }, "wrong typed fixed context recipe");
  const auto different_rng = (directory / "wrong-rng-policy.pt").string();
  changed_audit(candidate0, different_rng, "rng_policy", embedding::archive::text_tensor("different-counter-law"));
  rejects([&] { rpb::audit_context_replication_initialization(different_rng, reference, fit, 2); }, "wrong original counter stream law");
  const auto different_audit_protocol = (directory / "historical-producer.pt").string();
  changed_audit(candidate0, different_audit_protocol, "protocol_id", embedding::archive::text_tensor("native-curve-v1/lag_sign"));
  rejects([&] { rpb::audit_context_replication_initialization(different_audit_protocol, reference, fit, 2); }, "historical producer cannot claim fresh cohort");
  const auto different_samples = (directory / "wrong-samples.pt").string();
  changed_audit(base2, different_samples, "sampled_rows", torch::tensor(int64_t{5}));
  wrong_reference = reference; wrong_reference.reference_checkpoint = different_samples;
  rejects([&] { rpb::audit_context_replication_initialization(candidate0, wrong_reference, fit, 2); }, "incorrect reference row-presentation counter");
  for (const auto &[path, bytes] : originals) check(file_bytes(path) == bytes, "all original checkpoint/audit fixture bytes preserved");
  std::cout << "Fresh RPB context replication CUDA initialization/association contracts passed; artifacts=" << directory << '\n';
}
} // namespace

int main() {
  try {
    at::set_num_threads(1); check(torch::cuda::is_available(), "context replication integration requires actual CUDA");
    fresh_cuda_replication_initialization_contract(); return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
