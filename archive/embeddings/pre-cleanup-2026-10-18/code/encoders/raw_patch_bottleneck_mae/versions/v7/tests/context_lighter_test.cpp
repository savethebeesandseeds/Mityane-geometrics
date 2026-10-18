// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replication_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replay_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <algorithm>
#include <array>
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
namespace ctx = rpb::context_deletion;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;
constexpr auto lighter = rpb::ContextDeletionRecipe::coordinate15_v1;

std::string text(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true); return embedding::archive::tensor_text(value);
}
int64_t integer(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true);
  check(value.scalar_type() == torch::kInt64 && value.numel() == 1, "typed int64 companion: " + name);
  return value.item<int64_t>();
}
double floating(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true);
  check(value.scalar_type() == torch::kFloat64 && value.numel() == 1, "typed float64 companion: " + name);
  return value.item<double>();
}
std::string file_bytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary); check(bool(input), "protected fixture readable");
  std::ostringstream out; out << input.rdbuf(); return out.str();
}
template <typename Function>
void rejects_for(Function function, const std::string &reason, const std::string &label) {
  bool rejected = false;
  try { function(); } catch (const std::exception &error) {
    rejected = true;
    check(std::string(error.what()).find(reason) != std::string::npos, label + " rejected for wrong reason: " + error.what());
  }
  check(rejected, label + " was accepted");
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
    check(threads == at::get_num_threads(), label + " threads");
    for (size_t i = 0; i < generators.size(); ++i)
      check(torch::equal(states[i], generators[i].get_state()), label + " generator " + std::to_string(i));
  }
};
void same_model(const rpb::Model &a, const rpb::Model &b) {
  const auto x = a->named_parameters(), y = b->named_parameters();
  check(x.size() == y.size(), "exact same parameter set");
  for (const auto &p : x)
    check(y.contains(p.key()) && torch::equal(p.value(), y[p.key()]), "exact named model tensor: " + p.key());
  const auto u = a->named_buffers(), v = b->named_buffers();
  check(u.size() == v.size(), "exact same buffer set");
  for (const auto &p : u)
    check(v.contains(p.key()) && torch::equal(p.value(), v[p.key()]), "exact named buffer: " + p.key());
}
void same_optimizer(rpb::Checkpoint &a, const std::string &ap, rpb::Checkpoint &b, const std::string &bp) {
  torch::optim::AdamW x(a.model->parameters(), torch::optim::AdamWOptions(a.settings.learning_rate).weight_decay(a.settings.weight_decay));
  torch::optim::AdamW y(b.model->parameters(), torch::optim::AdamWOptions(b.settings.learning_rate).weight_decay(b.settings.weight_decay));
  rpb::load_optimizer(ap, x, a.settings.model.device); rpb::load_optimizer(bp, y, b.settings.model.device);
  check(!x.state().empty() && x.state().size() == y.state().size(), "same nonempty active AdamW states");
  const auto yp = b.model->named_parameters();
  for (const auto &p : a.model->named_parameters()) {
    const auto i = x.state().find(p.value().unsafeGetTensorImpl());
    const auto j = y.state().find(yp[p.key()].unsafeGetTensorImpl());
    check((i == x.state().end()) == (j == y.state().end()), "same named AdamW support");
    if (i == x.state().end()) continue;
    const auto *u = dynamic_cast<const torch::optim::AdamWParamState *>(i->second.get());
    const auto *v = dynamic_cast<const torch::optim::AdamWParamState *>(j->second.get());
    check(u && v && u->step() == a.completed_steps && v->step() == b.completed_steps &&
        u->exp_avg().is_cuda() && u->exp_avg_sq().is_cuda() &&
        torch::equal(u->exp_avg(), v->exp_avg()) && torch::equal(u->exp_avg_sq(), v->exp_avg_sq()),
        "exact CUDA AdamW absolute steps and moments: " + p.key());
  }
}
void same_trace(const ev::CurveProgress &a, const ev::CurveProgress &b) {
  check(a.attempted == b.attempted && a.completed == b.completed && a.sampled_rows == b.sampled_rows &&
      a.preprocessing_id == b.preprocessing_id && a.training_dataset_id == b.training_dataset_id && a.losses.size() == b.losses.size(),
      "exact default/historical numeric trace and identities");
  for (size_t i = 0; i < a.losses.size(); ++i) {
    const auto &x = a.losses[i], &y = b.losses[i];
    check(x.attempted == y.attempted && x.completed == y.completed && x.target_cells == y.target_cells &&
        x.loss == y.loss && x.gradient_norm == y.gradient_norm, "exact default/historical loss point");
  }
}
void trace_prefix(const ev::CurveProgress &a, const ev::CurveProgress &b) {
  check(a.losses.size() <= b.losses.size(), "loss trace prefix length");
  for (size_t i = 0; i < a.losses.size(); ++i) {
    const auto &x = a.losses[i], &y = b.losses[i];
    check(x.attempted == y.attempted && x.completed == y.completed && x.target_cells == y.target_cells &&
        x.loss == y.loss && x.gradient_norm == y.gradient_norm, "queried milestone trace stays an exact prefix");
  }
}
void freeze(rpb::Checkpoint &checkpoint) {
  checkpoint.model->eval(); for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(false);
}
std::array<int64_t, 3> companion(const std::string &path, int64_t completed) {
  torch::serialize::InputArchive audit; audit.load_from(path + ".audit.pt", torch::kCPU);
  check(text(audit, "artifact_kind") == "rpb_learning_curve_training_audit_v1" &&
      text(audit, "model_tag") == "RPB-v7" && text(audit, "training_policy_id") == ctx::descriptor(lighter).policy_id &&
      text(audit, "context_deletion_ratio") == "0.15" && floating(audit, "context_deletion_ratio_value") == .15 &&
      text(audit, "context_deletion_stream") == "0x6374782d64726f70" &&
      integer(audit, "context_deletion_stream_value") == static_cast<int64_t>(ctx::stream) &&
      text(audit, "context_deletion_rng_policy") == ctx::rng_policy &&
      text(audit, "context_deletion_repair_policy") == ctx::repair_policy &&
      text(audit, "context_deletion_visibility_policy") == ctx::visibility_policy &&
      text(audit, "context_deletion_count_policy") == "cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only" &&
      text(audit, "context_deletion_resume_policy") == "fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API" &&
      integer(audit, "attempted_steps") == completed && integer(audit, "completed_steps") == completed &&
      integer(audit, "sampled_rows") == completed * 2, "exact lighter typed/text policy and absolute counters");
  const std::array<int64_t, 3> counts{integer(audit, "context_requested_deleted_coordinates"),
      integer(audit, "context_actual_deleted_coordinates"), integer(audit, "context_restored_coordinates")};
  check(counts[0] >= counts[1] && counts[1] >= 0 && counts[2] == counts[0] - counts[1], "typed cumulative context count identity");
  if (!completed) check(counts == std::array<int64_t, 3>{0, 0, 0}, "point zero has not consumed E");
  return counts;
}
void changed_companion(const std::string &path, const std::string &copy,
                       const std::string &field, const torch::Tensor &replacement) {
  fs::copy_file(path, copy, fs::copy_options::none);
  torch::serialize::InputArchive old; old.load_from(path + ".audit.pt", torch::kCPU);
  torch::serialize::OutputArchive changed; bool found = false;
  for (const auto &name : old.keys()) {
    torch::Tensor value; old.read(name, value, true);
    if (name == field) { value = replacement; found = true; }
    changed.write(name, value, true);
  }
  check(found, "negative policy companion key exists"); embedding::archive::save_archive(copy + ".audit.pt", changed);
}
void ordinary_resume_rejects(const std::string &path, const fs::path &directory) {
  std::vector<std::string> args{"rpb", "train", "--resume", path, "--checkpoint", (directory / "forbidden-resume.pt").string(),
      "--input", (directory / "never-read-training.pt").string(), "--steps", "1", "--device", "cuda"};
  std::vector<char *> argv; for (auto &arg : args) argv.push_back(arg.data());
  rejects_for([&] { rpb::run_cli(static_cast<int>(argv.size()), argv.data()); },
      "ordinary train cannot resume this saved training policy", "ordinary15 resume before TRAIN access");
  check(!fs::exists(directory / "forbidden-resume.pt") && !fs::exists(directory / "never-read-training.pt"),
      "policy rejection writes no checkpoint or training archive");
}

void lighter_cuda_contract() {
  auto settings = rpb::default_settings(); auto &c = settings.model;
  c.device = torch::Device(torch::kCUDA, 0); c.global_bottleneck_mode = 2; c.channel_mixer_layers = 1;
  c.channel_ids = {101, 202, 303}; c.dropout = .1; c.sampling_interval = .5;
  settings.steps = 6; settings.batch_size = 2; settings.log_every = 3; settings.attempt_limit = 32;
  auto raw = input(c, 6); const auto order = torch::tensor({2, 0, 1}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order);
  raw.channel_ids = raw.channel_ids.index_select(0, order); raw.observed[0][0][14][1] = false;
  raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch legal{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{legal, {3, 32, 3, torch::kFloat64, torch::kCPU}, 4404,
      {"fresh-a", "fresh-a", "fresh-b", "fresh-b", "fresh-c", "fresh-c"}, {303, 101, 202},
      "volts,amperes,kelvin", "native-development-v1/lag_sign", c.sampling_interval, raw.endpoints[0].item<double>()};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-context-lighter-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);

  const RuntimeWitness invalid_runtime;
  rejects_for([&] { rpb::make_learning_curve_trainer(settings, {true, static_cast<rpb::ContextDeletionRecipe>(99)}); },
      "unknown bounded context recipe", "unknown enabled selector");
  rejects_for([&] { rpb::make_learning_curve_trainer(settings, {false, static_cast<rpb::ContextDeletionRecipe>(99)}); },
      "unknown bounded context recipe", "unknown disabled selector");
  rejects_for([&] { rpb::make_learning_curve_trainer(settings, {false, lighter}); },
      "disabled context", "disabled15 selector");
  invalid_runtime.unchanged("invalid factories reject before runtime mutation");

  auto baseline = rpb::make_learning_curve_trainer(settings)(fit);
  const RuntimeWitness baseline_initial_rng;
  auto disabled = rpb::make_learning_curve_trainer(settings, rpb::ContextDeletionOptions{})(fit);
  baseline_initial_rng.unchanged("default and explicitly disabled initialization consume identical RNG");
  auto heavy = rpb::make_learning_curve_trainer(settings, rpb::ContextDeletionOptions{true})(fit);
  const RuntimeWitness heavy_initial_rng;
  auto explicit_heavy = rpb::make_learning_curve_trainer(settings,
      {true, rpb::ContextDeletionRecipe::coordinate30_v1})(fit);
  heavy_initial_rng.unchanged("old aggregate and explicit30 initialization consume identical RNG");
  rpb::ContextDeletionOptions captured{true, lighter};
  const auto light_factory = rpb::make_learning_curve_trainer(settings, captured);
  captured.enabled = false; captured.recipe = rpb::ContextDeletionRecipe::coordinate30_v1;
  auto light = light_factory(fit), direct = light_factory(fit);
  check(light.audit_fields.at("model_tag") == "RPB-v7" && light.audit_fields.at("context_deletion_ratio") == "0.15" &&
      light.audit_fields.at("training_policy_id") == ctx::descriptor(lighter).policy_id &&
      light.audit_fields.at("rng_policy") == heavy.audit_fields.at("rng_policy") &&
      light.audit_fields.at("resolved_settings") == heavy.audit_fields.at("resolved_settings") &&
      heavy.audit_fields == explicit_heavy.audit_fields && baseline.audit_fields == disabled.audit_fields,
      "recipe captured by value; default30 and disabled audits unchanged");
  const auto base0 = (directory / "base0.pt").string(), light0 = (directory / "light0.pt").string();
  baseline.save_checkpoint(base0); light.save_checkpoint(light0); companion(light0, 0);
  auto base_initial = rpb::load_checkpoint(base0), light_initial = rpb::load_checkpoint(light0);
  same_model(base_initial.model, light_initial.model);
  const auto zero = light.snapshot(light0);
  const auto zero_features = zero.features.extract(legal).at("curve_global");
  check(zero_features.values.size(1) == 32 && zero_features.valid.all().item<bool>() &&
      zero.features.provenance.find("model_tag=RPB-v7") != std::string::npos &&
      zero_features.provenance.find("model_tag=RPB-v7") != std::string::npos &&
      zero_features.provenance.find(ctx::descriptor(lighter).policy_id) != std::string::npos,
      "ordinary native32 surfaces retain explicit15 provenance");
  const auto base_two = baseline.train_to(2), disabled_two = disabled.train_to(2);
  const auto heavy_two = heavy.train_to(2), explicit_two = explicit_heavy.train_to(2);
  same_trace(base_two, disabled_two); same_trace(heavy_two, explicit_two);
  const auto base2 = (directory / "base2.pt").string(), disabled2 = (directory / "disabled2.pt").string();
  const auto heavy2 = (directory / "heavy2.pt").string(), explicit2 = (directory / "explicit30-2.pt").string();
  baseline.save_checkpoint(base2); disabled.save_checkpoint(disabled2); heavy.save_checkpoint(heavy2); explicit_heavy.save_checkpoint(explicit2);
  auto base_cp = rpb::load_checkpoint(base2, c.device), disabled_cp = rpb::load_checkpoint(disabled2, c.device);
  auto heavy_cp = rpb::load_checkpoint(heavy2, c.device), explicit_cp = rpb::load_checkpoint(explicit2, c.device);
  same_model(base_cp.model, disabled_cp.model); same_optimizer(base_cp, base2, disabled_cp, disabled2);
  same_model(heavy_cp.model, explicit_cp.model); same_optimizer(heavy_cp, heavy2, explicit_cp, explicit2);
  check(base_cp.training_policy_id.empty() && heavy_cp.training_policy_id == ctx::policy_id,
      "disabled and default30 checkpoint identities remain unchanged");
  std::map<std::string, std::string> originals;
  for (const auto &path : {base0, base2, light0, heavy2})
    for (const std::string suffix : {"", ".audit.pt", ".training-raw.pt", ".scaler.pt"})
      originals.emplace(path + suffix, file_bytes(path + suffix));
  const auto base_zero_snapshot = baseline.snapshot(base0);
  close(zero_features.values, base_zero_snapshot.features.extract(legal).at("curve_global").values,
      "all initial native32 weights/scaler match ordinary v4", 0, 0);

  ev::RetainedPoolingCohort reference; reference.master_seed = fit.seed;
  reference.reference_initial_checkpoint = base0; reference.reference_checkpoint = base2;
  const RuntimeWitness audit_runtime;
  const auto audited = rpb::audit_context_replication_initialization(light0, reference, fit, lighter, 2);
  check(audited.at("common_parameter_count") == "225805" && audited.at("training_policy_id") == ctx::descriptor(lighter).policy_id &&
      audited.at("training_policy_companion_exact") == "true", "strict full-mode explicit15 initialization gate");
  rejects_for([&] { rpb::audit_context_replication_initialization(light0, reference, fit, 2); },
      "checkpoint policy tags differ", "legacy30 initialization gate rejects15");
  rejects_for([&] { rpb::audit_context_replication_initialization(light0, reference, fit,
      rpb::ContextDeletionRecipe::coordinate30_v1, 2); }, "checkpoint policy tags differ", "explicit30 gate rejects15");
  rejects_for([&] { rpb::audit_context_replication_initialization(light0, reference, fit,
      static_cast<rpb::ContextDeletionRecipe>(99), 2); }, "unknown bounded context recipe", "unknown initialization recipe");
  audit_runtime.unchanged("all successful and rejected gates restore CPU/all-CUDA RNG and threads");
  const auto wrong_ratio = (directory / "wrong-typed-ratio.pt").string();
  changed_companion(light0, wrong_ratio, "context_deletion_ratio_value", torch::tensor(.30, torch::kFloat64));
  rejects_for([&] { rpb::audit_context_replication_initialization(wrong_ratio, reference, fit, lighter, 2); },
      "fixed context policy", "text15 typed30 mismatch");
  const auto wrong_tag = (directory / "wrong-text-tag.pt").string();
  changed_companion(light0, wrong_tag, "model_tag", embedding::archive::text_tensor("RPB-v6"));
  rejects_for([&] { rpb::audit_context_replication_initialization(wrong_tag, reference, fit, lighter, 2); },
      "fixed context policy", "15 checkpoint with30 companion model tag");

  const auto two = light.train_to(2); const auto light2 = (directory / "light2.pt").string(); light.save_checkpoint(light2);
  const auto counts_two = companion(light2, 2);
  check(two.attempted == 2 && two.completed == 2 && two.sampled_rows == 4 && two.parameter_count == 225805 &&
      two.cuda_parameter_count == 225805 && two.last_input_cuda && two.last_loss_cuda && two.finite_gradients && two.weights_changed &&
      two.preprocessing_id == base_two.preprocessing_id && two.training_dataset_id == base_two.training_dataset_id,
      "actual unchanged-size CUDA inputs/loss/gradients/updates/scaler");
  for (size_t i = 0; i < two.losses.size(); ++i)
    check(two.losses[i].target_cells == heavy_two.losses[i].target_cells && two.losses[i].attempted == heavy_two.losses[i].attempted,
        "original A/Q target-cell counts and absolute attempts remain exact30/15");
  {
    auto manual = rpb::load_checkpoint(light0, c.device); manual.model->train();
    std::array<int64_t, 3> expected_counts{0, 0, 0};
    for (int64_t attempt = 0; attempt < 2; ++attempt) {
      const auto rows = rpb::training_detail::sampled_indices(raw.data.size(0), 2, manual.settings.seed, attempt);
      const auto batch = manual.scaler.transform(rpb::training_detail::selected(raw, rows), c);
      const auto original = rpb::make_training_mask(batch.observed, c,
          rpb::training_detail::counter_seed(manual.settings.seed, attempt, 0x6d61736bULL));
      const auto context = ctx::make_plan(original, batch.channel_ids, c, manual.settings.seed, attempt, lighter);
      expected_counts[0] += context.requested_count; expected_counts[1] += context.actual_count; expected_counts[2] += context.restored_count;
      if (attempt == 0) {
        torch::manual_seed(rpb::training_detail::counter_seed(manual.settings.seed, attempt, 0x746f726368ULL));
        const auto out = ctx::training_forward(*manual.model, batch, original, context); out.loss.backward();
        const auto norm = torch::nn::utils::clip_grad_norm_(manual.model->parameters(), manual.settings.gradient_clip_norm, 2.0, true);
        check(out.loss.is_cuda() && out.loss.item<double>() == two.losses.front().loss &&
            norm == two.losses.front().gradient_norm && out.target_cell_count == two.losses.front().target_cells,
            "actual first CUDA update uses exact15 visibility, served32 and original Huber targets");
      }
    }
    check(expected_counts == counts_two, "cumulative E counts match independently regenerated absolute attempts");
  }
  auto hidden = torch::zeros_like(raw.observed); hidden.narrow(2, 0, c.patch_length).fill_(true);
  const auto snapshot = light.snapshot(light2);
  const auto features = snapshot.features.extract(legal).at("curve_global");
  const auto prediction = snapshot.reconstruct(legal, hidden);
  {
    torch::NoGradGuard no_grad;
    auto cpu = rpb::load_checkpoint(light2, torch::kCPU); freeze(cpu);
    const auto normalized = cpu.scaler.transform(raw, cpu.settings.model);
    const auto encoded = cpu.model->encode(normalized);
    close(features.values, rpb::compact_reconstruction_export(encoded, cpu.settings.model), "exact CPU checkpoint native32 export", 0, 0);
    auto gpu = rpb::load_checkpoint(light2, c.device); freeze(gpu);
    const auto active = gpu.model->forward(gpu.scaler.transform(raw, gpu.settings.model), hidden);
    close(prediction.prediction, active.reconstruction.to(torch::kCPU), "exact frozen CUDA checkpoint32 decoder", 0, 0);
    bool pool_changed = false, decoder_changed = false;
    for (const auto &p : gpu.model->named_parameters()) {
      if (p.key().rfind("global_pool_", 0) == 0) pool_changed |= !torch::equal(p.value().to(torch::kCPU), light_initial.model->named_parameters()[p.key()]);
      if (p.key().rfind("decoder_", 0) == 0) decoder_changed |= !torch::equal(p.value().to(torch::kCPU), light_initial.model->named_parameters()[p.key()]);
    }
    check(pool_changed && decoder_changed, "actual learned-global and decoder weights change");
  }
  auto changed = legal; changed.data = legal.data.clone(); changed.data.masked_fill_(hidden, 123456.0);
  close(snapshot.reconstruct(changed, hidden).prediction, prediction.prediction, "fixed-query hidden values never enter frozen inference", 0, 0);
  auto empty = legal; empty.feature_mask = torch::zeros_like(legal.feature_mask);
  empty.data = torch::full_like(legal.data, std::numeric_limits<double>::quiet_NaN());
  const auto absent = snapshot.features.extract(empty).at("curve_global");
  check(!absent.valid.any().item<bool>() && absent.values.eq(0).all().item<bool>(), "15 training never invents inference support");
  const auto frozen_counts = snapshot.features.audit_fields.at("context_actual_deleted_coordinates");
  const auto four = light.train_to(4); const auto light4 = (directory / "light4.pt").string(); light.save_checkpoint(light4);
  const auto counts_four = companion(light4, 4);
  const auto six = light.train_to(6); const auto light6 = (directory / "light6.pt").string(); light.save_checkpoint(light6);
  const auto counts_six = companion(light6, 6);
  trace_prefix(two, four); trace_prefix(four, six);
  check(counts_two[0] < counts_four[0] && counts_four[0] < counts_six[0] && frozen_counts == std::to_string(counts_two[1]),
      "live context counts accumulate without mutating snapshot counts");
  const auto uninterrupted = direct.train_to(6); const auto direct6 = (directory / "direct6.pt").string(); direct.save_checkpoint(direct6);
  auto split_cp = rpb::load_checkpoint(light6, c.device), direct_cp = rpb::load_checkpoint(direct6, c.device);
  same_model(split_cp.model, direct_cp.model); same_optimizer(split_cp, light6, direct_cp, direct6);
  check(companion(direct6, 6) == counts_six && uninterrupted.attempted == six.attempted &&
      uninterrupted.sampled_rows == six.sampled_rows, "split2/4/6 equals uninterrupted6 state/counts/optimizer");
  for (const auto &point : uninterrupted.losses) {
    const auto found = std::find_if(six.losses.begin(), six.losses.end(), [&](const ev::CurveLossPoint &p) { return p.completed == point.completed; });
    check(found != six.losses.end() && found->loss == point.loss && found->gradient_norm == point.gradient_norm &&
        found->target_cells == point.target_cells, "uninterrupted periodic loss points match split trajectory");
  }
  close(snapshot.features.extract(legal).at("curve_global").values, features.values, "point2 CPU feature snapshot immutable through6", 0, 0);
  close(snapshot.reconstruct(legal, hidden).prediction, prediction.prediction, "point2 CUDA reconstruction snapshot immutable through6", 0, 0);
  close(zero.features.extract(legal).at("curve_global").values, zero_features.values, "point0 snapshot immutable through6", 0, 0);
  ordinary_resume_rejects(light2, directory);
  const RuntimeWitness replay_runtime;
  rpb::ContextReplayOptions replay{light2, light2 + ".training-raw.pt", (directory / "forbidden-v6-replay.pt").string(), 2, 6};
  rejects_for([&] { rpb::make_context_replay_trainer(replay); }, "parent context policy differs from the fixed recipe",
      "old v6 replay rejects15 for exact policy reason");
  replay_runtime.unchanged("v6-only rejected replay runtime");
  check(!fs::exists(replay.replay_checkpoint), "v6 replay cannot write a15 replay point");
  for (const auto &[path, bytes] : originals)
    check(file_bytes(path) == bytes, "all original checkpoint/raw/scaler/audit fixture bytes remain preserved");
  std::cout << "RPB lighter .15 CUDA/policy/continuity tests passed; artifacts=" << directory << '\n';
}
} // namespace

int main() {
  try { at::set_num_threads(1); check(torch::cuda::is_available(), "lighter context test requires actual CUDA"); lighter_cuda_contract(); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
