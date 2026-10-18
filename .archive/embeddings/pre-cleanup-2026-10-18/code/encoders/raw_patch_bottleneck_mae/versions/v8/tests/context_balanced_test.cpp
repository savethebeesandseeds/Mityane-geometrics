// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replication_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replay_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/paired_pooling_adapter.h"
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
constexpr auto balanced = rpb::ContextDeletionRecipe::balanced30_v1;

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
      text(audit, "model_tag") == "RPB-v8" && text(audit, "training_policy_id") == ctx::descriptor(balanced).policy_id &&
      text(audit, "context_deletion_ratio") == "0.30" && floating(audit, "context_deletion_ratio_value") == .30 &&
      text(audit, "context_deletion_stream") == "0x6374782d64726f70" &&
      integer(audit, "context_deletion_stream_value") == static_cast<int64_t>(ctx::stream) &&
      text(audit, "context_deletion_rng_policy") == ctx::rng_policy &&
      text(audit, "context_deletion_repair_policy") == ctx::repair_policy &&
      text(audit, "context_deletion_visibility_policy") == ctx::visibility_policy &&
      text(audit, "context_deletion_count_policy") == "cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only" &&
      text(audit, "context_deletion_resume_policy") == "fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API" &&
      integer(audit, "attempted_steps") == completed && integer(audit, "completed_steps") == completed &&
      integer(audit, "sampled_rows") == completed * 2, "exact balanced typed/text policy and absolute counters");
  check(text(audit, "context_deletion_schedule_policy") == ctx::balanced_schedule_policy &&
      text(audit, "context_deletion_rate_scope") == ctx::balanced_rate_scope &&
      text(audit, "context_deletion_branch_count_policy") == ctx::balanced_branch_count_policy &&
      text(audit, "context_deletion_skip_policy") == ctx::balanced_skip_policy &&
      text(audit, "context_ordinary_attempts") == std::to_string(completed / 2 + completed % 2) &&
      text(audit, "context_deletion_attempts") == std::to_string(completed / 2) &&
      integer(audit, "context_ordinary_attempts_value") == completed / 2 + completed % 2 &&
      integer(audit, "context_deletion_attempts_value") == completed / 2,
      "exact active-.30 schedule and typed/text branch counters");
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
      "ordinary train cannot resume this saved training policy", "ordinary balanced resume before TRAIN access");
  check(!fs::exists(directory / "forbidden-resume.pt") && !fs::exists(directory / "never-read-training.pt"),
      "policy rejection writes no checkpoint or training archive");
}

void balanced_cuda_contract() {
  auto settings = rpb::default_settings(); auto &c = settings.model;
  c.device = torch::Device(torch::kCUDA, 0); c.global_bottleneck_mode = 2; c.channel_mixer_layers = 1;
  c.channel_ids = {101, 202, 303}; c.dropout = .1; c.sampling_interval = .5;
  settings.steps = 4; settings.batch_size = 2; settings.log_every = 1; settings.attempt_limit = 16;
  auto raw = input(c, 6); const auto order = torch::tensor({2, 0, 1}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order);
  raw.channel_ids = raw.channel_ids.index_select(0, order); raw.observed[0][0][14][1] = false;
  raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch legal{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{legal, {3, 32, 3, torch::kFloat64, torch::kCPU}, 4404,
      {"fresh-a", "fresh-a", "fresh-b", "fresh-b", "fresh-c", "fresh-c"}, {303, 101, 202},
      "volts,amperes,kelvin", "native-development-v1/lag_sign", c.sampling_interval, raw.endpoints[0].item<double>()};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-context-balanced-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(directory);
  const RuntimeWitness invalid_runtime;
  rejects_for([&] { rpb::make_learning_curve_trainer(settings, {false, balanced}); }, "disabled context", "disabled balanced selector");
  rejects_for([&] { rpb::make_learning_curve_trainer(settings, {true, static_cast<rpb::ContextDeletionRecipe>(99)}); },
      "unknown bounded context recipe", "unknown selector");
  invalid_runtime.unchanged("malformed options reject before RNG mutation");
  int ordinary = 0, deleted = 0;
  for (int64_t a = 0; a < 512; ++a) { if (ctx::deletion_attempt(balanced, a)) ++deleted; else ++ordinary; }
  check(ordinary == 256 && deleted == 256 && ctx::descriptor(balanced).ratio == .30 &&
      std::string(ctx::descriptor(balanced).model_tag) == "RPB-v8", "fixed512 is256 ordinary/256 active30, never uniform15");

  auto baseline = rpb::make_learning_curve_trainer(settings)(fit);
  const RuntimeWitness initial_rng;
  rpb::ContextDeletionOptions captured{true, balanced};
  const auto factory = rpb::make_learning_curve_trainer(settings, captured);
  captured.recipe = rpb::ContextDeletionRecipe::coordinate15_v1;
  auto split = factory(fit); initial_rng.unchanged("balanced initialization equals ordinary RNG state");
  auto direct = factory(fit);
  auto heavy = rpb::make_learning_curve_trainer(settings, {true})(fit);
  auto explicit_heavy = rpb::make_learning_curve_trainer(settings, {true, rpb::ContextDeletionRecipe::coordinate30_v1})(fit);
  auto disabled = rpb::make_learning_curve_trainer(settings, {})(fit);
  check(split.audit_fields.at("model_tag") == "RPB-v8" &&
      split.audit_fields.at("context_deletion_schedule_policy") == ctx::balanced_schedule_policy &&
      heavy.audit_fields == explicit_heavy.audit_fields && disabled.audit_fields == baseline.audit_fields,
      "captured balanced policy and legacy audits exact");
  const auto base0 = (directory / "base0.pt").string(), split0 = (directory / "split0.pt").string();
  baseline.save_checkpoint(base0); split.save_checkpoint(split0); companion(split0, 0);
  std::map<std::string, std::string> protected_bytes;
  for (const auto &path : {base0, split0})
    for (const std::string suffix : {"", ".audit.pt", ".training-raw.pt", ".scaler.pt"})
      protected_bytes.emplace(path + suffix, file_bytes(path + suffix));
  auto base_initial = rpb::load_checkpoint(base0), initial = rpb::load_checkpoint(split0);
  same_model(base_initial.model, initial.model);
  const auto zero = split.snapshot(split0); const auto zero_features = zero.features.extract(legal).at("curve_global");
  check(zero_features.values.size(1) == 32 && zero_features.valid.all().item<bool>() &&
      zero.features.provenance.find("model_tag=RPB-v8") != std::string::npos, "immutable ordinary native32 serving with balanced producer tag");
  const auto base_one = baseline.train_to(1), one = split.train_to(1), disabled_one = disabled.train_to(1);
  same_trace(base_one, one); same_trace(base_one, disabled_one);
  const auto base1 = (directory / "base1.pt").string(), split1 = (directory / "split1.pt").string();
  const auto disabled1 = (directory / "disabled1.pt").string();
  baseline.save_checkpoint(base1); split.save_checkpoint(split1); disabled.save_checkpoint(disabled1);
  check(companion(split1, 1) == std::array<int64_t, 3>{0, 0, 0}, "first ordinary step consumes no E coordinates");
  auto a1 = rpb::load_checkpoint(base1, c.device), b1 = rpb::load_checkpoint(split1, c.device);
  auto d1 = rpb::load_checkpoint(disabled1, c.device);
  same_model(a1.model, b1.model); same_optimizer(a1, base1, b1, split1);
  same_model(a1.model, d1.model); same_optimizer(a1, base1, d1, disabled1);
  const auto snapshot1 = split.snapshot(split1); const auto features1 = snapshot1.features.extract(legal).at("curve_global");
  auto hidden = torch::zeros_like(raw.observed); hidden.narrow(2, 0, c.patch_length).fill_(true);
  const auto prediction1 = snapshot1.reconstruct(legal, hidden);

  const auto two = split.train_to(2); const auto split2 = (directory / "split2.pt").string(); split.save_checkpoint(split2);
  const auto counts_two = companion(split2, 2);
  check(two.parameter_count == 225805 && two.cuda_parameter_count == 225805 && two.last_input_cuda && two.last_loss_cuda &&
      two.finite_gradients && two.weights_changed && two.attempted == 2 && two.completed == 2 && two.sampled_rows == 4,
      "actual CUDA unchanged architecture gradients and updates");
  {
    auto manual = rpb::load_checkpoint(split1, c.device); manual.model->train();
    const int64_t attempt = 1;
    const auto rows = rpb::training_detail::sampled_indices(raw.data.size(0), 2, manual.settings.seed, attempt);
    const auto batch = manual.scaler.transform(rpb::training_detail::selected(raw, rows), c);
    const auto original = rpb::make_training_mask(batch.observed, c,
        rpb::training_detail::counter_seed(manual.settings.seed, attempt, 0x6d61736bULL));
    const RuntimeWitness plan_rng;
    const auto plan = ctx::make_plan(original, batch.channel_ids, c, manual.settings.seed, attempt, balanced);
    const auto old = ctx::make_plan(original, batch.channel_ids, c, manual.settings.seed, attempt);
    const auto even = ctx::make_plan(original, batch.channel_ids, c, manual.settings.seed, 2, balanced);
    plan_rng.unchanged("stateless odd30/even plans consume no runtime RNG");
    check(torch::equal(plan.visible, old.visible) && torch::equal(plan.deleted, old.deleted) &&
        torch::equal(plan.requested_deleted, old.requested_deleted) && plan.requested_count == old.requested_count &&
        plan.actual_count == old.actual_count && plan.restored_count == old.restored_count,
        "odd branch exact legacy30 draws and repair at original absolute attempt");
    check(torch::equal(even.visible, original.visible) && !even.deleted.any().item<bool>() &&
        even.requested_count == 0 && even.actual_count == 0 && even.restored_count == 0, "even primitive preserves original V");
    check(counts_two == std::array<int64_t, 3>{plan.requested_count, plan.actual_count, plan.restored_count},
        "only odd attempt accrues coordinate counts");
    const auto perm = torch::tensor({1, 2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(c.device));
    const auto perm_o = batch.observed.index_select(1, perm), perm_h = original.hidden.index_select(1, perm);
    const auto perm_original = rpb::mask_from_hidden(perm_o, perm_h, c);
    const auto perm_plan = ctx::make_plan(perm_original, batch.channel_ids.index_select(batch.channel_ids.dim() == 1 ? 0 : 1, perm), c, manual.settings.seed, attempt, balanced);
    check(torch::equal(plan.visible.index_select(1, perm), perm_plan.visible) &&
        torch::equal(plan.requested_deleted.index_select(1, perm), perm_plan.requested_deleted), "semantic-order schedule draws and repair invariant");
    check(!plan.deleted.logical_and(original.visible.logical_not()).any().item<bool>() &&
        !plan.visible.logical_and(original.target).any().item<bool>(), "original query protected; no invented context support");
    torch::manual_seed(rpb::training_detail::counter_seed(manual.settings.seed, attempt, 0x746f726368ULL));
    const auto out = ctx::training_forward(*manual.model, batch, original, plan); out.loss.backward();
    const auto norm = torch::nn::utils::clip_grad_norm_(manual.model->parameters(), manual.settings.gradient_clip_norm, 2.0, true);
    check(out.loss.item<double>() == two.losses.back().loss && norm == two.losses.back().gradient_norm &&
        out.target_cell_count == two.losses.back().target_cells && torch::equal(out.target_counts, original.target.sum(std::vector<int64_t>{2, 3})),
        "actual odd update uses exact30 V and unchanged original Q/Huber counts");
  }
  const auto base_two = baseline.train_to(2); const auto base2 = (directory / "base2.pt").string(); baseline.save_checkpoint(base2);
  check(base_two.preprocessing_id == two.preprocessing_id && base_two.training_dataset_id == two.training_dataset_id &&
      base_two.losses.back().target_cells == two.losses.back().target_cells, "balanced preserves original scaler/TRAIN/Q counts");
  const auto heavy_two = heavy.train_to(2), explicit_two = explicit_heavy.train_to(2); same_trace(heavy_two, explicit_two);
  const auto heavy2 = (directory / "heavy2.pt").string(), explicit2 = (directory / "explicit2.pt").string();
  heavy.save_checkpoint(heavy2); explicit_heavy.save_checkpoint(explicit2);
  auto h2 = rpb::load_checkpoint(heavy2, c.device), e2 = rpb::load_checkpoint(explicit2, c.device);
  same_model(h2.model, e2.model); same_optimizer(h2, heavy2, e2, explicit2);
  check(h2.training_policy_id == ctx::policy_id && a1.training_policy_id.empty(), "old ordinary and30 checkpoint IDs preserved");
  ev::RetainedPoolingCohort reference; reference.master_seed = fit.seed;
  reference.reference_initial_checkpoint = base0; reference.reference_checkpoint = base2;
  const RuntimeWitness audit_rng;
  const auto audited = rpb::audit_context_replication_initialization(split0, reference, fit, balanced, 2);
  check(audited.at("common_parameter_count") == "225805" && audited.at("training_policy_companion_exact") == "true",
      "strict complete same-mode balanced initialization admission");
  rejects_for([&] { rpb::audit_context_replication_initialization(split0, reference, fit, 2); }, "checkpoint policy tags differ", "old30 gate rejects balanced");
  rejects_for([&] { rpb::audit_context_replication_initialization(split0, reference, fit, rpb::ContextDeletionRecipe::coordinate15_v1, 2); },
      "checkpoint policy tags differ", "15 gate rejects balanced");
  audit_rng.unchanged("positive/negative initialization audits restore RNG/thread state");
  const auto wrong_schedule = (directory / "wrong-schedule.pt").string();
  changed_companion(split0, wrong_schedule, "context_deletion_schedule_policy", embedding::archive::text_tensor("uniform15"));
  rejects_for([&] { rpb::audit_context_replication_initialization(wrong_schedule, reference, fit, balanced, 2); },
      "schedule/rate scope/branch-count companion differs", "wrong balanced schedule rejected");
  const auto wrong_count = (directory / "wrong-count.pt").string();
  changed_companion(split0, wrong_count, "context_deletion_attempts_value", torch::tensor(int64_t{1}));
  rejects_for([&] { rpb::audit_context_replication_initialization(wrong_count, reference, fit, balanced, 2); },
      "schedule/rate scope/branch-count companion differs", "point0 typed branch mismatch rejected");

  const auto snapshot2 = split.snapshot(split2); const auto features2 = snapshot2.features.extract(legal).at("curve_global");
  const auto prediction2 = snapshot2.reconstruct(legal, hidden);
  {
    torch::NoGradGuard no_grad;
    auto gpu = rpb::load_checkpoint(split2, c.device); freeze(gpu);
    const auto output = gpu.model->forward(gpu.scaler.transform(raw, c), hidden);
    close(prediction2.prediction, output.reconstruction.to(torch::kCPU), "exact ordinary frozen CUDA checkpoint32 decoder", 0, 0);
    auto cpu = rpb::load_checkpoint(split2); freeze(cpu);
    close(features2.values, rpb::compact_reconstruction_export(cpu.model->encode(cpu.scaler.transform(raw, cpu.settings.model)), cpu.settings.model),
        "exact native32 CPU checkpoint serving", 0, 0);
    bool pool_changed = false, decoder_changed = false;
    for (const auto &p : cpu.model->named_parameters()) {
      if (p.key().rfind("global_pool_", 0) == 0) pool_changed |= !torch::equal(p.value(), initial.model->named_parameters()[p.key()]);
      if (p.key().rfind("decoder_", 0) == 0) decoder_changed |= !torch::equal(p.value(), initial.model->named_parameters()[p.key()]);
    }
    check(pool_changed && decoder_changed, "actual global32 pool and decoder weights change");
  }
  auto perturbed = legal; perturbed.data = legal.data.clone(); perturbed.data.masked_fill_(hidden, 123456.0);
  close(snapshot2.reconstruct(perturbed, hidden).prediction, prediction2.prediction, "hidden query values never reach frozen decoder route", 0, 0);
  auto absent = legal; absent.feature_mask = torch::zeros_like(legal.feature_mask);
  absent.data = torch::full_like(legal.data, std::numeric_limits<double>::quiet_NaN());
  const auto unsupported = snapshot2.features.extract(absent).at("curve_global");
  check(!unsupported.valid.any().item<bool>() && unsupported.values.eq(0).all().item<bool>(), "balanced training invents no inference support");
  const auto four = split.train_to(4); const auto split4 = (directory / "split4.pt").string(); split.save_checkpoint(split4);
  const auto direct_four = direct.train_to(4); const auto direct4 = (directory / "direct4.pt").string(); direct.save_checkpoint(direct4);
  trace_prefix(one, two); trace_prefix(two, four); same_trace(four, direct_four);
  auto s4 = rpb::load_checkpoint(split4, c.device), u4 = rpb::load_checkpoint(direct4, c.device);
  same_model(s4.model, u4.model); same_optimizer(s4, split4, u4, direct4);
  check(companion(split4, 4) == companion(direct4, 4), "direct0/4 and split0/1/2/4 exact cumulative context/branch counts");
  {
    const auto active = split.snapshot(split4);
    const auto expected_features = active.features.extract(legal).at("curve_global");
    const auto expected_reconstruction = active.reconstruct(legal, hidden);
    const RuntimeWitness retained_runtime;
    const auto retained = rpb::make_retained_curve_snapshot(split4, fit);
    retained_runtime.unchanged("balanced retained loader restores CPU/all-CUDA RNG and threads");
    const auto &fields = retained.features.audit_fields;
    torch::serialize::InputArchive audit; audit.load_from(split4 + ".audit.pt", torch::kCPU);
    check(fields.at("model_tag") == "RPB-v8" && fields.at("training_policy_id") == ctx::descriptor(balanced).policy_id &&
        fields.at("context_deletion_schedule_policy") == ctx::balanced_schedule_policy &&
        fields.at("context_deletion_rate_scope") == ctx::balanced_rate_scope &&
        fields.at("context_deletion_branch_count_policy") == ctx::balanced_branch_count_policy &&
        fields.at("context_deletion_skip_policy") == ctx::balanced_skip_policy &&
        fields.at("context_ordinary_attempts") == text(audit, "context_ordinary_attempts") &&
        fields.at("context_deletion_attempts") == text(audit, "context_deletion_attempts") &&
        fields.at("context_ordinary_attempts") == "2" && fields.at("context_deletion_attempts") == "2" &&
        integer(audit, "context_ordinary_attempts_value") == 2 && integer(audit, "context_deletion_attempts_value") == 2,
        "restored balanced snapshot policy/schedule and typed/text two-plus-two branches");
    close(retained.features.extract(legal).at("curve_global").values, expected_features.values,
        "retained balanced native32 parity", 0, 0);
    const auto reconstructed = retained.reconstruct(legal, hidden);
    close(reconstructed.prediction, expected_reconstruction.prediction, "retained balanced CUDA reconstruction parity", 0, 0);
    close(reconstructed.target, expected_reconstruction.target, "retained balanced frozen-scaler target parity", 0, 0);
    check(torch::equal(reconstructed.eligible, expected_reconstruction.eligible), "retained balanced eligibility parity");
    same_trace(four, split.train_to(4));
    const auto after_path = (directory / "after-retained4.pt").string(); split.save_checkpoint(after_path);
    auto after = rpb::load_checkpoint(after_path, c.device);
    same_model(s4.model, after.model); same_optimizer(s4, split4, after, after_path);
    check(companion(after_path, 4) == companion(split4, 4), "retained snapshot leaves live model/AdamW/counter state unchanged");
  }
  check(snapshot1.features.audit_fields.at("context_ordinary_attempts") == "1" &&
      snapshot1.features.audit_fields.at("context_deletion_attempts") == "0" &&
      snapshot2.features.audit_fields.at("context_deletion_attempts") == "1", "saved snapshot branch counts immutable");
  close(snapshot1.features.extract(legal).at("curve_global").values, features1.values, "point1 feature immutable through4", 0, 0);
  close(snapshot1.reconstruct(legal, hidden).prediction, prediction1.prediction, "point1 reconstruction immutable through4", 0, 0);
  close(snapshot2.features.extract(legal).at("curve_global").values, features2.values, "point2 feature immutable through4", 0, 0);
  close(snapshot2.reconstruct(legal, hidden).prediction, prediction2.prediction, "point2 reconstruction immutable through4", 0, 0);
  close(zero.features.extract(legal).at("curve_global").values, zero_features.values, "point0 immutable through4", 0, 0);
  ordinary_resume_rejects(split2, directory);
  rpb::ContextReplayOptions replay{split2, split2 + ".training-raw.pt", (directory / "forbidden-v6-replay.pt").string(), 2, 4};
  rejects_for([&] { rpb::make_context_replay_trainer(replay); }, "parent context policy differs from the fixed recipe", "v6 replay rejects balanced policy");

  auto poor_fit = fit; poor_fit.training_observations.feature_mask = torch::zeros_like(raw.observed);
  poor_fit.training_observations.feature_mask.narrow(2, 0, c.patch_length).fill_(true);
  poor_fit.training_observations.data = torch::where(poor_fit.training_observations.feature_mask, raw.data, torch::zeros_like(raw.data));
  auto poor = factory(poor_fit);
  rejects_for([&] { poor.train_to(1); }, "aborts an ineligible original masking attempt", "balanced ineligible attempt abort before update");
  rejects_for([&] { poor.train_to(1); }, "previously aborted", "failed schedule cannot replace/shift attempt");
  const auto forbidden = (directory / "failed-balanced.pt").string();
  rejects_for([&] { poor.save_checkpoint(forbidden); }, "previously aborted", "failed policy cannot save usable checkpoint");
  check(!fs::exists(forbidden), "skip abort writes no checkpoint");
  for (const auto &[path, bytes] : protected_bytes)
    check(file_bytes(path) == bytes, "initial checkpoint/raw/scaler/audit bytes remain immutable");
  std::cout << "RPB balanced .30 CUDA/policy/continuity tests passed; artifacts=" << directory << '\n';
}
} // namespace

int main() {
  try { at::set_num_threads(1); check(torch::cuda::is_available(), "balanced context test requires actual CUDA"); balanced_cuda_contract(); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
