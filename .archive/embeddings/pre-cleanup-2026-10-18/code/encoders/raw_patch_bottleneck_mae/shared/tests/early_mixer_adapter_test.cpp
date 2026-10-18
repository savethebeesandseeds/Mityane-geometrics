// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/early_mixer_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_native_feature_adapter.h"
#include "embedding/shared/data.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#ifndef EARLY_MIXER_ADAPTER_SOURCE_ID
#define EARLY_MIXER_ADAPTER_SOURCE_ID "unrecorded"
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;
struct RuntimeWitness {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeWitness() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &g : generators) states.push_back(g.get_state().clone());
  }
  void unchanged(const std::string &label) const {
    check(at::get_num_threads() == threads, label + " threads");
    for (size_t i = 0; i < generators.size(); ++i) check(torch::equal(states[i], generators[i].get_state()), label + " RNG");
  }
};
// Only historical fixture construction/loading/training lacks an ambient-RNG
// promise. New adapter calls remain unwrapped and have their own witnesses.
struct FixtureIsolation : RuntimeWitness {
  ~FixtureIsolation() {
    for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]);
    at::set_num_threads(threads);
  }
};
template <typename Function> auto fixture(Function function) { const FixtureIsolation isolation; return function(); }
void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &label) {
  check(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() && torch::equal(a.to(torch::kCPU), b.to(torch::kCPU)), label);
}
void same_model(const rpb::Model &a, const rpb::Model &b, const std::string &label) {
  const auto reference = b->named_parameters(); check(a->named_parameters().size() == reference.size(), label + " names");
  for (const auto &p : a->named_parameters()) exact(p.value(), reference[p.key()], label + '/' + p.key());
  const auto buffers = b->named_buffers(); check(a->named_buffers().size() == buffers.size(), label + " buffers");
  for (const auto &v : a->named_buffers()) exact(v.value(), buffers[v.key()], label + '/' + v.key());
}
std::string text(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor value; a.read(key, value, true); return embedding::archive::tensor_text(value);
}
std::string bytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary); check(bool(in), "read artifact bytes");
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
rpb::EarlyMixerSnapshotOptions options(const std::string &path, int64_t placement, int64_t budget, const ev::CurveTrainer &trainer) {
  return {path, placement, budget, rpb::EarlyMixerScope::engineering,
      trainer.audit_fields.at("core_writer_source_fingerprint"), trainer.audit_fields.at("training_producer_source_fingerprint")};
}
void same_optimizer(const std::string &left, const std::string &right) {
  fixture([&] {
    auto a = rpb::load_checkpoint(left, torch::Device(torch::kCUDA, 0));
    auto b = rpb::load_checkpoint(right, torch::Device(torch::kCUDA, 0));
    torch::optim::AdamW oa(a.model->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
    torch::optim::AdamW ob(b.model->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
    rpb::load_optimizer(left, oa, torch::Device(torch::kCUDA, 0)); rpb::load_optimizer(right, ob, torch::Device(torch::kCUDA, 0));
    const auto pa = a.model->parameters(), pb = b.model->parameters(); check(oa.state().size() == ob.state().size(), "AdamW state names");
    for (size_t i = 0; i < pa.size(); ++i) {
      const auto aa = oa.state().find(pa[i].unsafeGetTensorImpl()), bb = ob.state().find(pb[i].unsafeGetTensorImpl());
      check((aa == oa.state().end()) == (bb == ob.state().end()), "AdamW active parameter set");
      if (aa == oa.state().end()) continue;
      const auto *sa = dynamic_cast<const torch::optim::AdamWParamState *>(aa->second.get());
      const auto *sb = dynamic_cast<const torch::optim::AdamWParamState *>(bb->second.get());
      check(sa && sb && sa->step() == sb->step() && sa->exp_avg().is_cuda() && sb->exp_avg_sq().is_cuda(), "CUDA AdamW steps/devices");
      exact(sa->exp_avg(), sb->exp_avg(), "continuous first moments"); exact(sa->exp_avg_sq(), sb->exp_avg_sq(), "continuous second moments");
    }
  });
}
void same_trace(const ev::CurveProgress &left, const ev::CurveProgress &right) {
  check(left.losses.size() == right.losses.size(), "all-step trace lengths");
  for (size_t i = 0; i < left.losses.size(); ++i) {
    const auto &a = left.losses[i], &b = right.losses[i];
    check(a.attempted == b.attempted && a.completed == b.completed && a.target_cells == b.target_cells &&
        a.loss == b.loss && a.gradient_norm == b.gradient_norm, "exact continuous loss/gradient/target trace");
  }
}
void change_audit(const std::string &path, const std::string &key, const torch::Tensor &replacement) {
  torch::serialize::InputArchive input; input.load_from(path, torch::kCPU); torch::serialize::OutputArchive output;
  for (const auto &name : input.keys()) { torch::Tensor value; input.read(name, value, true); output.write(name, name == key ? replacement : value, true); }
  embedding::archive::save_archive(path, output);
}
std::string copy_parent(const fs::path &directory, const std::string &parent, const std::string &name) {
  const auto path = (directory / name).string();
  for (const auto &suffix : {std::string(), std::string(".audit.pt"), std::string(".scaler.pt"), std::string(".training-raw.pt")})
    check(fs::copy_file(parent + suffix, path + suffix), "new negative fixture parent");
  return path;
}
void run() {
  check(torch::cuda::is_available(), "actual CUDA required; no skip or CPU fallback");
  const RuntimeWitness overall;
  auto settings = rpb::default_settings(); settings.model.channel_mixer_layers = 1; settings.model.global_bottleneck_mode = 2;
  settings.model.device = torch::Device(torch::kCUDA, 0); settings.steps = 4; settings.batch_size = 8;
  settings.attempt_limit = 8; settings.log_every = 1; settings.threads = 1;
  const auto raw = input(settings.model, 8);
  auto observed = raw.observed.clone(); observed.select(2, 3).select(2, 2).fill_(false);
  const auto legal = raw.data.masked_fill(observed.logical_not(), 0);
  ev::ProviderFitInput fit{{legal.clone(), observed.clone()}, {3,32,3,torch::kFloat64,torch::kCPU}, 202,
      {"a","a","b","b","c","c","d","d"}, {0,1,2}, "unitless,unitless,unitless", rpb::kEarlyMixerFixtureFitProtocol, 1,31};
  auto original = fit; original.training_observations = {legal.clone(), observed.clone()};
  const embedding::Batch batch{legal.clone(), observed.clone()};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-early-mixer-adapter-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory), "exclusive fixture directory");
  auto early_settings = settings; early_settings.model.channel_mixer_placement = 1;
  const RuntimeWitness declaration;
  rejects([&] { rpb::make_early_mixer_trainer(settings, static_cast<rpb::EarlyMixerScope>(99)); }, "unknown scope");
  auto invalid = early_settings; invalid.model.device = torch::kCPU;
  rejects([&] { rpb::make_early_mixer_trainer(invalid, rpb::EarlyMixerScope::engineering); }, "CPU settings");
  rejects([&] { rpb::make_early_mixer_trainer(settings); }, "engineering budget under quality scope");
  declaration.unchanged("malformed declarations before RNG changes");
  auto late = rpb::make_early_mixer_trainer(settings, rpb::EarlyMixerScope::engineering)(fit);
  auto early = rpb::make_early_mixer_trainer(early_settings, rpb::EarlyMixerScope::engineering)(fit);
  auto direct = rpb::make_early_mixer_trainer(early_settings, rpb::EarlyMixerScope::engineering)(fit);
  check(late.audit_fields.at("model_tag") == "RPB-v7" && early.audit_fields.at("model_tag") == "RPB-v10" &&
      early.audit_fields.at("training_policy_id") == "rpb-training-context-deletion-015-v1", "architecture tag and unchanged15 policy");
  const auto late0 = (directory/"late0.pt").string(), early0 = (directory/"early0.pt").string();
  late.train_to(0); early.train_to(0); late.save_checkpoint(late0); early.save_checkpoint(early0);
  const auto pairing = rpb::audit_early_mixer_initialization(options(late0,0,0,late),options(early0,1,0,early),fit);
  for (const auto &key : {"common_parameters_exact","common_buffers_exact","scaler_exact","training_dataset_exact","counter_streams_exact"})
    check(pairing.at(key) == "true", std::string("complete initialized pairing/") + key);
  check(pairing.at("initial_features_equality_required") == "false", "different architecture need not have equal initial features");
  const auto zero = early.snapshot(early0); const auto zero_values = zero.features.extract(batch).at("curve_global");
  check(zero.features.surfaces.size() == 1 && zero.features.surfaces.at("curve_global").kind == ev::SurfaceKind::global &&
      zero_values.values.scalar_type() == torch::kFloat32 && zero_values.values.device().is_cpu() && zero_values.valid.all().item<bool>(), "only CPU evidence from native CUDA32");
  // The factory owns its fit rows, support and identities; caller mutation cannot
  // alter its saved points or their admission metadata.
  fit.training_observations.data.fill_(999); fit.training_observations.feature_mask.fill_(false); fit.training_source_ids[0] = "changed";
  const auto two = early.train_to(2); check(two.completed == 2 && two.attempted == 2 && two.sampled_rows == 16 &&
      two.parameter_count == 225805 && two.cuda_parameter_count == 225805 && two.last_input_cuda && two.last_loss_cuda &&
      two.finite_gradients && two.weights_changed && two.losses.size() == 2, "real full-size CUDA updates and every-step trace");
  const auto early2 = (directory/"early2.pt").string(); early.save_checkpoint(early2); const auto point_two = early.snapshot(early2);
  const auto features_two = point_two.features.extract(batch).at("curve_global");
  auto hidden = torch::zeros_like(observed); hidden.narrow(2,0,8).fill_(true);
  const auto query_two = point_two.reconstruct(batch,hidden);
  fixture([&] {
    auto cp = rpb::load_checkpoint(early2, settings.model.device); cp.model->eval();
    for (auto &p : cp.model->parameters()) p.set_requires_grad(false);
    torch::NoGradGuard no_grad;
    const rpb::Input raw_input{original.training_observations.data,original.training_observations.feature_mask,
        torch::tensor(original.channel_ids,torch::kInt64),torch::full({8},31,torch::kFloat64),1};
    const auto normalized = cp.scaler.transform(raw_input,cp.settings.model);
    const auto encoded = cp.model->encode(normalized);
    exact(features_two.values, rpb::compact_reconstruction_export(encoded,cp.settings.model).to(torch::kCPU), "exact CUDA checkpoint global32 export");
    const auto forward = cp.model->forward(normalized,hidden);
    exact(query_two.prediction,forward.reconstruction.to(torch::kCPU),"exact checkpoint ordinary-query reconstruction");
    exact(query_two.target,normalized.data.to(torch::kCPU),"frozen TRAIN-scaled targets");
    check(!torch::equal(cp.model->named_parameters()["global_pool_first.weight"],
        rpb::load_checkpoint(early0,settings.model.device).model->named_parameters()["global_pool_first.weight"]),"global pool receives actual training gradient");
  });
  auto poisoned = batch; poisoned.data = batch.data.clone(); poisoned.data.masked_fill_(hidden,1e12);
  exact(query_two.prediction,point_two.reconstruct(poisoned,hidden).prediction,"no hidden target bypass");
  auto absent = batch; absent.data = torch::full_like(batch.data,std::numeric_limits<double>::quiet_NaN()); absent.feature_mask = torch::zeros_like(observed);
  const auto absent_values = point_two.features.extract(absent).at("curve_global");
  check(!absent_values.valid.any().item<bool>() && absent_values.values.eq(0).all().item<bool>(),"all-absent support exact zero/no inference");
  const auto four = early.train_to(4), direct_four = direct.train_to(4); same_trace(four,direct_four);
  const auto early4 = (directory/"early4.pt").string(), direct4 = (directory/"direct4.pt").string();
  early.save_checkpoint(early4); direct.save_checkpoint(direct4);
  fixture([&] { same_model(rpb::load_checkpoint(early4,settings.model.device).model,rpb::load_checkpoint(direct4,settings.model.device).model,"split/direct parameters"); });
  same_optimizer(early4,direct4);
  exact(zero_values.values,zero.features.extract(batch).at("curve_global").values,"point0 immutable after later updates");
  exact(features_two.values,point_two.features.extract(batch).at("curve_global").values,"point2 immutable after later updates");
  exact(query_two.prediction,point_two.reconstruct(batch,hidden).prediction,"point2 CUDA query immutable after later updates");
  const auto reloaded = rpb::make_early_mixer_snapshot(options(early4,1,4,early),original);
  exact(early.snapshot(early4).features.extract(batch).at("curve_global").values,reloaded.features.extract(batch).at("curve_global").values,"public reloaded exact native snapshot");
  const auto assets = directory/"assets"; check(fs::create_directory(assets),"new assets"); reloaded.features.save_assets(assets.string());
  torch::serialize::InputArchive asset; asset.load_from((assets/rpb::kEarlyMixerSnapshotAuditFile).string(),torch::kCPU);
  check(text(asset,"artifact_kind") == rpb::kEarlyMixerSnapshotArtifact && text(asset,"model_tag") == "RPB-v10" &&
      text(asset,"architecture_id") == rpb::kEarlyMixerArchitectureId && text(asset,"no_optimizer_created") == "true" &&
      text(asset,"snapshot_loader_source_fingerprint") == EARLY_MIXER_ADAPTER_SOURCE_ID,"snapshot typed identity/scopes");
  torch::Tensor placement; asset.read("channel_mixer_placement_value",placement,true);
  check(placement.scalar_type() == torch::kInt64 && placement.dim() == 0 && placement.item<int64_t>() == 1,"typed placement witness");
  torch::serialize::InputArchive named; asset.read("model_parameters",named); check(named.keys().size() > 2,"named frozen CPU parameter witnesses");
  rejects([&] { reloaded.features.save_assets(assets.string()); },"immutable audit replacement");
  // New late training is numerically the historical ordinary-placement15 loop.
  const auto late_four = late.train_to(4); const auto late4 = (directory/"late4.pt").string(); late.save_checkpoint(late4);
  check(late_four.losses.size() == four.losses.size(),"paired per-attempt trace length");
  for (size_t i = 0; i < four.losses.size(); ++i)
    check(late_four.losses[i].attempted == four.losses[i].attempted &&
        late_four.losses[i].completed == four.losses[i].completed &&
        late_four.losses[i].target_cells == four.losses[i].target_cells,"paired original target/counter traces");
  torch::serialize::InputArchive late_audit, early_audit;
  late_audit.load_from(late4+".audit.pt",torch::kCPU); early_audit.load_from(early4+".audit.pt",torch::kCPU);
  for (const auto &key : {"context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates"}) {
    torch::Tensor left,right; late_audit.read(key,left,true); early_audit.read(key,right,true);
    exact(left,right,std::string("paired cumulative context counts/")+key);
  }
  const auto legacy4 = (directory/"legacy-late4.pt").string();
  fixture([&] {
    at::set_num_threads(1);
    auto legacy_fit = original; legacy_fit.protocol_id = "historical-late-learner-fixture-v1";
    auto legacy = rpb::make_learning_curve_trainer(settings,{true,rpb::ContextDeletionRecipe::coordinate15_v1})(legacy_fit);
    const auto progress = legacy.train_to(4); same_trace(late.train_to(4),progress); legacy.save_checkpoint(legacy4);
    same_model(rpb::load_checkpoint(late4,settings.model.device).model,rpb::load_checkpoint(legacy4,settings.model.device).model,"default late numeric contract");
  });
  same_optimizer(late4,legacy4);
  auto bad = options(early4,1,4,early); bad.expected_channel_mixer_placement = 0;
  rejects([&] { rpb::make_early_mixer_snapshot(bad,original); },"wrong placement binding");
  bad = options(early4,1,4,early); bad.expected_core_source_fingerprint = std::string(64,'0');
  rejects([&] { rpb::make_early_mixer_snapshot(bad,original); },"wrong source scope");
  bad = options(early4,1,4,early); bad.expected_completed_updates = 2;
  rejects([&] { rpb::make_early_mixer_snapshot(bad,original); },"wrong absolute budget");
  auto wrong_fit = original; wrong_fit.protocol_id = "early-mixer-reliability-v1/amplitude";
  rejects([&] { rpb::make_early_mixer_snapshot(options(early4,1,4,early),wrong_fit); },"amplitude fit forbidden");
  wrong_fit = original; wrong_fit.seed++;
  rejects([&] { rpb::make_early_mixer_snapshot(options(early4,1,4,early),wrong_fit); },"wrong training seed");
  wrong_fit = original; std::swap(wrong_fit.training_source_ids[0],wrong_fit.training_source_ids[2]);
  rejects([&] { rpb::make_early_mixer_snapshot(options(early4,1,4,early),wrong_fit); },"source order forgery");
  const auto forged = copy_parent(directory,early4,"forged-policy.pt");
  change_audit(forged+".audit.pt","context_deletion_ratio_value",torch::tensor(.30,torch::kFloat64));
  rejects([&] { rpb::make_early_mixer_snapshot(options(forged,1,4,early),original); },"typed policy forgery");
  fixture([&] {
    auto historical = rpb::make_learning_curve_trainer(early_settings,{true,rpb::ContextDeletionRecipe::coordinate15_v1})(original);
    const auto path = (directory/"historical-early0.pt").string(); historical.save_checkpoint(path);
    rejects([&] { historical.snapshot(path); },"historical CPU snapshot rejects early placement");
    rpb::EvaluationOptions evaluation; evaluation.checkpoint_path = early4;
    rejects([&] { rpb::make_evaluation_provider(evaluation)(original); },"legacy evaluator rejects early placement");
  });
  auto historical_fit = original; historical_fit.protocol_id = "fresh-decoder-replication-v1/lag_sign";
  rpb::FrozenNativeFeatureOptions historical_options{early4,rpb::FrozenDecoderParentPolicy::coordinate15_v7,4,
      early.audit_fields.at("core_writer_source_fingerprint"),early.audit_fields.at("training_producer_source_fingerprint")};
  rejects([&] { rpb::make_frozen_native_feature_provider(historical_options,historical_fit); },"old strict frozen parent rejects early placement");
  const auto previous_bytes = bytes(early2); (void)point_two.features.extract(batch); check(previous_bytes == bytes(early2),"immutable original checkpoint bytes");
  overall.unchanged("new CUDA adapter lifecycle");
  std::cout << "Generated artificial early mixer fixtures at " << directory << '\n';
}
} // namespace

int main() {
  try {
    run();
    std::cout << "EARLY_MIXER_ADAPTER_SOURCE_ID=" << EARLY_MIXER_ADAPTER_SOURCE_ID << '\n';
    std::cout << "Early mixer CUDA adapter admission passed\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
