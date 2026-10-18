// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/early_mixer_adapter.h"
#include "embedding/shared/data.h"
#include "embedding/shared/feature_harness.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

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
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  void unchanged(const std::string &label) const {
    check(at::get_num_threads() == threads, label + " threads");
    for (size_t i = 0; i < generators.size(); ++i)
      check(torch::equal(states[i], generators[i].get_state()), label + " RNG");
  }
};
// Test-owned isolation covers only historical checkpoint loading and CPU head
// construction, which have no promised ambient runtime contract. New curve
// callbacks remain unwrapped and are checked by the retained broad witness.
struct FixtureIsolation : RuntimeWitness {
  ~FixtureIsolation() {
    for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]);
    at::set_num_threads(threads);
  }
};

torch::Tensor read(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::Tensor tensor; archive.read(key, tensor, true); return tensor;
}
std::string text(torch::serialize::InputArchive &archive, const std::string &key) {
  return embedding::archive::tensor_text(read(archive, key));
}
void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &label) {
  check(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() &&
      torch::equal(a.to(torch::kCPU), b.to(torch::kCPU)), label);
}
std::string bytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary); check(bool(input), "read fixture bytes");
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
std::map<std::string, std::string> byte_witness(const std::string &path) {
  std::map<std::string, std::string> result;
  for (const auto &suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt", ".continuation.pt"})
    result.emplace(suffix, bytes(path + suffix));
  return result;
}
void same_flat(torch::serialize::InputArchive &a, torch::serialize::InputArchive &b,
               const std::string &label) {
  check(a.keys() == b.keys(), label + " keys");
  for (const auto &key : a.keys()) exact(read(a,key),read(b,key),label + '/' + key);
}
void same_named(torch::serialize::InputArchive &a, torch::serialize::InputArchive &b,
                const std::string &key) {
  torch::serialize::InputArchive left,right; a.read(key,left); b.read(key,right);
  const auto count = read(left,"count").item<int64_t>(); exact(read(left,"count"),read(right,"count"),key + " count");
  for (int64_t i = 0; i < count; ++i) {
    torch::serialize::InputArchive l,r; left.read("tensor_" + std::to_string(i),l); right.read("tensor_" + std::to_string(i),r);
    same_flat(l,r,key + '/' + std::to_string(i));
  }
}
void same_state(const std::string &left, const std::string &right) {
  torch::serialize::InputArchive a,b;
  a.load_from(left + rpb::kLearningCurveContinuationSuffix,torch::kCPU);
  b.load_from(right + rpb::kLearningCurveContinuationSuffix,torch::kCPU);
  for (const auto &key : {"attempted_steps","completed_steps","sampled_rows",
      "context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates",
      "loss_trace_counters","loss_trace_values"}) exact(read(a,key),read(b,key),std::string("continuous live state/") + key);
  same_named(a,b,"model_parameters"); same_named(a,b,"model_buffers");
  torch::serialize::InputArchive sa,sb; a.read("scaler",sa); b.read("scaler",sb); same_flat(sa,sb,"frozen scaler");
  torch::serialize::InputArchive oa,ob; a.read("optimizer_state",oa); b.read("optimizer_state",ob);
  exact(read(oa,"parameter_count"),read(ob,"parameter_count"),"AdamW named parameter count");
  exact(read(oa,"active_state_count"),read(ob,"active_state_count"),"AdamW named active set count");
  for (int64_t i = 0; i < read(oa,"parameter_count").item<int64_t>(); ++i) {
    torch::serialize::InputArchive l,r; oa.read("parameter_" + std::to_string(i),l); ob.read("parameter_" + std::to_string(i),r);
    same_flat(l,r,"exact name-aligned AdamW state/" + std::to_string(i));
  }
}
void same_trace(const ev::CurveProgress &a, const ev::CurveProgress &b, bool prefix = false) {
  check(prefix ? a.losses.size() <= b.losses.size() : a.losses.size() == b.losses.size(),"complete trace length");
  for (size_t i = 0; i < a.losses.size(); ++i) {
    const auto &x = a.losses[i], &y = b.losses[i];
    check(x.attempted == y.attempted && x.completed == y.completed && x.target_cells == y.target_cells &&
        x.loss == y.loss && x.gradient_norm == y.gradient_norm,"exact per-update trace/prefix");
  }
}
rpb::EarlyMixerSnapshotOptions options(const std::string &path, int64_t placement, int64_t updates,
                                       const ev::CurveTrainer &trainer) {
  return {path,placement,updates,rpb::EarlyMixerScope::curve_engineering,
      trainer.audit_fields.at("core_writer_source_fingerprint"),trainer.audit_fields.at("training_producer_source_fingerprint")};
}
void audit_live_vs_checkpoint(const std::string &path, const rpb::Config &config) {
  const FixtureIsolation fixture;
  auto checkpoint = rpb::load_checkpoint(path,config.device);
  torch::optim::AdamW optimizer(checkpoint.model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  rpb::load_optimizer(path,optimizer,config.device);
  torch::serialize::InputArchive state; state.load_from(path + rpb::kLearningCurveContinuationSuffix,torch::kCPU);
  torch::serialize::InputArchive named,moments; state.read("model_parameters",named); state.read("optimizer_state",moments);
  const auto parameters = checkpoint.model->named_parameters();
  for (int64_t i = 0; i < read(moments,"parameter_count").item<int64_t>(); ++i) {
    torch::serialize::InputArchive p,m; named.read("tensor_"+std::to_string(i),p); moments.read("parameter_"+std::to_string(i),m);
    const auto name = text(m,"parameter_name"); check(name == text(p,"parameter_name"),"lexical model/optimizer identity");
    exact(read(p,"value"),parameters[name],"live state equals ordinary CUDA checkpoint/"+name);
    const auto found = optimizer.state().find(parameters[name].unsafeGetTensorImpl());
    check(read(m,"has_state").item<bool>() == (found != optimizer.state().end()),"saved active AdamW parameter association");
    if (found == optimizer.state().end()) continue;
    const auto *value = dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
    check(value && value->step() == read(m,"step").item<int64_t>() && value->exp_avg().is_cuda() &&
        value->exp_avg_sq().is_cuda(),"exact CUDA optimizer continuation step/device");
    exact(read(m,"exp_avg"),value->exp_avg(),"live first moment equals saved ordinary optimizer");
    exact(read(m,"exp_avg_sq"),value->exp_avg_sq(),"live second moment equals saved ordinary optimizer");
  }
}
void change_flat(const std::string &path, const std::string &key, const torch::Tensor &replacement) {
  torch::serialize::InputArchive input; input.load_from(path,torch::kCPU); torch::serialize::OutputArchive output;
  for (const auto &name : input.keys()) output.write(name,name == key ? replacement : read(input,name),true);
  embedding::archive::save_archive(path,output);
}
std::string copy_parent(const fs::path &directory, const std::string &path, const std::string &name) {
  const auto result = (directory/name).string();
  for (const auto &suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt", ".continuation.pt"})
    check(fs::copy_file(path+suffix,result+suffix),"new negative copied fixture");
  return result;
}

void run() {
  check(torch::cuda::is_available(),"actual CUDA required; no CPU fallback/skip");
  const RuntimeWitness overall;
  auto settings = rpb::default_settings(); settings.model.device = torch::Device(torch::kCUDA);
  settings.model.global_bottleneck_mode = 2; settings.model.channel_mixer_layers = 1;
  settings.batch_size = 8; settings.threads = 1; settings.log_every = 1; settings.steps = 4; settings.attempt_limit = 8;
  auto raw = input(settings.model,8); raw.observed.select(2,3).select(2,2).fill_(false);
  raw.data.masked_fill_(raw.observed.logical_not(),0);
  ev::ProviderFitInput fit{{raw.data.clone(),raw.observed.clone()},{3,32,3,torch::kFloat64,torch::kCPU},303,
      {"a","a","b","b","c","c","d","d"},{0,1,2},"unitless,unitless,unitless",
      rpb::kEarlyMixerCurveFixtureFitProtocol,1,31};
  const embedding::Batch batch{raw.data.clone(),raw.observed.clone()};
  auto hidden = torch::zeros_like(raw.observed); hidden.narrow(2,0,8).fill_(true);
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-early-mixer-curve-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory),"exclusive artificial directory");
  std::array<rpb::EarlyMixerSnapshotOptions,2> initial_options;
  for (int64_t placement : {0,1}) {
    auto active = settings; active.model.channel_mixer_placement = placement;
    auto longer = active; longer.steps = 8; longer.attempt_limit = 16;
    const RuntimeWitness runtime;
    auto split = rpb::make_early_mixer_curve_trainer(active,true)(fit);
    auto direct = rpb::make_early_mixer_curve_trainer(active,true)(fit);
    auto horizon = rpb::make_early_mixer_curve_trainer(longer,true)(fit);
    const auto name = std::to_string(placement);
    const auto zero_path = (directory/ ("zero-"+name+".pt")).string();
    const auto zero_progress = split.train_to(0); split.save_checkpoint(zero_path);
    initial_options[static_cast<size_t>(placement)] = options(zero_path,placement,0,split);
    auto zero = split.snapshot(zero_path); const auto zero_features = zero.features.extract(batch).at("curve_global");
    const auto zero_query = zero.reconstruct(batch,hidden); const auto zero_bytes = byte_witness(zero_path);
    const auto one = split.train_to(1); const auto one_path = (directory/("one-"+name+".pt")).string();
    split.save_checkpoint(one_path); const auto one_snapshot = split.snapshot(one_path);
    check(one.parameter_count == 225805 && one.cuda_parameter_count == 225805 && one.attempted == 1 &&
        one.completed == 1 && one.last_input_cuda && one.last_loss_cuda && one.finite_gradients && one.weights_changed,
        "actual CUDA full-size first update");
    const auto two = split.train_to(2); same_trace(one,two,true);
    const auto two_path = (directory/("two-"+name+".pt")).string(); split.save_checkpoint(two_path);
    const auto two_bytes = byte_witness(two_path); const auto two_snapshot = split.snapshot(two_path);
    const auto two_features = two_snapshot.features.extract(batch).at("curve_global");
    const auto two_query = two_snapshot.reconstruct(batch,hidden);
    // Real fixed CPU heads intervene between live CUDA segments. The historical
    // probe constructor is test-isolated; no fresh encoder callback is wrapped.
    {
      const FixtureIsolation head_fixture; at::set_num_threads(1);
      const auto labels = torch::tensor({0,1,0,1,0,1,0,1},torch::kInt64);
      const ev::RidgeProbe ridge(two_features,labels,1);
      const ev::TinyProbe neural(two_features,labels,ev::stream_seed(2701,32),100,16,.01);
      check(ridge.predict(two_features).numel() == 8 && neural.predict(two_features).numel() == 8,
          "intervening CPU fitted readout work");
    }
    const auto four = split.train_to(4), direct_four = direct.train_to(4), horizon_four = horizon.train_to(4);
    same_trace(two,four,true); same_trace(four,direct_four); same_trace(four,horizon_four);
    check(four.attempted == 4 && four.completed == 4 && four.sampled_rows == 32 &&
        four.preprocessing_id == zero_progress.preprocessing_id && four.training_dataset_id == zero_progress.training_dataset_id,
        "continuous counters/scaler/source identity");
    const auto four_path = (directory/("four-"+name+".pt")).string();
    const auto direct_path = (directory/("direct-"+name+".pt")).string();
    const auto horizon_path = (directory/("horizon-"+name+".pt")).string();
    split.save_checkpoint(four_path); direct.save_checkpoint(direct_path); horizon.save_checkpoint(horizon_path);
    same_state(four_path,direct_path); same_state(four_path,horizon_path); audit_live_vs_checkpoint(four_path,active.model);
    check(zero_bytes == byte_witness(zero_path) && two_bytes == byte_witness(two_path),"earlier five-file saved bytes unchanged");
    exact(zero_features.values,zero.features.extract(batch).at("curve_global").values,"immutable point0 native export");
    exact(zero_query.prediction,zero.reconstruct(batch,hidden).prediction,"immutable point0 CUDA query");
    exact(two_features.values,two_snapshot.features.extract(batch).at("curve_global").values,"immutable point2 native export");
    exact(two_query.prediction,two_snapshot.reconstruct(batch,hidden).prediction,"immutable point2 CUDA query");
    const auto reloaded = rpb::make_early_mixer_snapshot(options(two_path,placement,2,split),fit);
    exact(two_features.values,reloaded.features.extract(batch).at("curve_global").values,"exact independently reloaded CUDA native32");
    exact(two_query.prediction,reloaded.reconstruct(batch,hidden).prediction,"exact independently reloaded CUDA reconstruction");
    auto poisoned = batch; poisoned.data = batch.data.clone(); poisoned.data.masked_fill_(hidden,1e12);
    exact(two_query.prediction,reloaded.reconstruct(poisoned,hidden).prediction,"hidden-query values do not bypass native32");
    const auto assets = directory/("assets-"+name); check(fs::create_directory(assets),"new curve snapshot assets");
    reloaded.features.save_assets(assets.string());
    torch::serialize::InputArchive audit; audit.load_from((assets/rpb::kEarlyMixerSnapshotAuditFile).string(),torch::kCPU);
    check(text(audit,"artifact_kind") == rpb::kEarlyMixerCurveSnapshotArtifact &&
        text(audit,"protocol_id") == rpb::kEarlyMixerCurveProtocol &&
        text(audit,"original_training_protocol_id") == rpb::kEarlyMixerCurveFixtureFitProtocol &&
        text(audit,"continuation_state_suffix") == rpb::kLearningCurveContinuationSuffix &&
        !text(audit,"parent_content_id.continuation.pt").empty(),"new curve protocol/state-bound snapshot identity");
    // New ceilings and state capture do not change the untouched fixed4 recipe.
    auto historical_fit = fit; historical_fit.protocol_id = rpb::kEarlyMixerFixtureFitProtocol;
    auto historical = rpb::make_early_mixer_trainer(active,rpb::EarlyMixerScope::engineering)(historical_fit);
    const auto historical_four = historical.train_to(4); same_trace(four,historical_four);
    const auto historical_path = (directory/("historical-"+name+".pt")).string(); historical.save_checkpoint(historical_path);
    check(!fs::exists(historical_path+rpb::kLearningCurveContinuationSuffix),"oldscope companion set unchanged");
    {
      const FixtureIsolation cp_fixture;
      const auto a = rpb::load_checkpoint(four_path,active.model.device), b = rpb::load_checkpoint(historical_path,active.model.device);
      const auto expected = b.model->named_parameters();
      for (const auto &entry : a.model->named_parameters()) exact(entry.value(),expected[entry.key()],"oldfixed4 numeric parameter parity");
      check(a.scaler.identity() == b.scaler.identity(),"oldfixed4 scaler parity");
    }
    auto wrong = options(two_path,placement,2,split); wrong.scope = rpb::EarlyMixerScope::engineering;
    rejects([&]{rpb::make_early_mixer_snapshot(wrong,fit);},"historicalscope rejects curve namespace/state");
    wrong = options(two_path,placement,2,split); wrong.expected_completed_updates = 4;
    rejects([&]{rpb::make_early_mixer_snapshot(wrong,fit);},"wrong absolute point counters");
    wrong = options(two_path,placement,2,split); wrong.expected_channel_mixer_placement = 1-placement;
    rejects([&]{rpb::make_early_mixer_snapshot(wrong,fit);},"wrong placement");
    wrong = options(two_path,placement,2,split); wrong.expected_training_producer_source_fingerprint = std::string(64,'0');
    rejects([&]{rpb::make_early_mixer_snapshot(wrong,fit);},"wrong source producer");
    auto wrong_fit = fit; wrong_fit.seed++;
    rejects([&]{rpb::make_early_mixer_snapshot(options(two_path,placement,2,split),wrong_fit);},"wrong original training seed");
    const auto forged = copy_parent(directory,two_path,"forged-"+name+".pt");
    change_flat(forged+".audit.pt","completed_steps",torch::tensor(int64_t(4)));
    rejects([&]{rpb::make_early_mixer_snapshot(options(forged,placement,2,split),fit);},"forged saved counter binding");
    rejects([&]{split.train_to(2);},"decreasing completed budget");
    rejects([&]{split.train_to(4);},"failed trainer permanently blocked");
    runtime.unchanged("new curve callbacks plus test-isolated CPU/checkpoint fixtures");
  }
  const auto pairing = rpb::audit_early_mixer_initialization(initial_options[0],initial_options[1],fit);
  for (const auto &key : {"common_parameters_exact","common_buffers_exact","scaler_exact","training_dataset_exact","counter_streams_exact"})
    check(pairing.at(key) == "true",std::string("paired curve point0/") + key);
  const RuntimeWitness negative;
  auto bad_settings = settings; bad_settings.steps = 512;
  rejects([&]{rpb::make_early_mixer_curve_trainer(bad_settings);},"curvequality requires2048 ceiling");
  auto quality = settings; quality.steps = 2048; quality.attempt_limit = 4096;
  auto quality_factory = rpb::make_early_mixer_curve_trainer(quality);
  rejects([&]{quality_factory(fit);},"curvequality rejects fixture namespace");
  auto wrong_fit = fit; wrong_fit.protocol_id = rpb::kEarlyMixerFitProtocol;
  rejects([&]{rpb::make_early_mixer_curve_trainer(settings,true)(wrong_fit);},"curveengineering rejects oldquality namespace");
  auto sparse = fit; sparse.training_observations.data = batch.data.clone(); sparse.training_observations.feature_mask = torch::zeros_like(batch.feature_mask);
  sparse.training_observations.feature_mask.narrow(2,0,8).fill_(true);
  sparse.training_observations.data.masked_fill_(sparse.training_observations.feature_mask.logical_not(),0);
  auto skip = rpb::make_early_mixer_curve_trainer(settings,true)(sparse);
  const auto skip_zero = (directory/"skip-zero.pt").string(); skip.save_checkpoint(skip_zero);
  const auto skip_before = byte_witness(skip_zero);
  rejects([&]{skip.train_to(2);},"curve aborts skipped original masking attempt");
  check(skip_before == byte_witness(skip_zero),"rejected skipped attempt leaves original point0 bytes unchanged");
  rejects([&]{skip.save_checkpoint((directory/"skip.pt").string());},"failed skip cannot save");
  auto short_settings = settings; short_settings.attempt_limit = 1;
  auto exhausted = rpb::make_early_mixer_curve_trainer(short_settings,true)(fit);
  rejects([&]{exhausted.train_to(2);},"exhausted cumulative attempt cap");
  rejects([&]{exhausted.train_to(1);},"exhausted trainer cannot resume");
  negative.unchanged("malformed/skip/exhausted declarations and callbacks");
  overall.unchanged("full new curve engineering lifecycle");
  std::cout << "Generated artificial curve fixtures at " << directory << '\n';
}
} // namespace

int main() {
  try {
    run();
    std::cout << "EARLY_MIXER_ADAPTER_SOURCE_ID=" << EARLY_MIXER_ADAPTER_SOURCE_ID << '\n';
    std::cout << "Early mixer curve CUDA admission passed\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
