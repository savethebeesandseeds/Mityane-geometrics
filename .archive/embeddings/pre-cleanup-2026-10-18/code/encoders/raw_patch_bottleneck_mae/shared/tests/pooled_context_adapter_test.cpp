// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/pooled_context_adapter.h"
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

#ifndef POOLED_CONTEXT_ADAPTER_SOURCE_ID
#define POOLED_CONTEXT_ADAPTER_SOURCE_ID "unrecorded"
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


rpb::PooledContextSnapshotOptions options(const std::string &path,rpb::PooledContextRole role,int64_t budget,
    const ev::CurveTrainer &trainer) {
  return {path,role,budget,rpb::PooledContextScope::engineering,trainer.audit_fields.at("core_writer_source_fingerprint"),
    trainer.audit_fields.at("training_producer_source_fingerprint")};
}
void run() {
  check(torch::cuda::is_available(),"actual CUDA required; no fallback or skipped gate");
  const RuntimeWitness overall;
  auto settings=rpb::default_settings(); settings.model.device=torch::Device(torch::kCUDA);
  settings.model.global_bottleneck_mode=2; settings.model.channel_mixer_layers=1; settings.model.channel_mixer_placement=1;
  settings.steps=4; settings.attempt_limit=8; settings.batch_size=8; settings.log_every=1; settings.threads=1;
  auto raw=input(settings.model,8); raw.observed.select(2,3).select(2,2).fill_(false); raw.data.masked_fill_(raw.observed.logical_not(),0);
  ev::ProviderFitInput fit{{raw.data.clone(),raw.observed.clone()},{3,32,3,torch::kFloat64,torch::kCPU},719,
    {"a","a","b","b","c","c","d","d"},{0,1,2},"unitless,unitless,unitless",rpb::kPooledContextFixtureFitProtocol,1,31};
  const embedding::Batch batch{raw.data.clone(),raw.observed.clone()};
  auto hidden=torch::zeros_like(raw.observed); hidden.narrow(2,0,8).fill_(true);
  const auto directory=fs::path(std::getenv("TMPDIR")?std::getenv("TMPDIR"):"/tmp")/
    ("rpb-pooled-context-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory),"exclusive artificial directory");
  auto reference=rpb::make_pooled_context_trainer(settings,rpb::PooledContextRole::compact_control,"",rpb::PooledContextScope::engineering)(fit);
  const auto reference_path=(directory/"reference-zero.pt").string(); reference.train_to(0); reference.save_checkpoint(reference_path);
  const auto reference_bytes=byte_witness(reference_path);
  std::string candidate_zero;
  for (auto role : {rpb::PooledContextRole::compact_control,rpb::PooledContextRole::pooled_width_candidate}) {
    const bool candidate=role==rpb::PooledContextRole::pooled_width_candidate;
    auto active=settings; active.model.global_pool_input_source=candidate ? 1 : 0;
    const auto name=candidate ? "pooled" : "compact"; const auto ref=candidate ? reference_path : std::string();
    const RuntimeWitness callbacks;
    auto split=rpb::make_pooled_context_trainer(active,role,ref,rpb::PooledContextScope::engineering)(fit);
    auto direct=rpb::make_pooled_context_trainer(active,role,ref,rpb::PooledContextScope::engineering)(fit);
    const auto zero_path=(directory/(std::string(name)+"-zero.pt")).string();
    const auto zero_progress=split.train_to(0); split.save_checkpoint(zero_path); if(candidate) candidate_zero=zero_path;
    const auto zero=split.snapshot(zero_path); const auto zero_z=zero.features.extract(batch).at("curve_global");
    const auto zero_q=zero.reconstruct(batch,hidden); const auto zero_bytes=byte_witness(zero_path);
    const auto one=split.train_to(1);
    check(one.attempted==1 && one.completed==1 && one.parameter_count==(candidate?231949:225805) &&
      one.cuda_parameter_count==one.parameter_count && one.last_input_cuda && one.last_loss_cuda && one.finite_gradients && one.weights_changed,
      "actual CUDA full-size input/loss/gradients/update");
    const auto one_path=(directory/(std::string(name)+"-one.pt")).string(); split.save_checkpoint(one_path);
    const auto two=split.train_to(2); same_trace(one,two,true);
    const auto two_path=(directory/(std::string(name)+"-two.pt")).string(); split.save_checkpoint(two_path);
    const auto two_snapshot=split.snapshot(two_path); const auto two_z=two_snapshot.features.extract(batch).at("curve_global");
    const auto two_q=two_snapshot.reconstruct(batch,hidden); const auto two_bytes=byte_witness(two_path);
    {
      const FixtureIsolation head_fixture; at::set_num_threads(1);
      const auto labels=torch::tensor({0,1,0,1,0,1,0,1},torch::kInt64);
      const ev::RidgeProbe head(two_z,labels,1); check(head.predict(two_z).numel()==8,"CPU head work between live segments");
    }
    const auto four=split.train_to(4),direct_four=direct.train_to(4); same_trace(two,four,true); same_trace(four,direct_four);
    check(four.attempted==4 && four.completed==4 && four.sampled_rows==32 &&
      four.preprocessing_id==zero_progress.preprocessing_id && four.training_dataset_id==zero_progress.training_dataset_id,
      "continuous absolute counters/scaler/data");
    const auto four_path=(directory/(std::string(name)+"-four.pt")).string(),direct_path=(directory/(std::string(name)+"-direct.pt")).string();
    split.save_checkpoint(four_path); direct.save_checkpoint(direct_path); same_state(four_path,direct_path);
    const auto four_bytes=byte_witness(four_path);
    rejects([&]{split.save_checkpoint(four_path);},"existing checkpoint companions cannot be overwritten");
    check(four_bytes==byte_witness(four_path),"rejected overwrite preserves every saved byte");
    audit_live_vs_checkpoint(four_path,active.model);
    check(zero_bytes==byte_witness(zero_path) && two_bytes==byte_witness(two_path) && reference_bytes==byte_witness(reference_path),
      "all earlier five-file points and copy reference immutable");
    exact(zero_z.values,zero.features.extract(batch).at("curve_global").values,"point0 CUDA native immutable");
    exact(zero_q.prediction,zero.reconstruct(batch,hidden).prediction,"point0 CUDA query immutable");
    exact(two_z.values,two_snapshot.features.extract(batch).at("curve_global").values,"point2 CUDA native immutable");
    exact(two_q.prediction,two_snapshot.reconstruct(batch,hidden).prediction,"point2 CUDA query immutable");
    const auto reload=rpb::make_pooled_context_snapshot(options(two_path,role,2,split),fit);
    exact(two_z.values,reload.features.extract(batch).at("curve_global").values,"independent CUDA native reload exact");
    exact(two_q.prediction,reload.reconstruct(batch,hidden).prediction,"independent CUDA query reload exact");
    auto poison=batch; poison.data=batch.data.clone(); poison.data.masked_fill_(hidden,1e12);
    exact(two_q.prediction,reload.reconstruct(poison,hidden).prediction,"hidden originalQ storage excluded");
    auto absent=batch; absent.feature_mask=torch::zeros_like(batch.feature_mask); absent.data=torch::zeros_like(batch.data);
    const auto invalid=reload.features.extract(absent).at("curve_global");
    check(!invalid.valid.any().item<bool>() && !invalid.values.ne(0).any().item<bool>(),"all absent zero/invalid");
    const auto assets=directory/(std::string(name)+"-assets"); check(fs::create_directory(assets),"new snapshot assets");
    reload.features.save_assets(assets.string()); torch::serialize::InputArchive a;
    a.load_from((assets/rpb::kPooledContextSnapshotAuditFile).string(),torch::kCPU);
    check(text(a,"artifact_kind")==rpb::kPooledContextSnapshotArtifact && text(a,"protocol_id")==rpb::kPooledContextProtocol &&
      read(a,"global_pool_input_source_value").item<int64_t>()==(candidate?1:0),"distinct pooled snapshot identity");
    if(candidate) {
      torch::serialize::InputArchive state; state.load_from(four_path+rpb::kLearningCurveContinuationSuffix,torch::kCPU);
      check(text(state,"artifact_kind")==rpb::kPooledContextContinuationArtifact && read(state,"copied_parameter_values").item<int64_t>()==219469 &&
        read(state,"inactive_projection_values").item<int64_t>()==2080,"typed copy/inactive witness");
      torch::serialize::InputArchive optimizer; state.read("optimizer_state",optimizer);
      for(int64_t i=0;i<read(optimizer,"parameter_count").item<int64_t>();++i) {
        torch::serialize::InputArchive p; optimizer.read("parameter_"+std::to_string(i),p); const auto n=text(p,"parameter_name");
        if(n=="export_projection.weight" || n=="export_projection.bias") check(!read(p,"has_state").item<bool>(),"inactive projection has no fabricated Adam state");
      }
    } else {
      auto oldfit=fit; oldfit.protocol_id=rpb::kEarlyMixerFixtureFitProtocol;
      auto old=rpb::make_early_mixer_trainer(active,rpb::EarlyMixerScope::engineering)(oldfit);
      same_trace(four,old.train_to(4)); const auto oldpath=(directory/"old-compact.pt").string(); old.save_checkpoint(oldpath);
      const FixtureIsolation fixture; const auto x=rpb::load_checkpoint(four_path,active.model.device),y=rpb::load_checkpoint(oldpath,active.model.device);
      const auto yp=y.model->named_parameters(); for(const auto &p:x.model->named_parameters()) exact(p.value(),yp[p.key()],"source0 old .15 parameter parity");
    }
    auto wrong=options(two_path,role,2,split); wrong.expected_completed_updates=4;
    rejects([&]{rpb::make_pooled_context_snapshot(wrong,fit);},"wrong saved budget");
    wrong=options(two_path,role,2,split); wrong.role=candidate?rpb::PooledContextRole::compact_control:rpb::PooledContextRole::pooled_width_candidate;
    rejects([&]{rpb::make_pooled_context_snapshot(wrong,fit);},"wrong declared input source");
    wrong=options(two_path,role,2,split); wrong.expected_training_producer_source_fingerprint=std::string(64,'0');
    rejects([&]{rpb::make_pooled_context_snapshot(wrong,fit);},"wrong producer pin");
    rejects([&]{split.train_to(2);},"decreasing budget"); rejects([&]{split.train_to(4);},"failed trainer poisoned");
    callbacks.unchanged("all new pooled callbacks plus test-isolated historical loading/head fixture");
  }
  const auto gate=rpb::audit_pooled_context_initialization(reference_path,candidate_zero,fit,rpb::PooledContextScope::engineering);
  check(gate.at("common_parameters_exact")=="true" && gate.at("shared_parameter_values")=="219469" &&
    gate.at("copied_before_AdamW")=="true" && gate.at("initial_features_equality_required")=="false","exact shared initialization without false initial-output parity");
  auto candidate=settings; candidate.model.global_pool_input_source=1;
  const RuntimeWitness negative;
  const auto positive_reference=(directory/"compact-four.pt").string();
  rejects([&]{rpb::make_pooled_context_trainer(candidate,rpb::PooledContextRole::pooled_width_candidate,positive_reference,rpb::PooledContextScope::engineering)(fit);},
    "positive trained initialization reference rejected");
  rejects([&]{rpb::make_learning_curve_trainer(candidate,rpb::ContextDeletionOptions{true,rpb::ContextDeletionRecipe::coordinate15_v1});},"old learner rejects source1");
  rejects([&]{rpb::make_pooled_context_trainer(candidate,rpb::PooledContextRole::pooled_width_candidate,"",rpb::PooledContextScope::engineering);},"missing compact initialization reference");
  auto wrongfit=fit; wrongfit.seed++;
  rejects([&]{rpb::make_pooled_context_trainer(candidate,rpb::PooledContextRole::pooled_width_candidate,reference_path,rpb::PooledContextScope::engineering)(wrongfit);},"wrong initialization seed");
  wrongfit=fit; wrongfit.training_source_ids[0]="wrong";
  rejects([&]{rpb::make_pooled_context_trainer(candidate,rpb::PooledContextRole::pooled_width_candidate,reference_path,rpb::PooledContextScope::engineering)(wrongfit);},"wrong source order");
  wrongfit=fit; wrongfit.training_observations.data=fit.training_observations.data.clone(); wrongfit.training_observations.data[0][0][0][0]+=1;
  rejects([&]{rpb::make_pooled_context_trainer(candidate,rpb::PooledContextRole::pooled_width_candidate,reference_path,rpb::PooledContextScope::engineering)(wrongfit);},"wrong original data/scaler");
  wrongfit=fit; wrongfit.protocol_id=rpb::kEarlyMixerFixtureFitProtocol;
  rejects([&]{rpb::make_pooled_context_trainer(candidate,rpb::PooledContextRole::pooled_width_candidate,reference_path,rpb::PooledContextScope::engineering)(wrongfit);},"wrong namespace");
  auto quality=settings; quality.steps=512; quality.attempt_limit=1024;
  rejects([&]{rpb::make_pooled_context_trainer(quality,rpb::PooledContextRole::compact_control)(fit);},"quality rejects engineering namespace");
  const auto altered=copy_parent(directory,reference_path,"altered-reference.pt");
  change_flat(altered+".audit.pt","completed_steps",torch::tensor(int64_t(1)));
  rejects([&]{rpb::make_pooled_context_trainer(candidate,rpb::PooledContextRole::pooled_width_candidate,altered,rpb::PooledContextScope::engineering)(fit);},"altered copy reference");
  auto sparse=fit; sparse.training_observations.feature_mask=torch::zeros_like(raw.observed);
  sparse.training_observations.feature_mask.narrow(2,0,8).fill_(true); sparse.training_observations.data=raw.data.clone();
  sparse.training_observations.data.masked_fill_(sparse.training_observations.feature_mask.logical_not(),0);
  auto skipped=rpb::make_pooled_context_trainer(settings,rpb::PooledContextRole::compact_control,"",rpb::PooledContextScope::engineering)(sparse);
  rejects([&]{skipped.train_to(2);},"ineligible original query aborts"); rejects([&]{skipped.save_checkpoint((directory/"skip.pt").string());},"aborted skip cannot save");
  auto short_settings=settings; short_settings.attempt_limit=1;
  auto exhausted=rpb::make_pooled_context_trainer(short_settings,rpb::PooledContextRole::compact_control,"",rpb::PooledContextScope::engineering)(fit);
  rejects([&]{exhausted.train_to(2);},"exhausted attempt cap");
  negative.unchanged("rejected pooled declarations restore ambient state"); overall.unchanged("entire new pooled lifecycle");
}
} // namespace
int main() { try { run(); std::cout << "Pooled context adapter CUDA admission passed\n" << POOLED_CONTEXT_ADAPTER_SOURCE_ID << '\n'; return 0; }
  catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; } }
