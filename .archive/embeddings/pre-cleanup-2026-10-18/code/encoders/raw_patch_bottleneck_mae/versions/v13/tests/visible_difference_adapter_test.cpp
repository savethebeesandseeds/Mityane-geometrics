// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/visible_difference_adapter.h"
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

#ifndef VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID
#define VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID "unrecorded"
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
  for (const auto &suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt", ".continuation.pt", ".visible-difference.pt"})
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
  for (const auto &suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt", ".continuation.pt", ".visible-difference.pt"})
    check(fs::copy_file(path+suffix,result+suffix),"new negative copied fixture");
  return result;
}


rpb::VisibleDifferenceSnapshotOptions options(const std::string &path,int64_t budget,const ev::CurveTrainer &trainer) {
  return {path,budget,rpb::VisibleDifferenceScope::engineering,trainer.audit_fields.at("core_writer_source_fingerprint"),
    trainer.audit_fields.at("training_producer_source_fingerprint")};
}
void run() {
  check(torch::cuda::is_available(),"actual CUDA required; no fallback"); const RuntimeWitness overall;
  auto settings=rpb::default_settings(); settings.model.device=torch::kCUDA;
  settings.model.global_bottleneck_mode=2; settings.model.channel_mixer_layers=1; settings.model.channel_mixer_placement=1;
  settings.steps=4; settings.attempt_limit=8; settings.batch_size=8; settings.log_every=1; settings.threads=1;
  auto raw=input(settings.model,8); raw.observed.select(2,3).select(2,2).fill_(false); raw.data.masked_fill_(raw.observed.logical_not(),0);
  ev::ProviderFitInput original{{raw.data.clone(),raw.observed.clone()},{3,32,3,torch::kFloat64,torch::kCPU},719,
    {"a","a","b","b","c","c","d","d"},{0,1,2},"unitless,unitless,unitless",rpb::kEarlyMixerFixtureFitProtocol,1,31};
  const embedding::Batch batch{raw.data.clone(),raw.observed.clone()};
  auto hidden=torch::zeros_like(raw.observed); hidden.narrow(2,0,8).fill_(true);
  const auto directory=fs::temp_directory_path()/("rpb-visible-difference-adapter-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory),"exclusive preserved artificial leaf");
  auto parent=rpb::make_early_mixer_trainer(settings,rpb::EarlyMixerScope::engineering)(original);
  parent.train_to(0); const auto parent_path=(directory/"original-zero.pt").string(); parent.save_checkpoint(parent_path);
  std::map<std::string,std::string> original_bytes;
  for(const auto &suffix:{"",".audit.pt",".scaler.pt",".training-raw.pt"}) original_bytes.emplace(suffix,bytes(parent_path+suffix));
  auto candidate=settings; candidate.model.temporal_difference_input=1;
  const rpb::VisibleDifferenceOptions binding{true,parent_path,parent.audit_fields.at("core_writer_source_fingerprint"),
    parent.audit_fields.at("training_producer_source_fingerprint"),original.protocol_id};
  auto fit=original; fit.protocol_id=rpb::kVisibleDifferenceFixtureFitProtocol;
  auto split=rpb::make_visible_difference_trainer(candidate,binding,rpb::VisibleDifferenceScope::engineering)(fit);
  auto direct=rpb::make_visible_difference_trainer(candidate,binding,rpb::VisibleDifferenceScope::engineering)(fit);
  check(split.audit_fields.at("model_tag")=="RPB-v13" && split.audit_fields.at("visible_difference_copied_parameter_values")=="0",
    "new tag and no copied initialization");
  const auto zero_path=(directory/"candidate-zero.pt").string(); split.train_to(0); split.save_checkpoint(zero_path);
  const auto zero=split.snapshot(zero_path),old_zero=parent.snapshot(parent_path);
  const auto zero_z=zero.features.extract(batch).at("curve_global"),old_z=old_zero.features.extract(batch).at("curve_global");
  exact(zero_z.values,old_z.values,"initial native32 equals original early0"); exact(zero_z.valid,old_z.valid,"initial support equals original");
  const auto zero_q=zero.reconstruct(batch,hidden); exact(zero_q.prediction,old_zero.reconstruct(batch,hidden).prediction,"initial original-query equality");
  const auto zero_bytes=byte_witness(zero_path); const auto two=split.train_to(2);
  check(two.completed==2 && two.attempted==2 && two.sampled_rows==16 && two.parameter_count==228877 && two.cuda_parameter_count==228877 &&
    two.last_input_cuda && two.last_loss_cuda && two.finite_gradients && two.weights_changed,"actual CUDA candidate updates and counters");
  const auto two_path=(directory/"candidate-two.pt").string(); split.save_checkpoint(two_path);
  const auto two_snapshot=split.snapshot(two_path);
  const auto two_z=two_snapshot.features.extract(batch).at("curve_global");
  const auto two_q=two_snapshot.reconstruct(batch,hidden);
  const auto two_bytes=byte_witness(two_path);
  { const FixtureIsolation artificial; torch::rand({29}); torch::rand({19},torch::TensorOptions().device(torch::kCUDA)); }
  const auto four=split.train_to(4),direct_four=direct.train_to(4); same_trace(two,four,true); same_trace(four,direct_four);
  const auto four_path=(directory/"candidate-four.pt").string(),direct_path=(directory/"candidate-direct.pt").string();
  split.save_checkpoint(four_path); direct.save_checkpoint(direct_path); same_state(four_path,direct_path); audit_live_vs_checkpoint(four_path,candidate.model);
  exact(zero_z.values,zero.features.extract(batch).at("curve_global").values,"saved0 remains exact after live4");
  exact(zero_q.prediction,zero.reconstruct(batch,hidden).prediction,"saved0 originalQ remains exact");
  exact(two_z.values,two_snapshot.features.extract(batch).at("curve_global").values,"saved2 remains exact after live4");
  exact(two_q.prediction,two_snapshot.reconstruct(batch,hidden).prediction,"saved2 originalQ remains exact");
  check(zero_bytes==byte_witness(zero_path) && two_bytes==byte_witness(two_path),"six-role earlier points preserved");
  for(const auto &[suffix,value]:original_bytes) check(bytes(parent_path+suffix)==value,"original parent four bytes preserved");
  const auto reload=rpb::make_visible_difference_snapshot(options(two_path,2,split),fit);
  exact(two_z.values,reload.features.extract(batch).at("curve_global").values,"immutable independent CUDA reload");
  auto poison=batch; poison.data=batch.data.clone(); poison.data.masked_fill_(hidden,1e12);
  exact(two_q.prediction,reload.reconstruct(poison,hidden).prediction,"hidden query storage cannot enter difference branch");
  auto absent=batch; absent.feature_mask=torch::zeros_like(batch.feature_mask); absent.data=torch::zeros_like(batch.data);
  const auto invalid=reload.features.extract(absent).at("curve_global");
  check(!invalid.valid.any().item<bool>() && !invalid.values.ne(0).any().item<bool>(),"all absent zero invalid");
  const auto assets=directory/"assets"; check(fs::create_directory(assets),"exclusive assets"); reload.features.save_assets(assets.string());
  torch::serialize::InputArchive saved; saved.load_from((assets/rpb::kVisibleDifferenceSnapshotAuditFile).string(),torch::kCPU);
  check(text(saved,"artifact_kind")==rpb::kVisibleDifferenceSnapshotArtifact && text(saved,"protocol_id")==rpb::kVisibleDifferenceProtocol &&
    text(saved,"original_training_protocol_id")==fit.protocol_id &&
    read(saved,"temporal_difference_input_value").item<int64_t>()==1,"new CUDA snapshot typed identity");
  torch::serialize::InputArchive continuation; continuation.load_from(four_path+rpb::kLearningCurveContinuationSuffix,torch::kCPU);
  check(text(continuation,"artifact_kind")==rpb::kVisibleDifferenceContinuationArtifact && read(continuation,"copied_parameter_values").item<int64_t>()==0,
    "new live witness no concealed copy");
  torch::serialize::InputArchive optimizer; continuation.read("optimizer_state",optimizer); bool new_active=false;
  for(int64_t i=0;i<read(optimizer,"parameter_count").item<int64_t>();++i) {
    torch::serialize::InputArchive entry; optimizer.read("parameter_"+std::to_string(i),entry);
    if(text(entry,"parameter_name")=="visible_difference_projection.weight") new_active=read(entry,"has_state").item<bool>() && read(entry,"step").item<int64_t>()==4;
  }
  check(new_active,"new difference branch participates in real AdamW state");
  auto wrong=options(two_path,2,split); wrong.expected_completed_updates=4;
  rejects([&]{rpb::make_visible_difference_snapshot(wrong,fit);},"wrong budget");
  wrong=options(two_path,2,split); wrong.expected_training_producer_source_fingerprint=std::string(64,'0');
  rejects([&]{rpb::make_visible_difference_snapshot(wrong,fit);},"wrong new producer");
  auto altered_fit=fit; altered_fit.training_observations.data=fit.training_observations.data.clone(); altered_fit.training_observations.data[0][0][0][0]+=1;
  rejects([&]{rpb::make_visible_difference_snapshot(options(two_path,2,split),altered_fit);},"snapshot altered original TRAIN");
  rejects([&]{rpb::make_learning_curve_trainer(candidate);},"old disabled factory rejects flag1");
  auto wrong_binding=binding; wrong_binding.expected_parent_core_source_fingerprint=std::string(64,'0');
  rejects([&]{rpb::make_visible_difference_trainer(candidate,wrong_binding,rpb::VisibleDifferenceScope::engineering)(fit);},"wrong original core pin");
  altered_fit=fit; altered_fit.seed++;
  rejects([&]{rpb::make_visible_difference_trainer(candidate,binding,rpb::VisibleDifferenceScope::engineering)(altered_fit);},"wrong original seed");
  altered_fit=fit; altered_fit.training_source_ids[0]="wrong";
  rejects([&]{rpb::make_visible_difference_trainer(candidate,binding,rpb::VisibleDifferenceScope::engineering)(altered_fit);},"wrong original source order");
  altered_fit=fit; altered_fit.protocol_id=rpb::kEarlyMixerFixtureFitProtocol;
  rejects([&]{rpb::make_visible_difference_trainer(candidate,binding,rpb::VisibleDifferenceScope::engineering)(altered_fit);},"wrong external namespace");
  auto bad_parent=(directory/"bad-original.pt").string();
  for(const auto &suffix:{"",".audit.pt",".scaler.pt",".training-raw.pt"}) check(fs::copy_file(parent_path+suffix,bad_parent+suffix),"copy negative original roles");
  change_flat(bad_parent+".audit.pt","initialization_seed",embedding::archive::text_tensor("wrong")); wrong_binding=binding; wrong_binding.parent_point0_checkpoint_path=bad_parent;
  rejects([&]{rpb::make_visible_difference_trainer(candidate,wrong_binding,rpb::VisibleDifferenceScope::engineering)(fit);},"forged initialization metadata");
  const auto corrupt=copy_parent(directory,two_path,"corrupt.pt"); change_flat(corrupt+rpb::kVisibleDifferenceBindingSuffix,"temporal_difference_input_value",torch::tensor(int64_t(0)));
  rejects([&]{rpb::make_visible_difference_snapshot(options(corrupt,2,split),fit);},"forged branch binding before weight load");
  const auto four_bytes=byte_witness(four_path); rejects([&]{split.save_checkpoint(four_path);},"reject saved output replacement");
  check(four_bytes==byte_witness(four_path),"rejected output replacement preserves bytes");
  rejects([&]{split.train_to(2);},"decreasing absolute budget"); rejects([&]{split.train_to(4);},"failed trainer poisoned");
  auto sparse_original=original;
  sparse_original.training_observations.feature_mask=torch::zeros_like(raw.observed);
  sparse_original.training_observations.feature_mask.narrow(2,0,8).fill_(true);
  sparse_original.training_observations.data=raw.data.clone();
  sparse_original.training_observations.data.masked_fill_(sparse_original.training_observations.feature_mask.logical_not(),0);
  auto sparse_parent=rpb::make_early_mixer_trainer(settings,rpb::EarlyMixerScope::engineering)(sparse_original);
  const auto sparse_path=(directory/"sparse-original-zero.pt").string(); sparse_parent.train_to(0); sparse_parent.save_checkpoint(sparse_path);
  auto sparse_binding=binding; sparse_binding.parent_point0_checkpoint_path=sparse_path;
  auto sparse_fit=sparse_original; sparse_fit.protocol_id=rpb::kVisibleDifferenceFixtureFitProtocol;
  auto skipped=rpb::make_visible_difference_trainer(candidate,sparse_binding,rpb::VisibleDifferenceScope::engineering)(sparse_fit);
  rejects([&]{skipped.train_to(2);},"ineligible original query aborts before update");
  rejects([&]{skipped.save_checkpoint((directory/"skipped.pt").string());},"skipped trainer cannot save");
  check(!fs::exists(directory/"skipped.pt"),"failed prefix creates no point artifact");
  overall.unchanged("new live state/snapshots/negative paths plus isolated ordinary loader");
  std::cout<<"Preserved artificial CUDA adapter fixtures: "<<directory.string()<<'\n';
}
} // namespace
int main() try { run(); std::cout<<"Visible difference adapter CUDA admission passed\n"<<VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID<<'\n'; return 0; }
catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
