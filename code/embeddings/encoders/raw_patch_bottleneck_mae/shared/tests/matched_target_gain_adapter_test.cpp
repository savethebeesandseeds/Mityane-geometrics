// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/early_mixer_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/matched_target_gain_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
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
#include <limits>

#ifndef MATCHED_TARGET_GAIN_ADAPTER_SOURCE_ID
#define MATCHED_TARGET_GAIN_ADAPTER_SOURCE_ID "unrecorded"
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
  for (const auto &suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt", ".continuation.pt", ".gain-view.pt"})
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
  for (const auto &suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt", ".continuation.pt", ".gain-view.pt"})
    check(fs::copy_file(path+suffix,result+suffix),"new negative copied fixture");
  return result;
}

rpb::MatchedTargetGainSnapshotOptions options(const std::string &path,rpb::MatchedTargetGainPolicy policy,int64_t updates,
    const ev::CurveTrainer &trainer) {
  return {path,policy,rpb::MatchedTargetGainScope::engineering,updates,
      trainer.audit_fields.at("core_writer_source_fingerprint"),trainer.audit_fields.at("training_producer_source_fingerprint")};
}
void same_gain_view(const std::string &left,const std::string &right) {
  torch::serialize::InputArchive a,b; a.load_from(left+rpb::kMatchedTargetGainViewSuffix,torch::kCPU);
  b.load_from(right+rpb::kMatchedTargetGainViewSuffix,torch::kCPU);
  for (const auto &key:{"source_ranks","row_source_ranks","source_top53","source_uniforms","source_gains","row_gains",
      "attempt_indices","sampled_row_indices","sampled_gains","actual_normalized_targets"}) exact(read(a,key),read(b,key),std::string("gain continuous witness/")+key);
}
void primitives(const rpb::Input &raw,const std::vector<std::string> &ids) {
  namespace gain=rpb::training_source_gain;
  const RuntimeWitness runtime;
  const auto m=gain::make_manifest(ids,303,{rpb::TrainingSourceGainRecipe::source_log2_v1});
  check(m.source_ids == std::vector<std::string>({"a","b","c","d"}),"lexical source ranks");
  for (int64_t i=0;i<8;i+=2) check(m.row_gains[i].item<double>() == m.row_gains[i+1].item<double>(),"source pair gain without labels");
  const auto order=torch::tensor({6,7,2,3,0,1,4,5},torch::kInt64);
  std::vector<std::string> reordered;
  for (int64_t i=0;i<8;++i) reordered.push_back(ids[order[i].item<int64_t>()]);
  const auto permuted=gain::make_manifest(reordered,303,{rpb::TrainingSourceGainRecipe::source_log2_v1});
  exact(m.source_gains,permuted.source_gains,"row permutation preserves source law");
  exact(m.row_gains.index_select(0,order),permuted.row_gains,"row gain association permutes correctly");
  const auto indices=torch::arange(8,torch::kInt64);
  auto poisoned=raw; poisoned.data=raw.data.clone(); poisoned.data.masked_fill_(raw.observed.logical_not(),std::numeric_limits<double>::quiet_NaN());
  const auto gained=gain::apply(poisoned,indices,m);
  check(torch::isfinite(gained.data).all().item<bool>() && (gained.data.masked_select(raw.observed.logical_not())==0).all().item<bool>(),"hidden NaN never reaches gain and absent storage stays zero");
  exact(gained.observed,raw.observed,"gain preserves natural O");
  rejects([&]{gain::validate_options({static_cast<rpb::TrainingSourceGainRecipe>(99)});},"invalid gain enum");
  rejects([&]{gain::make_manifest({"a","a","a","b"},303,{rpb::TrainingSourceGainRecipe::source_log2_v1});},"wrong source multiplicity");
  rejects([&]{gain::make_manifest(ids,uint64_t{1}<<63,{rpb::TrainingSourceGainRecipe::source_log2_v1});},"unrepresentable gain seed");
  for (int malformed=0;malformed<4;++malformed) {
    auto bad=m;
    if (malformed==0) bad.source_gains=m.source_gains.to(torch::kCUDA);
    if (malformed==1) bad.source_gains=torch::stack({m.source_gains,m.source_gains},1).select(1,0);
    if (malformed==2) bad.source_gains=m.source_gains.to(torch::kFloat32);
    if (malformed==3) bad.source_gains=m.source_gains.reshape({2,2});
    rejects([&]{gain::apply(raw,indices,bad);},"malformed manifest storage before raw-byte hashing");
  }
  runtime.unchanged("source gain primitive no RNG/thread consumption");
}
void run() {
  check(torch::cuda::is_available(),"actual CUDA mandatory");
  const RuntimeWitness overall;
  auto settings=rpb::default_settings(); settings.model.device=torch::Device(torch::kCUDA);
  settings.model.global_bottleneck_mode=2; settings.model.channel_mixer_layers=1; settings.model.channel_mixer_placement=1;
  settings.model.dropout=0; settings.steps=4; settings.attempt_limit=8; settings.batch_size=8; settings.log_every=1; settings.threads=1;
  auto raw=input(settings.model,8); raw.data=raw.data+3;
  raw.observed.select(2,3).select(2,2).fill_(false); raw.data.masked_fill_(raw.observed.logical_not(),0);
  const std::vector<std::string> ids{"a","a","b","b","c","c","d","d"}; primitives(raw,ids);
  ev::ProviderFitInput fit{{raw.data.clone(),raw.observed.clone()},{3,32,3,torch::kFloat64,torch::kCPU},303,ids,
      {0,1,2},"unitless,unitless,unitless",rpb::kMatchedTargetGainFixtureFitProtocol,1,31};
  const embedding::Batch batch{raw.data.clone(),raw.observed.clone()};
  auto hidden=torch::zeros_like(raw.observed); hidden.narrow(2,0,8).fill_(true);
  const auto directory=fs::path(std::getenv("TMPDIR")?std::getenv("TMPDIR"):"/tmp") /
      ("rpb-matched-target-gain-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory),"exclusive test artifacts");
  std::array<std::string,2> zero_paths;
  std::array<ev::FeatureSurface,2> initial_features;
  std::array<ev::CurveReconstruction,2> initial_queries;
  std::array<std::string,2> four_paths;
  std::array<ev::CurveProgress,2> progresses;
  for (size_t k=0;k<2;++k) {
    const auto policy=k?rpb::MatchedTargetGainPolicy::coordinate15_gain:rpb::MatchedTargetGainPolicy::coordinate15_control;
    const RuntimeWitness lifecycle;
    auto split=rpb::make_matched_target_gain_trainer(settings,policy,rpb::MatchedTargetGainScope::engineering)(fit);
    auto direct=rpb::make_matched_target_gain_trainer(settings,policy,rpb::MatchedTargetGainScope::engineering)(fit);
    split.train_to(0); const auto zero=(directory/("zero-"+std::to_string(k)+".pt")).string(); zero_paths[k]=zero;
    split.save_checkpoint(zero); const auto zero_snapshot=split.snapshot(zero); const auto initial_bytes=byte_witness(zero);
    initial_features[k]=zero_snapshot.features.extract(batch).at("curve_global"); initial_queries[k]=zero_snapshot.reconstruct(batch,hidden);
    const auto one=split.train_to(1); const auto one_path=(directory/("one-"+std::to_string(k)+".pt")).string(); split.save_checkpoint(one_path);
    const auto two=split.train_to(2); same_trace(one,two,true);
    const auto two_path=(directory/("two-"+std::to_string(k)+".pt")).string(); split.save_checkpoint(two_path);
    const auto two_snapshot=split.snapshot(two_path); const auto two_native=two_snapshot.features.extract(batch).at("curve_global");
    const auto two_query=two_snapshot.reconstruct(batch,hidden); const auto earlier_bytes=byte_witness(two_path);
    const auto four=split.train_to(4),direct_four=direct.train_to(4); same_trace(two,four,true); same_trace(four,direct_four);
    check(four.completed==4 && four.attempted==4 && four.sampled_rows==32 && four.parameter_count==225805 &&
        four.cuda_parameter_count==225805 && four.last_input_cuda && four.last_loss_cuda && four.finite_gradients && four.weights_changed,
        "actual full-sized CUDA loss/grad/update evidence"); progresses[k]=four;
    const auto four_path=(directory/("four-"+std::to_string(k)+".pt")).string(); four_paths[k]=four_path;
    const auto direct_path=(directory/("direct-"+std::to_string(k)+".pt")).string(); split.save_checkpoint(four_path); direct.save_checkpoint(direct_path);
    same_state(four_path,direct_path); same_gain_view(four_path,direct_path); audit_live_vs_checkpoint(four_path,settings.model);
    check(initial_bytes==byte_witness(zero) && earlier_bytes==byte_witness(two_path),"all six earlier companion bytes remain immutable");
    exact(initial_features[k].values,zero_snapshot.features.extract(batch).at("curve_global").values,"point0 serving immutable after4");
    exact(initial_queries[k].prediction,zero_snapshot.reconstruct(batch,hidden).prediction,"point0 original-Q immutable");
    const auto reloaded=rpb::make_matched_target_gain_snapshot(options(two_path,policy,2,split),fit);
    exact(two_native.values,reloaded.features.extract(batch).at("curve_global").values,"independent exact frozen CUDA native parity");
    exact(two_query.prediction,reloaded.reconstruct(batch,hidden).prediction,"independent exact original-Q decoder parity");
    auto poison=batch; poison.data=batch.data.clone(); poison.data.masked_fill_(hidden,1e12);
    exact(two_query.prediction,reloaded.reconstruct(poison,hidden).prediction,"no hidden target bypass");
    const embedding::Batch absent{torch::full_like(batch.data,std::numeric_limits<double>::quiet_NaN()),torch::zeros_like(batch.feature_mask)};
    const auto missing=reloaded.features.extract(absent).at("curve_global"); check(!missing.valid.any().item<bool>() && (missing.values==0).all().item<bool>(),"no inferred all-absent support");
    auto wrong=fit; wrong.seed++; rejects([&]{rpb::make_matched_target_gain_snapshot(options(two_path,policy,2,split),wrong);},"wrong original seed");
    wrong=fit; std::swap(wrong.training_source_ids[0],wrong.training_source_ids[2]);
    rejects([&]{rpb::make_matched_target_gain_snapshot(options(two_path,policy,2,split),wrong);},"wrong original source order");
    auto wrong_options=options(two_path,policy,2,split); wrong_options.policy=k?rpb::MatchedTargetGainPolicy::coordinate15_control:rpb::MatchedTargetGainPolicy::coordinate15_gain;
    rejects([&]{rpb::make_matched_target_gain_snapshot(wrong_options,fit);},"wrong explicit policy");
    wrong_options=options(two_path,policy,2,split); wrong_options.expected_training_producer_source_fingerprint=std::string(64,'0');
    rejects([&]{rpb::make_matched_target_gain_snapshot(wrong_options,fit);},"wrong producer pin");
    rejects([&]{split.save_checkpoint(two_path);},"no overwrite");
    rejects([&]{split.train_to(2);},"decreasing budget"); rejects([&]{split.train_to(4);},"failed actor permanently blocks");
    lifecycle.unchanged("new matched-view API lifecycle");
  }
  const auto initialization=rpb::audit_matched_target_gain_initialization(zero_paths[0],zero_paths[1],fit,rpb::MatchedTargetGainScope::engineering);
  check(initialization.at("common_parameters_exact")=="true" && initialization.at("scaler_exact")=="true","full same-early point0 state/scaler gate");
  exact(initial_features[0].values,initial_features[1].values,"same original CUDA point0 native values");
  exact(initial_features[0].valid,initial_features[1].valid,"same point0 support");
  exact(initial_queries[0].prediction,initial_queries[1].prediction,"same point0 original query before gain");
  {
    const FixtureIsolation fixture;
    const auto control=rpb::load_checkpoint(four_paths[0],settings.model.device),candidate=rpb::load_checkpoint(four_paths[1],settings.model.device);
    check(control.training_policy_id=="rpb-training-context-deletion-015-v1" && candidate.training_policy_id==rpb::kMatchedTargetGainPolicy &&
        control.scaler.identity()==candidate.scaler.identity() && control.dataset_id==candidate.dataset_id,"original scaler/data and distinct composed policies");
    const auto m=rpb::training_source_gain::make_manifest(ids,303,{rpb::TrainingSourceGainRecipe::source_log2_v1});
    torch::serialize::InputArchive view; view.load_from(four_paths[1]+rpb::kMatchedTargetGainViewSuffix,torch::kCPU);
    const auto indices=read(view,"sampled_row_indices"),targets=read(view,"actual_normalized_targets");
    for (int64_t a=0;a<4;++a) {
      const auto selected=rpb::training_detail::selected(raw,indices[a]);
      const auto gain_raw=rpb::training_source_gain::apply(selected,indices[a],m);
      const auto normalized=candidate.scaler.transform(gain_raw,settings.model);
      exact(targets[a],normalized.data,"actual CUDA normalized gained target affine replay");
      const auto ordinary=candidate.scaler.transform(selected,settings.model);
      check(!torch::equal(normalized.data,ordinary.data*m.row_gains.index_select(0,indices[a]).to(torch::kCUDA,torch::kFloat32).reshape({8,1,1,1})),
          "gain before nonzero-mean centering, not centered-value multiplication");
      const auto mask=rpb::make_training_mask(normalized.observed,settings.model,rpb::training_detail::counter_seed(303,a,0x6d61736bULL));
      const auto original_mask=rpb::make_training_mask(ordinary.observed,settings.model,rpb::training_detail::counter_seed(303,a,0x6d61736bULL));
      exact(mask.hidden,original_mask.hidden,"original A exact under gain"); exact(mask.target,original_mask.target,"original Q exact under gain");
    }
  }
  // The new unit-control evidence scope must not alter any old .15 mathematics.
  auto historical_fit=fit; historical_fit.protocol_id=rpb::kEarlyMixerFixtureFitProtocol;
  auto historical=rpb::make_early_mixer_trainer(settings,rpb::EarlyMixerScope::engineering)(historical_fit);
  const auto historical_four=historical.train_to(4); same_trace(progresses[0],historical_four);
  const auto old_path=(directory/"historical.pt").string(); historical.save_checkpoint(old_path);
  check(!fs::exists(old_path+rpb::kMatchedTargetGainViewSuffix) && !fs::exists(old_path+rpb::kLearningCurveContinuationSuffix),"old artifact matrix unchanged");
  {
    const FixtureIsolation fixture; const auto old=rpb::load_checkpoint(old_path,settings.model.device),control=rpb::load_checkpoint(four_paths[0],settings.model.device);
    const auto named=old.model->named_parameters(); for (const auto &p:control.model->named_parameters()) exact(p.value(),named[p.key()],"old default015 exact parameter parity");
  }
  rejects([&]{rpb::make_early_mixer_trainer(settings,rpb::EarlyMixerScope::engineering)(fit);},"historical factory rejects new namespace");
  rejects([&]{rpb::make_learning_curve_trainer(settings,{true,rpb::ContextDeletionRecipe::coordinate15_v1},{true},
      {rpb::TrainingSourceGainRecipe::disabled})(fit);},"disabled cannot admit new protocol");
  rejects([&]{rpb::make_matched_target_gain_trainer(settings,static_cast<rpb::MatchedTargetGainPolicy>(99),rpb::MatchedTargetGainScope::engineering);},"malformed policy before RNG");
  {
    const FixtureIsolation fixture;
    std::vector<std::string> arguments{"rpb-mae","train","--resume",four_paths[1],"--device","cuda","--checkpoint",(directory/"forbidden-resume.pt").string()};
    std::vector<char*> argv; for (auto &argument:arguments) argv.push_back(argument.data());
    bool rejected=false;
    try { rpb::run_cli(static_cast<int>(argv.size()),argv.data()); }
    catch (const std::exception &e) { rejected=std::string(e.what()).find("ordinary train/resume rejects early mixer placement")!=std::string::npos; }
    check(rejected && !fs::exists(directory/"forbidden-resume.pt"),"ordinary CLI rejects unfamiliar early tagged training before output");
  }
  {
    rpb::EarlyMixerSnapshotOptions old_options{four_paths[1],1,4,rpb::EarlyMixerScope::engineering,
        historical.audit_fields.at("core_writer_source_fingerprint"),historical.audit_fields.at("training_producer_source_fingerprint")};
    rejects([&]{rpb::make_early_mixer_snapshot(old_options,historical_fit);},"historical CUDA snapshot rejects combined gain policy");
  }
  auto sparse=fit; sparse.training_observations.data=batch.data.clone(); sparse.training_observations.feature_mask=torch::zeros_like(batch.feature_mask);
  sparse.training_observations.feature_mask.narrow(2,0,8).fill_(true); sparse.training_observations.data.masked_fill_(sparse.training_observations.feature_mask.logical_not(),0);
  auto skipped=rpb::make_matched_target_gain_trainer(settings,rpb::MatchedTargetGainPolicy::coordinate15_gain,rpb::MatchedTargetGainScope::engineering)(sparse);
  rejects([&]{skipped.train_to(1);},"new gain scope aborts ineligible original query");
  rejects([&]{skipped.save_checkpoint((directory/"skipped.pt").string());},"failed gain scope cannot save");
  auto bad=settings; bad.attempt_limit=1;
  rejects([&]{rpb::make_matched_target_gain_trainer(bad,rpb::MatchedTargetGainPolicy::coordinate15_gain,rpb::MatchedTargetGainScope::engineering);},"closed attempt budget prevents exhausted prefix");
  for (size_t i=0;i<progresses[0].losses.size();++i) check(progresses[0].losses[i].target_cells==progresses[1].losses[i].target_cells,"gain never changes original Q/loss denominator");
  torch::serialize::InputArchive ca,ga; ca.load_from(four_paths[0]+".audit.pt",torch::kCPU); ga.load_from(four_paths[1]+".audit.pt",torch::kCPU);
  for (const auto &key:{"context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates"}) exact(read(ca,key),read(ga,key),"gain preserves absolute E/support streams");
  overall.unchanged("new factory/gain/CUDA snapshots preserve ambient runtime");
}
} // namespace
int main() {
  try { run(); std::cout << "Matched target gain CUDA admission passed\n" << MATCHED_TARGET_GAIN_ADAPTER_SOURCE_ID << '\n'; return 0; }
  catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
