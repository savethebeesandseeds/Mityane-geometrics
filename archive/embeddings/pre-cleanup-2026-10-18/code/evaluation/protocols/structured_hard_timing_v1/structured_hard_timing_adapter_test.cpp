// SPDX-License-Identifier: MIT
#include "structured_hard_timing_adapter.h"
#include "analytic_support.h"
#include "embedding/shared/data.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#ifndef STRUCTURED_HARD_TIMING_ADAPTER_SOURCE_ID
#define STRUCTURED_HARD_TIMING_ADAPTER_SOURCE_ID "unrecorded"
#endif
namespace {
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
namespace ev=embedding::evaluation;
namespace fs=std::filesystem;
using namespace rpb_test;
struct RuntimeWitness {
  int threads{at::get_num_threads()}; std::vector<at::Generator> generators; std::vector<torch::Tensor> states;
  RuntimeWitness() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for(size_t i=0;i<at::getNumGPUs();++i)generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for(const auto &g:generators)states.push_back(g.get_state().clone());
  }
  void unchanged(const std::string &label) const {
    check(threads==at::get_num_threads(),label+" threads");
    for(size_t i=0;i<generators.size();++i)check(torch::equal(states[i],generators[i].get_state()),label+" RNG");
  }
};
struct FixtureIsolation:RuntimeWitness {
  ~FixtureIsolation(){for(size_t i=0;i<generators.size();++i)generators[i].set_state(states[i]);at::set_num_threads(threads);}
};
void exact(const torch::Tensor &a,const torch::Tensor &b,const std::string &why) {
  check(a.scalar_type()==b.scalar_type()&&a.sizes()==b.sizes()&&torch::equal(a.to(torch::kCPU),b.to(torch::kCPU)),why);
}
std::string bytes(const std::string &path) {
  std::ifstream in(path,std::ios::binary);check(bool(in),"fixture bytes readable");std::ostringstream value;value<<in.rdbuf();return value.str();
}
std::string text(torch::serialize::InputArchive &a,const std::string &key) {
  torch::Tensor value;a.read(key,value,true);return embedding::archive::tensor_text(value);
}
void replace(const std::string &path,const std::string &key,const torch::Tensor &replacement) {
  torch::serialize::InputArchive in;in.load_from(path,torch::kCPU);torch::serialize::OutputArchive out;
  for(const auto &name:in.keys()){torch::Tensor value;in.read(name,value,true);out.write(name,name==key?replacement:value,true);}
  embedding::archive::save_archive(path,out);
}
std::string copy_point(const std::string &parent,const fs::path &directory,const std::string &name,bool structured_timing=true) {
  const auto path=(directory/name).string();
  for(const auto &suffix:{std::string(),std::string(".audit.pt"),std::string(".scaler.pt"),std::string(".training-raw.pt"),std::string(rpb::kStructuredHardTimingSuffix)})
    if(structured_timing || suffix!=rpb::kStructuredHardTimingSuffix) check(fs::copy_file(parent+suffix,path+suffix),"exclusive negative fixture copy");
  return path;
}
void same_trace(const ev::CurveProgress &a,const ev::CurveProgress &b) {
  check(a.attempted==b.attempted&&a.completed==b.completed&&a.sampled_rows==b.sampled_rows&&a.losses.size()==b.losses.size(),"exact counters/trace length");
  for(size_t i=0;i<a.losses.size();++i){const auto &x=a.losses[i],&y=b.losses[i];check(x.attempted==y.attempted&&x.completed==y.completed&&
      x.target_cells==y.target_cells&&x.loss==y.loss&&x.gradient_norm==y.gradient_norm,"exact per-attempt targets/loss/gradient norm");}
}
void same_checkpoint(const std::string &left,const std::string &right) {
  // Historical fixture checkpoint construction/loading is test-isolated. New
  // callbacks below are unwrapped and retain their own ambient-runtime witness.
  const FixtureIsolation isolation;
  auto a=rpb::load_checkpoint(left,torch::Device(torch::kCUDA,0)),b=rpb::load_checkpoint(right,torch::Device(torch::kCUDA,0));
  const auto ap=a.model->named_parameters(),bp=b.model->named_parameters();std::vector<std::string> an,bn;
  for(const auto &p:ap){check(p.value().is_cuda(),"actual CUDA checkpoint parameters");an.push_back(p.key());exact(p.value(),bp[p.key()],"exact parameter "+p.key());}
  for(const auto &p:bp)bn.push_back(p.key());
  check(an==bn,"registration order exact");
  const auto ab=a.model->named_buffers(),bb=b.model->named_buffers();
  for(const auto &p:ab)exact(p.value(),bb[p.key()],"exact buffer "+p.key());
  check(a.scaler.identity()==b.scaler.identity()&&a.dataset_id==b.dataset_id&&a.schema_id==b.schema_id&&
      a.attempted_steps==b.attempted_steps&&a.completed_steps==b.completed_steps,"exact scaler/dataset/counter identities");
  exact(a.scaler.mean,b.scaler.mean,"scaler mean");exact(a.scaler.scale,b.scaler.scale,"scaler scale");exact(a.scaler.count,b.scaler.count,"scaler count");
  torch::optim::AdamW oa(a.model->parameters(),torch::optim::AdamWOptions(a.settings.learning_rate).weight_decay(a.settings.weight_decay));
  torch::optim::AdamW ob(b.model->parameters(),torch::optim::AdamWOptions(b.settings.learning_rate).weight_decay(b.settings.weight_decay));
  rpb::load_optimizer(left,oa,torch::Device(torch::kCUDA,0));rpb::load_optimizer(right,ob,torch::Device(torch::kCUDA,0));
  check(oa.state().size()==ob.state().size(),"named AdamW state count");const auto pa=a.model->parameters(),pb=b.model->parameters();
  for(size_t i=0;i<pa.size();++i){const auto x=oa.state().find(pa[i].unsafeGetTensorImpl()),y=ob.state().find(pb[i].unsafeGetTensorImpl());
    check((x==oa.state().end())==(y==ob.state().end()),"same active AdamW set");if(x==oa.state().end())continue;
    const auto *sx=dynamic_cast<const torch::optim::AdamWParamState *>(x->second.get());const auto *sy=dynamic_cast<const torch::optim::AdamWParamState *>(y->second.get());
    check(sx&&sy&&sx->step()==sy->step()&&sx->exp_avg().is_cuda()&&sy->exp_avg_sq().is_cuda(),"CUDA AdamW steps and device");
    exact(sx->exp_avg(),sy->exp_avg(),"exact first moments");exact(sx->exp_avg_sq(),sy->exp_avg_sq(),"exact second moments");
  }
}
rpb::StructuredHardTimingSnapshotOptions options(const std::string &path,int64_t placement,int64_t updates,const ev::CurveTrainer &trainer) {
  return {path,placement,updates,rpb::StructuredHardTimingScope::engineering,
      trainer.audit_fields.at("core_writer_source_fingerprint"),trainer.audit_fields.at("training_producer_source_fingerprint")};
}
void run() {
  check(torch::cuda::is_available(),"actual CUDA mandatory; no skipped test");const RuntimeWitness overall;
  const auto analytic_valid=torch::tensor(std::vector<int64_t>{1,1,0,0},torch::kInt64).to(torch::kBool);
  const auto analytic_margin=torch::tensor(std::vector<double>{1,-1,1,0},torch::kFloat64);
  exact(ev::structured_timing::supported_analytic_predictions(analytic_valid,analytic_margin),
      torch::tensor(std::vector<int64_t>{1,0,0,0},torch::kInt64),"positive-margin insufficient-center row remains invalid zero");
  auto settings=rpb::default_settings();settings.model.channel_mixer_layers=1;settings.model.global_bottleneck_mode=2;settings.model.global_pool_input_source=0;
  settings.model.device=torch::Device(torch::kCUDA,0);settings.steps=4;settings.attempt_limit=8;settings.batch_size=8;settings.threads=1;settings.log_every=1;
  const auto raw=input(settings.model,8);auto observed=raw.observed.clone();observed.select(2,3).select(2,2).fill_(false);
  const auto legal=raw.data.masked_fill(observed.logical_not(),0);const embedding::Batch batch{legal.clone(),observed.clone()};
  const std::string prefix=std::string(rpb::kStructuredHardTimingFixtureFitProtocol)+"/";
  ev::ProviderFitInput fit{batch,{3,32,3,torch::kFloat64,torch::kCPU},202,
      {prefix+"a",prefix+"a",prefix+"b",prefix+"b",prefix+"c",prefix+"c",prefix+"d",prefix+"d"},
      {0,1,2},"unitless,unitless,unitless",rpb::kStructuredHardTimingFixtureFitProtocol,1,31};
  const auto directory=fs::path(std::getenv("TMPDIR")?std::getenv("TMPDIR"):"/tmp")/
      ("rpb-structured-hard-timing-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory),"exclusive artificial directory");
  rejects([&]{rpb::make_structured_hard_timing_trainer(settings,static_cast<rpb::StructuredHardTimingScope>(99));},"malformed scope");
  rejects([&]{rpb::make_structured_hard_timing_trainer(settings);},"fixture cannot enter quality scope");
  std::array<ev::CurveTrainer,2> segmented;std::array<std::string,2> zero_paths;
  for(int64_t placement=0;placement<2;++placement){
    settings.model.channel_mixer_placement=placement;
    const auto declaration=rpb::make_structured_hard_timing_trainer(settings,rpb::StructuredHardTimingScope::engineering);
    auto wrong=fit;wrong.protocol_id=rpb::kEarlyMixerFixtureFitProtocol;rejects([&]{declaration(wrong);},"internal namespace is not external structured_timing");
    wrong=fit;wrong.protocol_id="structured-hard-timing-comparison-engineering-v1/lag_sign";
    rejects([&]{declaration(wrong);},"preserved stopped v1 namespace cannot enter v2 binding");
    wrong=fit;wrong.training_source_ids[0]="historical-source";rejects([&]{declaration(wrong);},"old source outside external namespace");
    segmented[placement]=declaration(fit);auto direct=declaration(fit);
    auto legacy_fit=fit;legacy_fit.protocol_id=rpb::kEarlyMixerFixtureFitProtocol;
    auto legacy=rpb::make_early_mixer_trainer(settings,rpb::EarlyMixerScope::engineering)(legacy_fit);
    check(segmented[placement].audit_fields.at("protocol_id")==rpb::kEarlyMixerFixtureFitProtocol &&
        segmented[placement].audit_fields.at("external_fit_protocol_id")==rpb::kStructuredHardTimingFixtureFitProtocol,
        "truthful distinct external/implementation scopes");
    zero_paths[placement]=(directory/("zero-"+std::to_string(placement)+".pt")).string();segmented[placement].save_checkpoint(zero_paths[placement]);
    const auto zero_bytes=bytes(zero_paths[placement]);auto zero=segmented[placement].snapshot(zero_paths[placement]);
    const auto z0=zero.features.extract(batch).at("curve_global");check(z0.values.scalar_type()==torch::kFloat32&&z0.values.device().is_cpu()&&z0.valid.all().item<bool>(),"CUDA native32 CPU evidence");
    const auto p2=segmented[placement].train_to(2);check(p2.cuda_parameter_count==225805&&p2.last_input_cuda&&p2.last_loss_cuda&&p2.finite_gradients&&p2.weights_changed,"real CUDA parameters/input/loss/gradients/updates");
    const auto two=(directory/("two-"+std::to_string(placement)+".pt")).string();segmented[placement].save_checkpoint(two);auto snap=segmented[placement].snapshot(two);
    const auto before=snap.features.extract(batch).at("curve_global");auto hidden=torch::zeros_like(observed);hidden.narrow(2,0,8).fill_(true);
    const auto query=snap.reconstruct(batch,hidden);auto changed=batch;changed.data=batch.data.clone();changed.data.masked_fill_(hidden.logical_and(observed),123456);
    exact(snap.reconstruct(changed,hidden).prediction,query.prediction,"Q values excluded from decoder signal");
    auto poisoned=batch;poisoned.data=batch.data.clone();poisoned.data.masked_fill_(observed.logical_not(),std::numeric_limits<double>::quiet_NaN());
    exact(snap.features.extract(poisoned).at("curve_global").values,before.values,"absent storage isolation");
    const auto absent=snap.features.extract({torch::zeros_like(legal),torch::zeros_like(observed)}).at("curve_global");
    check(!absent.valid.any().item<bool>()&&absent.values.eq(0).all().item<bool>(),"all-absent native support remains invalid zero");
    const auto asset=directory/("assets-"+std::to_string(placement));check(fs::create_directory(asset),"new snapshot asset directory");snap.features.save_assets(asset.string());
    torch::serialize::InputArchive saved;saved.load_from((asset/rpb::kStructuredHardTimingSnapshotFile).string(),torch::kCPU);
    check(text(saved,"artifact_kind")==rpb::kStructuredHardTimingSnapshotArtifact &&
        text(saved,"information_rule_id")==rpb::kStructuredHardTimingInformationRule &&
        text(saved,"structured_timing_protocol_id")=="structured-hard-timing-comparison-v2" &&
        text(saved,"training_implementation_fit_protocol_id")==rpb::kEarlyMixerFixtureFitProtocol,
        "new snapshot artifact retains actual delegate scope");
    const auto p4=segmented[placement].train_to(4),d4=direct.train_to(4),l4=legacy.train_to(4);same_trace(p4,d4);same_trace(p4,l4);
    const auto final=(directory/("final-"+std::to_string(placement)+".pt")).string(),other=(directory/("direct-"+std::to_string(placement)+".pt")).string(),old=(directory/("legacy-"+std::to_string(placement)+".pt")).string();
    segmented[placement].save_checkpoint(final);direct.save_checkpoint(other);legacy.save_checkpoint(old);same_checkpoint(final,other);same_checkpoint(final,old);
    exact(snap.features.extract(batch).at("curve_global").values,before.values,"saved point2 immutable after continuation");
    exact(zero.features.extract(batch).at("curve_global").values,z0.values,"point0 immutable after continuation");check(bytes(zero_paths[placement])==zero_bytes,"earlier checkpoint bytes unchanged");
    auto o=options(final,placement,4,segmented[placement]);const auto reloaded=rpb::make_structured_hard_timing_snapshot(o,fit);
    exact(reloaded.features.extract(batch).at("curve_global").values,segmented[placement].snapshot(final).features.extract(batch).at("curve_global").values,"independent reload parity");
    wrong=fit;wrong.seed+=1;rejects([&]{rpb::make_structured_hard_timing_snapshot(o,wrong);},"wrong master");
    wrong=fit;std::swap(wrong.training_source_ids[0],wrong.training_source_ids[2]);rejects([&]{rpb::make_structured_hard_timing_snapshot(o,wrong);},"wrong source order");
    wrong=fit;wrong.training_observations.data=fit.training_observations.data.clone();wrong.training_observations.data.index_put_({0,0,0,0},999);rejects([&]{rpb::make_structured_hard_timing_snapshot(o,wrong);},"wrong TRAIN data");
    auto bad=o;bad.expected_channel_mixer_placement=1-placement;rejects([&]{rpb::make_structured_hard_timing_snapshot(bad,fit);},"wrong role");
    bad=o;bad.expected_completed_updates=2;rejects([&]{rpb::make_structured_hard_timing_snapshot(bad,fit);},"wrong counters");
    bad=o;bad.expected_core_source_fingerprint=std::string(64,'f');rejects([&]{rpb::make_structured_hard_timing_snapshot(bad,fit);},"wrong core source");
    const auto corrupted=copy_point(final,directory,"corrupt-"+std::to_string(placement)+".pt");bad=o;bad.checkpoint_path=corrupted;
    replace(corrupted+rpb::kStructuredHardTimingSuffix,"external_fit_protocol_id",embedding::archive::text_tensor("wrong"));
    rejects([&]{rpb::make_structured_hard_timing_snapshot(bad,fit);},"altered fifth companion");
    const auto stale=copy_point(final,directory,"stale-"+std::to_string(placement)+".pt");bad=o;bad.checkpoint_path=stale;
    {std::ofstream stream(stale,std::ios::binary|std::ios::app);stream<<'x';}
    rejects([&]{rpb::make_structured_hard_timing_snapshot(bad,fit);},"stale ordinary parent before model load");
    const auto missing=copy_point(final,directory,"missing-"+std::to_string(placement)+".pt",false);bad=o;bad.checkpoint_path=missing;
    rejects([&]{rpb::make_structured_hard_timing_snapshot(bad,fit);},"missing fifth companion before any parent body");
    const auto malformed=copy_point(final,directory,"malformed-"+std::to_string(placement)+".pt");bad=o;bad.checkpoint_path=malformed;
    replace(malformed+rpb::kStructuredHardTimingSuffix,"completed_steps",torch::tensor(std::vector<int64_t>{4}));
    rejects([&]{rpb::make_structured_hard_timing_snapshot(bad,fit);},"non-scalar typed fifth binding");
    rejects([&]{segmented[placement].train_to(2);},"decreasing absolute budget");
    overall.unchanged("new wrapper lifecycle and fixture-isolated comparison preserve runtime");
  }
  const auto pairing=rpb::audit_structured_hard_timing_initialization(options(zero_paths[0],0,0,segmented[0]),options(zero_paths[1],1,0,segmented[1]),fit);
  check(pairing.at("common_parameters_exact")=="true"&&pairing.at("scaler_exact")=="true"&&pairing.at("historical_quality_input_roles")=="0","exact paired new initialization");
  settings.model.channel_mixer_placement=1;auto stopped=rpb::make_structured_hard_timing_trainer(settings,rpb::StructuredHardTimingScope::engineering)(fit);
  const auto stopped0=(directory/"stopped0.pt").string();stopped.save_checkpoint(stopped0);
  replace(stopped0+rpb::kStructuredHardTimingSuffix,"completed_steps",torch::tensor(int64_t(2)));
  rejects([&]{stopped.train_to(2);},"changed saved point prevents further updates");rejects([&]{stopped.save_checkpoint((directory/"forbidden.pt").string());},"failed wrapper permanently blocks saving");
  auto ineligible=fit;ineligible.training_observations.feature_mask=torch::zeros_like(observed);
  ineligible.training_observations.feature_mask.narrow(2,0,8).fill_(true);
  ineligible.training_observations.data=legal.masked_fill(ineligible.training_observations.feature_mask.logical_not(),0);
  auto skipped=rpb::make_structured_hard_timing_trainer(settings,rpb::StructuredHardTimingScope::engineering)(ineligible);
  rejects([&]{skipped.train_to(2);},"ineligible/skipped trajectory cannot satisfy absolute prefix");
  rejects([&]{skipped.train_to(4);},"failed skipped trajectory cannot continue");
  rejects([&]{skipped.save_checkpoint((directory/"skipped.pt").string());},"failed skipped trajectory cannot save");
  overall.unchanged("all negative declarations and byte guards preserve runtime");
  std::cout<<"STRUCTURED_HARD_TIMING_ADAPTER_SOURCE_ID="<<STRUCTURED_HARD_TIMING_ADAPTER_SOURCE_ID<<'\n'
      <<"Structured hard timing CUDA admission passed\n";
}
} // namespace
int main(){try{run();return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
