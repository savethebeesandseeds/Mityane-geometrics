// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Bounded engineering comparison. Synthetic inputs only; no research rescoring.
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v16.h"
#include <torch/cuda.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace old=embedding::encoders::causal_patch_bottleneck_mae::v16;
namespace tr=cp::training;
namespace ot=old::training;
namespace impl=cp::implementation;
namespace fs=std::filesystem;
namespace {
int64_t checks=0,physical_updates=0,gradient_tensors_compared=0;
void check(bool condition,const char *message){++checks;TORCH_CHECK(condition,"[CPB reference equivalence] ",message);}
bool bits(const torch::Tensor &a,const torch::Tensor &b){
  if(!a.defined()||!b.defined())return a.defined()==b.defined();
  return cp::boundary::same_bits(a,b);
}
std::string quote(const std::string &text){
  std::ostringstream out;out<<'"';
  for(const unsigned char c:text){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}
  out<<'"';return out.str();
}
old::RawHistory historical(const cp::RawHistory &raw){return{raw.data,raw.observed,raw.channel_ids,raw.timestamps.select(1,0).contiguous(),raw.sampling_interval};}
cp::RawHistory history(const cp::Config &config,int64_t phase=0){
  const auto &c=config.level;const int64_t B=3,N=c.history_length+17;
  auto data=torch::zeros({B,c.channel_count,N,c.input_width},torch::kFloat64);
  auto observed=torch::ones(data.sizes(),torch::kBool);
  auto x=data.accessor<double,4>();auto m=observed.accessor<bool,4>();
  for(int64_t b=0;b<B;++b)for(int64_t j=0;j<c.channel_count;++j)
    for(int64_t t=0;t<N;++t)for(int64_t f=0;f<c.input_width;++f){
      x[b][j][t][f]=20.+j*2.+f*.3+b*.4+.013*t+.006*phase+
          .8*std::sin(t*.13+f*.23+j*.37+b*.29+.01*phase)+.17*std::cos(t*.67+f*.11);
      m[b][j][t][f]=(t+f*3+j+b)%31!=0 && !(j==1 && t>=42 && t<=48);
    }
  const auto times=(torch::arange(N,torch::kFloat64)*c.sampling_interval).view({1,1,N}).expand({B,c.channel_count,N}).contiguous();
  return{data,observed,torch::tensor(impl::resolved_channel_ids(c),torch::kInt64),times,c.sampling_interval};
}
torch::Tensor cutoffs(const cp::Config &config){
  const auto H=config.level.history_length;
  return torch::tensor(std::vector<double>{double(H-1),double(H+7),double(H+16)},torch::kFloat64)*config.level.sampling_interval;
}
cp::RawHistory poisoned(const cp::RawHistory &raw,const torch::Tensor &cuts){
  auto out=raw;out.data=raw.data.clone();out.observed=raw.observed.clone();
  for(int64_t b=0;b<raw.data.size(0);++b){
    const auto future=raw.timestamps[b].gt(cuts[b]).unsqueeze(-1).expand_as(out.data[b]);
    out.data[b].masked_fill_(future,std::numeric_limits<double>::quiet_NaN());
    out.observed[b].copy_(torch::where(future,out.observed[b].logical_not(),out.observed[b]));
  }
  return out;
}
template<class A,class B>void same_model(const A &a,const B &b){
  const auto ap=a->named_parameters(),bp=b->named_parameters();
  check(ap.size()==bp.size(),"same parameter inventory");
  for(const auto &p:ap){check(bp.contains(p.key()),"matching parameter key");const auto &q=bp[p.key()];
    check(p.value().device()==torch::Device(torch::kCUDA,0)&&q.device()==p.value().device()&&
        p.value().requires_grad()==q.requires_grad()&&bits(p.value(),q),"exact CUDA parameter values/shape/type/flags");}
  const auto ab=a->named_buffers(),bb=b->named_buffers();check(ab.size()==bb.size(),"same buffer inventory");
  for(const auto &p:ab)check(bb.contains(p.key())&&p.value().device()==bb[p.key()].device()&&bits(p.value(),bb[p.key()]),"exact registered buffer");
  check(a->normalizer().identity()==b->normalizer().identity(),"same frozen bank identity");
}
template<class A,class B>void same_gradients(const A &a,const B &b){
  const auto ap=a->named_parameters(),bp=b->named_parameters();
  check(ap.size()==bp.size(),"gradient inventory matches");
  for(const auto &p:ap){const auto g=p.value().grad(),h=bp[p.key()].grad();
    check(bits(g,h),"exact per-parameter gradient or matching absence");
    if(g.defined()){check(p.value().requires_grad()&&g.is_cuda()&&torch::isfinite(g).all().template item<bool>(),"finite learned CUDA gradient");++gradient_tensors_compared;}
  }
}
template<class A,class B>void same_optimizer(const A &a,const torch::optim::AdamW &oa,const B &b,const torch::optim::AdamW &ob){
  const auto ap=a->named_parameters(),bp=b->named_parameters();
  check(oa.state().size()==ob.state().size(),"same optimizer state inventory");
  for(const auto &p:ap)if(p.value().requires_grad()){
    const auto ia=oa.state().find(p.value().unsafeGetTensorImpl()),ib=ob.state().find(bp[p.key()].unsafeGetTensorImpl());
    check((ia==oa.state().end())==(ib==ob.state().end()),"same optimizer state presence");if(ia==oa.state().end())continue;
    const auto *sa=dynamic_cast<const torch::optim::AdamWParamState *>(ia->second.get());
    const auto *sb=dynamic_cast<const torch::optim::AdamWParamState *>(ib->second.get());
    check(sa&&sb&&sa->step()==sb->step(),"same AdamW step");
    check(bits(sa->exp_avg(),sb->exp_avg())&&bits(sa->exp_avg_sq(),sb->exp_avg_sq())&&bits(sa->max_exp_avg_sq(),sb->max_exp_avg_sq()),"exact AdamW moment content");
  }
}
void same_progress(const tr::TrainingProgress &a,const tr::TrainingProgress &b){
  check(a.completed_updates==b.completed_updates&&a.attempted_batches==b.attempted_batches&&
      a.row_seed==b.row_seed&&a.mask_seed==b.mask_seed&&a.context_seed==b.context_seed,"same update/attempt and row/mask/context counters");
  for(const auto &seed:std::vector<std::pair<int64_t,int64_t>>{{a.row_seed,b.row_seed},{a.mask_seed,b.mask_seed},{a.context_seed,b.context_seed}})
    check(tr::training_counter_seed(seed.first,a.attempted_batches)==tr::training_counter_seed(seed.second,b.attempted_batches),"same next counter stream");
}
void same_forward(const impl::LocationScaleForwardTrace &a,const impl::LocationScaleForwardTrace &b){
  for(const auto &pair:std::vector<std::pair<torch::Tensor,torch::Tensor>>{
      {a.output.encoding.z_contextual_global,b.output.encoding.z_contextual_global},
      {a.output.encoding.sample_valid_mask,b.output.encoding.sample_valid_mask},
      {a.targets,b.targets},{a.target_mask,b.target_mask},{a.visible,b.visible},
      {a.distribution.mean,b.distribution.mean},{a.distribution.variance,b.distribution.variance},
      {a.nll.loss,b.nll.loss},{a.objective.temporal.loss,b.objective.temporal.loss},{a.output.loss,b.output.loss}})
    check(bits(pair.first,pair.second),"exact masked encoding/distribution/support/objective");
  check(a.objective.temporal.lags.size()==3&&b.objective.temporal.lags.size()==3,"three declared temporal losses");
  for(size_t i=0;i<3;++i)check(bits(a.objective.temporal.lags[i].pair_mask,b.objective.temporal.lags[i].pair_mask)&&
      bits(a.objective.temporal.lags[i].loss,b.objective.temporal.lags[i].loss),"exact lag support and loss");
}
struct Snapshot {
  std::map<std::string,torch::Tensor> parameters,buffers;
  std::string bank,scaler;
  template<class Model>Snapshot(const Model &model,const cp::FrozenScaler &input):bank(model->normalizer().identity()),scaler(input.identity()){
    for(const auto &p:model->named_parameters())parameters.emplace(p.key(),p.value().detach().clone());
    for(const auto &p:model->named_buffers())buffers.emplace(p.key(),p.value().detach().clone());
  }
  template<class Model>void unchanged(const Model &model,const cp::FrozenScaler &input)const{
    const auto now=model->named_parameters(),now_buffers=model->named_buffers();
    check(now.size()==parameters.size()&&now_buffers.size()==buffers.size(),"unchanged reference inventory");
    for(const auto &p:parameters)check(bits(p.second,now[p.first]),"reference weights unchanged");
    for(const auto &p:buffers)check(bits(p.second,now_buffers[p.first]),"reference buffers unchanged");
    check(bank==model->normalizer().identity()&&scaler==input.identity(),"reference frozen preprocessing unchanged");
  }
};
cp::Encoder reference_adapter(const old::Loaded &reference){
  // Evaluation-only adaptation: all tensors are copied into separate storage.
  // Its constructor metadata is NEVER exported as a fresh-family artifact.
  auto core=impl::LocationScaleDecoderModel(reference.config,reference.model->normalizer());
  const auto source=reference.model->core()->named_parameters(),destination=core->named_parameters();
  const auto source_buffers=reference.model->core()->named_buffers(),destination_buffers=core->named_buffers();
  check(source.size()==destination.size()&&source_buffers.size()==destination_buffers.size(),"full reference adaptation inventory");
  {torch::NoGradGuard guard;
    for(const auto &p:source){check(destination.contains(p.key()),"reference adaptation parameter key");
      destination[p.key()].copy_(p.value());check(destination[p.key()].unsafeGetTensorImpl()!=p.value().unsafeGetTensorImpl()&&
          destination[p.key()].const_data_ptr()!=p.value().const_data_ptr(),"reference parameters use independent storage");}
    for(const auto &p:source_buffers){check(destination_buffers.contains(p.key()),"reference adaptation buffer key");destination_buffers[p.key()].copy_(p.value());
      check(destination_buffers[p.key()].unsafeGetTensorImpl()!=p.value().unsafeGetTensorImpl()&&
          (p.value().numel()==0||destination_buffers[p.key()].const_data_ptr()!=p.value().const_data_ptr()),"reference buffers use independent storage");}
  }
  auto model=cp::Encoder(core,reference.initialization_seed,reference.scaler.identity(),reference.causal_updates,reference.initial_state_identity);
  cp::validate_model_counts(model);model->eval();model->clear_gradients();return model;
}
torch::Tensor context(const impl::MaskPlan &masks,const tr::TrainingProgress &progress){
  const auto visible=masks.visible.to(torch::kCPU);auto deleted=torch::zeros_like(visible);auto out=deleted.accessor<bool,4>();
  const auto v=visible.accessor<bool,4>();const auto shift=tr::training_counter_seed(progress.context_seed,progress.attempted_batches)%37;
  for(int64_t b=0;b<visible.size(0);++b)for(int64_t c=0;c<visible.size(1);++c)
    for(int64_t t=0;t<visible.size(2);++t)for(int64_t f=0;f<visible.size(3);++f)
      out[b][c][t][f]=v[b][c][t][f]&&((b+3*c+t+7*f+shift)%37==0);
  return deleted;
}
tr::StepResult old_step(old::Encoder &model,const cp::FrozenScaler &scaler,torch::optim::AdamW &optimizer,
    tr::TrainingProgress &progress,const old::RawHistory &raw,const torch::Tensor &cuts,const impl::MaskPlan &masks,const torch::Tensor &deleted){
  ot::validate_optimizer(model,optimizer,progress);check(model->causal_updates()==progress.completed_updates,"old live update counter");
  optimizer.zero_grad(true);model->train();auto forward=model->forward_at(raw,cuts,masks.hidden,scaler,deleted);
  check(forward.nll.eligible_example_count>0&&torch::isfinite(forward.output.loss).item<bool>(),"supported finite historical training loss");
  tr::StepResult result;result.gaussian_nll=forward.nll.loss.item<double>();result.temporal_loss=forward.objective.temporal.loss.item<double>();result.total_loss=forward.output.loss.item<double>();
  forward.output.loss.backward();
  result.gradient_norm=torch::nn::utils::clip_grad_norm_(ot::learned_parameters(model),1.);
  check(std::isfinite(result.gradient_norm),"finite historical gradient norm");optimizer.step();++physical_updates;
  ++progress.completed_updates;++progress.attempted_batches;model->set_causal_updates(progress.completed_updates);
  optimizer.zero_grad(true);model->clear_gradients();model->core()->validate_state();model->core()->validate_fixed_prior();
  ot::validate_optimizer(model,optimizer,progress);result.updated=true;return result;
}
void same_result(const tr::StepResult &a,const tr::StepResult &b){
  check(a.updated&&b.updated&&a.gaussian_nll==b.gaussian_nll&&a.temporal_loss==b.temporal_loss&&
      a.total_loss==b.total_loss&&a.gradient_norm==b.gradient_norm,"exact training loss components and unclipped norm");
}
struct ReferenceResult {double gaussian,temporal,total;};
ReferenceResult trained_reference(old::Loaded &reference){
  auto adapted=reference_adapter(reference);same_model(reference.model,adapted);
  const Snapshot before(reference.model,reference.scaler),adapted_before(adapted,reference.scaler);
  const auto raw=history(reference.config);const auto cuts=cutoffs(reference.config);const auto poison=poisoned(raw,cuts);
  const auto legacy=historical(raw),legacy_poison=historical(poison);
  const auto a=reference.model->encode_at(legacy,cuts,reference.scaler);
  const auto b=adapted->encode_at(raw,cuts,reference.scaler);
  check(bits(a.native,b.native)&&bits(a.valid,b.valid)&&bits(a.cutoffs,b.cutoffs),"selected trained checkpoint exact native/valid/cutoff output");
  const auto future=adapted->encode_at(poison,cuts,reference.scaler);
  check(bits(b.native,future.native)&&bits(b.valid,future.valid),"trained family reference crops future values and masks");
  const tr::TrainingProgress counters{0,0,221101,221102,221103};
  const auto masks=tr::make_mask(raw,cuts,reference.config,counters.mask_seed,0);const auto deleted=context(masks,counters);
  reference.model->train();adapted->train();
  const auto old_forward=reference.model->forward_at(legacy,cuts,masks.hidden,reference.scaler,deleted);
  const auto new_forward=adapted->forward_at(raw,cuts,masks.hidden,reference.scaler,deleted);
  const auto old_poison=reference.model->forward_at(legacy_poison,cuts,masks.hidden,reference.scaler,deleted);
  const auto new_poison=adapted->forward_at(poison,cuts,masks.hidden,reference.scaler,deleted);
  same_forward(old_forward,new_forward);same_forward(old_forward,old_poison);same_forward(new_forward,new_poison);
  old_forward.output.loss.backward();new_forward.output.loss.backward();same_gradients(reference.model,adapted);
  check(gradient_tensors_compared>0,"trained objective reaches actual learned parameters");
  before.unchanged(reference.model,reference.scaler);adapted_before.unchanged(adapted,reference.scaler);
  reference.model->clear_gradients();adapted->clear_gradients();reference.model->eval();adapted->eval();
  reference.model->validate_frozen();adapted->validate_frozen();
  return{old_forward.nll.loss.item<double>(),old_forward.objective.temporal.loss.item<double>(),old_forward.output.loss.item<double>()};
}
void matched_fresh_training(const old::Loaded &reference,const fs::path &output){
  // Use the exact saved preprocessing to isolate the API change. No fitter is called.
  auto legacy=old::fresh(reference.config,reference.scaler,reference.model->normalizer(),946103);
  auto current=cp::fresh(reference.config,reference.scaler,reference.model->normalizer(),946103);
  check(legacy.model->initial_state_identity()==reference.initial_state_identity,"fresh historical initialization recreates selected checkpoint's recorded initial state");
  same_model(legacy.model,current.model);const auto initial=legacy.model->state_identity();
  auto legacy_optimizer=ot::make_optimizer(legacy.model),current_optimizer=tr::make_optimizer(current.model);
  const tr::OptimizerConfig options;check(tr::adamw_options(options)==ot::adamw_options(),"configured defaults reproduce historical AdamW recipe");
  tr::TrainingProgress lp{0,0,222001,222002,222003},cp_progress=lp;
  const tr::TrainingIdentity identity{"synthetic-reference-equivalence-v1","engineering-only-TRAIN/no-market-read","old-family-mask-context-counter-AdamW-defaults-v1"};
  const auto perform=[&](old::Encoder &old_model,torch::optim::AdamW &old_optimizer,tr::TrainingProgress &old_progress,
      cp::Encoder &new_model,torch::optim::AdamW &new_optimizer,tr::TrainingProgress &new_progress){
    same_progress(old_progress,new_progress);const auto phase=tr::training_counter_seed(old_progress.row_seed,old_progress.attempted_batches)%97;
    const auto raw=history(reference.config,phase);const auto cuts=cutoffs(reference.config);
    const auto masks=tr::make_mask(raw,cuts,reference.config,old_progress.mask_seed,old_progress.attempted_batches);
    const auto retained=old::boundary::retain(historical(raw),cuts,reference.config);
    const auto historical_masks=impl::configurable_spectral_make_training_mask(retained.raw.observed.to(reference.config.level.device),
        reference.config.level,ot::training_counter_seed(old_progress.mask_seed,old_progress.attempted_batches));
    check(bits(masks.hidden,historical_masks.hidden)&&bits(masks.target,historical_masks.target)&&bits(masks.visible,historical_masks.visible),"portable mask stream matches historical path");
    const auto deleted=context(masks,old_progress);
    const auto a=old_step(old_model,reference.scaler,old_optimizer,old_progress,historical(raw),cuts,historical_masks,deleted);
    const auto b=tr::train_step(new_model,reference.scaler,new_optimizer,new_progress,raw,cuts,options,deleted);++physical_updates;
    same_result(a,b);same_model(old_model,new_model);same_optimizer(old_model,old_optimizer,new_model,new_optimizer);same_progress(old_progress,new_progress);
  };
  perform(legacy.model,*legacy_optimizer,lp,current.model,*current_optimizer,cp_progress);
  check(legacy.model->state_identity()!=initial,"fresh matched training changes learned weights");
  const auto old_path=output/"fresh-old-training-1.pt",new_path=output/"fresh-family-training-1.pt";
  ot::save_training_state(old_path.string(),legacy.model,legacy.scaler,*legacy_optimizer,identity,lp);
  tr::save_training_state(new_path.string(),current.model,current.scaler,*current_optimizer,identity,cp_progress,options);
  auto old_resumed=ot::load_training_state(old_path.string(),reference.config,identity);
  auto new_resumed=tr::load_training_state(new_path.string(),reference.config,identity,options);
  same_model(legacy.model,old_resumed.model);same_optimizer(legacy.model,*legacy_optimizer,old_resumed.model,*old_resumed.optimizer);same_progress(lp,old_resumed.progress);
  same_model(current.model,new_resumed.model);same_optimizer(current.model,*current_optimizer,new_resumed.model,*new_resumed.optimizer);same_progress(cp_progress,new_resumed.progress);
  for(int step=0;step<2;++step){
    perform(legacy.model,*legacy_optimizer,lp,current.model,*current_optimizer,cp_progress);
    perform(old_resumed.model,*old_resumed.optimizer,old_resumed.progress,new_resumed.model,*new_resumed.optimizer,new_resumed.progress);
    same_model(legacy.model,old_resumed.model);same_optimizer(legacy.model,*legacy_optimizer,old_resumed.model,*old_resumed.optimizer);same_progress(lp,old_resumed.progress);
    same_model(current.model,new_resumed.model);same_optimizer(current.model,*current_optimizer,new_resumed.model,*new_resumed.optimizer);same_progress(cp_progress,new_resumed.progress);
  }
  check(lp.completed_updates==3&&old_resumed.progress.completed_updates==3&&cp_progress.completed_updates==3&&new_resumed.progress.completed_updates==3,"all fresh trajectories end at logical update3");
  for(auto *model:{&legacy.model,&old_resumed.model}){(*model)->clear_gradients();(*model)->eval();(*model)->validate_frozen();}
  for(auto *model:{&current.model,&new_resumed.model}){(*model)->clear_gradients();(*model)->eval();(*model)->validate_frozen();}
  const auto raw=history(reference.config,73);const auto cuts=cutoffs(reference.config);
  check(bits(legacy.model->encode_at(historical(raw),cuts,legacy.scaler).native,current.model->encode_at(raw,cuts,current.scaler).native),"matched final inference native output");
  check(legacy.scaler.identity()==reference.scaler.identity()&&current.scaler.identity()==reference.scaler.identity()&&
      legacy.model->normalizer().identity()==reference.model->normalizer().identity()&&current.model->normalizer().identity()==reference.model->normalizer().identity(),"selected TRAIN preprocessing stays exact without refit");
}
}
int main(int argc,char **argv){try{
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  check(argc==7,"usage: --checkpoint FILE --expected-sha256 HASH --output NEWLEAF");
  std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2)check(args.emplace(argv[i],argv[i+1]).second,"unique options");
  check(args.size()==3&&args.count("--checkpoint")&&args.count("--expected-sha256")&&args.count("--output"),"only declared CLI options");
  const fs::path checkpoint=args.at("--checkpoint"),output=args.at("--output");const auto expected=args.at("--expected-sha256");
  check(expected.size()==64&&expected.find_first_not_of("0123456789abcdef")==std::string::npos,"explicit lowercase SHA256 required");
  check(torch::cuda::is_available(),"actual CUDA required");check(old::boundary::sha256_file(checkpoint)==expected,"reference byte hash before loading");
  check(!fs::exists(output)&&fs::create_directories(output),"exclusive new evidence directory");
  auto reference=old::load_checkpoint(checkpoint.string(),torch::Device(torch::kCUDA,0));
  check(reference.initialization_seed==946103&&reference.source_updates==0&&reference.causal_updates==2048&&
      reference.config.level.channel_count==3&&reference.config.level.history_length==128&&reference.config.level.input_width==9&&
      reference.config.level.patch_length==8&&reference.config.level.export_width==32&&reference.config.change_width==4,"selected trained seed946103/update2048/native36 reference");
  const Snapshot preserved(reference.model,reference.scaler);const auto state=reference.model->state_identity();
  const auto result=trained_reference(reference);matched_fresh_training(reference,output);
  preserved.unchanged(reference.model,reference.scaler);check(reference.model->state_identity()==state,"original loaded trained reference remains unchanged");
  check(old::boundary::sha256_file(checkpoint)==expected,"reference archive bytes unchanged after comparison");
  check(physical_updates==10,"exact bounded physical training budget");
  std::ostringstream json;json<<std::setprecision(17)<<"{\"kind\":\"cpb_family_trained_reference_equivalence_v1\",\"status\":\"passed\",\"checkpoint\":"<<quote(checkpoint.string())
      <<",\"checkpoint_sha256\":"<<quote(expected)<<",\"selected_reference_seed\":946103,\"selected_reference_logical_update\":2048,\"native_width\":36"
      <<",\"trained_reference_native_loss_gradient_parity\":true,\"trained_reference_future_poison_invariance\":true,\"trained_reference_weights_and_statistics_unchanged\":true"
      <<",\"borrowed_reference_adapter_evaluation_only\":true,\"borrowed_weight_artifacts_written\":0,\"borrowed_weight_optimizer_updates\":0,\"preprocessing_fits\":0"
      <<",\"fresh_constructor_matches_recorded_initial_state\":true,\"matched_training_logical_updates\":3,\"fresh_trajectory_count\":4,\"resume_save_update\":1,\"matched_resumed_continuation_updates\":2"
      <<",\"physical_synthetic_optimizer_updates\":"<<physical_updates<<",\"gradient_tensors_compared\":"<<gradient_tensors_compared<<",\"checks\":"<<checks
      <<",\"same_frozen_TRAIN_statistics_used\":true,\"fresh_fixture_calibration_policy\":\"reference statistics reused only to isolate API equivalence; not an application training recipe\""
      <<",\"new_fitter_compared_to_historical_fitter\":false,\"market_datasets_read\":0,\"quality_evaluated\":false"
      <<",\"trained_reference_synthetic_objective_witness\":{\"gaussian_nll\":"<<result.gaussian<<",\"temporal_loss\":"<<result.temporal<<",\"total_loss\":"<<result.total<<"}}";
  std::ofstream stream(output/"summary.json");stream<<json.str()<<'\n';stream.close();TORCH_CHECK(bool(stream),"write summary");std::cout<<json.str()<<std::endl;return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
