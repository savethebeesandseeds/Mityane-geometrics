// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Artificial CUDA engineering only. No historical encoder training or quality.
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_training_state.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_settings.h"
#include "cuwacunu/encoder/parent_modulated_encoder.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/relative_change_training_state.h"
#include <torch/cuda.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

namespace {
namespace r=embedding::encoders::raw_patch_bottleneck_mae;
namespace tr=r::parent_modulated_decoder_training;
namespace oldtr=r::separated_change_training;
void check(bool value,const char *why){TORCH_CHECK(value,"[parent-modulated CUDA admission] ",why);}
template<class F>void rejects(F operation,const char *why){bool caught=false;try{operation();}catch(const std::exception&){caught=true;}check(caught,why);}
void near(double actual,double expected,const char *why,double tolerance=2e-6){check(std::isfinite(actual)&&std::abs(actual-expected)<=tolerance*std::max(1.,std::abs(expected)),why);}

struct Evidence {
  std::filesystem::path path;
  Evidence(){
    std::filesystem::create_directories("output/engineering/parent-modulated-decoder-fixture");
    auto pattern=std::filesystem::absolute("output/engineering/parent-modulated-decoder-fixture/run-XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
    check(::mkdtemp(name.data()),"exclusive fixture evidence directory");path=name.data();
    std::cout<<"PARENT_MODULATED_FIXTURE_DIRECTORY="<<path.string()<<std::endl;
  }
  // Keep engineering archives on success and failure as separate local evidence.
};

struct MomentOracle{torch::Tensor values,valid;};
MomentOracle moment_oracle(const r::Input &input,const r::Config &c){
  const auto data=input.data.to(torch::kCPU).to(torch::kFloat64).contiguous();
  const auto observed=input.observed.to(torch::kCPU).contiguous();
  const auto ids=input.channel_ids.to(torch::kCPU).contiguous();
  check(ids.dim()==1,"fixture scalar oracle uses rank1 semantic ids");
  const auto B=data.size(0),C=c.channel_count,F=c.input_width,H=c.history_length;
  auto values=torch::zeros({B,C,F,10},torch::kFloat64);
  auto valid=torch::zeros({B,C,F,10},torch::kBool);
  const auto x=data.accessor<double,4>();const auto mask=observed.accessor<bool,4>();
  auto v=values.accessor<double,4>();auto good=valid.accessor<bool,4>();
  for(int64_t b=0;b<B;++b)for(int64_t channel=0;channel<C;++channel){
    int64_t physical=-1;
    for(int64_t j=0;j<C;++j)if(ids[j].item<int64_t>()==c.channel_ids[channel])physical=j;
    check(physical>=0,"all semantic channels present in oracle");
    for(int64_t f=0;f<F;++f){
      std::vector<double> delta(H-1);std::vector<bool> support(H-1);
      int64_t count=0;double sum=0,sumsq=0;
      for(int64_t t=0;t<H-1;++t)if(mask[b][physical][t][f]&&mask[b][physical][t+1][f]){
        const double a=x[b][physical][t+1][f],previous=x[b][physical][t][f],m=std::max(std::abs(a),std::abs(previous));
        const double aa=a/(m>0?m:1.),bb=previous/(m>0?m:1.),denominator=std::abs(aa)+std::abs(bb);
        const double d=float(2.*(aa-bb)/(denominator>0?denominator:1.));
        check(std::isfinite(d),"oracle only reads legal finite endpoints");
        support[t]=true;delta[t]=d;++count;sum+=d;sumsq+=d*d;
      }
      if(count<2)continue;
      const double rms=std::sqrt(std::max(sumsq/count,1e-12));
      v[b][channel][f][0]=sum/count;v[b][channel][f][1]=sumsq>0?rms:0;
      good[b][channel][f][0]=good[b][channel][f][1]=true;
      for(int64_t q=0;q<4;++q){
        const auto begin=q*(H-1)/4,end=(q+1)*(H-1)/4;
        int64_t n=0;double part_sum=0,part_square=0;
        for(auto t=begin;t<end;++t)if(support[t]){++n;part_sum+=delta[t];part_square+=delta[t]*delta[t];}
        if(n<2)continue;
        v[b][channel][f][2+2*q]=part_sum/n/rms;
        v[b][channel][f][3+2*q]=part_square>0?std::sqrt(std::max(part_square/n,1e-12))/rms:0;
        good[b][channel][f][2+2*q]=good[b][channel][f][3+2*q]=true;
      }
    }
  }
  return{values.flatten(1),valid.flatten(1)};
}
void check_moments(const r::Input &input,const r::Config &c){
  const auto actual=r::relative_change_targets(input,c);const auto expected=moment_oracle(input,c);
  check(actual.values.is_cuda()&&actual.valid.is_cuda(),"moment targets and support are actual CUDA tensors");
  check(torch::equal(actual.values,r::relative_change_bank(input,c).values),"raw relative bank primitive and target helper agree exactly");
  check(torch::equal(actual.valid.to(torch::kCPU),expected.valid),"scalar adjacent-pair and original-quarter support oracle");
  check(torch::allclose(actual.values.to(torch::kCPU).to(torch::kFloat64),expected.values,2e-6,2e-7),"scalar signed/global RMS/normalized-quarter oracle");
}
void check_normalizer(const r::RelativeBankStandardizer &norm,const r::RelativeChangeTargets &targets){
  const auto values=targets.values.to(torch::kCPU,torch::kFloat64).contiguous();
  const auto valid=targets.valid.to(torch::kCPU).contiguous();const auto x=values.accessor<double,2>();const auto mask=valid.accessor<bool,2>();
  const auto transformed=norm.transform(targets).to(torch::kCPU).contiguous();
  for(int64_t j=0;j<values.size(1);++j){
    int64_t count=0;double sum=0,square=0;
    for(int64_t b=0;b<values.size(0);++b)if(mask[b][j]){++count;sum+=x[b][j];square+=x[b][j]*x[b][j];}
    const double mean=count>=2?sum/count:0,scale=count>=2?std::max(std::sqrt(std::max(square/count-mean*mean,0.)),1e-6):1;
    check(norm.counts[j].item<int64_t>()==count,"streaming scalar counts");near(norm.sum[j].item<double>(),sum,"streaming scalar sums",1e-12);
    near(norm.sumsq[j].item<double>(),square,"streaming scalar sum of squares",1e-12);
    near(norm.mean[j].item<double>(),mean,"population mean with minimum2",1e-12);near(norm.scale[j].item<double>(),scale,"population standard deviation/floor",1e-12);
    check(norm.support[j].item<bool>()==(count>=2),"minimum2 support");
    for(int64_t b=0;b<values.size(0);++b){
      const float expected=mask[b][j]&&count>=2?(float(x[b][j])-float(mean))/float(scale):0.f;
      if(!mask[b][j]||count<2)check(transformed[b][j].item<float>()==0.f,"invalid and unsupported conditioning is exact0");
      else near(transformed[b][j].item<float>(),expected,"F32 conditioning scalar arithmetic",2e-7);
    }
  }
}
void same_model(const r::ParentModulatedDecoderModel &a,const r::ParentModulatedDecoderModel &b){
  const auto params=b->named_parameters(),buffers=b->named_buffers();
  for(const auto &p:a->named_parameters())check(torch::equal(p.value(),params[p.key()]),"complete parameter state exact");
  for(const auto &v:a->named_buffers())check(torch::equal(v.value(),buffers[v.key()]),"complete registered buffer state exact");
  check(a->normalizer().identity()==b->normalizer().identity(),"complete frozen standardizer state exact");
}
void same_optimizer(const r::ParentModulatedDecoderModel &a,const torch::optim::AdamW &oa,const r::ParentModulatedDecoderModel &b,const torch::optim::AdamW &ob){
  const auto ap=a->named_parameters(),bp=b->named_parameters();check(oa.state().size()==ob.state().size(),"optimizer state count exact");
  for(const auto &p:ap){
    const auto ai=oa.state().find(p.value().unsafeGetTensorImpl()),bi=ob.state().find(bp[p.key()].unsafeGetTensorImpl());
    check((ai==oa.state().end())==(bi==ob.state().end()),"optimizer support exact");if(ai==oa.state().end())continue;
    const auto *av=dynamic_cast<const torch::optim::AdamWParamState*>(ai->second.get()),*bv=dynamic_cast<const torch::optim::AdamWParamState*>(bi->second.get());
    check(av&&bv&&av->step()==bv->step()&&torch::equal(av->exp_avg(),bv->exp_avg())&&torch::equal(av->exp_avg_sq(),bv->exp_avg_sq()),"optimizer steps and moments exact");
  }
}
torch::Tensor update(const r::Input &input,r::ParentModulatedDecoderModel &model,torch::optim::AdamW &optimizer,tr::TrainingProgress &progress,const tr::FrozenLevelGuard &guard,const r::FrozenScaler &scaler){
  const auto &c=model->config().level;
  const auto masks=r::configurable_spectral_make_training_mask(input.observed,c,tr::training_counter_seed(progress.mask_seed,progress.attempted_batches));
  const auto context=r::context_deletion::make_plan(masks,input.channel_ids,c,progress.context_seed,progress.attempted_batches,r::ContextDeletionRecipe::coordinate15_v1);
  const r::Input visible{input.data,context.visible,input.channel_ids,input.endpoints,input.sampling_interval};
  model->train();optimizer.zero_grad();const auto native=model->encode(visible,scaler).z_contextual_global,reconstruction=model->decode(native,input.channel_ids);
  const auto wave=r::hierarchical_huber(reconstruction,r::relative_change_parent_input(input,scaler,c).data.detach(),masks.target,masks.eligible_channels,c.huber_delta);
  check(input.data.is_cuda()&&visible.observed.is_cuda()&&native.is_cuda()&&reconstruction.is_cuda()&&wave.loss.is_cuda(),"actual CUDA input/native/reconstruction/loss");
  check(torch::isfinite(wave.loss).item<bool>()&&wave.eligible_example_count>0,"finite eligible waveform-only loss");wave.loss.backward();
  const auto pg=model->encoder->change_projection.grad(),dg=model->encoder->change_decoder.grad(),ag=model->waveform_gain.grad();
  for(const auto &g:{pg,dg,ag})check(g.defined()&&g.is_cuda()&&torch::isfinite(g).all().item<bool>(),"all three actual branch gradients finite CUDA");
  check(dg.abs().sum().item<double>()>0,"waveform residual decoder has nonzero gradient");
  check(progress.completed_updates?pg.abs().sum().item<double>()>0:pg.eq(0).all().item<bool>(),"first zero-decoder then projection loss gradient");
  check(progress.completed_updates?ag.abs().sum().item<double>()>0:ag.eq(0).all().item<bool>(),"first zero-decoder then gain loss gradient");
  for(const auto &p:model->encoder->level->parameters()){
    check(p.is_cuda()&&!p.grad().defined(),"parent is structurally outside graph");check(!optimizer.state().count(p.unsafeGetTensorImpl()),"parent has no optimizer moments");
  }
  guard.discard_gradients(model,optimizer);const auto projection=model->encoder->change_projection.detach().clone(),decoder=model->encoder->change_decoder.detach().clone(),gain=model->waveform_gain.detach().clone();
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer.step();
  check(!torch::equal(decoder,model->encoder->change_decoder),"actual CUDA decoder update");
  if(progress.completed_updates){check(!torch::equal(projection,model->encoder->change_projection),"actual CUDA projection update");check(!torch::equal(gain,model->waveform_gain),"actual CUDA gain update");}
  else check(model->waveform_gain.eq(0).all().item<bool>()&&torch::equal(gain,model->waveform_gain),"first gain remains literal zero including AdamW decay");
  ++progress.completed_updates;++progress.attempted_batches;guard.validate(model,optimizer);tr::validate_optimizer(model,optimizer,progress);return wave.loss.detach().clone();
}
void check_decode_trace(r::ParentModulatedDecoderModel &model,const torch::Tensor &native,const torch::Tensor &ids){
  const auto &c=model->config().level;const auto B=native.size(0),C=c.channel_count,H=c.history_length,F=c.input_width;
  auto trace=model->decode_with_trace(native,ids);
  for(const auto &v:{trace.parent_prediction,trace.canonical_base_residual,trace.gain,trace.scaled_residual,trace.prediction})
    check(v.is_cuda()&&v.scalar_type()==torch::kFloat32&&torch::isfinite(v).all().item<bool>(),"actual finite CUDA F32 decoder trace");
  check(trace.gain.sizes()==torch::IntArrayRef({B,C,1,F}),"one canonical C/F gain broadcast over H");
  const auto direct_residual=native.narrow(1,c.export_width,model->config().change_width).matmul(model->encoder->change_decoder.t()).reshape({B,C,H,F});
  const auto direct_gain=1.+native.narrow(1,0,c.export_width).detach().matmul(model->waveform_gain.t()).reshape({B,C,1,F});
  check(torch::equal(trace.canonical_base_residual,direct_residual)&&torch::equal(trace.gain,direct_gain)&&
      torch::equal(trace.scaled_residual,direct_residual*direct_gain),"exact direct F32 gain/residual multiplication witness");
  check(torch::equal(trace.prediction,model->decode(native,ids)),"decode delegates the actual traced prediction");
  const auto n=native.detach().to(torch::kCPU).contiguous(),a=model->waveform_gain.detach().to(torch::kCPU).contiguous(),
      d=model->encoder->change_decoder.detach().to(torch::kCPU).contiguous(),semantic=ids.to(torch::kCPU).contiguous();
  const auto base=trace.parent_prediction.detach().to(torch::kCPU).contiguous(),gain=trace.gain.detach().to(torch::kCPU).contiguous(),
      residual=trace.canonical_base_residual.detach().to(torch::kCPU).contiguous(),scaled=trace.scaled_residual.detach().to(torch::kCPU).contiguous(),prediction=trace.prediction.detach().to(torch::kCPU).contiguous();
  const auto nv=n.accessor<float,2>(),av=a.accessor<float,2>(),dv=d.accessor<float,2>();
  const auto bv=base.accessor<float,4>(),gv=gain.accessor<float,4>(),rv=residual.accessor<float,4>(),sv=scaled.accessor<float,4>(),pv=prediction.accessor<float,4>();
  check(semantic.dim()==1,"decoder scalar fixture uses rank1 semantic ids");
  for(int64_t b=0;b<B;++b)for(int64_t physical=0;physical<C;++physical){
    int64_t canonical=-1;for(int64_t j=0;j<C;++j)if(c.channel_ids[j]==semantic[physical].item<int64_t>())canonical=j;
    check(canonical>=0,"decoder scalar semantic map complete");
    for(int64_t f=0;f<F;++f){
      float g=0;for(int64_t k=0;k<c.export_width;++k)g+=nv[b][k]*av[canonical*F+f][k];g=1.f+g;
      near(gv[b][canonical][0][f],g,"independent scalar F32 gain dot",1e-5);
      for(int64_t t=0;t<H;++t){
        float rvalue=0;for(int64_t k=0;k<model->config().change_width;++k)rvalue+=nv[b][c.export_width+k]*dv[(canonical*H+t)*F+f][k];
        near(rv[b][canonical][t][f],rvalue,"independent scalar F32 residual dot",1e-5);
        near(sv[b][canonical][t][f],float(rvalue*g),"independent scalar gain broadcast multiplication",1e-5);
        near(pv[b][physical][t][f],float(bv[b][physical][t][f]+float(rvalue*g)),"independent semantic gather and parent addition",1e-5);
      }
    }
  }
}
void relative_oracles(const r::Input &input,const r::Config &c){
  const auto baseline=r::relative_change_bank(input,c);
  check(baseline.increments.scalar_type()==torch::kFloat32&&baseline.increments.is_cuda()&&baseline.adjacent_valid.scalar_type()==torch::kBool&&baseline.increments.abs().le(2.).all().item<bool>(),"canonical relative increments are bounded actual CUDA F32");
  auto scaled=input;const auto units=(torch::arange(c.input_width,input.data.options())+1.).view({1,1,1,c.input_width})*3.7;
  scaled.data=input.data*units*torch::tensor({.25,2.,8.},input.data.options()).view({3,1,1,1});
  const auto changed=r::relative_change_bank(scaled,c);
  check(torch::equal(baseline.valid,changed.valid)&&torch::allclose(baseline.values,changed.values,2e-6,2e-7),"positive per-window/feature unit invariance");
  auto tiny=input;tiny.data=input.data*1e-250;check_moments(tiny,c);
  check(torch::allclose(baseline.values,r::relative_change_bank(tiny,c).values,2e-6,2e-7),"tiny finite units preserve relative bank");
  auto precise=input;precise.observed=torch::ones_like(input.observed);
  precise.data=(1.+torch::arange(c.history_length,input.data.options()).view({1,1,c.history_length,1})*1e-9).expand_as(input.data).clone();
  auto rounded=precise;rounded.data=precise.data.to(torch::kFloat32).to(torch::kFloat64);
  check(rounded.data.eq(1.).all().item<bool>()&&r::relative_change_bank(precise,c).values.select(1,0).gt(0).all().item<bool>()&&r::relative_change_bank(rounded,c).values.eq(0).all().item<bool>(),"genuine raw F64 increments survive where historical F32 rounding loses them");
  auto edge=input;edge.observed=torch::ones_like(input.observed);edge.data=torch::zeros_like(input.data);
  const std::vector<double> pattern{0.,0.,1e-300,-1e-300,1.7e308,-1.7e308,0.,1.,-1.,2.,2.,0.};
  for(int64_t t=0;t<c.history_length;++t)edge.data.select(2,t).fill_(pattern[size_t(t)%pattern.size()]);
  check_moments(edge,c);const auto extreme=r::relative_change_bank(edge,c);
  check(torch::isfinite(extreme.values).all().item<bool>()&&extreme.increments.abs().le(2.).all().item<bool>(),"bank-only stable ratio accepts finite extremes/sign crossings/both-zero pairs");
  auto invalid=input;invalid.data=input.data.clone();invalid.data[0][0][0][0]=std::numeric_limits<double>::infinity();
  rejects([&]{r::relative_change_bank(invalid,c);},"observed nonfinite raw values rejected");
}
} // namespace

int main()try{
  check(torch::cuda::is_available(),"CUDA required");torch::set_num_threads(1);torch::set_num_interop_threads(1);Evidence evidence;
  auto c=r::default_content_ordered_relation_config();c.channel_ids={7,41,99};c.history_length=32;c.encoder_width=32;c.num_layers=1;c.feedforward_width=64;c.decoder_hidden_width=64;
  const r::ParentModulatedDecoderConfig native36{c,4},native40{c,8};check(r::parent_modulated_decoder_native_width(native36)==36,"native36 contract");
  const auto time=torch::arange(32,torch::kFloat64).view({1,1,32,1}),feature=torch::arange(9,torch::kFloat64).view({1,1,1,9});
  const auto channel=torch::arange(3,torch::kFloat64).view({1,3,1,1}),batch=torch::arange(3,torch::kFloat64).view({3,1,1,1});
  auto values=(.13*time+.11*feature+.2*channel+.3*batch).sin()+.02*time;auto observed=torch::ones(values.sizes(),torch::kBool);
  observed[0][1][7][3]=false;observed[1][2][11][2]=false;
  r::Input raw{values,observed,torch::tensor(c.channel_ids,torch::kInt64),torch::tensor({1000.,2000.,3000.},torch::kFloat64),c.sampling_interval};
  auto historical_raw=raw;historical_raw.data=raw.data.to(torch::kFloat32);
  const auto scaler=r::fit_configurable_spectral_scaler(historical_raw,c);const auto parent_input=r::transform_configurable_spectral_input(scaler,historical_raw,c);const auto scaler_id=scaler.identity();
  const r::Input input{raw.data.to(c.device),raw.observed.to(c.device),raw.channel_ids.to(c.device),raw.endpoints.to(c.device),raw.sampling_interval};
  const auto prepared=r::relative_change_parent_input(input,scaler,c);
  check(torch::equal(prepared.data,parent_input.data)&&torch::equal(prepared.observed,parent_input.observed)&&torch::equal(prepared.channel_ids,parent_input.channel_ids)&&torch::equal(prepared.endpoints,parent_input.endpoints),"exact historical raw-F32 before scaler parent view and metadata");
  {auto wrong=input;wrong.data=input.data.to(torch::kFloat32);rejects([&]{r::relative_change_targets(wrong,c);},"raw API rejects F32 observations");}
  relative_oracles(input,c);
  check_moments(input,c);check(r::relative_change_bank(input,c).counts[0][1][3].item<int64_t>()==29,"one missing tick removes both adjacent pairs");
  auto isolated=input;isolated.observed=torch::arange(32,input.data.options().dtype(torch::kInt64)).remainder(2).eq(0).view({1,1,32,1}).expand_as(input.observed).clone();
  const auto isolated_targets=r::relative_change_targets(isolated,c);check(!isolated_targets.valid.any().item<bool>()&&isolated_targets.values.eq(0).all().item<bool>(),"isolated samples never bridge gaps");check_moments(isolated,c);
  auto constant=input;constant.data=torch::zeros_like(input.data);constant.observed=torch::ones_like(input.observed);
  const auto constant_targets=r::relative_change_targets(constant,c);check(constant_targets.values.eq(0).all().item<bool>()&&constant_targets.valid.all().item<bool>(),"valid constant changes retain support");check_moments(constant,c);
  auto threshold=input;threshold.observed=torch::zeros_like(input.observed);threshold.observed.narrow(2,0,2).fill_(true);
  check(!r::relative_change_targets(threshold,c).valid.any().item<bool>(),"one adjacent pair below global minimum");threshold.observed.narrow(2,2,1).fill_(true);check_moments(threshold,c);
  const auto threshold_valid=r::relative_change_targets(threshold,c).valid.reshape({3,3,9,10});
  check(threshold_valid.narrow(3,0,4).all().item<bool>()&&!threshold_valid.narrow(3,4,6).any().item<bool>(),"two pairs support global and first original quarter only");
  const auto targets=r::relative_change_targets(input,c);r::RelativeBankStandardizer norm(c,"artificial-natural-TRAIN-v1",scaler_id);
  norm.add({targets.values.narrow(0,0,2),targets.valid.narrow(0,0,2)});norm.add({targets.values.narrow(0,2,1),targets.valid.narrow(0,2,1)});norm.freeze();norm.validate(c);check_normalizer(norm,targets);
  rejects([&]{norm.add(targets);},"frozen calibration cannot refit");rejects([&]{norm.freeze();},"normalizer freezes once");
  r::RelativeBankStandardizer zero(c,"artificial-zero-TRAIN",scaler_id);zero.add(constant_targets);zero.freeze();
  check(zero.scale.eq(1e-6).all().item<bool>()&&zero.mean.eq(0).all().item<bool>()&&zero.support.all().item<bool>(),"constant zero bank supported with floor");
  r::RelativeBankStandardizer absent(c,"artificial-absent-TRAIN",scaler_id);absent.add(isolated_targets);absent.freeze();
  check(!absent.support.any().item<bool>()&&absent.scale.eq(1).all().item<bool>()&&absent.mean.eq(0).all().item<bool>(),"unsupported neutral metadata");
  {
    auto tv=torch::ones_like(targets.values),valid=torch::ones_like(targets.valid);tv[0][0]=5.;tv[1][0]=5.;tv[2][0]=5.;
    valid[1][1]=false;valid[2][1]=false;valid.select(1,2).fill_(false);tv.masked_fill_(valid.logical_not(),std::numeric_limits<float>::quiet_NaN());
    r::RelativeChangeTargets artificial{tv,valid};r::RelativeBankStandardizer special(c,"artificial-support-TRAIN",scaler_id);special.add(artificial);special.freeze();special.validate(c);check_normalizer(special,artificial);
    check(special.scale[0].item<double>()==1e-6&&special.mean[0].item<double>()==5.&&special.support[0].item<bool>(),"nonzero constant uses centered floor");
    check(!special.support[1].item<bool>()&&!special.support[2].item<bool>()&&special.transform(artificial).select(1,1).eq(0).all().item<bool>(),"single-count and zero-count excluded");
    auto hidden=artificial;hidden.values=tv.clone();hidden.values.masked_fill_(valid.logical_not(),std::numeric_limits<float>::infinity());
    check(torch::equal(special.transform(artificial),special.transform(hidden)),"invalid nonfinite storage sanitized before conditioning");
    auto bad=norm.clone();bad.scale[0]*=2.;rejects([&]{bad.validate(c);},"inconsistent scale rejected");
    bad=norm.clone();bad.layout_identity+="-wrong";rejects([&]{bad.validate(c);},"wrong bank layout rejected");
    bad=norm.clone();bad.recipe_identity+="-wrong";rejects([&]{bad.validate(c);},"wrong normalization recipe rejected");
    bad=norm.clone();bad.floor=1e-5;rejects([&]{bad.validate(c);},"wrong floor rejected");
    bad=norm.clone();bad.counts=norm.counts.to(torch::kFloat32);rejects([&]{bad.validate(c);},"statistics dtype rejected");
    bad=norm.clone();bad.bank_family="raw-difference-v26";rejects([&]{bad.validate(c);},"wrong outer bank family rejected");
    bad=norm.clone();bad.increment_recipe+="-wrong";rejects([&]{bad.validate(c);},"wrong raw increment recipe rejected");
    bad=norm.clone();bad.source_identity="old-v26-TRAIN-source";rejects([&]{bad.validate(c);},"source prefix swap rejected");
    const std::string wrong="wrong";rejects([&]{norm.validate(c,&wrong);},"wrong TRAIN source rejected");rejects([&]{norm.validate(c,nullptr,&wrong);},"wrong input scaler rejected");
    const auto recipe_path=(evidence.path/"rejected-bank-recipe.pt").string();
    bad=norm.clone();bad.recipe_identity+="-wrong";torch::serialize::OutputArchive bad_archive;bad.save(bad_archive);bad_archive.save_to(recipe_path);
    rejects([&]{torch::serialize::InputArchive a;a.load_from(recipe_path);r::RelativeBankStandardizer::load(a,c);},"typed normalizer refuses a changed archived recipe");
    const auto tamper_path=(evidence.path/"rejected-bank-statistics.pt").string();
    bad=norm.clone();bad.mean[0]+=1.;torch::serialize::OutputArchive tampered;bad.save(tampered);tampered.save_to(tamper_path);
    rejects([&]{torch::serialize::InputArchive a;a.load_from(tamper_path);r::RelativeBankStandardizer::load(a,c);},"typed normalizer refuses inconsistent archived statistics");
    r::BankStandardizer old_norm(c,"artificial-v26-raw-difference-TRAIN",scaler_id);old_norm.add(r::conditioned_change_targets(parent_input,c));old_norm.freeze();
    const auto old_norm_path=(evidence.path/"rejected-v26-standardizer.pt").string();torch::serialize::OutputArchive old_archive;old_norm.save(old_archive);old_archive.save_to(old_norm_path);
    rejects([&]{torch::serialize::InputArchive a;a.load_from(old_norm_path);r::RelativeBankStandardizer::load(a,c);},"provided V26 standardizer archive rejected by relative typed family");
    bad=norm.clone();static_cast<r::BankStandardizer&>(bad)=old_norm;rejects([&]{bad.validate(c);},"old-family statistics cannot replace relative wrapper source");
  }
  torch::manual_seed(941001);r::ContentOrderedRelationModel parent(c);{torch::NoGradGuard ng;parent->relation_projection->weight.fill_(.003);}
  const auto parent_path=(evidence.path/"parent-v22.pt").string();r::save_content_ordered_relation_checkpoint(parent_path,parent,scaler,2048);
  auto parent_saved=r::load_content_ordered_relation_checkpoint(parent_path,c.device,&c);
  torch::manual_seed(942001);r::SeparatedChangeModel old(native36);oldtr::warm_start_level(old,parent_saved.model);
  const auto old_path=(evidence.path/"initialized-v24.pt").string();r::save_separated_change_checkpoint(old_path,old,scaler,0);auto old_saved=r::load_separated_change_checkpoint(old_path,c.device,&native36);
  r::ParentModulatedDecoderModel model(native36,norm);tr::warm_start_from_separated(model,old_saved.model);
  r::RelativeChangeModel legacy(native36,norm);r::relative_change_training::warm_start_from_separated(legacy,old_saved.model);
  const auto legacy_path=(evidence.path/"initialized-v28.pt").string();r::save_relative_change_checkpoint(legacy_path,legacy,scaler,0);
  check(model->waveform_gain.eq(0).all().item<bool>()&&model->waveform_gain.sizes()==torch::IntArrayRef({27,32}),"gain has literal-zero canonical C/F by parent-width shape");
  int64_t old_count=0,new_count=0;for(const auto &p:old_saved.model->parameters())old_count+=p.numel();for(const auto &p:model->parameters())new_count+=p.numel();
  check(new_count==old_count+864,"new parameter count is exactly old encoder plus864 gain parameters");
  {
    auto full=r::default_content_ordered_relation_config();full.channel_ids={7,41,99};
    r::RelativeBankStandardizer full_norm(full,"artificial-full-geometry-TRAIN",scaler_id);full_norm.add(targets);full_norm.freeze();
    r::ParentModulatedDecoderModel full_model(r::ParentModulatedDecoderConfig{full,4},full_norm);int64_t total=0,active=0,protected_count=0;
    for(const auto &p:full_model->named_parameters()){total+=p.value().numel();if(p.key().rfind("encoder.level.",0)==0)protected_count+=p.value().numel();else active+=p.value().numel();}
    check(total==298289&&active==15768&&protected_count==282521,"admitted C3/H128/F9/W64 total/updated/protected parameter counts");
  }
  const auto copied=model->encoder->named_parameters();for(const auto &p:old_saved.model->named_parameters())check(torch::equal(p.value(),copied[p.key()]),"all saved v24 init parameter tensors copied exactly");
  const auto untouched=norm.identity();norm.mean=norm.mean.clone();norm.mean[0]+=1.;check(model->normalizer().identity()==untouched,"model owns deep-cloned calibration");norm=model->normalizer().clone();
  auto optimizer=std::make_unique<torch::optim::AdamW>(model->parameters(),tr::content_training_adamw_options());tr::FrozenLevelGuard guard(model);
  {
    torch::NoGradGuard ng;model->eval();const auto native=model->encode(input,scaler).z_contextual_global;
    const auto legacy_out=legacy->encode(input,scaler),new_out=model->encode(input,scaler);
    check(torch::equal(new_out.z_contextual_global,legacy_out.z_contextual_global)&&torch::equal(new_out.sample_valid_mask,legacy_out.sample_valid_mask),"full initial36 and sample validity exactly equal unchanged v28 initial encoding");
    check(torch::equal(native.narrow(1,0,32),parent_saved.model->encode(parent_input).z_contextual_global),"initial parent prefix32 exact");
    check(torch::equal(model->decode(native,input.channel_ids),parent_saved.model->decode(native.narrow(1,0,32),input.channel_ids)),"zero decoder initial reconstruction exact");
    const auto conditioned=norm.transform(targets);check(torch::equal(native.narrow(1,32,4),conditioned.matmul(model->encoder->change_projection.t())),"suffix is conditioned-bank projection exactly");
    auto overflow=input;overflow.data=input.data.clone();overflow.data[0][0][0][0]=1.7e308;
    rejects([&]{model->encode(overflow,scaler);},"full model rejects observed finite F64 that overflows historical F32");
    auto alternate=scaler;alternate.mean=scaler.mean.clone()+.25;const auto frozen_identity=model->normalizer().identity();
    const auto alternate_native=model->encode(input,alternate).z_contextual_global;
    check(torch::equal(alternate_native.narrow(1,0,32),parent_saved.model->encode(r::relative_change_parent_input(input,alternate,c)).z_contextual_global)&&
        torch::equal(alternate_native.narrow(1,32,4),native.narrow(1,32,4))&&model->normalizer().identity()==frozen_identity,"alternate external input scaler preserves exact matching parent view and frozen relative branch");
    auto poison=input;poison.data=input.data.clone();poison.data.masked_fill_(input.observed.logical_not(),std::numeric_limits<float>::quiet_NaN());
    check(torch::equal(model->encode(poison,scaler).z_contextual_global,native),"naturally hidden NaN cannot affect native");
    const auto masks=r::configurable_spectral_make_training_mask(input.observed,c,942005);const auto context=r::context_deletion::make_plan(masks,input.channel_ids,c,942004,0,r::ContextDeletionRecipe::coordinate15_v1);
    const r::Input visible{input.data,context.visible,input.channel_ids,input.endpoints,input.sampling_interval};auto hidden=visible;hidden.data=input.data.clone();hidden.data.masked_fill_(context.visible.logical_not(),std::numeric_limits<float>::quiet_NaN());
    check(torch::equal(model->encode(visible,scaler).z_contextual_global,model->encode(hidden,scaler).z_contextual_global),"query/context-hidden poison cannot enter bank or parent");
    auto missing=input;missing.observed=torch::zeros_like(input.observed);check(model->encode(missing,scaler).z_contextual_global.eq(0).all().item<bool>(),"all-missing native literal0");
    auto partial_c=native36;partial_c.level.history_length=33;
    r::Input partial_input{torch::cat({input.data,torch::zeros({3,3,1,9},input.data.options())},2),torch::cat({input.observed,torch::zeros({3,3,1,9},input.observed.options())},2),input.channel_ids,input.endpoints,input.sampling_interval};
    r::RelativeBankStandardizer partial_norm(partial_c.level,"artificial-partial-TRAIN",scaler_id);partial_norm.add(r::relative_change_targets(partial_input,partial_c.level));partial_norm.freeze();r::ParentModulatedDecoderModel partial(partial_c,partial_norm);
    const auto out=partial->forward(partial_input,r::configurable_spectral_make_training_mask(partial_input.observed,partial_c.level,1101).hidden,scaler);
    check(out.reconstruction.sizes()==torch::IntArrayRef({3,3,33,9})&&torch::isfinite(out.loss).item<bool>(),"partial last patch preserved");check_moments(partial_input,partial_c.level);
  }
  {
    auto actual_gain=model->waveform_gain;model->waveform_gain=actual_gain.clone();
    rejects([&]{model->validate_state();},"unregistered gain holder replacement rejected");model->waveform_gain=actual_gain;model->validate_state();
    torch::NoGradGuard ng;const auto native=model->encode(input,scaler).z_contextual_global;
    model->encoder->change_decoder.copy_(torch::arange(model->encoder->change_decoder.numel(),model->encoder->change_decoder.options()).sin().reshape_as(model->encoder->change_decoder)*.03);
    legacy->encoder->change_decoder.copy_(model->encoder->change_decoder);
    check(torch::equal(model->decode(native,input.channel_ids),legacy->decode(native,input.channel_ids)),"A zero preserves complete old decoder arithmetic even with nonzero residual");
    model->waveform_gain.copy_(torch::arange(model->waveform_gain.numel(),model->waveform_gain.options()).cos().reshape_as(model->waveform_gain)*.02);
    check_decode_trace(model,native,input.channel_ids);
    const auto permutation=torch::tensor({2,0,1},torch::TensorOptions().device(c.device).dtype(torch::kInt64));
    check_decode_trace(model,native,input.channel_ids.index_select(0,permutation));
    check(torch::equal(model->decode(native,input.channel_ids.index_select(0,permutation)),model->decode(native,input.channel_ids).index_select(1,permutation)),"nonzero gain gather is exactly semantic equivariant for fixed native");
  }
  {
    const auto native=model->encode(input,scaler).z_contextual_global.detach().clone().set_requires_grad(true);
    const auto prediction=model->decode(native,input.channel_ids);prediction.sum().backward();
    check(native.grad().narrow(1,0,32).eq(0).all().item<bool>()&&native.grad().narrow(1,32,4).abs().sum().item<double>()>0,"gain parent context and parent decode detach exactly while change4 retains gradients");
    for(const auto &p:model->encoder->level->parameters())check(!p.grad().defined(),"nonzero gain cannot create parent parameter gradients");
    torch::NoGradGuard ng;model->waveform_gain.zero_();model->encoder->change_decoder.zero_();legacy->encoder->change_decoder.zero_();
    for(const auto &p:model->parameters())p.mutable_grad()=torch::Tensor();
    model->validate_state();
  }
  {
    const auto wrong_recipe=(evidence.path/"rejected-decoder-recipe.pt").string();torch::serialize::OutputArchive a;
    r::relative_change_checkpoint_detail::write_recipe(a);a.write("decoder_recipe",r::configurable_checkpoint_detail::text_tensor("wrong-parent-gain-recipe"),true);a.save_to(wrong_recipe);
    rejects([&]{torch::serialize::InputArchive in;in.load_from(wrong_recipe);r::parent_modulated_decoder_checkpoint_detail::read_recipe(in);},"wrong archived decoder recipe rejected");
  }
  const auto settings=(evidence.path/"v29-settings.conf").string();{std::ofstream f(settings);f<<"ENCODER { MODEL_TAG=RPB-v29; LEVEL_EMBEDDING_SIZE=32; EMBEDDING_SIZE=36; HISTORY_LENGTH=32; CHANNEL_IDS=7,41,99; }\n";}
  check(r::read_parent_modulated_decoder_settings(settings).change_width==4,"explicit v29 settings parser");
  tr::TrainingIdentity identity{"artificial-source","artificial-TRAIN","RPB-v29;raw-F64-relative-bank;waveform-only;parent2048"};tr::TrainingProgress progress{0,0,942002,942003,942004};
  const auto legacy_state=(evidence.path/"training-v28-0.pt").string();
  {torch::optim::AdamW legacy_optimizer(legacy->parameters(),tr::content_training_adamw_options());
    r::relative_change_training::save_training_state(legacy_state,legacy,scaler,legacy_optimizer,identity,progress);}
  rejects([&]{tr::load_training_state(legacy_state,native36,identity,scaler_id,norm.source_identity);},"v29 training reader rejects preserved v28 TRAIN kind");
  update(input,model,*optimizer,progress,guard,scaler);rejects([&]{tr::warm_start_from_separated(model,old_saved.model);},"trained branch cannot restart silently");
  const auto state=(evidence.path/"training-1.pt").string();tr::save_training_state(state,model,scaler,*optimizer,identity,progress);
  rejects([&]{r::relative_change_training::load_training_state(state,native36,identity,scaler_id,norm.source_identity);},"v28 training reader rejects new v29 TRAIN kind");
  check(model->waveform_gain.eq(0).all().item<bool>(),"saved first-update gain remains zero");
  rejects([&]{tr::save_training_state(state,model,scaler,*optimizer,identity,progress);},"training archive immutable");
  rejects([&]{tr::load_training_state(state,native40,identity,scaler_id,norm.source_identity);},"resume width mismatch");
  for(int field=0;field<3;++field){auto wrong=identity;if(field==0)wrong.input_source_identity+="-wrong";if(field==1)wrong.split_identity+="-wrong";if(field==2)wrong.recipe_identity+="-wrong";rejects([&]{tr::load_training_state(state,native36,wrong,scaler_id,norm.source_identity);},"resume identity mismatch");}
  rejects([&]{tr::load_training_state(state,native36,identity,scaler_id+"-wrong",norm.source_identity);},"resume input scaler mismatch");
  rejects([&]{tr::load_training_state(state,native36,identity,scaler_id,norm.source_identity+"-wrong");},"resume normalizer source mismatch");
  const std::string wrong_id=norm.identity()+"-wrong";rejects([&]{tr::load_training_state(state,native36,identity,scaler_id,norm.source_identity,&wrong_id);},"resume normalizer content mismatch");
  auto restored=tr::load_training_state(state,native36,identity,scaler_id,norm.source_identity);tr::FrozenLevelGuard resumed_guard(restored.model);
  same_model(model,restored.model);same_optimizer(model,*optimizer,restored.model,*restored.optimizer);
  check(restored.progress.completed_updates==progress.completed_updates&&restored.progress.attempted_batches==progress.attempted_batches&&restored.progress.row_seed==progress.row_seed&&restored.progress.mask_seed==progress.mask_seed&&restored.progress.context_seed==progress.context_seed,"all counters and seeds exact");
  const auto loss=update(input,model,*optimizer,progress,guard,scaler),resumed=update(input,restored.model,*restored.optimizer,restored.progress,resumed_guard,scaler);
  check(torch::equal(loss,resumed),"next resumed waveform loss exact");same_model(model,restored.model);same_optimizer(model,*optimizer,restored.model,*restored.optimizer);
  {torch::NoGradGuard ng;check_decode_trace(model,model->encode(input,scaler).z_contextual_global,input.channel_ids);}
  {
    torch::NoGradGuard ng;model->eval();const auto native=model->encode(input,scaler).z_contextual_global;
    check(torch::equal(native.narrow(1,0,32),parent_saved.model->encode(parent_input).z_contextual_global),"trained prefix remains exact");
    const auto permutation=torch::tensor({2,0,1},torch::TensorOptions().device(c.device).dtype(torch::kInt64));
    r::Input shuffled{input.data.index_select(1,permutation),input.observed.index_select(1,permutation),input.channel_ids.index_select(0,permutation),input.endpoints,input.sampling_interval};
    const auto shuffled_native=model->encode(shuffled,scaler).z_contextual_global;
    check(torch::equal(native.narrow(1,32,4),shuffled_native.narrow(1,32,4)),"conditioned suffix semantic permutation exact");
    check(torch::equal(shuffled_native.narrow(1,0,32),parent_saved.model->encode(r::relative_change_parent_input(shuffled,scaler,c)).z_contextual_global),"permuted prefix exact same-layout parent");
    check(torch::allclose(model->decode(shuffled_native,shuffled.channel_ids),model->decode(native,input.channel_ids).index_select(1,permutation),1e-5,1e-6),"inherited decoder permutation tolerance");check_moments(shuffled,c);
  }
  const auto inference=(evidence.path/"inference-2.pt").string();r::save_parent_modulated_decoder_checkpoint(inference,model,scaler,2);
  auto loaded=r::load_parent_modulated_decoder_checkpoint(inference,c.device,&native36,&scaler_id,&norm.source_identity);same_model(model,loaded.model);
  {torch::NoGradGuard ng;check(torch::equal(model->encode(input,scaler).z_contextual_global,loaded.model->encode(input,scaler).z_contextual_global),"typed inference native reload exact");}
  rejects([&]{r::save_parent_modulated_decoder_checkpoint(inference,model,scaler,2);},"inference archive immutable");
  rejects([&]{r::load_parent_modulated_decoder_checkpoint(inference,c.device,&native40);},"inference width mismatch");
  rejects([&]{r::load_parent_modulated_decoder_checkpoint(inference,c.device,&native36,nullptr,nullptr,&wrong_id);},"inference normalizer content mismatch");
  rejects([&]{r::load_parent_modulated_decoder_checkpoint(old_path,c.device,&native36);},"new reader rejects old v24 kind");
  rejects([&]{r::load_parent_modulated_decoder_checkpoint(legacy_path,c.device,&native36);},"new reader rejects v28 model kind despite accepted unchanged v28 bank family");
  rejects([&]{r::load_relative_change_checkpoint(inference,c.device,&native36);},"v28 reader rejects v29 kind");
  rejects([&]{tr::load_training_state(legacy_path,native36,identity,scaler_id,norm.source_identity);},"v29 training reader rejects v28 inference kind");
  rejects([&]{r::load_separated_change_checkpoint(inference,c.device,&native36);},"old reader rejects new v29 kind");
  rejects([&]{r::load_parent_modulated_decoder_checkpoint(state,c.device,&native36);},"inference rejects training kind");
  rejects([&]{tr::load_training_state(inference,native36,identity,scaler_id,norm.source_identity);},"training rejects inference kind");
  {auto alias=model->normalizer().mean;alias[0]+=1.;}rejects([&]{model->validate_state();},"live frozen calibration mutation detected");
  std::cout<<"PASS RPB-v29 CUDA admission: unchanged genuine-F64 relative bank and exact initial v28 native36, canonical C/F gain/scalar oracle/broadcast/semantic gather and detached parent, exact zero-gain old decoder and initial parent reconstruction, actual first decoder then projection/gain learning with first zero gain/moments, preserved masks/prefix, new immutable typed inference/TRAIN, v28 kind refusal and exact next-update resume\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
