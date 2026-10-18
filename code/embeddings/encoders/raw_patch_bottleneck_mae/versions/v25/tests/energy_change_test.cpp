// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Artificial CUDA engineering only. No historical encoder training or quality.
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_training_state.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_settings.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include <torch/cuda.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

namespace {
namespace r=embedding::encoders::raw_patch_bottleneck_mae;
namespace tr=r::energy_change_training;
namespace oldtr=r::separated_change_training;
void check(bool value,const char *why){TORCH_CHECK(value,"[energy CUDA admission] ",why);}
template<class F>void rejects(F operation,const char *why){bool caught=false;try{operation();}catch(const std::exception&){caught=true;}check(caught,why);}
void near(double actual,double expected,const char *why,double tolerance=2e-6){check(std::isfinite(actual)&&std::abs(actual-expected)<=tolerance*std::max(1.,std::abs(expected)),why);}

struct Evidence {
  std::filesystem::path path;
  Evidence(){
    std::filesystem::create_directories("output/engineering/energy-change-fixture");
    auto pattern=std::filesystem::absolute("output/engineering/energy-change-fixture/run-XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
    check(::mkdtemp(name.data()),"exclusive fixture evidence directory");path=name.data();
    std::cout<<"ENERGY_FIXTURE_DIRECTORY="<<path.string()<<std::endl;
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
        const auto d=x[b][physical][t+1][f]-x[b][physical][t][f];
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
  const auto actual=r::energy_change_moment_targets(input,c);const auto expected=moment_oracle(input,c);
  check(actual.values.is_cuda()&&actual.valid.is_cuda(),"moment targets and support are actual CUDA tensors");
  check(torch::equal(actual.values,r::consecutive_change_bank(input,c).values),"target values are the unchanged existing bank exactly");
  check(torch::equal(actual.valid.to(torch::kCPU),expected.valid),"scalar adjacent-pair and original-quarter support oracle");
  check(torch::allclose(actual.values.to(torch::kCPU).to(torch::kFloat64),expected.values,2e-6,2e-7),"scalar signed/global RMS/normalized-quarter oracle");
}
void check_normalizer(const r::MomentRmsScaler &scaler,const r::EnergyChangeMomentTargets &targets){
  const auto values=targets.values.to(torch::kCPU).to(torch::kFloat64).contiguous();
  const auto valid=targets.valid.to(torch::kCPU).contiguous();
  const auto x=values.accessor<double,2>();const auto mask=valid.accessor<bool,2>();
  const auto counts=scaler.counts.to(torch::kCPU).contiguous();
  const auto squares=scaler.sumsq.to(torch::kCPU).to(torch::kFloat64).contiguous();
  const auto scales=scaler.scales.to(torch::kCPU).to(torch::kFloat64).contiguous();
  const auto support=scaler.support.to(torch::kCPU).contiguous();
  for(int64_t j=0;j<values.size(1);++j){
    int64_t n=0;double sumsq=0;for(int64_t b=0;b<values.size(0);++b)if(mask[b][j]){++n;sumsq+=x[b][j]*x[b][j];}
    check(counts[j].item<int64_t>()==n,"streaming calibration counts exact");
    near(squares[j].item<double>(),sumsq,"streaming uncentered sum of squares oracle",1e-12);
    check(support[j].item<bool>()==(n>0),"unsupported calibration coordinate excluded explicitly");
    near(scales[j].item<double>(),n?std::max(std::sqrt(sumsq/n),1e-6):1.,"TRAIN RMS scale/floor and unsupported neutral scale",1e-12);
  }
}

double huber(double e){const auto a=std::abs(e);return a<=1?.5*e*e:a-.5;}
struct LossOracle{double loss=0,signed_loss=0,energy_loss=0;int64_t examples=0;};
LossOracle loss_oracle(const torch::Tensor &prediction,const r::EnergyChangeMomentTargets &target,const r::MomentRmsScaler &norm){
  const auto pred=prediction.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
  const auto values=target.values.to(torch::kCPU).to(torch::kFloat64).contiguous();
  const auto valid=target.valid.to(torch::kCPU).contiguous();
  const auto scale=norm.scales.to(torch::kCPU).to(torch::kFloat64).contiguous();
  const auto supported=norm.support.to(torch::kCPU).contiguous();
  const auto p=pred.accessor<double,2>(),v=values.accessor<double,2>();const auto m=valid.accessor<bool,2>();
  LossOracle result;
  for(int64_t b=0;b<pred.size(0);++b){double signed_sum=0,energy_sum=0;int64_t signed_n=0,energy_n=0;
    for(int64_t j=0;j<pred.size(1);++j)if(m[b][j]&&supported[j].item<bool>()){
      const auto error=(p[b][j]-v[b][j])/scale[j].item<double>();
      if(j%2){energy_sum+=huber(error);++energy_n;}else{signed_sum+=huber(error);++signed_n;}
    }
    if(!signed_n||!energy_n)continue;
    result.signed_loss+=signed_sum/signed_n;result.energy_loss+=energy_sum/energy_n;++result.examples;
  }
  if(result.examples){result.signed_loss/=result.examples;result.energy_loss/=result.examples;result.loss=.5*(result.signed_loss+result.energy_loss);}
  return result;
}

void protected_state(const r::EnergyChangeModel &model,const torch::optim::AdamW &optimizer,const oldtr::FrozenLevelGuard &guard){
  guard.validate(model->encoder,optimizer);
  for(const auto &p:model->encoder->level->parameters()){
    check(p.is_cuda()&&!p.grad().defined(),"protected parent CUDA parameters have undefined gradients");
    check(!optimizer.state().count(p.unsafeGetTensorImpl()),"protected parent has no AdamW moments");
  }
}
struct Step{torch::Tensor total,wave,signed_loss,energy_loss;};
Step update(const r::Input &input,r::EnergyChangeModel &model,const r::MomentRmsScaler &norm,
    torch::optim::AdamW &optimizer,tr::TrainingProgress &progress,const oldtr::FrozenLevelGuard &guard){
  const auto &c=model->config().level;
  const auto masks=r::configurable_spectral_make_training_mask(input.observed,c,tr::training_counter_seed(progress.mask_seed,progress.attempted_batches));
  const auto context=r::context_deletion::make_plan(masks,input.channel_ids,c,progress.context_seed,progress.attempted_batches,r::ContextDeletionRecipe::coordinate15_v1);
  const r::Input visible{torch::where(context.visible,input.data,torch::zeros_like(input.data)),context.visible,input.channel_ids,input.endpoints,input.sampling_interval};
  model->train();optimizer.zero_grad();
  const auto native=r::energy_change_native(model->encode(visible),model->config());
  const auto reconstruction=model->decode(native,input.channel_ids);
  const auto wave=r::hierarchical_huber(reconstruction,input.data.detach(),masks.target,masks.eligible_channels,c.huber_delta);
  const auto moments=r::energy_change_moment_loss(model,visible,native,norm);
  const auto total=wave.loss+.1*moments.loss;
  check(input.data.is_cuda()&&visible.observed.is_cuda()&&native.is_cuda()&&reconstruction.is_cuda()&&total.is_cuda(),"actual CUDA parameters/input/native/loss");
  check(torch::isfinite(total).item<bool>()&&moments.eligible_example_count>0,"finite CUDA objective with auxiliary support");
  const auto target=r::energy_change_moment_targets(visible,c);
  const auto predicted=torch::matmul(native.narrow(1,c.export_width,model->config().change_width),model->moment_decoder.transpose(0,1));
  const auto oracle=loss_oracle(predicted,target,norm);
  near(moments.loss.item<double>(),oracle.loss,"balanced auxiliary scalar oracle");
  near(moments.signed_loss.item<double>(),oracle.signed_loss,"signed-group scalar oracle");
  near(moments.energy_loss.item<double>(),oracle.energy_loss,"energy-group scalar oracle");
  near(total.item<double>(),wave.loss.item<double>()+.1*.5*(moments.signed_loss.item<double>()+moments.energy_loss.item<double>()),"fixed waveform plus weighted balanced moment loss");
  total.backward();
  const auto pg=model->encoder->change_projection.grad(),dg=model->encoder->change_decoder.grad(),ag=model->moment_decoder.grad();
  for(const auto &g:{pg,dg,ag})check(g.defined()&&g.is_cuda()&&torch::isfinite(g).all().item<bool>(),"all three branch gradients are finite CUDA");
  check(dg.abs().sum().item<double>()>0&&ag.abs().sum().item<double>()>0,"waveform and moment decoders both receive nonzero CUDA gradients");
  check(progress.completed_updates?pg.abs().sum().item<double>()>0:pg.eq(0).all().item<bool>(),"first step opens zero decoders; second step learns projection");
  for(const auto &p:model->encoder->level->parameters())check(!p.grad().defined(),"parent structurally absent from both loss graphs");
  guard.discard_gradients(model->encoder,optimizer);
  const auto projection=model->encoder->change_projection.detach().clone(),decoder=model->encoder->change_decoder.detach().clone(),auxiliary=model->moment_decoder.detach().clone();
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer.step();
  check(!torch::equal(decoder,model->encoder->change_decoder)&&!torch::equal(auxiliary,model->moment_decoder),"actual CUDA weights of both decoders moved");
  if(progress.completed_updates)check(!torch::equal(projection,model->encoder->change_projection),"actual projection CUDA weights moved after opening");
  ++progress.completed_updates;++progress.attempted_batches;protected_state(model,optimizer,guard);
  return{total.detach().clone(),wave.loss.detach().clone(),moments.signed_loss.detach().clone(),moments.energy_loss.detach().clone()};
}

void same_optimizer(const r::EnergyChangeModel &a,const torch::optim::AdamW &oa,const r::EnergyChangeModel &b,const torch::optim::AdamW &ob){
  const auto ap=a->named_parameters(),bp=b->named_parameters();
  check(oa.state().size()==ob.state().size(),"resumed optimizer state count exact");
  for(const auto &p:ap){
    const auto ai=oa.state().find(p.value().unsafeGetTensorImpl()),bi=ob.state().find(bp[p.key()].unsafeGetTensorImpl());
    check((ai==oa.state().end())==(bi==ob.state().end()),"resumed optimizer support exact");
    if(ai==oa.state().end())continue;
    const auto *av=dynamic_cast<const torch::optim::AdamWParamState*>(ai->second.get());
    const auto *bv=dynamic_cast<const torch::optim::AdamWParamState*>(bi->second.get());
    check(av&&bv&&av->step()==bv->step()&&torch::equal(av->exp_avg(),bv->exp_avg())&&torch::equal(av->exp_avg_sq(),bv->exp_avg_sq()),"resumed AdamW steps and both moment tensors exact");
  }
}
} // namespace

int main() try {
  check(torch::cuda::is_available(),"CUDA required");torch::set_num_threads(1);torch::set_num_interop_threads(1);
  auto c=r::default_content_ordered_relation_config();c.channel_ids={7,41,99};c.history_length=32;c.encoder_width=32;c.num_layers=1;c.feedforward_width=64;c.decoder_hidden_width=64;
  const r::EnergyChangeConfig native36{c,4},native40{c,8};
  check(r::separated_change_native_width(native36)==36,"native36 remains exact v24 forward geometry");
  Evidence evidence;
  torch::manual_seed(941001);r::ContentOrderedRelationModel parent(c);
  {torch::NoGradGuard guard;parent->relation_projection->weight.fill_(.003);}
  const auto time=torch::arange(32,torch::kFloat64).view({1,1,32,1}),feature=torch::arange(9,torch::kFloat64).view({1,1,1,9});
  const auto channel=torch::arange(3,torch::kFloat64).view({1,3,1,1}),batch=torch::arange(3,torch::kFloat64).view({3,1,1,1});
  auto values=(.13*time+.11*feature+.2*channel+.3*batch).sin()+.02*time;
  auto observed=torch::ones(values.sizes(),torch::kBool);observed[0][1][7][3]=false;observed[1][2][11][2]=false;
  r::Input raw{values,observed,torch::tensor(c.channel_ids,torch::kInt64),torch::tensor({1000.,2000.,3000.},torch::kFloat64),c.sampling_interval};
  const auto scaler=r::fit_configurable_spectral_scaler(raw,c);const auto input=r::transform_configurable_spectral_input(scaler,raw,c);
  check_moments(input,c);
  check(r::consecutive_change_bank(input,c).counts[0][1][3].item<int64_t>()==29,"one missing tick removes both adjacent pairs");
  auto isolated=input;isolated.observed=torch::arange(32,input.data.options().dtype(torch::kInt64)).remainder(2).eq(0).view({1,1,32,1}).expand_as(input.observed).clone();
  const auto isolated_targets=r::energy_change_moment_targets(isolated,c);check(isolated_targets.values.eq(0).all().item<bool>()&&!isolated_targets.valid.any().item<bool>(),"isolated endpoints never bridge gaps or become valid zeros");check_moments(isolated,c);
  auto constant=input;constant.data=torch::zeros_like(input.data);constant.observed=torch::ones_like(input.observed);
  const auto constant_targets=r::energy_change_moment_targets(constant,c);check(constant_targets.values.eq(0).all().item<bool>()&&constant_targets.valid.all().item<bool>(),"observed constant changes are valid zero-energy targets");check_moments(constant,c);
  auto threshold=input;threshold.observed=torch::zeros_like(input.observed);threshold.observed.narrow(2,0,2).fill_(true);
  check(!r::energy_change_moment_targets(threshold,c).valid.any().item<bool>(),"one adjacent pair fails global minimum2");
  threshold.observed.narrow(2,2,1).fill_(true);check_moments(threshold,c);
  const auto threshold_valid=r::energy_change_moment_targets(threshold,c).valid.reshape({3,3,9,10});
  check(threshold_valid.narrow(3,0,4).all().item<bool>()&&!threshold_valid.narrow(3,4,6).any().item<bool>(),"exactly two pairs validate global and first quarter only");

  const auto targets=r::energy_change_moment_targets(input,c);
  r::MomentRmsScaler norm(c,"artificial-TRAIN-natural-mask-v1");
  norm.add({targets.values.narrow(0,0,2),targets.valid.narrow(0,0,2)});norm.add({targets.values.narrow(0,2,1),targets.valid.narrow(0,2,1)});norm.freeze();norm.validate(c);check_normalizer(norm,targets);
  rejects([&]{norm.add(targets);},"frozen RMS calibration refuses further data");
  r::MomentRmsScaler zero_norm(c,"artificial-zero-TRAIN");zero_norm.add(constant_targets);zero_norm.freeze();
  check(zero_norm.scales.eq(1e-6).all().item<bool>(),"valid zero-energy calibration uses declared floor");
  r::MomentRmsScaler absent_norm(c,"artificial-unsupported-TRAIN");absent_norm.add(isolated_targets);absent_norm.freeze();
  check(!absent_norm.support.any().item<bool>()&&absent_norm.counts.eq(0).all().item<bool>(),"unsupported TRAIN coordinates remain explicitly inactive");
  {
    // Deliberately unequal artificial group supports test the reduction itself;
    // real moment pairs naturally have matching signed/energy eligibility.
    auto custom_values=torch::ones_like(targets.values),custom_valid=torch::zeros_like(targets.valid);
    custom_valid[0][0]=true;custom_valid[0][2]=true;custom_valid[0][1]=true;
    custom_valid[1][0]=true;custom_valid[1][1]=true;custom_valid[1][3]=true;custom_valid[1][5]=true;
    custom_valid[2][0]=true; // This example has no energy target and is excluded.
    const r::EnergyChangeMomentTargets custom{custom_values,custom_valid};
    r::MomentRmsScaler unit_norm(c,"artificial-group-reduction-TRAIN");unit_norm.add(custom);unit_norm.freeze();
    auto predicted=torch::zeros_like(custom_values);predicted[0][0]=3.;predicted[0][2]=2.;predicted[0][1]=4.;
    predicted[1][0]=1.5;predicted[1][1]=0.;predicted[1][3]=2.;predicted[1][5]=3.;predicted[2][0]=10000.;predicted.set_requires_grad(true);
    const auto reduced=r::energy_change_balanced_loss(predicted,custom,unit_norm);
    const auto expected=loss_oracle(predicted,custom,unit_norm);
    check(reduced.eligible_example_count==2&&expected.examples==2,"both groups required per example despite unequal coordinate counts");
    near(reduced.signed_loss.item<double>(),expected.signed_loss,"unequal signed-group scalar oracle");near(reduced.energy_loss.item<double>(),expected.energy_loss,"unequal energy-group scalar oracle");near(reduced.loss.item<double>(),expected.loss,"equal group weighting instead of pooled coordinate weighting");
    reduced.loss.backward();check(predicted.grad().is_cuda()&&predicted.grad()[2].eq(0).all().item<bool>(),"one-group-only example contributes no gradient");
    auto bad=norm;bad.scales=norm.scales.clone();bad.scales[0]*=2.;rejects([&]{bad.validate(c);},"normalizer rejects scales inconsistent with sufficient statistics");
    bad=norm;bad.layout_identity+="-wrong";rejects([&]{bad.validate(c);},"normalizer rejects wrong moment layout");
    bad=norm;bad.floor=1e-5;rejects([&]{bad.validate(c);},"normalizer rejects changed floor recipe");
    bad=norm;bad.counts=norm.counts.to(torch::kFloat32);rejects([&]{bad.validate(c);},"normalizer rejects wrong sufficient-statistics dtype");
  }

  const auto parent_file=(evidence.path/"parent-v22.pt").string();r::save_content_ordered_relation_checkpoint(parent_file,parent,scaler,2048);
  auto loaded_parent=r::load_content_ordered_relation_checkpoint(parent_file,c.device,&c);
  torch::manual_seed(942001);r::SeparatedChangeModel baseline(native36);oldtr::warm_start_level(baseline,loaded_parent.model);
  const auto old_file=(evidence.path/"initialized-v24.pt").string();r::save_separated_change_checkpoint(old_file,baseline,scaler,0);
  auto loaded_old=r::load_separated_change_checkpoint(old_file,c.device,&native36);
  r::EnergyChangeModel model(native36);tr::warm_start_from_separated(model,loaded_old.model);
  auto optimizer=std::make_unique<torch::optim::AdamW>(model->parameters(),tr::content_training_adamw_options());oldtr::FrozenLevelGuard guard(model->encoder);protected_state(model,*optimizer,guard);
  check(model->moment_decoder.sizes()==torch::IntArrayRef({270,4})&&model->moment_decoder.eq(0).all().item<bool>(),"new1080-parameter auxiliary decoder starts at literal zero");
  {
    torch::NoGradGuard no_grad;model->eval();
    const auto native=model->encode(input).z_contextual_global,old=loaded_old.model->encode(input).z_contextual_global;
    check(torch::equal(native,old)&&torch::equal(native.narrow(1,0,32),loaded_parent.model->encode(input).z_contextual_global),"initialized entire native36 and protected prefix32 exactly match saved controls");
    check(torch::equal(model->decode(native,input.channel_ids),loaded_parent.model->decode(native.narrow(1,0,32),input.channel_ids)),"zero residual retains parent reconstruction exactly");
    auto poisoned=input;poisoned.data=input.data.clone();poisoned.data.masked_fill_(input.observed.logical_not(),std::numeric_limits<float>::quiet_NaN());
    check(torch::equal(model->encode(poisoned).z_contextual_global,native),"unobserved NaN storage cannot affect native");
    check(torch::equal(r::energy_change_moment_targets(poisoned,c).values,targets.values),"unobserved NaN storage cannot affect auxiliary targets");
    const auto masks=r::configurable_spectral_make_training_mask(input.observed,c,942005);
    const auto context=r::context_deletion::make_plan(masks,input.channel_ids,c,942004,0,r::ContextDeletionRecipe::coordinate15_v1);
    const r::Input visible{input.data,context.visible,input.channel_ids,input.endpoints,input.sampling_interval};
    auto hidden_poison=visible;hidden_poison.data=input.data.clone();hidden_poison.data.masked_fill_(context.visible.logical_not(),std::numeric_limits<float>::quiet_NaN());
    const auto clean_native=model->encode(visible).z_contextual_global,poison_native=model->encode(hidden_poison).z_contextual_global;
    const auto clean_target=r::energy_change_moment_targets(visible,c),poison_target=r::energy_change_moment_targets(hidden_poison,c);
    const auto clean_loss=r::energy_change_moment_loss(model,visible,clean_native,norm),poison_loss=r::energy_change_moment_loss(model,hidden_poison,poison_native,norm);
    check(torch::equal(clean_native,poison_native)&&torch::equal(clean_target.values,poison_target.values)&&torch::equal(clean_target.valid,poison_target.valid)&&torch::equal(clean_loss.loss,poison_loss.loss),"hidden/context-deleted poison cannot alter native, moment targets, support or auxiliary loss");
    auto missing=input;missing.observed=torch::zeros_like(input.observed);check(model->encode(missing).z_contextual_global.eq(0).all().item<bool>(),"all-missing native remains zero");
    auto partial_config=native36;partial_config.level.history_length=33;r::EnergyChangeModel partial(partial_config);
    r::Input partial_input{torch::cat({input.data,torch::zeros({3,3,1,9},input.data.options())},2),torch::cat({input.observed,torch::zeros({3,3,1,9},input.observed.options())},2),input.channel_ids,input.endpoints,input.sampling_interval};
    const auto output=partial->forward(partial_input,r::configurable_spectral_make_training_mask(partial_input.observed,partial_config.level,1101).hidden);
    check(output.reconstruction.sizes()==torch::IntArrayRef({3,3,33,9})&&torch::isfinite(output.loss).item<bool>(),"partial patch history retained");check_moments(partial_input,partial_config.level);
  }
  {
    optimizer->zero_grad();const auto native=model->encode(isolated).z_contextual_global;
    const auto empty=r::energy_change_moment_loss(model,isolated,native,norm);
    check(empty.eligible_example_count==0&&empty.loss.requires_grad()&&empty.loss.item<double>()==0.,"zero-eligible auxiliary remains differentiable zero");
    empty.loss.backward();check(model->moment_decoder.grad().defined()&&model->moment_decoder.grad().eq(0).all().item<bool>(),"zero-support auxiliary backward has finite zero gradients");optimizer->zero_grad();
  }
  const auto settings=(evidence.path/"v25-settings.conf").string();{std::ofstream f(settings);f<<"ENCODER { MODEL_TAG=RPB-v25; LEVEL_EMBEDDING_SIZE=32; EMBEDDING_SIZE=36; HISTORY_LENGTH=32; CHANNEL_IDS=7,41,99; }\n";}
  const auto parsed=r::read_energy_change_settings(settings);check(parsed.change_width==4&&parsed.level.export_width==32,"typed v25 settings preserve native36 geometry");
  tr::TrainingIdentity identity{"artificial-source","artificial-TRAIN-split","RPB-v25;wave-plus-balanced-moment.1;parent2048"};tr::TrainingProgress progress{0,0,942002,942003,942004};
  update(input,model,norm,*optimizer,progress,guard);
  rejects([&]{tr::warm_start_from_separated(model,loaded_old.model);},"trained auxiliary cannot silently restart");
  const auto state=(evidence.path/"training-1.pt").string();tr::save_training_state(state,model,scaler,norm,*optimizer,identity,progress);
  rejects([&]{tr::save_training_state(state,model,scaler,norm,*optimizer,identity,progress);},"immutable typed training state refuses overwrite");
  rejects([&]{tr::load_training_state(state,native40,identity,scaler.identity(),norm.source_identity);},"native40 rejects native36 energy training state");
  for(int field=0;field<3;++field){auto wrong=identity;if(field==0)wrong.input_source_identity+="-wrong";if(field==1)wrong.split_identity+="-wrong";if(field==2)wrong.recipe_identity+="-wrong";rejects([&]{tr::load_training_state(state,native36,wrong,scaler.identity(),norm.source_identity);},"typed resume rejects wrong source/split/recipe");}
  rejects([&]{tr::load_training_state(state,native36,identity,scaler.identity()+"-wrong",norm.source_identity);},"typed resume rejects wrong input TRAIN scaler");
  rejects([&]{tr::load_training_state(state,native36,identity,scaler.identity(),norm.source_identity+"-wrong");},"typed resume rejects wrong moment TRAIN normalization identity");
  auto restored=tr::load_training_state(state,native36,identity,scaler.identity(),norm.source_identity);oldtr::FrozenLevelGuard resumed_guard(restored.model->encoder);
  check(restored.progress.completed_updates==progress.completed_updates&&restored.progress.attempted_batches==progress.attempted_batches&&restored.progress.row_seed==progress.row_seed&&restored.progress.mask_seed==progress.mask_seed&&restored.progress.context_seed==progress.context_seed,"typed resume preserves every counter and seed exactly");
  check(torch::equal(norm.counts,restored.moment_scaler.counts)&&torch::equal(norm.sumsq,restored.moment_scaler.sumsq)&&torch::equal(norm.scales,restored.moment_scaler.scales)&&torch::equal(norm.support,restored.moment_scaler.support),"typed resume preserves all frozen moment calibration state");same_optimizer(model,*optimizer,restored.model,*restored.optimizer);
  const auto next=update(input,model,norm,*optimizer,progress,guard),resumed=update(input,restored.model,restored.moment_scaler,*restored.optimizer,restored.progress,resumed_guard);
  check(torch::equal(next.total,resumed.total)&&torch::equal(next.wave,resumed.wave)&&torch::equal(next.signed_loss,resumed.signed_loss)&&torch::equal(next.energy_loss,resumed.energy_loss),"resumed next CUDA total and all component losses are exact");
  const auto other=restored.model->named_parameters();for(const auto &p:model->named_parameters())check(torch::equal(p.value(),other[p.key()]),"resumed next complete parameter state exact including moment decoder");
  const auto other_buffers=restored.model->named_buffers();for(const auto &b:model->named_buffers())check(torch::equal(b.value(),other_buffers[b.key()]),"resumed next complete buffer state exact");same_optimizer(model,*optimizer,restored.model,*restored.optimizer);
  {
    torch::NoGradGuard no_grad;model->eval();const auto native=model->encode(input).z_contextual_global;
    check(torch::equal(native.narrow(1,0,32),loaded_parent.model->encode(input).z_contextual_global),"trained protected parent prefix stays exact");
    const auto permutation=torch::tensor({2,0,1},torch::TensorOptions().device(c.device).dtype(torch::kInt64));
    r::Input shuffled{input.data.index_select(1,permutation),input.observed.index_select(1,permutation),input.channel_ids.index_select(0,permutation),input.endpoints,input.sampling_interval};
    const auto shuffled_native=model->encode(shuffled).z_contextual_global;
    check(torch::equal(shuffled_native.narrow(1,0,32),loaded_parent.model->encode(shuffled).z_contextual_global),"permuted protected prefix equals same-layout parent exactly");
    check(torch::equal(native.narrow(1,32,4),shuffled_native.narrow(1,32,4)),"change coordinates exactly invariant to physical channel permutation");
    check(torch::allclose(model->decode(shuffled_native,shuffled.channel_ids),model->decode(native,input.channel_ids).index_select(1,permutation),1e-5,1e-6),"decoder follows inherited semantic permutation tolerance");check_moments(shuffled,c);
  }
  const auto inference=(evidence.path/"inference-2.pt").string();r::save_energy_change_checkpoint(inference,model,scaler,norm,2);auto reloaded=r::load_energy_change_checkpoint(inference,c.device,&native36);
  {torch::NoGradGuard no_grad;check(torch::equal(model->encode(input).z_contextual_global,reloaded.model->encode(input).z_contextual_global),"new typed immutable inference native reload exact");}
  check(torch::equal(reloaded.moment_scaler.counts,norm.counts)&&torch::equal(reloaded.moment_scaler.sumsq,norm.sumsq)&&torch::equal(reloaded.moment_scaler.scales,norm.scales)&&torch::equal(reloaded.moment_scaler.support,norm.support),"typed inference preserves moment normalization metadata");
  rejects([&]{r::save_energy_change_checkpoint(inference,model,scaler,norm,2);},"immutable inference refuses overwrite");
  {auto bad=norm;bad.layout_identity+="-wrong";rejects([&]{r::save_energy_change_checkpoint((evidence.path/"rejected-normalizer.pt").string(),model,scaler,bad,2);},"typed inference refuses invalid normalization recipe");}
  rejects([&]{r::load_energy_change_checkpoint(inference,c.device,&native40);},"native40 geometry rejects native36 inference");
  const auto wrong_scaler=scaler.identity()+"-wrong",wrong_moment_source=norm.source_identity+"-wrong";
  rejects([&]{r::load_energy_change_checkpoint(inference,c.device,&native36,&wrong_scaler);},"typed inference rejects wrong input TRAIN scaler identity");
  rejects([&]{r::load_energy_change_checkpoint(inference,c.device,&native36,nullptr,&wrong_moment_source);},"typed inference rejects wrong auxiliary TRAIN normalizer identity");
  rejects([&]{r::load_energy_change_checkpoint(old_file,c.device,&native36);},"v25 reader rejects old v24 kind");
  rejects([&]{r::load_separated_change_checkpoint(inference,c.device,&native36);},"old v24 reader rejects new v25 kind");
  rejects([&]{r::load_energy_change_checkpoint(state,c.device,&native36);},"inference reader rejects new training state");
  rejects([&]{tr::load_training_state(inference,native36,identity,scaler.identity(),norm.source_identity);},"training reader rejects inference kind");
  {torch::NoGradGuard no_grad;model->encoder->level->relation_projection->weight.add_(.001);}rejects([&]{guard.validate(model->encoder,*optimizer);},"protected parent mutation detected");
  std::cout<<"PASS RPB-v25 CUDA admission: exact initialized native36 and frozen prefix32, scalar moment/support/RMS/group-loss oracles, poison/gap/partial-history/semantic order, first zero-decoder then projection learning, protected parent no moments, immutable typed inference/state and exact all-component next-update resume\n";
  return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
