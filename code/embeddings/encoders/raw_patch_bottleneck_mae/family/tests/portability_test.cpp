// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Waajacu. See the repository LICENSE.
// Bounded artificial CUDA portability proof; no datasets, heads or quality score.
#include "embedding/encoders/raw_patch_bottleneck_mae/family/training.h"
#include <torch/cuda.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

namespace {
namespace r = embedding::encoders::raw_patch_bottleneck_mae;
namespace f = r::family;
namespace fs = std::filesystem;
int64_t assertions=0, refusals=0;
void check(bool condition,const char *message) {
  ++assertions; TORCH_CHECK(condition,"[rpb portability] ",message);
}
template<class Action> void refuses(Action action,const char *message) {
  bool rejected=false;
  try { action(); } catch(const std::exception &) { rejected=true; }
  check(rejected,message); ++refusals;
}
bool exact(const torch::Tensor &a,const torch::Tensor &b) {
  if(!a.defined()||!b.defined()) return a.defined()==b.defined();
  const auto x=a.detach().to(torch::kCPU).contiguous();
  const auto y=b.detach().to(torch::kCPU).contiguous();
  return x.scalar_type()==y.scalar_type() && x.sizes()==y.sizes() &&
      (!x.numel()||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.numel()*x.element_size())==0);
}
r::Input fixture(const r::Config &c) {
  constexpr int64_t B=4;
  auto data=torch::empty({B,c.channel_count,c.history_length,c.input_width},torch::kFloat64);
  auto observed=torch::ones(data.sizes(),torch::kBool);
  auto a=data.accessor<double,4>();
  for(int64_t b=0;b<B;++b) for(int64_t j=0;j<c.channel_count;++j)
    for(int64_t t=0;t<c.history_length;++t) for(int64_t k=0;k<c.input_width;++k)
      a[b][j][t][k]=k==0?1099511627776.+.25*t+.5*b+.125*j:
          (k==c.input_width-1?200.+j:std::sin(.13*t+.17*j+.11*k+.19*b)+.3*std::cos(.29*t+.07*b));
  observed[0][0][5][1]=false;
  data[0][0][5][1]=std::numeric_limits<double>::quiet_NaN();
  return {data,observed,torch::tensor(r::resolved_channel_ids(c),torch::kInt64),
      torch::full({B},double(c.history_length-1),torch::kFloat64),c.sampling_interval};
}
r::Input slice(const r::Input &input,int64_t first,int64_t last) {
  return {input.data.slice(0,first,last),input.observed.slice(0,first,last),
      input.channel_ids.dim()==2?input.channel_ids.slice(0,first,last):input.channel_ids,
      input.endpoints.slice(0,first,last),input.sampling_interval};
}
// This fixture's fully named row policy is MT19937_64 modulo sampling with
// replacement, recreated from the absolute attempt counter; no iterator state.
r::Input selected(const r::Input &input,const r::TrainingProgress &p) {
  std::mt19937_64 generator(r::training_counter_seed(p.row_seed,p.attempted_batches));
  auto rows=torch::tensor(std::vector<int64_t>{int64_t(generator()%input.data.size(0)),
      int64_t(generator()%input.data.size(0))},torch::kInt64);
  return {input.data.index_select(0,rows),input.observed.index_select(0,rows),input.channel_ids,
      input.endpoints.index_select(0,rows),input.sampling_interval};
}
void same_scaler(const r::FrozenScaler &a,const r::FrozenScaler &b) {
  check(torch::allclose(a.mean,b.mean,0.,1e-10) && torch::allclose(a.scale,b.scale,0.,1e-10),
        "streamed and whole-input float64 moments agree within declared tolerance");
  check(exact(a.count,b.count)&&exact(a.channel_ids,b.channel_ids)&&exact(a.floor_applied,b.floor_applied),
        "fit support, semantic IDs and floor flags agree exactly");
}
std::string run(const fs::path &config_path,const fs::path &out,const std::string &name,int64_t seed) {
  auto c=f::read_settings(config_path);
  const auto d=f::dimensions(c);
  const auto raw=fixture(c);
  f::PreprocessingFitter fitter(c); fitter.add(slice(raw,0,1)); fitter.add(slice(raw,1,4));
  auto scaler=fitter.freeze();
  same_scaler(scaler,r::fit_configurable_spectral_scaler(raw,c));
  check(scaler.floor_applied.select(1,c.input_width-1).all().item<bool>(),"constant feature is floored");
  refuses([&]{fitter.add(raw);},"frozen fitter refuses additional TRAIN support");
  f::PreprocessingFitter unseen(c);
  auto absent=raw; absent.observed=torch::zeros_like(raw.observed);
  unseen.add(absent); refuses([&]{(void)unseen.freeze();},"globally unobserved TRAIN fit refuses freeze");
  auto invalid=raw; invalid.data=raw.data.clone(); invalid.data[0][0][0][0]=std::numeric_limits<double>::infinity();
  f::PreprocessingFitter nonfinite(c);
  refuses([&]{nonfinite.add(invalid);},"observed nonfinite TRAIN value refuses fitting");
  const auto normalized=r::transform_configurable_spectral_input(scaler,raw,c);
  check(normalized.data.is_cuda()&&torch::isfinite(normalized.data).all().item<bool>(),"masked poison is absent after actual CUDA transform");
  auto model=f::fresh(c,scaler,seed);
  const auto inventory=f::parameter_inventory(model);
  check(inventory.total==inventory.trainable+inventory.frozen && inventory.total>0,"derived parameter inventory partitions model");
  check(inventory.frozen==d.channel_pairs*4*c.input_width*c.input_width*4,"fixed-prior count derives from geometry");
  const auto initial=f::model_identity(model);
  auto twin=f::fresh(c,scaler,seed);
  check(initial==f::model_identity(twin),"fresh seeded construction agrees exactly");
  auto optimizer=f::make_optimizer(model);
  r::TrainingIdentity identity{"artificial-rpb-portability-source-v1",name+"-TRAIN-only-v1",
      "rpb-family-reference-no-extra-context-v1;"+f::config_identity(c)+";seed="+std::to_string(seed)};
  r::TrainingProgress p{0,0,seed+1,seed+2,seed+3};
  const auto seeds=p;
  auto before=f::encode(model,scaler,raw);
  const auto fixed=before.z_contextual_global.narrow(1,d.learned_width,d.spectral_width).clone();
  const auto masks=r::configurable_spectral_make_training_mask(normalized.observed,c,
      r::training_counter_seed(p.mask_seed,p.attempted_batches));
  const auto plan=r::configurable_spectral_mask_from_hidden(normalized.observed,masks.hidden,c);
  check(exact(plan.target,normalized.observed.logical_and(masks.hidden)),"fixed target support is original observed AND hidden");
  check(!plan.target.logical_and(normalized.observed.logical_not()).any().item<bool>(),"missing cells are not targets");
  auto illegal=masks.hidden.clone(); illegal[0][0][0][0]=!illegal[0][0][0][0].item<bool>();
  refuses([&]{(void)r::configurable_spectral_mask_from_hidden(normalized.observed,illegal,c);},"partial artificial patch mask is refused");
  // Hidden values are absent from both learned and moment routes.
  auto hidden_input=normalized; hidden_input.observed=plan.visible;
  const auto visible=model->encode(hidden_input).z_contextual_global.detach().clone();
  hidden_input.data=normalized.data.clone(); hidden_input.data.masked_fill_(masks.hidden,1e20);
  check(exact(visible,model->encode(hidden_input).z_contextual_global),"hidden raw storage cannot enter visible encoding");
  const auto first=f::train_step(model,scaler,*optimizer,p,selected(raw,p));
  check(first.updated&&p.completed_updates==1&&p.attempted_batches==1&&first.target_cells>0,
        "first eligible raw batch makes one real update");
  check(initial!=f::model_identity(model),"actual optimizer action changes learned model weights");
  if(d.channel_pairs) check(model->relation_projection->weight.abs().sum().item<double>()>0,
        "zero-start content relation receives a nonzero update");
  check(exact(fixed,f::encode(model,scaler,raw).z_contextual_global.narrow(1,d.learned_width,d.spectral_width)),
        "literal spectral coordinates remain fixed after training");
  const auto old_model=f::model_identity(model),old_optimizer=f::optimizer_identity(model,*optimizer);
  const auto skip=f::train_step(model,scaler,*optimizer,p,slice(absent,0,2));
  check(!skip.updated&&p.completed_updates==1&&p.attempted_batches==2,
        "ineligible batch consumes an attempt and no update");
  check(old_model==f::model_identity(model)&&old_optimizer==f::optimizer_identity(model,*optimizer),
        "skip preserves model and optimizer bytes without weight decay");
  const auto state=(out/(name+"-training-1.pt")).string();
  f::save_training_state(state,model,scaler,*optimizer,identity,p);
  const auto state_sha=cuwacunu::artifacts::v1::sha256_file(state);
  auto restored=f::load_training_state(state,c,identity,scaler.identity(),seeds);
  check(restored.progress.attempted_batches==2&&restored.progress.completed_updates==1,
        "saved attempt and update counters reload");
  check(old_model==f::model_identity(restored.model)&&old_optimizer==f::optimizer_identity(restored.model,*restored.optimizer),
        "model and named AdamW moments/steps reload exactly");
  check(scaler.identity()==restored.scaler.identity(),"exact fitted scaler reloads");
  const auto rows=selected(raw,p),resumed_rows=selected(raw,restored.progress);
  check(exact(rows.data,resumed_rows.data)&&exact(rows.observed,resumed_rows.observed),"next row sampling repeats exactly");
  const auto next=f::train_step(model,scaler,*optimizer,p,rows);
  const auto resumed=f::train_step(restored.model,restored.scaler,*restored.optimizer,restored.progress,resumed_rows);
  check(next.updated&&resumed.updated&&next.loss==resumed.loss&&next.gradient_norm==resumed.gradient_norm&&
      next.gradients_sha256==resumed.gradients_sha256&&exact(next.hidden,resumed.hidden),
      "next loss, original mask and all preclip gradients repeat exactly");
  check(f::model_identity(model)==f::model_identity(restored.model)&&
      f::optimizer_identity(model,*optimizer)==f::optimizer_identity(restored.model,*restored.optimizer),
      "next updated weights and all AdamW state repeat exactly");
  const auto final=f::encode(model,scaler,raw);
  check(exact(final.z_contextual_global,f::encode(restored.model,restored.scaler,raw).z_contextual_global),
        "next resumed inference output repeats exactly");
  check(p.completed_updates==2&&p.attempted_batches==3&&restored.progress.completed_updates==2&&
      restored.progress.attempted_batches==3,"skip and both continuation counters agree");
  const auto inference=(out/(name+"-inference.pt")).string();
  f::save_inference(inference,model,scaler,p.completed_updates);
  auto loaded=f::load_inference(inference,c.device,&c);
  check(exact(final.z_contextual_global,f::encode(loaded.model,loaded.scaler,raw).z_contextual_global),
        "inference export loads exact native output");
  refuses([&]{f::save_inference(inference,model,scaler,p.completed_updates);},"immutable inference publication refuses overwrite");
  refuses([&]{f::save_training_state(state,model,scaler,*optimizer,identity,p);},"immutable training publication refuses overwrite");
  check(state_sha==cuwacunu::artifacts::v1::sha256_file(state),"refused replacement preserves original state bytes");
  refuses([&]{(void)f::load_training_state(inference,c,identity,scaler.identity(),seeds);},"inference artifact cannot resume optimizer");
  refuses([&]{(void)f::load_inference(state,c.device,&c);},"training artifact cannot masquerade as inference");
  for(int stream=0;stream<3;++stream) {
    auto changed=seeds;
    ++(stream==0?changed.row_seed:stream==1?changed.mask_seed:changed.context_seed);
    refuses([&]{(void)f::load_training_state(state,c,identity,scaler.identity(),changed);},"independently expected stream seed drift refuses resume");
  }
  auto changed_identity=identity; changed_identity.recipe_identity+="-changed";
  refuses([&]{(void)f::load_training_state(state,c,changed_identity,scaler.identity(),seeds);},"recipe identity drift refuses resume");
  changed_identity=identity; changed_identity.split_identity+="-changed";
  refuses([&]{(void)f::load_training_state(state,c,changed_identity,scaler.identity(),seeds);},"TRAIN split identity drift refuses resume");
  changed_identity=identity; changed_identity.input_source_identity+="-changed";
  refuses([&]{(void)f::load_training_state(state,c,changed_identity,scaler.identity(),seeds);},"source identity drift refuses resume");
  refuses([&]{(void)f::load_training_state(state,c,identity,"different-fitted-scaler",seeds);},"independent TRAIN scaler drift refuses resume");
  auto other=c; ++other.history_length;
  refuses([&]{(void)f::load_training_state(state,other,identity,scaler.identity(),seeds);},"logical history drift refuses resume even with equal padding");
  other=c; other.device=torch::kCPU;
  refuses([&]{(void)f::load_inference(inference,other.device,&other);},"CPU model loading is refused");
  other=c; other.export_width=d.spectral_width;
  refuses([&]{f::validate_config(other);},"width without learned capacity is refused");
  const auto probe_model=f::model_identity(model),probe_optimizer=f::optimizer_identity(model,*optimizer);
  auto invalid_interval=raw; invalid_interval.sampling_interval=2.;
  refuses([&]{(void)f::encode(model,scaler,invalid_interval);},"changed sampling interval refuses inference");
  auto invalid_ids=raw; invalid_ids.channel_ids=raw.channel_ids+100000;
  refuses([&]{(void)f::encode(model,scaler,invalid_ids);},"unknown semantic IDs refuse inference");
  if(c.channel_count>1) {
    invalid_ids.channel_ids=raw.channel_ids.clone(); invalid_ids.channel_ids[1]=invalid_ids.channel_ids[0];
    refuses([&]{(void)f::encode(model,scaler,invalid_ids);},"duplicate semantic IDs refuse inference");
  }
  other=c; other.num_heads=3;
  refuses([&]{f::validate_config(other);},"incompatible attention head count refuses configuration");
  other=c; other.dtype=torch::kFloat64;
  refuses([&]{f::validate_config(other);},"non-float32 model dtype refuses configuration");
  auto finite_overflow=raw; finite_overflow.data=raw.data.clone();
  finite_overflow.data[0][0][0][1]=std::numeric_limits<double>::max()/4.;
  refuses([&]{(void)f::encode(model,scaler,finite_overflow);},"finite observed value overflowing normalized float32 refuses inference");
  auto reordered_parameters=model->parameters(); std::reverse(reordered_parameters.begin(),reordered_parameters.end());
  torch::optim::AdamW reordered(reordered_parameters,r::content_training_adamw_options());
  refuses([&]{f::save_training_state((out/(name+"-reordered-optimizer.pt")).string(),model,scaler,reordered,identity,p);},
        "reordered optimizer parameter group refuses state export");
  const auto original_slots=optimizer->state().size();
  auto orphan_tensor=torch::zeros({1},torch::TensorOptions().device(c.device).dtype(torch::kFloat32));
  auto *orphan_key=orphan_tensor.unsafeGetTensorImpl();
  optimizer->state()[orphan_key]=std::make_unique<torch::optim::AdamWParamState>();
  refuses([&]{f::save_training_state((out/(name+"-orphan-optimizer.pt")).string(),model,scaler,*optimizer,identity,p);},
        "orphan optimizer state refuses state export");
  check(optimizer->state().erase(orphan_key)==1&&optimizer->state().size()==original_slots,
        "orphan probe removes its only temporary state slot");
  auto *moment_state=dynamic_cast<torch::optim::AdamWParamState *>(optimizer->state().begin()->second.get());
  check(moment_state&&moment_state->exp_avg().defined(),"nonfinite-moment probe locates an actual active AdamW slot");
  const auto original_moment=moment_state->exp_avg().detach().clone();
  moment_state->exp_avg().flatten()[0]=std::numeric_limits<float>::quiet_NaN();
  refuses([&]{f::save_training_state((out/(name+"-nonfinite-optimizer.pt")).string(),model,scaler,*optimizer,identity,p);},
        "nonfinite actual AdamW moment refuses state export");
  moment_state->exp_avg().copy_(original_moment);
  check(probe_model==f::model_identity(model)&&probe_optimizer==f::optimizer_identity(model,*optimizer),
        "all no-update refusal probes preserve exact model and optimizer witness");
  auto old_p=p;
  refuses([&]{(void)f::train_step(model,scaler,*optimizer,p,slice(invalid,0,2));},"nonfinite raw update is refused");
  check(p.attempted_batches==old_p.attempted_batches&&p.completed_updates==old_p.completed_updates,
        "input refusal does not consume an attempt");
  auto exhausted=p; exhausted.attempted_batches=std::numeric_limits<int64_t>::max()-1;
  const auto before_limit_model=f::model_identity(model),before_limit_optimizer=f::optimizer_identity(model,*optimizer);
  refuses([&]{(void)f::train_step(model,scaler,*optimizer,exhausted,slice(raw,0,2));},
        "near-limit attempted counter refuses before numerical action");
  check(before_limit_model==f::model_identity(model)&&before_limit_optimizer==f::optimizer_identity(model,*optimizer)&&
      exhausted.attempted_batches==std::numeric_limits<int64_t>::max()-1,
        "counter-limit refusal preserves state and counters");
  torch::optim::AdamW wrong(model->parameters(),torch::optim::AdamWOptions(.002).weight_decay(.0001));
  refuses([&]{f::save_training_state((out/(name+"-wrong-optimizer.pt")).string(),model,scaler,wrong,identity,{});},
        "changed optimizer recipe refuses state export");
  const auto zero=f::encode(model,scaler,absent);
  check(!zero.sample_valid_mask.any().item<bool>()&&zero.z_contextual_global.eq(0).all().item<bool>(),
        "allmissing inference is exact zero and invalid");
  // Window endpoint preparation excludes all later storage; legal final-window
  // observations remain available to the non-prefix-causal global encoder.
  const auto tail=torch::full({4,c.channel_count,3,c.input_width},1e8,torch::kFloat64);
  const auto long_data=torch::cat({raw.data,tail},2);
  const auto long_mask=torch::cat({raw.observed,torch::ones(tail.sizes(),torch::kBool)},2);
  const auto time=torch::arange(c.history_length+3,torch::kFloat64);
  const auto crop=r::prepare_endpoint_history(long_data,long_mask,raw.channel_ids,time,raw.endpoints,
      r::configurable_spectral_input_config(c));
  check(exact(final.z_contextual_global,f::encode(model,scaler,crop).z_contextual_global),
        "post-endpoint storage cannot affect cropped retained-window inference");
  auto late=raw; late.data=raw.data.clone(); late.data.select(2,c.history_length-1).select(2,1).add_(10.);
  check(!exact(final.z_contextual_global,f::encode(model,scaler,late).z_contextual_global),
        "legal later observations within retained window may influence global output");
  f::save_training_state((out/(name+"-training-2.pt")).string(),model,scaler,*optimizer,identity,p);
  std::ostringstream json; json<<"{\"name\":\""<<name<<"\",\"config_sha256\":\""<<f::config_identity(c)
      <<"\",\"parameters\":"<<inventory.total<<",\"trainable_parameters\":"<<inventory.trainable
      <<",\"frozen_parameters\":"<<inventory.frozen<<",\"completed_updates\":2,\"attempted_batches\":3"
      <<",\"physical_optimizer_actions\":3,\"scaler_identity\":\""<<scaler.identity()
      <<"\",\"model_sha256\":\""<<f::model_identity(model)<<"\",\"optimizer_sha256\":\""<<f::optimizer_identity(model,*optimizer)
      <<"\",\"native_sha256\":\""<<f::tensor_identity(final.z_contextual_global)<<"\",\"next_gradients_sha256\":\""<<next.gradients_sha256
      <<"\",\"next_loss\":"<<std::setprecision(17)<<next.loss<<",\"inference_sha256\":\""<<cuwacunu::artifacts::v1::sha256_file(inference)<<"\"}";
  return json.str();
}
}

int main(int argc,char **argv) try {
  TORCH_CHECK(argc==4,"usage: rpb_family_portability_test single.conf multi.conf NEW_OUTPUT_DIRECTORY");
  check(torch::cuda::is_available(),"actual CUDA device is available");
  torch::set_num_threads(1);
  const fs::path output(argv[3]);
  TORCH_CHECK(!fs::exists(output),"[rpb portability] output directory must be new");
  fs::create_directories(output);
  const auto single=run(argv[1],output,"single",163100);
  const auto multi=run(argv[2],output,"multi",163200);
  torch::cuda::synchronize();
  std::ofstream result(output/"test-result.json",std::ios::binary);
  TORCH_CHECK(result,"[rpb portability] cannot write result");
  result<<"{\"schema\":\"rpb-family-portability-result-v1\",\"status\":\"PASS\",\"cuda\":true,\"assertions\":"<<assertions
      <<",\"refusals\":"<<refusals<<",\"physical_optimizer_actions\":6,\"cases\":["<<single<<','<<multi<<"]}\n";
  result.close(); TORCH_CHECK(result,"[rpb portability] result write failure");
  std::cout<<"PASS RPB family portable training: two configurations; six artificial CUDA optimizer actions; "
      <<assertions<<" assertions and "<<refusals<<" refusals; exact resumed numerical state; no quality experiment\n";
  return 0;
} catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
