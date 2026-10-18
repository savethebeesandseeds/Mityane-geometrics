// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Artificial numerical/causal witnesses. No market observations or fit.
#include "cuwacunu/networks/direct_embedding_control/v1/control.h"
#include <ATen/CPUGeneratorImpl.h>
#include <iostream>

namespace dc=cuwacunu::networks::direct_embedding_control::v1;
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace fs=std::filesystem;
namespace {
int checks=0;
void check(bool ok,const std::string& message){++checks;if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F action,const std::string& message){bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}check(rejected,message);}
ev::ArrayBank bank(){
  ev::ArrayBank b;b.coordinate_id="synthetic-control-engineering-v1:native-e";b.interval_ms=10;b.window_ticks=4;std::vector<double> values;
  for(const auto start:{0,200})for(int64_t j=start;j<start+120;++j){ev::ArrayRow r;r.row_id=j;r.endpoint_ms=j*10;r.available_ms=j*10+2;r.support_begin_ms=(j-3)*10;r.support_end_ms=j*10;r.role=start==0?ev::Role::train:ev::Role::validation;b.rows.push_back(r);
    values.push_back(std::sin(.11*j));values.push_back(std::cos(.11*j));}
  b.values=torch::tensor(values,torch::kFloat64).reshape({240,2});b.valid=torch::ones({240},torch::kBool);return b;
}
std::vector<ev::Association> pairs(int64_t begin,int64_t end){
  std::vector<ev::Association> out;for(int64_t j=begin;j<end;++j)for(const auto horizon:{1,4,8}){ev::Association p;p.pair_id=j*10+horizon;p.origin_row_id=j;p.target_row_id=j+horizon;p.horizon_ticks=horizon;p.decision_ms=j*10+3;p.history_row_ids={j-3,j-2,j-1,j};p.sample_weight=1+(j%3);p.dependence_group="synthetic-block-"+std::to_string(j/16);p.actual_overlap_fraction=horizon<4?.75:0;out.push_back(p);}return out;
}
dc::Config config(){dc::Config c;c.width=2;c.hidden_width=8;c.horizons={1,4,8};c.fit_cutoff_ms=1200;c.train_begin_ms=0;c.train_end_ms=1190;c.updates=120;c.microbatch=16;c.learning_rate=.01;c.parameter_budget=1000;return c;}
dc::Identity identity(){return {"synthetic-control-engineering-v1:native-e","artificial-source-v1","train0-119-validation200-319","synthetic-pairs-v1",dc::av::sha256_bytes("synthetic-Encoder"),"synthetic-engineering-budget-v1"};}
double loss(const ev::ArrayBank& b,const ev::PredictionSet& p){double sum=0;int64_t n=0;const auto idx=dc::detail::index(b);for(size_t j=0;j<p.associations.size();++j)if(p.valid[j].item<bool>()){const auto target=dc::detail::find(idx,p.associations[j].target_row_id);sum+=(p.values[j]-b.values[target]).square().mean().item<double>();++n;}return sum/n;}
void registered_geometry_witness(){
  ev::ArrayBank b;b.coordinate_id="SYNTHETIC-native36-registered-geometry";b.interval_ms=10;b.window_ticks=128;
  std::vector<double> values;
  for(int64_t j=0;j<400;++j){ev::ArrayRow r;r.row_id=j;r.endpoint_ms=j*10;r.available_ms=j*10+2;
    r.support_begin_ms=(j-127)*10;r.support_end_ms=j*10;r.role=ev::Role::train;b.rows.push_back(r);
    for(int64_t k=0;k<36;++k)values.push_back(std::sin(.007*j*(k+1))+.001*j);}
  b.values=torch::tensor(values,torch::kFloat64).reshape({400,36});b.valid=torch::ones({400},torch::kBool);
  std::vector<ev::Association> associations;
  for(int64_t j=140;j<260;++j)for(const auto h:{1,16,128}){ev::Association p;p.pair_id=associations.size()+1;
    p.origin_row_id=j;p.target_row_id=j+h;p.horizon_ticks=h;p.decision_ms=j*10+3;p.history_row_ids={j-3,j-2,j-1,j};associations.push_back(p);}
  dc::Config c;c.fit_cutoff_ms=4000;c.train_begin_ms=0;c.train_end_ms=3990;
  auto id=identity();id.coordinate_id=b.coordinate_id;dc::Trainer model(c,id,b,associations);
  int64_t actual=0;for(const auto& h:model.horizons)for(const auto& p:h.model->parameters())actual+=p.numel();
  check(actual==17484&&model.horizons.size()==3,"registered D36/K4/tanh32 actual tensor geometry differs");
  for(const auto h:c.horizons)model.step(h);
  for(const auto& h:model.horizons)check(h.completed_updates==1&&h.sampler_draws==32,
    "registered horizon cannot perform its declared AdamW update");
  model.validate_runtime();
}
}
int main(int argc,char** argv)try{
  check(argc==2,"exclusive temporary directory required");const fs::path tmp(argv[1]);check(fs::create_directories(tmp),"temporary leaf already exists");torch::set_num_threads(1);
  registered_geometry_witness();
  const auto b=bank();const auto c=config();const auto id=identity();const auto train=pairs(6,105),valid=pairs(206,305);auto all=train;all.insert(all.end(),valid.begin(),valid.end());
  const auto rng=at::detail::getDefaultCPUGenerator().get_state().clone();dc::Trainer model(c,id,b,all);
  check(torch::equal(rng,at::detail::getDefaultCPUGenerator().get_state()),"construction changed global CPU RNG");
  check(dc::parameter_count(dc::Config{})==17484,"registered native36 parameter count");check(model.excluded_rows==static_cast<int64_t>(valid.size()),"held-out associations entered fit");
  ev::BaselineRecipe recipe;recipe.fit_cutoff_ms=c.fit_cutoff_ms;recipe.train_begin_ms=c.train_begin_ms;recipe.train_end_ms=c.train_end_ms;recipe.history_size=4;recipe.spacing_ticks=c.spacing_ticks;
  const auto baseline=ev::fit_baselines(b,all,recipe);
  for(size_t j=0;j<model.horizons.size();++j){const auto& h=model.horizons[j];const auto& expected=baseline.horizons[j];check(h.training_rows==expected.training_rows,"baseline fitting population count differs");check(torch::equal(h.model->input_mean,expected.input_mean)&&torch::equal(h.model->input_scale,expected.input_scale),"baseline weighted transform differs");}
  const auto initial=model.predict(b,valid);const auto persist=ev::predict_baselines(b,valid,baseline)[1];check(torch::equal(initial.values,persist.values)&&torch::equal(initial.valid,persist.valid),"initial prediction is not persistence");
  auto changed=b;changed.values=b.values.clone();changed.values.slice(0,120).fill_(123456.);dc::Trainer unaffected(c,id,changed,all);check(model.model_state_sha256()==unaffected.model_state_sha256(),"held-out values changed fit state");
  changed.values.slice(0,120).fill_(std::numeric_limits<double>::quiet_NaN());dc::Trainer heldout_nan(c,id,changed,all);check(model.model_state_sha256()==heldout_nan.model_state_sha256(),"held-out observed NaNs changed fit state");
  // Exact baseline population rejection covers missing history, late maturity,
  // partial raw support and TRAIN envelopes without inventing compressed rows.
  auto difficult=b;difficult.rows[20].available_ms=1201;difficult.rows[30].support_begin_ms=-1;difficult.rows[40].complete_support=false;
  auto special=all;special[0].input_eligible=false;special[1].history_row_ids[0]=-99;special[2].target_eligible=false;
  dc::Trainer restricted(c,id,difficult,special);const auto expected=ev::fit_baselines(difficult,special,recipe);
  for(size_t j=0;j<restricted.horizons.size();++j)check(restricted.horizons[j].training_rows==expected.horizons[j].training_rows&&torch::equal(restricted.horizons[j].model->input_mean,expected.horizons[j].input_mean),"eligibility differs from baseline");
  auto one=std::vector<ev::Association>{valid[2]};auto future=b;future.values=b.values.clone();future.valid=b.valid.clone();const auto target=dc::detail::find(dc::detail::index(b),one[0].target_row_id);future.valid[target]=false;future.values[target].fill_(std::numeric_limits<double>::quiet_NaN());
  check(torch::equal(model.predict(b,one).values,model.predict(future,one).values),"future target value/mask changed issuance");one[0].target_eligible=false;one[0].target_row_id=-55;check(model.predict(b,one).valid.item<bool>(),"target-side eligibility changed issuance");
  auto observed_nan=b;observed_nan.values=b.values.clone();observed_nan.values[target].fill_(std::numeric_limits<double>::quiet_NaN());
  check(torch::equal(model.predict(b,{valid[2]}).values,model.predict(observed_nan,{valid[2]}).values),"future target observed NaN changed issuance");
  auto invalid=one;invalid[0].history_row_ids[0]=-99;invalid[0].ledger_input_reasons=0x123;const auto invalid_prediction=model.predict(b,invalid);check(!invalid_prediction.valid.item<bool>()&&(invalid_prediction.reasons.item<int64_t>()&ev::missing_input)&&invalid_prediction.associations[0].ledger_input_reasons==0x123,"invalid association/reason lost");
  auto before=valid;before[0].decision_ms=c.fit_cutoff_ms-1;check(model.predict(b,before).reasons[0].item<int64_t>()&ev::fit_not_available,"pre-fit forecast issued");
  auto old_origin=std::vector<ev::Association>{train.back()};old_origin[0].decision_ms=2000;
  check(model.predict(b,old_origin).reasons.item<int64_t>()&ev::fit_not_available,"origin before fitting accepted through later decision");
  auto equal_config=c;equal_config.fit_cutoff_ms=2060;dc::Trainer equal_fit(equal_config,id,b,all);
  check(equal_fit.predict(b,{valid.front()}).reasons.item<int64_t>()&ev::fit_not_available,"origin equal to fit cutoff accepted");
  auto wrong_bank=b;wrong_bank.coordinate_id+="changed";rejects([&]{model.predict(wrong_bank,valid);},"changed coordinates accepted");
  const double initial_loss=loss(b,initial);model.train();const auto trained=model.predict(b,valid);check(loss(b,trained)<initial_loss*.25,"synthetic temporal learning failed");
  check(torch::equal(rng,at::detail::getDefaultCPUGenerator().get_state()),"training changed global CPU RNG");
  dc::Trainer split(c,id,b,all);for(auto h:c.horizons)for(int j=0;j<47;++j)split.step(h);const auto checkpoint=(tmp/"training.pt").string();split.save_training(checkpoint);auto resumed=dc::Trainer::load_training(checkpoint,c,id);resumed->train();
  check(model.model_state_sha256()==resumed->model_state_sha256()&&torch::equal(model.predict(b,valid).values,resumed->predict(b,valid).values),"resumed weights/predictions differ from uninterrupted fit");
  for(size_t j=0;j<model.horizons.size();++j){const auto& a=model.horizons[j];const auto& z=resumed->horizons[j];check(a.completed_updates==z.completed_updates&&a.sampler_draws==z.sampler_draws&&a.sampler==z.sampler,"resume sampler/counters differ");for(size_t k=0;k<a.model->parameters().size();++k){const auto* aa=dynamic_cast<torch::optim::AdamWParamState*>(a.optimizer->state().at(a.model->parameters()[k].unsafeGetTensorImpl()).get());const auto* zz=dynamic_cast<torch::optim::AdamWParamState*>(z.optimizer->state().at(z.model->parameters()[k].unsafeGetTensorImpl()).get());check(torch::equal(aa->exp_avg(),zz->exp_avg())&&torch::equal(aa->exp_avg_sq(),zz->exp_avg_sq())&&aa->step()==zz->step(),"resume optimizer differs");}}
  const auto inference=(tmp/"inference.pt").string();model.save_inference(inference);auto frozen=dc::Trainer::load_inference(inference,c,id);check(frozen->model_state_sha256()==model.model_state_sha256()&&torch::equal(frozen->predict(b,valid).values,trained.values),"frozen roundtrip changed mapping");
  rejects([&]{frozen->train();},"inference could continue training");rejects([&]{dc::Trainer::load_training(inference,c,id);},"inference loaded as training");rejects([&]{dc::Trainer::load_inference(checkpoint,c,id);},"training loaded as inference");rejects([&]{model.save_inference(inference);},"archive overwrite accepted");
  auto config_change=c;config_change.learning_rate=.02;rejects([&]{dc::Trainer::load_training(checkpoint,config_change,id);},"changed optimizer accepted");auto parent_change=id;parent_change.encoder_model_ref=dc::av::sha256_bytes("changed Encoder");rejects([&]{dc::Trainer::load_inference(inference,c,parent_change);},"changed Encoder accepted");
  check(torch::equal(rng,at::detail::getDefaultCPUGenerator().get_state()),"archive loading changed global CPU RNG");
  model.horizons[0].model->input_scale[0]=99;rejects([&]{model.predict(b,valid);},"frozen transform mutation accepted");
  std::cout<<"Direct embedding control v1: "<<checks<<" artificial checks passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
