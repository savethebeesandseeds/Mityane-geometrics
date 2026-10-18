// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/artifacts/v1/tensor_state.h"
#include "cuwacunu/evaluation/latent_dynamics/v1/baselines.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>

namespace cuwacunu::networks::direct_embedding_control::v1 {
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace av=cuwacunu::artifacts::v1;
inline constexpr const char* inference_kind="direct_embedding_control_inference_v1";
inline constexpr const char* training_kind="direct_embedding_control_training_v1";
struct Config {
  int64_t width{36},history_size{4},hidden_width{32};
  std::vector<int64_t> horizons{1,16,128};
  int64_t spacing_ticks{1},updates{2048},microbatch{32},repetition_seed{944101};
  int64_t fit_cutoff_ms{},train_begin_ms{},train_end_ms{},parameter_budget{18000};
  double std_floor{1e-8},learning_rate{1e-3},beta1{.9},beta2{.999},optimizer_epsilon{1e-8},weight_decay{1e-4},clip_norm{1};
  bool require_complete_support{true};
};
struct Identity {
  std::string coordinate_id,source,split,pair_ledger,encoder_model_ref,numeric_card;
};
inline int64_t parameter_count(const Config& c) {
  TORCH_CHECK(c.width>0&&c.width<=4096&&c.history_size>0&&c.history_size<=128&&
    c.hidden_width>0&&c.hidden_width<=4096&&!c.horizons.empty()&&c.horizons.size()<=128,"[direct control v1] unsafe geometry");
  return static_cast<int64_t>(c.horizons.size())*(c.width*c.history_size*c.hidden_width+c.hidden_width+c.hidden_width*c.width+c.width);
}
inline void validate(const Config& c) {
  const auto count=parameter_count(c);
  TORCH_CHECK(c.parameter_budget>=count&&c.parameter_budget<=100000000&&c.spacing_ticks>0&&c.updates>=0&&
    c.microbatch>0&&c.microbatch<=1000000&&c.updates<=INT64_MAX/c.microbatch&&c.repetition_seed>=0&&c.train_begin_ms>=0&&c.train_begin_ms<=c.train_end_ms&&
    c.train_end_ms<=c.fit_cutoff_ms,"[direct control v1] invalid budget/history/fit chronology");
  std::set<int64_t> unique;
  for(const auto h:c.horizons)TORCH_CHECK(h>0&&h<=INT64_MAX-c.repetition_seed&&unique.insert(h).second,"[direct control v1] invalid/duplicate horizon");
  TORCH_CHECK(std::is_sorted(c.horizons.begin(),c.horizons.end()),"[direct control v1] horizons must increase");
  for(const auto x:{c.std_floor,c.learning_rate,c.optimizer_epsilon,c.clip_norm})TORCH_CHECK(std::isfinite(x)&&x>0,"[direct control v1] positive finite optimizer/scale setting required");
  TORCH_CHECK(std::isfinite(c.weight_decay)&&c.weight_decay>=0&&std::isfinite(c.beta1)&&c.beta1>=0&&c.beta1<1&&
    std::isfinite(c.beta2)&&c.beta2>=0&&c.beta2<1,"[direct control v1] invalid AdamW setting");
}
inline void validate(const Identity& id) {
  for(const auto* text:{&id.coordinate_id,&id.source,&id.split,&id.pair_ledger,&id.encoder_model_ref,&id.numeric_card})
    TORCH_CHECK(!text->empty()&&text->size()<=16384&&text->find('\0')==std::string::npos,"[direct control v1] immutable identity required");
  TORCH_CHECK(av::is_sha256(id.encoder_model_ref),"[direct control v1] exact Encoder ModelRef required");
}
namespace detail {
inline void need(bool value,const std::string& message){TORCH_CHECK(value,"[direct control v1] ",message);}
inline std::string canonical(const Config& c) {
  validate(c);std::ostringstream s;s<<std::setprecision(17)<<"direct-embedding-control-v1\n"<<c.width<<' '<<c.history_size<<' '<<c.hidden_width<<' '<<c.horizons.size();
  for(auto h:c.horizons)s<<' '<<h;
  s<<'\n'<<c.spacing_ticks<<' '<<c.updates<<' '<<c.microbatch<<' '<<c.repetition_seed<<' '<<c.fit_cutoff_ms<<' '<<c.train_begin_ms<<' '<<c.train_end_ms<<' '<<c.parameter_budget<<' '<<c.require_complete_support
    <<'\n'<<c.std_floor<<' '<<c.learning_rate<<' '<<c.beta1<<' '<<c.beta2<<' '<<c.optimizer_epsilon<<' '<<c.weight_decay<<' '<<c.clip_norm
    <<"\nCPU-float64;independent-horizons;oldest-first;linear-tanh-linear;last-embedding-residual;weighted-TRAIN-input-population-std;weighted-TRAIN-delta-RMS;"
      "local-mt19937_64;seed=repetition+horizon;Xavier-uniform-gain1;zero-bias-and-output;uniform-replacement-rejection-sampler-v1;weighted-sampled-coordinate-MSE;fixed-budget\n";
  return s.str();
}
inline std::string canonical(const Identity& id) {
  validate(id);std::ostringstream s;
  for(const auto* field:{&id.coordinate_id,&id.source,&id.split,&id.pair_ledger,&id.encoder_model_ref,&id.numeric_card})s<<field->size()<<':'<<*field<<'\n';
  return s.str();
}
using Index=std::unordered_map<int64_t,int64_t>;
inline Index index(const ev::ArrayBank& b){Index out;for(size_t j=0;j<b.rows.size();++j)out.emplace(b.rows[j].row_id,j);return out;}
inline int64_t find(const Index& i,int64_t id){const auto it=i.find(id);return it==i.end()?-1:it->second;}
inline void validate_pairs(const std::vector<ev::Association>& pairs){
  std::set<int64_t> ids;
  for(const auto& p:pairs)need(ids.insert(p.pair_id).second&&p.horizon_ticks>0&&std::isfinite(p.sample_weight)&&p.sample_weight>0&&
    std::isfinite(p.actual_overlap_fraction)&&p.actual_overlap_fraction>=0&&p.actual_overlap_fraction<=1,"invalid association identity/weight/overlap");
}
// Issuance may inspect input values only. A full-bank finite-value reduction
// would make a later target's bytes or validity affect a past prediction.
inline void validate_bank_structure(const ev::ArrayBank& b){
  const auto n=static_cast<int64_t>(b.rows.size());
  need(!b.coordinate_id.empty()&&b.interval_ms>0&&b.window_ticks>0,"invalid bank identity/geometry");
  need(b.values.defined()&&b.values.device().is_cpu()&&(b.values.scalar_type()==torch::kFloat64||b.values.scalar_type()==torch::kFloat32)&&b.values.dim()==2&&b.values.size(0)==n&&b.values.size(1)>0,"invalid bank value storage");
  need(b.valid.defined()&&b.valid.device().is_cpu()&&b.valid.scalar_type()==torch::kBool&&b.valid.sizes()==torch::IntArrayRef({n}),"invalid bank validity storage");
  std::set<int64_t> ids,endpoints;
  for(const auto& r:b.rows){
    need(ids.insert(r.row_id).second&&endpoints.insert(r.endpoint_ms).second,"duplicate bank row/time");
    need((r.role==ev::Role::train||r.role==ev::Role::validation||r.role==ev::Role::test)&&r.support_begin_ms<=r.support_end_ms&&r.support_end_ms<=r.endpoint_ms&&r.endpoint_ms<=r.available_ms,"invalid bank role/support/availability");
    need(std::isfinite(r.observed_fraction)&&r.observed_fraction>=0&&r.observed_fraction<=1,"invalid observed fraction");
  }
}
inline bool exact_offset(int64_t from,int64_t to,int64_t ticks,int64_t interval){
  if(ticks<0||ticks>INT64_MAX/interval)return false;
  const auto delta=ticks*interval;return from<=INT64_MAX-delta&&from+delta==to;
}
inline std::vector<int64_t> history(const ev::Association& p,int64_t count,const Index& idx){
  std::vector<int64_t> out;if(static_cast<int64_t>(p.history_row_ids.size())<count)return out;
  for(auto it=p.history_row_ids.end()-count;it!=p.history_row_ids.end();++it)out.push_back(find(idx,*it));
  return out;
}
inline int64_t input_reasons(const ev::ArrayBank& b,const ev::Association& p,const Index& idx,const Config& c){
  int64_t why=p.input_eligible?int64_t{0}:static_cast<int64_t>(ev::ledger_input_invalid);const auto origin=find(idx,p.origin_row_id);const auto hist=history(p,c.history_size,idx);
  if(origin<0||static_cast<int64_t>(hist.size())!=c.history_size)return why|ev::missing_input;
  const auto& o=b.rows[origin];if(c.fit_cutoff_ms>p.decision_ms||c.fit_cutoff_ms>=o.endpoint_ms)why|=ev::fit_not_available;
  if(p.history_row_ids.back()!=p.origin_row_id)why|=ev::history_grid_mismatch;
  for(int64_t k=0;k<c.history_size;++k){const auto i=hist[k];if(i<0){why|=ev::missing_input;continue;}const auto& r=b.rows[i];
    if(!b.valid[i].item<bool>()||!torch::isfinite(b.values[i]).all().item<bool>())why|=ev::invalid_input;
    if(r.available_ms>p.decision_ms)why|=ev::input_not_available;
    if(r.role!=o.role)why|=ev::role_mismatch;
    if(c.history_size-1-k>INT64_MAX/c.spacing_ticks||!exact_offset(r.endpoint_ms,o.endpoint_ms,(c.history_size-1-k)*c.spacing_ticks,b.interval_ms))why|=ev::history_grid_mismatch;
    if(c.require_complete_support&&(!r.complete_support||r.stale))why|=ev::incomplete_support;
  }return why;
}
inline int64_t fit_reasons(const ev::ArrayBank& b,const ev::Association& p,const Index& idx,const Config& c){
  // Reject non-TRAIN metadata before consulting held-out values or masks.
  const auto fit_origin=find(idx,p.origin_row_id),fit_target=find(idx,p.target_row_id);
  if(fit_origin>=0&&b.rows[fit_origin].role!=ev::Role::train)return ev::role_mismatch;
  if(fit_target>=0&&b.rows[fit_target].role!=ev::Role::train)return ev::role_mismatch;
  for(const auto i:history(p,c.history_size,idx))if(i>=0&&b.rows[i].role!=ev::Role::train)return ev::role_mismatch;
  auto why=input_reasons(b,p,idx,c)&~ev::fit_not_available;
  why|=p.target_eligible?int64_t{0}:static_cast<int64_t>(ev::ledger_target_invalid);const auto origin=find(idx,p.origin_row_id),target=find(idx,p.target_row_id);
  if(origin<0||target<0)why|=ev::missing_target;
  else {const auto& o=b.rows[origin];const auto& t=b.rows[target];
    if(!b.valid[target].item<bool>()||!torch::isfinite(b.values[target]).all().item<bool>())why|=ev::invalid_target;
    if(t.available_ms>c.fit_cutoff_ms)why|=ev::target_not_mature;
    if(t.role!=o.role)why|=ev::role_mismatch;
    if(!exact_offset(o.endpoint_ms,t.endpoint_ms,p.horizon_ticks,b.interval_ms))why|=ev::target_grid_mismatch;
    if(c.require_complete_support&&(!t.complete_support||t.stale))why|=ev::incomplete_support;
  }
  if(p.decision_ms>c.fit_cutoff_ms)why|=ev::input_not_available;
  auto rows=history(p,c.history_size,idx);rows.push_back(target);
  for(auto i:rows){if(i<0)continue;const auto& r=b.rows[i];if(r.role!=ev::Role::train)why|=ev::role_mismatch;
    if(r.support_begin_ms<c.train_begin_ms||r.endpoint_ms>c.train_end_ms||r.support_end_ms>c.train_end_ms||r.available_ms>c.fit_cutoff_ms)why|=ev::outside_train_support;
  }return why;
}
inline torch::Tensor flattened(const ev::ArrayBank& b,const ev::Association& p,const Index& idx,int64_t k){
  return b.values.index_select(0,torch::tensor(history(p,k,idx),torch::kInt64)).to(torch::kFloat64).reshape({-1});
}
inline void text(torch::serialize::OutputArchive& a,const std::string& name,const std::string& value){a.write(name,torch::IValue(value));}
inline std::string text(torch::serialize::InputArchive& a,const std::string& name){torch::IValue v;a.read(name,v);need(v.isString(),"malformed archived text "+name);return v.toStringRef();}
inline torch::Tensor tensor(torch::serialize::InputArchive& a,const std::string& name){torch::Tensor t;a.read(name,t,true);return t;}
inline void shape(const torch::Tensor& t,torch::ScalarType type,torch::IntArrayRef dims,const std::string& name){
  need(t.defined()&&t.device().is_cpu()&&t.scalar_type()==type&&t.sizes()==dims,"malformed tensor "+name);
}
inline uint64_t sample(std::mt19937_64& rng,uint64_t count){
  need(count>0,"empty sampler population");const uint64_t threshold=-count%count;for(;;){const auto value=rng();if(value>=threshold)return value%count;}
}
inline void publish(torch::serialize::OutputArchive& a,const std::string& path){
  const std::filesystem::path destination(path);need(!destination.empty()&&std::filesystem::is_directory(destination.parent_path().empty()?".":destination.parent_path()),"existing archive parent required");
  need(!std::filesystem::exists(destination),"immutable archive already exists");static std::atomic<uint64_t> serial{0};
  const auto temp=path+".stage-"+std::to_string(::getpid())+"-"+std::to_string(serial++);const int fd=::open(temp.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);
  need(fd>=0,"cannot reserve archive staging file");::close(fd);
  try{a.save_to(temp);std::filesystem::create_hard_link(temp,destination);std::filesystem::remove(temp);}catch(...){std::error_code ignored;std::filesystem::remove(temp,ignored);throw;}
}
inline torch::optim::AdamWOptions optimizer_options(const Config& c){return torch::optim::AdamWOptions(c.learning_rate).betas(std::make_tuple(c.beta1,c.beta2)).eps(c.optimizer_epsilon).weight_decay(c.weight_decay).amsgrad(false);}
} // namespace detail

// Parameter storage is created directly. No torch module constructor or random
// operator can consume a process-wide RNG stream.
struct ControlModelImpl:torch::nn::Module {
  torch::Tensor w1,b1,w2,b2,input_mean,input_scale,delta_scale;
  int64_t width,history_size,hidden_width;
  ControlModelImpl(const Config& c,int64_t horizon,const torch::Tensor& mean,const torch::Tensor& scale,const torch::Tensor& residual)
    :width(c.width),history_size(c.history_size),hidden_width(c.hidden_width){
    const auto options=torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    w1=register_parameter("w1",torch::empty({c.hidden_width,c.width*c.history_size},options));
    b1=register_parameter("b1",torch::zeros({c.hidden_width},options));w2=register_parameter("w2",torch::zeros({c.width,c.hidden_width},options));b2=register_parameter("b2",torch::zeros({c.width},options));
    input_mean=register_buffer("input_mean",mean.detach().clone());input_scale=register_buffer("input_scale",scale.detach().clone());delta_scale=register_buffer("delta_scale",residual.detach().clone());
    std::mt19937_64 initializer(static_cast<uint64_t>(c.repetition_seed+horizon));const auto bound=std::sqrt(6.0/(c.width*c.history_size+c.hidden_width));
    auto* values=w1.data_ptr<double>();for(int64_t j=0;j<w1.numel();++j){const double u=static_cast<double>(initializer()>>11)*0x1.0p-53;values[j]=(2*u-1)*bound;}
  }
  torch::Tensor forward(const torch::Tensor& raw_history)const{
    const auto x=raw_history.dim()==3?raw_history.flatten(1):raw_history;
    TORCH_CHECK(x.device().is_cpu()&&x.scalar_type()==torch::kFloat64&&x.dim()==2&&x.size(1)==width*history_size,"[direct control v1] native CPU F64 history required");
    return x.slice(1,(history_size-1)*width,history_size*width)+delta_scale*(torch::tanh(((x-input_mean)/input_scale).matmul(w1.t())+b1).matmul(w2.t())+b2);
  }
};
TORCH_MODULE(ControlModel);
struct HorizonState {
  int64_t horizon_ticks{},training_rows{},completed_updates{},sampler_draws{};
  double training_weight{};
  ControlModel model{nullptr};
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::mt19937_64 sampler;
  torch::Tensor training_pair_ids,training_weights,training_x,training_y;
  torch::Tensor bound_mean,bound_scale,bound_delta;
};

class Trainer {
 public:
  const Config config;
  const Identity identity;
  std::vector<HorizonState> horizons;
  int64_t interval_ms{},window_ticks{},submitted_rows{},excluded_rows{};
  Trainer(const Config& c,const Identity& id,const ev::ArrayBank& bank,const std::vector<ev::Association>& pairs)
    :config(c),identity(id){
    validate(c);validate(id);detail::validate_bank_structure(bank);detail::validate_pairs(pairs);
    detail::need(bank.coordinate_id==id.coordinate_id&&bank.values.size(1)==c.width,"fit bank coordinate/width mismatch");
    interval_ms=bank.interval_ms;window_ticks=bank.window_ticks;submitted_rows=pairs.size();const auto idx=detail::index(bank);
    for(const auto horizon:c.horizons){
      std::vector<torch::Tensor> xx,yy;std::vector<double> ww;std::vector<int64_t> ids;
      for(const auto& p:pairs){if(p.horizon_ticks!=horizon)continue;if(detail::fit_reasons(bank,p,idx,c)){++excluded_rows;continue;}
        xx.push_back(detail::flattened(bank,p,idx,c.history_size));yy.push_back(bank.values[detail::find(idx,p.target_row_id)].to(torch::kFloat64));ww.push_back(p.sample_weight);ids.push_back(p.pair_id);
      }
      detail::need(!xx.empty(),"registered horizon has no eligible mature TRAIN associations");
      HorizonState h;h.horizon_ticks=horizon;h.training_rows=xx.size();h.training_x=torch::stack(xx);h.training_y=torch::stack(yy);h.training_weights=torch::tensor(ww,torch::kFloat64);h.training_pair_ids=torch::tensor(ids,torch::kInt64);
      h.training_weight=h.training_weights.sum().item<double>();const auto w=h.training_weights.unsqueeze(1);
      const auto mean=(h.training_x*w).sum(0)/h.training_weight;
      const auto scale=(((h.training_x-mean).square()*w).sum(0)/h.training_weight).sqrt().clamp_min(c.std_floor);
      const auto origin=h.training_x.slice(1,(c.history_size-1)*c.width,c.history_size*c.width);
      const auto residual=(((h.training_y-origin).square()*w).sum(0)/h.training_weight).sqrt().clamp_min(c.std_floor);
      initialize(h,mean,scale,residual);horizons.push_back(std::move(h));
    }
    validate_runtime();
  }
  static int64_t parameter_count(const Config& c){return v1::parameter_count(c);}
  void step(int64_t horizon){
    validate_runtime();auto& h=at(horizon);detail::need(!inference_only_&&h.training_x.defined(),"inference archive cannot continue training");
    detail::need(h.completed_updates<config.updates,"registered update budget exhausted");
    std::vector<int64_t> chosen;chosen.reserve(config.microbatch);
    for(int64_t j=0;j<config.microbatch;++j)chosen.push_back(detail::sample(h.sampler,h.training_rows));
    h.sampler_draws+=config.microbatch;
    const auto indices=torch::tensor(chosen,torch::kInt64),x=h.training_x.index_select(0,indices),y=h.training_y.index_select(0,indices),w=h.training_weights.index_select(0,indices).unsqueeze(1);
    h.optimizer->zero_grad();const auto error=(h.model->forward(x)-y)/h.model->delta_scale;
    const auto loss=(error.square()*w).sum()/(w.sum()*config.width);detail::need(torch::isfinite(loss).item<bool>(),"nonfinite training loss");loss.backward();
    for(const auto& p:h.model->parameters())detail::need(p.grad().defined()&&torch::isfinite(p.grad()).all().item<bool>(),"nonfinite/missing gradient");
    torch::nn::utils::clip_grad_norm_(h.model->parameters(),config.clip_norm);h.optimizer->step();++h.completed_updates;validate_runtime();
  }
  void train(){detail::need(!inference_only_,"inference archive cannot continue training");for(auto horizon:config.horizons)while(at(horizon).completed_updates<config.updates)step(horizon);}
  ev::PredictionSet predict(const ev::ArrayBank& bank,const std::vector<ev::Association>& pairs)const{
    torch::NoGradGuard guard;validate_runtime();detail::validate_bank_structure(bank);detail::validate_pairs(pairs);
    detail::need(bank.coordinate_id==identity.coordinate_id&&bank.values.size(1)==config.width&&bank.interval_ms==interval_ms&&bank.window_ticks==window_ticks,"prediction coordinate/geometry mismatch");
    ev::PredictionSet out;out.coordinate_id=identity.coordinate_id;out.method="residual-mlp-k"+std::to_string(config.history_size);out.fit_cutoff_ms=config.fit_cutoff_ms;out.history_size=config.history_size;out.spacing_ticks=config.spacing_ticks;out.associations=pairs;
    const auto n=static_cast<int64_t>(pairs.size());out.values=torch::zeros({n,config.width},torch::kFloat64);out.valid=torch::zeros({n},torch::kBool);out.reasons=torch::zeros({n},torch::kInt64);const auto idx=detail::index(bank);
    for(int64_t j=0;j<n;++j){const auto& p=pairs[j];auto why=detail::input_reasons(bank,p,idx,config);const auto* h=lookup(p.horizon_ticks);if(!h)why|=ev::no_fitted_horizon;
      out.reasons[j]=why;if(why)continue;const auto x=detail::flattened(bank,p,idx,config.history_size).unsqueeze(0);const auto value=h->model->forward(x).squeeze(0);
      detail::need(torch::isfinite(value).all().item<bool>(),"nonfinite issued prediction");out.values[j].copy_(value);out.valid[j]=true;
    }return out;
  }
  std::string model_state_sha256()const{
    validate_runtime();std::vector<av::NamedTensor> tensors{{"geometry",torch::tensor(std::vector<int64_t>{interval_ms,window_ticks,submitted_rows,excluded_rows},torch::kInt64)}};
    for(const auto& h:horizons){const auto prefix="h"+std::to_string(h.horizon_ticks)+".";
      for(const auto& p:h.model->named_parameters(true))tensors.emplace_back(prefix+"parameter."+p.key(),p.value());
      for(const auto& p:h.model->named_buffers(true))tensors.emplace_back(prefix+"buffer."+p.key(),p.value());
      tensors.emplace_back(prefix+"fit_pair_ids",h.training_pair_ids);tensors.emplace_back(prefix+"fit_weights",h.training_weights);
      tensors.emplace_back(prefix+"fit_progress",torch::tensor(std::vector<int64_t>{h.horizon_ticks,h.training_rows,h.completed_updates,h.sampler_draws},torch::kInt64));
      tensors.emplace_back(prefix+"fit_weight",torch::tensor(h.training_weight,torch::kFloat64));
    }return av::tensor_state_sha256(std::move(tensors));
  }
  void validate_runtime()const{
    validate(config);validate(identity);detail::need(interval_ms>0&&window_ticks>0&&horizons.size()==config.horizons.size()&&submitted_rows>=0&&excluded_rows>=0,"invalid runtime geometry/count");
    const auto f=config.width*config.history_size;
    for(size_t i=0;i<horizons.size();++i){const auto& h=horizons[i];detail::need(h.horizon_ticks==config.horizons[i]&&h.training_rows>0&&std::isfinite(h.training_weight)&&h.training_weight>0&&h.completed_updates>=0&&h.completed_updates<=config.updates&&h.sampler_draws==h.completed_updates*config.microbatch,"invalid horizon/progress state");
      detail::need(bool(h.model)&&h.model->width==config.width&&h.model->history_size==config.history_size&&h.model->hidden_width==config.hidden_width,"module configuration drift");
      const auto named_parameters=h.model->named_parameters(),named_buffers=h.model->named_buffers();
      detail::need(named_parameters.size()==4&&named_buffers.size()==3,"registered module state drift");
      for(const auto& field:std::vector<av::NamedTensor>{{"w1",h.model->w1},{"b1",h.model->b1},{"w2",h.model->w2},{"b2",h.model->b2}})
        detail::need(named_parameters[field.first].unsafeGetTensorImpl()==field.second.unsafeGetTensorImpl(),"forward parameter differs from registered state");
      for(const auto& field:std::vector<av::NamedTensor>{{"input_mean",h.model->input_mean},{"input_scale",h.model->input_scale},{"delta_scale",h.model->delta_scale}})
        detail::need(named_buffers[field.first].unsafeGetTensorImpl()==field.second.unsafeGetTensorImpl(),"forward buffer differs from registered state");
      detail::shape(h.model->w1,torch::kFloat64,{config.hidden_width,f},"w1");detail::shape(h.model->b1,torch::kFloat64,{config.hidden_width},"b1");detail::shape(h.model->w2,torch::kFloat64,{config.width,config.hidden_width},"w2");detail::shape(h.model->b2,torch::kFloat64,{config.width},"b2");
      detail::shape(h.model->input_mean,torch::kFloat64,{f},"input_mean");detail::shape(h.model->input_scale,torch::kFloat64,{f},"input_scale");detail::shape(h.model->delta_scale,torch::kFloat64,{config.width},"delta_scale");
      for(const auto& p:h.model->parameters())detail::need(torch::isfinite(p).all().item<bool>()&&p.requires_grad()!=inference_only_,"invalid parameter state/trainability");
      for(const auto& b:h.model->buffers())detail::need(torch::isfinite(b).all().item<bool>()&&!b.requires_grad(),"invalid frozen buffer");
      detail::need(h.model->input_scale.ge(config.std_floor).all().item<bool>()&&h.model->delta_scale.ge(config.std_floor).all().item<bool>()&&torch::equal(h.bound_mean,h.model->input_mean)&&torch::equal(h.bound_scale,h.model->input_scale)&&torch::equal(h.bound_delta,h.model->delta_scale),"frozen transform changed");
      detail::shape(h.training_pair_ids,torch::kInt64,{h.training_rows},"fit_pair_ids");detail::shape(h.training_weights,torch::kFloat64,{h.training_rows},"fit_weights");
      detail::need(torch::isfinite(h.training_weights).all().item<bool>()&&h.training_weights.gt(0).all().item<bool>()&&h.training_weights.sum().item<double>()==h.training_weight,"invalid fitting weights");
      std::set<int64_t> ids;for(int64_t j=0;j<h.training_rows;++j)detail::need(ids.insert(h.training_pair_ids[j].item<int64_t>()).second,"duplicate fitting pair ID");
      if(!inference_only_){detail::shape(h.training_x,torch::kFloat64,{h.training_rows,f},"training_x");detail::shape(h.training_y,torch::kFloat64,{h.training_rows,config.width},"training_y");detail::need(torch::isfinite(h.training_x).all().item<bool>()&&torch::isfinite(h.training_y).all().item<bool>(),"nonfinite training state");validate_optimizer(h);}
    }
  }
  void save_training(const std::string& path)const{detail::need(!inference_only_,"inference cannot be saved as training");torch::serialize::OutputArchive a;write(a,true);detail::publish(a,path);}
  void save_inference(const std::string& path)const{torch::serialize::OutputArchive a;write(a,false);detail::publish(a,path);}
  static std::unique_ptr<Trainer> load_training(std::istream& in,const Config& c,const Identity& id){torch::serialize::InputArchive a;a.load_from(in,torch::kCPU);return read(a,c,id,true);}
  static std::unique_ptr<Trainer> load_inference(std::istream& in,const Config& c,const Identity& id){torch::serialize::InputArchive a;a.load_from(in,torch::kCPU);return read(a,c,id,false);}
  static std::unique_ptr<Trainer> load_training(const std::string& path,const Config& c,const Identity& id){std::ifstream in(path,std::ios::binary);detail::need(bool(in),"cannot open training archive");return load_training(in,c,id);}
  static std::unique_ptr<Trainer> load_inference(const std::string& path,const Config& c,const Identity& id){std::ifstream in(path,std::ios::binary);detail::need(bool(in),"cannot open inference archive");return load_inference(in,c,id);}
 private:
  bool inference_only_{false};
  Trainer(const Config& c,const Identity& id):config(c),identity(id){validate(c);validate(id);}
  HorizonState& at(int64_t horizon){for(auto& h:horizons)if(h.horizon_ticks==horizon)return h;throw std::invalid_argument("unregistered control horizon");}
  const HorizonState* lookup(int64_t horizon)const{for(const auto& h:horizons)if(h.horizon_ticks==horizon)return &h;return nullptr;}
  void initialize(HorizonState& h,const torch::Tensor& mean,const torch::Tensor& scale,const torch::Tensor& residual){
    h.model=ControlModel(config,h.horizon_ticks,mean,scale,residual);h.bound_mean=mean.detach().clone();h.bound_scale=scale.detach().clone();h.bound_delta=residual.detach().clone();
    h.sampler.seed(static_cast<uint64_t>(config.repetition_seed+h.horizon_ticks));h.optimizer=std::make_unique<torch::optim::AdamW>(h.model->parameters(),detail::optimizer_options(config));
  }
  void validate_optimizer(const HorizonState& h)const{
    detail::need(bool(h.optimizer),"missing optimizer");const auto expected=detail::optimizer_options(config);const auto* defaults=dynamic_cast<const torch::optim::AdamWOptions*>(&h.optimizer->defaults());
    detail::need(defaults&&*defaults==expected&&h.optimizer->param_groups().size()==1,"optimizer recipe drift");const auto& group=h.optimizer->param_groups().front();const auto* options=dynamic_cast<const torch::optim::AdamWOptions*>(&group.options());
    const auto parameters=h.model->parameters();detail::need(options&&*options==expected&&group.params().size()==parameters.size(),"optimizer parameter/group drift");std::set<void*> keys;
    for(size_t j=0;j<parameters.size();++j){const auto& p=parameters[j];auto* key=p.unsafeGetTensorImpl();detail::need(group.params()[j].unsafeGetTensorImpl()==key&&keys.insert(key).second,"optimizer parameter order drift");const auto it=h.optimizer->state().find(key);
      if(h.completed_updates==0){detail::need(it==h.optimizer->state().end(),"unexpected initialized optimizer state");continue;}
      detail::need(it!=h.optimizer->state().end(),"missing updated optimizer state");const auto* state=dynamic_cast<const torch::optim::AdamWParamState*>(it->second.get());detail::need(state&&state->step()==h.completed_updates&&!state->max_exp_avg_sq().defined(),"optimizer counter/AMSGrad mismatch");
      for(const auto& moment:{state->exp_avg(),state->exp_avg_sq()}){detail::shape(moment,torch::kFloat64,p.sizes(),"optimizer moment");detail::need(torch::isfinite(moment).all().item<bool>(),"nonfinite optimizer moment");}detail::need(state->exp_avg_sq().ge(0).all().item<bool>(),"negative optimizer variance");
    }for(const auto& s:h.optimizer->state())detail::need(keys.count(s.first),"orphan optimizer state");
  }
  void write(torch::serialize::OutputArchive& a,bool training)const{
    validate_runtime();detail::text(a,"artifact_kind",training?training_kind:inference_kind);detail::text(a,"config",detail::canonical(config));detail::text(a,"identity",detail::canonical(identity));
    a.write("geometry",torch::tensor(std::vector<int64_t>{interval_ms,window_ticks,submitted_rows,excluded_rows},torch::kInt64),true);
    for(size_t i=0;i<horizons.size();++i){const auto& h=horizons[i];torch::serialize::OutputArchive child,module;
      child.write("progress",torch::tensor(std::vector<int64_t>{h.horizon_ticks,h.training_rows,h.completed_updates,h.sampler_draws},torch::kInt64),true);
      child.write("training_weight",torch::tensor(h.training_weight,torch::kFloat64),true);child.write("fit_pair_ids",h.training_pair_ids,true);child.write("fit_weights",h.training_weights,true);h.model->save(module);child.write("model",module);
      if(training){child.write("training_x",h.training_x,true);child.write("training_y",h.training_y,true);std::ostringstream engine;engine<<h.sampler;detail::text(child,"sampler",engine.str());
        // Store AdamW state by stable named parameter instead of pointer IDs.
        for(const auto& p:h.model->named_parameters()){torch::serialize::OutputArchive state;const auto it=h.optimizer->state().find(p.value().unsafeGetTensorImpl());
          state.write("step",torch::tensor(h.completed_updates,torch::kInt64),true);
          if(it!=h.optimizer->state().end()){const auto* value=dynamic_cast<const torch::optim::AdamWParamState*>(it->second.get());state.write("exp_avg",value->exp_avg(),true);state.write("exp_avg_sq",value->exp_avg_sq(),true);}child.write("optimizer_"+p.key(),state);
        }
      }a.write("horizon_"+std::to_string(i),child);
    }
  }
  static std::unique_ptr<Trainer> read(torch::serialize::InputArchive& a,const Config& c,const Identity& id,bool training){
    detail::need(detail::text(a,"artifact_kind")==std::string(training?training_kind:inference_kind),"wrong inference/training archive kind");detail::need(detail::text(a,"config")==detail::canonical(c)&&detail::text(a,"identity")==detail::canonical(id),"archive configuration/identity mismatch");
    auto result=std::unique_ptr<Trainer>(new Trainer(c,id));const auto geometry=detail::tensor(a,"geometry");detail::shape(geometry,torch::kInt64,{4},"geometry");result->interval_ms=geometry[0].item<int64_t>();result->window_ticks=geometry[1].item<int64_t>();result->submitted_rows=geometry[2].item<int64_t>();result->excluded_rows=geometry[3].item<int64_t>();
    for(size_t i=0;i<c.horizons.size();++i){torch::serialize::InputArchive child,module;a.read("horizon_"+std::to_string(i),child);HorizonState h;
      const auto progress=detail::tensor(child,"progress");detail::shape(progress,torch::kInt64,{4},"progress");h.horizon_ticks=progress[0].item<int64_t>();h.training_rows=progress[1].item<int64_t>();h.completed_updates=progress[2].item<int64_t>();h.sampler_draws=progress[3].item<int64_t>();
      const auto weight=detail::tensor(child,"training_weight");detail::shape(weight,torch::kFloat64,{},"training_weight");h.training_weight=weight.item<double>();h.training_pair_ids=detail::tensor(child,"fit_pair_ids");h.training_weights=detail::tensor(child,"fit_weights");
      child.read("model",module);const auto mean=detail::tensor(module,"input_mean"),scale=detail::tensor(module,"input_scale"),residual=detail::tensor(module,"delta_scale");result->initialize(h,mean,scale,residual);h.model->load(module);
      if(training){h.training_x=detail::tensor(child,"training_x");h.training_y=detail::tensor(child,"training_y");std::istringstream sampler(detail::text(child,"sampler"));sampler>>h.sampler;detail::need(!sampler.fail(),"malformed sampler state");sampler>>std::ws;detail::need(sampler.eof(),"trailing sampler state");
        for(const auto& p:h.model->named_parameters()){torch::serialize::InputArchive state;child.read("optimizer_"+p.key(),state);const auto step=detail::tensor(state,"step");detail::shape(step,torch::kInt64,{},"optimizer step");detail::need(step.item<int64_t>()==h.completed_updates,"optimizer step manifest mismatch");
          if(h.completed_updates){auto value=std::make_unique<torch::optim::AdamWParamState>();value->step(h.completed_updates);value->exp_avg(detail::tensor(state,"exp_avg"));value->exp_avg_sq(detail::tensor(state,"exp_avg_sq"));h.optimizer->state().emplace(p.value().unsafeGetTensorImpl(),std::move(value));}
          else detail::need(state.keys().size()==1,"unexpected initial optimizer moments");
        }
      }else {for(auto& p:h.model->parameters())p.set_requires_grad(false);h.model->eval();h.optimizer.reset();}
      result->horizons.push_back(std::move(h));
    }result->inference_only_=!training;result->validate_runtime();return result;
  }
};
} // namespace cuwacunu::networks::direct_embedding_control::v1
