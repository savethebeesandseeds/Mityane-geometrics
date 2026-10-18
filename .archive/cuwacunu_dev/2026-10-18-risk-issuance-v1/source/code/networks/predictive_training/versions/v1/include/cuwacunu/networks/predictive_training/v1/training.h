// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/networks/state_representation/v1/state.h"
#include "cuwacunu/networks/latent_dynamics/v1/dynamics.h"
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <torch/cuda.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>

namespace cuwacunu::networks::predictive_training::v1 {
namespace sr=cuwacunu::networks::state_representation::v1;
namespace ld=cuwacunu::networks::latent_dynamics::v1;
inline constexpr const char *training_kind="predictive_state_training_v1";
inline constexpr const char *inference_kind="predictive_state_inference_v1";
inline constexpr const char *estimator_contract="distinct-origin-equal-weight-centered-sample-covariance-full-graph-v1";
enum class Arm : int64_t { frozen_state=0, blocked_prediction=1, full=2 };
struct Config {
  sr::Config state;
  ld::Config dynamics;
  Arm arm{Arm::full};
  double lambda_prediction{1.},lambda_variance{1.},lambda_covariance{.01};
  double variance_floor{1.},covariance_epsilon{1e-4},ema_decay{.99};
  double learning_rate{1e-3},beta1{.9},beta2{.999},optimizer_epsilon{1e-8},weight_decay{0.},clip_norm{0.};
  int64_t sampler_seed{1};
};
// Identities bind immutable manifests, not paths discovered opportunistically.
// The external runner supplies hashes, role/fit cutoffs and export layout.
struct Identity {
  std::string encoder_checkpoint,encoder_scaler,embedding_transform,reference_scale;
  std::string source,split,pair_ledger,sampler_recipe,loss_recipe,serving_layout;
};
struct Progress {
  int64_t attempted_batches{0},completed_updates{0},online_updates{0},dynamics_updates{0},ema_updates{0},sampler_draws{0};
};
struct RegularizerPool {
  torch::Tensor embeddings; // [N,D]; N is independent of optimizer B.
  torch::Tensor origin_ids; // [N] distinct completed-candle origins, never asset IDs.
};
struct PredictiveBatch {
  torch::Tensor history_embeddings,future_embeddings,prediction_valid;
  RegularizerPool pool;
};
struct RegularizerLoss {
  torch::Tensor variance,covariance,sample_covariance;
  int64_t n_cov{0},rank_upper_bound{0};
};
struct StepReceipt {
  bool updated{false};
  double prediction_loss{0.},variance_loss{0.},covariance_loss{0.},total_loss{0.};
  int64_t eligible_predictions{0},n_cov{0},rank_upper_bound{0};
};
inline void validate(const Config &c) {
  sr::validate(c.state); ld::validate(c.dynamics);
  TORCH_CHECK(c.state.state_width==c.dynamics.state_width && c.state.device==c.dynamics.device &&
    c.state.dtype==c.dynamics.dtype,"[predictive v1] component geometry/device/dtype mismatch");
  TORCH_CHECK(c.arm==Arm::frozen_state || c.arm==Arm::blocked_prediction || c.arm==Arm::full,
              "[predictive v1] unknown learning arm");
  for(double x:{c.lambda_prediction,c.lambda_variance,c.lambda_covariance,c.variance_floor,
                c.weight_decay,c.clip_norm}) TORCH_CHECK(std::isfinite(x)&&x>=0,"[predictive v1] nonnegative finite setting required");
  TORCH_CHECK(c.lambda_prediction>0 && std::isfinite(c.covariance_epsilon)&&c.covariance_epsilon>0 &&
    std::isfinite(c.ema_decay)&&c.ema_decay>=0&&c.ema_decay<1 &&
    std::isfinite(c.learning_rate)&&c.learning_rate>0 &&
    std::isfinite(c.beta1)&&c.beta1>=0&&c.beta1<1 && std::isfinite(c.beta2)&&c.beta2>=0&&c.beta2<1 &&
    std::isfinite(c.optimizer_epsilon)&&c.optimizer_epsilon>0 && c.sampler_seed>=0,
    "[predictive v1] invalid predictive/EMA/optimizer/sampler recipe");
}
inline void validate(const Identity &i) {
  for(const auto *s:{&i.encoder_checkpoint,&i.encoder_scaler,&i.embedding_transform,&i.reference_scale,
    &i.source,&i.split,&i.pair_ledger,&i.sampler_recipe,&i.loss_recipe,&i.serving_layout})
    TORCH_CHECK(!s->empty()&&s->size()<=16384&&s->find('\0')==std::string::npos,
                "[predictive v1] every immutable identity is required and bounded");
}
// No memory bank, duplicated origin, detached activation or per-microbatch
// averaging is hidden here. All N activations share one centering operation.
inline RegularizerLoss regularizer_loss(const torch::Tensor &states,const torch::Tensor &origin_ids,
    const torch::Tensor &reference_scale,double floor,double epsilon) {
  TORCH_CHECK(states.dim()==2 && states.size(0)>=2 && states.size(1)>0 &&
    reference_scale.dim()==1&&reference_scale.size(0)==states.size(1)&&
    reference_scale.device()==states.device()&&reference_scale.scalar_type()==states.scalar_type()&&
    !reference_scale.requires_grad()&&torch::isfinite(reference_scale).all().item<bool>()&&
    reference_scale.gt(0).all().item<bool>()&&torch::isfinite(states).all().item<bool>()&&
    std::isfinite(floor)&&floor>=0&&std::isfinite(epsilon)&&epsilon>0,
    "[predictive v1] valid full-state covariance population/reference required");
  TORCH_CHECK(origin_ids.dim()==1&&origin_ids.size(0)==states.size(0)&&origin_ids.scalar_type()==torch::kInt64,
              "[predictive v1] one int64 origin per regularizer row required");
  auto ids=origin_ids.to(torch::kCPU).contiguous();
  std::set<int64_t> distinct;
  for(int64_t j=0;j<ids.numel();++j) TORCH_CHECK(distinct.insert(ids[j].item<int64_t>()).second,
    "[predictive v1] repeated origins cannot count as independent covariance rows");
  auto z=states/reference_scale;
  auto centered=z-z.mean(0,true);
  auto covariance=torch::matmul(centered.transpose(0,1),centered)/(z.size(0)-1);
  auto diagonal=covariance.diag();
  auto variance=torch::relu(floor-torch::sqrt(diagonal+epsilon)).mean();
  auto off_diagonal=covariance-torch::diag(diagonal);
  auto penalty=off_diagonal.square().sum()/z.size(1);
  return {variance,penalty,covariance,z.size(0),std::min(z.size(1),z.size(0)-1)};
}

namespace detail {
inline std::string canonical(const Config &c) {
  validate(c);std::ostringstream s;s<<std::setprecision(17)<<"predictive-v1\n";
  s<<c.state.embedding_width<<' '<<c.state.state_width<<' '<<c.state.identity_initialization<<' '
   <<c.state.device.str()<<' '<<static_cast<int>(c.state.dtype)<<'\n';
  s<<c.dynamics.history_length<<' '<<c.dynamics.residual<<' '<<c.dynamics.horizons.size();
  for(auto h:c.dynamics.horizons)s<<' '<<h;
  s<<'\n'<<static_cast<int64_t>(c.arm)<<' '<<c.lambda_prediction<<' '<<c.lambda_variance<<' '
   <<c.lambda_covariance<<' '<<c.variance_floor<<' '<<c.covariance_epsilon<<' '<<c.ema_decay<<' '
   <<c.learning_rate<<' '<<c.beta1<<' '<<c.beta2<<' '<<c.optimizer_epsilon<<' '<<c.weight_decay<<' '
   <<c.clip_norm<<' '<<c.sampler_seed<<'\n'<<estimator_contract<<'\n'
   <<"MSE-reference-scaled-valid-pair-mean;EMA-after-success;skip-all-if-no-valid-prediction;clip-separately-by-component;lambda-ret=0\n";
  return s.str();
}
inline std::string canonical(const Identity &i) {
  validate(i);std::ostringstream s;
  for(const auto *v:{&i.encoder_checkpoint,&i.encoder_scaler,&i.embedding_transform,&i.reference_scale,
    &i.source,&i.split,&i.pair_ledger,&i.sampler_recipe,&i.loss_recipe,&i.serving_layout}) s<<v->size()<<':'<<*v<<'\n';
  return s.str();
}
inline void write_text(torch::serialize::OutputArchive &a,const std::string &k,const std::string &value) {
  a.write(k,torch::IValue(value));
}
inline std::string read_text(torch::serialize::InputArchive &a,const std::string &k) {
  torch::IValue v;a.read(k,v);TORCH_CHECK(v.isString(),"[predictive v1] malformed text: ",k);return v.toStringRef();
}
inline torch::Tensor read_tensor(torch::serialize::InputArchive &a,const std::string &k) {
  torch::Tensor v;a.read(k,v,true);return v;
}
inline void validate_progress(const Progress &p) {
  TORCH_CHECK(p.attempted_batches>=0&&p.attempted_batches<std::numeric_limits<int64_t>::max()&&
    p.completed_updates>=0&&p.completed_updates<=p.attempted_batches&&
    p.online_updates>=0&&p.online_updates<=p.completed_updates&&p.dynamics_updates==p.completed_updates&&
    p.ema_updates==p.completed_updates&&p.sampler_draws>=0,
    "[predictive v1] invalid update/attempt/teacher/sampler counters");
}
inline void write_progress(torch::serialize::OutputArchive &a,const Progress &p) {
  validate_progress(p);
  a.write("progress",torch::tensor(std::vector<int64_t>{p.attempted_batches,p.completed_updates,
    p.online_updates,p.dynamics_updates,p.ema_updates,p.sampler_draws},torch::kInt64),true);
}
inline Progress read_progress(torch::serialize::InputArchive &a) {
  auto p=read_tensor(a,"progress").to(torch::kCPU).contiguous();
  TORCH_CHECK(p.scalar_type()==torch::kInt64&&p.dim()==1&&p.numel()==6,"[predictive v1] malformed progress");
  auto v=p.const_data_ptr<int64_t>();Progress r{v[0],v[1],v[2],v[3],v[4],v[5]};validate_progress(r);return r;
}
inline void publish(torch::serialize::OutputArchive &a,const std::string &filename) {
  namespace fs=std::filesystem;const fs::path destination(filename);
  TORCH_CHECK(!destination.empty()&&fs::is_directory(destination.parent_path().empty()?fs::path("."):destination.parent_path()),
    "[predictive v1] existing checkpoint directory required");
  TORCH_CHECK(!fs::exists(destination),"[predictive v1] immutable checkpoint already exists: ",filename);
  static std::atomic<uint64_t> serial{0};
  auto temporary=destination;temporary+=".tmp-"+std::to_string(::getpid())+"-"+std::to_string(serial.fetch_add(1));
  // The parent belongs to this caller; O_EXCL reserves a private staging name.
  const int fd=::open(temporary.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);
  TORCH_CHECK(fd>=0,"[predictive v1] cannot reserve checkpoint staging path");::close(fd);
  try {a.save_to(temporary.string());fs::create_hard_link(temporary,destination);fs::remove(temporary);}
  catch(...) {std::error_code error;fs::remove(temporary,error);throw;}
}
inline torch::optim::AdamWOptions optimizer_options(const Config &c) {
  return torch::optim::AdamWOptions(c.learning_rate).betas(std::make_tuple(c.beta1,c.beta2))
    .eps(c.optimizer_epsilon).weight_decay(c.weight_decay).amsgrad(false);
}
inline void save_rng(torch::serialize::OutputArchive &a) {
  a.write("torch_cpu_rng",at::detail::getDefaultCPUGenerator().get_state(),true);
  const auto count=static_cast<int64_t>(torch::cuda::device_count());
  a.write("cuda_rng_count",torch::tensor(count,torch::kInt64),true);
  for(int64_t j=0;j<count;++j) a.write("torch_cuda_rng_"+std::to_string(j),
    at::cuda::detail::getDefaultCUDAGenerator(j).get_state(),true);
}
inline void restore_rng(torch::serialize::InputArchive &a) {
  auto cpu=read_tensor(a,"torch_cpu_rng").to(torch::kCPU);
  auto count_tensor=read_tensor(a,"cuda_rng_count");
  TORCH_CHECK(count_tensor.scalar_type()==torch::kInt64&&count_tensor.dim()==0,
              "[predictive v1] malformed CUDA RNG count");
  const auto count=count_tensor.item<int64_t>();
  TORCH_CHECK(count==static_cast<int64_t>(torch::cuda::device_count()),"[predictive v1] CUDA RNG device inventory changed");
  std::vector<torch::Tensor> cuda_states;
  for(int64_t j=0;j<count;++j) cuda_states.push_back(read_tensor(a,"torch_cuda_rng_"+std::to_string(j)).to(torch::kCPU));
  TORCH_CHECK(cpu.scalar_type()==torch::kUInt8&&cpu.dim()==1,"[predictive v1] malformed CPU RNG state");
  for(const auto &s:cuda_states) TORCH_CHECK(s.scalar_type()==torch::kUInt8&&s.dim()==1,"[predictive v1] malformed CUDA RNG state");
  auto generator=at::detail::getDefaultCPUGenerator();generator.set_state(cpu);
  for(int64_t j=0;j<count;++j) {auto gen=at::cuda::detail::getDefaultCUDAGenerator(j);gen.set_state(cuda_states[j]);}
}
} // namespace detail

struct InferenceSnapshot {
  Config config;
  Identity identity;
  sr::AffineState online{nullptr},target{nullptr};
  ld::AffineDynamics dynamics{nullptr};
  torch::Tensor reference_scale;
  Progress progress;
};

class Trainer {
public:
  const Config config;
  const Identity identity;
  sr::AffineState online{nullptr},target{nullptr};
  ld::AffineDynamics dynamics{nullptr};
  torch::Tensor reference_scale;
  Progress progress;
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::mt19937_64 sampler;
  Trainer(const Config &c,const Identity &i,torch::Tensor mean={},torch::Tensor scale={},torch::Tensor reference={})
    :config(c),identity(i),sampler(static_cast<uint64_t>(c.sampler_seed)) {
    validate(c);validate(i);
    online=sr::AffineState(c.state,mean,scale);target=sr::frozen_copy(online);dynamics=ld::AffineDynamics(c.dynamics);
    if(c.arm==Arm::frozen_state) {online->eval();for(auto &p:online->parameters())p.set_requires_grad(false);}
    auto options=torch::TensorOptions().device(c.state.device).dtype(c.state.dtype);
    if(!reference.defined())reference=torch::ones({c.state.state_width},options);
    TORCH_CHECK(reference.dim()==1&&reference.size(0)==c.state.state_width&&
      torch::isfinite(reference).all().item<bool>()&&reference.gt(0).all().item<bool>(),"[predictive v1] positive fixed state reference required");
    reference_scale=reference.detach().to(options).clone();
    bound_mean_=online->embedding_mean.detach().clone();bound_scale_=online->embedding_scale.detach().clone();
    bound_reference_=reference_scale.detach().clone();
    optimizer=std::make_unique<torch::optim::AdamW>(trainable_parameters(),detail::optimizer_options(c));
  }
  std::vector<torch::Tensor> trainable_parameters() const {
    std::vector<torch::Tensor> parameters;
    if(config.arm!=Arm::frozen_state) for(const auto &p:online->parameters())parameters.push_back(p);
    for(const auto &p:dynamics->parameters())parameters.push_back(p);
    return parameters;
  }
  std::vector<int64_t> sample_indices(int64_t population,int64_t count) {
    TORCH_CHECK(population>0&&count>=0&&count<=std::numeric_limits<int64_t>::max()-progress.sampler_draws,
      "[predictive v1] invalid sampler request");
    std::uniform_int_distribution<int64_t> distribution(0,population-1);std::vector<int64_t> rows;
    rows.reserve(static_cast<size_t>(count));for(int64_t j=0;j<count;++j)rows.push_back(distribution(sampler));
    progress.sampler_draws+=count;return rows;
  }
  StepReceipt step(const PredictiveBatch &batch) {
    validate_runtime();
    TORCH_CHECK(config.state.device.is_cuda(),"[predictive v1] neural training requires managed-container CUDA");
    const auto &h=batch.history_embeddings;const auto &f=batch.future_embeddings;const auto &mask=batch.prediction_valid;
    const auto q=static_cast<int64_t>(config.dynamics.horizons.size());
    TORCH_CHECK(h.dim()==3&&h.size(0)>0&&h.size(1)==config.dynamics.history_length&&h.size(2)==config.state.embedding_width&&
      f.dim()==3&&f.size(0)==h.size(0)&&f.size(1)==q&&f.size(2)==config.state.embedding_width&&
      mask.dim()==2&&mask.size(0)==h.size(0)&&mask.size(1)==q&&mask.scalar_type()==torch::kBool,
      "[predictive v1] batch axes/validity mismatch");
    for(const auto &v:{h,f}) TORCH_CHECK(v.device()==config.state.device&&v.scalar_type()==config.state.dtype&&
      torch::isfinite(v).all().item<bool>(),"[predictive v1] finite saved embeddings on declared CUDA device/dtype required");
    TORCH_CHECK(mask.device()==config.state.device,"[predictive v1] validity mask device mismatch");
    StepReceipt receipt;receipt.eligible_predictions=mask.sum().item<int64_t>();
    optimizer->zero_grad(true);
    if(!receipt.eligible_predictions) {++progress.attempted_batches;return receipt;}
    online->train(config.arm!=Arm::frozen_state);dynamics->train();target->eval();
    // This explicit boundary protects E and its scaler even when a caller
    // accidentally supplies embeddings carrying an encoder gradient graph.
    auto history=online->forward(h.detach());
    auto prediction=dynamics->forward(config.arm==Arm::full?history:history.detach());
    torch::Tensor future;
    {torch::NoGradGuard guard;future=target->forward(f.detach());}
    auto pair_error=((prediction-future.detach())/reference_scale).square().mean(-1);
    auto prediction_loss=pair_error.masked_select(mask).mean();
    auto variance_loss=torch::zeros({},prediction_loss.options());auto covariance_loss=torch::zeros_like(variance_loss);
    if(config.lambda_variance>0||config.lambda_covariance>0) {
      TORCH_CHECK(batch.pool.embeddings.dim()==2&&batch.pool.embeddings.size(1)==config.state.embedding_width&&
        batch.pool.embeddings.device()==config.state.device&&batch.pool.embeddings.scalar_type()==config.state.dtype,
        "[predictive v1] explicit separately sampled regularizer embedding pool required");
      const auto losses=regularizer_loss(online->forward(batch.pool.embeddings.detach()),batch.pool.origin_ids,
        reference_scale,config.variance_floor,config.covariance_epsilon);
      variance_loss=losses.variance;covariance_loss=losses.covariance;
      receipt.n_cov=losses.n_cov;receipt.rank_upper_bound=losses.rank_upper_bound;
    }
    auto loss=config.lambda_prediction*prediction_loss+config.lambda_variance*variance_loss+
      config.lambda_covariance*covariance_loss;
    TORCH_CHECK(loss.is_cuda()&&torch::isfinite(loss).item<bool>(),"[predictive v1] finite actual CUDA loss required");
    loss.backward();
    bool online_gradient=false;
    for(const auto &p:trainable_parameters()) if(p.grad().defined()) {
      TORCH_CHECK(p.is_cuda()&&p.grad().is_cuda()&&torch::isfinite(p.grad()).all().item<bool>(),
        "[predictive v1] finite actual CUDA gradients required");
    }
    for(const auto &p:online->parameters()) online_gradient|=p.grad().defined();
    for(const auto &p:target->parameters()) TORCH_CHECK(!p.requires_grad()&&!p.grad().defined(),"[predictive v1] target received gradient");
    // Separate component clipping prevents a G gradient from changing A's
    // clipping factor between full and blocked arms.
    if(config.clip_norm>0) {
      if(config.arm!=Arm::frozen_state)torch::nn::utils::clip_grad_norm_(online->parameters(),config.clip_norm);
      torch::nn::utils::clip_grad_norm_(dynamics->parameters(),config.clip_norm);
    }
    optimizer->step();
    {torch::NoGradGuard guard;const auto current=online->named_parameters();
      for(auto &p:target->named_parameters())p.value().add_((current[p.key()]-p.value())*(1-config.ema_decay));}
    ++progress.attempted_batches;++progress.completed_updates;++progress.dynamics_updates;++progress.ema_updates;
    if(online_gradient)++progress.online_updates;
    receipt.updated=true;receipt.prediction_loss=prediction_loss.detach().item<double>();
    receipt.variance_loss=variance_loss.detach().item<double>();receipt.covariance_loss=covariance_loss.detach().item<double>();
    receipt.total_loss=loss.detach().item<double>();validate_runtime();(void)validate_optimizer();return receipt;
  }
  void save_training(const std::string &filename) const {
    validate_runtime();torch::serialize::OutputArchive archive;write_common(archive,training_kind);
    const auto steps=validate_optimizer();
    torch::serialize::OutputArchive state;optimizer->save(state);archive.write("optimizer",state);
    detail::write_text(archive,"optimizer_parameter_order",parameter_order());
    detail::write_text(archive,"optimizer_pointer_order",optimizer_pointer_order());
    archive.write("optimizer_steps",torch::tensor(steps,torch::kInt64),true);
    std::ostringstream stream;stream<<sampler;detail::write_text(archive,"sampler_state",stream.str());
    detail::save_rng(archive);detail::publish(archive,filename);
  }
  void save_inference(const std::string &filename) const {
    validate_runtime();torch::serialize::OutputArchive archive;write_common(archive,inference_kind);detail::publish(archive,filename);
  }
  static std::unique_ptr<Trainer> load_training(const std::string &filename,const Config &expected,const Identity &identity) {
    torch::serialize::InputArchive archive;archive.load_from(filename,expected.state.device);
    auto restored=read_common(archive,training_kind,expected,identity);
    TORCH_CHECK(detail::read_text(archive,"optimizer_parameter_order")==restored->parameter_order(),
      "[predictive v1] optimizer named parameter order changed");
    torch::serialize::InputArchive state;archive.read("optimizer",state);
    restored->validate_optimizer_archive(state,detail::read_text(archive,"optimizer_pointer_order"));
    restored->optimizer->load(state);
    const auto actual=restored->validate_optimizer();auto saved=detail::read_tensor(archive,"optimizer_steps").to(torch::kCPU).contiguous();
    TORCH_CHECK(saved.scalar_type()==torch::kInt64&&saved.dim()==1&&saved.numel()==static_cast<int64_t>(actual.size())&&
      std::equal(actual.begin(),actual.end(),saved.const_data_ptr<int64_t>()),"[predictive v1] AdamW step manifest mismatch");
    const auto sampler_state=detail::read_text(archive,"sampler_state");std::istringstream stream(sampler_state);
    stream>>restored->sampler;TORCH_CHECK(!stream.fail(),"[predictive v1] malformed sampler engine state");
    stream>>std::ws;TORCH_CHECK(stream.eof(),"[predictive v1] trailing sampler engine data");
    detail::restore_rng(archive);return restored;
  }
  static InferenceSnapshot load_inference(const std::string &filename,const Config &expected,const Identity &identity) {
    torch::serialize::InputArchive archive;archive.load_from(filename,expected.state.device);
    auto restored=read_common(archive,inference_kind,expected,identity);
    for(const auto &model:{static_cast<torch::nn::Module*>(restored->online.get()),
      static_cast<torch::nn::Module*>(restored->target.get()),static_cast<torch::nn::Module*>(restored->dynamics.get())}) {
      model->eval();for(auto &p:model->parameters())p.set_requires_grad(false);
    }
    return {expected,identity,restored->online,restored->target,restored->dynamics,restored->reference_scale,restored->progress};
  }
  void validate_runtime() const {
    validate(config);validate(identity);detail::validate_progress(progress);online->validate_state();target->validate_state();dynamics->validate_state();
    TORCH_CHECK(reference_scale.dim()==1&&reference_scale.size(0)==config.state.state_width&&
      reference_scale.device()==config.state.device&&reference_scale.scalar_type()==config.state.dtype&&
      !reference_scale.requires_grad()&&torch::isfinite(reference_scale).all().item<bool>()&&reference_scale.gt(0).all().item<bool>(),
      "[predictive v1] invalid fixed reference scale");
    TORCH_CHECK(torch::equal(online->embedding_mean,target->embedding_mean)&&
      torch::equal(online->embedding_scale,target->embedding_scale),"[predictive v1] online/target transform drift");
    TORCH_CHECK(torch::equal(online->embedding_mean,bound_mean_)&&torch::equal(online->embedding_scale,bound_scale_)&&
      torch::equal(reference_scale,bound_reference_),"[predictive v1] admitted frozen transform/reference mutated");
    for(const auto &model:{online,target}) TORCH_CHECK(model->config.embedding_width==config.state.embedding_width&&
      model->config.state_width==config.state.state_width&&model->config.identity_initialization==config.state.identity_initialization&&
      model->config.device==config.state.device&&model->config.dtype==config.state.dtype,
      "[predictive v1] state module configuration drift");
    TORCH_CHECK(dynamics->config.state_width==config.dynamics.state_width&&
      dynamics->config.history_length==config.dynamics.history_length&&dynamics->config.horizons==config.dynamics.horizons&&
      dynamics->config.residual==config.dynamics.residual&&dynamics->config.device==config.dynamics.device&&
      dynamics->config.dtype==config.dynamics.dtype,"[predictive v1] dynamics module configuration drift");
    for(const auto &p:online->parameters()) TORCH_CHECK(p.requires_grad()==(config.arm!=Arm::frozen_state),"[predictive v1] online trainability drift");
    for(const auto &p:target->parameters()) TORCH_CHECK(!p.requires_grad()&&!p.grad().defined(),"[predictive v1] detached frozen teacher required");
    for(const auto &p:dynamics->parameters()) TORCH_CHECK(p.requires_grad(),"[predictive v1] dynamics trainability drift");
    TORCH_CHECK(config.arm!=Arm::frozen_state||progress.online_updates==0,"[predictive v1] frozen online update counter must remain zero");
  }
private:
  torch::Tensor bound_mean_,bound_scale_,bound_reference_;
  std::string parameter_order() const {
    std::string text;
    if(config.arm!=Arm::frozen_state)for(const auto &p:online->named_parameters())text+="online."+p.key()+"\n";
    for(const auto &p:dynamics->named_parameters())text+="dynamics."+p.key()+"\n";
    return text;
  }
  std::string optimizer_pointer_order() const {
    std::string s;
    for(const auto &p:optimizer->param_groups().front().params())
      s+=std::to_string(reinterpret_cast<uintptr_t>(p.unsafeGetTensorImpl()))+'\n';
    return s;
  }
  std::vector<int64_t> validate_optimizer() const {
    const auto expected=detail::optimizer_options(config);const auto parameters=trainable_parameters();
    const auto *defaults=dynamic_cast<const torch::optim::AdamWOptions*>(&optimizer->defaults());
    TORCH_CHECK(defaults&&*defaults==expected&&optimizer->param_groups().size()==1,"[predictive v1] AdamW recipe/group drift");
    const auto &group=optimizer->param_groups().front();const auto *options=dynamic_cast<const torch::optim::AdamWOptions*>(&group.options());
    TORCH_CHECK(options&&*options==expected&&group.params().size()==parameters.size(),"[predictive v1] AdamW options/count mismatch");
    std::set<void*> known;std::vector<int64_t> steps;int64_t maximum=0;
    const auto online_count=config.arm==Arm::frozen_state?size_t{0}:online->parameters().size();
    for(size_t j=0;j<parameters.size();++j) {
      const auto &p=parameters[j];auto *key=p.unsafeGetTensorImpl();
      const auto expected_step=j<online_count?progress.online_updates:progress.dynamics_updates;
      TORCH_CHECK(group.params()[j].unsafeGetTensorImpl()==key&&known.insert(key).second,"[predictive v1] optimizer order/duplicate mismatch");
      const auto it=optimizer->state().find(key);
      if(it==optimizer->state().end()) {
        TORCH_CHECK(expected_step==0,"[predictive v1] missing AdamW state for an updated component");
        steps.push_back(-1);continue;
      }
      const auto *s=dynamic_cast<const torch::optim::AdamWParamState*>(it->second.get());
      TORCH_CHECK(s&&s->step()>0&&s->step()==expected_step&&!s->max_exp_avg_sq().defined(),"[predictive v1] invalid AdamW component step/state");
      for(const auto &moment:{s->exp_avg(),s->exp_avg_sq()}) TORCH_CHECK(moment.defined()&&
        moment.device()==p.device()&&moment.scalar_type()==p.scalar_type()&&moment.sizes()==p.sizes()&&
        torch::isfinite(moment).all().item<bool>(),"[predictive v1] invalid AdamW moments");
      TORCH_CHECK(s->exp_avg_sq().ge(0).all().item<bool>(),"[predictive v1] negative second moment");
      steps.push_back(s->step());maximum=std::max(maximum,s->step());
    }
    for(const auto &s:optimizer->state())TORCH_CHECK(known.count(s.first),"[predictive v1] orphan AdamW state");
    TORCH_CHECK(maximum==progress.completed_updates,"[predictive v1] update counter differs from AdamW state");return steps;
  }
  void validate_optimizer_archive(torch::serialize::InputArchive &a,const std::string &expected_keys) const {
    torch::serialize::InputArchive groups,group,states;a.read("param_groups",groups);a.read("state",states);
    torch::Tensor count;groups.read("param_groups/size",count);
    TORCH_CHECK(count.scalar_type()==torch::kInt64&&count.dim()==0&&count.item<int64_t>()==1,"[predictive v1] one archived optimizer group required");
    groups.read("param_groups/0",group);torch::Tensor size;group.read("params/size",size);
    const auto n=static_cast<int64_t>(trainable_parameters().size());
    TORCH_CHECK(size.scalar_type()==torch::kInt64&&size.dim()==0&&size.item<int64_t>()==n,"[predictive v1] archived optimizer count mismatch");
    std::set<std::string> keys;std::string actual;
    for(int64_t j=0;j<n;++j) {
      torch::IValue v;group.read("params/"+std::to_string(j),v);
      TORCH_CHECK(v.isString(),"[predictive v1] optimizer pointer must be text");std::string key=v.toStringRef();
      TORCH_CHECK(!key.empty()&&std::all_of(key.begin(),key.end(),[](char ch){return ch>='0'&&ch<='9';})&&
        keys.insert(key).second,"[predictive v1] malformed/duplicate optimizer key");actual+=key+'\n';
    }
    TORCH_CHECK(actual==expected_keys,"[predictive v1] archived optimizer mapping differs");
    for(const auto &key:states.keys())TORCH_CHECK(keys.count(key),"[predictive v1] orphan archived optimizer state");
  }
  void write_common(torch::serialize::OutputArchive &a,const char *kind) const {
    detail::write_text(a,"artifact_kind",kind);detail::write_text(a,"config",detail::canonical(config));
    detail::write_text(a,"identity",detail::canonical(identity));detail::write_progress(a,progress);
    torch::serialize::OutputArchive o,t,d;online->save(o);target->save(t);dynamics->save(d);
    a.write("online",o);a.write("target",t);a.write("dynamics",d);a.write("reference_scale",reference_scale,true);
    a.write("embedding_mean",online->embedding_mean,true);a.write("embedding_scale",online->embedding_scale,true);
  }
  static std::unique_ptr<Trainer> read_common(torch::serialize::InputArchive &a,const char *kind,
      const Config &expected,const Identity &identity) {
    TORCH_CHECK(detail::read_text(a,"artifact_kind")==kind,"[predictive v1] wrong inference/training archive kind");
    TORCH_CHECK(detail::read_text(a,"config")==detail::canonical(expected),"[predictive v1] exact config/arm/loss/optimizer/EMA/estimator mismatch");
    TORCH_CHECK(detail::read_text(a,"identity")==detail::canonical(identity),"[predictive v1] source/split/pair/encoder/transform/reference/serving identity mismatch");
    const auto mean=detail::read_tensor(a,"embedding_mean"),scale=detail::read_tensor(a,"embedding_scale");
    const auto reference=detail::read_tensor(a,"reference_scale");
    auto r=std::make_unique<Trainer>(expected,identity,mean,scale,reference);r->progress=detail::read_progress(a);
    torch::serialize::InputArchive o,t,d;a.read("online",o);a.read("target",t);a.read("dynamics",d);
    r->online->load(o);r->target->load(t);r->dynamics->load(d);
    TORCH_CHECK(torch::equal(mean,r->online->embedding_mean)&&torch::equal(scale,r->online->embedding_scale),
      "[predictive v1] duplicate frozen-transform fields disagree");
    r->validate_runtime();return r;
  }
};
inline InferenceSnapshot load_inference(const std::string &filename,const Config &config,const Identity &identity) {
  return Trainer::load_inference(filename,config,identity);
}
} // namespace cuwacunu::networks::predictive_training::v1
