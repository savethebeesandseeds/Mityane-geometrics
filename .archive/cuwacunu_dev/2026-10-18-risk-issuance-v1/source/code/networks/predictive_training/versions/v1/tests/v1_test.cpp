// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Artificial engineering witnesses only; no market training or quality claims.
#include "cuwacunu/networks/predictive_training/v1/training.h"
#include "cuwacunu/networks/predictive_training/v1/serving.h"
#include <torch/cuda.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

namespace pt=cuwacunu::networks::predictive_training::v1;
namespace sr=cuwacunu::networks::state_representation::v1;
namespace ld=cuwacunu::networks::latent_dynamics::v1;
namespace fs=std::filesystem;
namespace {
void check(bool value,const char *message) {TORCH_CHECK(value,"[predictive v1 test] ",message);}
template<class F> void rejects(F action,const char *message) {
  bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}check(rejected,message);
}
bool exact(const torch::Tensor &a,const torch::Tensor &b) {
  if(!a.defined()||!b.defined())return a.defined()==b.defined();
  auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
  return x.scalar_type()==y.scalar_type()&&x.sizes()==y.sizes()&&
    (!x.numel()||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.numel()*x.element_size())==0);
}
void close(const torch::Tensor &a,const torch::Tensor &b,const char *message) {
  check(torch::allclose(a,b,1e-11,1e-11),message);
}
struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    auto pattern=(fs::temp_directory_path()/"predictive-v1-XXXXXX").string();
    std::vector<char> text(pattern.begin(),pattern.end());text.push_back('\0');
    const auto created=::mkdtemp(text.data());check(created,"private fixture directory");path=created;
  }
  ~TemporaryDirectory(){std::error_code error;fs::remove_all(path,error);}
};
std::vector<char> bytes(const fs::path &path) {
  std::ifstream f(path,std::ios::binary);return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
pt::Config config(pt::Arm arm) {
  pt::Config c;c.state.embedding_width=3;c.state.state_width=3;c.state.dtype=torch::kFloat64;
  c.dynamics.state_width=3;c.dynamics.history_length=2;c.dynamics.horizons={1,9};c.dynamics.dtype=torch::kFloat64;
  c.arm=arm;c.lambda_prediction=1.;c.lambda_variance=.3;c.lambda_covariance=.2;
  c.variance_floor=1.4;c.ema_decay=.8;c.learning_rate=.003;c.clip_norm=.7;c.weight_decay=.01;c.sampler_seed=903;
  return c;
}
pt::Identity identity() {
  return {"artificial-encoder-sha","artificial-scaler-sha","artificial-transform-sha","artificial-reference-sha",
    "artificial-source-sha","TRAIN-role-sha","pair-ledger-sha","declared-mt19937-64-sampler-v1",
    "no-retention-auxiliary-v1","online-history-target-teacher-v1"};
}
torch::Tensor values(int64_t n,const torch::TensorOptions &o,double phase=0.) {
  const auto a=torch::arange(n,o);return (a*.37+phase).sin()+.31*(a*.71-phase).cos()+a*.013;
}
pt::PredictiveBatch batch(const pt::Config &c) {
  auto o=torch::TensorOptions().device(c.state.device).dtype(c.state.dtype);
  return {values(24,o).reshape({4,2,3}).set_requires_grad(true),
    values(24,o,.8).reshape({4,2,3}).set_requires_grad(true),torch::ones({4,2},o.dtype(torch::kBool)),
    {values(21,o,1.7).reshape({7,3}).set_requires_grad(true),torch::arange(7,torch::kInt64)}};
}
std::vector<torch::Tensor> snapshot(const torch::nn::Module &model) {
  std::vector<torch::Tensor> result;for(const auto &p:model.parameters())result.push_back(p.detach().clone());return result;
}
bool same(const std::vector<torch::Tensor> &before,const torch::nn::Module &after) {
  const auto now=after.parameters();if(before.size()!=now.size())return false;
  for(size_t j=0;j<now.size();++j)if(!exact(before[j],now[j]))return false;
  return true;
}
void same_models(const pt::Trainer &a,const pt::Trainer &b) {
  check(same(snapshot(*a.online),*b.online)&&same(snapshot(*a.target),*b.target)&&same(snapshot(*a.dynamics),*b.dynamics),
    "all online/target/dynamics tensor bytes exact");
  check(exact(a.online->embedding_mean,b.online->embedding_mean)&&exact(a.online->embedding_scale,b.online->embedding_scale)&&
    exact(a.reference_scale,b.reference_scale),"all frozen transform/reference tensor bytes exact");
}
void same_optimizers(const pt::Trainer &a,const pt::Trainer &b) {
  auto x=a.trainable_parameters(),y=b.trainable_parameters();
  check(a.optimizer->state().size()==b.optimizer->state().size(),"optimizer populated slot counts exact");
  for(size_t j=0;j<x.size();++j) {
    auto ai=a.optimizer->state().find(x[j].unsafeGetTensorImpl()),bi=b.optimizer->state().find(y[j].unsafeGetTensorImpl());
    check((ai==a.optimizer->state().end())==(bi==b.optimizer->state().end()),"optimizer inactive slots exact");
    if(ai==a.optimizer->state().end())continue;
    const auto *as=dynamic_cast<const torch::optim::AdamWParamState*>(ai->second.get());
    const auto *bs=dynamic_cast<const torch::optim::AdamWParamState*>(bi->second.get());
    check(as&&bs&&as->step()==bs->step()&&exact(as->exp_avg(),bs->exp_avg())&&exact(as->exp_avg_sq(),bs->exp_avg_sq())&&
      exact(as->max_exp_avg_sq(),bs->max_exp_avg_sq()),"every AdamW step/moment byte exact");
  }
}
double gradient_sum(const torch::nn::Module &model) {
  double total=0.;for(const auto &p:model.parameters())if(p.grad().defined()) {
    check(p.is_cuda()&&p.grad().is_cuda()&&torch::isfinite(p.grad()).all().item<bool>(),"actual finite CUDA gradients");
    total+=p.grad().abs().sum().item<double>();
  }return total;
}
void gradient_isolation() {
  for(auto arm:{pt::Arm::frozen_state,pt::Arm::blocked_prediction,pt::Arm::full}) {
    auto c=config(arm);c.lambda_variance=0;c.lambda_covariance=0;
    torch::manual_seed(719);pt::Trainer t(c,identity());auto b=batch(c);
    const auto old=snapshot(*t.online),teacher=snapshot(*t.target),dynamic=snapshot(*t.dynamics);
    const auto receipt=t.step(b);check(receipt.updated,"prediction-only CUDA optimizer update");
    check(gradient_sum(*t.dynamics)>0,"predictive signal trains dynamics in every arm");
    check((gradient_sum(*t.online)>0)==(arm==pt::Arm::full),"complete prediction history detached in blocked/frozen arm");
    check(same(old,*t.online)==(arm!=pt::Arm::full),"prediction-only state changes only for full arm");
    check(!same(dynamic,*t.dynamics),"prediction-only dynamics parameters actually change");
    check(gradient_sum(*t.target)==0,"teacher has no gradient");
    check(!b.history_embeddings.grad().defined()&&!b.future_embeddings.grad().defined()&&!b.pool.embeddings.grad().defined(),
      "encoder-side input/future/pool gradients blocked at saved-embedding boundary");
    const auto current=t.online->parameters(),target=t.target->parameters();
    for(size_t j=0;j<current.size();++j)
      check(exact(target[j],teacher[j]+(current[j]-teacher[j])*(1-c.ema_decay)),"exact declared successful-update EMA arithmetic");
    if(arm!=pt::Arm::full)check(same(teacher,*t.target),"unchanged online gives bitwise unchanged teacher");
  }
  auto c=config(pt::Arm::blocked_prediction);torch::manual_seed(719);pt::Trainer blocked(c,identity());
  const auto initial=snapshot(*blocked.online);const auto receipt=blocked.step(batch(c));
  check(receipt.n_cov==7&&receipt.rank_upper_bound==3&&gradient_sum(*blocked.online)>0&&!same(initial,*blocked.online),
    "blocked arm retains active state-regularizer CUDA gradients and real updates");
}
void dimension_admission() {
  auto c=config(pt::Arm::full);auto state=c.state;
  state.device=torch::Device(torch::kCUDA);
  rejects([&]{sr::validate(state);},"ambiguous CUDA ordinal refused before tensor allocation");
  state=c.state;
  state.identity_initialization=false;state.embedding_width=std::numeric_limits<int64_t>::max();state.state_width=1;
  rejects([&]{sr::validate(state);},"state INT64_MAX elements/bytes refused before allocation");
  state=c.state;state.identity_initialization=false;state.embedding_width=INT64_C(1)<<61;state.state_width=2;
  rejects([&]{sr::validate(state);},"state float64 byte-count overflow refused before allocation");
  auto dynamics=c.dynamics;dynamics.history_length=std::numeric_limits<int64_t>::max();
  rejects([&]{ld::validate(dynamics);},"K*d signed overflow refused before multiplication/allocation");
  dynamics=c.dynamics;dynamics.state_width=std::numeric_limits<int64_t>::max();dynamics.history_length=1;
  rejects([&]{ld::validate(dynamics);},"Q*d signed overflow refused before multiplication/allocation");
  dynamics=c.dynamics;dynamics.history_length=INT64_C(1)<<60;dynamics.state_width=1;
  rejects([&]{ld::validate(dynamics);},"dynamics weight byte-count overflow refused before allocation");
}
void covariance_estimator() {
  auto o=torch::TensorOptions().device(torch::kCUDA,0).dtype(torch::kFloat64);
  auto input=values(20,o,.3).reshape({5,4}).set_requires_grad(true);
  auto scale=torch::tensor({1.,2.,.7,1.3},o);auto ids=torch::arange(5,torch::kInt64);
  auto result=pt::regularizer_loss(input,ids,scale,1.2,1e-4);
  auto z=input/scale;auto center=z-z.mean(0,true);auto expected=center.transpose(0,1).matmul(center)/4.;
  close(result.sample_covariance,expected,"centered sample covariance denominator n-1 across full graph");
  check(result.n_cov==5&&result.rank_upper_bound==4,"sample rather than asset count controls rank bound");
  (result.variance+result.covariance).backward();check(input.grad().defined()&&input.grad().is_cuda()&&
    input.grad().abs().sum().item<double>()>0,"pooled covariance preserves every activation gradient graph");
  auto duplicate=torch::tensor({0,1,1,3,4},torch::kInt64);
  rejects([&]{pt::regularizer_loss(input,duplicate,scale,1.,1e-4);},"duplicate origin cannot inflate covariance population");
  rejects([&]{pt::regularizer_loss(input.slice(0,0,1),ids.slice(0,0,1),scale,1.,1e-4);},"single-row covariance refused");
  // Pooling across microbatches includes between-group differences.
  auto shifted=torch::cat({torch::zeros({3,2},o),torch::ones({3,2},o)*4.},0);
  auto pooled=pt::regularizer_loss(shifted,torch::arange(6,torch::kInt64),torch::ones({2},o),1.,1e-4);
  check(pooled.sample_covariance.abs().sum().item<double>()>0,"pooled centering differs from separate constant microbatch covariances");
  auto low_rank=pt::regularizer_loss(values(24,o).reshape({3,8}),torch::arange(3,torch::kInt64),torch::ones({8},o),1.,1e-4);
  check(low_rank.rank_upper_bound==2&&torch::isfinite(low_rank.covariance).item<bool>(),"d>n-1 is admissible with explicit rank bound");
}
void folding() {
  auto c=config(pt::Arm::full);c.state.state_width=2;c.state.identity_initialization=false;c.dynamics.state_width=2;
  auto o=torch::TensorOptions().device(c.state.device).dtype(c.state.dtype);
  for(bool residual:{false,true}) {
    c.dynamics.residual=residual;
    sr::AffineState g(c.state,torch::tensor({.2,-.3,.5},o),torch::tensor({1.2,.7,2.},o));ld::AffineDynamics a(c.dynamics);
    {torch::NoGradGuard guard;a->affine->weight.copy_(values(a->affine->weight.numel(),o).reshape_as(a->affine->weight)*.2);
      a->affine->bias.copy_(values(a->affine->bias.numel(),o,.6)*.1);}
    const auto h=values(24,o,.4).reshape({4,2,3});
    auto folded=pt::fold_state_forecast(g,a);const auto forecast=a->forward(g->forward(h)).flatten(1);
    close(folded.forward(h),forecast,"affine forecast folds preprocessing, rectangular G, horizon A and residual without refit");
    auto history=pt::fold_state_history(g,2);close(history.forward(h),g->forward(h).flatten(1),"state-history folding exact to roundoff");
    auto joined=pt::concatenate(history,folded);auto w=values(2*joined.weight.size(0),o).reshape({2,joined.weight.size(0)});
    auto bias=torch::tensor({.3,-.7},o);auto readout=pt::fold_linear_readout(joined,w,bias);
    close(readout.forward(h),torch::cat({g->forward(h).flatten(1),forecast},1).matmul(w.transpose(0,1))+bias,
      "unchanged linear readout folds state plus forecast into native embedding history");
  }
}
void resume_and_admission() {
  TemporaryDirectory temporary;
  for(auto arm:{pt::Arm::frozen_state,pt::Arm::blocked_prediction,pt::Arm::full}) {
    auto c=config(arm);torch::manual_seed(382);auto i=identity();
    auto o=torch::TensorOptions().device(c.state.device).dtype(c.state.dtype);
    pt::Trainer t(c,i,torch::tensor({.1,.2,-.2},o),torch::tensor({.9,1.2,1.4},o),torch::tensor({1.,.8,1.3},o));
    auto b=batch(c);t.step(b);const auto before_skip=snapshot(*t.target);
    auto invalid=b;invalid.prediction_valid=torch::zeros_like(b.prediction_valid);
    check(!t.step(invalid).updated&&same(before_skip,*t.target)&&t.progress.completed_updates==1&&t.progress.attempted_batches==2,
      "ineligible attempt consumes no optimizer/teacher update");
    const auto file=temporary.path/("train-"+std::to_string(static_cast<int>(arm))+".pt");
    t.sample_indices(23,5);t.save_training(file.string());const auto immutable=bytes(file);
    auto expected_cpu=torch::rand({5},torch::TensorOptions().dtype(torch::kFloat64));
    auto expected_cuda=torch::rand({5},o);auto rows=t.sample_indices(23,9);
    const auto first=t.step(b);auto first_gradients=std::vector<torch::Tensor>{};
    for(const auto &p:t.trainable_parameters())first_gradients.push_back(p.grad().defined()?p.grad().detach().clone():torch::Tensor());
    auto restored=pt::Trainer::load_training(file.string(),c,i);
    check(exact(expected_cpu,torch::rand({5},torch::TensorOptions().dtype(torch::kFloat64)))&&
      exact(expected_cuda,torch::rand({5},o))&&rows==restored->sample_indices(23,9),
      "CPU/CUDA generators and next sampler draws restore exactly");
    const auto second=restored->step(b);
    check(first.prediction_loss==second.prediction_loss&&first.variance_loss==second.variance_loss&&
      first.covariance_loss==second.covariance_loss&&first.total_loss==second.total_loss,"exact next-update CUDA losses in every arm");
    const auto parameters=restored->trainable_parameters();
    for(size_t j=0;j<parameters.size();++j)check(exact(first_gradients[j],parameters[j].grad()),"every next-update gradient byte exact");
    same_models(t,*restored);same_optimizers(t,*restored);
    check(t.progress.attempted_batches==restored->progress.attempted_batches&&t.progress.completed_updates==2&&
      t.progress.ema_updates==restored->progress.ema_updates&&t.progress.online_updates==restored->progress.online_updates&&
      t.progress.sampler_draws==restored->progress.sampler_draws,"all attempt/component/EMA/sampler progress exact");
    rejects([&]{t.save_training(file.string());},"immutable publication refuses overwrite");
    check(bytes(file)==immutable,"refused overwrite leaves existing archive unchanged");
    const auto inference=temporary.path/("inference-"+std::to_string(static_cast<int>(arm))+".pt");
    restored->save_inference(inference.string());auto snapshot=pt::load_inference(inference.string(),c,i);
    check(same(::snapshot(*restored->online),*snapshot.online)&&same(::snapshot(*restored->target),*snapshot.target)&&
      same(::snapshot(*restored->dynamics),*snapshot.dynamics),"typed inference preserves the frozen coordinate tuple");
    for(const auto &p:snapshot.online->parameters())check(!p.requires_grad(),"served online frozen");
    rejects([&]{pt::Trainer::load_training(inference.string(),c,i);},"inference archive cannot resume training");
    rejects([&]{pt::load_inference(file.string(),c,i);},"training archive cannot masquerade as inference");
    auto wrong=c;wrong.ema_decay=.7;
    rejects([&]{pt::Trainer::load_training(file.string(),wrong,i);},"EMA/loss/optimizer recipe drift refused");
    wrong=c;wrong.dynamics.horizons={1,10};
    rejects([&]{pt::Trainer::load_training(file.string(),wrong,i);},"horizon association drift refused");
    auto drift=i;drift.pair_ledger+="-changed";
    rejects([&]{pt::Trainer::load_training(file.string(),c,drift);},"pair-ledger identity drift refused");
    drift=i;drift.encoder_checkpoint+="-changed";
    rejects([&]{pt::Trainer::load_training(file.string(),c,drift);},"encoder identity drift refused");
    auto bad_progress=t.progress;t.progress.completed_updates=1;t.progress.dynamics_updates=1;t.progress.ema_updates=1;
    t.progress.online_updates=std::min(t.progress.online_updates,INT64_C(1));
    rejects([&]{t.save_training((temporary.path/"bad-counter.pt").string());},"actual optimizer steps prevent counter reset");t.progress=bad_progress;
    const auto original=t.optimizer->param_groups().front().params();
    auto &params=t.optimizer->param_groups().front().params();std::reverse(params.begin(),params.end());
    rejects([&]{t.save_training((temporary.path/"bad-order.pt").string());},"optimizer remapping refused before archive publication");params=original;
  }
  for(const auto &entry:fs::directory_iterator(temporary.path))
    check(entry.path().filename().string().find(".tmp-")==std::string::npos,"no staging files left behind");
}
} // namespace
int main() try {
  check(torch::cuda::is_available(),"actual CUDA required");torch::set_num_threads(1);
  dimension_admission();gradient_isolation();covariance_estimator();folding();resume_and_admission();torch::cuda::synchronize();
  std::cout<<"PASS predictive v1: actual CUDA gradients/updates, frozen/blocked/full attribution, detached teacher and encoder boundary, "
    "distinct-origin pooled covariance and rank bound, fixed transforms, successful-update EMA/skip, affine/readout folding, "
    "all-arm exact next-update AdamW/RNG/sampler/EMA resume, typed identity admission and immutable archives; artificial engineering only\n";
  return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
