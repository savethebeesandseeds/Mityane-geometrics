// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Synthetic CUDA mechanism fixture; no historical checkpoint or dataset access.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v15.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v4.h"
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v15;
namespace old=embedding::encoders::causal_patch_bottleneck_mae::v4;
namespace impl=cp::implementation;
int64_t checks=0;
void check(bool condition,const char *message){++checks;TORCH_CHECK(condition,"[cpb-v15 objective] ",message);}
template<class F> void refuses(F action,const char *message){bool rejected=false;try{action();}catch(const c10::Error &){rejected=true;}check(rejected,message);}
bool zero_gradient(const torch::Tensor &value){return !value.grad().defined()||value.grad().eq(0).all().item<bool>();}
bool live(const torch::Tensor &value){return value.grad().defined()&&torch::isfinite(value.grad()).all().item<bool>()&&value.grad().abs().sum().item<double>()>0.;}
void clear(const std::vector<torch::Tensor> &parameters){for(auto value:parameters)value.mutable_grad()=torch::Tensor();}
bool bits(const torch::Tensor &a,const torch::Tensor &b){
  if(a.sizes()!=b.sizes()||a.scalar_type()!=b.scalar_type())return false;
  const auto x=a.detach().cpu().contiguous(),y=b.detach().cpu().contiguous();
  return std::memcmp(x.data_ptr(),y.data_ptr(),static_cast<size_t>(x.numel())*x.element_size())==0;
}
void near(double actual,double expected,double tolerance,const char *message){check(std::isfinite(actual)&&std::abs(actual-expected)<=tolerance*(1.+std::abs(expected)),message);}

void scalar_objective(){
  constexpr int B=3,C=3,H=2,F=2,N=B*C*H*F;
  const auto options=torch::TensorOptions().device(torch::kCUDA,0).dtype(torch::kFloat32);
  std::vector<float> m(N),v(N),t(N);std::vector<uint8_t> mask(N,0);
  for(int i=0;i<N;++i){m[i]=float((i%7)-3)*.6f;v[i]=.3f+.2f*float(i%5);t[i]=float((i%3)-1)*.25f;}
  for(int i:{0,1,8,9,10,11,16})mask[i]=1;
  for(int i=0;i<N;++i)if(!mask[i])m[i]=v[i]=t[i]=std::numeric_limits<float>::quiet_NaN();
  auto mean=torch::tensor(m,options).reshape({B,C,H,F}).clone().set_requires_grad(true);
  auto variance=torch::tensor(v,options).reshape({B,C,H,F}).clone().set_requires_grad(true);
  auto target=torch::tensor(t,options).reshape({B,C,H,F}).clone().set_requires_grad(true);
  const auto support=torch::tensor(mask,torch::kUInt8).reshape({B,C,H,F}).to(torch::kCUDA,torch::kBool);
  const auto eligible=support.sum(std::vector<int64_t>{2,3}).gt(0);
  auto objective=impl::location_scale_decoder_objective(mean,variance,target,support,eligible);
  check(objective.variance_nll.eligible_example_count==2&&objective.variance_nll.eligible_channel_count==3&&objective.variance_nll.target_cell_count==7,"hierarchical nonuniform support fixture");
  double huber=0.,nll=0.;std::vector<double> gm(N,0.),gv(N,0.);
  for(int b=0;b<B;++b){int channels=0;for(int c=0;c<C;++c){int count=0;for(int h=0;h<H;++h)for(int f=0;f<F;++f)count+=mask[((b*C+c)*H+h)*F+f];channels+=count>0;}
    for(int c=0;c<C;++c){int count=0;for(int h=0;h<H;++h)for(int f=0;f<F;++f)count+=mask[((b*C+c)*H+h)*F+f];if(!count)continue;
      const double weight=1./(2.*channels*count);
      for(int h=0;h<H;++h)for(int f=0;f<F;++f){const int i=((b*C+c)*H+h)*F+f;if(!mask[i])continue;
        const double error=double(m[i])-double(t[i]),a=std::abs(error),var=double(v[i]);
        huber+=weight*(a<=1.?.5*error*error:a-.5);nll+=weight*.5*(error*error/var+std::log(var));
        gm[i]=weight*std::clamp(error,-1.,1.);gv[i]=weight*.5*(1./var-error*error/(var*var));
      }
    }
  }
  near(objective.mean_huber.loss.item<double>(),huber,1e-11,"independent F64 hierarchical Huber sum");
  near(objective.variance_nll.loss.item<double>(),nll,1e-11,"independent F64 hierarchical variance NLL sum");
  near(objective.total_loss.item<double>(),huber+nll,1e-11,"exact unit coefficient composite");
  objective.variance_nll.loss.backward({},true);
  check(zero_gradient(mean)&&zero_gradient(target)&&live(variance),"variance calibration detaches error and target while retaining variance graph");
  clear({mean,variance,target});objective.mean_huber.loss.backward({},true);
  check(live(mean)&&zero_gradient(variance)&&zero_gradient(target),"Huber directly supervises mean only");
  clear({mean,variance,target});objective.total_loss.backward();
  const auto actual_m=mean.grad().cpu().contiguous(),actual_v=variance.grad().cpu().contiguous();
  for(int i=0;i<N;++i){near(actual_m.data_ptr<float>()[i],gm[i],2e-6,"manual mean derivative without inverse variance");near(actual_v.data_ptr<float>()[i],gv[i],2e-6,"manual detached-error variance derivative");}
  check(zero_gradient(target),"frozen target has no gradient");
  auto changed_mean=mean.detach().clone().set_requires_grad(true),changed_variance=variance.detach()*7.;
  impl::location_scale_decoder_objective(changed_mean,changed_variance,target,support,eligible).total_loss.backward();
  check(bits(mean.grad(),changed_mean.grad()),"changing variance cannot reweight direct Huber mean gradient");
  auto old_mean=mean.detach().clone().set_requires_grad(true);
  impl::location_scale_decoder_nll(old_mean,variance.detach(),target,support,eligible).loss.backward();
  check(!bits(old_mean.grad(),mean.grad()),"attached mean-NLL positive control is a different gradient");
  const auto empty=torch::zeros_like(support),empty_eligible=torch::zeros_like(eligible);
  auto empty_mean=mean.detach().clone().set_requires_grad(true),empty_variance=variance.detach().clone().set_requires_grad(true);
  auto empty_objective=impl::location_scale_decoder_objective(empty_mean,empty_variance,target,empty,empty_eligible);
  check(empty_objective.total_loss.item<double>()==0.,"all-unsupported NaN operands sanitize to exact zero");empty_objective.total_loss.backward();
  check(zero_gradient(empty_mean)&&zero_gradient(empty_variance),"empty population contributes exact zero gradients");
  auto poison=mean.detach().clone();poison.flatten().select(0,0).fill_(std::numeric_limits<float>::quiet_NaN());
  refuses([&]{(void)impl::location_scale_decoder_objective(poison,variance,target,support,eligible);},"supported nonfinite mean refuses");
  auto nonpositive=variance.detach().clone();nonpositive.flatten().select(0,0).fill_(0.);
  refuses([&]{(void)impl::location_scale_decoder_objective(mean,nonpositive,target,support,eligible);},"supported nonpositive variance refuses");
  refuses([&]{(void)impl::location_scale_decoder_objective(mean,variance,target,support,torch::ones_like(eligible));},"eligibility must match exact support");
  refuses([&]{(void)impl::location_scale_decoder_objective(mean.to(torch::kFloat64),variance,target,support,eligible);},"distribution dtype change refuses");
}

struct Synthetic {torch::Tensor raw,observed,ids,times,cuts;};
Synthetic synthetic(){
  auto raw=torch::zeros({8,3,128,9},torch::kFloat64);auto x=raw.accessor<double,4>();
  for(int b=0;b<8;++b)for(int c=0;c<3;++c)for(int h=0;h<128;++h)for(int f=0;f<9;++f)
    x[b][c][h][f]=10.+2.*c+.2*f+.01*h+.07*b+std::sin(.13*h+.19*f+.31*c+.17*b);
  return{raw,torch::ones_like(raw,torch::kBool),torch::tensor({0,1,2},torch::kInt64),torch::arange(128,torch::kFloat64)*60000.,torch::full({8},127.*60000.,torch::kFloat64)};
}
void paired_models(const std::string &settings,const std::string &control_settings,uint64_t seed){
  const auto c=cp::read_settings(settings);const auto oc=old::read_settings(control_settings);const auto data=synthetic();
  const cp::RawHistory history{data.raw,data.observed,data.ids,data.times,60000.};
  const old::RawHistory old_history{data.raw,data.observed,data.ids,data.times,60000.};
  const auto retained=cp::boundary::retain(history,data.cuts,c);const auto old_retained=old::boundary::retain(old_history,data.cuts,oc);
  const impl::Input rounded{retained.raw.data.to(torch::kFloat32),retained.raw.observed,retained.raw.channel_ids,retained.raw.endpoints,60000.};
  const old::implementation::Input old_rounded{old_retained.raw.data.to(torch::kFloat32),old_retained.raw.observed,old_retained.raw.channel_ids,old_retained.raw.endpoints,60000.};
  auto scaler=impl::fit_configurable_spectral_scaler(rounded,c.level);
  auto old_scaler=old::implementation::fit_configurable_spectral_scaler(old_rounded,oc.level);
  cp::BankStandardizer bank(c.level,"v15-mechanism-generated-TRAIN-only",scaler.identity());
  old::BankStandardizer old_bank(oc.level,"v15-mechanism-generated-TRAIN-only",old_scaler.identity());
  bank.add(impl::relative_change_targets(retained.raw,c.level));bank.freeze();
  old_bank.add(old::implementation::relative_change_targets(old_retained.raw,oc.level));old_bank.freeze();
  auto initialized=cp::fresh(c,scaler,bank,seed);auto old_initialized=old::fresh(oc,old_scaler,old_bank,seed);
  auto model=initialized.model;auto control=old_initialized.model;
  const auto parameters=model->named_parameters(),old_parameters=control->named_parameters();
  check(parameters.size()==old_parameters.size(),"same registration count as V4");
  int64_t learned=0,total=0,tensors=0;
  for(const auto &p:parameters){const auto *q=old_parameters.find(p.key());check(q&&q->requires_grad()==p.value().requires_grad()&&bits(p.value(),*q),"every fresh parameter bit-paired with same-seed V4");
    total+=p.value().numel();if(p.value().requires_grad()){learned+=p.value().numel();++tensors;}}
  check(total==297560&&learned==293672&&tensors==75,"exact unchanged native36 capacity; no uncertainty4 added");
  const auto buffers=model->named_buffers(),old_buffers=control->named_buffers();check(buffers.size()==old_buffers.size(),"same registered buffer count");
  for(const auto &p:buffers){const auto *q=old_buffers.find(p.key());check(q&&bits(p.value(),*q),"every fresh buffer bit-paired");}
  model->eval();control->eval();{
    torch::NoGradGuard guard;const auto a=model->encode_at(history,data.cuts,scaler);const auto b=control->encode_at(old_history,data.cuts,old_scaler);
    check(bits(a.native,b.native)&&bits(a.valid,b.valid)&&a.native.size(1)==36,"full unchanged native forward is bit-paired at initialization");
  }
  auto core=model->core();auto old_core=control->core();
  {torch::NoGradGuard guard;
    core->variance_projection.copy_(torch::linspace(-.08,.13,108,core->variance_projection.options()).reshape({27,4}));
    core->variance_bias.copy_(torch::linspace(-.2,.2,27,core->variance_bias.options()));
    core->encoder->change_decoder.copy_(torch::linspace(-.01,.02,core->encoder->change_decoder.numel(),core->encoder->change_decoder.options()).reshape_as(core->encoder->change_decoder));
    old_core->variance_projection.copy_(core->variance_projection);old_core->variance_bias.copy_(core->variance_bias);old_core->encoder->change_decoder.copy_(core->encoder->change_decoder);
  }
  auto native=torch::linspace(-.4,.7,72,torch::TensorOptions().device(c.level.device).dtype(torch::kFloat32)).reshape({2,36}).clone().set_requires_grad(true);
  const auto distribution=core->decode_with_trace(native,data.ids);const auto old_distribution=old_core->decode_with_trace(native.detach(),data.ids);
  check(bits(distribution.mean,old_distribution.mean)&&bits(distribution.variance,old_distribution.variance),"V4/V15 forward Gaussian capacity unchanged with nonzero conditional heads");
  const auto support=torch::ones_like(distribution.mean,torch::kBool),eligible=torch::ones({2,3},support.options());
  const auto target=torch::zeros_like(distribution.mean);
  auto objective=impl::location_scale_decoder_objective(distribution.mean,distribution.variance,target,support,eligible);
  clear(core->parameters());objective.variance_nll.loss.backward({},true);
  check(zero_gradient(native)&&live(core->variance_projection)&&live(core->variance_bias),"calibration trains both variance-head tensors while stopping native input and mean error");
  for(const auto &p:core->encoder->parameters())check(zero_gradient(p),"calibration cannot reach any parent/mean encoder parameter");
  clear(core->parameters());native.mutable_grad()=torch::Tensor();objective.mean_huber.loss.backward({},true);
  check(live(native)&&zero_gradient(core->variance_projection)&&zero_gradient(core->variance_bias)&&live(core->encoder->change_decoder),"Huber keeps native/mean decoder live without variance-head gradient");
  clear(core->parameters());native.mutable_grad()=torch::Tensor();(objective.total_loss*1000.).backward();
  check(live(core->variance_projection)&&live(core->encoder->change_decoder),"synthetic clipping stress contains both objective branches");
  std::vector<torch::Tensor> gradients;std::vector<torch::Tensor> learned_parameters;
  for(auto p:core->parameters())if(p.requires_grad()){learned_parameters.push_back(p);gradients.push_back(p.grad().defined()?p.grad().clone():torch::Tensor());}
  const double norm=torch::nn::utils::clip_grad_norm_(learned_parameters,1.);const double factor=std::min(1.,1./(norm+1e-6));
  check(norm>1.&&factor<1.,"one actual shared norm clips the synthetic combined objective stress");
  for(size_t i=0;i<gradients.size();++i)if(gradients[i].defined())check(torch::allclose(learned_parameters[i].grad(),gradients[i]*factor,3e-6,1e-8),"both routes use the same single recorded clip factor");
}
} // namespace
int main(int argc,char **argv){try{
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  std::string settings,control_settings;uint64_t seed=945101;std::set<std::string> seen;
  for(int i=1;i<argc;i+=2){TORCH_CHECK(i+1<argc,"value required");const std::string key=argv[i];
    TORCH_CHECK(seen.insert(key).second,"duplicate argument");
    if(key=="--settings")settings=argv[i+1];else if(key=="--control-settings")control_settings=argv[i+1];else if(key=="--seed"){
      const std::string text=argv[i+1];size_t consumed=0;seed=std::stoull(text,&consumed);
      TORCH_CHECK(consumed==text.size()&&seed<=uint64_t(std::numeric_limits<int64_t>::max())&&!text.empty()&&text.front()!='-',"explicit nonnegative exact signed64 seed required");
    }else TORCH_CHECK(false,"unknown argument");}
  check(!settings.empty()&&!control_settings.empty()&&torch::cuda::is_available(),"explicit paired configs and actual CUDA required");
  scalar_objective();paired_models(settings,control_settings,seed);
  std::cout<<"{\"kind\":\"cpb_v15_native36_objective_mechanism_fixture_v1\",\"status\":\"passed\",\"checks\":"<<checks
    <<",\"seed\":"<<seed<<",\"model_calls\":2,\"decoder_calls\":2,\"historical_checkpoints_opened\":0,\"head_fits\":0,\"optimizer_steps\":0,\"fixture_clip_loss_multiplier\":1000,\"shared_clip_calls\":1}"<<std::endl;
  return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
