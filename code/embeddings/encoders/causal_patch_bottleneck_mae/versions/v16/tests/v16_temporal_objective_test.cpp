// Copyright (c) 2026 Waajacu. See LICENSE.
// Artificial CUDA mechanism fixture only. No dataset/checkpoint/head access.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v16.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v4.h"
#include <torch/cuda.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v16;
namespace old=embedding::encoders::causal_patch_bottleneck_mae::v4;
namespace impl=cp::implementation;
int64_t checks=0,finite_differences=0;
void check(bool value,const char *message){++checks;TORCH_CHECK(value,"[cpb-v16 temporal fixture] ",message);}
template<class F>void refuses(F action,const char *message){bool rejected=false;try{action();}catch(const c10::Error &){rejected=true;}check(rejected,message);}
void near(double actual,double expected,double tolerance,const char *message){check(std::isfinite(actual)&&std::abs(actual-expected)<=tolerance*(1.+std::abs(expected)),message);}
bool bits(const torch::Tensor &a,const torch::Tensor &b){if(a.sizes()!=b.sizes()||a.scalar_type()!=b.scalar_type())return false;const auto x=a.detach().cpu().contiguous(),y=b.detach().cpu().contiguous();return std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size())==0;}
bool zero_gradient(const torch::Tensor &x){return !x.grad().defined()||(torch::isfinite(x.grad()).all().item<bool>()&&x.grad().eq(0).all().item<bool>());}
bool live(const torch::Tensor &x){return x.grad().defined()&&torch::isfinite(x.grad()).all().item<bool>()&&x.grad().abs().sum().item<double>()>0.;}
void clear(const std::vector<torch::Tensor> &parameters){for(auto x:parameters)x.mutable_grad()=torch::Tensor();}
constexpr int B=3,C=3,H=32,F=2,N=B*C*H*F;
size_t index(int b,int c,int t,int f){return ((b*C+c)*H+t)*F+f;}
struct Oracle {double loss{0};std::array<double,3> losses{};std::array<int64_t,3> counts{},channels{},examples{};std::vector<double> gradient=std::vector<double>(N,0.);};
// Independent scalar reference: no producer tensor helper/reduction/arithmetic.
Oracle oracle(const std::vector<double> &mean,const std::vector<double> &target,const std::vector<uint8_t> &q){
  Oracle out;const std::array<int,3> lags{1,4,16};
  for(size_t a=0;a<lags.size();++a){const int lag=lags[a];int count[B][C]{};int channels[B]{};int examples=0;
    for(int b=0;b<B;++b){for(int c=0;c<C;++c){for(int t=lag;t<H;++t)for(int f=0;f<F;++f)count[b][c]+=q[index(b,c,t,f)]&&q[index(b,c,t-lag,f)];channels[b]+=count[b][c]>0;}examples+=channels[b]>0;}
    out.examples[a]=examples;
    for(int b=0;b<B;++b)for(int c=0;c<C;++c){out.counts[a]+=count[b][c];out.channels[a]+=count[b][c]>0;if(!count[b][c])continue;
      const double weight=1./(double(examples)*channels[b]*count[b][c]);
      for(int t=lag;t<H;++t)for(int f=0;f<F;++f){const auto hi=index(b,c,t,f),lo=index(b,c,t-lag,f);if(!q[hi]||!q[lo])continue;
        const double e=((mean[hi]-mean[lo])-(target[hi]-target[lo]))/std::sqrt(2.),absolute=std::abs(e);
        out.losses[a]+=weight*(absolute<=1.?.5*e*e:absolute-.5);
        const double derivative=weight*std::clamp(e,-1.,1.)/(3.*std::sqrt(2.));out.gradient[hi]+=derivative;out.gradient[lo]-=derivative;
      }
    }out.loss+=out.losses[a]/3.;
  }return out;
}
torch::Tensor tensor(const std::vector<double> &x){return torch::tensor(x,torch::kFloat64).to(torch::kCUDA,torch::kFloat32).reshape({B,C,H,F}).clone();}
torch::Tensor query(const std::vector<uint8_t> &q){return torch::tensor(q,torch::kUInt8).reshape({B,C,H,F}).to(torch::kCUDA,torch::kBool);}
void scalar_mechanism(){
  std::vector<double> m(N),y(N),variance(N);std::vector<uint8_t> q(N,0);
  for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int t=0;t<H;++t)for(int f=0;f<F;++f){const auto i=index(b,c,t,f);m[i]=.125*((t*3+c*5+b*7+f)%23-11);y[i]=.0625*((t+c+f+b)%9-4);variance[i]=.25+.25*(i%5);
    q[i]=(b==0&&c<2&&(t<8||(t>=16&&t<24)))||(b==1&&c==2&&f==0&&(t==0||t==1||t==4||t==16));}
  const auto expected=oracle(m,y,q);auto mean=tensor(m).set_requires_grad(true),target=tensor(y).set_requires_grad(true),var=tensor(variance).set_requires_grad(true);const auto mask=query(q);
  auto actual=impl::location_scale_decoder_temporal_loss(mean,target,mask);
  check(actual.lags.size()==3&&actual.active_lag_count==3,"all three fixed lags active on unequal support");
  near(actual.loss.item<double>(),expected.loss,2e-12,"independent hierarchical temporal scalar parity");
  for(size_t a=0;a<3;++a){const auto &term=actual.lags[a];near(term.loss.item<double>(),expected.losses[a],2e-12,"independent per-lag scalar parity");
    check(term.pair_count==expected.counts[a]&&term.eligible_channel_count==expected.channels[a]&&term.eligible_example_count==expected.examples[a],"exact pair/channel/example support counts");
    check(term.cell_error.scalar_type()==torch::kFloat64&&term.cell_huber.scalar_type()==torch::kFloat64&&term.pair_mask.size(2)==H-term.lag,"F64 cells and nonwrapping H-minus-lag geometry");}
  actual.loss.backward();const auto gm=mean.grad().cpu().to(torch::kFloat64).contiguous().reshape({N});const auto gradient=gm.accessor<double,1>();
  check(zero_gradient(target)&&zero_gradient(var),"no direct target or variance gradient from temporal objective");
  for(int i=0;i<N;++i)near(gradient[i],expected.gradient[i],3e-6,"manual endpoint adjoint including reduction and fixed third weights");
  for(const auto i:{index(0,0,0,0),index(0,0,4,0),index(0,1,17,1),index(1,2,1,0),index(1,2,16,0),index(2,1,8,0)}){
    auto plus=m,minus=m;constexpr double step=1e-5;plus[i]+=step;minus[i]-=step;
    const double numerical=(oracle(plus,y,q).loss-oracle(minus,y,q).loss)/(2.*step);++finite_differences;
    near(numerical,gradient[i],5e-6,"independent F64 scalar finite difference agrees with actual CUDA adjoint");}
  const auto order=torch::tensor({2,0,1},torch::kInt64).cuda();
  const auto reordered=impl::location_scale_decoder_temporal_loss(mean.detach().index_select(1,order),target.detach().index_select(1,order),mask.index_select(1,order));
  near(reordered.loss.item<double>(),expected.loss,2e-12,"channel storage permutation preserves temporal reduction");
  const auto reordered_examples=impl::location_scale_decoder_temporal_loss(mean.detach().index_select(0,order),target.detach().index_select(0,order),mask.index_select(0,order));
  near(reordered_examples.loss.item<double>(),expected.loss,2e-12,"example permutation preserves temporal reduction");
  const auto original=impl::location_scale_decoder_temporal_loss(mean.detach(),target.detach(),mask);
  const auto offset=impl::location_scale_decoder_temporal_loss(mean.detach()+8.,target.detach()-4.,mask);
  near(offset.loss.item<double>(),original.loss.item<double>(),2e-12,"temporal term is blind to constant level offsets; main NLL must anchor levels");
  const auto gaussian=impl::location_scale_decoder_nll(mean.detach(),var,target.detach(),mask,mask.sum(std::vector<int64_t>{2,3}).gt(0));
  const auto changed_gaussian=impl::location_scale_decoder_nll(mean.detach()+8.,var,target.detach(),mask,mask.sum(std::vector<int64_t>{2,3}).gt(0));
  check(std::abs(gaussian.loss.item<double>()-changed_gaussian.loss.item<double>())>1.,"retained base Gaussian penalizes the level shift");
  mean=tensor(m).set_requires_grad(true);target=tensor(y).set_requires_grad(true);var=tensor(variance).set_requires_grad(true);
  const auto base=impl::location_scale_decoder_nll(mean,var,target,mask,mask.sum(std::vector<int64_t>{2,3}).gt(0));
  auto total=impl::location_scale_decoder_objective(base,mean,target,mask);near(total.total_loss.item<double>(),base.loss.item<double>()+.1*expected.loss,2e-12,"Gaussian coefficient one and auxiliary coefficient point one");
  total.total_loss.backward();check(live(var)&&zero_gradient(target),"one Gaussian still supervises variance and target remains detached");
  auto base_mean=tensor(m).set_requires_grad(true),base_var=tensor(variance).set_requires_grad(true);
  impl::location_scale_decoder_nll(base_mean,base_var,target.detach(),mask,mask.sum(std::vector<int64_t>{2,3}).gt(0)).loss.backward();
  check(torch::equal(var.grad(),base_var.grad()),"temporal term adds exactly zero variance-output derivative");
  check(torch::allclose(mean.grad(),base_mean.grad()+tensor(expected.gradient)*.1,4e-6,1e-8),"actual combined mean gradient is base plus point-one independent temporal adjoint");
}
void boundary_support_cases(){
  const auto options=torch::TensorOptions().device(torch::kCUDA,0).dtype(torch::kFloat32);
  auto m=torch::arange(128,options).reshape({1,1,128,1}).clone().set_requires_grad(true),target=torch::zeros_like(m);
  auto q=torch::zeros_like(m,torch::kBool);q.narrow(2,40,8).fill_(true);
  const auto fixed=impl::location_scale_decoder_temporal_loss(m,target,q);
  check(fixed.lags[0].pair_count==7&&fixed.lags[1].pair_count==4&&fixed.lags[2].pair_count==0&&fixed.active_lag_count==2,"one hidden eight-step fixed-query patch has unsupported lag16");
  near(fixed.loss.item<double>(),(fixed.lags[0].loss.item<double>()+fixed.lags[1].loss.item<double>())/3.,1e-12,"empty lag keeps fixed one-third weight");
  q.zero_();q[0][0][0][0]=true;q[0][0][127][0]=true;
  const auto edges=impl::location_scale_decoder_temporal_loss(m,target,q);check(edges.active_lag_count==0,"no circular last-to-first pair or compressed-observation indexing");
  auto empty_mean=torch::full_like(m,std::numeric_limits<float>::quiet_NaN()).detach().set_requires_grad(true);
  const auto empty=impl::location_scale_decoder_temporal_loss(empty_mean,empty_mean.detach(),torch::zeros_like(q));
  check(empty.loss.requires_grad()&&empty.loss.item<double>()==0.&&empty.active_lag_count==0,"empty unsupported auxiliary is finite connected zero");empty.loss.backward();check(zero_gradient(empty_mean),"empty auxiliary has zero finite gradient even with unsupported NaNs");
  q.zero_();q.narrow(2,32,8).fill_(true);const auto safe=torch::where(q,m.detach(),torch::full_like(m,std::numeric_limits<float>::infinity()));
  check(torch::isfinite(impl::location_scale_decoder_temporal_loss(safe,torch::where(q,target,torch::full_like(target,std::numeric_limits<float>::quiet_NaN())),q).loss).item<bool>(),"unsupported NaN/Inf endpoints sanitized before arithmetic");
  refuses([&]{(void)impl::location_scale_decoder_temporal_loss(torch::full_like(m,std::numeric_limits<float>::infinity()),target,q);},"supported nonfinite mean rejected");
  refuses([&]{(void)impl::location_scale_decoder_temporal_loss(m,torch::full_like(target,std::numeric_limits<float>::quiet_NaN()),q);},"supported nonfinite target rejected");
  refuses([&]{(void)impl::location_scale_decoder_temporal_loss(m.to(torch::kFloat64),target,q);},"wrong distribution dtype rejected");
  refuses([&]{(void)impl::location_scale_decoder_temporal_loss(m,target,q.to(torch::kUInt8));},"nonbool original Q rejected");
  const auto exact=impl::location_scale_decoder_temporal_loss(m.detach(),m.detach(),q);check(exact.loss.item<double>()==0.,"correct slope has zero difference loss");
  const auto reversed=impl::location_scale_decoder_temporal_loss(127.-m.detach(),m.detach(),q);check(reversed.loss.item<double>()>0.,"wrong slope with equal window level is penalized");
  // A F32 subtraction would round (2^24 - 1); independently promoted endpoints do not.
  auto large=torch::zeros_like(m);large[0][0][32][0]=1.;large[0][0][33][0]=16777216.;
  auto two=torch::zeros_like(q);two[0][0][32][0]=true;two[0][0][33][0]=true;
  const auto precision=impl::location_scale_decoder_temporal_loss(large,target,two);
  near(precision.lags[0].cell_error[0][0][32][0].item<double>(),16777215./std::sqrt(2.),1e-15,"endpoint promotion precedes difference arithmetic");
}
struct Synthetic {torch::Tensor raw,observed,ids,times,cuts;};
Synthetic synthetic(){
  auto raw=torch::zeros({2,3,193,9},torch::kFloat64);auto a=raw.accessor<double,4>();
  for(int b=0;b<2;++b)for(int c=0;c<3;++c)for(int t=0;t<193;++t)for(int f=0;f<9;++f)a[b][c][t][f]=10.+c+.1*f+.01*t+.05*b+std::sin(.13*t+.17*c+.19*f+.23*b);
  return {raw,torch::ones_like(raw,torch::kBool),torch::tensor({0,1,2},torch::kInt64),torch::arange(193,torch::kFloat64)*60000.,torch::full({2},127.*60000.,torch::kFloat64)};
}
void paired_models(const std::string &settings,const std::string &control_settings,int64_t seed){
  const auto c=cp::read_settings(settings);const auto oc=old::read_settings(control_settings);const auto data=synthetic();
  const cp::RawHistory h{data.raw,data.observed,data.ids,data.times,60000.};const old::RawHistory oh{data.raw,data.observed,data.ids,data.times,60000.};
  const auto w=cp::boundary::retain(h,data.cuts,c);const auto ow=old::boundary::retain(oh,data.cuts,oc);
  const impl::Input rounded{w.raw.data.to(torch::kFloat32),w.raw.observed,w.raw.channel_ids,w.raw.endpoints,60000.};
  const old::implementation::Input old_rounded{ow.raw.data.to(torch::kFloat32),ow.raw.observed,ow.raw.channel_ids,ow.raw.endpoints,60000.};
  auto scaler=impl::fit_configurable_spectral_scaler(rounded,c.level);auto old_scaler=old::implementation::fit_configurable_spectral_scaler(old_rounded,oc.level);
  cp::BankStandardizer bank(c.level,"v16-fixture-generated-TRAIN-only",scaler.identity());old::BankStandardizer old_bank(oc.level,"v16-fixture-generated-TRAIN-only",old_scaler.identity());
  bank.add(impl::relative_change_targets(w.raw,c.level));bank.freeze();old_bank.add(old::implementation::relative_change_targets(ow.raw,oc.level));old_bank.freeze();
  auto fresh=cp::fresh(c,scaler,bank,seed);auto old_fresh=old::fresh(oc,old_scaler,old_bank,seed);auto model=fresh.model;auto control=old_fresh.model;
  const auto p=model->named_parameters(),op=control->named_parameters();check(p.size()==op.size(),"same V4 parameter registration count");int64_t total=0,learned=0,tensors=0;
  for(const auto &x:p){const auto *y=op.find(x.key());check(y&&y->requires_grad()==x.value().requires_grad()&&bits(x.value(),*y),"every fresh V16 parameter bit-equals same-seed V4");total+=x.value().numel();if(x.value().requires_grad()){learned+=x.value().numel();++tensors;}}
  check(total==297560&&learned==293672&&tensors==75,"no auxiliary parameter or width added");
  const auto buffers=model->named_buffers(),old_buffers=control->named_buffers();check(buffers.size()==old_buffers.size(),"same buffer registration count");for(const auto &x:buffers){const auto *y=old_buffers.find(x.key());check(y&&bits(x.value(),*y),"every initial buffer bit-paired with V4");}
  model->train();control->train();const auto hidden=torch::zeros_like(w.raw.observed);hidden.narrow(2,16,8).fill_(true);hidden.narrow(2,32,8).fill_(true);hidden.narrow(2,64,8).fill_(true);
  const auto candidate=model->forward_at(h,data.cuts,hidden,scaler);const auto baseline=control->forward_at(oh,data.cuts,hidden,old_scaler);
  check(bits(candidate.output.encoding.z_contextual_global,baseline.output.encoding.z_contextual_global)&&bits(candidate.distribution.mean,baseline.distribution.mean)&&bits(candidate.distribution.variance,baseline.distribution.variance)&&bits(candidate.nll.loss,baseline.nll.loss),"V4 native36 forward/mean/variance/base NLL exact parity");
  check(candidate.objective.temporal.active_lag_count==3&&candidate.objective.temporal.lags[2].pair_count>0,"whole-model mechanism includes actual cross-hidden-patch lag16 support");
  const auto deletion=torch::zeros_like(hidden);deletion[0][0][8][0]=true;
  const auto with_context=model->forward_at(h,data.cuts,hidden,scaler,deletion);
  check(bits(candidate.target_mask,with_context.target_mask)&&!with_context.visible[0][0][8][0].item<bool>(),"context deletion removes visible input without adding original-Q targets");
  for(size_t a=0;a<3;++a)check(bits(candidate.objective.temporal.lags[a].pair_mask,with_context.objective.temporal.lags[a].pair_mask),"all lag supports remain original-Q after context deletion");
  auto bad_deletion=deletion.clone();bad_deletion[0][0][16][0]=true;refuses([&]{(void)model->forward_at(h,data.cuts,hidden,scaler,bad_deletion);},"context deletion cannot claim a hidden target endpoint");
  clear(model->parameters());candidate.objective.temporal.loss.backward();check(zero_gradient(model->core()->variance_projection)&&zero_gradient(model->core()->variance_bias),"auxiliary has no direct variance-head path");
  check(live(model->core()->encoder->change_decoder),"auxiliary reaches the live residual decoder");bool parent_live=false;for(const auto &x:model->core()->encoder->level->parameters())parent_live|=live(x);check(parent_live,"auxiliary reaches live parent/backbone mean path");
  clear(model->parameters());const auto stress=model->forward_at(h,data.cuts,hidden,scaler);(1000.*stress.output.loss).backward();
  std::vector<torch::Tensor> gradients,parameters;for(auto x:model->parameters())if(x.requires_grad()){parameters.push_back(x);gradients.push_back(x.grad().defined()?x.grad().clone():torch::Tensor());}
  const double norm=torch::nn::utils::clip_grad_norm_(parameters,1.),factor=std::min(1.,1./(norm+1e-6));check(norm>1.,"actual synthetic combined objective exceeds one global clip bound");
  for(size_t i=0;i<parameters.size();++i)if(gradients[i].defined())check(torch::allclose(parameters[i].grad(),gradients[i]*factor,3e-6,1e-8),"all routes retain one shared global clip factor");
  // Deliberate engineering-only nonzero heads avoid vacuous unit-variance parity.
  clear(model->parameters());auto core=model->core();auto old_core=control->core();
  {torch::NoGradGuard guard;core->variance_projection.copy_(torch::linspace(-.08,.13,108,core->variance_projection.options()).reshape({27,4}));
    core->variance_bias.copy_(torch::linspace(-.2,.2,27,core->variance_bias.options()));
    core->encoder->change_decoder.copy_(torch::linspace(-.01,.02,core->encoder->change_decoder.numel(),core->encoder->change_decoder.options()).reshape_as(core->encoder->change_decoder));
    old_core->variance_projection.copy_(core->variance_projection);old_core->variance_bias.copy_(core->variance_bias);old_core->encoder->change_decoder.copy_(core->encoder->change_decoder);}
  auto native=torch::linspace(-.4,.7,72,core->variance_projection.options()).reshape({2,36}).clone().set_requires_grad(true);
  const auto distribution=core->decode_with_trace(native,data.ids);const auto old_distribution=old_core->decode_with_trace(native.detach(),data.ids);
  check(bits(distribution.mean,old_distribution.mean)&&bits(distribution.variance,old_distribution.variance),"nonzero conditional/residual heads retain bit-exact V4 distribution arithmetic");
  distribution.variance.sum().backward();check(zero_gradient(native)&&live(core->variance_projection)&&live(core->variance_bias),"original variance-input stop remains nonvacuous and head learning stays live");
}
} // namespace
int main(int argc,char **argv)try{
  torch::set_num_threads(1);torch::set_num_interop_threads(1);std::string settings,control_settings;int64_t seed=-1;std::set<std::string> seen;
  for(int i=1;i<argc;i+=2){TORCH_CHECK(i+1<argc,"argument value required");const std::string key=argv[i];TORCH_CHECK(seen.insert(key).second,"duplicate argument");if(key=="--settings")settings=argv[i+1];else if(key=="--control-settings")control_settings=argv[i+1];else if(key=="--seed"){const std::string text=argv[i+1];size_t consumed=0;seed=std::stoll(text,&consumed);TORCH_CHECK(consumed==text.size()&&seed>=0,"exact nonnegative signed64 seed required");}else TORCH_CHECK(false,"unknown argument");}
  check(!settings.empty()&&!control_settings.empty()&&seed>=0&&torch::cuda::is_available(),"explicit configurations/seed and actual CUDA required");
  scalar_mechanism();boundary_support_cases();paired_models(settings,control_settings,seed);
  std::cout<<"{\"kind\":\"cpb_v16_temporal_objective_mechanism_fixture_v1\",\"status\":\"passed\",\"checks\":"<<checks<<",\"finite_difference_coordinates\":"<<finite_differences<<",\"seed\":"<<seed<<",\"new_parameters\":0,\"native_width\":36,\"historical_checkpoints_opened\":0,\"head_fits\":0,\"optimizer_steps\":0,\"shared_clip_calls\":1}"<<std::endl;return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
