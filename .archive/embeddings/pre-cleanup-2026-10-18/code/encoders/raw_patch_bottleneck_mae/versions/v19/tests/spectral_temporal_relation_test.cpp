// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/spectral_temporal_relation.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "rpb_test_support.h"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#ifndef MULTIBAND_SCREEN_SOURCE_ID
#define MULTIBAND_SCREEN_SOURCE_ID "unrecorded"
#endif
namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;
constexpr double tau=6.283185307179586476925286766559;
rpb::Config configuration() {
  rpb::Config c;c.device=torch::kCUDA;c.channel_ids={101,202,303};
  c.channel_mixer_layers=1;c.channel_mixer_placement=1;c.global_bottleneck_mode=2;return c;
}
void exact(const torch::Tensor &a,const torch::Tensor &b,const std::string &why) {
  check(a.scalar_type()==b.scalar_type()&&a.sizes()==b.sizes()&&torch::equal(a,b),why);
}
rpb::Input wave_input(const rpb::Config &c,double slow_period=15.5,double fast_period=5.6,double ratio=1.1) {
  auto out=input(c,2);out.data=torch::zeros({2,3,32,3},torch::kFloat32);
  auto x=out.data.accessor<float,4>();
  for(int64_t b=0;b<2;++b)for(int64_t channel=0;channel<3;++channel)
    for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f) {
      const double gain=.6+.3*channel+.07*f,offset=-.4+.23*channel+.11*f;
      const double slow=ratio*std::cos(tau*t/slow_period+.4*channel+.21*b);
      const double fast=std::cos(tau*t/fast_period-.3*channel+.17*b);
      x[b][channel][t][f]=float(offset+gain*(slow+fast));
    }
  out.data=out.data.to(torch::kCUDA);out.observed=out.observed.to(torch::kCUDA);return out;
}
struct ScalarBank {torch::Tensor values,counts,eligible;};
// Independent scalar2x2 fixed-bin arithmetic on artificial input copies only.
// No model, frequency search, true period, label or data-generating API runs
// here. It independently checks every saved support and all108 metrics.
ScalarBank scalar_bank(const rpb::Input &input) {
  const auto host=input.data.detach().to(torch::kCPU,torch::kFloat64).contiguous();
  const auto mask=input.observed.to(torch::kCPU).contiguous();
  const auto x=host.accessor<double,4>();const auto m=mask.accessor<bool,4>();const auto B=host.size(0);
  ScalarBank out{torch::zeros({B,108},torch::kFloat32),torch::zeros({B,27},torch::kInt64),torch::zeros({B,27},torch::kBool)};
  auto v=out.values.accessor<float,2>();auto count=out.counts.accessor<int64_t,2>();auto ok=out.eligible.accessor<bool,2>();
  constexpr std::array<std::array<int64_t,2>,3> pairs{{{0,1},{0,2},{1,2}}};
  for(int64_t b=0;b<B;++b)for(int64_t p=0;p<3;++p)for(int64_t fi=0;fi<3;++fi)for(int64_t fj=0;fj<3;++fj) {
    std::vector<int64_t> times;double am=0,bm=0;
    for(int64_t t=0;t<32;++t)if(m[b][pairs[p][0]][t][fi]&&m[b][pairs[p][1]][t][fj]) {
      times.push_back(t);am+=x[b][pairs[p][0]][t][fi];bm+=x[b][pairs[p][1]][t][fj];
    }
    const auto index=p*9+fi*3+fj;const auto n=times.size();count[b][index]=int64_t(n);
    if(n<4)continue;
    am/=n;bm/=n;
    std::array<double,8> ac{},as{},bc{},bs{};bool full_rank=true;double ea=0,eb=0;
    for(int64_t q=1;q<=8;++q) {
      double cm=0,sm=0;for(const auto t:times){cm+=std::cos(tau*q*t/32.);sm+=std::sin(tau*q*t/32.);}
      cm/=n;sm/=n;
      double cc=0,ss=0,cs=0,ayc=0,ays=0,byc=0,bys=0;
      for(const auto t:times) {
        const double c=std::cos(tau*q*t/32.)-cm,s=std::sin(tau*q*t/32.)-sm;
        const double a=x[b][pairs[p][0]][t][fi]-am,z=x[b][pairs[p][1]][t][fj]-bm;
        cc+=c*c;ss+=s*s;cs+=c*s;ayc+=a*c;ays+=a*s;byc+=z*c;bys+=z*s;
      }
      cc/=n;ss/=n;cs/=n;ayc/=n;ays/=n;byc/=n;bys/=n;
      const auto det=cc*ss-cs*cs,trace=cc+ss;
      if(!(trace>1e-12&&det>1e-8*trace*trace)){full_rank=false;continue;}
      ac[q-1]=(ayc*ss-ays*cs)/det;as[q-1]=(ays*cc-ayc*cs)/det;
      bc[q-1]=(byc*ss-bys*cs)/det;bs[q-1]=(bys*cc-byc*cs)/det;
      ea+=ac[q-1]*ac[q-1]+as[q-1]*as[q-1];eb+=bc[q-1]*bc[q-1]+bs[q-1]*bs[q-1];
    }
    if(!full_rank||!(ea>1e-12&&eb>1e-12))continue;
    ok[b][index]=true;const auto denominator=std::sqrt(ea)*std::sqrt(eb);
    std::array<double,4> metrics{};
    for(int64_t q=0;q<8;++q) {
      const int band=q<3?0:2;
      metrics[band]+=(ac[q]*bc[q]+as[q]*bs[q])/denominator;
      metrics[band+1]+=(as[q]*bc[q]-ac[q]*bs[q])/denominator;
    }
    for(int64_t coordinate=0;coordinate<4;++coordinate)v[b][index*4+coordinate]=float(metrics[coordinate]);
  }
  return out;
}
torch::Tensor scalar_native(const ScalarBank &bank) {
  const auto v=bank.values.accessor<float,2>();auto out=torch::zeros({bank.values.size(0),12},torch::kFloat32);auto z=out.accessor<float,2>();
  for(int64_t b=0;b<out.size(0);++b)for(int64_t p=0;p<3;++p) {
    std::array<double,4> four{};
    for(int64_t f=0;f<9;++f)for(int64_t j=0;j<4;++j)four[j]+=double(v[b][(p*9+f)*4+j])/9.;
    double energy=0;for(double value:four)energy+=value*value;const auto norm=std::max(std::sqrt(energy),1e-12);
    for(int64_t j=0;j<4;++j)z[b][p*4+j]=float(four[j]/norm);
  }
  return out;
}
void compare_scalar(const rpb::Input &x,const rpb::Config &c) {
  const auto reference=scalar_bank(x);
  const auto actual=rpb::spectral_relation_bank(x,c);
  close(actual.values,reference.values.to(torch::kCUDA),"independent scalar fixed-bin real/imag relations",3e-6,2e-6);
  exact(actual.common_counts,reference.counts.to(torch::kCUDA),"all27 common-visible original-time counts");
  exact(actual.eligible,reference.eligible.to(torch::kCUDA),"all-eight trace/determinant and energy eligibility");
}
void primitive_contract() {
  const auto c=configuration();auto x=wave_input(c);
  compare_scalar(x,c);
  auto m=x.observed.to(torch::kCPU).contiguous();auto mask=m.accessor<bool,4>();
  for(int64_t b=0;b<2;++b)for(int64_t ch=0;ch<3;++ch)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f)
    mask[b][ch][t][f]=(t+2*f+ch+b)%7!=0&&!(t>=9+ch&&t<12+ch);
  x.observed=m.to(torch::kCUDA);x.data=torch::where(x.observed,x.data,torch::zeros_like(x.data));
  compare_scalar(x,c);const auto original=rpb::spectral_relation_bank(x,c);
  check(original.eligible.all().item<bool>(),"independent gappy feature support remains eligible");
  auto poison=x;poison.data=torch::where(x.observed,x.data,torch::full_like(x.data,std::numeric_limits<float>::quiet_NaN()));
  exact(rpb::spectral_relation_bank(poison,c).values,original.values,"hidden NaNs sanitized before all arithmetic");
  auto affine=x;affine.data=x.data.clone();
  for(int64_t ch=0;ch<3;++ch)for(int64_t f=0;f<3;++f)
    affine.data.select(1,ch).select(2,f).mul_(.7+.4*ch+.13*f).add_(-.9+.5*f-.17*ch);
  close(rpb::spectral_relation_bank(affine,c).values,original.values,"per-channel/feature positive gain and constant offset invariance",2e-5,4e-6);
  auto reversal=x;reversal.data=x.data.flip({2});reversal.observed=x.observed.flip({2});
  const auto reversed=rpb::spectral_relation_bank(reversal,c);
  exact(reversed.common_counts,original.common_counts,"reversal common tick count");
  exact(reversed.eligible,original.eligible,"trace/determinant eligibility invariant under rotated basis");
  const auto signs=torch::tensor({1.f,-1.f,1.f,-1.f},torch::kFloat32).to(torch::kCUDA);
  close(reversed.values.reshape({2,27,4}),original.values.reshape({2,27,4})*signs,"spectral Re-even/Im-odd joint value and mask reversal",2e-5,4e-6);
  const auto order=torch::tensor({2,0,1},torch::kInt64).to(torch::kCUDA);auto permutation=x;
  permutation.data=x.data.index_select(1,order);permutation.observed=x.observed.index_select(1,order);
  permutation.channel_ids=x.channel_ids.to(torch::kCUDA).index_select(0,order);
  exact(rpb::spectral_relation_bank(permutation,c).values,original.values,"canonical semantic pair layout under channel permutation");
  auto isolate=x;isolate.observed=x.observed.clone();isolate.observed.select(1,2).fill_(false);
  const auto isolated=rpb::spectral_relation_bank(isolate,c);
  exact(isolated.values.narrow(1,0,36),original.values.narrow(1,0,36),"pair01 unaffected by other channel mask");
  exact(isolated.values.narrow(1,36,72),torch::zeros({2,72},original.values.options()),"unsupported02/12 exactzero");
  auto singular=wave_input(c);auto sparse=torch::zeros_like(singular.observed);
  for(int64_t t=0;t<32;t+=2)sparse.select(2,t).fill_(true);
  singular.observed=sparse;const auto rank_loss=rpb::spectral_relation_bank(singular,c);
  check(rank_loss.common_counts.eq(16).all().item<bool>()&&!rank_loss.eligible.any().item<bool>(),"q8 even-tick rank loss rejects whole shared bank");
  exact(rank_loss.values,torch::zeros_like(rank_loss.values),"rank loss is exactzero, never a fabricated spectrum");
  auto four=wave_input(c);four.observed=torch::zeros_like(four.observed);
  for(const int64_t t:{0,3,13,22})four.observed.select(2,t).fill_(true);
  const auto four_bank=rpb::spectral_relation_bank(four,c);compare_scalar(four,c);
  check(four_bank.common_counts.eq(4).all().item<bool>()&&four_bank.eligible.all().item<bool>(),"four distinct legal ticks with full-rank fixed basis accepted");
  auto three=four;three.observed=four.observed.clone();three.observed.select(2,22).fill_(false);
  const auto three_bank=rpb::spectral_relation_bank(three,c);
  check(three_bank.common_counts.eq(3).all().item<bool>()&&!three_bank.eligible.any().item<bool>(),"three ticks below support rule");
  exact(three_bank.values,torch::zeros_like(three_bank.values),"too few legal ticks exactzero");
  auto empty=poison;empty.observed=torch::zeros_like(x.observed);
  const auto none=rpb::spectral_relation_bank(empty,c);check(!none.eligible.any().item<bool>(),"all absent has no spectral eligibility");
  exact(none.values,torch::zeros_like(none.values),"all absent including hidden NaNs exactzero");
  auto constant=wave_input(c);constant.data=torch::zeros_like(constant.data).set_requires_grad(true);
  const auto zero=rpb::spectral_relation_bank(constant,c);exact(zero.values,torch::zeros_like(zero.values),"zero total bank energy exactzero");
  zero.values.sum().backward();finite(constant.data.grad(),"inactive energy CUDA derivatives finite");
  auto differentiable=x;differentiable.data=x.data.detach().clone().set_requires_grad(true);
  const auto grad_bank=rpb::spectral_relation_bank(differentiable,c);grad_bank.values.square().sum().backward();
  finite(differentiable.data.grad(),"active fixed-bank CUDA derivatives finite");
  check(differentiable.data.grad().abs().sum().item<double>()>0,"spectral bank has nonzero visible-data derivative");
  exact(differentiable.data.grad().masked_select(~x.observed),torch::zeros_like(differentiable.data.grad().masked_select(~x.observed)),"hidden data has zero bank gradient");
  auto invalid=x;invalid.data=x.data.clone();invalid.data[0][0][0][0]=std::numeric_limits<float>::infinity();invalid.observed=x.observed.clone();invalid.observed[0][0][0][0]=true;
  rejects([&]{rpb::spectral_relation_bank(invalid,c);},"nonfinite visible value rejected");
  auto wrong=c;wrong.device=torch::kCPU;rejects([&]{rpb::spectral_relation_bank(x,wrong);},"CPU model helper rejected");
  rejects([&]{rpb::unit_spectral_relations(torch::zeros({2,12},torch::kFloat32));},"CPU unit norm rejected");
  const auto arbitrary=torch::tensor(std::vector<float>{0,0,0,0,3,4,0,0,1e-15f,-2e-15f,0,0},torch::kFloat32).reshape({1,12}).to(torch::kCUDA).set_requires_grad(true);
  const auto normalized=rpb::unit_spectral_relations(arbitrary);exact(normalized.narrow(1,0,4),torch::zeros({1,4},arbitrary.options()),"unit zero norm exactzero");
  check(normalized[0][8].item<float>()>0&&normalized[0][8].item<float>()<.01,"subfloor norm divided by floor");
  normalized.sum().backward();finite(arbitrary.grad(),"unit norm zero/subfloor derivatives finite");
}
void off_grid_contract() {
  const auto c=configuration();
  for(const int edge:{0,1})for(const int slow_sign:{-1,1})for(const int fast_sign:{-1,1})for(const double ratio:{2.,.5}) {
    auto x=wave_input(c);auto host=x.data.to(torch::kCPU).contiguous();auto values=host.accessor<float,4>();
    auto masks=x.observed.to(torch::kCPU).contiguous();auto legal=masks.accessor<bool,4>();
    const double ps=edge?19.7:12.3,pf=edge?6.9:4.6;
    for(int64_t b=0;b<2;++b)for(int64_t ch=0;ch<3;++ch)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f) {
      const double phase=b?3.19:.37,ds=ch*slow_sign*.73,df=ch*fast_sign*.47;
      values[b][ch][t][f]=float((.7+.2*ch+.07*f)*(ratio*std::cos(tau*(t+ds)/ps+phase)+
          std::cos(tau*(t+df)/pf-phase))-.31+.19*f);
      legal[b][ch][t][f]=(t+f+2*ch+b)%6!=0&&!(t>=5+ch&&t<8+ch);
    }
    x.data=host.to(torch::kCUDA);x.observed=masks.to(torch::kCUDA);
    compare_scalar(x,c); // Values, not task accuracy: off-grid/mask leakage remains real.
  }
}
void exact_bin_semantics() {
  const auto c=configuration();auto pure=wave_input(c,16.,16.,1.); // both terms occupy the same genericq2.
  pure.data=torch::zeros_like(pure.data);auto host=pure.data.to(torch::kCPU).contiguous();auto x=host.accessor<float,4>();
  const std::array<double,3> phases{.13,.41,-.29};
  for(int64_t b=0;b<2;++b)for(int64_t ch=0;ch<3;++ch)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f)
    x[b][ch][t][f]=float(.2*f+(.8+.1*f+.2*ch)*std::cos(tau*2*t/32.+phases[ch]));
  pure.data=host.to(torch::kCUDA);rpb::SpectralTemporalRelationModel model(c);model->eval();
  const auto native=model->encode(pure).z_contextual_global.narrow(1,20,12);
  constexpr std::array<std::array<int64_t,2>,3> pairs{{{0,1},{0,2},{1,2}}};
  for(int64_t p=0;p<3;++p) {
    const double delta=phases[pairs[p][1]]-phases[pairs[p][0]];
    check(std::abs(native[0][4*p].item<double>()-std::cos(delta))<2e-6&&
          std::abs(native[0][4*p+1].item<double>()-std::sin(delta))<2e-6,"independent all-pair conjugate sign and real phase reference");
    check(native[0].narrow(0,4*p+2,2).abs().max().item<double>()<1e-10,"pure low-frequency has no high-band coordinates");
  }
  for(const double ratio:{2.,.5}) {
    const auto mixture=wave_input(c,16.,32./6.,ratio);const auto z=model->encode(mixture).z_contextual_global.narrow(1,20,12).reshape({2,3,4});
    const auto low=z.narrow(2,0,2).square().sum(-1).sqrt(),high=z.narrow(2,2,2).square().sum(-1).sqrt();
    close(low/high,torch::full_like(low,ratio*ratio),"joint pair norm retains generic low/high energy balance",1e-5,3e-6);
  }
}
void cuda_model_contract() {
  const auto c=configuration();auto raw=input(c,3);raw.observed[0][2].narrow(0,13,3).fill_(false);
  raw.data=torch::where(raw.observed,raw.data,torch::zeros_like(raw.data));const auto scaler=rpb::fit_scaler(raw,c);
  const auto mean=scaler.mean.clone(),scale=scaler.scale.clone();const auto x=scaler.transform(raw,c);
  torch::manual_seed(19101);rpb::UnitTemporalRelationModel v18(c);
  const auto cpu_rng=torch::rand({8}),cuda_rng=torch::rand({8},torch::TensorOptions().device(torch::kCUDA));
  torch::manual_seed(19101);rpb::SpectralTemporalRelationModel v19(c);
  exact(cpu_rng,torch::rand({8}),"v19 preserves v18 CPU constructor RNG");
  exact(cuda_rng,torch::rand({8},torch::TensorOptions().device(torch::kCUDA)),"v19 preserves v18 CUDA constructor RNG");
  v18->eval();v19->eval();const auto p18=v18->named_parameters(),p19=v19->named_parameters();
  check(p18.keys()==p19.keys(),"same registered parameter names/order");int64_t total=0,trainable=0,frozen=0;
  for(const auto &p:p19) {
    exact(p.value(),p18[p.key()],"same initialization and every common value");check(p.value().is_cuda(),"model parameters actual CUDA");total+=p.value().numel();
    if(p.value().requires_grad())trainable+=p.value().numel();else frozen+=p.value().numel();
  }
  check(total==rpb::kSpectralTemporalRelationParameterCount&&trainable==rpb::kSpectralTemporalRelationTrainableParameterCount&&
        frozen==rpb::kSpectralTemporalRelationFrozenParameterCount&&total==226877&&trainable==226445&&frozen==432,"unchanged registered/trainable/frozen capacity");
  const auto b18=v18->named_buffers(),b19=v19->named_buffers();check(b18.keys()==b19.keys(),"no new buffer registration");
  for(const auto &b:b18)exact(b.value(),b19[b.key()],"same initial buffers");v19->validate_fixed_prior();
  check(std::string(rpb::kSpectralTemporalRelationModelTag)=="RPB-v19"&&
        std::string(rpb::kSpectralTemporalRelationArchitectureId)!=rpb::kUnitTemporalRelationArchitectureId&&
        std::string(rpb::kSpectralTemporalRelationLayout).find("ReLow-ImLow-ReHigh-ImHigh")!=std::string::npos,"distinct mixed spectral model/layout identity");
  const auto old=v18->encode(x),initial=v19->encode(x);exact(initial.z_contextual_global.narrow(1,0,20),old.z_contextual_global.narrow(1,0,20),"shape20 exactly preserved");
  exact(initial.z_local,old.z_local,"local diagnostics preserved");exact(initial.z_contextual,old.z_contextual,"contextual diagnostics preserved");
  close(initial.z_contextual_global.narrow(1,20,12),scalar_native(scalar_bank(x)).to(torch::kCUDA),"actual fixed432 feature means and joint norm scalar reference",3e-6,3e-6);
  const auto spectral_initial=initial.z_contextual_global.narrow(1,20,12).detach().clone();
  const auto q=rpb::make_training_mask(x.observed,c,19103);const auto masks=rpb::mask_from_hidden(x.observed,q.hidden,c);
  const rpb::Input visible{x.data,masks.visible,x.channel_ids,x.endpoints,x.sampling_interval};
  const auto new_q=v19->forward(x,q.hidden),old_q=v18->forward(x,q.hidden);
  exact(new_q.encoding.z_contextual_global,v19->encode(visible).z_contextual_global,"derived forward binds new mask-visible spectral encode");
  exact(new_q.reconstruction,v19->decode(new_q.encoding.z_contextual_global,x.channel_ids),"sole32 native decoder input");
  exact(new_q.target_counts,old_q.target_counts,"original query support unchanged");
  auto changed=x;changed.data=torch::where(q.target,x.data+1000.,x.data);
  exact(v19->forward(changed,q.hidden).encoding.z_contextual_global,new_q.encoding.z_contextual_global,"Q-hidden target values cannot enter spectral/shape input");
  auto poison=x;poison.data=torch::where(x.observed,x.data,torch::full_like(x.data,std::numeric_limits<float>::quiet_NaN()));
  exact(v19->encode(poison).z_contextual_global,initial.z_contextual_global,"natural hidden NaNs cannot enter model");
  const auto prior=v19->grouped_odd_relation_projection->weight.detach().clone(),shape_initial=v19->native_shape_projection->weight.detach().clone();
  const auto backbone_weight=v19->named_parameters()["backbone.patch_projection.weight"],decoder_weight=v19->named_parameters()["backbone.decoder_first.weight"];
  const auto backbone_initial=backbone_weight.detach().clone(),decoder_initial=decoder_weight.detach().clone();
  torch::optim::AdamW optimizer(v19->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for(int64_t step=0;step<4;++step) {
    const auto train_mask=rpb::make_training_mask(x.observed,c,19110+step);
    const auto context=rpb::context_deletion::make_plan(train_mask,x.channel_ids,c,19107,step,rpb::ContextDeletionRecipe::coordinate15_v1);
    const rpb::Input current{torch::where(context.visible,x.data,torch::zeros_like(x.data)),context.visible,x.channel_ids,x.endpoints,x.sampling_interval};
    optimizer.zero_grad();const auto encoding=v19->encode(current);
    const auto native=rpb::compact_reconstruction_export(encoding,c);
    const auto prediction=v19->decode(native,x.channel_ids);
    const auto loss=rpb::hierarchical_huber(prediction,x.data.detach(),train_mask.target,train_mask.eligible_channels,1.);
    check(native.sizes()==torch::IntArrayRef({3,32})&&native.is_cuda()&&loss.loss.is_cuda()&&torch::isfinite(loss.loss).item<bool>(),"original CUDA waveform Huber1 through native32");
    loss.loss.backward();check(!v19->grouped_odd_relation_projection->weight.grad().defined(),"fixed432 has no gradient");
    for(const auto &w:{v19->native_shape_projection->weight,backbone_weight,decoder_weight}) {
      finite(w.grad(),"finite trainable CUDA gradient");check(w.grad().is_cuda()&&w.grad().abs().sum().item<double>()>0,"shape/backbone/decoder nonzero gradient");
    }
    torch::nn::utils::clip_grad_norm_(v19->parameters(),1.,2.,true);optimizer.step();v19->validate_fixed_prior();
    exact(v19->grouped_odd_relation_projection->weight,prior,"literal frozen432 unchanged after optimizer");
    exact(v19->encode(x).z_contextual_global.narrow(1,20,12),spectral_initial,"fixed spectral coordinates exactly preserved after actual update");
  }
  check(!torch::equal(v19->native_shape_projection->weight,shape_initial)&&!torch::equal(backbone_weight,backbone_initial)&&!torch::equal(decoder_weight,decoder_initial),"four updates change learned shape/backbone/decoder");
  check(optimizer.state().find(v19->grouped_odd_relation_projection->weight.unsafeGetTensorImpl())==optimizer.state().end(),"no fabricated fixed432 AdamW state");
  for(const auto &w:{v19->native_shape_projection->weight,backbone_weight,decoder_weight}) {
    const auto it=optimizer.state().find(w.unsafeGetTensorImpl());check(it!=optimizer.state().end(),"actual trainable AdamW state exists");
    const auto *adam=dynamic_cast<const torch::optim::AdamWParamState *>(it->second.get());
    check(adam&&adam->step()==4&&adam->exp_avg().is_cuda()&&adam->exp_avg_sq().is_cuda(),"four-step CUDA first/second moments");
  }
  exact(scaler.mean,mean,"TRAIN scaler mean immutable");exact(scaler.scale,scale,"TRAIN scaler scale immutable");
  {
    torch::NoGradGuard guard;std::vector<std::pair<std::string,torch::Tensor>> before;
    for(const auto &p:v19->named_parameters())before.emplace_back(p.key(),p.value().detach().clone());
    auto reverse=x;reverse.data=x.data.flip({2});reverse.observed=x.observed.flip({2});
    const auto signs=torch::tensor({1.f,-1.f,1.f,-1.f},torch::kFloat32).to(torch::kCUDA);
    close(v19->encode(reverse).z_contextual_global.narrow(1,20,12).reshape({3,3,4}),spectral_initial.reshape({3,3,4})*signs,"trained fixed12 retains mixed even/odd reversal",3e-5,1e-5);
    auto absent=poison;absent.observed=torch::zeros_like(x.observed);const auto empty=v19->encode(absent);
    check(!empty.sample_valid_mask.any().item<bool>(),"all absent invalid sample");exact(empty.z_contextual_global,torch::zeros_like(empty.z_contextual_global),"all absent native32 exactzero");
    auto one=poison;one.observed=torch::zeros_like(x.observed);one.observed.select(1,0).copy_(x.observed.select(1,0));
    exact(v19->encode(one).z_contextual_global.narrow(1,20,12),torch::zeros_like(spectral_initial),"single channel cannot fabricate spectral relations");
    for(const auto &p:before)exact(p.second,v19->named_parameters()[p.first],"serving never changes parameter state");
  }
  torch::Tensor native;{torch::NoGradGuard guard;native=v19->encode(x).z_contextual_global.detach().clone();}
  std::stringstream bytes;torch::serialize::OutputArchive archive_out;v19->save(archive_out);archive_out.save_to(bytes);
  rpb::SpectralTemporalRelationModel restored(c);restored->grouped_odd_relation_projection->weight.set_requires_grad(true);
  std::shared_ptr<torch::nn::Module> erased=restored.ptr();torch::serialize::InputArchive archive_in;archive_in.load_from(bytes,torch::kCUDA);erased->load(archive_in);
  restored->eval();restored->validate_fixed_prior();check(!restored->grouped_odd_relation_projection->weight.requires_grad(),"type-erased inherited load refreezes unchanged literal432");
  {torch::NoGradGuard guard;exact(restored->encode(x).z_contextual_global,native,"trained CUDA save/load exact spectral native32");}
  const auto original=v19->named_parameters(),loaded=restored->named_parameters();check(original.keys()==loaded.keys(),"saved named association preserved");
  for(const auto &p:original)exact(p.value(),loaded[p.key()],"all saved parameter values exact");
  rpb::SpectralTemporalRelationModel corrupt(c);{torch::NoGradGuard guard;corrupt->grouped_odd_relation_projection->weight[0][0][0]=0;}
  std::stringstream invalid;torch::serialize::OutputArchive invalid_out;corrupt->save(invalid_out);invalid_out.save_to(invalid);
  torch::serialize::InputArchive invalid_in;invalid_in.load_from(invalid,torch::kCUDA);rejects([&]{restored->load(invalid_in);},"changed fixed432 checkpoint rejected");
}
} // namespace
int main() try {
  check(torch::cuda::is_available(),"spectral relation admission requires actual CUDA");torch::set_num_threads(1);
  primitive_contract();off_grid_contract();exact_bin_semantics();cuda_model_contract();
  std::cout<<"Spectral temporal relation CUDA admission passed\n"<<MULTIBAND_SCREEN_SOURCE_ID<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
