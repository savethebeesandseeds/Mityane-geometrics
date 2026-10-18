// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Prospective synthetic CUDA fixture only. No quality data or old artifacts.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v13.h"
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v13;
namespace im=cp::implementation;
int64_t checks=0,direct_pool_kernel_calls=0;
constexpr double kManualTolerance=3e-5,kGaugeTolerance=2e-6;
constexpr double kF64FiniteDifferenceStep=1e-5,kF64FiniteDifferenceTolerance=2e-8;
constexpr double kConstantVarianceTolerance=1e-10; // times (1+max supported |hidden| squared), F32 only
constexpr double kF32FiniteDifferenceStep=2e-3,kF32FiniteDifferenceTolerance=2e-3;
constexpr double kFiniteDifferenceSignificantDerivative=1e-2;
constexpr int kSigns[3][4]={{1,1,-1,-1},{1,-1,1,-1},{1,-1,-1,1}};
void check(bool v,const std::string &why){++checks;TORCH_CHECK(v,"[CPB-v13 hybrid-pool proof] ",why);}
void near(const torch::Tensor &a,const torch::Tensor &b,const std::string &why,double tol=kManualTolerance){
  check(torch::allclose(a.detach().to(torch::kCPU,torch::kFloat64),b.detach().to(torch::kCPU,torch::kFloat64),tol,tol),why);
}
template<class F>void rejects(F f,const std::string &why){bool yes=false;try{f();}catch(const std::exception&){yes=true;}check(yes,why);}
torch::Tensor f64(const torch::Tensor &t){return t.detach().to(torch::kCPU,torch::kFloat64).contiguous();}
im::TemporalPoolOutput counted_pool(const torch::Tensor &h,const torch::Tensor &s,const torch::Tensor &v,
    const torch::Tensor &p,const torch::Tensor &w,const torch::Tensor &b,const im::Config &c){
  ++direct_pool_kernel_calls;return im::pool_original_position_bins(h,s,v,p,w,b,c);
}
struct Reference{
  torch::Tensor output,weights,bins,local,global,mixture,bin_mean,bin_variance,bin_regularized_root,bin_dispersion;
  torch::Tensor hidden_grad,weight_grad,bias_grad,position_grad,score_weight_grad;
};

// Independent scalar reference and chain rule: no producer helper or autograd.
Reference reference(const torch::Tensor &h0,const torch::Tensor &s0,const torch::Tensor &p0,
    const torch::Tensor &w0,const torch::Tensor &b0,const torch::Tensor &scorew0,
    const torch::Tensor &up0,int64_t K,const im::TemporalPoolOutput *captured=nullptr){
  const auto h=f64(h0),s=f64(s0),w=f64(w0),b=f64(b0),scorew=f64(scorew0),up=f64(up0);
  const auto p=p0.detach().to(torch::kCPU).contiguous();const auto R=h.size(0),T=h.size(1),W=h.size(2),D=w.size(0);
  Reference out{torch::zeros({R,D},torch::kFloat64),torch::zeros({R,T},torch::kFloat64),torch::zeros({R,4},torch::kBool),
      torch::zeros({R,T},torch::kFloat64),torch::zeros({R,T},torch::kFloat64),torch::zeros({R,4,T},torch::kFloat64),
      torch::zeros({R,4,W},torch::kFloat64),torch::zeros({R,4,W},torch::kFloat64),
      torch::full({R,4,W},cp::kTemporalDispersionRootOffset,torch::kFloat64),torch::zeros({R,4,W},torch::kFloat64),
      torch::zeros_like(h),torch::zeros_like(w),torch::zeros_like(b),torch::zeros({K,1},torch::kFloat64),torch::zeros_like(scorew)};
  const auto hv=h.accessor<double,3>();const auto sv=s.accessor<double,2>(),wv=w.accessor<double,2>();
  const auto bv=b.accessor<double,1>();const auto pv=p.accessor<int64_t,2>();
  const auto uv=up.accessor<double,2>(),sw=scorew.accessor<double,2>();
  auto ov=out.output.accessor<double,2>(),aw=out.weights.accessor<double,2>(),lw=out.local.accessor<double,2>(),gw0=out.global.accessor<double,2>();
  auto mw=out.mixture.accessor<double,3>(),bm=out.bin_mean.accessor<double,3>(),bv0=out.bin_variance.accessor<double,3>();auto bins=out.bins.accessor<bool,2>();
  auto root0=out.bin_regularized_root.accessor<double,3>(),disp0=out.bin_dispersion.accessor<double,3>();
  // Captured-F32 adjoint mode uses actual saved intermediates, never observed
  // gradients. F64 forward/FD mode independently recomputes the whole graph.
  const auto cl=captured?f64(captured->local_weights):torch::Tensor{},cg=captured?f64(captured->global_weights):torch::Tensor{};
  const auto cm=captured?f64(captured->mixture_weights):torch::Tensor{},cu=captured?f64(captured->bin_mean):torch::Tensor{};
  const auto cv=captured?f64(captured->bin_variance):torch::Tensor{},cr=captured?f64(captured->bin_regularized_root):torch::Tensor{};
  const auto cd=captured?f64(captured->bin_dispersion):torch::Tensor{};
  auto gh=out.hidden_grad.accessor<double,3>();auto gw=out.weight_grad.accessor<double,2>();
  auto gb=out.bias_grad.accessor<double,1>();auto gp=out.position_grad.accessor<double,2>(),gsw=out.score_weight_grad.accessor<double,2>();
  for(int64_t r=0;r<R;++r){
    std::vector<int64_t> all;for(int64_t t=0;t<T;++t)if(pv[r][t]>=0)all.push_back(t);
    if(all.empty())continue;
    double maximum=-INFINITY,denominator=0.;for(auto t:all)maximum=std::max(maximum,sv[r][t]);
    for(auto t:all)denominator+=std::exp(sv[r][t]-maximum);
    std::vector<double> global(T,0.),global_hidden(W,0.);
    for(auto t:all){global[t]=captured?cg[r][t].item<double>():std::exp(sv[r][t]-maximum)/denominator;gw0[r][t]=global[t];
      for(int64_t k=0;k<W;++k)global_hidden[k]+=global[t]*hv[r][t][k];}
    for(int64_t j=0;j<4;++j){
      std::vector<int64_t> keys;for(auto t:all)if(pv[r][t]>=j*(K/4)&&pv[r][t]<(j+1)*(K/4))keys.push_back(t);
      if(keys.empty())continue;bins[r][j]=true;maximum=-INFINITY;denominator=0.;
      for(auto t:keys)maximum=std::max(maximum,sv[r][t]);for(auto t:keys)denominator+=std::exp(sv[r][t]-maximum);
      std::vector<double> local(T,0.),local_hidden(W,0.),pooled(W,0.),variance(W,0.),residual(W,0.),a(W,0.),b_adj(W,0.),mixture(T,0.),q(T,0.);
      for(auto t:keys){local[t]=captured?cl[r][t].item<double>():std::exp(sv[r][t]-maximum)/denominator;lw[r][t]=local[t];
        for(int64_t k=0;k<W;++k)local_hidden[k]+=local[t]*hv[r][t][k];}
      for(int64_t k=0;k<W;++k)pooled[k]=.5*local_hidden[k]+.5*global_hidden[k];
      if(captured)for(int64_t k=0;k<W;++k)pooled[k]=cu[r][j][k].item<double>();
      for(auto t:all){mixture[t]=captured?cm[r][j][t].item<double>():.5*local[t]+.5*global[t];mw[r][j][t]=mixture[t];aw[r][t]+=mixture[t];
        for(int64_t k=0;k<W;++k){const double delta=hv[r][t][k]-pooled[k];
          variance[k]+=mixture[t]*delta*delta;residual[k]+=mixture[t]*delta;}}
      std::vector<double> roots(W),dispersion(W);
      for(int64_t k=0;k<W;++k){if(captured)variance[k]=cv[r][j][k].item<double>();
        roots[k]=captured?cr[r][j][k].item<double>():std::sqrt(variance[k]+cp::kTemporalDispersionRegularizer);
        dispersion[k]=captured?cd[r][j][k].item<double>():roots[k]-cp::kTemporalDispersionRootOffset;
        bm[r][j][k]=pooled[k];bv0[r][j][k]=variance[k];root0[r][j][k]=roots[k];disp0[r][j][k]=dispersion[k];}
      for(int64_t d=j*(D/4);d<(j+1)*(D/4);++d){ov[r][d]=bv[d];gb[d]+=uv[r][d];
        const bool is_mean=d-j*(D/4)<D/8;
        for(int64_t k=0;k<W;++k){const double feature=is_mean?pooled[k]:dispersion[k];ov[r][d]+=feature*wv[d][k];gw[d][k]+=uv[r][d]*feature;
          (is_mean?a[k]:b_adj[k])+=uv[r][d]*wv[d][k];}}
      // Full actual-expression derivative. Do not assume residual=sum p*(h-mu)
      // is exactly zero: float softmax mass/two-sum mean need not be exact.
      for(int64_t k=0;k<W;++k){b_adj[k]/=2.*roots[k];a[k]-=2.*b_adj[k]*residual[k];}
      double local_q=0.,global_q=0.;
      for(auto t:all){for(int64_t k=0;k<W;++k){const double delta=hv[r][t][k]-pooled[k];q[t]+=a[k]*hv[r][t][k]+b_adj[k]*delta*delta;}
        local_q+=local[t]*q[t];global_q+=global[t]*q[t];}
      for(auto t:all){const double ds=.5*local[t]*(q[t]-local_q)+.5*global[t]*(q[t]-global_q);
        gp[pv[r][t]][0]+=ds;
        for(int64_t k=0;k<W;++k){const double th=std::tanh(hv[r][t][k]);
          // Mean is two separate sums; its coefficient must not be replaced by
          // the rounded stored mixture when checking actual F32 residuals.
          gh[r][t][k]+=(.5*local[t]+.5*global[t])*a[k]+2.*mixture[t]*b_adj[k]*(hv[r][t][k]-pooled[k])+ds*sw[0][k]*(1.-th*th);gsw[0][k]+=ds*th;}
      }
    }
  }
  return out;
}

// Finite differences independently evaluate the complete score(tanh(hidden))
// and moment graph. The F64 scalar reference is never a detached-weight oracle.
void finite_differences(const torch::Tensor &h,const torch::Tensor &positions,const torch::Tensor &w,
    const torch::Tensor &b,const torch::Tensor &sw,const torch::Tensor &sb,const torch::Tensor &pp,
    const torch::Tensor &up,const Reference &analytic,const im::Config &c){
  torch::NoGradGuard guard;
  const auto score_of=[&](const torch::Tensor &hidden,const torch::Tensor &score_weight,const torch::Tensor &score_bias,const torch::Tensor &pos){
    const auto valid=positions.ge(0).to(hidden.device());
    const auto safe=torch::where(valid.unsqueeze(-1),hidden,torch::zeros_like(hidden));
    return torch::matmul(torch::tanh(safe),score_weight.transpose(0,1)).squeeze(-1)+score_bias+
        pos.index_select(0,positions.to(hidden.device()).clamp_min(0).reshape({-1})).reshape(positions.sizes());
  };
  const auto objective64=[&](const torch::Tensor &hidden,const torch::Tensor &weight,const torch::Tensor &pos){
    const auto score=score_of(hidden,f64(sw),f64(sb),pos);
    return (reference(hidden,score,positions,weight,b,sw,up,16).output*f64(up)).sum().item<double>();
  };
  const auto objective32=[&](const torch::Tensor &hidden,const torch::Tensor &weight,const torch::Tensor &pos){
    return (counted_pool(hidden,score_of(hidden,sw,sb,pos),positions.ge(0),positions,weight,b,c).projected*up).sum().item<double>();
  };
  const auto initial_h64=f64(h),initial_w64=f64(w),initial_p64=f64(pp);
  const auto analytic64=reference(initial_h64,score_of(initial_h64,f64(sw),f64(sb),initial_p64),positions,
      initial_w64,b,sw,up,16);
  const auto assert_fd=[&](double actual,double expected,double tolerance,const std::string &why){
    check(std::isfinite(actual)&&std::isfinite(expected)&&std::abs(actual-expected)<=tolerance+tolerance*std::abs(expected),why);
  };
  int64_t significant_points=0;
  for(int which=0;which<3;++which)for(int coordinate=0;coordinate<6;++coordinate){
    auto h64=f64(h),w64=f64(w),p64=f64(pp);double expected=0.;
    // Supported hidden entries, both mean/variance row weights, and original
    // position entries cover attention, centering, layout and projection routes.
    const int64_t hidden_index=coordinate*64+coordinate,weight_index=(coordinate*5)*64+coordinate,position_index=coordinate*3%16;
    auto flat=which==0?h64.flatten():which==1?w64.flatten():p64.flatten();
    const auto index=which==0?hidden_index:which==1?weight_index:position_index;
    expected=(which==0?analytic64.hidden_grad:which==1?analytic64.weight_grad:analytic64.position_grad).flatten()[index].item<double>();
    const auto old=flat[index].item<double>();flat[index]=old+kF64FiniteDifferenceStep;const auto plus64=objective64(h64,w64,p64);
    flat[index]=old-kF64FiniteDifferenceStep;const auto minus64=objective64(h64,w64,p64);
    assert_fd((plus64-minus64)/(2*kF64FiniteDifferenceStep),expected,kF64FiniteDifferenceTolerance,"independent full-expression F64 finite difference");
    expected=(which==0?analytic.hidden_grad:which==1?analytic.weight_grad:analytic.position_grad).flatten()[index].item<double>();
    significant_points+=std::abs(expected)>kFiniteDifferenceSignificantDerivative;
    auto h32=h.detach().clone(),w32=w.detach().clone(),p32=pp.detach().clone();auto flat32=which==0?h32.flatten():which==1?w32.flatten():p32.flatten();
    const auto old32=flat32[index].item<float>();flat32[index]=old32+kF32FiniteDifferenceStep;const auto plus32=objective32(h32,w32,p32);
    flat32[index]=old32-kF32FiniteDifferenceStep;const auto minus32=objective32(h32,w32,p32);
    assert_fd((plus32-minus32)/(2*kF32FiniteDifferenceStep),expected,kF32FiniteDifferenceTolerance,"actual CUDA F32 kernel finite difference");
  }
  check(significant_points>=3,"at least three actual FD reference derivatives exceed fixed1e-2 noise-floor threshold");
}

// Fixed unit-orthonormal R16 directions: signs repeated over four positions,
// divided by4. For four bin means, the three coefficients are signed sums.
torch::Tensor between_contrasts(const torch::Tensor &positions){
  const auto means=positions.to(torch::kFloat64).reshape({4,4}).mean(1);
  const auto signs=torch::tensor(std::vector<double>{1,1,-1,-1,1,-1,1,-1,1,-1,-1,1},means.options()).reshape({3,4});
  return torch::matmul(signs,means);
}

void actual_model_paths(im::Model &model,const im::Config &c){
  torch::NoGradGuard guard;model->eval();const int64_t B=2,C=3,H=128,F=9;
  auto data=torch::sin(torch::arange(B*C*H*F,torch::TensorOptions().dtype(torch::kFloat32).device(c.device))*.017).reshape({B,C,H,F});
  auto observed=torch::ones({B,C,H,F},torch::TensorOptions().dtype(torch::kBool).device(c.device));
  observed[0][1].zero_();observed[0][1].narrow(0,120,8).fill_(true);observed[0][2].zero_();
  observed[1][0].zero_();for(auto p:{0,4,8,12,15})observed[1][0].narrow(0,p*8,8).fill_(true);
  im::Input input{data,observed,torch::tensor(std::vector<int64_t>{2,0,1},torch::kInt64),torch::tensor(std::vector<double>{100.,200.},torch::kFloat64),c.sampling_interval};
  im::TemporalExecutionTrace trace;const auto out=model->encode(input,&trace);const auto named=model->named_parameters();
  const auto w=named["export_projection.weight"],b=named["export_projection.bias"],sw=named["pool_score.weight"],sb=named["pool_score.bias"],pp=named["pool_positions.weight"];
  for(int path=0;path<2;++path){
    const auto h=path?trace.contextual_temporal_hidden:trace.temporal_hidden;
    const auto scores=torch::matmul(torch::tanh(h),sw.transpose(0,1)).squeeze(-1)+sb+
        pp.index_select(0,trace.patch_positions.clamp_min(0).reshape({-1})).reshape(trace.patch_positions.sizes());
    const auto manual=reference(h,scores,trace.patch_positions,w,b,sw,torch::zeros({h.size(0),32},h.options()),16);
    near(path?trace.contextual_pool_weights:trace.pool_weights,manual.weights,"actual aggregate is the sum of per-export mixtures");
    near(path?trace.contextual_pool_local_weights:trace.pool_local_weights,manual.local,"actual anchored local trace");
    near(path?trace.contextual_pool_global_weights:trace.pool_global_weights,manual.global,"actual shared valid-history global trace");
    near(path?trace.contextual_pool_mixture_weights:trace.pool_mixture_weights,manual.mixture,"actual per-export half-mixture trace");
    near(path?trace.contextual_pool_bin_mean:trace.pool_bin_mean,manual.bin_mean,"actual two-sum bin mean trace");
    near(path?trace.contextual_pool_bin_variance:trace.pool_bin_variance,manual.bin_variance,"actual centered variance trace");
    near(path?trace.contextual_pool_bin_regularized_root:trace.pool_bin_regularized_root,manual.bin_regularized_root,"actual raw root trace, absent root1/32");
    near(path?trace.contextual_pool_bin_dispersion:trace.pool_bin_dispersion,manual.bin_dispersion,"actual zero-based dispersion trace");
    check(torch::equal((path?trace.contextual_pool_bin_valid:trace.pool_bin_valid).to(torch::kCPU),manual.bins),"actual original-bin support trace");
    const auto z=(path?out.z_contextual:out.z_local).reshape({B*C,32}).index_select(0,trace.patch_rows);
    near(z,manual.output,"both real paths mix hidden before one row-projection/bias application");
  }
  check(out.z_local[0][1].narrow(0,0,24).eq(0).all().item<bool>()&&out.z_contextual[0][1].narrow(0,0,24).eq(0).all().item<bool>(),
      "empty local bins stay zero despite available global history and nonzero bias");
  check(out.z_local[0][2].eq(0).all().item<bool>()&&out.z_contextual[0][2].eq(0).all().item<bool>(),"fully absent channel zero");
}
} // namespace

int main(int argc,char **argv){try{
  TORCH_CHECK(argc==3&&std::string(argv[1])=="--settings","--settings CONFIG required");
  check(torch::cuda::is_available(),"actual CUDA required");const auto settings=cp::read_settings(argv[2]);auto c=im::configurable_spectral_backbone_config(settings.level);
  check(c.history_length==128&&c.patch_length==8&&c.export_width==32&&c.encoder_width==64&&
      cp::temporal_pooling_partition(settings)==std::vector<int64_t>({4,16,32,4,8})&&cp::temporal_pooling_mix()==std::vector<double>({.5,.5})&&
      cp::temporal_moment_partition()==std::vector<int64_t>({4,4}),"declared exact profile, moment layout and fixed mix");
  torch::manual_seed(943001);im::Model model(c);const auto named=model->named_parameters();auto w=named["export_projection.weight"],b=named["export_projection.bias"],sw=named["pool_score.weight"],sb=named["pool_score.bias"],pp=named["pool_positions.weight"];
  {torch::NoGradGuard guard;w.copy_(torch::sin(torch::arange(w.numel(),w.options()).reshape_as(w)*.031)*.07);
    b.copy_(torch::arange(32,b.options())*.003+.04);sw.copy_(torch::cos(torch::arange(64,sw.options()).reshape_as(sw)*.021)*.04);
    sb.fill_(.17);pp.copy_(torch::arange(16,pp.options()).reshape_as(pp)*.011);}
  actual_model_paths(model,c);
  const auto positions=torch::tensor(std::vector<int64_t>{0,3,4,7,8,11,12,15,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,4,8,12,-1,-1,-1,-1},torch::kInt64).reshape({4,8}).to(c.device);
  const auto valid=positions.ge(0);auto h=torch::sin(torch::arange(4*8*64,w.options()).reshape({4,8,64})*.029)*.7;
  h=torch::where(valid.unsqueeze(-1),h,torch::full_like(h,NAN)).detach().set_requires_grad(true);
  const auto safe=torch::where(valid.unsqueeze(-1),h,torch::zeros_like(h));
  auto scores=torch::matmul(torch::tanh(safe),sw.transpose(0,1)).squeeze(-1)+sb+pp.index_select(0,positions.clamp_min(0).reshape({-1})).reshape({4,8});
  scores=torch::where(valid,scores,torch::full_like(scores,NAN));
  const auto up=torch::arange(4*32,w.options()).reshape({4,32})*.007+.13;
  const auto manual=reference(h,scores,positions,w,b,sw,up,16);
  const auto pooled=counted_pool(h,scores,valid,positions,w,b,c);
  check(pooled.projected.is_cuda()&&torch::isfinite(pooled.projected).all().item<bool>()&&torch::isfinite(pooled.mixture_weights).all().item<bool>(),"finite CUDA despite unsupported NaNs");
  near(pooled.projected,manual.output,"manual original-position half-mixture pooling parity");near(pooled.weights,manual.weights,"manual aggregate mixture parity");
  near(pooled.local_weights,manual.local,"manual local weights");near(pooled.global_weights,manual.global,"manual global weights");near(pooled.mixture_weights,manual.mixture,"manual per-export mixtures");
  near(pooled.bin_mean,manual.bin_mean,"manual two-sum means");near(pooled.bin_variance,manual.bin_variance,"manual centered variance");
  near(pooled.bin_regularized_root,manual.bin_regularized_root,"independent root values");near(pooled.bin_dispersion,manual.bin_dispersion,"independent dispersion values");
  check(torch::equal(pooled.bin_regularized_root,torch::sqrt(pooled.bin_variance+cp::kTemporalDispersionRegularizer))&&
      torch::equal(pooled.bin_dispersion,pooled.bin_regularized_root-cp::kTemporalDispersionRootOffset),"exact actual F32 add-sqrt-subtract trace");
  const auto absent=pooled.bin_valid.logical_not().unsqueeze(-1).expand_as(pooled.bin_variance);
  check(pooled.bin_variance.masked_select(absent).eq(0).all().item<bool>()&&
      pooled.bin_regularized_root.masked_select(absent).eq(cp::kTemporalDispersionRootOffset).all().item<bool>()&&
      pooled.bin_dispersion.masked_select(absent).eq(0).all().item<bool>(),"absent root1/32 exception and zero dispersion");
  check(torch::equal(pooled.bin_valid.to(torch::kCPU),manual.bins),"original positions determine local membership");
  check(pooled.projected[1].narrow(0,0,24).eq(0).all().item<bool>()&&pooled.projected[2].eq(0).all().item<bool>()&&pooled.weights[2].eq(0).all().item<bool>(),"empty-bin finite zero after bias including all-padding row");
  for(int64_t j=0;j<4;++j){const auto parent_mean=torch::matmul(pooled.bin_mean.select(1,j),w.narrow(0,j*8,4).transpose(0,1))+b.narrow(0,j*8,4);
    check(torch::equal(pooled.projected.narrow(1,j*8,4),torch::where(pooled.bin_valid.select(1,j).unsqueeze(1),parent_mean,torch::zeros_like(parent_mean))),
        "first4 unchanged parent mean projection expression is bitwise exact");}
  for(int64_t r=0;r<4;++r){check(std::abs(pooled.weights[r].sum().item<double>()-pooled.bin_valid[r].sum().item<int64_t>())<8e-6,"aggregate mass equals number of nonempty local bins");
    check(std::abs(pooled.global_weights[r].sum().item<double>()-double(valid[r].any().item<bool>()))<2e-6,"shared global mass1 or all-padding0");
    for(int64_t j=0;j<4;++j){const auto member=valid[r].logical_and(positions[r].ge(j*4)).logical_and(positions[r].lt((j+1)*4));const bool present=member.any().item<bool>();
      check(std::abs(pooled.local_weights[r].masked_select(member).sum().item<double>()-double(present))<2e-6,"anchored local bin mass1 or0");
      check(std::abs(pooled.mixture_weights[r][j].sum().item<double>()-double(present))<2e-6,"per-export mixture mass1 or exact empty0");}}
  check(pooled.mixture_weights[0][0].masked_select(positions[0].ge(4)).gt(0).all().item<bool>(),"nonempty first-bin mixture genuinely includes outside-bin global states");
  {torch::NoGradGuard guard;
    const auto common=counted_pool(h,scores+2.,valid,positions,w,b,c);
    near(common.projected,pooled.projected,"common score-shift invariance",kGaugeTolerance);near(common.mixture_weights,pooled.mixture_weights,"common bias remains softmax gauge",kGaugeTolerance);
    for(int direction=0;direction<3;++direction){auto shifts=torch::zeros_like(scores);
      for(int j=0;j<4;++j)shifts=torch::where(valid.logical_and(positions.ge(j*4)).logical_and(positions.lt((j+1)*4)),torch::full_like(scores,kSigns[direction][j]/4.),shifts);
      const auto shifted=counted_pool(h,scores+shifts,valid,positions,w,b,c);
      near(shifted.local_weights,pooled.local_weights,"bin-constant shifts still cancel inside local denominators",kGaugeTolerance);
      check(!torch::allclose(shifted.global_weights[0],pooled.global_weights[0],kGaugeTolerance,kGaugeTolerance)&&
          !torch::allclose(shifted.projected[0],pooled.projected[0],kGaugeTolerance,kGaugeTolerance),"three relative-bin Hadamard directions affect shared-global pooling");
    }
    const auto all_positions=torch::arange(16,positions.options()).reshape({1,16}),all_valid=all_positions.ge(0);
    const auto constant=torch::linspace(-.7,.9,64,w.options()).reshape({1,1,64}).expand({1,16,64}).clone();
    const auto constant_scores=torch::arange(16,w.options()).reshape({1,16})*.037;
    const auto constant_pool=counted_pool(constant,constant_scores,all_valid,all_positions,w,b,c);
    const double constant_bound=kConstantVarianceTolerance*(1.+constant.abs().max().item<double>()*constant.abs().max().item<double>());
    check(constant_pool.bin_variance.abs().max().item<double>()<=constant_bound,"present constant hidden variance uses declared F32 tolerance, never exact-zero claim");
    const auto sign=torch::remainder(torch::arange(16,positions.options()),2).to(c.dtype)*2.-1.;
    const auto spread=sign.reshape({1,16,1})*torch::linspace(.2,.8,64,w.options()).reshape({1,1,64});
    const auto zero_scores=torch::zeros({1,16},w.options());
    const auto spread1=counted_pool(spread,zero_scores,all_valid,all_positions,w,b,c);
    const auto spread2=counted_pool(2.*spread,zero_scores,all_valid,all_positions,w,b,c);
    near(spread1.bin_mean,spread2.bin_mean,"equal-mean different-spread synthetic pair");
    check(spread2.bin_variance.gt(spread1.bin_variance).all().item<bool>(),"centered variances distinguish equal means with different spread");
    for(int64_t j=0;j<4;++j){near(spread1.projected.narrow(1,j*8,4),spread2.projected.narrow(1,j*8,4),"first4 mean rows preserve equal means");
      check(!torch::allclose(spread1.projected.narrow(1,j*8+4,4),spread2.projected.narrow(1,j*8+4,4),kManualTolerance,kManualTolerance),"last4 regularized_stddev rows detect spread with same mean");}
  }
  // The complete graph stays differentiable at zero and at small/large finite
  // variance. Each case is one direct kernel call and one isolated grad call.
  const auto test_positions=torch::arange(16,positions.options()).reshape({1,16}),test_valid=test_positions.ge(0);
  const auto signs=(torch::remainder(test_positions,2).to(c.dtype)*2.-1.).unsqueeze(-1);
  auto test_weight=torch::zeros_like(w),test_bias=torch::zeros_like(b);
  for(int j=0;j<4;++j)test_weight.narrow(0,j*8+4,4).fill_(.25);
  for(const double amplitude:{0.,0.000244140625,1024.}){
    auto test_hidden=(signs*torch::linspace(.5,1.,64,w.options()).reshape({1,1,64})*amplitude).detach().set_requires_grad(true);
    auto test_scores=(torch::arange(16,w.options()).reshape({1,16})*.037).detach().set_requires_grad(true);
    const auto test=counted_pool(test_hidden,test_scores,test_valid,test_positions,test_weight,test_bias,c);
    check(torch::equal(test.bin_regularized_root,torch::sqrt(test.bin_variance+cp::kTemporalDispersionRegularizer))&&
        torch::equal(test.bin_dispersion,test.bin_regularized_root-cp::kTemporalDispersionRootOffset),"zero/tiny/large exact F32 transform");
    const auto grads=torch::autograd::grad({test.projected.sum()},{test_hidden,test_scores});
    const auto analytic=reference(test_hidden,test_scores,test_positions,test_weight,test_bias,torch::zeros_like(sw),torch::ones({1,32},w.options()),16,&test);
    near(grads[0],analytic.hidden_grad,"zero/tiny/large captured-F32 analytic hidden adjoints");
    near(grads[1],analytic.position_grad.reshape({1,16}),"zero/tiny/large captured-F32 analytic live score adjoints");
    check((.5/test.bin_regularized_root).le(16.).all().item<bool>(),"regularized transform derivative bound16 uses ACTUAL saved root");
    check(torch::isfinite(grads[0]).all().item<bool>()&&torch::isfinite(grads[1]).all().item<bool>(),"zero/tiny/large finite live dispersion derivatives");
    if(amplitude==0.)check(test.bin_variance.eq(0).all().item<bool>()&&test.bin_regularized_root.eq(.03125).all().item<bool>()&&
        test.bin_dispersion.eq(0).all().item<bool>()&&grads[0].eq(0).all().item<bool>()&&grads[1].eq(0).all().item<bool>(),"exact zero variance/dispersion and finite zero hidden/score derivatives");
    else check(test.bin_variance.gt(0).all().item<bool>()&&test.bin_dispersion.gt(0).all().item<bool>()&&grads[0].ne(0).any().item<bool>(),"tiny/large variance preserves nonzero dispersion and hidden route");
  }
  {torch::NoGradGuard guard;
    const auto singleton_positions=torch::tensor(std::vector<int64_t>{15,-1},positions.options()).reshape({1,2});
    const auto singleton_hidden=torch::cat({torch::full({1,1,64},1e20,w.options()),torch::full({1,1,64},NAN,w.options())},1);
    const auto singleton_scores=torch::tensor(std::vector<float>{0.f,NAN},w.options()).reshape({1,2});
    const auto singleton=counted_pool(singleton_hidden,singleton_scores,
        singleton_positions.ge(0),singleton_positions,w,b,c);
    check(torch::isfinite(singleton.projected).all().item<bool>()&&singleton.bin_variance.eq(0).all().item<bool>()&&
        singleton.bin_regularized_root.eq(.03125).all().item<bool>()&&singleton.bin_dispersion.eq(0).all().item<bool>()&&
        singleton.projected.narrow(1,0,24).eq(0).all().item<bool>(),"finite huge padded singleton: absent local AND unsupported token operands sanitized before subtract/square, no zero-times-infinity");
    auto invalid_hidden=h.detach().clone();invalid_hidden[0][0][0]=NAN;
    rejects([&]{(void)counted_pool(invalid_hidden,scores.detach(),valid,positions,w,b,c);},"supported NaN hidden refused");
    auto invalid_scores=scores.detach().clone();invalid_scores[0][0]=INFINITY;
    rejects([&]{(void)counted_pool(h.detach(),invalid_scores,valid,positions,w,b,c);},"supported infinite score refused");
    const auto overflow_hidden=signs.expand({1,16,64})*1e20;
    rejects([&]{(void)counted_pool(overflow_hidden,torch::zeros({1,16},w.options()),test_valid,test_positions,w,b,c);},"finite supported hidden with overflowed raw variance refused before export");
  }
  const auto captured_manual=reference(h,scores,positions,w,b,sw,up,16,&pooled);
  const auto safe_captured=torch::where(valid.unsqueeze(-1),h.detach(),torch::zeros_like(h));
  const auto residual=(pooled.mixture_weights.detach().unsqueeze(-1)*(safe_captured.unsqueeze(1)-pooled.bin_mean.detach().unsqueeze(2))).sum(2);
  check(residual.ne(0).any().item<bool>(),"actual captured F32 centered residual is observed nonzero, retained in adjoint");
  auto dispersion_up=torch::zeros_like(up);for(int j=0;j<4;++j)dispersion_up.narrow(1,j*8+4,4).copy_(up.narrow(1,j*8+4,4));
  const auto isolated_score=torch::autograd::grad({(pooled.projected*dispersion_up).sum()},{scores},{},true,false)[0];
  check(torch::isfinite(isolated_score).all().item<bool>()&&isolated_score.masked_select(valid).ne(0).any().item<bool>(),"dispersion-only loss has live attention-score gradient");
  torch::optim::AdamW optimizer(model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));optimizer.zero_grad();
  (pooled.projected*up).sum().backward();near(h.grad(),captured_manual.hidden_grad,"captured-F32 full attention hidden-plus-score/root adjoint parity");
  near(w.grad(),captured_manual.weight_grad,"manual existing four8-row export weight adjoints");near(b.grad(),captured_manual.bias_grad,"manual once-applied bias adjoints");
  near(pp.grad(),captured_manual.position_grad,"manual local-plus-global position adjoints");near(sw.grad(),captured_manual.score_weight_grad,"manual shared score-weight adjoints");
  check(sb.grad().defined()&&torch::isfinite(sb.grad()).all().item<bool>(),"finite observed common-bias gauge gradient, no nonzero requirement");
  finite_differences(h,positions,w,b,sw,sb,pp,up,captured_manual,c);
  std::vector<torch::Tensor> before{w.detach().clone(),b.detach().clone(),pp.detach().clone()};const auto gauge_before=sb.detach().item<double>();
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.);const auto gauge_gradient=sb.grad().item<double>();
  for(int64_t j=0;j<4;++j){for(int64_t moment=0;moment<2;++moment)
      check(w.grad().narrow(0,j*8+moment*4,4).ne(0).any().item<bool>()&&b.grad().narrow(0,j*8+moment*4,4).ne(0).any().item<bool>(),"each mean4 and regularized_stddev4 export subgroup has data-gradient events");
    const auto g=pp.grad().narrow(0,j*4,4);check((g-g.mean()).ne(0).any().item<bool>(),"four within-bin-centered position gradient contrasts");}
  const auto contrasts=between_contrasts(pp.grad());check(torch::isfinite(contrasts).all().item<bool>()&&contrasts.ne(0).all().item<bool>(),"all three orthonormal between-bin position data-gradient coefficients nonzero");
  const auto bin_means=pp.grad().to(torch::kFloat64).reshape({4,4}).mean(1);
  near(contrasts.square().sum(),4.*(bin_means-bin_means.mean()).square().sum(),"Hadamard coefficients preserve full-position between-bin norm");
  optimizer.step();
  for(int64_t j=0;j<4;++j){for(int64_t moment=0;moment<2;++moment)
      check((w.detach().to(torch::kFloat64)-before[0].to(torch::kFloat64)).narrow(0,j*8+moment*4,4).ne(0).any().item<bool>()&&
          (b.detach().to(torch::kFloat64)-before[1].to(torch::kFloat64)).narrow(0,j*8+moment*4,4).ne(0).any().item<bool>(),"each mean4 and regularized_stddev4 subgroup moves in actual AdamW step");
    const auto change=pp.detach().to(torch::kFloat64)-before[2].to(torch::kFloat64);const auto bin=change.narrow(0,j*4,4);
    check((bin-bin.mean()).ne(0).any().item<bool>(),"four within-bin-centered position movements, F64 before subtraction");}
  check(between_contrasts(pp.detach().to(torch::kFloat64)-before[2].to(torch::kFloat64)).ne(0).all().item<bool>(),"all three between-bin position contrast movements");
  auto wrong=c;wrong.history_length=112;rejects([&]{im::validate_temporal_pooling_config(wrong);},"nondivisible original patch count refused");wrong=c;wrong.export_width=30;rejects([&]{im::validate_temporal_pooling_config(wrong);},"nondivisible export width refused");
  wrong=c;wrong.export_width=64;rejects([&]{im::validate_temporal_pooling_config(wrong);},"standalone divisible width cannot contradict fixed4/4 moment layout");
  check(direct_pool_kernel_calls==51,"source-derived51 calls:44 parent+3 zero/tiny/large+1 huge singleton+3 numerical refusals");
  std::cout<<std::setprecision(17)<<"PASS CPB-v13 "<<cp::kTemporalPoolingPolicy<<": manual full-attention mean/raw-variance/regularized-stddev parity; independent full-expression F64 finite differences; actual CUDA F32 kernel finite differences; four mean4 and regularized_stddev4 projection gradient and movement; three between-bin Hadamard contrast gradient and movement; present constant variance dtype tolerance; equal mean different spread; empty-bin finite zero; relative-bin global sensitivity; common softmax gauge invariance; synthetic optimizer steps=1; scope=owned-backbone-pooling; backbone_encode_calls=1; public_encoder_calls=0; direct_pool_kernel_calls="<<direct_pool_kernel_calls<<"; head_fits=0; manual_F32_abs_rel_tolerance="<<kManualTolerance<<"; F64_FD_step="<<kF64FiniteDifferenceStep<<"; F64_FD_abs_rel_tolerance="<<kF64FiniteDifferenceTolerance<<"; actual_F32_FD_step="<<kF32FiniteDifferenceStep<<"; actual_F32_FD_abs_rel_tolerance="<<kF32FiniteDifferenceTolerance<<"; present_constant_F32_variance_bound=1e-10*(1+max_abs_hidden_squared); gauge_abs_rel_tolerance=2e-6; F64_finite_difference_coordinates=18; F32_finite_difference_coordinates=18; isolated_autograd_grad_calls=4; mean4_regularized_stddev4_export_layout=true; full_attention_dependent_regularized_moment_adjoints=true; live_dispersion_score_gradient=true; raw_variance_root_dispersion_trace=true; zero_tiny_large_analytic_dispersion_adjoints=true; supported_nonfinite_and_variance_overflow_refused=true; absent_local_huge_singleton_sanitized=true; captured_F32_residual_correction=true; raw_root_absent_offset_exception=true; temporal_dispersion_regularizer=0.0009765625; transform_derivative_upper_bound=16; gauge_bias_data_gradient_observed="<<gauge_gradient<<"; gauge_bias_movement_observed="<<(sb.detach().item<double>()-gauge_before)<<"; gauge_bias_gradient_required_nonzero=false; gauge_bias_decay_movement_is_data_learning=false; checks="<<checks<<"\n";
  return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
