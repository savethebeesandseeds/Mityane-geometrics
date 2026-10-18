// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Prospective synthetic CUDA fixture only. No quality data or old artifacts.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v10.h"
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v10;
namespace im=cp::implementation;
int64_t checks=0;
constexpr double kManualTolerance=2e-5,kGaugeTolerance=2e-6;
constexpr int kSigns[3][4]={{1,1,-1,-1},{1,-1,1,-1},{1,-1,-1,1}};
void check(bool v,const std::string &why){++checks;TORCH_CHECK(v,"[CPB-v10 hybrid-pool proof] ",why);}
void near(const torch::Tensor &a,const torch::Tensor &b,const std::string &why,double tol=kManualTolerance){
  check(torch::allclose(a.detach().to(torch::kCPU,torch::kFloat64),b.detach().to(torch::kCPU,torch::kFloat64),tol,tol),why);
}
template<class F>void rejects(F f,const std::string &why){bool yes=false;try{f();}catch(const std::exception&){yes=true;}check(yes,why);}
torch::Tensor f64(const torch::Tensor &t){return t.detach().to(torch::kCPU,torch::kFloat64).contiguous();}
struct Reference{
  torch::Tensor output,weights,bins,local,global,mixture;
  torch::Tensor hidden_grad,weight_grad,bias_grad,position_grad,score_weight_grad;
};

// Independent scalar reference and chain rule: no producer helper or autograd.
Reference reference(const torch::Tensor &h0,const torch::Tensor &s0,const torch::Tensor &p0,
    const torch::Tensor &w0,const torch::Tensor &b0,const torch::Tensor &scorew0,
    const torch::Tensor &up0,int64_t K){
  const auto h=f64(h0),s=f64(s0),w=f64(w0),b=f64(b0),scorew=f64(scorew0),up=f64(up0);
  const auto p=p0.detach().to(torch::kCPU).contiguous();const auto R=h.size(0),T=h.size(1),W=h.size(2),D=w.size(0);
  Reference out{torch::zeros({R,D},torch::kFloat64),torch::zeros({R,T},torch::kFloat64),torch::zeros({R,4},torch::kBool),
      torch::zeros({R,T},torch::kFloat64),torch::zeros({R,T},torch::kFloat64),torch::zeros({R,4,T},torch::kFloat64),
      torch::zeros_like(h),torch::zeros_like(w),torch::zeros_like(b),torch::zeros({K,1},torch::kFloat64),torch::zeros_like(scorew)};
  const auto hv=h.accessor<double,3>();const auto sv=s.accessor<double,2>(),wv=w.accessor<double,2>();
  const auto bv=b.accessor<double,1>();const auto pv=p.accessor<int64_t,2>();
  const auto uv=up.accessor<double,2>(),sw=scorew.accessor<double,2>();
  auto ov=out.output.accessor<double,2>(),aw=out.weights.accessor<double,2>(),lw=out.local.accessor<double,2>(),gw0=out.global.accessor<double,2>();
  auto mw=out.mixture.accessor<double,3>();auto bins=out.bins.accessor<bool,2>();
  auto gh=out.hidden_grad.accessor<double,3>();auto gw=out.weight_grad.accessor<double,2>();
  auto gb=out.bias_grad.accessor<double,1>();auto gp=out.position_grad.accessor<double,2>(),gsw=out.score_weight_grad.accessor<double,2>();
  for(int64_t r=0;r<R;++r){
    std::vector<int64_t> all;for(int64_t t=0;t<T;++t)if(pv[r][t]>=0)all.push_back(t);
    if(all.empty())continue;
    double maximum=-INFINITY,denominator=0.;for(auto t:all)maximum=std::max(maximum,sv[r][t]);
    for(auto t:all)denominator+=std::exp(sv[r][t]-maximum);
    std::vector<double> global(T,0.),global_hidden(W,0.);
    for(auto t:all){global[t]=std::exp(sv[r][t]-maximum)/denominator;gw0[r][t]=global[t];
      for(int64_t k=0;k<W;++k)global_hidden[k]+=global[t]*hv[r][t][k];}
    for(int64_t j=0;j<4;++j){
      std::vector<int64_t> keys;for(auto t:all)if(pv[r][t]>=j*(K/4)&&pv[r][t]<(j+1)*(K/4))keys.push_back(t);
      if(keys.empty())continue;bins[r][j]=true;maximum=-INFINITY;denominator=0.;
      for(auto t:keys)maximum=std::max(maximum,sv[r][t]);for(auto t:keys)denominator+=std::exp(sv[r][t]-maximum);
      std::vector<double> local(T,0.),local_hidden(W,0.),pooled(W,0.),q(W,0.);
      for(auto t:keys){local[t]=std::exp(sv[r][t]-maximum)/denominator;lw[r][t]=local[t];
        for(int64_t k=0;k<W;++k)local_hidden[k]+=local[t]*hv[r][t][k];}
      for(int64_t k=0;k<W;++k)pooled[k]=.5*local_hidden[k]+.5*global_hidden[k];
      for(int64_t d=j*(D/4);d<(j+1)*(D/4);++d){ov[r][d]=bv[d];gb[d]+=uv[r][d];
        for(int64_t k=0;k<W;++k){ov[r][d]+=pooled[k]*wv[d][k];gw[d][k]+=uv[r][d]*pooled[k];q[k]+=uv[r][d]*wv[d][k];}}
      for(auto t:all){const double mixture=.5*local[t]+.5*global[t];mw[r][j][t]=mixture;aw[r][t]+=mixture;
        double ds=0.;for(int64_t k=0;k<W;++k)
          ds+=.5*q[k]*(local[t]*(hv[r][t][k]-local_hidden[k])+global[t]*(hv[r][t][k]-global_hidden[k]));
        gp[pv[r][t]][0]+=ds;
        for(int64_t k=0;k<W;++k){const double th=std::tanh(hv[r][t][k]);
          gh[r][t][k]+=mixture*q[k]+ds*sw[0][k]*(1.-th*th);gsw[0][k]+=ds*th;}
      }
    }
  }
  return out;
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
      cp::temporal_pooling_partition(settings)==std::vector<int64_t>({4,16,32,4,8})&&cp::temporal_pooling_mix()==std::vector<double>({.5,.5}),"declared exact profile and fixed mix");
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
  const auto pooled=im::pool_original_position_bins(h,scores,valid,positions,w,b,c); // direct call1
  check(pooled.projected.is_cuda()&&torch::isfinite(pooled.projected).all().item<bool>()&&torch::isfinite(pooled.mixture_weights).all().item<bool>(),"finite CUDA despite unsupported NaNs");
  near(pooled.projected,manual.output,"manual original-position half-mixture pooling parity");near(pooled.weights,manual.weights,"manual aggregate mixture parity");
  near(pooled.local_weights,manual.local,"manual local weights");near(pooled.global_weights,manual.global,"manual global weights");near(pooled.mixture_weights,manual.mixture,"manual per-export mixtures");
  check(torch::equal(pooled.bin_valid.to(torch::kCPU),manual.bins),"original positions determine local membership");
  check(pooled.projected[1].narrow(0,0,24).eq(0).all().item<bool>()&&pooled.projected[2].eq(0).all().item<bool>()&&pooled.weights[2].eq(0).all().item<bool>(),"empty-bin finite zero after bias including all-padding row");
  for(int64_t r=0;r<4;++r){check(std::abs(pooled.weights[r].sum().item<double>()-pooled.bin_valid[r].sum().item<int64_t>())<8e-6,"aggregate mass equals number of nonempty local bins");
    check(std::abs(pooled.global_weights[r].sum().item<double>()-double(valid[r].any().item<bool>()))<2e-6,"shared global mass1 or all-padding0");
    for(int64_t j=0;j<4;++j){const auto member=valid[r].logical_and(positions[r].ge(j*4)).logical_and(positions[r].lt((j+1)*4));const bool present=member.any().item<bool>();
      check(std::abs(pooled.local_weights[r].masked_select(member).sum().item<double>()-double(present))<2e-6,"anchored local bin mass1 or0");
      check(std::abs(pooled.mixture_weights[r][j].sum().item<double>()-double(present))<2e-6,"per-export mixture mass1 or exact empty0");}}
  check(pooled.mixture_weights[0][0].masked_select(positions[0].ge(4)).gt(0).all().item<bool>(),"nonempty first-bin mixture genuinely includes outside-bin global states");
  {torch::NoGradGuard guard;
    const auto common=im::pool_original_position_bins(h,scores+2.,valid,positions,w,b,c); // direct call2
    near(common.projected,pooled.projected,"common score-shift invariance",kGaugeTolerance);near(common.mixture_weights,pooled.mixture_weights,"common bias remains softmax gauge",kGaugeTolerance);
    for(int direction=0;direction<3;++direction){auto shifts=torch::zeros_like(scores);
      for(int j=0;j<4;++j)shifts=torch::where(valid.logical_and(positions.ge(j*4)).logical_and(positions.lt((j+1)*4)),torch::full_like(scores,kSigns[direction][j]/4.),shifts);
      const auto shifted=im::pool_original_position_bins(h,scores+shifts,valid,positions,w,b,c); // direct calls3..5
      near(shifted.local_weights,pooled.local_weights,"bin-constant shifts still cancel inside local denominators",kGaugeTolerance);
      check(!torch::allclose(shifted.global_weights[0],pooled.global_weights[0],kGaugeTolerance,kGaugeTolerance)&&
          !torch::allclose(shifted.projected[0],pooled.projected[0],kGaugeTolerance,kGaugeTolerance),"three relative-bin Hadamard directions affect shared-global pooling");
    }
  }
  torch::optim::AdamW optimizer(model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));optimizer.zero_grad();
  (pooled.projected*up).sum().backward();near(h.grad(),manual.hidden_grad,"manual half-mixture hidden-plus-score adjoint parity");
  near(w.grad(),manual.weight_grad,"manual existing four8-row export weight adjoints");near(b.grad(),manual.bias_grad,"manual once-applied bias adjoints");
  near(pp.grad(),manual.position_grad,"manual local-plus-global position adjoints");near(sw.grad(),manual.score_weight_grad,"manual shared score-weight adjoints");
  check(sb.grad().defined()&&torch::isfinite(sb.grad()).all().item<bool>(),"finite observed common-bias gauge gradient, no nonzero requirement");
  std::vector<torch::Tensor> before{w.detach().clone(),b.detach().clone(),pp.detach().clone()};const auto gauge_before=sb.detach().item<double>();
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.);const auto gauge_gradient=sb.grad().item<double>();
  for(int64_t j=0;j<4;++j){check(w.grad().narrow(0,j*8,8).ne(0).any().item<bool>()&&b.grad().narrow(0,j*8,8).ne(0).any().item<bool>(),"four8-row export data-gradient events");
    const auto g=pp.grad().narrow(0,j*4,4);check((g-g.mean()).ne(0).any().item<bool>(),"four within-bin-centered position gradient contrasts");}
  const auto contrasts=between_contrasts(pp.grad());check(torch::isfinite(contrasts).all().item<bool>()&&contrasts.ne(0).all().item<bool>(),"all three orthonormal between-bin position data-gradient coefficients nonzero");
  const auto bin_means=pp.grad().to(torch::kFloat64).reshape({4,4}).mean(1);
  near(contrasts.square().sum(),4.*(bin_means-bin_means.mean()).square().sum(),"Hadamard coefficients preserve full-position between-bin norm");
  optimizer.step();
  for(int64_t j=0;j<4;++j){check((w.detach()-before[0]).narrow(0,j*8,8).ne(0).any().item<bool>()&&(b.detach()-before[1]).narrow(0,j*8,8).ne(0).any().item<bool>(),"four8-row export actual AdamW movements");
    const auto change=pp.detach().to(torch::kFloat64)-before[2].to(torch::kFloat64);const auto bin=change.narrow(0,j*4,4);
    check((bin-bin.mean()).ne(0).any().item<bool>(),"four within-bin-centered position movements, F64 before subtraction");}
  check(between_contrasts(pp.detach().to(torch::kFloat64)-before[2].to(torch::kFloat64)).ne(0).all().item<bool>(),"all three between-bin position contrast movements");
  auto wrong=c;wrong.history_length=112;rejects([&]{im::validate_temporal_pooling_config(wrong);},"nondivisible original patch count refused");wrong=c;wrong.export_width=30;rejects([&]{im::validate_temporal_pooling_config(wrong);},"nondivisible export width refused");
  std::cout<<std::setprecision(17)<<"PASS CPB-v10 original-position-four-bin-shared-global-half-mixture-v1: manual original-position half-mixture pooling parity; four 8-row projection gradient and movement; three between-bin Hadamard contrast gradient and movement; empty-bin finite zero; relative-bin global sensitivity; common softmax gauge invariance; synthetic optimizer steps=1; scope=owned-backbone-pooling; backbone_encode_calls=1; public_encoder_calls=0; direct_pool_kernel_calls=5; head_fits=0; manual_F32_abs_rel_tolerance=2e-5; gauge_abs_rel_tolerance=2e-6; gauge_bias_data_gradient_observed="<<gauge_gradient<<"; gauge_bias_movement_observed="<<(sb.detach().item<double>()-gauge_before)<<"; gauge_bias_gradient_required_nonzero=false; gauge_bias_decay_movement_is_data_learning=false; checks="<<checks<<"\n";
  return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
