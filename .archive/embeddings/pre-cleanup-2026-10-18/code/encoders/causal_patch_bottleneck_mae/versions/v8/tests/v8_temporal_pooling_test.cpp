// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Prospective CUDA correctness fixture. No quality data, fits or old artifacts.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v8.h"
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v8;
namespace im=cp::implementation;
int64_t checks=0;
constexpr double kManualTolerance=2e-5,kGaugeTolerance=2e-6;
void check(bool v,const std::string &why){++checks;TORCH_CHECK(v,"[CPB-v8 four-bin proof] ",why);}
void near(const torch::Tensor &a,const torch::Tensor &b,const std::string &why,double tol=kManualTolerance){
  check(torch::allclose(a.detach().to(torch::kCPU,torch::kFloat64),b.detach().to(torch::kCPU,torch::kFloat64),tol,tol),why);
}
template<class F>void rejects(F f,const std::string &why){bool yes=false;try{f();}catch(const std::exception&){yes=true;}check(yes,why);}
torch::Tensor f64(const torch::Tensor &t){return t.detach().to(torch::kCPU,torch::kFloat64).contiguous();}
struct Reference{torch::Tensor output,weights,bins,hidden_grad,weight_grad,bias_grad,position_grad,score_weight_grad;};

// Independent scalar implementation: no producer pooling helper/autograd.
Reference reference(const torch::Tensor &h0,const torch::Tensor &s0,const torch::Tensor &p0,
    const torch::Tensor &w0,const torch::Tensor &b0,const torch::Tensor &scorew0,
    const torch::Tensor &up0,int64_t K){
  const auto h=f64(h0),s=f64(s0),w=f64(w0),b=f64(b0),scorew=f64(scorew0),up=f64(up0);
  const auto p=p0.detach().to(torch::kCPU).contiguous();const auto R=h.size(0),T=h.size(1),W=h.size(2),D=w.size(0);
  Reference out{torch::zeros({R,D},torch::kFloat64),torch::zeros({R,T},torch::kFloat64),torch::zeros({R,4},torch::kBool),
      torch::zeros_like(h),torch::zeros_like(w),torch::zeros_like(b),torch::zeros({K,1},torch::kFloat64),torch::zeros_like(scorew)};
  const auto hv=h.accessor<double,3>();const auto sv=s.accessor<double,2>();const auto wv=w.accessor<double,2>();const auto bv=b.accessor<double,1>();
  const auto pv=p.accessor<int64_t,2>();const auto uv=up.accessor<double,2>(),sw=scorew.accessor<double,2>();
  auto ov=out.output.accessor<double,2>(),aw=out.weights.accessor<double,2>();auto bins=out.bins.accessor<bool,2>();
  auto gh=out.hidden_grad.accessor<double,3>();auto gw=out.weight_grad.accessor<double,2>();auto gb=out.bias_grad.accessor<double,1>();
  auto gp=out.position_grad.accessor<double,2>(),gsw=out.score_weight_grad.accessor<double,2>();
  for(int64_t r=0;r<R;++r)for(int64_t j=0;j<4;++j){
    std::vector<int64_t> keys;for(int64_t t=0;t<T;++t)if(pv[r][t]>=j*(K/4)&&pv[r][t]<(j+1)*(K/4))keys.push_back(t);
    if(keys.empty())continue;bins[r][j]=true;double maximum=-INFINITY,denominator=0.;
    for(auto t:keys)maximum=std::max(maximum,sv[r][t]);for(auto t:keys)denominator+=std::exp(sv[r][t]-maximum);
    std::vector<double> pooled(W,0.),q(W,0.);
    for(auto t:keys){aw[r][t]=std::exp(sv[r][t]-maximum)/denominator;for(int64_t k=0;k<W;++k)pooled[k]+=aw[r][t]*hv[r][t][k];}
    for(int64_t d=j*(D/4);d<(j+1)*(D/4);++d){ov[r][d]=bv[d];gb[d]+=uv[r][d];
      for(int64_t k=0;k<W;++k){ov[r][d]+=pooled[k]*wv[d][k];gw[d][k]+=uv[r][d]*pooled[k];q[k]+=uv[r][d]*wv[d][k];}}
    for(auto t:keys){double ds=0.;for(int64_t k=0;k<W;++k)ds+=aw[r][t]*q[k]*(hv[r][t][k]-pooled[k]);
      gp[pv[r][t]][0]+=ds;for(int64_t k=0;k<W;++k){const double th=std::tanh(hv[r][t][k]);
        gh[r][t][k]=aw[r][t]*q[k]+ds*sw[0][k]*(1.-th*th);gsw[0][k]+=ds*th;}}
  }
  return out;
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
    auto scores=torch::matmul(torch::tanh(h),sw.transpose(0,1)).squeeze(-1)+sb+
        pp.index_select(0,trace.patch_positions.clamp_min(0).reshape({-1})).reshape(trace.patch_positions.sizes());
    const auto manual=reference(h,scores,trace.patch_positions,w,b,sw,torch::zeros({h.size(0),32},h.options()),16);
    const auto weights=path?trace.contextual_pool_weights:trace.pool_weights;
    const auto bins=path?trace.contextual_pool_bin_valid:trace.pool_bin_valid;
    near(weights,manual.weights,"actual ordinary/contextual trace matches independent per-original-bin weights");
    check(torch::equal(bins.to(torch::kCPU),manual.bins),"actual ordinary/contextual bin support trace");
    const auto z=(path?out.z_contextual:out.z_local).reshape({B*C,32}).index_select(0,trace.patch_rows);
    near(z,manual.output,"actual ordinary/contextual encode uses four disjoint projection row slices");
  }
  check(out.z_local[0][1].narrow(0,0,24).eq(0).all().item<bool>()&&out.z_contextual[0][1].narrow(0,0,24).eq(0).all().item<bool>(),
      "single recent original bin does not migrate to packed bin0; empty slices zero after nonzero bias");
  check(out.z_local[0][2].eq(0).all().item<bool>()&&out.z_contextual[0][2].eq(0).all().item<bool>(),"fully absent channel zero");
}
} // namespace

int main(int argc,char **argv){try{
  TORCH_CHECK(argc==3&&std::string(argv[1])=="--settings","--settings CONFIG required");
  check(torch::cuda::is_available(),"actual CUDA required");const auto settings=cp::read_settings(argv[2]);auto c=im::configurable_spectral_backbone_config(settings.level);
  check(c.history_length==128&&c.patch_length==8&&c.export_width==32&&c.encoder_width==64&&cp::temporal_pooling_partition(settings)==std::vector<int64_t>({4,16,32,4,8}),"declared exact partition and profile");
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
  const auto up=(torch::arange(4*32,w.options()).reshape({4,32})*.007+.13);
  const auto manual=reference(h,scores,positions,w,b,sw,up,16);
  const auto pooled=im::pool_original_position_bins(h,scores,valid,positions,w,b,c);
  check(pooled.projected.is_cuda()&&torch::isfinite(pooled.projected).all().item<bool>()&&torch::isfinite(pooled.weights).all().item<bool>(),"actual finite CUDA pooling despite unsupported NaN");
  near(pooled.projected,manual.output,"manual original-position pooling parity");near(pooled.weights,manual.weights,"manual original-position weight parity");
  check(torch::equal(pooled.bin_valid.to(torch::kCPU),manual.bins),"bin membership is original time, not packed rank");
  check(pooled.projected[1].narrow(0,0,24).eq(0).all().item<bool>()&&pooled.projected[2].eq(0).all().item<bool>()&&pooled.weights[2].eq(0).all().item<bool>(),"empty-bin finite zero after nonzero projection bias");
  for(int64_t r=0;r<4;++r){check(std::abs(pooled.weights[r].sum().item<double>()-pooled.bin_valid[r].sum().item<int64_t>())<8e-6,"true flat mass equals number nonempty bins");
    for(int64_t j=0;j<4;++j){const auto member=valid[r].logical_and(positions[r].ge(j*4)).logical_and(positions[r].lt((j+1)*4));
      check(std::abs(pooled.weights[r].masked_select(member).sum().item<double>()-double(member.any().item<bool>()))<2e-6,"per-original-bin mass1 or exact empty0");}}
  {torch::NoGradGuard guard;auto shifts=torch::zeros_like(scores);for(int64_t j=0;j<4;++j)shifts=torch::where(valid.logical_and(positions.ge(j*4)).logical_and(positions.lt((j+1)*4)),torch::full_like(scores,double(j)-1.5),shifts);
    const auto shifted=im::pool_original_position_bins(h,scores+shifts+2.,valid,positions,w,b,c);
    near(shifted.projected,pooled.projected,"softmax common-bias and four-bin offset gauge invariance",kGaugeTolerance);
    near(shifted.weights,pooled.weights,"within-bin constant score offsets preserve probabilities",kGaugeTolerance);}
  torch::optim::AdamW optimizer(model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));optimizer.zero_grad();
  (pooled.projected*up).sum().backward();near(h.grad(),manual.hidden_grad,"manual hidden-plus-score native adjoint parity");
  near(w.grad(),manual.weight_grad,"manual four8-row export weight adjoint parity");near(b.grad(),manual.bias_grad,"manual four8-row export bias adjoint parity");
  near(pp.grad(),manual.position_grad,"manual original-position score adjoint parity");near(sw.grad(),manual.score_weight_grad,"manual shared score weight adjoint parity");
  // One scalar common score bias is a gauge. Its observed roundoff gradient is
  // recorded but is NEVER required nonzero or counted as meaningful learning.
  check(sb.grad().defined()&&torch::isfinite(sb.grad()).all().item<bool>(),"finite observed gauge-bias gradient, no nonzero learning claim");
  std::vector<torch::Tensor> before{w.detach().clone(),b.detach().clone(),pp.detach().clone()};const auto gauge_bias_before=sb.detach().item<double>();
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.);
  const auto gauge_bias_data_gradient=sb.grad().item<double>();
  for(int64_t j=0;j<4;++j){
    check(w.grad().narrow(0,j*8,8).ne(0).any().item<bool>()&&b.grad().narrow(0,j*8,8).ne(0).any().item<bool>(),"four8-row projection nonzero data-gradient events");
    const auto g=pp.grad().narrow(0,j*4,4);check((g-g.mean()).ne(0).any().item<bool>(),"four within-bin-centered position data-gradient contrasts");}
  optimizer.step();
  for(int64_t j=0;j<4;++j){check((w.detach()-before[0]).narrow(0,j*8,8).ne(0).any().item<bool>()&&(b.detach()-before[1]).narrow(0,j*8,8).ne(0).any().item<bool>(),"four8-row export weight/bias actual AdamW movement");
    const auto change=(pp.detach()-before[2]).narrow(0,j*4,4);check((change-change.mean()).ne(0).any().item<bool>(),"four within-bin-centered position contrast movements");}
  auto wrong=c;wrong.history_length=112;rejects([&]{im::validate_temporal_pooling_config(wrong);},"nondivisible original patch count refused");wrong=c;wrong.export_width=30;rejects([&]{im::validate_temporal_pooling_config(wrong);},"nondivisible export width refused");
  std::cout<<std::setprecision(17)<<"PASS CPB-v8 original-position-four-bin-disjoint-export-rows-v1: manual original-position pooling parity; four 8-row projection gradient and movement; empty-bin finite zero; softmax gauge invariance; synthetic optimizer steps=1; scope=owned-backbone-pooling; backbone_encode_calls=1; public_encoder_calls=0; direct_pool_kernel_calls=2; head_fits=0; manual_F32_abs_rel_tolerance=2e-5; gauge_abs_rel_tolerance=2e-6; gauge_bias_data_gradient_observed="<<gauge_bias_data_gradient<<"; gauge_bias_movement_observed="<<(sb.detach().item<double>()-gauge_bias_before)<<"; gauge_bias_gradient_required_nonzero=false; gauge_bias_decay_movement_is_data_learning=false; checks="<<checks<<"\n";
  return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
