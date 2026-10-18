// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Artificial CUDA fixture only; no encoder forward, old artifact, or quality fit.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v6.h"
#include <torch/cuda.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v6;
namespace im=cp::implementation;
int64_t checks=0,decoder_calls=0;
void check(bool x,const char *why){++checks;TORCH_CHECK(x,"[CPB-v6 objective fixture] ",why);}
bool has_nonzero_gradient(const torch::Tensor &x){return x.grad().defined()&&x.grad().is_cuda()&&torch::isfinite(x.grad()).all().item<bool>()&&x.grad().abs().sum().item<double>()>0.;}
bool has_zero_gradient(const torch::Tensor &x){return !x.grad().defined()||(torch::isfinite(x.grad()).all().item<bool>()&&x.grad().eq(0).all().item<bool>());}
void clear(im::LocationScaleDecoderModel &m){for(auto &p:m->parameters())p.mutable_grad()=torch::Tensor();}
struct Stats{cp::FrozenScaler scaler;cp::BankStandardizer bank;};
Stats statistics(const cp::Config &c){
  auto x=torch::zeros({8,3,128,9},torch::kFloat64);auto a=x.accessor<double,4>();
  for(int64_t b=0;b<8;++b)for(int64_t j=0;j<3;++j)for(int64_t h=0;h<128;++h)for(int64_t f=0;f<9;++f)
    a[b][j][h][f]=10.+2*j+.2*f+.01*h+.07*b+std::sin(.13*h+.19*f+.31*j+.17*b);
  cp::RawHistory raw{x,torch::ones_like(x,torch::kBool),torch::tensor({0,1,2},torch::kInt64),torch::arange(128,torch::kFloat64)*60000.,60000.};
  auto retained=cp::boundary::retain(raw,torch::full({8},127.*60000.,torch::kFloat64),c);
  const im::Input rounded{retained.raw.data.to(torch::kFloat32),retained.raw.observed,retained.raw.channel_ids,retained.raw.endpoints,retained.raw.sampling_interval};
  auto s=im::fit_configurable_spectral_scaler(rounded,c.level);
  cp::BankStandardizer n(c.level,"v6-split-objective-engineering-TRAIN-only",s.identity());
  n.add(im::relative_change_targets(retained.raw,c.level));n.freeze();const auto id=s.identity();n.validate(c.level,nullptr,&id);return{s,n};
}
im::LocationScaleDecoderTrace decode(im::LocationScaleDecoderModel &m,const torch::Tensor &z,const torch::Tensor &ids){
  ++decoder_calls;auto d=m->decode_with_trace(z,ids);check(d.mean.is_cuda()&&d.variance.is_cuda(),"actual CUDA distribution");return d;
}
torch::Tensor fresh_native(const torch::TensorOptions &o){return torch::linspace(-.4,.7,80,o).reshape({2,40}).clone().set_requires_grad(true);}
torch::Tensor manual_variance(im::LocationScaleDecoderModel &m,const torch::Tensor &z,const torch::Tensor &ids){
  auto score=z.narrow(1,36,4).matmul(m->variance_projection.transpose(0,1))+m->variance_bias;
  const auto k=im::location_scale_decoder_constants();auto e=torch::tensor(k[0],z.options()),shift=torch::tensor(k[1],z.options());
  auto v=e+(torch::ones({},z.options())-e)*(torch::softplus(score+shift,1.,20.)/torch::softplus(shift,1.,20.));
  auto order=im::channel_indices(ids,m->config().level,2,z.device());
  return v.reshape({2,3,1,9}).expand({2,3,128,9}).gather(1,order.unsqueeze(-1).unsqueeze(-1).expand({2,3,128,9}));
}
torch::Tensor hierarchical_fixture_reduce(const torch::Tensor &cell,const torch::Tensor &support,const torch::Tensor &eligible){
  auto counts=support.sum(std::vector<int64_t>{2,3});
  auto channels=cell.sum(std::vector<int64_t>{2,3})/counts.clamp_min(1).to(torch::kFloat64);
  auto rows=torch::where(eligible,channels,torch::zeros_like(channels)).sum(1)/eligible.sum(1).clamp_min(1).to(torch::kFloat64);
  return rows.masked_select(eligible.any(1)).mean();
}
int run(int argc,char **argv){
  check(argc==3&&std::string(argv[1])=="--settings","--settings CONFIG");
  check(std::filesystem::exists("/.dockerenv")&&torch::cuda::is_available(),"existing managed CUDA container");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);auto c=cp::read_settings(argv[2]);
  check(c.change_width==4&&c.uncertainty_width==4&&c.level.export_width==32&&c.level.huber_delta==1.,"same native40 geometry and delta1");
  auto stats=statistics(c);auto loaded=cp::fresh(c,stats.scaler,stats.bank,943001);auto m=loaded.model->core();
  const auto scaler_id=stats.scaler.identity(),bank_id=stats.bank.identity();const auto o=m->variance_projection.options();
  std::vector<std::pair<std::string,torch::Tensor>> fixed;
  int64_t learned=0,protected_values=0;for(const auto &p:m->named_parameters())if(p.value().requires_grad())learned+=p.value().numel();else{fixed.push_back({p.key(),p.value().detach().clone()});protected_values+=p.value().numel();}
  check(learned==294752&&protected_values==3888,"same learned/fixed counts");
  {torch::NoGradGuard guard;
    m->variance_projection.copy_(torch::sin(torch::arange(108,o)*.17).reshape({27,4})*.13+.025);
    m->variance_bias.copy_(torch::linspace(-.2,.3,27,o));
    m->encoder->change_decoder.copy_(torch::sin(torch::arange(13824,o)*.013).reshape({3456,4})*.04+.003);
  }
  auto ids=torch::tensor({{2,0,1},{1,2,0}},torch::kInt64).to(c.level.device);
  auto mask=torch::zeros({2,3,128,9},torch::kBool);auto ma=mask.accessor<bool,4>();
  for(int64_t b=0;b<2;++b)for(int64_t j=0;j<3;++j)for(int64_t h=0;h<128;++h)for(int64_t f=0;f<9;++f)
    ma[b][j][h][f]=!(b==1&&j==2)&&((h+2*f+j+b)%(3+j)==0);
  auto support=mask.to(c.level.device),eligible=support.sum(std::vector<int64_t>{2,3}).gt(0);
  check(eligible.sum().item<int64_t>()==5&&support[0][0].sum().item<int64_t>()!=support[0][1].sum().item<int64_t>(),"unequal support and one ineligible channel");
  auto z=fresh_native(o);auto d=decode(m,z,ids);
  auto target=d.mean.detach()-torch::linspace(-2.7,2.5,6912,o).reshape({2,3,128,9});
  target=torch::where(support,target,torch::full_like(target,std::numeric_limits<float>::quiet_NaN()));
  check((d.mean-target).abs().masked_select(support).lt(1).any().item<bool>()&&(d.mean-target).abs().masked_select(support).gt(1).any().item<bool>(),"nonvacuous quadratic and clipped Huber branches");
  check(d.variance.masked_select(support).max().item<double>()>d.variance.masked_select(support).min().item<double>(),"nonconstant heteroscedastic variance");
  check(torch::equal(d.variance,manual_variance(m,z,ids)),"unchanged positive-link forward");
  {torch::NoGradGuard guard;auto zm=z.detach().clone();zm.narrow(1,32,4).add_(.37);auto dm=decode(m,zm,ids);
    check(torch::equal(d.variance,dm.variance)&&!torch::equal(d.mean,dm.mean),"mean4 only changes mean");
    auto zu=z.detach().clone();zu.narrow(1,36,4).add_(.29);auto du=decode(m,zu,ids);
    check(torch::equal(d.mean,du.mean)&&!torch::equal(d.variance,du.variance),"uncertainty4 only changes variance");}
  auto objective=im::location_scale_decoder_objective(d.mean,d.variance,target,support,eligible);
  check(torch::equal(objective.total_loss,objective.mean_huber.loss+objective.variance_nll.loss),"explicit unit-coefficient total");
  auto reporting=im::location_scale_decoder_nll(d.mean.detach(),d.variance.detach(),target,support,eligible);
  check(!reporting.loss.requires_grad()&&torch::equal(reporting.loss,objective.variance_nll.loss),"reporting Gaussian NLL keeps ordinary numeric value and never backpropagates");
  auto poison=torch::full_like(d.mean,std::numeric_limits<float>::quiet_NaN());
  auto sanitized=im::location_scale_decoder_objective(torch::where(support,d.mean,poison),torch::where(support,d.variance,poison),target,support,eligible);
  check(torch::equal(sanitized.total_loss,objective.total_loss)&&torch::isfinite(sanitized.total_loss).item<bool>(),"unsupported nonfinite distributions/target sanitized before arithmetic");
  clear(m);objective.variance_nll.loss.backward();
  check(z.grad().narrow(1,0,36).eq(0).all().item<bool>()&&z.grad().narrow(1,36,4).abs().sum().item<double>()>0.,"variance NLL directly excludes parent32/mean4 and reaches uncertainty4");
  check(has_zero_gradient(m->encoder->change_decoder)&&has_nonzero_gradient(m->variance_projection)&&has_nonzero_gradient(m->variance_bias),"detached-error variance excludes mean decoder, learns positive-link head");
  clear(m);z=fresh_native(o);d=decode(m,z,ids);objective=im::location_scale_decoder_objective(d.mean,d.variance,target,support,eligible);objective.mean_huber.loss.backward();
  auto huber_gradient=z.grad().detach().clone();
  check(huber_gradient.narrow(1,32,4).abs().sum().item<double>()>0.&&huber_gradient.narrow(1,36,4).eq(0).all().item<bool>(),"Huber mean4 live, direct uncertainty4 gradient zero");
  check(has_nonzero_gradient(m->encoder->change_decoder)&&has_zero_gradient(m->variance_projection)&&has_zero_gradient(m->variance_bias),"Huber excludes variance head");
  clear(m);z=fresh_native(o);d=decode(m,z,ids);objective=im::location_scale_decoder_objective(d.mean,d.variance*7.,target,support,eligible);objective.mean_huber.loss.backward();
  check(torch::equal(z.grad(),huber_gradient),"fixing mean/target while changing variance leaves raw Huber native gradient exact");
  clear(m);z=fresh_native(o);d=decode(m,z,ids);objective=im::location_scale_decoder_objective(d.mean,d.variance,target,support,eligible);objective.total_loss.backward();
  auto actual=z.grad().detach().cpu().to(torch::kFloat64),manual=torch::zeros({2,8},torch::kFloat64);
  auto mean=d.mean.detach().cpu().to(torch::kFloat64),var=d.variance.detach().cpu().to(torch::kFloat64),truth=target.cpu().to(torch::kFloat64);
  auto D=m->encoder->change_decoder.detach().cpu().to(torch::kFloat64),V=m->variance_projection.detach().cpu().to(torch::kFloat64),score=d.score.detach().cpu().to(torch::kFloat64);
  auto ia=ids.cpu().contiguous();auto id=ia.accessor<int64_t,2>();auto out=manual.accessor<double,2>();auto mm=mean.accessor<double,4>(),vv=var.accessor<double,4>(),tt=truth.accessor<double,4>();
  const auto k=im::location_scale_decoder_constants();const double shift=k[1],coef=double(float(1.f-k[0])),u0=std::log1p(std::exp(shift));
  for(int64_t b=0;b<2;++b){int64_t channels=0;for(int64_t j=0;j<3;++j)channels+=mask[b][j].any().item<bool>();
    for(int64_t j=0;j<3;++j){const auto count=mask[b][j].sum().item<int64_t>();if(!count)continue;const double factor=1./(2.*channels*count);
      for(int64_t h=0;h<128;++h)for(int64_t f=0;f<9;++f)if(ma[b][j][h][f]){
        const auto cf=id[b][j]*9+f,di=(id[b][j]*128+h)*9+f;const double error=mm[b][j][h][f]-tt[b][j][h][f],variance=vv[b][j][h][f];
        const double q=score[b][cf].item<double>()+shift,link=coef/(u0*(1.+std::exp(-q)));
        for(int64_t j2=0;j2<4;++j2){out[b][j2]+=factor*std::max(-1.,std::min(1.,error))*D[di][j2].item<double>();out[b][4+j2]+=factor*.5*(1./variance-error*error/(variance*variance))*link*V[cf][j2].item<double>();}
      }}}
  check(torch::allclose(actual.narrow(1,32,8),manual,2e-4,2e-6),"manual composite mean4/uncertainty4 adjoints with semantic gather");
  auto reference=fresh_native(o);++decoder_calls;auto parent_mean=m->encoder->level->decode(reference.narrow(1,0,32),ids);
  auto order=im::channel_indices(ids,c.level,2,c.level.device).unsqueeze(-1).unsqueeze(-1).expand({2,3,128,9});
  auto residual=reference.narrow(1,32,4).matmul(m->encoder->change_decoder.transpose(0,1)).reshape({2,3,128,9}).gather(1,order);
  auto expected_mean=parent_mean+residual,expected_var=manual_variance(m,reference,ids);
  auto safe_mean=torch::where(support,expected_mean,torch::zeros_like(expected_mean)).to(torch::kFloat64);
  auto safe_target=torch::where(support,target.detach(),torch::zeros_like(target)).to(torch::kFloat64);
  auto error=safe_mean-safe_target,safe_var=torch::where(support,expected_var,torch::ones_like(expected_var)).to(torch::kFloat64);
  auto mean_cell=torch::where(error.abs()<=1.,.5*error.square(),error.abs()-.5);
  auto variance_cell=.5*(error.detach().square()/safe_var+safe_var.log());
  auto mean_loss=hierarchical_fixture_reduce(mean_cell,support,eligible),variance_loss=hierarchical_fixture_reduce(variance_cell,support,eligible),total=mean_loss+variance_loss;
  auto reference_gradient=torch::autograd::grad({total},{reference})[0].detach().cpu().to(torch::kFloat64);
  check(std::abs(mean_loss.item<double>()-objective.mean_huber.loss.item<double>())<1e-10,"manual F64 Huber component parity");
  check(std::abs(variance_loss.item<double>()-objective.variance_nll.loss.item<double>())<1e-10,"manual F64 variance NLL component parity");
  check(std::abs(total.item<double>()-objective.total_loss.item<double>())<1e-10&&torch::allclose(actual,reference_gradient,2e-5,2e-7),"manual composite-objective native40 gradient parity");
  check(torch::allclose(actual.narrow(1,0,36),huber_gradient.cpu().to(torch::kFloat64).narrow(1,0,36),2e-5,2e-7),"total direct parent32/mean4 gradient equals only Huber component");
  check(actual.narrow(1,0,20).abs().sum().item<double>()>0.,"Huber keeps learned parent path live");
  // Show the retained shared clipping operation couples branch step scaling.
  auto a=torch::zeros({2},o).set_requires_grad(true),b=torch::zeros({2},o).set_requires_grad(true);
  a.mutable_grad()=torch::tensor({3.f,4.f},o);b.mutable_grad()=torch::tensor({0.f,12.f},o);
  const auto raw_a=a.grad().clone();const double norm1=torch::nn::utils::clip_grad_norm_(std::vector<torch::Tensor>{a,b},1.);auto clipped1=a.grad().clone();
  a.mutable_grad()=raw_a.clone();b.mutable_grad()=torch::tensor({0.f,24.f},o);
  const double norm2=torch::nn::utils::clip_grad_norm_(std::vector<torch::Tensor>{a,b},1.);
  check(norm2>norm1&&!torch::equal(a.grad(),clipped1)&&a.grad().norm().item<double>()<clipped1.norm().item<double>(),"unchanged direct mean gradient can receive a different joint clip factor from variance gradients");
  clear(m);auto bank=torch::sin(torch::arange(540,o)*.019).reshape({2,270}),parent=torch::linspace(-.3,.4,64,o).reshape({2,32});
  auto composed=torch::cat({parent,bank.matmul(m->encoder->change_projection.transpose(0,1)),bank.matmul(m->uncertainty_projection.transpose(0,1))},1);
  d=decode(m,composed,ids);objective=im::location_scale_decoder_objective(d.mean,d.variance,target,support,eligible);objective.variance_nll.loss.backward();
  check(has_zero_gradient(m->encoder->change_projection)&&has_nonzero_gradient(m->uncertainty_projection),"variance NLL directly learns only uncertainty bank projection");
  clear(m);composed=torch::cat({parent,bank.matmul(m->encoder->change_projection.transpose(0,1)),bank.matmul(m->uncertainty_projection.transpose(0,1))},1);
  d=decode(m,composed,ids);objective=im::location_scale_decoder_objective(d.mean,d.variance,target,support,eligible);objective.mean_huber.loss.backward();
  check(has_nonzero_gradient(m->encoder->change_projection)&&has_zero_gradient(m->uncertainty_projection),"Huber directly learns mean bank projection only");
  clear(m);composed=torch::cat({parent,bank.matmul(m->encoder->change_projection.transpose(0,1)),bank.matmul(m->uncertainty_projection.transpose(0,1))},1);
  d=decode(m,composed,ids);objective=im::location_scale_decoder_objective(d.mean,d.variance,target,support,eligible);objective.total_loss.backward();
  std::vector<torch::Tensor> params{m->encoder->change_projection,m->uncertainty_projection,m->encoder->change_decoder,m->variance_projection,m->variance_bias},before;
  for(const auto &p:params){check(has_nonzero_gradient(p),"both projections/decoder heads have composite-objective CUDA gradients");before.push_back(p.detach().clone());}
  torch::nn::utils::clip_grad_norm_(params,1.);torch::optim::AdamW optimizer(params,cp::training::adamw_options());optimizer.step();torch::cuda::synchronize();
  for(size_t i=0;i<params.size();++i)check(!torch::equal(params[i],before[i]),"actual CUDA AdamW moves both projections and distribution heads");
  for(const auto &[name,value]:fixed)check(torch::equal(m->named_parameters()[name],value)&&!m->named_parameters()[name].grad().defined(),"literal prior remains exact without gradients");
  check(stats.scaler.identity()==scaler_id&&stats.bank.identity()==bank_id,"fixture statistics remain frozen");
  std::cout<<"PASS CPB-v6 "<<cp::kObjectivePolicy<<": "<<checks<<" CUDA checks; "<<decoder_calls<<" decoder calls; Huber mean live, detached-error NLL mean gradient zero, manual composite-objective native40 gradient parity, uncertainty projection learning, shared global clip coupling; synthetic optimizer steps=1; optimizer=AdamW, parameter tensors moved=5; encoder forwards=0, old artifacts=0, quality fits=0\n";return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
