// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Artificial CUDA mechanism fixture; no encoder forward, old artifact or quality fit.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v5.h"
#include <torch/cuda.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v5;
namespace im=cp::implementation;
int64_t checks=0,decoder_calls=0;
void check(bool yes,const char *why){++checks;TORCH_CHECK(yes,"[CPB-v5 separation fixture] ",why);}
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
  cp::BankStandardizer n(c.level,"v5-disjoint-projection-engineering-TRAIN-only",s.identity());
  n.add(im::relative_change_targets(retained.raw,c.level));n.freeze();const auto id=s.identity();n.validate(c.level,nullptr,&id);return{s,n};
}
im::LocationScaleDecoderTrace decode(im::LocationScaleDecoderModel &m,const torch::Tensor &z,const torch::Tensor &ids){++decoder_calls;auto d=m->decode_with_trace(z,ids);check(d.mean.is_cuda()&&d.variance.is_cuda(),"actual CUDA distribution");return d;}
torch::Tensor fresh_native(const torch::TensorOptions &o){return torch::linspace(-.4,.7,80,o).reshape({2,40}).clone().set_requires_grad(true);}
torch::Tensor manual_variance(im::LocationScaleDecoderModel &m,const torch::Tensor &z,const torch::Tensor &ids){
  auto score=torch::matmul(z.narrow(1,36,4),m->variance_projection.transpose(0,1))+m->variance_bias;
  const auto k=im::location_scale_decoder_constants();const auto e=torch::tensor(k[0],z.options()),c=torch::tensor(k[1],z.options());
  auto v=e+(torch::ones({},z.options())-e)*(torch::softplus(score+c,im::kLocationScaleDecoderSoftplusBeta,im::kLocationScaleDecoderSoftplusThreshold)/torch::softplus(c,im::kLocationScaleDecoderSoftplusBeta,im::kLocationScaleDecoderSoftplusThreshold));
  auto order=im::channel_indices(ids,m->config().level,2,z.device());return v.reshape({2,3,1,9}).expand({2,3,128,9}).gather(1,order.unsqueeze(-1).unsqueeze(-1).expand({2,3,128,9}));
}
int run(int argc,char **argv){
  check(argc==3&&std::string(argv[1])=="--settings","--settings CONFIG required");
  check(std::filesystem::exists("/.dockerenv")&&torch::cuda::is_available(),"existing managed CUDA container required");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);auto c=cp::read_settings(argv[2]);
  check(c.change_width==4&&c.uncertainty_width==4&&c.level.export_width==32,"parent32+mean4+uncertainty4");
  auto s=statistics(c);auto loaded=cp::fresh(c,s.scaler,s.bank,943001);auto m=loaded.model->core();
  const auto scaler_id=s.scaler.identity(),bank_id=s.bank.identity();
  std::vector<std::pair<std::string,torch::Tensor>> fixed;
  int64_t learned=0,protected_values=0;for(const auto &p:m->named_parameters())if(p.value().requires_grad())learned+=p.value().numel();else{fixed.push_back({p.key(),p.value().detach().clone()});protected_values+=p.value().numel();}
  check(learned==294752&&protected_values==3888,"exact learned/fixed counts");
  const auto o=m->variance_projection.options();
  // Engineering-only nonzero matrices prevent the zero-start decoder/variance heads
  // from making gradient separation or projection learning vacuously true.
  {torch::NoGradGuard guard;
    m->variance_projection.copy_(torch::sin(torch::arange(108,o)*.17).reshape({27,4})*.13+.025);
    m->variance_bias.copy_(torch::linspace(-.2,.3,27,o));
    m->encoder->change_decoder.copy_(torch::sin(torch::arange(13824,o)*.013).reshape({3456,4})*.04+.003);
  }
  auto ids=torch::tensor({{2,0,1},{1,2,0}},torch::kInt64).to(c.level.device);
  auto z=fresh_native(o);auto d=decode(m,z,ids);
  check(torch::equal(d.variance,manual_variance(m,z,ids)),"variance equals independent live-uncertainty4 formula");
  {torch::NoGradGuard guard;auto zm=z.detach().clone();zm.narrow(1,32,4).add_(.37);auto dm=decode(m,zm,ids);
    check(torch::equal(d.variance,dm.variance)&&!torch::equal(d.mean,dm.mean),"mean4 changes mean only");
    auto zu=z.detach().clone();zu.narrow(1,36,4).add_(.29);auto du=decode(m,zu,ids);
    check(torch::equal(d.mean,du.mean)&&!torch::equal(d.variance,du.variance),"uncertainty4 changes variance only");}
  clear(m);d.variance.sum().backward();
  check(z.grad().narrow(1,0,36).eq(0).all().item<bool>()&&z.grad().narrow(1,36,4).abs().sum().item<double>()>0.,"variance-only gradient reaches uncertainty4 and excludes parent32/mean4");
  check(has_nonzero_gradient(m->variance_projection)&&has_nonzero_gradient(m->variance_bias)&&has_zero_gradient(m->encoder->change_decoder),"variance head live, mean decoder excluded");
  clear(m);z=fresh_native(o);d=decode(m,z,ids);d.mean.square().mean().backward();
  check(z.grad().narrow(1,32,4).abs().sum().item<double>()>0.&&z.grad().narrow(1,36,4).eq(0).all().item<bool>(),"mean gradient live at mean4 and absent at uncertainty4");
  check(has_nonzero_gradient(m->encoder->change_decoder)&&has_zero_gradient(m->variance_projection)&&has_zero_gradient(m->variance_bias),"mean decoder live, variance parameters excluded");
  auto mask=torch::zeros({2,3,128,9},torch::kBool);auto ma=mask.accessor<bool,4>();
  for(int64_t b=0;b<2;++b)for(int64_t j=0;j<3;++j)for(int64_t h=0;h<128;++h)for(int64_t f=0;f<9;++f)
    ma[b][j][h][f]=!(b==1&&j==2)&&((h+2*f+j+b)%(3+j)==0);
  auto support=mask.to(c.level.device),eligible=support.sum(std::vector<int64_t>{2,3}).gt(0);
  clear(m);z=fresh_native(o);d=decode(m,z,ids);auto target=torch::cos(torch::arange(6912,o)*.011).reshape({2,3,128,9});
  auto loss=im::location_scale_decoder_nll(d.mean,d.variance,target,support,eligible);loss.loss.backward();
  auto actual=z.grad().detach().cpu().to(torch::kFloat64),manual=torch::zeros({2,8},torch::kFloat64);
  auto mean=d.mean.detach().cpu().to(torch::kFloat64),var=d.variance.detach().cpu().to(torch::kFloat64),truth=target.cpu().to(torch::kFloat64);
  auto D=m->encoder->change_decoder.detach().cpu().to(torch::kFloat64),V=m->variance_projection.detach().cpu().to(torch::kFloat64),score=d.score.detach().cpu().to(torch::kFloat64);
  auto ia=ids.cpu().contiguous();auto id=ia.accessor<int64_t,2>();auto out=manual.accessor<double,2>();auto mm=mean.accessor<double,4>(),vv=var.accessor<double,4>(),tt=truth.accessor<double,4>();
  const auto constants=im::location_scale_decoder_constants();const double eps=constants[0],shift=constants[1],coef=double(float(1.f-float(eps))),u0=std::log1p(std::exp(shift));
  for(int64_t b=0;b<2;++b){int64_t channels=0;for(int64_t j=0;j<3;++j)channels+=mask[b][j].any().item<bool>();
    for(int64_t j=0;j<3;++j){const auto count=mask[b][j].sum().item<int64_t>();if(!count)continue;const double factor=1./(2.*channels*count);
      for(int64_t h=0;h<128;++h)for(int64_t f=0;f<9;++f)if(ma[b][j][h][f]){
        const auto cf=id[b][j]*9+f,di=(id[b][j]*128+h)*9+f;const double error=mm[b][j][h][f]-tt[b][j][h][f],variance=vv[b][j][h][f];
        const double q=score[b][cf].item<double>()+shift,link=coef/(u0*(1.+std::exp(-q)));
        for(int64_t k=0;k<4;++k){out[b][k]+=factor*error/variance*D[di][k].item<double>();out[b][4+k]+=factor*.5*(1./variance-error*error/(variance*variance))*link*V[cf][k].item<double>();}
      }} }
  const auto tail=actual.narrow(1,32,8);check((tail-manual).abs().max().item<double>()<=2e-6+2e-4*manual.abs().max().item<double>(),"manual full Gaussian native mean4/uncertainty4 gradients with semantic gather and unequal support");
  // Independently assemble the complete distribution/loss expression. The
  // preserved parent decoder supplies its unchanged nonlinear Jacobian; the
  // analytic tail oracle above independently checks both new branch derivatives.
  auto reference=fresh_native(o);++decoder_calls;
  auto parent_mean=m->encoder->level->decode(reference.narrow(1,0,32),ids);
  auto order=im::channel_indices(ids,c.level,2,c.level.device).unsqueeze(-1).unsqueeze(-1).expand({2,3,128,9});
  auto residual=reference.narrow(1,32,4).matmul(m->encoder->change_decoder.transpose(0,1)).reshape({2,3,128,9}).gather(1,order);
  auto expected_mean=parent_mean+residual,expected_var=manual_variance(m,reference,ids);
  auto safe_mean=torch::where(support,expected_mean,torch::zeros_like(expected_mean)).to(torch::kFloat64);
  auto safe_target=torch::where(support,target.detach(),torch::zeros_like(target)).to(torch::kFloat64);
  auto safe_error=safe_mean-safe_target;
  auto safe_var=torch::where(support,expected_var,torch::ones_like(expected_var)).to(torch::kFloat64);
  auto cell=.5*(safe_error.square()/safe_var+safe_var.log());
  auto counts=support.sum(std::vector<int64_t>{2,3});
  auto channel=cell.sum(std::vector<int64_t>{2,3})/counts.clamp_min(1).to(torch::kFloat64);
  auto example=torch::where(eligible,channel,torch::zeros_like(channel)).sum(1)/eligible.sum(1).clamp_min(1).to(torch::kFloat64);
  auto reference_loss=example.mean();auto reference_gradient=torch::autograd::grad({reference_loss},{reference})[0].detach().cpu().to(torch::kFloat64);
  check(std::abs(reference_loss.item<double>()-loss.loss.item<double>())<1e-10&&torch::allclose(actual,reference_gradient,2e-5,2e-7),"manual full-NLL native40 gradient parity");
  check(actual.narrow(1,0,20).abs().sum().item<double>()>0.,"full NLL parent learned path stays live");
  check(has_nonzero_gradient(m->variance_projection)&&has_nonzero_gradient(m->variance_bias)&&has_nonzero_gradient(m->encoder->change_decoder),"full NLL learns both distribution heads");
  clear(m);auto bank=torch::sin(torch::arange(540,o)*.019).reshape({2,270});auto parent=torch::linspace(-.3,.4,64,o).reshape({2,32});
  auto composed=torch::cat({parent,bank.matmul(m->encoder->change_projection.transpose(0,1)),bank.matmul(m->uncertainty_projection.transpose(0,1))},1);
  d=decode(m,composed,ids);d.variance.sum().backward();
  check(has_zero_gradient(m->encoder->change_projection)&&has_nonzero_gradient(m->uncertainty_projection),"variance objective learns uncertainty bank projection without reaching mean bank projection");
  clear(m);composed=torch::cat({parent,bank.matmul(m->encoder->change_projection.transpose(0,1)),bank.matmul(m->uncertainty_projection.transpose(0,1))},1);
  d=decode(m,composed,ids);auto full=im::location_scale_decoder_nll(d.mean,d.variance,target,support,eligible);full.loss.backward();
  check(has_nonzero_gradient(m->encoder->change_projection)&&has_nonzero_gradient(m->uncertainty_projection),"actual distinct bank projections receive Gaussian gradients");
  std::vector<torch::Tensor> params{m->encoder->change_projection,m->uncertainty_projection,m->encoder->change_decoder,m->variance_projection,m->variance_bias},before;
  for(const auto &p:params)before.push_back(p.detach().clone());torch::optim::SGD optimizer(params,torch::optim::SGDOptions(.001));optimizer.step();torch::cuda::synchronize();
  for(size_t i=0;i<params.size();++i)check(!torch::equal(params[i],before[i]),"actual CUDA optimizer moves mean/uncertainty projections and both decoder heads");
  for(const auto &[name,value]:fixed)check(torch::equal(m->named_parameters()[name],value)&&!m->named_parameters()[name].grad().defined(),"fixed spectral prior exact without gradients");
  check(s.scaler.identity()==scaler_id&&s.bank.identity()==bank_id,"fresh fixture statistics remain exact");
  std::cout<<"PASS CPB-v5 "<<cp::kGradientPolicy<<": "<<checks<<" CUDA checks; "<<decoder_calls<<" decoder calls; disjoint native gradients, manual full-NLL native40 gradient parity, uncertainty projection learning; synthetic optimizer steps=1; encoder forwards=0, old artifacts=0, quality fits=0\n";return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
