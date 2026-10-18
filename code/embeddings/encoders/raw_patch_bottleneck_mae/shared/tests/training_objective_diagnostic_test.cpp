// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/training_objective_diagnostic.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/cuda.h>
#include <iostream>
#include <map>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace od = rpb::objective_diagnostic;
using namespace rpb_test;

struct RuntimeWitness {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeWitness() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i=0;i<at::getNumGPUs();++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for(const auto &generator:generators)states.push_back(generator.get_state().clone());
  }
  void unchanged() const {
    check(threads==at::get_num_threads(),"diagnostic thread state unchanged");
    for(size_t i=0;i<generators.size();++i)check(torch::equal(states[i],generators[i].get_state()),"diagnostic Torch RNG unchanged");
  }
};
std::map<std::string,torch::Tensor> weights(const rpb::Model &model) {
  std::map<std::string,torch::Tensor> out;
  for(const auto &p:model->named_parameters())out.emplace("parameter/"+p.key(),p.value().detach().cpu().clone());
  for(const auto &p:model->named_buffers())out.emplace("buffer/"+p.key(),p.value().detach().cpu().clone());
  return out;
}
void exact_weights(const rpb::Model &model,const std::map<std::string,torch::Tensor> &before) {
  const auto after=weights(model);check(after.size()==before.size(),"same frozen parameter/buffer set");
  for(const auto &[key,value]:before)check(torch::equal(value,after.at(key)),"zero-update exact tensor preservation: "+key);
}
od::FrozenRidgeHead head() {
  const auto a=torch::arange(32,torch::kFloat64);
  return {a*.01,.7+a*.02,-a*.003,1.2+a*.005,
      torch::sin(torch::arange(64,torch::kFloat64)*.31).reshape({32,2}),torch::tensor({.2,-.3},torch::kFloat64)};
}
void primitive_contracts() {
  auto prediction=torch::tensor({.2,.1,-.1},torch::kFloat64).view({1,3,1,1}).clone().set_requires_grad(true);
  auto target=torch::zeros_like(prediction),query=torch::ones_like(prediction,torch::kBool);
  const auto value=od::pair_difference_huber(prediction,target,query,1);
  const double expected=(.01+.09+.04)/12;
  check(std::abs(value.loss.item<double>()-expected)<1e-14,"sqrt2-normalized three-pair quadratic arithmetic");
  check(value.supported_examples==1&&value.supported_pairs==3&&value.target_pair_cells==3,"supported pair/cell counts");
  const auto gradient=torch::autograd::grad({value.loss},{prediction})[0];
  check(std::abs(gradient.sum().item<double>())<1e-14,"common normalized residual direction cancels");
  const auto order=torch::tensor({2,0,1},torch::kInt64);
  const auto reordered=od::pair_difference_huber(prediction.index_select(1,order),target.index_select(1,order),query.index_select(1,order),1);
  check(std::abs(reordered.loss.item<double>()-value.loss.item<double>())<1e-14,"all unordered pairs are semantic-storage permutation invariant");
  auto sparse=torch::zeros({2,3,4,1},torch::kBool);sparse[0][0][0][0]=true;sparse[0][1][0][0]=true;sparse[0][2][1][0]=true;
  sparse[1][0][0][0]=true;sparse[1][1][1][0]=true;sparse[1][2][2][0]=true;
  auto safe=torch::ones({2,3,4,1},torch::kFloat64).set_requires_grad(true);
  auto targets=torch::zeros_like(safe);targets=torch::where(sparse,targets,torch::full_like(targets,std::numeric_limits<double>::quiet_NaN()));
  const auto partial=od::pair_difference_huber(safe,targets,sparse,1);
  check(partial.supported_examples==1&&partial.supported_pairs==1&&partial.target_pair_cells==1,"sparse original-Q intersection; unsupported row preserved");
  const auto empty=od::pair_difference_huber(safe,targets,torch::zeros_like(sparse),1);
  check(empty.supported_examples==0&&empty.loss.item<double>()==0&&empty.loss.requires_grad(),"unsupported auxiliary differentiable zero");
  check(torch::autograd::grad({empty.loss},{safe})[0].eq(0).all().item<bool>(),"unsupported zero gradient");
  auto large=torch::tensor({1e20f,-1e20f,1e20f},torch::kFloat32).reshape({1,3,1,1}).clone().set_requires_grad(true);
  const auto linear=od::pair_difference_huber(large,torch::zeros_like(large),torch::ones_like(large,torch::kBool),1);
  check(torch::isfinite(linear.loss).item<bool>()&&torch::isfinite(torch::autograd::grad({linear.loss},{large})[0]).all().item<bool>(),
      "large finite linear-region Huber avoids unused quadratic overflow");
  const auto good=od::compare_gradients(torch::tensor({1.,0.}),torch::tensor({0.,1.}),torch::tensor({-1.,-1.}));
  check(good.supported&&good.base_descent_margin_cosine>0&&good.combined_descent_margin_cosine>good.base_descent_margin_cosine&&
      good.combined_descent_base_directional_derivative<0,"unit descent signs and equal weight1 combination");
  const auto bad=od::compare_gradients(torch::tensor({1.,0.}),torch::zeros({2}),torch::tensor({1.,0.}));
  check(bad.supported&&bad.base_descent_margin_cosine==-1,"positive reconstruction-gradient dot correct-margin is conflict");
  const auto undefined=od::compare_gradients(torch::zeros({2}),torch::ones({2}),torch::ones({2}));
  check(!undefined.supported,"zero norm cannot become helpful or conflicting");
}
void cuda_contracts() {
  check(torch::cuda::is_available(),"actual CUDA required; admission must not silently skip");
  rpb::Config c;c.channel_mixer_layers=1;c.global_bottleneck_mode=2;c.device=torch::kCUDA;
  auto raw=input(c,4);auto scaler=rpb::fit_scaler(raw,c);auto normalized=scaler.transform(raw,c);
  torch::manual_seed(901234);rpb::Model model(c);model->eval();
  int64_t count=0;for(const auto &p:model->parameters()){check(p.is_cuda(),"actual CUDA parameter");count+=p.numel();}
  check(count==225805&&normalized.data.is_cuda(),"fixed225805 parameters and input on CUDA");
  auto hidden=torch::zeros_like(normalized.observed);hidden.narrow(2,8,8).fill_(true);
  const auto original=rpb::mask_from_hidden(normalized.observed,hidden,c);const auto labels=torch::tensor({0,1,0,1},torch::kInt64);
  const auto frozen=head();const auto before=weights(model);RuntimeWitness runtime;
  const auto bank=od::inspect_bank(model,normalized,original,original.visible,labels,frozen);
  runtime.unchanged();exact_weights(model,before);
  check(bank.cuda_parameter_count==225805&&bank.frozen_z.sizes()==torch::IntArrayRef({4,32})&&
      bank.graph_export_max_error<=1e-6&&bank.encoder_base_gradient.numel()>0&&bank.comparison.supported,"real CUDA decoder/encoder-gradient branch admission");
  for(const auto &g:{bank.encoder_base_gradient,bank.encoder_auxiliary_gradient,bank.encoder_margin_gradient,
      bank.latent_base_gradient,bank.latent_auxiliary_gradient,bank.latent_margin_gradient})
    check(torch::isfinite(g).all().item<bool>(),"finite saved gradient arrays");
  check(bank.encoder_base_gradient.norm().item<double>()>0&&bank.encoder_auxiliary_gradient.norm().item<double>()>0&&
      bank.encoder_margin_gradient.norm().item<double>()>0,"nonzero actual CUDA encoder gradients");
  for(const auto &name:bank.parameter_names)check(name.rfind("decoder_",0)!=0,"strict nondecoder gradient boundary");
  check(std::is_sorted(bank.parameter_names.begin(),bank.parameter_names.end())&&
      bank.parameter_offsets[-1].item<int64_t>()==bank.encoder_base_gradient.numel(),"lexical named unused-zero gradient offsets");
  // Independent ordinary graph parity: identical flags, same fixed masks and
  // scalar objective, no optimizer construction or parameter .grad mutation.
  for(const auto &p:model->named_parameters())p.value().set_requires_grad(p.key().rfind("decoder_",0)!=0);
  auto forward=model->forward(normalized,hidden);
  auto actual=torch::autograd::grad({forward.loss}, [&]{std::vector<torch::Tensor> p;const auto named=model->named_parameters();for(const auto &name:bank.parameter_names)p.push_back(named[name]);return p;}(),{},false,false,true);
  std::vector<torch::Tensor> pieces;const auto named=model->named_parameters();
  for(size_t i=0;i<actual.size();++i)pieces.push_back((actual[i].defined()?actual[i]:torch::zeros_like(named[bank.parameter_names[i]])).detach().to(torch::kCPU,torch::kFloat64).flatten());
  check(torch::equal(torch::cat(pieces),bank.encoder_base_gradient),"diagnostic parameter gradient exactly equals ordinary core forward gradient");
  for(auto &p:model->parameters())p.set_requires_grad(false);
  auto z=bank.frozen_z.to(torch::kCUDA).clone().set_requires_grad(true);
  const auto prediction=model->decode(z,normalized.channel_ids);
  const auto base=rpb::hierarchical_huber(prediction,normalized.data,original.target,original.eligible_channels,1);
  const auto gz=torch::autograd::grad({base.loss},{z})[0].detach().to(torch::kCPU,torch::kFloat64);
  check(torch::equal(gz*(frozen.outer_scale*frozen.ridge_scale),bank.latent_base_gradient),"final Ridge leaf gradient multiplies both frozen scales");
  auto u=torch::zeros({4,32},torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64).requires_grad(true));
  const auto exact=od::latent_from_zero_perturbation(bank.frozen_z.to(torch::kCUDA),u,frozen);
  check(torch::equal(exact.detach().cpu(),bank.frozen_z),"zero perturbation has no inverse-normalizer rounding");
  const auto margin=od::signed_ridge_margin(exact,labels,frozen).sum();
  const auto gu=torch::autograd::grad({margin},{u})[0].detach().cpu();
  // The decoder perturbation deliberately casts to model float32. Its real
  // cast-chain derivative can round; the declared margin direction is the
  // analytical float64 saved-head law in final Ridge coordinates.
  close(gu,bank.latent_margin_gradient,"bounded model-float32 cast-chain margin discrepancy",1e-6,1e-7);
  auto direct_u=torch::zeros({4,32},torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64).requires_grad(true));
  const auto logits=direct_u.matmul(frozen.weights.to(torch::kCUDA))+frozen.intercept.to(torch::kCUDA);
  const auto direct_margin=((labels.to(torch::kCUDA,torch::kFloat64)*2-1)*(logits.select(1,1)-logits.select(1,0))).sum();
  const auto direct_gradient=torch::autograd::grad({direct_margin},{direct_u})[0].detach().cpu();
  check(torch::equal(direct_gradient,bank.latent_margin_gradient),"exact analytical final-Ridge signed margin law");
  auto poison=normalized;poison.data=normalized.data.clone();poison.data.masked_fill_(original.target,90000);
  const auto poisoned=od::inspect_bank(model,poison,original,original.visible,labels,frozen);
  check(torch::equal(poisoned.frozen_z,bank.frozen_z)&&torch::equal(poisoned.prediction,bank.prediction),"query values never enter encoder or frozen decoder");
  check(!torch::equal(poisoned.encoder_base_gradient,bank.encoder_base_gradient),"query changes affect targets/objective, not visible representation");
  const auto deletion=rpb::context_deletion::make_plan(original,normalized.channel_ids,c,4404,513,rpb::ContextDeletionRecipe::coordinate15_v1);
  const auto augmented=od::inspect_bank(model,normalized,original,deletion.visible,labels,frozen);
  check(torch::equal(augmented.query,bank.query)&&torch::equal(augmented.target,bank.target)&&
      !augmented.visible.logical_and(bank.visible.logical_not()).any().item<bool>(),"actual-policy context preserves original Q and targets");
  const auto reordered=torch::tensor({2,0,1},torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  auto permuted=normalized;permuted.data=normalized.data.index_select(1,reordered);permuted.observed=normalized.observed.index_select(1,reordered);
  permuted.channel_ids=normalized.channel_ids.index_select(normalized.channel_ids.dim()-1,reordered);
  const auto permuted_mask=rpb::mask_from_hidden(permuted.observed,hidden.index_select(1,reordered),c);
  const auto permuted_bank=od::inspect_bank(model,permuted,permuted_mask,permuted_mask.visible,labels,frozen);
  close(permuted_bank.frozen_z,bank.frozen_z,"semantic permutation global32 serving",2e-6,2e-6);
  check(std::abs(permuted_bank.auxiliary_loss-bank.auxiliary_loss)<1e-6,"semantic permutation all-pair scalar objective");
  exact_weights(model,before);
  for(const auto &p:model->parameters())check(!p.grad().defined(),"autograd::grad leaves parameter .grad untouched");
  auto malformed=frozen;malformed.outer_scale=torch::zeros_like(malformed.outer_scale);
  rejects([&]{od::inspect_bank(model,normalized,original,original.visible,labels,malformed);},"malformed saved scale rejected");
  rejects([&]{od::inspect_bank(model,normalized,original,normalized.observed,labels,frozen);},"hidden target support rejected at encoder boundary");
  auto corrupt=original;corrupt.visible=normalized.observed.clone();
  rejects([&]{od::inspect_bank(model,normalized,corrupt,original.visible,labels,frozen);},"corrupt original visible plan rejected");
  auto no_context=torch::zeros_like(original.visible);
  rejects([&]{od::inspect_bank(model,normalized,original,no_context,labels,frozen);},"context cannot remove required visible patch groups");
  std::cout<<"TRAIN objective CUDA admission passed; cuda_parameters="<<count
      <<"; input_cuda=true; gradient_cuda=true; updates=0; optimizer_created=false; weights_unchanged=true; graph_export_max_error="
      <<bank.graph_export_max_error<<'\n';
}
}
int main() {
  try { primitive_contracts();cuda_contracts();return 0; }
  catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
