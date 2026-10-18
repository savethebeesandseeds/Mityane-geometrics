// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/unit_temporal_relation.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "rpb_test_support.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#ifndef UNIT_RELATION_SCREEN_SOURCE_ID
#define UNIT_RELATION_SCREEN_SOURCE_ID "unrecorded"
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;

rpb::Config configuration() {
  rpb::Config c; c.device = torch::kCUDA; c.channel_ids = {101, 202, 303};
  c.channel_mixer_layers = 1; c.channel_mixer_placement = 1; c.global_bottleneck_mode = 2;
  return c;
}
void exact(const torch::Tensor &a,const torch::Tensor &b,const std::string &why) {
  check(a.scalar_type()==b.scalar_type()&&a.sizes()==b.sizes()&&torch::equal(a,b),why);
}
torch::Tensor scalar_reference(const torch::Tensor &odd12) {
  const auto x=odd12.detach().to(torch::kCPU).contiguous();
  auto out=torch::zeros_like(x);const auto values=x.accessor<float,2>();auto result=out.accessor<float,2>();
  for(int64_t b=0;b<x.size(0);++b)for(int64_t pair=0;pair<3;++pair) {
    double energy=0;for(int64_t k=0;k<4;++k)energy+=double(values[b][pair*4+k])*values[b][pair*4+k];
    const auto denominator=std::max(std::sqrt(energy),1e-12);
    for(int64_t k=0;k<4;++k)result[b][pair*4+k]=float(values[b][pair*4+k]/denominator);
  }
  return out.to(odd12.device());
}
void normalization_contract() {
  const auto host=torch::tensor(std::vector<float>{
      0,0,0,0, 3,4,0,0, -2,1,2,-1,
      1e-15f,-2e-15f,0,0, .3f,-.2f,.4f,.1f, -1,2,-3,4},torch::kFloat32).reshape({2,12});
  auto odd=host.to(torch::kCUDA).set_requires_grad(true);
  const auto normalized=rpb::unit_grouped_odd_relations(odd);
  close(normalized,scalar_reference(odd),"independent scalar normalization including zero/subfloor",2e-6,1e-7);
  exact(normalized[0].narrow(0,0,4),torch::zeros({4},normalized.options()),"allzero exactzero");
  check(normalized[1][0].item<float>()>0&&normalized[1][0].item<float>()<.01f,"subfloor vector divided by floor without quantization");
  exact(rpb::unit_grouped_odd_relations(-odd),-normalized,"normalization is exactly time-odd on arbitrary vectors");
  auto scaled=odd.detach().clone();scaled[0].narrow(0,4,4).mul_(7);scaled[1].narrow(0,8,4).mul_(.25);
  close(rpb::unit_grouped_odd_relations(scaled),normalized,"positive scale invariance abovefloor and pair isolation",2e-6,1e-7);
  auto isolated=odd.detach().clone();isolated[0].narrow(0,4,4).mul_(-2);
  const auto changed=rpb::unit_grouped_odd_relations(isolated);
  exact(changed[0].narrow(0,0,4),normalized[0].narrow(0,0,4),"first semantic pair isolated");
  exact(changed[0].narrow(0,8,4),normalized[0].narrow(0,8,4),"third semantic pair isolated");
  close(changed[0].narrow(0,4,4),-normalized[0].narrow(0,4,4),"only changed pair reverses",2e-6,1e-7);
  normalized.sum().backward();finite(odd.grad(),"finite CUDA normalization gradient at zero and floor");
  check(odd.grad().is_cuda(),"normalization gradient actual CUDA");
  rejects([&]{rpb::unit_grouped_odd_relations(host);},"CPU input outside CUDA model helper");
  rejects([&]{rpb::unit_grouped_odd_relations(odd.narrow(1,0,11));},"wrong odd width");
  rejects([&]{rpb::unit_grouped_odd_relations(odd.to(torch::kFloat64));},"wrong dtype");
  auto invalid=odd.detach().clone();invalid[0][0]=std::numeric_limits<float>::quiet_NaN();
  rejects([&]{rpb::unit_grouped_odd_relations(invalid);},"nonfinite relation input");
}
void initial_contract(rpb::FixedPriorTemporalRelationModel &v17,rpb::UnitTemporalRelationModel &v18) {
  const auto a=v17->named_parameters(),b=v18->named_parameters();check(a.keys()==b.keys(),"v17/v18 exact registration order");
  int64_t registered=0,trainable=0,frozen=0;
  for(const auto &p:b) {
    exact(p.value(),a[p.key()],"no extra initializer and exact common parameter values");
    check(p.value().is_cuda(),"all model parameters CUDA");registered+=p.value().numel();
    if(p.key()=="grouped_odd_relation_projection.weight") {
      check(!p.value().requires_grad()&&!p.value().grad().defined()&&p.value().sizes()==torch::IntArrayRef({3,4,36}),"same frozen432 geometry");
      frozen+=p.value().numel();
    }else{check(p.value().requires_grad(),"shape/backbone/decoder trainable");trainable+=p.value().numel();}
  }
  check(registered==226877&&trainable==226445&&frozen==432&&registered==rpb::kUnitTemporalRelationParameterCount&&
      trainable==rpb::kUnitTemporalRelationTrainableParameterCount&&frozen==rpb::kUnitTemporalRelationFrozenParameterCount,"unchanged exact capacity");
  const auto ab=v17->named_buffers(),bb=v18->named_buffers();check(ab.keys()==bb.keys(),"same buffer registration order");
  for(const auto &p:ab)exact(p.value(),bb[p.key()],"initial buffers exact");
  v18->validate_fixed_prior();
}
void cuda_contract() {
  const auto c=configuration();auto raw=input(c,3);
  raw.observed[0][2].narrow(0,13,3).fill_(false);raw.data=torch::where(raw.observed,raw.data,torch::zeros_like(raw.data));
  const auto scaler=rpb::fit_scaler(raw,c);const auto mean=scaler.mean.clone(),scale=scaler.scale.clone();const auto x=scaler.transform(raw,c);
  torch::manual_seed(18101);rpb::FixedPriorTemporalRelationModel v17(c);
  const auto cpu_rng=torch::rand({8}),cuda_rng=torch::rand({8},torch::TensorOptions().device(torch::kCUDA));
  torch::manual_seed(18101);rpb::UnitTemporalRelationModel v18(c);
  exact(cpu_rng,torch::rand({8}),"unchanged v17 CPU constructor RNG");
  exact(cuda_rng,torch::rand({8},torch::TensorOptions().device(torch::kCUDA)),"unchanged v17 CUDA constructor RNG");
  v17->eval();v18->eval();initial_contract(v17,v18);
  const auto old=v17->encode(x),initial=v18->encode(x);
  exact(initial.z_contextual_global.narrow(1,0,20),old.z_contextual_global.narrow(1,0,20),"shape20 exactly unchanged");
  close(initial.z_contextual_global.narrow(1,20,12),scalar_reference(old.z_contextual_global.narrow(1,20,12)),"initial actual-native scalar norm reference",2e-6,1e-7);
  exact(initial.z_local,old.z_local,"independent local diagnostic unchanged");
  exact(initial.z_contextual,old.z_contextual,"contextual channel diagnostics unchanged");
  const auto odd_initial=initial.z_contextual_global.narrow(1,20,12).detach().clone();
  const auto norms=odd_initial.reshape({3,3,4}).square().sum(-1);
  close(norms,torch::ones_like(norms),"nonzero pair relation vectors have unitL2",2e-6,1e-6);
  const auto q=rpb::make_training_mask(x.observed,c,18103);
  const auto masks=rpb::mask_from_hidden(x.observed,q.hidden,c);
  const rpb::Input visible{x.data,masks.visible,x.channel_ids,x.endpoints,x.sampling_interval};
  const auto new_q=v18->forward(x,q.hidden),old_q=v17->forward(x,q.hidden);
  exact(new_q.encoding.z_contextual_global,v18->encode(visible).z_contextual_global,"public forward uses derived normalization, never base nonvirtual encode");
  exact(new_q.reconstruction,v18->decode(new_q.encoding.z_contextual_global,x.channel_ids),"sole original decoder receives normalized32");
  exact(new_q.target_counts,old_q.target_counts,"original query support unchanged");
  auto changed=x;changed.data=torch::where(q.target,x.data+1000.,x.data);
  exact(v18->forward(changed,q.hidden).encoding.z_contextual_global,new_q.encoding.z_contextual_global,"hidden targets cannot enter shape or relation direction");
  auto poison=x;poison.data=torch::where(x.observed,x.data,torch::full_like(x.data,std::numeric_limits<float>::quiet_NaN()));
  exact(v18->encode(poison).z_contextual_global,initial.z_contextual_global,"hidden NaNs isolated by actual mask");
  const auto prior=v18->grouped_odd_relation_projection->weight.detach().clone();
  const auto shape_initial=v18->native_shape_projection->weight.detach().clone();
  const auto backbone_weight=v18->named_parameters()["backbone.patch_projection.weight"];
  const auto decoder_weight=v18->named_parameters()["backbone.decoder_first.weight"];
  const auto backbone_initial=backbone_weight.detach().clone(),decoder_initial=decoder_weight.detach().clone();
  torch::optim::AdamW optimizer(v18->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for(int64_t step=0;step<4;++step) {
    const auto train_mask=rpb::make_training_mask(x.observed,c,18110+step);
    const auto context=rpb::context_deletion::make_plan(train_mask,x.channel_ids,c,18107,step,rpb::ContextDeletionRecipe::coordinate15_v1);
    const rpb::Input current{torch::where(context.visible,x.data,torch::zeros_like(x.data)),context.visible,x.channel_ids,x.endpoints,x.sampling_interval};
    optimizer.zero_grad();const auto e=v18->encode(current);const auto z=rpb::compact_reconstruction_export(e,c);
    const auto prediction=v18->decode(z,x.channel_ids);
    const auto loss=rpb::hierarchical_huber(prediction,x.data.detach(),train_mask.target,train_mask.eligible_channels,1.);
    check(z.sizes()==torch::IntArrayRef({3,32})&&z.is_cuda()&&loss.loss.is_cuda()&&torch::isfinite(loss.loss).item<bool>(),"original CUDA waveform objective through same32");
    loss.loss.backward();check(!v18->grouped_odd_relation_projection->weight.grad().defined(),"no gradient for frozen odd432");
    for(const auto &w:{v18->native_shape_projection->weight,backbone_weight,decoder_weight}) {
      finite(w.grad(),"finite trainable CUDA gradients");check(w.grad().is_cuda()&&w.grad().abs().sum().item<double>()>0,"nonzero shape/backbone/decoder learning");
    }
    torch::nn::utils::clip_grad_norm_(v18->parameters(),1.,2.,true);optimizer.step();v18->validate_fixed_prior();
    exact(v18->grouped_odd_relation_projection->weight,prior,"frozen prior unaffected by AdamW/weight decay");
    exact(v18->encode(x).z_contextual_global.narrow(1,20,12),odd_initial,"normalized odd coordinates exactly fixed after each update");
  }
  check(!torch::equal(v18->native_shape_projection->weight,shape_initial)&&!torch::equal(backbone_weight,backbone_initial)&&
      !torch::equal(decoder_weight,decoder_initial),"four actual CUDA updates change trainable graph");
  check(optimizer.state().find(v18->grouped_odd_relation_projection->weight.unsafeGetTensorImpl())==optimizer.state().end(),"no fabricated frozen AdamW state");
  for(const auto &w:{v18->native_shape_projection->weight,backbone_weight,decoder_weight}) {
    const auto it=optimizer.state().find(w.unsafeGetTensorImpl());check(it!=optimizer.state().end(),"trainable AdamW state exists");
    const auto *adam=dynamic_cast<const torch::optim::AdamWParamState *>(it->second.get());
    check(adam&&adam->step()==4&&adam->exp_avg().is_cuda(),"four-step CUDA moments retained");
  }
  exact(scaler.mean,mean,"TRAIN scaler mean frozen");exact(scaler.scale,scale,"TRAIN scaler scale frozen");
  {
    torch::NoGradGuard guard;auto reverse=x;reverse.data=x.data.flip({2});reverse.observed=x.observed.flip({2});
    close(v18->encode(reverse).z_contextual_global.narrow(1,20,12),-odd_initial,"unit odd block reverses with values and masks",4e-5,2e-5);
    const auto order=torch::tensor({2,0,1},torch::kInt64).to(torch::kCUDA);auto perm=x;
    perm.data=x.data.index_select(1,order);perm.observed=x.observed.index_select(1,order);perm.channel_ids=x.channel_ids.to(torch::kCUDA).index_select(0,order);
    close(v18->encode(perm).z_contextual_global,v18->encode(x).z_contextual_global,"semantic permutation preserves normalized native32",4e-5,2e-5);
    auto single=poison;single.observed=torch::zeros_like(x.observed);single.observed.select(1,0).copy_(x.observed.select(1,0));
    const auto one=v18->encode(single);check(one.sample_valid_mask.all().item<bool>(),"single-channel context can remain valid");
    exact(one.z_contextual_global.narrow(1,20,12),torch::zeros_like(odd_initial),"no legal cross-channel pair produces exactzero relations");
    auto absent=poison;absent.observed=torch::zeros_like(x.observed);const auto empty=v18->encode(absent);
    check(!empty.sample_valid_mask.any().item<bool>(),"all absent remains invalid");
    exact(empty.z_contextual_global,torch::zeros_like(empty.z_contextual_global),"all-absent native32 exactzero");
  }
  torch::Tensor native;
  {torch::NoGradGuard guard;native=v18->encode(x).z_contextual_global.detach().clone();}
  std::stringstream bytes;torch::serialize::OutputArchive out;v18->save(out);out.save_to(bytes);
  rpb::UnitTemporalRelationModel restored(c);restored->grouped_odd_relation_projection->weight.set_requires_grad(true);
  std::shared_ptr<torch::nn::Module> erased=restored.ptr();torch::serialize::InputArchive archive;archive.load_from(bytes,torch::kCUDA);erased->load(archive);
  restored->eval();restored->validate_fixed_prior();check(!restored->grouped_odd_relation_projection->weight.requires_grad(),"type-erased inherited load refreezes literal prior");
  {torch::NoGradGuard guard;exact(restored->encode(x).z_contextual_global,native,"trained save/load exact normalized32");}
  const auto before=v18->named_parameters(),after=restored->named_parameters();check(before.keys()==after.keys(),"save/load parameter order preserved");
  for(const auto &p:before)exact(p.value(),after[p.key()],"save/load parameter equality");
  rpb::UnitTemporalRelationModel corrupt(c);
  {torch::NoGradGuard guard;corrupt->grouped_odd_relation_projection->weight[0][0][0]=0;}
  std::stringstream invalid;torch::serialize::OutputArchive invalid_out;corrupt->save(invalid_out);invalid_out.save_to(invalid);
  torch::serialize::InputArchive invalid_archive;invalid_archive.load_from(invalid,torch::kCUDA);
  rejects([&]{restored->load(invalid_archive);},"corrupted literal prior rejected");
}
} // namespace

int main() try {
  check(torch::cuda::is_available(),"unit relation admission requires actual CUDA");torch::set_num_threads(1);
  normalization_contract();cuda_contract();
  std::cout<<"Unit temporal relation CUDA admission passed\n"<<UNIT_RELATION_SCREEN_SOURCE_ID<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
