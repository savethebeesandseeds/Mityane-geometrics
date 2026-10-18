// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
#include "embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_training_state.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include <torch/cuda.h>
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace {
namespace r=embedding::encoders::raw_patch_bottleneck_mae;
namespace train=r::consecutive_change_training;
void check(bool v,const char *why) { TORCH_CHECK(v,"[change test] ",why); }
template<class F> void rejects(F f,const char *why) { bool refused=false;try { f(); }catch(const std::exception &) { refused=true; }check(refused,why); }
struct Temporary {
  std::filesystem::path path;
  Temporary() { auto p=(std::filesystem::temp_directory_path()/"rpb-v23-check-XXXXXX").string();
    std::vector<char> b(p.begin(),p.end());b.push_back(0);check(::mkdtemp(b.data()),"exclusive fixture");path=b.data(); }
  ~Temporary() { std::error_code e;std::filesystem::remove_all(path,e); }
};
torch::Tensor update(const r::Input &population,r::ConsecutiveChangeModel &model,
    torch::optim::AdamW &optimizer,train::TrainingProgress &progress) {
  const auto &c=model->config();
  const auto masks=r::configurable_spectral_make_training_mask(population.observed,c,
      train::training_counter_seed(progress.mask_seed,progress.attempted_batches));
  const auto context=r::context_deletion::make_plan(masks,population.channel_ids,c,
      progress.context_seed,progress.attempted_batches,r::ContextDeletionRecipe::coordinate15_v1);
  const r::Input visible{torch::where(context.visible,population.data,torch::zeros_like(population.data)),
      context.visible,population.channel_ids,population.endpoints,population.sampling_interval};
  model->train();optimizer.zero_grad();
  const auto predicted=model->decode(model->encode(visible).z_contextual_global,population.channel_ids);
  const auto loss=r::hierarchical_huber(predicted,population.data.detach(),masks.target,masks.eligible_channels,1.);
  check(loss.loss.is_cuda() && torch::isfinite(loss.loss).item<bool>(),"actual finite CUDA loss");
  loss.loss.backward();const auto gradient=model->change_weight.grad();
  check(gradient.defined() && gradient.is_cuda() && torch::isfinite(gradient).all().item<bool>() &&
      gradient.abs().sum().item<double>()>0,"active consecutive-change CUDA gradient");
  const auto before=model->change_weight.detach().clone();
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer.step();
  check(!torch::equal(before,model->change_weight.detach()),"change weights move on CUDA");
  ++progress.completed_updates;++progress.attempted_batches;return loss.loss.detach().clone();
}
}

int main() try {
  check(torch::cuda::is_available(),"GPU required");torch::set_num_threads(1);torch::set_num_interop_threads(1);
  auto single=r::default_content_ordered_relation_config();single.channel_count=1;single.channel_ids={7};
  single.input_width=1;single.history_length=17;single.encoder_width=16;single.num_heads=4;
  single.num_layers=1;single.feedforward_width=32;single.decoder_hidden_width=32;single.export_width=8;
  auto x=torch::arange(17,torch::kFloat32).view({1,1,17,1});auto observed=torch::ones(x.sizes(),torch::kBool);
  r::Input ramp{x,observed,torch::tensor({7},torch::kInt64),torch::tensor({1000.},torch::kFloat64),single.sampling_interval};
  auto bank=r::consecutive_change_bank(ramp,single);
  check(torch::equal(bank.values,torch::ones({1,10},bank.values.options())) && bank.counts.item<int64_t>()==16,
      "exact ramp mean/RMS/quarter moments");
  auto gap=ramp;gap.observed=observed.clone();gap.observed[0][0][4][0]=false;
  gap.data=x.clone();gap.data[0][0][4][0]=std::numeric_limits<float>::quiet_NaN();
  const auto gapped=r::consecutive_change_bank(gap,single);
  check(gapped.counts.item<int64_t>()==14,"gap removes both adjacent pairs, never bridges");
  gap.data[0][0][4][0]=1e30f;
  check(torch::equal(gapped.values,r::consecutive_change_bank(gap,single).values),"masked storage cannot leak");
  auto constant=ramp;constant.data=torch::ones(x.sizes(),torch::TensorOptions().device(single.device).dtype(torch::kFloat32).requires_grad(true));
  auto constant_bank=r::consecutive_change_bank(constant,single);constant_bank.values.sum().backward();
  check(constant_bank.values.eq(0).all().item<bool>() && torch::isfinite(constant.data.grad()).all().item<bool>(),
      "zero-energy branch gives zero finite moments and gradients");
  auto missing=ramp;missing.observed=torch::zeros_like(observed);
  check(r::consecutive_change_bank(missing,single).values.eq(0).all().item<bool>(),"all missing branch is zero");
  auto tiny=single;tiny.history_length=1;auto one=ramp;one.data=x.narrow(2,0,1);one.observed=observed.narrow(2,0,1);
  rejects([&] { r::consecutive_change_bank(one,tiny); },"inherited spectral minimum rejects H1");

  auto c=r::default_content_ordered_relation_config();c.history_length=32;c.channel_ids={7,41,99};
  c.encoder_width=32;c.num_layers=1;c.feedforward_width=64;c.decoder_hidden_width=64;
  auto t=torch::arange(c.history_length,torch::kFloat64).view({1,1,c.history_length,1});
  auto f=torch::arange(c.input_width,torch::kFloat64).view({1,1,1,c.input_width});
  auto ch=torch::arange(3,torch::kFloat64).view({1,3,1,1});auto b=torch::arange(3,torch::kFloat64).view({3,1,1,1});
  auto values=(t*.13+f*.07+ch*.3+b*.2).sin()+.03*t+.4*(t*.27-ch*.4).cos();
  auto mask=torch::ones(values.sizes(),torch::kBool);mask[0][1][7]=false;mask[1][2][11][3]=false;
  r::Input raw{values,mask,torch::tensor(c.channel_ids,torch::kInt64),torch::tensor({1000.,2000.,3000.},torch::kFloat64),c.sampling_interval};
  const auto scaler=r::fit_configurable_spectral_scaler(raw,c);const auto normalized=r::transform_configurable_spectral_input(scaler,raw,c);
  {
    auto partial_config=c;partial_config.history_length=33;
    const r::Input partial{torch::cat({normalized.data,torch::zeros({3,3,1,9},normalized.data.options())},2),
        torch::cat({normalized.observed,torch::zeros({3,3,1,9},normalized.observed.options())},2),
        normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    r::ConsecutiveChangeModel partial_model(partial_config);torch::NoGradGuard guard;
    const auto partial_mask=r::configurable_spectral_make_training_mask(partial.observed,partial_config,1101);
    const auto partial_output=partial_model->forward(partial,partial_mask.hidden);
    check(partial_output.encoding.z_contextual_global.sizes()==torch::IntArrayRef({3,32}) &&
        partial_output.reconstruction.size(2)==33 && torch::isfinite(partial_output.loss).item<bool>(),
        "partial-patch inference and masked waveform forward retain logical history");
  }
  const auto permutation=torch::tensor({2,0,1},torch::TensorOptions().device(c.device).dtype(torch::kInt64));
  const r::Input reordered{normalized.data.index_select(1,permutation),normalized.observed.index_select(1,permutation),
      normalized.channel_ids.index_select(0,permutation),normalized.endpoints,normalized.sampling_interval};
  check(torch::equal(r::consecutive_change_bank(normalized,c).values,r::consecutive_change_bank(reordered,c).values),
      "change bank follows canonical semantic IDs");
  torch::manual_seed(942001);r::ContentOrderedRelationModel original(c);
  torch::manual_seed(942001);r::ConsecutiveChangeModel model(c);
  {
    torch::NoGradGuard guard;original->eval();model->eval();
    check(torch::equal(original->encode(normalized).z_contextual_global,model->encode(normalized).z_contextual_global),
        "zero-start native equals equally initialized v22 exactly");
  }
  auto optimizer=std::make_unique<torch::optim::AdamW>(model->parameters(),train::content_training_adamw_options());
  train::TrainingProgress progress{0,0,942002,942003,942004};
  train::TrainingIdentity identity{"artificial-input","artificial-split","change-candidate-v1"};
  update(normalized,model,*optimizer,progress);Temporary temp;
  const auto inference=(temp.path/"inference.pt").string(),state=(temp.path/"state.pt").string();
  r::save_consecutive_change_checkpoint(inference,model,scaler,1);
  train::save_content_training_state(state,model,scaler,*optimizer,identity,progress);
  auto loaded=r::load_consecutive_change_checkpoint(inference,c.device,&c);
  {
    torch::NoGradGuard guard;model->eval();loaded.model->eval();
    check(torch::equal(model->encode(normalized).z_contextual_global,loaded.model->encode(normalized).z_contextual_global),
        "typed inference reload retains exact native");
  }
  rejects([&] { r::load_content_ordered_relation_checkpoint(inference,c.device,&c); },"v22 refuses v23 archive");
  rejects([&] { r::save_consecutive_change_checkpoint(inference,model,scaler,1); },"immutable checkpoints refuse overwrite");
  auto restored=train::load_content_training_state(state,c,identity,scaler.identity());
  const auto loss_a=update(normalized,model,*optimizer,progress);
  const auto loss_b=update(normalized,restored.model,*restored.optimizer,restored.progress);
  check(torch::equal(loss_a,loss_b),"next resumed CUDA loss exact");
  const auto a=model->named_parameters(),z=restored.model->named_parameters();
  for(size_t j=0;j<a.size();++j) check(a[j].key()==z[j].key() && torch::equal(a[j].value(),z[j].value()),
      "next resumed CUDA weights exact");
  model->validate_state();model->validate_fixed_prior();
  std::cout<<"PASS consecutive changes: adjacent masks/gaps/poison, zero-energy gradients, short-history rejection/partial patches, semantic order, v22 zero-start parity, actual CUDA learning, typed immutable reload and exact next-update resume\n";
  return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
