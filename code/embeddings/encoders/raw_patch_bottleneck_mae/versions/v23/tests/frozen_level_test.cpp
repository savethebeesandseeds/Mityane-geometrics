// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
#include "embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_frozen_level.h"
#include <torch/cuda.h>
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace {
namespace r=embedding::encoders::raw_patch_bottleneck_mae;
namespace tr=r::consecutive_change_training;
void check(bool v,const char *why) { TORCH_CHECK(v,"[frozen level test] ",why); }
template<class F> void rejects(F f,const char *why) {
  bool refused=false;try { f(); }catch(const std::exception &) { refused=true; }check(refused,why);
}
struct Temporary {
  std::filesystem::path path;
  Temporary() {
    auto p=(std::filesystem::temp_directory_path()/"rpb-frozen-level-XXXXXX").string();
    std::vector<char> b(p.begin(),p.end());b.push_back(0);check(::mkdtemp(b.data()),"exclusive fixture");path=b.data();
  }
  ~Temporary() { std::error_code e;std::filesystem::remove_all(path,e); }
};
torch::Tensor update(const r::Input &input,r::ConsecutiveChangeModel &model,
    torch::optim::AdamW &optimizer,tr::TrainingProgress &progress,const tr::FrozenLevelGuard &guard) {
  model->train();optimizer.zero_grad();
  const auto masks=r::configurable_spectral_make_training_mask(input.observed,model->config(),
      tr::training_counter_seed(progress.mask_seed,progress.attempted_batches));
  const auto result=model->forward(input,masks.hidden);
  check(result.loss.is_cuda() && torch::isfinite(result.loss).item<bool>(),"finite CUDA loss");
  result.loss.backward();const auto g=model->change_weight.grad();
  check(g.defined() && g.is_cuda() && torch::isfinite(g).all().item<bool>() &&
      g.abs().sum().item<double>()>0,"nonzero finite CUDA change gradient");
  guard.discard_gradients(model,optimizer);
  const auto before=model->change_weight.detach().clone();
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer.step();
  check(!torch::equal(before,model->change_weight.detach()),"change projection moved");
  ++progress.completed_updates;++progress.attempted_batches;guard.validate(model,optimizer);
  return result.loss.detach().clone();
}
}
int main() try {
  check(torch::cuda::is_available(),"CUDA required");torch::set_num_threads(1);torch::set_num_interop_threads(1);
  auto c=r::default_content_ordered_relation_config();c.channel_ids={0,1,2};c.history_length=32;c.encoder_width=32;
  c.num_layers=1;c.feedforward_width=64;c.decoder_hidden_width=64;
  torch::manual_seed(941001);r::ContentOrderedRelationModel parent(c);
  { torch::NoGradGuard g;parent->relation_projection->weight.fill_(.003); }
  auto t=torch::arange(32,torch::kFloat64).view({1,1,32,1});
  auto f=torch::arange(9,torch::kFloat64).view({1,1,1,9});
  auto ch=torch::arange(3,torch::kFloat64).view({1,3,1,1});
  auto b=torch::arange(3,torch::kFloat64).view({3,1,1,1});
  auto values=(.13*t+.11*f+.2*ch+.3*b).sin()+.02*t;
  r::Input raw{values,torch::ones(values.sizes(),torch::kBool),torch::tensor(c.channel_ids,torch::kInt64),
      torch::tensor({1000.,2000.,3000.},torch::kFloat64),c.sampling_interval};
  const auto scaler=r::fit_configurable_spectral_scaler(raw,c);
  const auto input=r::transform_configurable_spectral_input(scaler,raw,c);Temporary temp;
  const auto parent_file=(temp.path/"parent.pt").string();
  r::save_content_ordered_relation_checkpoint(parent_file,parent,scaler,2048);
  auto loaded=r::load_content_ordered_relation_checkpoint(parent_file,c.device,&c);
  torch::manual_seed(942001);r::ConsecutiveChangeModel model(c);tr::warm_start_level(model,loaded.model);
  { torch::NoGradGuard g;model->eval();
    check(torch::equal(model->encode(input).z_contextual_global,loaded.model->encode(input).z_contextual_global),
        "zero-start native equals typed warm parent exactly"); }
  auto wrong=c;wrong.history_length=40;r::ContentOrderedRelationModel mismatch(wrong);
  rejects([&] { tr::warm_start_level(model,mismatch); },"wrong geometry rejected");
  torch::optim::AdamW optimizer(model->parameters(),tr::content_training_adamw_options());
  tr::FrozenLevelGuard guard(model);tr::TrainingProgress progress{0,0,942002,942003,942004};
  tr::TrainingIdentity identity{"artificial-input","artificial-split","frozen-level-v1;parent-fixture2048"};
  update(input,model,optimizer,progress,guard);
  rejects([&] { tr::warm_start_level(model,parent); },"nonzero branch cannot silently restart");
  const auto state=(temp.path/"training.pt").string();
  tr::save_content_training_state(state,model,scaler,optimizer,identity,progress);
  auto restored=tr::load_content_training_state(state,c,identity,scaler.identity());
  tr::FrozenLevelGuard restored_guard(restored.model);restored_guard.validate(restored.model,*restored.optimizer);
  const auto a=update(input,model,optimizer,progress,guard);
  const auto z=update(input,restored.model,*restored.optimizer,restored.progress,restored_guard);
  check(torch::equal(a,z),"resumed next CUDA loss exact");
  const auto first=model->named_parameters(),second=restored.model->named_parameters();
  for(const auto &p:first)check(torch::equal(p.value(),second[p.key()]),"resumed next weights exact");
  const auto inference=(temp.path/"inference.pt").string();
  r::save_consecutive_change_checkpoint(inference,model,scaler,2);
  auto reload=r::load_consecutive_change_checkpoint(inference,c.device,&c);
  { torch::NoGradGuard g;model->eval();
    check(torch::equal(model->encode(input).z_contextual_global,reload.model->encode(input).z_contextual_global),
        "same typed v23 inference arithmetic reloads exactly"); }
  { torch::NoGradGuard g;model->level->relation_projection->weight.add_(.001); }
  rejects([&] { guard.validate(model,optimizer); },"protected state mutation detected");
  std::cout<<"PASS frozen level: typed parent/zero-start parity, config/nonzero restart guards, CUDA branch updates, exact protected parameters/buffers, no protected AdamW moments/decay, typed training resume/next loss and weights, inference reload, mutation detection\n";
  return 0;
}catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
