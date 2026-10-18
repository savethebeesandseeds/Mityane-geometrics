// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_frozen_level.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_settings.h"
#include <torch/cuda.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <unistd.h>
namespace {
namespace r=embedding::encoders::raw_patch_bottleneck_mae;namespace tr=r::separated_change_training;
void check(bool v,const char *why){TORCH_CHECK(v,"[separate test] ",why);}
template<class F>void rejects(F f,const char *why){bool caught=false;try{f();}catch(const std::exception &){caught=true;}check(caught,why);}
struct Temporary{std::filesystem::path path;Temporary(){auto p=(std::filesystem::temp_directory_path()/"rpb-separate-XXXXXX").string();
 std::vector<char>b(p.begin(),p.end());b.push_back(0);check(::mkdtemp(b.data()),"exclusive fixture");path=b.data();}
 ~Temporary(){std::error_code e;std::filesystem::remove_all(path,e);}};
torch::Tensor update(const r::Input &input,r::SeparatedChangeModel &model,torch::optim::AdamW &optimizer,
    tr::TrainingProgress &progress,const tr::FrozenLevelGuard &guard){
  model->train();optimizer.zero_grad();const auto &c=model->config().level;
  const auto masks=r::configurable_spectral_make_training_mask(input.observed,c,tr::training_counter_seed(progress.mask_seed,progress.attempted_batches));
  const auto out=model->forward(input,masks.hidden);check(out.loss.is_cuda() && torch::isfinite(out.loss).item<bool>(),"finite CUDA loss");
  out.loss.backward();const auto encoder=model->change_projection.grad(),decoder=model->change_decoder.grad();
  for(const auto &g:{encoder,decoder})check(g.defined() && g.is_cuda() && torch::isfinite(g).all().item<bool>(),"finite CUDA branch gradients");
  if(progress.completed_updates==0)check(encoder.eq(0).all().item<bool>() && decoder.abs().sum().item<double>()>0,"zero decoder opens through actual first CUDA decoder gradient");
  else check(encoder.abs().sum().item<double>()>0 && decoder.abs().sum().item<double>()>0,"both branch gradients active after first step");
  for(const auto &p:model->level->parameters())check(!p.grad().defined(),"level absent from gradient graph");
  guard.discard_gradients(model,optimizer);const auto old_encoder=model->change_projection.detach().clone(),old_decoder=model->change_decoder.detach().clone();
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer.step();
  check(!torch::equal(old_decoder,model->change_decoder.detach()),"CUDA decoder weights moved");
  if(progress.completed_updates>0)check(!torch::equal(old_encoder,model->change_projection.detach()),"CUDA projection weights moved with nonzero gradient");
  ++progress.completed_updates;++progress.attempted_batches;guard.validate(model,optimizer);return out.loss.detach().clone();
}
}
int main()try{
  check(torch::cuda::is_available(),"CUDA required");torch::set_num_threads(1);torch::set_num_interop_threads(1);
  auto c=r::default_content_ordered_relation_config();c.channel_ids={7,41,99};c.history_length=32;
  c.encoder_width=32;c.num_layers=1;c.feedforward_width=64;c.decoder_hidden_width=64;
  r::SeparatedChangeConfig full{c,8};auto invalid=full;invalid.change_width=0;
  rejects([&]{r::validate_separated_change_config(invalid);},"zero change size rejected");
  torch::manual_seed(941001);r::ContentOrderedRelationModel parent(c);
  {torch::NoGradGuard g;parent->relation_projection->weight.fill_(.003);}
  const auto t=torch::arange(32,torch::kFloat64).view({1,1,32,1}),f=torch::arange(9,torch::kFloat64).view({1,1,1,9});
  const auto ch=torch::arange(3,torch::kFloat64).view({1,3,1,1}),b=torch::arange(3,torch::kFloat64).view({3,1,1,1});
  auto values=(.13*t+.11*f+.2*ch+.3*b).sin()+.02*t;
  auto observed=torch::ones(values.sizes(),torch::kBool);observed[0][1][7][3]=false;observed[1][2][11][2]=false;
  r::Input raw{values,observed,torch::tensor(c.channel_ids,torch::kInt64),torch::tensor({1000.,2000.,3000.},torch::kFloat64),c.sampling_interval};
  const auto scaler=r::fit_configurable_spectral_scaler(raw,c);
  const auto input=r::transform_configurable_spectral_input(scaler,raw,c);
  check(r::consecutive_change_bank(input,c).counts[0][1][3].item<int64_t>()==29,
      "one missing tick removes both adjacent pairs without bridging");
  Temporary temp;const auto parent_file=(temp.path/"parent.pt").string();r::save_content_ordered_relation_checkpoint(parent_file,parent,scaler,2048);
  auto loaded=r::load_content_ordered_relation_checkpoint(parent_file,c.device,&c);
  torch::manual_seed(942001);r::SeparatedChangeModel model(full);tr::warm_start_level(model,loaded.model);
  auto optimizer=std::make_unique<torch::optim::AdamW>(model->parameters(),tr::content_training_adamw_options());tr::FrozenLevelGuard guard(model);
  {
    torch::NoGradGuard g;model->eval();const auto native=model->encode(input).z_contextual_global;
    const auto old=loaded.model->encode(input).z_contextual_global;
    check(native.sizes()==torch::IntArrayRef({3,40}) && torch::equal(native.narrow(1,0,32),old),"rank2 native40 with exact trained-parent prefix32");
    check(native.narrow(1,32,8).abs().sum().item<double>()>0,"initial separate projection exposes change information");
    check(torch::equal(model->decode(native,input.channel_ids),loaded.model->decode(old,input.channel_ids)),"zero residual decoder preserves initial reconstruction exactly");
    auto poisoned=input;poisoned.data=input.data.clone();poisoned.data.masked_fill_(input.observed.logical_not(),std::numeric_limits<float>::quiet_NaN());
    check(torch::equal(model->encode(poisoned).z_contextual_global,native),"unobserved NaN storage cannot affect either block");
    auto missing=poisoned;missing.observed=torch::zeros_like(input.observed);
    check(model->encode(missing).z_contextual_global.eq(0).all().item<bool>(),"all-missing output zero");
    auto partial=full;partial.level.history_length=33;r::SeparatedChangeModel partial_model(partial);
    r::Input partial_input{torch::cat({input.data,torch::zeros({3,3,1,9},input.data.options())},2),
      torch::cat({input.observed,torch::zeros({3,3,1,9},input.observed.options())},2),input.channel_ids,input.endpoints,input.sampling_interval};
    const auto out=partial_model->forward(partial_input,r::configurable_spectral_make_training_mask(partial_input.observed,partial.level,1101).hidden);
    check(out.reconstruction.sizes()==torch::IntArrayRef({3,3,33,9}) && torch::isfinite(out.loss).item<bool>(),"partial history retained by both decoder paths");
  }
  const auto settings=(temp.path/"settings.conf").string();{
    std::ofstream file(settings);file<<"ENCODER { MODEL_TAG=RPB-v24; LEVEL_EMBEDDING_SIZE=32; EMBEDDING_SIZE=40; HISTORY_LENGTH=32; CHANNEL_IDS=7,41,99; }\n";
  }
  const auto parsed=r::read_separated_change_settings(settings);check(parsed.change_width==8 && parsed.level.export_width==32,"programmatic level/native dimension settings");
  tr::TrainingIdentity identity{"artificial-source","artificial-split","v24-separate-8;parent-fixture2048"};tr::TrainingProgress progress{0,0,942002,942003,942004};
  update(input,model,*optimizer,progress,guard);
  rejects([&]{tr::warm_start_level(model,parent);},"trained residual cannot silently restart");
  const auto state=(temp.path/"training.pt").string();tr::save_training_state(state,model,scaler,*optimizer,identity,progress);
  auto restored=tr::load_training_state(state,full,identity,scaler.identity());tr::FrozenLevelGuard resumed_guard(restored.model);
  const auto a=update(input,model,*optimizer,progress,guard),z=update(input,restored.model,*restored.optimizer,restored.progress,resumed_guard);
  check(torch::equal(a,z),"resumed second CUDA loss exact");
  const auto first=model->named_parameters(),second=restored.model->named_parameters();
  for(const auto &p:first)check(torch::equal(p.value(),second[p.key()]),"resumed second CUDA weights exact");
  {
    torch::NoGradGuard g;model->eval();const auto native=model->encode(input).z_contextual_global;
    check(torch::equal(native.narrow(1,0,32),loaded.model->encode(input).z_contextual_global),"trained prefix remains exact");
    const auto perm=torch::tensor({2,0,1},torch::TensorOptions().device(c.device).dtype(torch::kInt64));
    r::Input shuffled{input.data.index_select(1,perm),input.observed.index_select(1,perm),input.channel_ids.index_select(0,perm),input.endpoints,input.sampling_interval};
    const auto shuffled_native=model->encode(shuffled).z_contextual_global;
    const auto parent_shuffled=loaded.model->encode(shuffled).z_contextual_global;
    check(torch::equal(shuffled_native.narrow(1,0,32),parent_shuffled),"permuted prefix equals the parent in that same physical layout exactly");
    check(torch::equal(native.narrow(1,32,8),shuffled_native.narrow(1,32,8)),"new change block is exactly permutation invariant");
    // Match the existing v20 learned-path contract; physical batch-row reorder
    // can change GPU rounding. The new block and each parent-prefix parity stay exact.
    check(torch::allclose(native.narrow(1,0,32),parent_shuffled,1e-5,1e-6),"inherited learned prefix follows existing semantic permutation tolerance");
    const auto reordered=model->decode(shuffled_native,shuffled.channel_ids),expected=model->decode(native,input.channel_ids).index_select(1,perm);
    check(torch::allclose(reordered,expected,1e-5,1e-6),"combined decoder follows inherited semantic permutation tolerance");
    std::cout<<"permutation_prefix_max_abs="<<(native.narrow(1,0,32)-parent_shuffled).abs().max().item<double>()
      <<" permutation_decoder_max_abs="<<(reordered-expected).abs().max().item<double>()<<'\n';
  }
  const auto inference=(temp.path/"inference.pt").string();r::save_separated_change_checkpoint(inference,model,scaler,2);
  auto reload=r::load_separated_change_checkpoint(inference,c.device,&full);
  {torch::NoGradGuard g;check(torch::equal(model->encode(input).z_contextual_global,reload.model->encode(input).z_contextual_global),"typed v24 native reload exact");}
  rejects([&]{r::load_consecutive_change_checkpoint(inference,c.device,&c);},"v23 reader rejects v24");
  rejects([&]{r::load_separated_change_checkpoint(parent_file,c.device,&full);},"v24 reader rejects v22");
  rejects([&]{r::save_separated_change_checkpoint(inference,model,scaler,2);},"immutable v24 checkpoint refuses overwrite");
  {torch::NoGradGuard g;model->level->relation_projection->weight.add_(.001);}
  rejects([&]{guard.validate(model,*optimizer);},"protected state corruption detected");
  std::cout<<"PASS separate changes: native40/exact parent32, zero initial residual, masked poison/missingness, partial history, settings dimensions, semantic encoder/decoder order, structural no-level-gradients, actual CUDA decoder then projection learning, protected state/no moments, typed immutable archives/refusal, exact resumed next loss/weights\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
