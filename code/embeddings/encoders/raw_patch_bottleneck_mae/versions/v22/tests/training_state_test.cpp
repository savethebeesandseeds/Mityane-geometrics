// Copyright (c) 2026 Waajacu. All rights reserved.
// See ../../../../../../../LICENSE for the copyright and permission notice.
// Bounded artificial CUDA resume witness; no market or quality measurement.
#include "embedding/encoders/raw_patch_bottleneck_mae/training_state.h"
#include <torch/cuda.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <unistd.h>

namespace {
namespace r = embedding::encoders::raw_patch_bottleneck_mae;
namespace fs = std::filesystem;
void check(bool c, const char *message) { TORCH_CHECK(c, "[training state test] ", message); }
template<class F> void rejects(F action, const char *message) {
  bool refused = false;
  try { action(); } catch (const std::exception &) { refused = true; }
  check(refused, message);
}
bool exact(const torch::Tensor &a, const torch::Tensor &b) {
  if (!a.defined() || !b.defined()) return a.defined() == b.defined();
  const auto x = a.detach().to(torch::kCPU).contiguous();
  const auto y = b.detach().to(torch::kCPU).contiguous();
  return x.scalar_type() == y.scalar_type() && x.sizes() == y.sizes() &&
      (!x.numel() || !std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.numel()*x.element_size()));
}
struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    const auto pattern = (fs::temp_directory_path()/"rpb-v22-resume-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
    const auto created = ::mkdtemp(name.data()); check(created, "private fixture directory");
    path = created;
  }
  ~TemporaryDirectory() { std::error_code e; fs::remove_all(path, e); }
};
std::vector<char> bytes(const fs::path &path) {
  std::ifstream file(path, std::ios::binary); check(file.good(), "checkpoint bytes readable");
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
r::Input population(const r::Config &c) {
  const auto o = torch::TensorOptions().dtype(torch::kFloat64);
  const auto t = torch::arange(c.history_length,o).view({1,1,c.history_length,1});
  const auto channel = torch::arange(c.channel_count,o).view({1,c.channel_count,1,1});
  const auto feature = torch::arange(c.input_width,o).view({1,1,1,c.input_width});
  const auto row = torch::arange(6,o).view({6,1,1,1});
  auto values = (t*.07+channel*.13+feature*.17+row*.23).sin() +
      .4*(t*.31-channel*.17+feature*.11).cos() + feature*.03 + row*.1;
  auto observed = torch::ones(values.sizes(), torch::kBool);
  observed[0][1][7] = false; observed[1][2][11][8] = false;
  return {torch::where(observed,values,torch::zeros_like(values)), observed,
      torch::tensor(c.channel_ids,torch::kInt64), torch::arange(6,o)+1700000000., c.sampling_interval};
}
struct UpdateReceipt {
  torch::Tensor rows, hidden, visible, loss;
  std::vector<torch::Tensor> gradients;
};
UpdateReceipt update(const r::Input &p, r::ContentOrderedRelationModel &model,
    torch::optim::AdamW &optimizer, r::TrainingProgress &progress) {
  const auto &c = model->config();
  std::mt19937_64 row_rng(r::training_counter_seed(progress.row_seed,progress.attempted_batches));
  std::uniform_int_distribution<int64_t> select(0,p.data.size(0)-1);
  auto rows = torch::tensor(std::vector<int64_t>{select(row_rng),select(row_rng)},torch::kInt64);
  auto device_rows = rows.to(c.device);
  r::Input batch{p.data.index_select(0,device_rows),p.observed.index_select(0,device_rows),
      p.channel_ids,p.endpoints.index_select(0,device_rows),p.sampling_interval};
  const auto masks = r::configurable_spectral_make_training_mask(batch.observed,c,
      r::training_counter_seed(progress.mask_seed,progress.attempted_batches));
  auto visible = masks.visible.to(torch::kCPU).contiguous();
  auto *bits = visible.data_ptr<bool>();
  std::mt19937_64 context_rng(r::training_counter_seed(progress.context_seed,progress.attempted_batches));
  std::uniform_real_distribution<double> erase(0.,1.);
  for (int64_t j=0;j<visible.numel();++j) if (bits[j] && erase(context_rng)<.15) bits[j]=false;
  visible = visible.to(c.device);
  const auto patch_support = r::configurable_spectral_pad_observed(visible,c)
      .reshape({2,c.channel_count,-1,c.patch_length,c.input_width}).any(-1).any(-1).sum(-1);
  check(patch_support.ge(2).all().item<bool>(), "artificial context retains two patches per channel");
  r::Input context{torch::where(visible,batch.data,torch::zeros_like(batch.data)),visible,
                   batch.channel_ids,batch.endpoints,batch.sampling_interval};
  model->train(); optimizer.zero_grad();
  const auto encoding = model->encode(context);
  const auto prediction = model->decode(encoding.z_contextual_global,batch.channel_ids);
  const auto losses = r::hierarchical_huber(prediction,batch.data.detach(),masks.target,
                                          masks.eligible_channels,c.huber_delta);
  check(losses.loss.is_cuda() && torch::isfinite(losses.loss).item<bool>() &&
            losses.eligible_example_count == 2, "actual finite eligible CUDA waveform loss");
  losses.loss.backward();
  const auto g = model->relation_projection->weight.grad();
  check(g.defined() && g.is_cuda() && torch::isfinite(g).all().item<bool>() &&
            g.abs().sum().item<double>()>0, "active content projection has nonzero CUDA gradient");
  UpdateReceipt receipt{rows,masks.hidden.clone(),visible.clone(),losses.loss.detach().clone(),{}};
  for (const auto &parameter : model->parameters()) {
    check(parameter.is_cuda(), "all model parameters execute on CUDA");
    if (parameter.grad().defined()) {
      check(parameter.grad().is_cuda() && torch::isfinite(parameter.grad()).all().item<bool>(),
            "all actual gradients are finite CUDA");
      receipt.gradients.push_back(parameter.grad().detach().clone());
    } else receipt.gradients.push_back({});
  }
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.); optimizer.step();
  ++progress.completed_updates; ++progress.attempted_batches;
  return receipt;
}
void same_models(const r::ContentOrderedRelationModel &a,const r::ContentOrderedRelationModel &b) {
  const auto x=a->named_parameters(), y=b->named_parameters();
  check(x.size()==y.size(), "model parameter count exact");
  for (size_t j=0;j<x.size();++j)
    check(x[j].key()==y[j].key() && exact(x[j].value(),y[j].value()), "all model parameter bytes exact");
  const auto u=a->named_buffers(), v=b->named_buffers();
  check(u.size()==v.size(), "model buffer count exact");
  for (size_t j=0;j<u.size();++j)
    check(u[j].key()==v[j].key() && exact(u[j].value(),v[j].value()), "all model buffer bytes exact");
}
void same_optimizers(const r::ContentOrderedRelationModel &a,const torch::optim::AdamW &oa,
    const r::ContentOrderedRelationModel &b,const torch::optim::AdamW &ob) {
  const auto ap=a->parameters(),bp=b->parameters();
  check(oa.state().size()==ob.state().size(), "optimizer state count exact");
  for(size_t j=0;j<ap.size();++j) {
    const auto ai=oa.state().find(ap[j].unsafeGetTensorImpl()),bi=ob.state().find(bp[j].unsafeGetTensorImpl());
    check((ai==oa.state().end())==(bi==ob.state().end()), "optimizer inactive parameter slots exact");
    if(ai==oa.state().end()) continue;
    const auto *x=dynamic_cast<const torch::optim::AdamWParamState*>(ai->second.get());
    const auto *y=dynamic_cast<const torch::optim::AdamWParamState*>(bi->second.get());
    check(x&&y&&x->step()==y->step()&&exact(x->exp_avg(),y->exp_avg())&&
              exact(x->exp_avg_sq(),y->exp_avg_sq())&&exact(x->max_exp_avg_sq(),y->max_exp_avg_sq()),
          "every AdamW step/first/second moment byte exact");
  }
}
} // namespace

int main() try {
  check(torch::cuda::is_available(), "actual CUDA required"); torch::set_num_threads(1);
  check(r::training_counter_seed(0,0)==INT64_C(7070836379803831727), "versioned SplitMix64 reference value");
  rejects([]{r::training_counter_seed(-1,0);}, "negative base seed refused");
  rejects([]{r::training_counter_seed(0,-1);}, "negative attempt refused");
  auto c=r::default_content_ordered_relation_config();
  c.history_length=129;c.channel_ids={10,20,30};c.sampling_interval=60.;
  c.encoder_width=16;c.num_layers=1;c.feedforward_width=32;c.decoder_hidden_width=16;
  const auto raw=population(c);
  const auto scaler=r::fit_configurable_spectral_scaler(raw,c);
  const auto normalized=r::transform_configurable_spectral_input(scaler,raw,c);
  torch::manual_seed(220016);r::ContentOrderedRelationModel model(c);
  torch::optim::AdamW optimizer(model->parameters(),r::content_training_adamw_options());
  r::TrainingIdentity identity{"artificial-source-manifest-v1","artificial-TRAIN-split-v1","artificial-fixed-adamw-counter-recipe-v1"};
  r::TrainingProgress progress{0,0,220017,220018,220019};
  const auto initial_relation=model->relation_projection->weight.detach().clone();
  (void)update(normalized,model,optimizer,progress);
  check(!exact(initial_relation,model->relation_projection->weight), "first actual update changes learned projection");
  // Simulate the runner's ineligible guard without executing an encoder.
  const auto absent=torch::zeros_like(normalized.observed);
  const auto skip=r::configurable_spectral_make_training_mask(absent,c,
      r::training_counter_seed(progress.mask_seed,progress.attempted_batches));
  check(!skip.eligible_channels.any().item<bool>(), "absent attempt is ineligible before model execution");
  ++progress.attempted_batches;
  TemporaryDirectory fixture;const auto file=fixture.path/"training-1.pt";
  r::save_content_training_state(file.string(),model,scaler,optimizer,identity,progress);
  const auto saved=bytes(file);
  auto restored=r::load_content_training_state(file.string(),c,identity,scaler.identity());
  check(restored.progress.completed_updates==1&&restored.progress.attempted_batches==2&&
            restored.progress.row_seed==progress.row_seed&&restored.progress.mask_seed==progress.mask_seed&&
            restored.progress.context_seed==progress.context_seed, "all counters and next stream seeds restore exactly");
  check(restored.model->is_training()&&exact(scaler.mean,restored.scaler.mean)&&
            exact(scaler.scale,restored.scaler.scale)&&exact(scaler.count,restored.scaler.count)&&
            exact(scaler.channel_ids,restored.scaler.channel_ids)&&exact(scaler.floor_applied,restored.scaler.floor_applied),
        "TRAIN scaler and train mode restore exactly");
  same_models(model,restored.model);same_optimizers(model,optimizer,restored.model,*restored.optimizer);
  const auto resumed_input=r::transform_configurable_spectral_input(restored.scaler,raw,restored.config);
  check(exact(normalized.data,resumed_input.data)&&exact(normalized.observed,resumed_input.observed),
        "resumed TRAIN standardization bytes exact");
  const auto uninterrupted=update(normalized,model,optimizer,progress);
  const auto resumed=update(resumed_input,restored.model,*restored.optimizer,restored.progress);
  check(exact(uninterrupted.rows,resumed.rows)&&exact(uninterrupted.hidden,resumed.hidden)&&
            exact(uninterrupted.visible,resumed.visible)&&exact(uninterrupted.loss,resumed.loss),
        "next rows/original target mask/context erasure/loss bytes exact");
  check(uninterrupted.gradients.size()==resumed.gradients.size(), "gradient manifest sizes exact");
  for(size_t j=0;j<uninterrupted.gradients.size();++j)
    check(exact(uninterrupted.gradients[j],resumed.gradients[j]), "every next-update gradient byte exact");
  same_models(model,restored.model);same_optimizers(model,optimizer,restored.model,*restored.optimizer);
  check(progress.completed_updates==2&&progress.attempted_batches==3&&
            restored.progress.completed_updates==2&&restored.progress.attempted_batches==3,
        "skipped attempt consumes a slot without resetting completed optimizer updates");
  r::save_content_training_state((fixture.path/"training-2.pt").string(),restored.model,restored.scaler,
      *restored.optimizer,identity,restored.progress);
  rejects([&]{r::save_content_training_state(file.string(),model,scaler,optimizer,identity,progress);},
          "immutable new filenames refuse existing destination");
  check(bytes(file)==saved, "refused replacement preserves training archive bytes");
  const auto inference=fixture.path/"inference.pt";
  r::save_content_ordered_relation_checkpoint(inference.string(),model,scaler,progress.completed_updates);
  rejects([&]{r::load_content_training_state(inference.string(),c,identity,scaler.identity());},
          "inference archive cannot resume training");
  rejects([&]{r::load_content_ordered_relation_checkpoint(file.string(),c.device,&c);},
          "training archive cannot masquerade as inference");
  auto wrong=c;wrong.history_length=130;
  rejects([&]{r::load_content_training_state(file.string(),wrong,identity,scaler.identity());},
          "same padded but changed logical history refuses resume");
  wrong=c;wrong.device=torch::kCPU;
  rejects([&]{r::load_content_training_state(file.string(),wrong,identity,scaler.identity());},
          "CPU configuration refuses training resume");
  for(int j=0;j<3;++j) {
    auto drift=identity;
    (j==0?drift.input_source_identity:j==1?drift.split_identity:drift.recipe_identity)+="-changed";
    rejects([&]{r::load_content_training_state(file.string(),c,drift,scaler.identity());},
            "source/split/recipe drift refuses resume");
  }
  rejects([&]{r::load_content_training_state(file.string(),c,identity,"changed-TRAIN-scaler");},
          "independently expected TRAIN scaler drift refuses resume");
  auto malformed=progress;malformed.completed_updates=malformed.attempted_batches+1;
  rejects([&]{r::save_content_training_state((fixture.path/"bad-counter.pt").string(),model,scaler,optimizer,identity,malformed);},
          "completed greater than attempted refuses save");
  malformed=progress;malformed.completed_updates=1;
  rejects([&]{r::save_content_training_state((fixture.path/"reset-step.pt").string(),model,scaler,optimizer,identity,malformed);},
          "counter reset versus actual AdamW steps refuses save");
  torch::optim::AdamW altered(model->parameters(),torch::optim::AdamWOptions(.002).weight_decay(.0001));
  rejects([&]{r::save_content_training_state((fixture.path/"wrong-options.pt").string(),model,scaler,altered,identity,{});},
          "optimizer recipe change refuses save");
  auto reversed=model->parameters();std::reverse(reversed.begin(),reversed.end());
  torch::optim::AdamW remapped(reversed,r::content_training_adamw_options());
  rejects([&]{r::save_content_training_state((fixture.path/"wrong-order.pt").string(),model,scaler,remapped,identity,{});},
          "optimizer parameter remapping refuses save");
  for(const auto &entry:fs::directory_iterator(fixture.path))
    check(entry.path().filename().string().find(".tmp-")==std::string::npos, "staging files cleaned after failed saves");
  torch::cuda::synchronize();
  std::cout<<"PASS training state: W16/H129 CUDA, one update plus skip, exact restored next update "
      "(three optimizer actions total), AdamW moments/steps/model/scaler/counters, typed/config/source/split/recipe "
      "refusals and immutable atomic new files; artificial engineering, no heads/quality\n";
  return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
