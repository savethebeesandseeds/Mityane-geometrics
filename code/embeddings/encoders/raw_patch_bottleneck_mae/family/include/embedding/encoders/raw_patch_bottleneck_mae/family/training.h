// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Waajacu. See the repository LICENSE.
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/family/family.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_state.h"

namespace embedding::encoders::raw_patch_bottleneck_mae::family {

using TrainingIdentity = raw_patch_bottleneck_mae::TrainingIdentity;
using TrainingProgress = raw_patch_bottleneck_mae::TrainingProgress;
using LoadedTrainingState = raw_patch_bottleneck_mae::LoadedContentTrainingState;

class PreprocessingFitter {
public:
  explicit PreprocessingFitter(const Config &c)
      : input_(checked_input(c)), stream_(input_) {}
  void add(const Input &raw_train) {
    TORCH_CHECK(!finished_,"[rpb family] frozen fitter cannot accept more TRAIN batches");
    stream_.add_train_batch(raw_train);
  }
  FrozenScaler freeze() {
    TORCH_CHECK(!finished_,"[rpb family] fitter already frozen");
    auto result=stream_.finish();
    finished_=true;
    return result;
  }
private:
  static Config checked_input(const Config &c) {
    family::validate_config(c);
    return configurable_spectral_input_config(c);
  }
  Config input_;
  StreamingScaler stream_;
  bool finished_{false};
};

inline std::unique_ptr<torch::optim::AdamW> make_optimizer(const Model &model) {
  TORCH_CHECK(model,"[rpb family] model required");
  model->validate_state();
  return std::make_unique<torch::optim::AdamW>(model->parameters(),content_training_adamw_options());
}
struct StepResult {
  bool updated{false};
  int64_t attempted_batches{0}, completed_updates{0}, eligible_examples{0}, target_cells{0};
  double loss{0}, gradient_norm{0};
  torch::Tensor hidden;
  std::string gradients_sha256;
};

// One caller-selected raw batch, one counter-derived original-patch mask; no
// extra context deletion. A rejected call leaves counters unchanged. Numerical
// failures after backward do not promise transactional model/optimizer rollback.
inline StepResult train_step(Model &model, const FrozenScaler &scaler,
    torch::optim::AdamW &optimizer, TrainingProgress &progress, const Input &raw) {
  TORCH_CHECK(model,"[rpb family] model required");
  TORCH_CHECK(progress.attempted_batches<std::numeric_limits<int64_t>::max()-1 &&
      progress.completed_updates<std::numeric_limits<int64_t>::max()-1,
      "[rpb family] next completed attempt must retain incrementable counters");
  training_state_detail::validate_optimizer(model,optimizer,progress);
  const auto input=transform_configurable_spectral_input(scaler,raw,model->config());
  const auto mask=configurable_spectral_make_training_mask(input.observed,model->config(),
      training_counter_seed(progress.mask_seed,progress.attempted_batches));
  StepResult result;
  auto next=progress;
  result.hidden=mask.hidden.detach().clone();
  optimizer.zero_grad();
  model->train();
  if(mask.eligible_channels.any().item<bool>()) {
    auto output=model->forward(input,mask.hidden);
    TORCH_CHECK(output.eligible_example_count>0 && torch::isfinite(output.loss).item<bool>(),
                "[rpb family] eligible finite loss required");
    output.loss.backward();
    for(const auto &p:model->named_parameters()) {
      const auto grad=p.value().grad();
      TORCH_CHECK(p.value().requires_grad() || !grad.defined(),
                  "[rpb family] frozen parameter received a gradient");
      if(grad.defined()) TORCH_CHECK(grad.is_cuda() && grad.scalar_type()==torch::kFloat32 &&
          torch::isfinite(grad).all().item<bool>(),"[rpb family] nonfinite/non-CUDA gradient: ",p.key());
    }
    result.gradients_sha256=gradient_identity(model); // pre-clip witness
    result.gradient_norm=torch::nn::utils::clip_grad_norm_(model->parameters(),1.,2.,true);
    TORCH_CHECK(std::isfinite(result.gradient_norm),"[rpb family] nonfinite gradient norm");
    optimizer.step();
    result.loss=output.loss.item<double>();
    result.eligible_examples=output.eligible_example_count;
    result.target_cells=output.target_cell_count;
    result.updated=true;
    ++next.completed_updates;
  } else {
    result.gradients_sha256=gradient_identity(model);
  }
  ++next.attempted_batches;
  training_state_detail::validate_optimizer(model,optimizer,next);
  progress=next;
  result.attempted_batches=progress.attempted_batches;
  result.completed_updates=progress.completed_updates;
  return result;
}

inline void save_training_state(const std::string &path, const Model &model,
    const FrozenScaler &scaler, const torch::optim::AdamW &optimizer,
    const TrainingIdentity &identity, const TrainingProgress &progress) {
  TORCH_CHECK(model,"[rpb family] model required");
  // Caller saves only after a completed attempt, never during accumulation.
  // Completed-step gradients may remain defined but are not continuation state.
  for(const auto &p:model->parameters()) TORCH_CHECK(!p.grad().defined() ||
      torch::isfinite(p.grad()).all().item<bool>(),"[rpb family] nonfinite pending gradient");
  save_content_training_state(path,model,scaler,optimizer,identity,progress);
}
inline LoadedTrainingState load_training_state(const std::string &path,
    const Config &expected, const TrainingIdentity &identity,
    const std::string &expected_scaler_identity, const TrainingProgress &expected_seeds) {
  TORCH_CHECK(expected_seeds.row_seed>=0 && expected_seeds.mask_seed>=0 && expected_seeds.context_seed>=0,
              "[rpb family] independently declared nonnegative stream seeds required");
  auto result=load_content_training_state(path,expected,identity,expected_scaler_identity);
  TORCH_CHECK(result.progress.row_seed==expected_seeds.row_seed &&
      result.progress.mask_seed==expected_seeds.mask_seed &&
      result.progress.context_seed==expected_seeds.context_seed,
      "[rpb family] restored seeds disagree with independently declared recipe seeds");
  return result;
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae::family
