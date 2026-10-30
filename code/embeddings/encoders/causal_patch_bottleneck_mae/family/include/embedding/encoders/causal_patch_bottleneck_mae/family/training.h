// Copyright (c) 2026 Waajacu. See LICENSE for the permission notice.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::family::training {

// The caller supplies an admitted TRAIN stream and immutable manifests. This
// numerical helper cannot infer split membership from observation values.
struct PreprocessingIdentity {
  std::string input_source_identity, split_identity, recipe_identity;
};

inline std::string preprocessing_binding(const PreprocessingIdentity &identity) {
  std::string result="cpb-family-TRAIN-window-Chan-input-moments-v1;";
  for(const auto *field:{&identity.input_source_identity,&identity.split_identity,&identity.recipe_identity}) {
    TORCH_CHECK(!field->empty() && field->find('\0')==std::string::npos,
        "[cpb family training] explicit source/split/preprocessing-recipe identities required");
    result+=std::to_string(field->size())+":"+*field;
  }
  TORCH_CHECK(result.size()+std::string(implementation::kRelativeChangeSourcePrefix).size()<=4096,
      "[cpb family training] preprocessing identity binding is too long");
  return result;
}

struct FittedPreprocessing {
  FrozenScaler scaler;
  BankStandardizer bank;
  PreprocessingIdentity identity;
  int64_t windows{0};
};

// Window-weighted population sufficient statistics. The parent path rounds
// genuine raw F64 to F32 before scaling; bank targets retain raw F64 arithmetic
// before their own documented F32 rounding. Neither path calls the encoder.
class PreprocessingFitter {
 public:
  PreprocessingFitter(const Config &config,PreprocessingIdentity identity)
      :config_(config),identity_(std::move(identity)),binding_(preprocessing_binding(identity_)) {
    family::validate_config(config_);
    const auto &c=config_.level;
    counts_=torch::zeros({c.channel_count,c.input_width},torch::kInt64);
    mean_=torch::zeros(counts_.sizes(),torch::kFloat64);m2_=torch::zeros_like(mean_);
    const auto width=implementation::consecutive_change_bank_width(c);
    bank_counts_=torch::zeros({width},torch::kInt64);
    bank_sum_=torch::zeros({width},torch::kFloat64);bank_sumsq_=torch::zeros_like(bank_sum_);
  }

  void add(const RawHistory &raw,const torch::Tensor &cutoffs) {
    TORCH_CHECK(!frozen_,"[cpb family training] frozen preprocessing cannot refit");
    torch::NoGradGuard guard;
    const auto retained=boundary::retain(raw,cutoffs,config_);const auto &c=config_.level;
    const auto B=retained.raw.data.size(0);
    TORCH_CHECK(windows_<=std::numeric_limits<int64_t>::max()-B,
        "[cpb family training] preprocessing window count overflow");
    // Batch channels may arrive in different semantic orders. Accumulate all
    // coordinates in the configuration's canonical channel-ID order.
    const auto indices=implementation::channel_indices(retained.raw.channel_ids,c,B,torch::kCPU);
    const auto order=indices.argsort(int64_t{1}).unsqueeze(-1).unsqueeze(-1)
        .expand({B,c.channel_count,c.history_length,c.input_width});
    const auto observed=retained.raw.observed.gather(1,order);
    const auto values=retained.raw.data.gather(1,order);
    const auto rounded=torch::where(observed,values,torch::zeros_like(values))
        .to(torch::kFloat32).to(torch::kFloat64);
    TORCH_CHECK(torch::isfinite(rounded).all().item<bool>(),
        "[cpb family training] raw observations overflow the historical F32 parent cast");
    const auto n=observed.sum(std::vector<int64_t>{0,2});
    TORCH_CHECK(counts_.le(std::numeric_limits<int64_t>::max()-n).all().item<bool>(),
        "[cpb family training] input preprocessing count overflow");
    // Center within this batch, then combine canonical C,F moments with Chan's
    // parallel variance formula. E[x*x]-E[x]^2 loses rare F32-ULP variation
    // around a large offset even though the parent values themselves are exact
    // F32 numbers promoted to F64. Keep only count/mean/M2 across batches.
    const auto batch_count=n.to(torch::kFloat64);
    const auto batch_mean=rounded.sum(std::vector<int64_t>{0,2})/batch_count.clamp_min(1);
    const auto centered=torch::where(observed,rounded-batch_mean.unsqueeze(0).unsqueeze(2),torch::zeros_like(rounded));
    const auto batch_m2=centered.square().sum(std::vector<int64_t>{0,2});
    const auto combined_counts=counts_+n;
    const auto denominator=combined_counts.to(torch::kFloat64).clamp_min(1);
    const auto delta=batch_mean-mean_;
    const auto combined_mean=mean_+delta*(batch_count/denominator);
    const auto combined_m2=m2_+batch_m2+delta.square()*(counts_.to(torch::kFloat64)/denominator)*batch_count;
    TORCH_CHECK(torch::isfinite(combined_mean).all().item<bool>() &&
        torch::isfinite(combined_m2).all().item<bool>() && combined_m2.ge(0).all().item<bool>(),
        "[cpb family training] input preprocessing centered-moment overflow");
    counts_=combined_counts;mean_=combined_mean;m2_=combined_m2;
    const auto targets=implementation::relative_change_targets(retained.raw,c);
    const auto valid=targets.valid.detach().to(torch::kCPU);
    const auto safe=torch::where(targets.valid,targets.values,torch::zeros_like(targets.values))
        .detach().to(torch::kCPU,torch::kFloat64);
    const auto bank_n=valid.sum(0);
    TORCH_CHECK(bank_counts_.le(std::numeric_limits<int64_t>::max()-bank_n).all().item<bool>(),
        "[cpb family training] bank preprocessing count overflow");
    bank_counts_+=bank_n;bank_sum_+=safe.sum(0);bank_sumsq_+=safe.square().sum(0);
    TORCH_CHECK(torch::isfinite(bank_sum_).all().item<bool>() && torch::isfinite(bank_sumsq_).all().item<bool>(),
        "[cpb family training] preprocessing sufficient-statistic overflow");
    windows_+=B;
  }

  FittedPreprocessing freeze() {
    TORCH_CHECK(!frozen_ && windows_>0 && counts_.gt(0).all().item<bool>(),
        "[cpb family training] freeze once after observed TRAIN support for every input coordinate");
    torch::NoGradGuard guard;const auto &c=config_.level;
    const auto deviation=(m2_/counts_.to(torch::kFloat64)).sqrt();
    FrozenScaler scaler{mean_.clone(),deviation.clamp_min(c.scale_floor),counts_.clone(),
        torch::tensor(implementation::resolved_channel_ids(c),torch::kInt64),
        c.scale_floor,deviation.lt(c.scale_floor)};
    scaler.validate(input_config(config_));
    BankStandardizer bank(c,binding_,scaler.identity());
    bank.counts=bank_counts_.clone();bank.sum=bank_sum_.clone();bank.sumsq=bank_sumsq_.clone();
    bank.freeze();const auto scaler_id=scaler.identity();bank.validate(c,nullptr,&scaler_id);
    frozen_=true;return{scaler,bank,identity_,windows_};
  }

  bool frozen()const{return frozen_;}
  int64_t windows()const{return windows_;}
 private:
  Config config_;PreprocessingIdentity identity_;std::string binding_;bool frozen_{false};int64_t windows_{0};
  torch::Tensor counts_,mean_,m2_,bank_counts_,bank_sum_,bank_sumsq_;
};

using MaskBatch=implementation::MaskPlan;
inline MaskBatch make_mask(const RawHistory &raw,const torch::Tensor &cutoffs,
    const Config &config,int64_t base_seed,int64_t attempt) {
  family::validate_config(config);
  const auto retained=boundary::retain(raw,cutoffs,config);
  const auto seed=training_counter_seed(base_seed,attempt);
  return implementation::configurable_spectral_make_training_mask(retained.raw.observed,config.level,seed);
}

struct StepResult {
  bool updated{false};
  int64_t attempt_index{0},completed_updates{0},eligible_examples{0},target_cells{0},gradient_tensors{0};
  double gaussian_nll{0},temporal_loss{0},total_loss{0},gradient_norm{0};
};

// One completed attempt, with no accumulation. Ineligible masks consume their
// attempt but no optimizer step. Row selection and any optional context
// deletion remain caller-owned counter streams; bind their recipes in the
// TrainingIdentity used by save/load_training_state. This family requires
// dropout0, so these explicit streams suffice without an implicit Torch RNG.
inline StepResult train_step(Encoder &model,const FrozenScaler &scaler,
    torch::optim::AdamW &optimizer,TrainingProgress &progress,const RawHistory &raw,
    const torch::Tensor &cutoffs,const OptimizerConfig &options={},
    const torch::Tensor &context_deleted={},double temporal_weight=0.1) {
  TORCH_CHECK(model && torch::GradMode::is_enabled(),
      "[cpb family training] owned model and enabled gradients required");
  family::validate_config(model->config());validate_optimizer_config(options);
  TORCH_CHECK(model->causal_updates()==progress.completed_updates &&
      progress.attempted_batches<std::numeric_limits<int64_t>::max()-1,
      "[cpb family training] exact live progress and an incrementable attempt counter required");
  validate_optimizer(model,optimizer,progress,options);
  TORCH_CHECK(scaler.identity()==model->input_scaler_identity(),
      "[cpb family training] training must use the model's own frozen TRAIN scaler");
  const auto masks=make_mask(raw,cutoffs,model->config(),progress.mask_seed,progress.attempted_batches);
  if(context_deleted.defined()) {
    TORCH_CHECK(context_deleted.scalar_type()==torch::kBool && context_deleted.sizes()==masks.visible.sizes(),
        "[cpb family training] logical-window bool context deletion required");
    TORCH_CHECK(!(context_deleted.to(masks.visible.device())&masks.visible.logical_not()).any().item<bool>(),
        "[cpb family training] context deletions must be a subset of query-visible observations");
  }
  StepResult result;result.attempt_index=progress.attempted_batches;result.completed_updates=progress.completed_updates;
  if(!masks.eligible_channels.any().item<bool>()) {
    optimizer.zero_grad(true);model->clear_gradients();++progress.attempted_batches;return result;
  }
  const auto bank_id=model->normalizer().identity();
  optimizer.zero_grad(true);model->train();
  const auto trace=model->forward_at(raw,cutoffs,masks.hidden,scaler,context_deleted);
  TORCH_CHECK(torch::isfinite(trace.output.loss).item<bool>() && trace.nll.eligible_example_count>0,
      "[cpb family training] finite supported objective required before backward");
  result.gaussian_nll=trace.nll.loss.item<double>();result.temporal_loss=trace.objective.temporal.loss.item<double>();
  result.eligible_examples=trace.nll.eligible_example_count;
  result.target_cells=trace.nll.target_cell_count;
  // Weight 0.1 is the frozen V16 coefficient already stored in output.loss.
  // Any other finite nonnegative weight is this family's explicit reweight.
  torch::Tensor objective=trace.output.loss;
  if(temporal_weight!=0.1){
    TORCH_CHECK(std::isfinite(temporal_weight)&&temporal_weight>=0,
        "[cpb family training] temporal weight must be finite and nonnegative");
    objective=trace.nll.loss+temporal_weight*trace.objective.temporal.loss;
    TORCH_CHECK(torch::isfinite(objective).item<bool>(),
        "[cpb family training] finite reweighted objective required before backward");
  }
  result.total_loss=objective.item<double>();
  objective.backward();
  for(const auto &parameter:model->named_parameters()) {
    const auto gradient=parameter.value().grad();
    if(!gradient.defined())continue;
    TORCH_CHECK(parameter.value().requires_grad() && gradient.is_cuda() &&
        torch::isfinite(gradient).all().item<bool>(),"[cpb family training] finite learned CUDA gradients required");
    ++result.gradient_tensors;
  }
  TORCH_CHECK(result.gradient_tensors>0,"[cpb family training] objective did not reach a learned parameter");
  result.gradient_norm=torch::nn::utils::clip_grad_norm_(learned_parameters(model),options.clip_norm);
  TORCH_CHECK(std::isfinite(result.gradient_norm),"[cpb family training] nonfinite global gradient norm");
  optimizer.step();++progress.completed_updates;++progress.attempted_batches;
  model->set_causal_updates(progress.completed_updates);
  optimizer.zero_grad(true);model->clear_gradients();
  model->core()->validate_state();model->core()->validate_fixed_prior();
  TORCH_CHECK(model->normalizer().identity()==bank_id && scaler.identity()==model->input_scaler_identity(),
      "[cpb family training] optimizer changed frozen preprocessing");
  validate_optimizer(model,optimizer,progress,options);
  result.updated=true;result.completed_updates=progress.completed_updates;return result;
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::family::training
