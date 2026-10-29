// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// CPB-v17 changes the temporal objective weight over the configurable CPB family.
// Channel axis: one window is one instrument. Each channel is a cuwacunu_torch kline interval (timeframe_channel.h). {0,1,2} is the withdrawn instrument binding.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace embedding::encoders::causal_patch_bottleneck_mae::v17 {
namespace architecture=family;
namespace implementation=family::implementation;
namespace base_training=family::training;
using Config=family::Config;
using Encoder=family::Encoder;
using FrozenScaler=family::FrozenScaler;
using BankStandardizer=family::BankStandardizer;
using RawHistory=family::RawHistory;
using Encoding=family::Encoding;
using CausalTrace=family::CausalTrace;
using TrainingIdentity=base_training::TrainingIdentity;
using TrainingProgress=base_training::TrainingProgress;
using OptimizerConfig=base_training::OptimizerConfig;
using StepResult=base_training::StepResult;
using PreprocessingIdentity=base_training::PreprocessingIdentity;
using PreprocessingFitter=base_training::PreprocessingFitter;
using FittedPreprocessing=base_training::FittedPreprocessing;
// research.conf is the recorded binding. Engineering tests also load timeframe ids.
inline Config read_settings(const std::filesystem::path &path){return family::read_admitted_settings(path);}
using family::validate_config;
using family::dimensions;
using family::parameter_counts;
using family::fresh;
using base_training::make_optimizer;
using base_training::make_mask;
using base_training::training_counter_seed;
using base_training::optimizer_identity;
inline constexpr const char *kModelTag="CPB-v17";
inline constexpr const char *kInferenceKind="cpb_v17_weighted_temporal_inference_v1";
inline constexpr const char *kTrainingKind="cpb_v17_weighted_temporal_training_v1";
inline constexpr const char *kRecipeKind="cpb-fixed-weight-hidden-temporal-recipe-v1";
inline constexpr const char *kPayloadCodec="cpb-family-structural-state-container-v1";
inline constexpr const char *kPayloadMeaning="structural-model-config-fitted-state;inner-legacy-objective-fields-identify-codec-only;outer-v17-recipe-is-authoritative";

struct Recipe {double temporal_weight{.03};};
inline void validate_recipe(const Recipe &recipe) {
  TORCH_CHECK(std::isfinite(recipe.temporal_weight)&&recipe.temporal_weight>=0,
      "[cpb-v17] finite nonnegative temporal objective weight required");
}
inline std::string recipe_identity(const Recipe &recipe) {
  validate_recipe(recipe);std::ostringstream out;out<<std::setprecision(17)
      <<kModelTag<<';'<<kRecipeKind<<";GaussianNLL=1;temporal="<<recipe.temporal_weight
      <<";lags=1,4,16;fixed-mean3;scale=sqrt2;Huber=1;originalQ-both-endpoints;variance-input-gradient-stop;dropout0";
  return out.str();
}

// The unchanged architecture computes all three temporal components. Replace
// only their total coefficient; no target, support, reduction or graph is rebuilt.
inline torch::Tensor objective(const implementation::LocationScaleForwardTrace &trace,const Recipe &recipe={}) {
  validate_recipe(recipe);
  TORCH_CHECK(trace.nll.loss.defined()&&trace.objective.temporal.loss.defined(),"[cpb-v17] complete base and temporal loss required");
  const auto total=trace.nll.loss+recipe.temporal_weight*trace.objective.temporal.loss;
  TORCH_CHECK(total.dim()==0&&total.is_cuda()&&total.scalar_type()==torch::kFloat64&&
      torch::isfinite(total).item<bool>(),"[cpb-v17] finite supported CUDA F64 objective required");
  return total;
}
inline implementation::LocationScaleForwardTrace forward_at(Encoder &model,const RawHistory &raw,
    const torch::Tensor &cutoffs,const torch::Tensor &hidden,const FrozenScaler &scaler,
    const Recipe &recipe={},const torch::Tensor &context_deleted={}) {
  auto trace=model->forward_at(raw,cutoffs,hidden,scaler,context_deleted);
  trace.objective.total_loss=objective(trace,recipe);trace.output.loss=trace.objective.total_loss;return trace;
}

// Same mask, eligibility, backward, clipping and AdamW sequence as the family
// default path. Weight .1 is the unchanged-family control, not old trained weights.
inline StepResult train_step(Encoder &model,const FrozenScaler &scaler,
    torch::optim::AdamW &optimizer,TrainingProgress &progress,const RawHistory &raw,
    const torch::Tensor &cutoffs,const Recipe &recipe={},const OptimizerConfig &options={},
    const torch::Tensor &context_deleted={}) {
  validate_recipe(recipe);
  TORCH_CHECK(model&&torch::GradMode::is_enabled(),"[cpb-v17] model and enabled gradients required");
  family::validate_config(model->config());base_training::validate_optimizer_config(options);
  TORCH_CHECK(model->causal_updates()==progress.completed_updates&&
      progress.attempted_batches<std::numeric_limits<int64_t>::max()-1,
      "[cpb-v17] exact live progress and incrementable attempted count required");
  base_training::validate_optimizer(model,optimizer,progress,options);
  TORCH_CHECK(scaler.identity()==model->input_scaler_identity(),"[cpb-v17] own frozen TRAIN scaler required");
  const auto masks=make_mask(raw,cutoffs,model->config(),progress.mask_seed,progress.attempted_batches);
  if(context_deleted.defined()) {
    TORCH_CHECK(context_deleted.scalar_type()==torch::kBool&&context_deleted.sizes()==masks.visible.sizes(),
        "[cpb-v17] matching logical-window bool context deletion required");
    TORCH_CHECK(!(context_deleted.to(masks.visible.device())&masks.visible.logical_not()).any().item<bool>(),
        "[cpb-v17] deleted context must be a subset of visible observations");
  }
  StepResult result;result.attempt_index=progress.attempted_batches;result.completed_updates=progress.completed_updates;
  if(!masks.eligible_channels.any().item<bool>()) {
    optimizer.zero_grad(true);model->clear_gradients();++progress.attempted_batches;return result;
  }
  const auto bank_id=model->normalizer().identity();optimizer.zero_grad(true);model->train();
  const auto trace=forward_at(model,raw,cutoffs,masks.hidden,scaler,recipe,context_deleted);
  TORCH_CHECK(trace.nll.eligible_example_count>0,"[cpb-v17] supported objective required before backward");
  result.gaussian_nll=trace.nll.loss.item<double>();result.temporal_loss=trace.objective.temporal.loss.item<double>();
  result.total_loss=trace.output.loss.item<double>();result.eligible_examples=trace.nll.eligible_example_count;
  result.target_cells=trace.nll.target_cell_count;trace.output.loss.backward();
  for(const auto &parameter:model->named_parameters()) {
    const auto gradient=parameter.value().grad();if(!gradient.defined())continue;
    TORCH_CHECK(parameter.value().requires_grad()&&gradient.is_cuda()&&torch::isfinite(gradient).all().item<bool>(),
        "[cpb-v17] finite learned CUDA gradients required");++result.gradient_tensors;
  }
  TORCH_CHECK(result.gradient_tensors>0,"[cpb-v17] objective must reach learned parameters");
  result.gradient_norm=torch::nn::utils::clip_grad_norm_(base_training::learned_parameters(model),options.clip_norm);
  TORCH_CHECK(std::isfinite(result.gradient_norm),"[cpb-v17] finite global gradient norm required");
  optimizer.step();++progress.completed_updates;++progress.attempted_batches;model->set_causal_updates(progress.completed_updates);
  optimizer.zero_grad(true);model->clear_gradients();model->core()->validate_state();model->core()->validate_fixed_prior();
  TORCH_CHECK(model->normalizer().identity()==bank_id&&scaler.identity()==model->input_scaler_identity(),
      "[cpb-v17] optimizer changed frozen preprocessing");
  base_training::validate_optimizer(model,optimizer,progress,options);
  result.updated=true;result.completed_updates=progress.completed_updates;return result;
}

namespace artifact_detail {
namespace d=family::checkpoint_detail::d;
inline void write_recipe(torch::serialize::OutputArchive &archive,const Recipe &recipe,const char *kind) {
  validate_recipe(recipe);
  archive.write("artifact_kind",d::text_tensor(kind),true);archive.write("model_tag",d::text_tensor(kModelTag),true);
  archive.write("recipe_kind",d::text_tensor(kRecipeKind),true);archive.write("recipe_binding",d::text_tensor(recipe_identity(recipe)),true);
  archive.write("temporal_weight_f64",torch::tensor(recipe.temporal_weight,torch::kFloat64),true);
  archive.write("implementation_payload_codec",d::text_tensor(kPayloadCodec),true);
  archive.write("implementation_payload_meaning",d::text_tensor(kPayloadMeaning),true);
}
inline Recipe read_recipe(torch::serialize::InputArchive &archive,const Recipe &expected,const char *kind) {
  validate_recipe(expected);torch::Tensor weight;archive.read("temporal_weight_f64",weight,true);
  TORCH_CHECK(d::read_text(archive,"artifact_kind")==kind&&d::read_text(archive,"model_tag")==kModelTag&&
      d::read_text(archive,"recipe_kind")==kRecipeKind&&d::read_text(archive,"recipe_binding")==recipe_identity(expected)&&
      d::read_text(archive,"implementation_payload_codec")==kPayloadCodec&&
      d::read_text(archive,"implementation_payload_meaning")==kPayloadMeaning&&
      weight.dim()==0&&weight.scalar_type()==torch::kFloat64&&
      family::boundary::same_bits(weight,torch::tensor(expected.temporal_weight,torch::kFloat64)),
      "[cpb-v17] typed artifact/version/weight/recipe/payload meaning mismatch");
  return expected;
}
inline std::string progress_binding(const TrainingIdentity &identity,const TrainingProgress &progress,const Recipe &recipe) {
  return recipe_identity(recipe)+":"+base_training::progress_binding(identity,progress);
}
} // namespace artifact_detail

struct Loaded {family::Loaded numerical;Recipe recipe;};
inline void save_checkpoint(const std::string &path,const Encoder &model,const FrozenScaler &scaler,
    int64_t updates,const Recipe &recipe={}) {
  TORCH_CHECK(model,"[cpb-v17] model required");model->validate_frozen();
  torch::serialize::OutputArchive archive,payload;
  artifact_detail::write_recipe(archive,recipe,kInferenceKind);
  family::checkpoint_detail::write(payload,model,scaler,updates);archive.write("structural_encoder",payload);
  artifact_detail::d::publish_checkpoint(archive,path);
}
inline Loaded load_checkpoint(const std::string &path,torch::Device device=torch::Device(torch::kCUDA,0),
    const Config *expected_config=nullptr,const Recipe &expected_recipe={}) {
  torch::serialize::InputArchive archive,payload;archive.load_from(path,device);
  const auto recipe=artifact_detail::read_recipe(archive,expected_recipe,kInferenceKind);
  archive.read("structural_encoder",payload);
  return{family::checkpoint_detail::read(payload,device,expected_config),recipe};
}

struct LoadedTrainingState {
  Config config;Encoder model{nullptr};FrozenScaler scaler;
  std::unique_ptr<torch::optim::AdamW> optimizer;TrainingIdentity identity;TrainingProgress progress;
  OptimizerConfig optimizer_config;Recipe recipe;
};
inline void save_training_state(const std::string &path,const Encoder &model,const FrozenScaler &scaler,
    const torch::optim::AdamW &optimizer,const TrainingIdentity &identity,const TrainingProgress &progress,
    const Recipe &recipe={},const OptimizerConfig &options={}) {
  validate_recipe(recipe);base_training::common::validate_identity(identity);
  TORCH_CHECK(model&&model->causal_updates()==progress.completed_updates,"[cpb-v17] model and progress update count mismatch");
  const auto steps=base_training::validate_optimizer(model,optimizer,progress,options);
  torch::serialize::OutputArchive archive,encoder,moments;artifact_detail::write_recipe(archive,recipe,kTrainingKind);
  const auto write_text=[&](const char *key,const std::string &value){archive.write(key,artifact_detail::d::text_tensor(value),true);};
  write_text("counter_contract",implementation::consecutive_change_training::kChangeTrainingCounterContract);
  write_text("input_source_identity",identity.input_source_identity);write_text("split_identity",identity.split_identity);
  write_text("recipe_identity",identity.recipe_identity);write_text("progress_binding",artifact_detail::progress_binding(identity,progress,recipe));
  write_text("parameter_order",base_training::parameter_order(model));
  write_text("optimizer_parameter_keys",base_training::common::optimizer_parameter_keys(optimizer));
  write_text("optimizer_state_identity",base_training::optimizer_identity(model,optimizer));
  archive.write("optimizer_config_f64",torch::tensor(base_training::optimizer_config_values(options),torch::kFloat64),true);
  archive.write("progress_v1",torch::tensor(std::vector<int64_t>{progress.completed_updates,progress.attempted_batches,
      progress.row_seed,progress.mask_seed,progress.context_seed},torch::kInt64),true);
  archive.write("optimizer_steps_v1",torch::tensor(steps,torch::kInt64),true);
  family::checkpoint_detail::write(encoder,model,scaler,progress.completed_updates);optimizer.save(moments);
  archive.write("structural_encoder",encoder);archive.write("optimizer",moments);
  artifact_detail::d::publish_checkpoint(archive,path);
}
inline LoadedTrainingState load_training_state(const std::string &path,const Config &expected,
    const TrainingIdentity &identity,const Recipe &recipe={},const OptimizerConfig &options={}) {
  family::validate_config(expected);base_training::validate_optimizer_config(options);base_training::common::validate_identity(identity);
  torch::serialize::InputArchive archive,encoder,moments;archive.load_from(path,expected.level.device);
  artifact_detail::read_recipe(archive,recipe,kTrainingKind);
  TORCH_CHECK(artifact_detail::d::read_text(archive,"counter_contract")==implementation::consecutive_change_training::kChangeTrainingCounterContract&&
      family::boundary::same_bits(artifact_detail::d::read_vector(archive,"optimizer_config_f64",torch::kFloat64,3),
      torch::tensor(base_training::optimizer_config_values(options),torch::kFloat64)),"[cpb-v17] counter/optimizer recipe mismatch");
  TrainingIdentity actual{artifact_detail::d::read_text(archive,"input_source_identity"),
      artifact_detail::d::read_text(archive,"split_identity"),artifact_detail::d::read_text(archive,"recipe_identity")};
  base_training::common::validate_identity(actual);
  TORCH_CHECK(base_training::common::same_identity(actual,identity),"[cpb-v17] expected source/split/application recipe mismatch");
  const auto values=artifact_detail::d::read_vector(archive,"progress_v1",torch::kInt64,5);const auto p=values.accessor<int64_t,1>();
  TrainingProgress progress{p[0],p[1],p[2],p[3],p[4]};base_training::common::validate_progress(progress);
  TORCH_CHECK(artifact_detail::d::read_text(archive,"progress_binding")==artifact_detail::progress_binding(identity,progress,recipe),
      "[cpb-v17] progress/seed/recipe binding mismatch");
  archive.read("structural_encoder",encoder);auto child=family::checkpoint_detail::read(encoder,expected.level.device,&expected);
  TORCH_CHECK(child.causal_updates==progress.completed_updates&&artifact_detail::d::read_text(archive,"parameter_order")==base_training::parameter_order(child.model),
      "[cpb-v17] child update/parameter order mismatch");
  auto optimizer=base_training::make_optimizer(child.model,options);archive.read("optimizer",moments);
  base_training::common::validate_optimizer_archive(moments,artifact_detail::d::read_text(archive,"optimizer_parameter_keys"),
      base_training::learned_parameters(child.model).size());
  optimizer->load(moments);const auto steps=base_training::validate_optimizer(child.model,*optimizer,progress,options);
  TORCH_CHECK(artifact_detail::d::read_text(archive,"optimizer_state_identity")==base_training::optimizer_identity(child.model,*optimizer)&&
      torch::equal(artifact_detail::d::read_vector(archive,"optimizer_steps_v1",torch::kInt64,steps.size()),torch::tensor(steps,torch::kInt64)),
      "[cpb-v17] optimizer moment content/order/step mismatch");
  child.model->train();child.model->clear_gradients();
  return{child.config,child.model,child.scaler,std::move(optimizer),actual,progress,options,recipe};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::v17
