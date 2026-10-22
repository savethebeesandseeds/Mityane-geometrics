// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// CPB-v18 changes training-mask geometry over the configurable CPB family.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include <cmath>
#include <array>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

namespace embedding::encoders::causal_patch_bottleneck_mae::v18 {
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
using PreprocessingIdentity=base_training::PreprocessingIdentity;
using PreprocessingFitter=base_training::PreprocessingFitter;
using FittedPreprocessing=base_training::FittedPreprocessing;
using family::read_settings;
using family::validate_config;
using family::dimensions;
using family::parameter_counts;
using family::fresh;
using base_training::make_optimizer;
using base_training::training_counter_seed;
using base_training::optimizer_identity;
inline constexpr const char *kModelTag="CPB-v18";
inline constexpr const char *kInferenceKind="cpb_v18_mask_geometry_inference_v1";
inline constexpr const char *kTrainingKind="cpb_v18_mask_geometry_training_v1";
inline constexpr const char *kRecipeKind="cpb-original-grid-contiguous-query-mask-v1";
inline constexpr const char *kPayloadCodec="cpb-family-structural-state-container-v1";
inline constexpr const char *kPayloadMeaning="structural-model-config-fitted-state;inner-legacy-objective-fields-identify-codec-only;outer-v18-mask-recipe-is-authoritative";

enum class MaskMode : int64_t {scattered=0,contiguous=1};
struct Recipe {MaskMode mask_mode{MaskMode::contiguous};};
inline const char *mask_mode_name(MaskMode mode){
  TORCH_CHECK(mode==MaskMode::scattered||mode==MaskMode::contiguous,"[cpb-v18] supported training mask mode required");
  return mode==MaskMode::scattered?"scattered":"contiguous";
}
inline void validate_recipe(const Recipe &recipe) {
  (void)mask_mode_name(recipe.mask_mode);
}
inline std::string recipe_identity(const Recipe &recipe) {
  validate_recipe(recipe);std::ostringstream out;out<<std::setprecision(17)
      <<kModelTag<<';'<<kRecipeKind<<";mask="<<mask_mode_name(recipe.mask_mode)
      <<";GaussianNLL=1;temporal=0.1;lags=1,4,16;fixed-mean3;scale=sqrt2;Huber=1;originalQ-both-endpoints;variance-input-gradient-stop;dropout0"
      <<";quota=clamp-floor-ratio-times-observed-patches-plus-half-to-1-Mminus2"
      <<";contiguous=all-q-patches-observed-original-grid;nonwrapping;no-span-abstain;crop-before-mask;partial-final-patch"
      <<";span-start=uniform-legal-starts-mt19937_64-rejection-low-threshold;per-attempt-counter-seed;row-order=B*C";
  return out.str();
}

namespace mask_detail {
inline uint64_t bounded_index(std::mt19937_64 &generator,uint64_t bound){
  TORCH_CHECK(bound>0,"[cpb-v18] nonempty legal span set required");
  // Reject the low remainder to obtain exactly equal bucket cardinalities.
  const uint64_t threshold=(uint64_t{0}-bound)%bound;
  for(;;){const auto draw=generator();if(draw>=threshold)return draw%bound;}
}
inline std::vector<int64_t> legal_starts(const std::vector<bool> &observed,int64_t quota){
  TORCH_CHECK(quota>0&&quota<=static_cast<int64_t>(observed.size()),"[cpb-v18] positive quota within original grid required");
  std::vector<int64_t> starts;
  for(int64_t first=0;first<=static_cast<int64_t>(observed.size())-quota;++first){
    bool legal=true;for(int64_t j=0;j<quota;++j)legal=legal&&observed[static_cast<size_t>(first+j)];
    if(legal)starts.push_back(first);
  }
  return starts;
}
} // namespace mask_detail

struct MaskBatch : base_training::MaskBatch {
  int64_t selected_patch_count{0},unsupported_channels{0},no_legal_span_channels{0},insufficient_support_channels{0};
  torch::Tensor selected_starts,legal_span_counts,quotas;
};
inline MaskBatch make_mask(const RawHistory &raw,const torch::Tensor &cutoffs,const Config &config,
    int64_t base_seed,int64_t attempt,const Recipe &recipe={}){
  validate_recipe(recipe);family::validate_config(config);
  // Observation storage and support are retained before any selection reads it.
  const auto retained=family::boundary::retain(raw,cutoffs,config);const auto &c=config.level;
  const auto B=retained.raw.data.size(0),K=family::dimensions(config).patch_count;
  const auto padded=implementation::configurable_spectral_pad_observed(retained.raw.observed,c);
  const auto support=padded.to(torch::kCPU).reshape({B*c.channel_count,K,-1}).any(-1).contiguous();
  MaskBatch out;out.selected_starts=torch::full({B,c.channel_count},-1,torch::kInt64);
  out.legal_span_counts=torch::zeros({B,c.channel_count},torch::kInt64);
  out.quotas=torch::zeros({B,c.channel_count},torch::kInt64);
  auto observed=support.accessor<bool,2>();auto starts=out.selected_starts.accessor<int64_t,2>();
  auto counts=out.legal_span_counts.accessor<int64_t,2>();auto quotas=out.quotas.accessor<int64_t,2>();
  auto selected=torch::zeros(support.sizes(),torch::kBool);auto choices=selected.accessor<bool,2>();
  std::mt19937_64 generator(training_counter_seed(base_seed,attempt));
  for(int64_t row=0;row<B*c.channel_count;++row){
    std::vector<bool> grid(static_cast<size_t>(K));int64_t M=0;
    for(int64_t k=0;k<K;++k){grid[static_cast<size_t>(k)]=observed[row][k];M+=observed[row][k];}
    if(M<3){++out.insufficient_support_channels;continue;}
    const auto quota=std::clamp<int64_t>(static_cast<int64_t>(std::floor(c.mask_ratio*M+.5)),1,M-2);
    const auto b=row/c.channel_count,j=row%c.channel_count;quotas[b][j]=quota;
    if(recipe.mask_mode==MaskMode::scattered)continue;
    const auto legal=mask_detail::legal_starts(grid,quota);counts[b][j]=static_cast<int64_t>(legal.size());
    if(legal.empty()){++out.no_legal_span_channels;continue;}
    const auto first=legal[static_cast<size_t>(mask_detail::bounded_index(generator,static_cast<uint64_t>(legal.size())))];
    starts[b][j]=first;for(int64_t k=0;k<quota;++k)choices[row][first+k]=true;
  }
  if(recipe.mask_mode==MaskMode::scattered){
    // Preserve the complete existing shuffle implementation and its RNG draws.
    static_cast<base_training::MaskBatch&>(out)=base_training::make_mask(raw,cutoffs,config,base_seed,attempt);
  }else{
    const auto hidden=selected.reshape({B,c.channel_count,K,1,1}).expand({B,c.channel_count,K,c.patch_length,c.input_width})
        .reshape({B,c.channel_count,K*c.patch_length,c.input_width}).narrow(2,0,c.history_length).to(retained.raw.observed.device());
    static_cast<base_training::MaskBatch&>(out)=implementation::configurable_spectral_mask_from_hidden(retained.raw.observed,hidden,c);
  }
  const auto selected_padded=implementation::configurable_spectral_pad_observed(out.hidden,c);
  out.selected_patch_count=selected_padded.reshape({B,c.channel_count,K,-1}).any(-1).sum().item<int64_t>();
  out.unsupported_channels=B*c.channel_count-out.eligible_channels.sum().item<int64_t>();
  return out;
}
inline implementation::LocationScaleForwardTrace forward_at(Encoder &model,const RawHistory &raw,
    const torch::Tensor &cutoffs,const torch::Tensor &hidden,const FrozenScaler &scaler,
    const Recipe &recipe={},const torch::Tensor &context_deleted={}) {
  validate_recipe(recipe);return model->forward_at(raw,cutoffs,hidden,scaler,context_deleted);
}

struct StepResult : base_training::StepResult {
  std::array<int64_t,3> temporal_pair_counts{{0,0,0}};
  int64_t selected_patch_count{0},unsupported_channels{0},no_legal_span_channels{0},insufficient_support_channels{0};
  torch::Tensor selected_starts,legal_span_counts,quotas;
};

// Only mask selection changes. Keep the family's objective, backward, clipping
// and AdamW sequence; scattered mode is the unchanged-family control.
inline StepResult train_step(Encoder &model,const FrozenScaler &scaler,
    torch::optim::AdamW &optimizer,TrainingProgress &progress,const RawHistory &raw,
    const torch::Tensor &cutoffs,const Recipe &recipe={},const OptimizerConfig &options={},
    const torch::Tensor &context_deleted={}) {
  validate_recipe(recipe);
  TORCH_CHECK(model&&torch::GradMode::is_enabled(),"[cpb-v18] model and enabled gradients required");
  family::validate_config(model->config());base_training::validate_optimizer_config(options);
  TORCH_CHECK(model->causal_updates()==progress.completed_updates&&
      progress.attempted_batches<std::numeric_limits<int64_t>::max()-1,
      "[cpb-v18] exact live progress and incrementable attempted count required");
  base_training::validate_optimizer(model,optimizer,progress,options);
  TORCH_CHECK(scaler.identity()==model->input_scaler_identity(),"[cpb-v18] own frozen TRAIN scaler required");
  const auto masks=make_mask(raw,cutoffs,model->config(),progress.mask_seed,progress.attempted_batches,recipe);
  if(context_deleted.defined()) {
    TORCH_CHECK(context_deleted.scalar_type()==torch::kBool&&context_deleted.sizes()==masks.visible.sizes(),
        "[cpb-v18] matching logical-window bool context deletion required");
    TORCH_CHECK(!(context_deleted.to(masks.visible.device())&masks.visible.logical_not()).any().item<bool>(),
        "[cpb-v18] deleted context must be a subset of visible observations");
  }
  StepResult result;result.attempt_index=progress.attempted_batches;result.completed_updates=progress.completed_updates;
  result.selected_patch_count=masks.selected_patch_count;result.unsupported_channels=masks.unsupported_channels;
  result.no_legal_span_channels=masks.no_legal_span_channels;result.insufficient_support_channels=masks.insufficient_support_channels;
  result.selected_starts=masks.selected_starts;result.legal_span_counts=masks.legal_span_counts;result.quotas=masks.quotas;
  if(!masks.eligible_channels.any().item<bool>()) {
    optimizer.zero_grad(true);model->clear_gradients();++progress.attempted_batches;return result;
  }
  const auto bank_id=model->normalizer().identity();optimizer.zero_grad(true);model->train();
  const auto trace=forward_at(model,raw,cutoffs,masks.hidden,scaler,recipe,context_deleted);
  TORCH_CHECK(trace.nll.eligible_example_count>0,"[cpb-v18] supported objective required before backward");
  result.gaussian_nll=trace.nll.loss.item<double>();result.temporal_loss=trace.objective.temporal.loss.item<double>();
  result.total_loss=trace.output.loss.item<double>();result.eligible_examples=trace.nll.eligible_example_count;
  TORCH_CHECK(trace.objective.temporal.lags.size()==3,"[cpb-v18] unchanged ordered1/4/16 temporal terms required");
  for(size_t lag=0;lag<3;++lag)result.temporal_pair_counts[lag]=trace.objective.temporal.lags[lag].pair_count;
  result.target_cells=trace.nll.target_cell_count;trace.output.loss.backward();
  for(const auto &parameter:model->named_parameters()) {
    const auto gradient=parameter.value().grad();if(!gradient.defined())continue;
    TORCH_CHECK(parameter.value().requires_grad()&&gradient.is_cuda()&&torch::isfinite(gradient).all().item<bool>(),
        "[cpb-v18] finite learned CUDA gradients required");++result.gradient_tensors;
  }
  TORCH_CHECK(result.gradient_tensors>0,"[cpb-v18] objective must reach learned parameters");
  result.gradient_norm=torch::nn::utils::clip_grad_norm_(base_training::learned_parameters(model),options.clip_norm);
  TORCH_CHECK(std::isfinite(result.gradient_norm),"[cpb-v18] finite global gradient norm required");
  optimizer.step();++progress.completed_updates;++progress.attempted_batches;model->set_causal_updates(progress.completed_updates);
  optimizer.zero_grad(true);model->clear_gradients();model->core()->validate_state();model->core()->validate_fixed_prior();
  TORCH_CHECK(model->normalizer().identity()==bank_id&&scaler.identity()==model->input_scaler_identity(),
      "[cpb-v18] optimizer changed frozen preprocessing");
  base_training::validate_optimizer(model,optimizer,progress,options);
  result.updated=true;result.completed_updates=progress.completed_updates;return result;
}

namespace artifact_detail {
namespace d=family::checkpoint_detail::d;
inline void write_recipe(torch::serialize::OutputArchive &archive,const Recipe &recipe,const char *kind) {
  validate_recipe(recipe);
  archive.write("artifact_kind",d::text_tensor(kind),true);archive.write("model_tag",d::text_tensor(kModelTag),true);
  archive.write("recipe_kind",d::text_tensor(kRecipeKind),true);archive.write("recipe_binding",d::text_tensor(recipe_identity(recipe)),true);
  archive.write("mask_mode_i64",torch::tensor(static_cast<int64_t>(recipe.mask_mode),torch::kInt64),true);
  archive.write("temporal_weight_f64",torch::tensor(.1,torch::kFloat64),true);
  archive.write("implementation_payload_codec",d::text_tensor(kPayloadCodec),true);
  archive.write("implementation_payload_meaning",d::text_tensor(kPayloadMeaning),true);
}
inline Recipe read_recipe(torch::serialize::InputArchive &archive,const Recipe &expected,const char *kind) {
  validate_recipe(expected);torch::Tensor mode,weight;
  archive.read("mask_mode_i64",mode,true);archive.read("temporal_weight_f64",weight,true);
  TORCH_CHECK(d::read_text(archive,"artifact_kind")==kind&&d::read_text(archive,"model_tag")==kModelTag&&
      d::read_text(archive,"recipe_kind")==kRecipeKind&&d::read_text(archive,"recipe_binding")==recipe_identity(expected)&&
      d::read_text(archive,"implementation_payload_codec")==kPayloadCodec&&
      d::read_text(archive,"implementation_payload_meaning")==kPayloadMeaning&&
      mode.dim()==0&&mode.scalar_type()==torch::kInt64&&mode.item<int64_t>()==static_cast<int64_t>(expected.mask_mode)&&
      weight.dim()==0&&weight.scalar_type()==torch::kFloat64&&family::boundary::same_bits(weight,torch::tensor(.1,torch::kFloat64)),
      "[cpb-v18] typed artifact/version/mask/recipe/payload meaning mismatch");
  return expected;
}
inline std::string progress_binding(const TrainingIdentity &identity,const TrainingProgress &progress,const Recipe &recipe) {
  return recipe_identity(recipe)+":"+base_training::progress_binding(identity,progress);
}
} // namespace artifact_detail

struct Loaded {family::Loaded numerical;Recipe recipe;};
inline void save_checkpoint(const std::string &path,const Encoder &model,const FrozenScaler &scaler,
    int64_t updates,const Recipe &recipe={}) {
  TORCH_CHECK(model,"[cpb-v18] model required");model->validate_frozen();
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
  TORCH_CHECK(model&&model->causal_updates()==progress.completed_updates,"[cpb-v18] model and progress update count mismatch");
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
      torch::tensor(base_training::optimizer_config_values(options),torch::kFloat64)),"[cpb-v18] counter/optimizer recipe mismatch");
  TrainingIdentity actual{artifact_detail::d::read_text(archive,"input_source_identity"),
      artifact_detail::d::read_text(archive,"split_identity"),artifact_detail::d::read_text(archive,"recipe_identity")};
  base_training::common::validate_identity(actual);
  TORCH_CHECK(base_training::common::same_identity(actual,identity),"[cpb-v18] expected source/split/application recipe mismatch");
  const auto values=artifact_detail::d::read_vector(archive,"progress_v1",torch::kInt64,5);const auto p=values.accessor<int64_t,1>();
  TrainingProgress progress{p[0],p[1],p[2],p[3],p[4]};base_training::common::validate_progress(progress);
  TORCH_CHECK(artifact_detail::d::read_text(archive,"progress_binding")==artifact_detail::progress_binding(identity,progress,recipe),
      "[cpb-v18] progress/seed/recipe binding mismatch");
  archive.read("structural_encoder",encoder);auto child=family::checkpoint_detail::read(encoder,expected.level.device,&expected);
  TORCH_CHECK(child.causal_updates==progress.completed_updates&&artifact_detail::d::read_text(archive,"parameter_order")==base_training::parameter_order(child.model),
      "[cpb-v18] child update/parameter order mismatch");
  auto optimizer=base_training::make_optimizer(child.model,options);archive.read("optimizer",moments);
  base_training::common::validate_optimizer_archive(moments,artifact_detail::d::read_text(archive,"optimizer_parameter_keys"),
      base_training::learned_parameters(child.model).size());
  optimizer->load(moments);const auto steps=base_training::validate_optimizer(child.model,*optimizer,progress,options);
  TORCH_CHECK(artifact_detail::d::read_text(archive,"optimizer_state_identity")==base_training::optimizer_identity(child.model,*optimizer)&&
      torch::equal(artifact_detail::d::read_vector(archive,"optimizer_steps_v1",torch::kInt64,steps.size()),torch::tensor(steps,torch::kInt64)),
      "[cpb-v18] optimizer moment content/order/step mismatch");
  child.model->train();child.model->clear_gradients();
  return{child.config,child.model,child.scaler,std::move(optimizer),actual,progress,options,recipe};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::v18
