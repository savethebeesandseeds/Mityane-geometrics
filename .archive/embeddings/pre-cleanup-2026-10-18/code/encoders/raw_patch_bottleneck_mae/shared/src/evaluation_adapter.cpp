// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/masking.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {

namespace ev=embedding::evaluation;
using training_detail::mixed;
using training_detail::counter_seed;
using training_detail::sampled_indices;
using training_detail::selected;

void require(bool ok,const std::string &message) {
  if(!ok)throw std::runtime_error("[rpb evaluation adapter] "+message);
}

void write_text(torch::serialize::OutputArchive &archive,const char *key,const std::string &text) {
  archive.write(key,embedding::archive::text_tensor(text),true);
}

std::string source_manifest(const std::vector<std::string> &ids) {
  // Length prefixes preserve arbitrary identifiers without ambiguous separators.
  std::string text;
  for(const auto &id:ids)text+=std::to_string(id.size())+":"+id;
  return text;
}

std::string manifest_identity(const std::string &text) {
  uint64_t hash=14695981039346656037ULL;
  for(unsigned char byte:text){hash^=byte;hash*=1099511628211ULL;}
  std::ostringstream out;
  out<<"rpb-fit-source-manifest-fnv1a-v1-"<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;
  return out.str();
}

Input protocol_input(const embedding::Batch &batch,const Config &config,
                     const std::vector<int64_t> &channel_ids,double interval,double endpoint) {
  Input input;
  input.data=batch.data;input.observed=batch.feature_mask;
  input.channel_ids=torch::tensor(channel_ids,torch::kInt64);
  input.endpoints=torch::full({batch.data.size(0)},endpoint,torch::kFloat64);
  input.sampling_interval=interval;
  validate_input(input,config);
  return input;
}

struct EvaluationState {
  std::string prefix,provenance;
  Settings settings;
  Model model{nullptr};
  FrozenScaler scaler;
  std::string schema_id,dataset_id,checkpoint_source;
  int64_t attempted{0},completed{0};
  bool supplied_checkpoint{false};
  bool untrained{false};
};

std::string weight_training_policy(const EvaluationState &state) {
  if(state.untrained)return "zero_model_updates;permitted_scaler_fit_only";
  return state.supplied_checkpoint?"frozen_supplied_checkpoint;no_evaluation_model_updates":
      "fresh_permitted_training_observations_only";
}

std::string training_producer(const EvaluationState &state) {
  if(state.untrained)return "none";
  return state.supplied_checkpoint?"unrecorded_checkpoint_producer;core_writer_field_only":
      EVALUATION_SOURCE_ID;
}

} // namespace

ev::FeatureProviderFactory make_evaluation_provider(const EvaluationOptions &options) {
  require(options.pretraining_updates>=0,"pretraining_updates must be nonnegative");
  require(!options.surface_prefix.empty() && options.surface_prefix.find_first_not_of(
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")==std::string::npos,
      "surface_prefix must be a nonempty alphanumeric/underscore namespace");
  require(options.checkpoint_path.empty() ||
      (options.config_path.empty() && options.pretraining_updates==0),
      "a frozen checkpoint cannot be combined with config overrides or fresh training");
  const auto settings=options.config_path.empty()?default_settings():read_settings(options.config_path);
  return [settings,options](const ev::ProviderFitInput &fit) {
    require(fit.shape.channel_count>0 && fit.shape.history_length>0 && fit.shape.input_width>0,
        "the evaluation card must declare positive C/H/F dimensions");
    require(static_cast<int64_t>(fit.channel_ids.size())==fit.shape.channel_count &&
        std::set<int64_t>(fit.channel_ids.begin(),fit.channel_ids.end()).size()==fit.channel_ids.size(),
        "the evaluation card must declare unique semantic channel IDs");
    require(!fit.feature_units.empty(),"the evaluation card must declare feature units");
    require(std::isfinite(fit.sampling_interval) && fit.sampling_interval>0 && std::isfinite(fit.endpoint),
        "the evaluation card must declare a positive interval and finite endpoint");
    require(fit.training_observations.data.defined() && fit.training_observations.data.dim()==4 &&
        fit.training_source_ids.size()==static_cast<size_t>(fit.training_observations.data.size(0)),
        "training source IDs must accompany every permitted fit observation");
    for(const auto &id:fit.training_source_ids)require(!id.empty(),"empty training source ID");

    auto active=settings;
    if(options.config_path.empty() && options.checkpoint_path.empty()) {
      active.model.channel_count=fit.shape.channel_count;
      active.model.history_length=fit.shape.history_length;
      active.model.input_width=fit.shape.input_width;
      active.model.channel_ids=fit.channel_ids;
      active.model.sampling_interval=fit.sampling_interval;
    }
    active.model.device=torch::kCPU;
    Checkpoint checkpoint;
    if(!options.checkpoint_path.empty()) {
      checkpoint=load_checkpoint(options.checkpoint_path);
      active=checkpoint.settings;active.model.device=torch::kCPU;
    } else {
      // Only run metadata changes: the existing counter streams already use fit.seed.
      active.seed=static_cast<int64_t>(fit.seed&0x7fffffffffffffffULL);
      if(options.pretraining_updates>0)active.steps=options.pretraining_updates;
    }
    validate_settings(active);
    require(active.model.channel_mixer_placement==0,
        "historical CPU evaluation adapter rejects early channel mixer placement");
    require(!options.require_channel_mixer || active.model.channel_mixer_layers>0,
        "the channel-mixer registration requires an enabled mixer configuration/checkpoint");
    require(active.model.channel_count==fit.shape.channel_count &&
        active.model.history_length==fit.shape.history_length && active.model.input_width==fit.shape.input_width,
        "configuration/checkpoint dimensions differ from the evaluation card");
    const auto configured_ids=resolved_channel_ids(active.model);
    require(std::set<int64_t>(configured_ids.begin(),configured_ids.end())==
        std::set<int64_t>(fit.channel_ids.begin(),fit.channel_ids.end()),
        "configuration/checkpoint semantic channel IDs differ from the evaluation card");
    require(std::abs(active.model.sampling_interval-fit.sampling_interval)<=
        1e-12*std::max(1.0,std::abs(active.model.sampling_interval)),
        "configuration/checkpoint sampling interval differs from the evaluation card");

    // Fit input has no labels, clean hidden signals or held-out observations.
    auto raw=protocol_input(fit.training_observations,active.model,fit.channel_ids,
                            fit.sampling_interval,fit.endpoint);
    const auto training=describe_dataset(raw,active.model,fit.feature_units);
    const auto scaler=options.checkpoint_path.empty()?fit_scaler(raw,active.model):checkpoint.scaler;
    if(!options.checkpoint_path.empty())
      require(checkpoint.schema_id==training.schema_id,
          "evaluation source precision/units/schema differ from the loaded checkpoint");
    const auto pretraining_id=options.checkpoint_path.empty()?
        (options.pretraining_updates>0?training.dataset_id:"none"):checkpoint.dataset_id;
    const auto scaler_fit_id=options.checkpoint_path.empty()?training.dataset_id:checkpoint.scaler_fit_dataset_id;
    const auto core_source=workflow_source_fingerprint();
    const auto init_seed=mixed(fit.seed^0x7270622d696e6974ULL);
    auto states=std::make_shared<std::vector<EvaluationState>>();
    torch::manual_seed(init_seed);auto random=Model(active.model);random->eval();
    states->push_back({options.surface_prefix+"_untrained","matched architecture random weights; scaler="+scaler.identity()+
        "; scaler_fit_dataset="+scaler_fit_id,active,random,scaler,training.schema_id,
        "none",core_source,0,0,false,true});
    if(!options.checkpoint_path.empty()) {
      checkpoint.model->eval();
      states->push_back({options.surface_prefix+"_trained","frozen supplied checkpoint="+options.checkpoint_path+
          "; pretraining_dataset="+checkpoint.dataset_id+"; scaler="+scaler.identity()+
          "; scaler_fit_dataset="+scaler_fit_id+"; no evaluation refit",active,checkpoint.model,
          scaler,checkpoint.schema_id,checkpoint.dataset_id,checkpoint.source_fingerprint,
           checkpoint.attempted_steps,checkpoint.completed_steps,true});
    } else if(options.pretraining_updates>0) {
      torch::manual_seed(init_seed);auto model=Model(active.model);model->train();
      torch::optim::AdamW optimizer(model->parameters(),
          torch::optim::AdamWOptions(active.learning_rate).weight_decay(active.weight_decay));
      int64_t attempted=0,completed=0;
      const auto training_seed=static_cast<int64_t>(fit.seed&0x7fffffffffffffffULL);
      while(completed<options.pretraining_updates && attempted<active.attempt_limit) {
        auto indices=sampled_indices(raw.data.size(0),active.batch_size,training_seed,attempted);
        auto batch=scaler.transform(selected(raw,indices),active.model);
        auto mask=make_training_mask(batch.observed,active.model,
            counter_seed(training_seed,attempted,0x6d61736bULL));
        torch::manual_seed(counter_seed(training_seed,attempted,0x746f726368ULL));++attempted;
        if(!mask.eligible_channels.any().item<bool>())continue;
        optimizer.zero_grad();auto result=model->forward(batch,mask.hidden);
        require(result.eligible_example_count>0 && torch::isfinite(result.loss).all().item<bool>(),
            "invalid evaluation pretraining loss");
        result.loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(),active.gradient_clip_norm>0?
            active.gradient_clip_norm:std::numeric_limits<double>::infinity(),2.0,true);
        optimizer.step();++completed;
      }
      require(completed==options.pretraining_updates,"evaluation pretraining exhausted its attempt budget");
      model->eval();
      states->push_back({options.surface_prefix+"_trained","fresh unsupervised pretraining on permitted training sources only; dataset="+
          training.dataset_id+"; scaler="+scaler.identity()+"; completed="+std::to_string(completed),
          active,model,scaler,training.schema_id,training.dataset_id,core_source,attempted,completed});
    }

    ev::FeatureProvider provider;
    provider.provenance=std::string(output_semantics(active.model))+
        "; channel concatenation preserves declared semantic order; float64 centering before float32; fit API exposes no labels or held-out observations";
    const auto source_text=source_manifest(fit.training_source_ids);
    provider.audit_fields={{"provider_id",kEncoderId},{"protocol_id",fit.protocol_id},
        {"surface_prefix",options.surface_prefix},
        {"output_semantics",output_semantics(active.model)},
        {"reconstruction_export_semantics",reconstruction_output_semantics(active.model)},
        {"channel_mixer_layers",std::to_string(active.model.channel_mixer_layers)},
        {"fit_policy",options.checkpoint_path.empty()?"permitted_training_observations_only":"frozen_checkpoint_no_refit"},
        {"resolved_settings",settings_text(active)},
        {"actual_training_seed",std::to_string(active.seed)},
        {"training_seed_policy",options.checkpoint_path.empty()?
            "fit_seed_and_0x7fffffffffffffff":"preserved_checkpoint_pretraining_seed"},
        {"rng_policy","splitmix64-counter-rows-masks-torch-attempt-v1"},
        {"fit_source_manifest_id",manifest_identity(source_text)},
        {"fit_observation_dataset_id",training.dataset_id},{"pretraining_dataset_id",pretraining_id},
        {"permitted_fit_observation_dataset_id",training.dataset_id},
        {"feature_units",fit.feature_units},
        {"scaler_fit_dataset_id",scaler_fit_id},{"preprocessing_id",scaler.identity()},
        {"encoder_source_fingerprint",core_source},{"evaluation_source_fingerprint",EVALUATION_SOURCE_ID},
        {"source_fingerprint_scope","core_encoder/writer identity is separate from adapter producer;fresh training producer is evaluation source"},
        {"initialization_seed",std::to_string(init_seed)},
        {"fresh_pretraining_updates",std::to_string(options.pretraining_updates)},
        {"training_observation_rows",std::to_string(fit.training_source_ids.size())},
        {"training_source_groups",std::to_string(std::set<std::string>(fit.training_source_ids.begin(),fit.training_source_ids.end()).size())}};
    for(const auto &state:*states) {
      provider.audit_fields.emplace(state.prefix+"_attempted_steps",std::to_string(state.attempted));
      provider.audit_fields.emplace(state.prefix+"_completed_steps",std::to_string(state.completed));
      provider.audit_fields.emplace(state.prefix+"_model_weight_update_budget",std::to_string(state.completed));
      provider.audit_fields.emplace(state.prefix+"_pretraining_dataset_id",state.dataset_id);
      provider.audit_fields.emplace(state.prefix+"_weight_training_policy",weight_training_policy(state));
      provider.audit_fields.emplace(state.prefix+"_training_producer_source_fingerprint",training_producer(state));
      provider.surfaces.emplace(state.prefix+"_global",ev::SurfaceDescription{
          ev::SurfaceKind::global,global_readout_description(state.settings.model,false),{}});
      provider.surfaces.emplace(state.prefix+"_channel_concatenation",ev::SurfaceDescription{
          ev::SurfaceKind::channel_concatenation,"every declared channel has visible observations; concatenate independent local vectors",fit.channel_ids});
      if(state.settings.model.channel_mixer_layers>0) {
        provider.surfaces.emplace(state.prefix+"_contextual_global",ev::SurfaceDescription{
            ev::SurfaceKind::global,global_readout_description(state.settings.model,true),{}});
        provider.surfaces.emplace(state.prefix+"_contextual_channel_concatenation",ev::SurfaceDescription{
            ev::SurfaceKind::channel_concatenation,"every declared channel has visible observations; concatenate observed contextual vectors after original-patch-aligned channel attention",fit.channel_ids});
      }
    }
    provider.extract=[states,ids=fit.channel_ids,interval=fit.sampling_interval,endpoint=fit.endpoint](const embedding::Batch &batch) {
      torch::NoGradGuard no_grad;ev::FeatureMap surfaces;
      for(auto &state:*states) {
        state.model->eval();auto input=protocol_input(batch,state.settings.model,ids,interval,endpoint);
        auto output=state.model->encode(state.scaler.transform(input,state.settings.model));
        surfaces.emplace(state.prefix+"_global",ev::FeatureSurface{
            output.z_global.to(torch::kCPU),output.sample_valid_mask.to(torch::kCPU),state.provenance});
        surfaces.emplace(state.prefix+"_channel_concatenation",ev::FeatureSurface{
            output.z_local.flatten(1).to(torch::kCPU),output.channel_valid_mask.all(1).to(torch::kCPU),
            state.provenance+"; independent local vectors in declared channel order; requires every channel observed-valid"});
        if(state.settings.model.channel_mixer_layers>0) {
          surfaces.emplace(state.prefix+"_contextual_global",ev::FeatureSurface{
              output.z_contextual_global.to(torch::kCPU),output.sample_valid_mask.to(torch::kCPU),
              state.provenance+(state.settings.model.global_bottleneck_mode==0 ?
                "; equal-valid observed contextual mean; local branch shares contextually trained parameters" :
                std::string("; ")+global_readout_description(state.settings.model,true)+
                "; exact trained global reconstruction bottleneck; local branch shares globally trained parameters")});
          surfaces.emplace(state.prefix+"_contextual_channel_concatenation",ev::FeatureSurface{
              output.z_contextual.flatten(1).to(torch::kCPU),output.channel_valid_mask.all(1).to(torch::kCPU),
              state.provenance+"; observed contextual vectors in declared channel order; no inferred missing channels"});
        }
      }
      return surfaces;
    };
    provider.save_assets=[states,scaler_fit_id,source_text,permitted_fit_id=training.dataset_id,
                         protocol_id=fit.protocol_id,init_seed,units=fit.feature_units,
                         ids=fit.channel_ids,interval=fit.sampling_interval,endpoint=fit.endpoint](const std::string &directory) {
      for(const auto &state:*states) {
        save_scaler((std::filesystem::path(directory)/(state.prefix+"-scaler.pt")).string(),
            state.scaler,state.settings.model,state.schema_id,scaler_fit_id);
        torch::serialize::OutputArchive archive,weights;
        write_text(archive,"encoder_id",kEncoderId);archive.write("format_version",torch::tensor(int64_t{1}),true);
        write_text(archive,"artifact_kind","rpb_frozen_feature_model_v1");
        write_text(archive,"settings",settings_text(state.settings));write_text(archive,"schema_id",state.schema_id);
        write_text(archive,"pretraining_dataset_id",state.dataset_id);write_text(archive,"preprocessing_id",state.scaler.identity());
        write_text(archive,"permitted_fit_observation_dataset_id",permitted_fit_id);
        write_text(archive,"feature_units",units);
        archive.write("channel_order",torch::tensor(ids,torch::kInt64),true);
        archive.write("sampling_interval",torch::tensor(interval,torch::kFloat64),true);
        archive.write("endpoint",torch::tensor(endpoint,torch::kFloat64),true);
        write_text(archive,"scaler_fit_dataset_id",scaler_fit_id);write_text(archive,"output_semantics",output_semantics(state.settings.model));
        write_text(archive,"provenance",state.provenance);write_text(archive,"protocol_id",protocol_id);
        write_text(archive,"fit_source_manifest",source_text);write_text(archive,"fit_source_manifest_id",manifest_identity(source_text));
        write_text(archive,"checkpoint_source_fingerprint",state.checkpoint_source);
        write_text(archive,"inference_source_fingerprint",workflow_source_fingerprint());
        write_text(archive,"evaluation_source_fingerprint",EVALUATION_SOURCE_ID);
        write_text(archive,"training_producer_source_fingerprint",training_producer(state));
        write_text(archive,"initialization_seed",state.supplied_checkpoint?"unrecorded_supplied_checkpoint":std::to_string(init_seed));
        write_text(archive,"weight_training_policy",weight_training_policy(state));
        write_text(archive,"rng_policy","splitmix64-counter-rows-masks-torch-attempt-v1");
        write_text(archive,"source_fingerprint_scope",
            "checkpoint_source_fingerprint identifies the core/writer;fresh adapter training producer is training_producer_source_fingerprint");
        archive.write("actual_training_seed",torch::tensor(state.settings.seed),true);
        archive.write("model_weight_update_budget",torch::tensor(state.completed),true);
        archive.write("attempted_steps",torch::tensor(state.attempted),true);
        archive.write("completed_steps",torch::tensor(state.completed),true);
        state.model->save(weights);archive.write("model",weights);
        embedding::archive::save_archive((std::filesystem::path(directory)/(state.prefix+"-model.pt")).string(),archive);
      }
    };
    return provider;
  };
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
