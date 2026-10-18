// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/paired_pooling_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include <ATen/Context.h>
#include <filesystem>
#include <memory>
#include <set>
#include <stdexcept>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
void require(bool value,const std::string &message) {
  if (!value) throw std::runtime_error("[RPB paired pooling adapter] " + message);
}
struct RngIsolation {
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RngIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t index=0;index<at::getNumGPUs();++index)
      generators.push_back(at::globalContext().defaultGenerator(
          at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(index))));
    for (const auto &generator:generators) states.push_back(generator.get_state().clone());
  }
  ~RngIsolation() noexcept {
    try { for (size_t index=0;index<generators.size();++index) generators[index].set_state(states[index]); }
    catch (...) { std::terminate(); }
  }
};
Input raw_input(const embedding::Batch &batch,const Config &config,const ev::ProviderFitInput &fit) {
  Input input{batch.data,batch.feature_mask,torch::tensor(fit.channel_ids,torch::kInt64),
      torch::full({batch.data.size(0)},fit.endpoint,torch::kFloat64),fit.sampling_interval};
  validate_input(input,config);
  return input;
}
void freeze(Checkpoint &checkpoint) {
  checkpoint.model->eval();
  for (auto &parameter:checkpoint.model->parameters()) parameter.set_requires_grad(false);
}
std::string source_manifest(const std::vector<std::string> &ids) {
  std::string result;
  for (const auto &id:ids) result += std::to_string(id.size()) + ':' + id;
  return result;
}
void validate_training(const Checkpoint &checkpoint,const ev::ProviderFitInput &fit) {
  require(checkpoint.settings.model.channel_count==fit.shape.channel_count &&
      checkpoint.settings.model.history_length==fit.shape.history_length &&
      checkpoint.settings.model.input_width==fit.shape.input_width,
      "retained dimensions differ from declared observations");
  const auto described=describe_dataset(raw_input(fit.training_observations,checkpoint.settings.model,fit),
      checkpoint.settings.model,fit.feature_units);
  require(described.dataset_id==checkpoint.dataset_id && described.schema_id==checkpoint.schema_id &&
      checkpoint.scaler_fit_dataset_id==described.dataset_id,
      "retained model/scaler were not fitted on these exact TRAIN observations");
  require(checkpoint.settings.seed==static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL),
      "retained counter seed differs from the paired master");
}
bool specific_pool(const std::string &name) {
  return name.rfind("global_pool_",0)==0 || name.rfind("global_patch_pool_",0)==0;
}
} // namespace

ev::CurveSnapshot make_retained_curve_snapshot(const std::string &path,const ev::ProviderFitInput &fit) {
  const RngIsolation isolation;
  auto checkpoint=std::make_shared<Checkpoint>(load_checkpoint(path,torch::kCUDA));
  require(checkpoint->settings.model.channel_mixer_placement==0,
      "historical retained pooling adapter rejects early channel mixer placement");
  require(checkpoint->settings.model.global_bottleneck_mode==2 &&
      checkpoint->settings.model.channel_mixer_layers==1 && checkpoint->settings.model.export_width==32,
      "retained reference must be the native RPB-v4 global bottleneck");
  validate_training(*checkpoint,fit);
  freeze(*checkpoint);
  EvaluationOptions options; options.checkpoint_path=path; options.surface_prefix="curve_checkpoint";
  const auto provider=make_evaluation_provider(options)(fit);
  const std::string base="curve_checkpoint_trained_contextual";
  ev::CurveSnapshot snapshot;
  snapshot.features.provenance=provider.provenance + "; retained paired-pooling reference; no refit";
  snapshot.features.audit_fields=provider.audit_fields;
  for (auto field=snapshot.features.audit_fields.begin();field!=snapshot.features.audit_fields.end();) {
    if (field->first.rfind("curve_checkpoint_untrained_",0)==0) field=snapshot.features.audit_fields.erase(field);
    else ++field;
  }
  snapshot.features.audit_fields["snapshot_policy"]="retained_CPU_serving_and_CUDA_reconstruction;no_fit_no_training";
  snapshot.features.audit_fields["checkpoint_path"]=path;
  snapshot.features.audit_fields["completed_steps"]=std::to_string(checkpoint->completed_steps);
  snapshot.features.audit_fields["retained_checkpoint_writer_source_fingerprint"]=checkpoint->source_fingerprint;
  snapshot.features.audit_fields["snapshot_loader_source_fingerprint"]=EVALUATION_SOURCE_ID;
  torch::serialize::InputArchive producer_audit;
  producer_audit.load_from(path + ".audit.pt",torch::kCPU);
  for (const std::string name:{"training_producer_source_fingerprint","core_writer_source_fingerprint",
      "initialization_seed","actual_training_seed","fit_source_manifest_id","rng_policy","sampling_policy"}) {
    torch::Tensor value; producer_audit.read(name,value,true);
    snapshot.features.audit_fields["retained_"+name]=embedding::archive::tensor_text(value);
  }
  if (checkpoint->training_policy_id == "rpb-training-context-balanced-030-v1") {
    // Bind the balanced training witness without changing ordinary inference
    // or any historical uniform-policy metadata contract.
    for (const std::string name:{"model_tag","training_policy_id","context_deletion_ratio",
        "context_deletion_stream","context_deletion_rng_policy","context_deletion_repair_policy",
        "context_deletion_visibility_policy","context_deletion_count_policy","context_deletion_resume_policy",
        "context_deletion_schedule_policy","context_deletion_rate_scope","context_deletion_branch_count_policy",
        "context_deletion_skip_policy","context_ordinary_attempts","context_deletion_attempts"}) {
      torch::Tensor value; producer_audit.read(name,value,true);
      snapshot.features.audit_fields[name]=embedding::archive::tensor_text(value);
    }
    require(snapshot.features.audit_fields.at("training_policy_id")==checkpoint->training_policy_id &&
        snapshot.features.audit_fields.at("model_tag")=="RPB-v8" &&
        snapshot.features.audit_fields.at("context_deletion_ratio")=="0.30" &&
        snapshot.features.audit_fields.at("context_deletion_schedule_policy")=="absolute-attempt-even-ordinary-odd-coordinate30-v1" &&
        snapshot.features.audit_fields.at("context_deletion_rate_scope")=="active-odd-attempts-only;not-uniform-effective-rate" &&
        snapshot.features.audit_fields.at("context_deletion_branch_count_policy")=="absolute-attempt-prefix;no-skips;ordinary-plus-deletion-equals-attempted" &&
        snapshot.features.audit_fields.at("context_deletion_skip_policy")=="abort-before-update-on-ineligible-original-mask;no-replacement-or-schedule-shift",
        "balanced retained companion policy differs from its checkpoint");
    for (const std::string name:{"context_requested_deleted_coordinates","context_actual_deleted_coordinates",
        "context_restored_coordinates"}) {
      torch::Tensor value; producer_audit.read(name,value,true);
      require(value.scalar_type()==torch::kInt64 && value.dim()==0 && value.numel()==1 && value.item<int64_t>()>=0,
          "balanced retained coordinate count is invalid: "+name);
      snapshot.features.audit_fields[name]=std::to_string(value.item<int64_t>());
    }
    for (const std::string name:{"context_ordinary_attempts","context_deletion_attempts"}) {
      torch::Tensor value; producer_audit.read(name+"_value",value,true);
      require(value.scalar_type()==torch::kInt64 && value.dim()==0 && value.numel()==1 && value.item<int64_t>()>=0 &&
          snapshot.features.audit_fields.at(name)==std::to_string(value.item<int64_t>()),
          "balanced retained branch text/typed count differs: "+name);
    }
    require(checkpoint->attempted_steps==checkpoint->completed_steps &&
        snapshot.features.audit_fields.at("context_ordinary_attempts")==
            std::to_string(checkpoint->attempted_steps/2+checkpoint->attempted_steps%2) &&
        snapshot.features.audit_fields.at("context_deletion_attempts")==
            std::to_string(checkpoint->attempted_steps/2),
        "balanced retained absolute-attempt prefix differs");
    snapshot.features.provenance += "; model_tag=RPB-v8; training_policy="+checkpoint->training_policy_id;
  }
  snapshot.features.audit_fields["curve_checkpoint_trained_training_producer_source_fingerprint"]=
      snapshot.features.audit_fields.at("retained_training_producer_source_fingerprint");
  snapshot.features.audit_fields[base+"_training_producer_source_fingerprint"]=
      snapshot.features.audit_fields.at("retained_training_producer_source_fingerprint");
  snapshot.features.surfaces.emplace("curve_global",provider.surfaces.at(base + "_global"));
  snapshot.features.surfaces.emplace("curve_channel_concatenation",provider.surfaces.at(base + "_channel_concatenation"));
  snapshot.features.extract=[provider,base](const embedding::Batch &batch) {
    const auto all=provider.extract(batch);
    ev::FeatureMap result;
    result.emplace("curve_global",all.at(base + "_global"));
    result.emplace("curve_channel_concatenation",all.at(base + "_channel_concatenation"));
    return result;
  };
  snapshot.features.save_assets=[fields=snapshot.features.audit_fields](const std::string &directory) {
    const auto path=(fs::path(directory)/"retained-snapshot-audit.pt").string();
    require(!fs::exists(path),"retained snapshot audit already exists");
    torch::serialize::OutputArchive output;
    for (const auto &[name,value]:fields) output.write(name,embedding::archive::text_tensor(value),true);
    embedding::archive::save_archive(path,output);
  };
  auto metadata=fit; metadata.training_observations={}; metadata.training_source_ids.clear();
  snapshot.reconstruct=[checkpoint,metadata](const embedding::Batch &batch,const torch::Tensor &hidden) {
    torch::NoGradGuard no_grad;
    const auto normalized=checkpoint->scaler.transform(raw_input(batch,checkpoint->settings.model,metadata),checkpoint->settings.model);
    const auto output=checkpoint->model->forward(normalized,hidden);
    require(output.reconstruction.is_cuda() && torch::isfinite(output.reconstruction).all().item<bool>(),
        "retained reconstruction must be finite CUDA output");
    return ev::CurveReconstruction{output.reconstruction.detach().to(torch::kCPU),
        normalized.data.detach().to(torch::kCPU),output.eligible_channels.to(torch::kCPU)};
  };
  return snapshot;
}

std::map<std::string,std::string> audit_pooling_initialization(const std::string &candidate_path,
    const ev::RetainedPoolingCohort &reference,const ev::ProviderFitInput &fit) {
  const RngIsolation isolation;
  const auto candidate=load_checkpoint(candidate_path,torch::kCPU);
  const auto initial=load_checkpoint(reference.reference_initial_checkpoint,torch::kCPU);
  const auto selected=load_checkpoint(reference.reference_checkpoint,torch::kCPU);
  require(candidate.settings.model.global_bottleneck_mode==3 &&
      initial.settings.model.global_bottleneck_mode==2 && selected.settings.model.global_bottleneck_mode==2,
      "paired audit requires RPB-v5 point0 and RPB-v4 point0/512");
  for (const auto *checkpoint:{&candidate,&initial,&selected}) validate_training(*checkpoint,fit);
  require(candidate.completed_steps==0 && candidate.attempted_steps==0 &&
      initial.completed_steps==0 && initial.attempted_steps==0 &&
      selected.completed_steps==512 && selected.attempted_steps==512,
      "paired counters differ from fixed unskipped point0/512 reference");
  auto reconciled=candidate.settings;
  reconciled.model.global_bottleneck_mode=2; reconciled.steps=selected.settings.steps;
  require(settings_text(reconciled)==settings_text(selected.settings) &&
      settings_text(initial.settings)==settings_text(selected.settings),
      "a setting beyond the declared pooling operation or completed session budget changed");
  require(candidate.scaler.identity()==initial.scaler.identity() && initial.scaler.identity()==selected.scaler.identity(),
      "paired TRAIN scalers differ");
  const auto same_scaler=[](const FrozenScaler &left,const FrozenScaler &right) {
    return torch::equal(left.mean,right.mean) && torch::equal(left.scale,right.scale) &&
        torch::equal(left.count,right.count) && torch::equal(left.channel_ids,right.channel_ids) &&
        torch::equal(left.floor_applied,right.floor_applied) && left.scale_floor==right.scale_floor;
  };
  require(same_scaler(candidate.scaler,initial.scaler) && same_scaler(initial.scaler,selected.scaler),
      "paired TRAIN scaler tensors/floors are not exact");
  const auto old=initial.model->named_parameters(),fresh=candidate.model->named_parameters();
  int64_t common_parameters=0,common_tensors=0;
  for (const auto &parameter:old) {
    if (specific_pool(parameter.key())) continue;
    require(fresh.contains(parameter.key()) && torch::equal(parameter.value(),fresh[parameter.key()]),
        "common initial weights differ: " + parameter.key());
    common_parameters += parameter.value().numel(); ++common_tensors;
  }
  for (const auto &parameter:fresh)
    require(specific_pool(parameter.key()) || old.contains(parameter.key()),
        "unregistered additional common module: " + parameter.key());
  require(common_parameters>0 && common_tensors>0,"no common initialized modules were verified");
  const auto read_audit=[&](const std::string &path) {
    torch::serialize::InputArchive archive; archive.load_from(path + ".audit.pt",torch::kCPU);
    std::map<std::string,std::string> fields;
    for (const std::string key:{"fit_source_manifest","initialization_seed","actual_training_seed","rng_policy","sampling_policy"}) {
      torch::Tensor value; archive.read(key,value,true); fields.emplace(key,embedding::archive::tensor_text(value));
    }
    return fields;
  };
  const auto reference_audit=read_audit(reference.reference_initial_checkpoint),candidate_audit=read_audit(candidate_path);
  require(reference_audit==candidate_audit && candidate_audit.at("fit_source_manifest")==source_manifest(fit.training_source_ids),
      "paired source order, initialization or attempted-row/mask counter streams differ");
  return {{"common_parameters_exact","true"},{"common_tensors",std::to_string(common_tensors)},
      {"common_parameter_count",std::to_string(common_parameters)},{"scaler_exact","true"},
      {"training_dataset_exact","true"},{"counter_streams_exact","true"},
      {"training_dataset_id",candidate.dataset_id},{"preprocessing_id",candidate.scaler.identity()},
      {"initialization_seed",candidate_audit.at("initialization_seed")},
      {"training_seed",candidate_audit.at("actual_training_seed")},
      {"scope","same common point0 modules/scaler/data/source order; different global pooling; reference512 has no skips"}};
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
