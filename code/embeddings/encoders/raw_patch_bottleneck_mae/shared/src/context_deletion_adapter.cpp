// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <ATen/Context.h>
#include <cmath>
#include <stdexcept>

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev=embedding::evaluation;
void require(bool value,const std::string &message) {
  if(!value)throw std::runtime_error("[RPB context deletion audit] "+message);
}
struct RuntimeIsolation {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for(size_t index=0;index<at::getNumGPUs();++index)
      generators.push_back(at::globalContext().defaultGenerator(
          at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(index))));
    for(const auto &generator:generators)states.push_back(generator.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try {
      for(size_t index=0;index<generators.size();++index)generators[index].set_state(states[index]);
      at::set_num_threads(threads);
    }catch(...){std::terminate();}
  }
};
std::string source_manifest(const std::vector<std::string> &ids) {
  std::string result;
  for(const auto &id:ids)result+=std::to_string(id.size())+':'+id;
  return result;
}
void validate_training(const Checkpoint &checkpoint,const ev::ProviderFitInput &fit) {
  const auto &config=checkpoint.settings.model;
  require(config.channel_mixer_placement==0,
      "historical context deletion admission rejects early channel mixer placement");
  require(config.global_bottleneck_mode==2 && config.channel_mixer_layers==1 && config.export_width==32 &&
      fit.shape.dtype==torch::kFloat64 && fit.shape.device.is_cpu() &&
      fit.shape.channel_count==config.channel_count && fit.shape.history_length==config.history_length &&
      fit.shape.input_width==config.input_width && fit.training_observations.data.defined() &&
      fit.training_observations.data.device().is_cpu() && fit.training_observations.data.scalar_type()==torch::kFloat64 &&
      fit.training_source_ids.size()==static_cast<size_t>(fit.training_observations.data.size(0)) &&
      fit.protocol_id=="native-curve-v1/lag_sign","same native v4 architecture and original TRAIN metadata required");
  Input raw{fit.training_observations.data,fit.training_observations.feature_mask,
      torch::tensor(fit.channel_ids,torch::kInt64),
      torch::full({fit.training_observations.data.size(0)},fit.endpoint,torch::kFloat64),fit.sampling_interval};
  validate_input(raw,config);
  const auto described=describe_dataset(raw,config,fit.feature_units);
  require(described.dataset_id==checkpoint.dataset_id && described.schema_id==checkpoint.schema_id &&
      checkpoint.scaler_fit_dataset_id==described.dataset_id &&
      checkpoint.settings.seed==static_cast<int64_t>(fit.seed&0x7fffffffffffffffULL),
      "checkpoint original TRAIN/schema/scaler-fit/counter identity differs");
}
bool same_scaler(const FrozenScaler &left,const FrozenScaler &right) {
  return left.identity()==right.identity() && torch::equal(left.mean,right.mean) &&
      torch::equal(left.scale,right.scale) && torch::equal(left.count,right.count) &&
      torch::equal(left.channel_ids,right.channel_ids) && torch::equal(left.floor_applied,right.floor_applied) &&
      left.scale_floor==right.scale_floor;
}
std::string text(torch::serialize::InputArchive &archive,const std::string &key) {
  torch::Tensor value;archive.read(key,value,true);return embedding::archive::tensor_text(value);
}
std::map<std::string,std::string> original_stream_audit(const std::string &path) {
  torch::serialize::InputArchive archive;archive.load_from(path+".audit.pt",torch::kCPU);
  std::map<std::string,std::string> result;
  for(const std::string key:{"fit_source_manifest","initialization_seed","actual_training_seed","rng_policy","sampling_policy"})
    result.emplace(key,text(archive,key));
  return result;
}
void validate_context_policy(const std::string &path) {
  torch::serialize::InputArchive archive;archive.load_from(path+".audit.pt",torch::kCPU);
  const auto ratio=text(archive,"context_deletion_ratio");size_t used=0;const auto value=std::stod(ratio,&used);
  const auto stream=text(archive,"context_deletion_stream");size_t stream_used=0;
  const auto stream_value=std::stoull(stream,&stream_used,0);
  require(text(archive,"model_tag")=="RPB-v6" && text(archive,"training_policy_id")==context_deletion::policy_id &&
      used==ratio.size() && std::isfinite(value) && value==context_deletion::ratio &&
      !stream.empty() && stream.front()>='0' && stream.front()<='9' && stream_used==stream.size() && stream_value==context_deletion::stream &&
      text(archive,"context_deletion_rng_policy")==context_deletion::rng_policy &&
      text(archive,"context_deletion_repair_policy")==context_deletion::repair_policy &&
      text(archive,"context_deletion_visibility_policy")==context_deletion::visibility_policy,
      "candidate checkpoint companion does not match the fixed context training policy");
  torch::Tensor typed_ratio,typed_stream;
  archive.read("context_deletion_ratio_value",typed_ratio,true);archive.read("context_deletion_stream_value",typed_stream,true);
  require(typed_ratio.scalar_type()==torch::kFloat64 && typed_ratio.numel()==1 &&
      typed_ratio.item<double>()==context_deletion::ratio && typed_stream.scalar_type()==torch::kInt64 &&
      typed_stream.numel()==1 && typed_stream.item<int64_t>()==static_cast<int64_t>(context_deletion::stream),
      "typed context policy ratio/stream differs from its declared companion");
  for(const std::string key:{"context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates"}) {
    torch::Tensor count;archive.read(key,count,true);
    require(count.scalar_type()==torch::kInt64 && count.numel()==1 && count.item<int64_t>()==0,
        "candidate point0 already consumed context corruption");
  }
}
} // namespace

std::map<std::string,std::string> audit_context_deletion_initialization(const std::string &candidate_path,
    const ev::RetainedPoolingCohort &reference,const ev::ProviderFitInput &fit) {
  const RuntimeIsolation isolation;
  const auto candidate=load_checkpoint(candidate_path,torch::kCPU);
  const auto initial=load_checkpoint(reference.reference_initial_checkpoint,torch::kCPU);
  const auto selected=load_checkpoint(reference.reference_checkpoint,torch::kCPU);
  for(const auto *checkpoint:{&candidate,&initial,&selected})validate_training(*checkpoint,fit);
  require(candidate.training_policy_id==context_deletion::policy_id && initial.training_policy_id.empty() &&
      selected.training_policy_id.empty(),"candidate/ref checkpoint policy tags differ from the declared context experiment");
  require(candidate.completed_steps==0 && candidate.attempted_steps==0 && initial.completed_steps==0 &&
      initial.attempted_steps==0 && selected.completed_steps==512 && selected.attempted_steps==512,
      "paired counters differ from fresh candidate and retained unskipped v4 point0/512");
  auto reconciled=candidate.settings;reconciled.steps=selected.settings.steps;
  require(settings_text(reconciled)==settings_text(selected.settings) &&
      settings_text(initial.settings)==settings_text(selected.settings),
      "a model/optimizer setting beyond the declared continuous training budget changed");
  require(same_scaler(candidate.scaler,initial.scaler) && same_scaler(initial.scaler,selected.scaler),
      "candidate/ref frozen original TRAIN scaler tensors or floors differ");
  const auto old=initial.model->named_parameters(),fresh=candidate.model->named_parameters();
  require(old.size()==fresh.size(),"candidate changed the native v4 parameter structure");
  int64_t parameters=0,tensors=0;
  for(const auto &parameter:old) {
    require(fresh.contains(parameter.key()) && parameter.value().scalar_type()==fresh[parameter.key()].scalar_type() &&
        torch::equal(parameter.value(),fresh[parameter.key()]),"initial parameter differs: "+parameter.key());
    parameters+=parameter.value().numel();++tensors;
  }
  require(parameters==225805 && tensors>0,"candidate does not have exactly the v4 parameter count");
  const auto old_buffers=initial.model->named_buffers(),fresh_buffers=candidate.model->named_buffers();
  require(old_buffers.size()==fresh_buffers.size(),"candidate changed the native v4 buffer structure");
  for(const auto &buffer:old_buffers)
    require(fresh_buffers.contains(buffer.key()) && buffer.value().scalar_type()==fresh_buffers[buffer.key()].scalar_type() &&
        torch::equal(buffer.value(),fresh_buffers[buffer.key()]),"initial buffer differs: "+buffer.key());
  const auto reference_audit=original_stream_audit(reference.reference_initial_checkpoint);
  const auto selected_audit=original_stream_audit(reference.reference_checkpoint);
  const auto candidate_audit=original_stream_audit(candidate_path);
  require(candidate_audit==reference_audit && selected_audit==reference_audit &&
      candidate_audit.at("fit_source_manifest")==source_manifest(fit.training_source_ids),
      "original paired initialization/source order/row/patch/Torch counter streams differ");
  validate_context_policy(candidate_path);
  return {{"common_parameters_exact","true"},{"all_parameters_exact_including_global_pool","true"},
      {"common_parameter_count",std::to_string(parameters)},{"common_tensors",std::to_string(tensors)},
      {"all_buffers_exact","true"},{"scaler_exact","true"},{"training_dataset_exact","true"},
      {"counter_streams_exact","true"},{"training_policy_id",context_deletion::policy_id},
      {"training_policy_companion_exact","true"},{"context_deletion_stream",std::to_string(context_deletion::stream)},
      {"training_dataset_id",candidate.dataset_id},{"preprocessing_id",candidate.scaler.identity()},
      {"initialization_seed",candidate_audit.at("initialization_seed")},
      {"training_seed",candidate_audit.at("actual_training_seed")},
      {"scope","same entire mode2 architecture/point0 weights/scaler/original TRAIN and row/patch/Torch streams; additional context-only E policy; retained512 has no skips"}};
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
