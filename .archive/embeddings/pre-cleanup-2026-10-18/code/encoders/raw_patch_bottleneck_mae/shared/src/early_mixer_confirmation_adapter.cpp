// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/early_mixer_confirmation_adapter.h"
#include "embedding/shared/data.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

#ifndef EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID
#define EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using Fields = std::map<std::string, std::string>;
using Integers = std::map<std::string, int64_t>;
const std::array<std::string,4> parent_suffixes{"", ".audit.pt", ".scaler.pt", ".training-raw.pt"};
const std::array<std::string,4> content_keys{"content_checkpoint", "content_audit", "content_scaler", "content_training_raw"};
void require(bool ok, const std::string &why) {
  if (!ok) throw std::runtime_error("[rpb early mixer confirmation] " + why);
}
bool fingerprint(const std::string &id) {
  return id.size() == 64 && id.find_first_not_of("0123456789abcdef") == std::string::npos;
}
struct RuntimeIsolation {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i=0;i<at::getNumGPUs();++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for (const auto &g:generators) states.push_back(g.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try { for(size_t i=0;i<generators.size();++i) generators[i].set_state(states[i]); at::set_num_threads(threads); }
    catch(...) { std::terminate(); }
  }
};
EarlyMixerScope delegate_scope(EarlyMixerConfirmationScope scope) {
  switch(scope) {
    case EarlyMixerConfirmationScope::quality:return EarlyMixerScope::quality;
    case EarlyMixerConfirmationScope::engineering:return EarlyMixerScope::engineering;
  }
  throw std::runtime_error("[rpb early mixer confirmation] explicit bounded scope required");
}
const char *external_protocol(EarlyMixerConfirmationScope scope) {
  return delegate_scope(scope)==EarlyMixerScope::quality ? kEarlyMixerConfirmationFitProtocol : kEarlyMixerConfirmationFixtureFitProtocol;
}
const char *implementation_protocol(EarlyMixerConfirmationScope scope) {
  return delegate_scope(scope)==EarlyMixerScope::quality ? kEarlyMixerFitProtocol : kEarlyMixerFixtureFitProtocol;
}
void budget(EarlyMixerConfirmationScope scope,int64_t value) {
  require(value==0 || (delegate_scope(scope)==EarlyMixerScope::quality ? value==512 : value==2 || value==4),
      "budget outside confirmation scope");
}
void fit_contract(const ev::ProviderFitInput &fit,EarlyMixerConfirmationScope scope) {
  require(fit.protocol_id==external_protocol(scope) && fit.seed<=static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "external fit namespace or seed differs");
  require(fit.shape.channel_count==3 && fit.shape.history_length==32 && fit.shape.input_width==3 &&
      fit.shape.dtype==torch::kFloat64 && fit.shape.device.is_cpu() && fit.channel_ids==std::vector<int64_t>({0,1,2}) &&
      fit.feature_units=="unitless,unitless,unitless" && fit.sampling_interval==1 && fit.endpoint==31 &&
      fit.training_observations.data.defined() && fit.training_observations.feature_mask.defined() &&
      fit.training_observations.data.device().is_cpu() && fit.training_observations.data.scalar_type()==torch::kFloat64 &&
      fit.training_observations.data.dim()==4 && fit.training_observations.data.size(0)==static_cast<int64_t>(fit.training_source_ids.size()) &&
      fit.training_observations.data.size(1)==3 && fit.training_observations.data.size(2)==32 && fit.training_observations.data.size(3)==3 &&
      fit.training_observations.feature_mask.device().is_cpu() && fit.training_observations.feature_mask.scalar_type()==torch::kBool &&
      fit.training_observations.feature_mask.sizes()==fit.training_observations.data.sizes() && !fit.training_source_ids.empty(),
      "original label-free CPU F64 TRAIN geometry and metadata required");
  require(torch::isfinite(fit.training_observations.data).all().item<bool>() &&
      fit.training_observations.data.masked_select(fit.training_observations.feature_mask.logical_not()).eq(0).all().item<bool>(),
      "legal TRAIN values and zero absent storage required");
  const std::string prefix=std::string(external_protocol(scope))+'/';
  for(const auto &id:fit.training_source_ids) require(id.rfind(prefix,0)==0 && id.size()>prefix.size(),"source identity outside new cohort namespace");
  if(delegate_scope(scope)==EarlyMixerScope::quality) {
    const std::array<uint64_t,5> masters{64161,65262,66363,67464,68565};
    require(std::find(masters.begin(),masters.end(),fit.seed)!=masters.end() && fit.training_source_ids.size()==256 &&
        std::set<std::string>(fit.training_source_ids.begin(),fit.training_source_ids.end()).size()==128,
        "quality requires declared new timing master/256 rows/128 sources");
  }
}
ev::ProviderFitInput delegate_fit(const ev::ProviderFitInput &external,EarlyMixerConfirmationScope scope) {
  auto fit=external; fit.protocol_id=implementation_protocol(scope); return fit;
}
std::string manifest(const std::vector<std::string> &ids) {
  std::string value; for(const auto &id:ids) value+=std::to_string(id.size())+':'+id; return value;
}
std::string path_key(const std::string &path) {
  require(!path.empty() && !fs::is_symlink(fs::symlink_status(path)),"empty/symlink checkpoint path");
  return fs::weakly_canonical(fs::absolute(path)).string();
}
std::string content(const std::string &path) {
  require(fs::is_regular_file(path) && !fs::is_symlink(fs::symlink_status(path)) && fs::hard_link_count(path)==1,
      "regular non-aliased parent/confirmation required: "+path);
  std::ifstream stream(path,std::ios::binary); require(bool(stream),"cannot read bound source bytes");
  uint64_t value=14695981039346656037ULL; std::array<char,65536> block{};
  while(stream) { stream.read(block.data(),block.size()); for(std::streamsize i=0;i<stream.gcount();++i) {
    value^=static_cast<unsigned char>(block[static_cast<size_t>(i)]); value*=1099511628211ULL;
  }}
  require(stream.eof(),"bound bytes read failed"); std::ostringstream out;
  out<<"fnv1a64-runtime-content-v1-"<<std::hex<<std::setw(16)<<std::setfill('0')<<value; return out.str();
}
Fields binding(const std::string &path,bool include_confirmation) {
  // Admit the entire explicit matrix before opening any file bodies.
  std::vector<std::string> paths;
  for(const auto &suffix:parent_suffixes) paths.push_back(path+suffix);
  if(include_confirmation) paths.push_back(path+kEarlyMixerConfirmationSuffix);
  std::set<std::string> canonical;
  for(const auto &p:paths) require(fs::is_regular_file(p) && !fs::is_symlink(fs::symlink_status(p)) &&
      fs::hard_link_count(p)==1 && canonical.emplace(fs::canonical(p).string()).second,"invalid/aliased confirmation parent matrix");
  Fields result; for(size_t i=0;i<parent_suffixes.size();++i) result.emplace(content_keys[i],content(paths[i]));
  if(include_confirmation) result.emplace("content_confirmation",content(paths.back()));
  return result;
}
void text(torch::serialize::OutputArchive &a,const std::string &key,const std::string &value) {
  a.write(key,embedding::archive::text_tensor(value),true);
}
std::string text(torch::serialize::InputArchive &a,const std::string &key) {
  torch::Tensor value; a.read(key,value,true);
  require(value.device().is_cpu() && value.scalar_type()==torch::kUInt8 && value.dim()==1,"CPU text required: "+key);
  return embedding::archive::tensor_text(value);
}
int64_t integer(torch::serialize::InputArchive &a,const std::string &key) {
  torch::Tensor value; a.read(key,value,true);
  require(value.device().is_cpu() && value.scalar_type()==torch::kInt64 && value.dim()==0,"CPU int64 scalar required: "+key);
  return value.item<int64_t>();
}
void save_new(const std::string &path,const Fields &fields,const Integers &values) {
  torch::serialize::OutputArchive a; for(const auto &[key,value]:fields) text(a,key,value);
  for(const auto &[key,value]:values) a.write(key,torch::tensor(value,torch::kInt64),true);
  const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  require(fd>=0,"exclusive new confirmation archive required"); ::close(fd); embedding::archive::save_archive(path,a);
}
Fields metadata(const ev::ProviderFitInput &fit,const Settings &settings,const Fields &base,EarlyMixerConfirmationScope scope) {
  Input input{fit.training_observations.data,fit.training_observations.feature_mask,torch::tensor(fit.channel_ids,torch::kInt64),
      torch::full({fit.training_observations.data.size(0)},fit.endpoint,torch::kFloat64),fit.sampling_interval};
  const auto raw=describe_dataset(input,settings.model,fit.feature_units);
  require(raw.dataset_id==base.at("training_dataset_id"),"delegate changed original TRAIN association");
  return {{"artifact_kind",kEarlyMixerConfirmationArtifact}, {"confirmation_protocol_id",kEarlyMixerConfirmationProtocol},
      {"external_fit_protocol_id",fit.protocol_id},{"training_implementation_fit_protocol_id",implementation_protocol(scope)},
      {"confirmation_scope",delegate_scope(scope)==EarlyMixerScope::quality ? "quality" : "engineering"},
      {"confirmation_role",settings.model.channel_mixer_placement ? "early" : "late"},
      {"confirmation_instance_group",delegate_scope(scope)==EarlyMixerScope::quality ?
          (settings.model.channel_mixer_placement ? "RPB-v10.alt-04" : "RPB-v7.alt-04") : "artificial-CUDA-confirmation-fixture"},
      {"confirmation_adapter_source_fingerprint",EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID},
      {"parent_core_writer_source_fingerprint",base.at("core_writer_source_fingerprint")},
      {"parent_training_producer_source_fingerprint",base.at("training_producer_source_fingerprint")},
      {"parent_snapshot_adapter_source_fingerprint",base.at("snapshot_adapter_source_fingerprint")},
      {"training_dataset_id",raw.dataset_id},{"training_schema_id",raw.schema_id},{"preprocessing_id",base.at("preprocessing_id")},
      {"original_training_source_manifest",manifest(fit.training_source_ids)},
      {"original_training_source_manifest_id",base.at("fit_source_manifest_id")},
      {"initialization_seed",base.at("initialization_seed")},{"actual_training_seed",base.at("actual_training_seed")},
      {"training_policy_id",base.at("training_policy_id")},{"architecture_id",base.at("architecture_id")},
      {"feature_units",fit.feature_units},{"snapshot_policy","new-confirmation-binding;unchanged-reliability-CUDA-serving;no-old-quality-input"}};
}
struct Record { Fields fields,bytes; Integers values; };
Record admit(const EarlyMixerConfirmationSnapshotOptions &options,const ev::ProviderFitInput &fit) {
  fit_contract(fit,options.scope); budget(options.scope,options.expected_completed_updates);
  require((options.expected_channel_mixer_placement==0 || options.expected_channel_mixer_placement==1) &&
      fingerprint(options.expected_core_source_fingerprint) && fingerprint(options.expected_training_producer_source_fingerprint) &&
      fingerprint(EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID),"explicit placement and captured source scopes required");
  const auto path=path_key(options.checkpoint_path); Record record; record.bytes=binding(path,true);
  torch::serialize::InputArchive audit; audit.load_from(path+".audit.pt",torch::kCPU);
  Fields base;
  for(const auto &key: {"training_dataset_id","preprocessing_id","fit_source_manifest_id","core_writer_source_fingerprint",
      "training_producer_source_fingerprint","initialization_seed","actual_training_seed","training_policy_id","architecture_id"})
    base.emplace(key,text(audit,key));
  // The literal historical snapshot source is retained, not relabelled as this wrapper.
  torch::serialize::InputArchive side; side.load_from(path+kEarlyMixerConfirmationSuffix,torch::kCPU);
  base.emplace("snapshot_adapter_source_fingerprint",text(side,"parent_snapshot_adapter_source_fingerprint"));
  require(fingerprint(base.at("snapshot_adapter_source_fingerprint")),"historical snapshot source fingerprint required");
  auto settings=parse_settings(text(audit,"resolved_settings"));
  require(settings.attempt_limit==(delegate_scope(options.scope)==EarlyMixerScope::quality ? 1024 : 8),"saved confirmation attempt cap differs");
  (void)make_early_mixer_trainer(settings,delegate_scope(options.scope)); // Declaration only; no model or fit.
  require(settings.model.channel_mixer_placement==options.expected_channel_mixer_placement &&
      base.at("actual_training_seed")==std::to_string(fit.seed) &&
      base.at("initialization_seed")==std::to_string(training_detail::mixed(fit.seed^0x7270622d696e6974ULL)) &&
      base.at("training_policy_id")=="rpb-training-context-deletion-015-v1" &&
      base.at("architecture_id")==architecture_id(settings.model) &&
      text(audit,"protocol_id")==implementation_protocol(options.scope) && text(audit,"fit_source_manifest")==manifest(fit.training_source_ids) &&
      base.at("core_writer_source_fingerprint")==options.expected_core_source_fingerprint &&
      base.at("training_producer_source_fingerprint")==options.expected_training_producer_source_fingerprint &&
      integer(audit,"attempted_steps")==options.expected_completed_updates &&
      integer(audit,"completed_steps")==options.expected_completed_updates && integer(audit,"sampled_rows")==options.expected_completed_updates*8,
      "original implementation/source/role/counters differ before model load");
  record.fields=metadata(fit,settings,base,options.scope);
  for(size_t i=0;i<content_keys.size();++i) record.fields.emplace(content_keys[i],record.bytes.at(content_keys[i]));
  record.values={{"channel_mixer_placement_value",options.expected_channel_mixer_placement},
      {"attempted_steps",options.expected_completed_updates},{"completed_steps",options.expected_completed_updates},
      {"sampled_rows",options.expected_completed_updates*8},{"parameter_count_value",225805}};
  for(const auto &key: {"context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates"})
    record.values.emplace(key,integer(audit,key));
  const auto requested=record.values.at("context_requested_deleted_coordinates"),actual=record.values.at("context_actual_deleted_coordinates"),
      restored=record.values.at("context_restored_coordinates");
  require(requested>=actual && actual>=0 && restored==requested-actual && requested<=options.expected_completed_updates*8*288 &&
      (options.expected_completed_updates!=0 || requested==0),"context counts differ before model load");
  for(const auto &[key,value]:record.fields) require(text(side,key)==value,"confirmation text binding differs: "+key);
  for(const auto &[key,value]:record.values) require(integer(side,key)==value,"confirmation typed binding differs: "+key);
  require(side.keys().size()==record.fields.size()+record.values.size(),"unexpected confirmation archive key");
  require(binding(path,true)==record.bytes,"confirmation bytes changed during admission"); return record;
}
EarlyMixerSnapshotOptions old_options(const EarlyMixerConfirmationSnapshotOptions &o) {
  return {o.checkpoint_path,o.expected_channel_mixer_placement,o.expected_completed_updates,delegate_scope(o.scope),
      o.expected_core_source_fingerprint,o.expected_training_producer_source_fingerprint};
}
} // namespace

ev::CurveSnapshot make_early_mixer_confirmation_snapshot(const EarlyMixerConfirmationSnapshotOptions &options,const ev::ProviderFitInput &fit) {
  const RuntimeIsolation isolation; at::set_num_threads(1);
  const auto record=admit(options,fit); const auto path=path_key(options.checkpoint_path);
  auto original=make_early_mixer_snapshot(old_options(options),delegate_fit(fit,options.scope));
  require(original.features.audit_fields.at("snapshot_loader_source_fingerprint")==record.fields.at("parent_snapshot_adapter_source_fingerprint"),
      "literal implementation snapshot source differs from confirmation binding");
  const auto verify=[path,expected=record.bytes] { require(binding(path,true)==expected,"immutable confirmation parent bytes changed"); };
  ev::CurveSnapshot result=original;
  for(const auto &[key,value]:record.fields) if(key!="artifact_kind") result.features.audit_fields.emplace(key,value);
  result.features.audit_fields["confirmation_content_id"]=record.bytes.at("content_confirmation");
  result.features.provenance+=";external-confirmation-cohort="+fit.protocol_id+";training-implementation="+implementation_protocol(options.scope);
  result.features.extract=[original,verify](const embedding::Batch &batch) {
    const RuntimeIsolation guard; verify(); auto value=original.features.extract(batch); verify(); return value;
  };
  result.reconstruct=[original,verify](const embedding::Batch &batch,const torch::Tensor &mask) {
    const RuntimeIsolation guard; verify(); auto value=original.reconstruct(batch,mask); verify(); return value;
  };
  result.features.save_assets=[original,record,verify](const std::string &directory) {
    const RuntimeIsolation guard; verify(); const auto out=(fs::path(directory)/kEarlyMixerConfirmationSnapshotFile).string();
    require(!fs::exists(out) && !fs::is_symlink(fs::symlink_status(out)),"confirmation snapshot asset already exists");
    original.features.save_assets(directory); auto fields=record.fields; fields["artifact_kind"]=kEarlyMixerConfirmationSnapshotArtifact;
    fields["content_confirmation"]=record.bytes.at("content_confirmation"); save_new(out,fields,record.values); verify();
  };
  verify(); return result;
}

ev::CurveTrainerFactory make_early_mixer_confirmation_trainer(const Settings &settings,EarlyMixerConfirmationScope scope) {
  const auto old_scope=delegate_scope(scope);
  require(settings.attempt_limit==(old_scope==EarlyMixerScope::quality ? 1024 : 8),"fixed confirmation attempt cap required");
  require(fingerprint(EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID),"captured confirmation adapter source required");
  const RuntimeIsolation isolation; const auto factory=make_early_mixer_trainer(settings,old_scope);
  return [settings,scope,factory](const ev::ProviderFitInput &fit) {
    fit_contract(fit,scope); const RuntimeIsolation guard; at::set_num_threads(1);
    auto captured=fit; captured.training_observations.data=fit.training_observations.data.detach().clone();
    captured.training_observations.feature_mask=fit.training_observations.feature_mask.detach().clone();
    auto base=factory(delegate_fit(captured,scope)); const auto static_fields=metadata(captured,settings,base.audit_fields,scope);
    auto saved=std::make_shared<std::map<std::string,std::pair<int64_t,Fields>>>();
    auto latest=std::make_shared<ev::CurveProgress>(); auto failed=std::make_shared<bool>(false);
    const auto verify=[saved] { for(const auto &[path,item]:*saved) require(binding(path,true)==item.second,"an earlier confirmation point changed"); };
    ev::CurveTrainer result; result.audit_fields=base.audit_fields;
    for(const auto &[key,value]:static_fields) if(key!="artifact_kind") result.audit_fields.emplace(key,value);
    result.train_to=[base,scope,latest,failed,verify](int64_t updates) {
      require(!*failed,"failed confirmation trainer cannot continue"); budget(scope,updates); const RuntimeIsolation isolate;
      try { verify(); auto value=base.train_to(updates);
        require(value.completed==updates && value.attempted==updates && value.sampled_rows==updates*8,"unskipped confirmation prefix required");
        *latest=value; verify(); return value;
      } catch(...) { *failed=true; throw; }
    };
    result.save_checkpoint=[base,static_fields,latest,saved,failed,verify,scope](const std::string &path) {
      require(!*failed,"failed confirmation trainer cannot save"); budget(scope,latest->completed); const RuntimeIsolation isolate;
      try {
        verify(); const auto key=path_key(path);
        for(const auto &suffix:parent_suffixes) require(!fs::exists(key+suffix) && !fs::is_symlink(fs::symlink_status(key+suffix)),"new point required");
        require(!fs::exists(key+kEarlyMixerConfirmationSuffix) && !fs::is_symlink(fs::symlink_status(key+kEarlyMixerConfirmationSuffix)),"new confirmation companion required");
        base.save_checkpoint(key); auto fields=static_fields; const auto parent=binding(key,false);
        fields.insert(parent.begin(),parent.end());
        Integers values{{"channel_mixer_placement_value",fields.at("confirmation_role")=="early" ? int64_t(1) : int64_t(0)},
            {"attempted_steps",latest->attempted},{"completed_steps",latest->completed},{"sampled_rows",latest->sampled_rows},{"parameter_count_value",225805}};
        torch::serialize::InputArchive audit; audit.load_from(key+".audit.pt",torch::kCPU);
        for(const auto &name:{"context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates"}) values.emplace(name,integer(audit,name));
        save_new(key+kEarlyMixerConfirmationSuffix,fields,values);
        require(saved->emplace(key,std::make_pair(latest->completed,binding(key,true))).second,"point already saved"); verify();
      } catch(...) { *failed=true; throw; }
    };
    result.snapshot=[captured,settings,scope,base,saved,failed,verify](const std::string &path) {
      require(!*failed,"failed confirmation trainer cannot snapshot"); const RuntimeIsolation isolate;
      try { verify(); const auto found=saved->find(path_key(path)); require(found!=saved->end(),"only this wrapper's saved new point can be served");
        EarlyMixerConfirmationSnapshotOptions options{path,settings.model.channel_mixer_placement,found->second.first,scope,
            base.audit_fields.at("core_writer_source_fingerprint"),base.audit_fields.at("training_producer_source_fingerprint")};
        auto value=make_early_mixer_confirmation_snapshot(options,captured); verify(); return value;
      } catch(...) { *failed=true; throw; }
    };
    return result;
  };
}

Fields audit_early_mixer_confirmation_initialization(const EarlyMixerConfirmationSnapshotOptions &late,
    const EarlyMixerConfirmationSnapshotOptions &early,const ev::ProviderFitInput &fit) {
  require(late.scope==early.scope && late.expected_channel_mixer_placement==0 && early.expected_channel_mixer_placement==1 &&
      late.expected_completed_updates==0 && early.expected_completed_updates==0,"declared new late0/early0 pair required");
  const RuntimeIsolation guard; at::set_num_threads(1); const auto a=admit(late,fit),b=admit(early,fit);
  auto result=audit_early_mixer_initialization(old_options(late),old_options(early),delegate_fit(fit,late.scope));
  require(binding(path_key(late.checkpoint_path),true)==a.bytes && binding(path_key(early.checkpoint_path),true)==b.bytes,"paired initial bytes changed");
  result.emplace("confirmation_protocol_id",kEarlyMixerConfirmationProtocol); result.emplace("external_fit_protocol_id",fit.protocol_id);
  result.emplace("training_implementation_fit_protocol_id",implementation_protocol(late.scope));
  result.emplace("confirmation_adapter_source_fingerprint",EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID);
  result.emplace("historical_quality_input_roles","0"); return result;
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
