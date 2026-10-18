// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/visible_difference_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

#ifndef VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID
#define VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev=embedding::evaluation;
namespace fs=std::filesystem;
using Named=std::map<std::string,torch::Tensor>;
constexpr const char *policy="rpb-training-context-deletion-015-v1";
constexpr const char *initialization="independent-original-common-state-exact;last-zero-branch-no-RNG;assert-before-AdamW;no-copy";
void require(bool ok,const std::string &why) { if (!ok) throw std::runtime_error("[rpb visible difference] "+why); }
bool pin(const std::string &v) { return v.size()==64 && v.find_first_not_of("0123456789abcdef")==std::string::npos; }
const char *fit_protocol(VisibleDifferenceScope scope) {
  switch(scope) { case VisibleDifferenceScope::quality:return kVisibleDifferenceFitProtocol;
    case VisibleDifferenceScope::engineering:return kVisibleDifferenceFixtureFitProtocol; }
  throw std::runtime_error("explicit visible-difference scope required");
}
void budget(VisibleDifferenceScope scope,int64_t n) {
  (void)fit_protocol(scope);
  require(scope==VisibleDifferenceScope::quality ? n==0 || n==512 : n==0 || n==2 || n==4,"budget outside fixed scope");
}
struct RuntimeIsolation {
  int threads{at::get_num_threads()}; std::vector<at::Generator> generators; std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for(size_t i=0;i<at::getNumGPUs();++i) generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for(const auto &g:generators) states.push_back(g.get_state().clone());
  }
  ~RuntimeIsolation() noexcept { try { for(size_t i=0;i<generators.size();++i) generators[i].set_state(states[i]); at::set_num_threads(threads); } catch(...) { std::terminate(); } }
};
torch::Tensor cpu(const torch::Tensor &v) { return v.detach().to(torch::kCPU).contiguous().clone(); }
torch::Tensor tensor(torch::serialize::InputArchive &a,const std::string &key) { torch::Tensor v; a.read(key,v,true); return v; }
std::string text(torch::serialize::InputArchive &a,const std::string &key) { return embedding::archive::tensor_text(tensor(a,key)); }
void text(torch::serialize::OutputArchive &a,const std::string &key,const std::string &value) { a.write(key,embedding::archive::text_tensor(value),true); }
int64_t integer(torch::serialize::InputArchive &a,const std::string &key) {
  auto v=tensor(a,key); require(v.device().is_cpu() && v.scalar_type()==torch::kInt64 && v.dim()==0,"typed CPU int64 required: "+key); return v.item<int64_t>();
}
std::string manifest(const std::vector<std::string> &ids) { std::string out; for(const auto &id:ids) out+=std::to_string(id.size())+':'+id; return out; }
std::string path_key(const std::string &path) {
  const auto p=fs::absolute(path).lexically_normal();
  require(!path.empty() && fs::canonical(p)==p,"direct canonical point path required"); return p.string();
}
std::string file_id(const std::string &path) {
  require(fs::is_regular_file(path) && !fs::is_symlink(fs::symlink_status(path)) && fs::hard_link_count(path)==1 &&
      fs::canonical(path)==fs::absolute(path).lexically_normal(),"regular direct nonaliased role required");
  std::ifstream in(path,std::ios::binary); require(bool(in),"cannot read role");
  uint64_t h=14695981039346656037ULL; std::array<char,65536> buf{};
  while(in) { in.read(buf.data(),buf.size()); for(std::streamsize i=0;i<in.gcount();++i) { h^=static_cast<unsigned char>(buf[i]); h*=1099511628211ULL; } }
  require(in.eof(),"role read failed"); std::ostringstream out; out<<"fnv1a64-runtime-content-v1-"<<std::hex<<std::setw(16)<<std::setfill('0')<<h; return out.str();
}
std::map<std::string,std::string> bindings(const std::string &path) {
  std::map<std::string,std::string> out;
  const std::array<std::string,6> suffixes{"",".audit.pt",".scaler.pt",".training-raw.pt",kLearningCurveContinuationSuffix,kVisibleDifferenceBindingSuffix};
  // Whole regular/path pass precedes every content hash.
  for(const auto &s:suffixes) require(fs::is_regular_file(path+s) && fs::hard_link_count(path+s)==1 &&
      fs::canonical(path+s)==fs::absolute(path+s).lexically_normal(),"complete direct six-role point required");
  for(const auto &s:suffixes) out.emplace(s,file_id(path+s));
  return out;
}
Named named(const Model &m,bool buffers=false) {
  Named out; if(buffers) { for(const auto &v:m->named_buffers()) out.emplace(v.key(),cpu(v.value())); }
  else { for(const auto &v:m->named_parameters()) out.emplace(v.key(),cpu(v.value())); } return out;
}
bool same(const Named &a,const Named &b) {
  if(a.size()!=b.size()) return false;
  for(const auto &[n,v]:a) { const auto at=b.find(n); if(at==b.end() || v.scalar_type()!=at->second.scalar_type() || v.sizes()!=at->second.sizes() || !torch::equal(v,at->second)) return false; } return true;
}
Named read_named(torch::serialize::InputArchive &a,const std::string &key) {
  torch::serialize::InputArchive group; a.read(key,group); const auto n=integer(group,"count"); require(n>=0,"negative named count"); Named out;
  for(int64_t i=0;i<n;++i) { torch::serialize::InputArchive entry; group.read("tensor_"+std::to_string(i),entry); auto v=tensor(entry,"value");
    require(v.device().is_cpu() && torch::isfinite(v).all().item<bool>() && out.emplace(text(entry,"parameter_name"),v).second,"finite unique CPU named state required"); }
  return out;
}
void write_named(torch::serialize::OutputArchive &a,const std::string &key,const Named &values) {
  torch::serialize::OutputArchive group; group.write("count",torch::tensor(static_cast<int64_t>(values.size())),true); int64_t i=0;
  for(const auto &[n,v]:values) { torch::serialize::OutputArchive entry; text(entry,"parameter_name",n); entry.write("value",v,true); group.write("tensor_"+std::to_string(i++),entry); } a.write(key,group);
}
void configuration(const Settings &s,VisibleDifferenceScope scope) {
  validate_settings(s); (void)fit_protocol(scope); const auto &c=s.model;
  require(c.device.is_cuda() && c.temporal_difference_input==1 && c.global_pool_input_source==0 && c.channel_mixer_placement==1 &&
      c.channel_count==3 && c.history_length==32 && c.input_width==3 && c.patch_length==8 && c.encoder_width==64 && c.export_width==32 &&
      c.num_layers==3 && c.num_heads==4 && c.feedforward_width==256 && c.decoder_hidden_width==128 && c.channel_mixer_layers==1 &&
      c.global_bottleneck_mode==2 && c.dropout==0 && c.huber_delta==1 && c.sampling_interval==1 && resolved_channel_ids(c)==std::vector<int64_t>({0,1,2}) &&
      s.batch_size==8 && s.threads==1 && s.learning_rate==.001 && s.weight_decay==.0001 && s.gradient_clip_norm==1 && s.log_every==1 &&
      (scope==VisibleDifferenceScope::quality ? s.steps==512 && s.attempt_limit==1024 : s.steps==4 && s.attempt_limit==8),"fixed candidate architecture/recipe/CUDA/ceiling required");
}
void fit_contract(const ev::ProviderFitInput &fit,VisibleDifferenceScope scope) {
  require(fit.protocol_id==fit_protocol(scope) && fit.shape.channel_count==3 && fit.shape.history_length==32 && fit.shape.input_width==3 &&
      fit.shape.dtype==torch::kFloat64 && fit.shape.device.is_cpu() && fit.channel_ids==std::vector<int64_t>({0,1,2}) &&
      fit.feature_units=="unitless,unitless,unitless" && fit.endpoint==31 && fit.sampling_interval==1 && fit.training_observations.data.defined() &&
      fit.training_observations.feature_mask.defined() && fit.training_observations.data.device().is_cpu() && fit.training_observations.data.scalar_type()==torch::kFloat64 &&
      fit.training_observations.feature_mask.device().is_cpu() && fit.training_observations.feature_mask.scalar_type()==torch::kBool &&
      fit.training_source_ids.size()==static_cast<size_t>(fit.training_observations.data.size(0)),"original label-free TRAIN fit contract required");
  for(const auto &id:fit.training_source_ids) require(!id.empty(),"empty original source ID");
  if(scope==VisibleDifferenceScope::quality) require(fit.training_source_ids.size()==256 &&
      std::set<std::string>(fit.training_source_ids.begin(),fit.training_source_ids.end()).size()==128,"original256 TRAIN rows/128 source pairs required");
}
Input input(const embedding::Batch &b,const Config &c,const ev::ProviderFitInput &fit) {
  Input out{b.data,b.feature_mask,torch::tensor(fit.channel_ids,torch::kInt64),torch::full({b.data.size(0)},fit.endpoint,torch::kFloat64),fit.sampling_interval}; validate_input(out,c); return out;
}
struct Parent {
  Checkpoint cp; ev::ProviderFitInput fit; std::string path; Named parameters,buffers; std::string scaler;
  std::map<std::string,std::string> content,fields;
};
void immutable(const std::shared_ptr<Parent> &p) {
  require(!p->cp.model->is_training() && same(p->parameters,named(p->cp.model)) && same(p->buffers,named(p->cp.model,true)) &&
      p->cp.scaler.identity()==p->scaler && bindings(p->path)==p->content,"immutable saved CUDA snapshot changed");
  for(const auto &v:p->cp.model->parameters()) require(v.is_cuda() && !v.requires_grad(),"frozen CUDA parameters required");
}
std::shared_ptr<Parent> admit(const VisibleDifferenceSnapshotOptions &o,const ev::ProviderFitInput &fit) {
  budget(o.scope,o.expected_completed_updates); fit_contract(fit,o.scope);
  require(pin(o.expected_core_source_fingerprint) && pin(o.expected_training_producer_source_fingerprint),"explicit new writer source pins required");
  auto p=std::make_shared<Parent>(); p->path=path_key(o.checkpoint_path); p->content=bindings(p->path); p->fit=fit;
  torch::serialize::InputArchive a,b,state; a.load_from(p->path+".audit.pt",torch::kCPU); b.load_from(p->path+kVisibleDifferenceBindingSuffix,torch::kCPU);
  state.load_from(p->path+kLearningCurveContinuationSuffix,torch::kCPU);
  auto settings=parse_settings(text(a,"resolved_settings")); settings.model.device=torch::Device(torch::kCUDA,0); configuration(settings,o.scope);
  const auto updates=o.expected_completed_updates;
  require(text(a,"artifact_kind")=="rpb_learning_curve_training_audit_v1" && text(b,"artifact_kind")==kVisibleDifferenceBindingArtifact &&
      text(state,"artifact_kind")==kVisibleDifferenceContinuationArtifact && text(a,"protocol_id")==fit.protocol_id && text(a,"model_tag")=="RPB-v13" &&
      text(a,"training_policy_id")==policy && text(a,"fit_source_manifest")==manifest(fit.training_source_ids) &&
      text(a,"actual_training_seed")==std::to_string(fit.seed&0x7fffffffffffffffULL) &&
      text(a,"initialization_seed")==std::to_string(training_detail::mixed(fit.seed^0x7270622d696e6974ULL)) &&
      text(a,"core_writer_source_fingerprint")==o.expected_core_source_fingerprint && text(a,"training_producer_source_fingerprint")==o.expected_training_producer_source_fingerprint &&
      text(a,"architecture_id")==architecture_id(settings.model) && text(a,"output_semantics")==output_semantics(settings.model) &&
      text(a,"reconstruction_export_semantics")==reconstruction_output_semantics(settings.model) &&
      text(a,"temporal_difference_input_semantics")==visible_difference_input_semantics(settings.model) &&
      text(a,"visible_difference_initialization_policy")==initialization && text(a,"visible_difference_common_parameter_values")=="225805" &&
      text(a,"visible_difference_added_parameter_values")=="3072" && text(a,"visible_difference_copied_parameter_values")=="0" &&
      text(a,"parameter_count")=="228877" && text(a,"cuda_parameter_count")=="228877","typed protocol/config/architecture/source/init audit differs");
  for(auto *archive:{&a,&b,&state}) {
    require(integer(*archive,"attempted_steps")==updates && integer(*archive,"completed_steps")==updates && integer(*archive,"sampled_rows")==updates*8 &&
        integer(*archive,"temporal_difference_input_value")==1,"typed unskipped candidate counters differ");
    for(const auto &key:{"protocol_id","resolved_settings","training_policy_id","fit_source_manifest","core_writer_source_fingerprint","training_producer_source_fingerprint",
        "training_dataset_id","preprocessing_id","architecture_id","output_semantics","reconstruction_export_semantics","temporal_difference_input_semantics",
        "visible_difference_initialization_policy","visible_difference_parent_point0_path","visible_difference_parent_core_source_fingerprint",
        "visible_difference_parent_training_producer_source_fingerprint","visible_difference_parent_fit_protocol"})
      require(text(*archive,key)==text(a,key),std::string("companion association differs: ")+key);
  }
  require(integer(state,"common_parameter_values")==225805 && integer(state,"copied_parameter_values")==0 && integer(state,"difference_parameter_values")==3072,
      "typed initial common/new branch counts differ");
  require(text(b,"checkpoint_path")==p->path && text(state,"checkpoint_path")==p->path,"saved point path association differs");
  for(const auto &suffix:{std::string(),std::string(".audit.pt"),std::string(".scaler.pt"),std::string(".training-raw.pt"),std::string(kLearningCurveContinuationSuffix)})
    require(text(b,"checkpoint_content_id"+suffix)==p->content.at(suffix),"point binding bytes differ before weight load");
  for(const auto &suffix:{std::string(),std::string(".audit.pt"),std::string(".scaler.pt"),std::string(".training-raw.pt")}) {
    const auto key="visible_difference_parent_content_id"+suffix;
    require(text(b,key)==text(a,key) && text(state,key)==text(a,key),"original point0 content association differs");
  }
  require(pin(text(a,"visible_difference_parent_core_source_fingerprint")) && pin(text(a,"visible_difference_parent_training_producer_source_fingerprint")),"typed original source pins required");
  const auto raw=load_dataset(p->path+".training-raw.pt",settings.model);
  const auto expected=input(fit.training_observations,settings.model,fit);
  require(torch::equal(raw.input.data,expected.data) && torch::equal(raw.input.observed,expected.observed) && torch::equal(raw.input.channel_ids,expected.channel_ids) &&
      torch::equal(raw.input.endpoints,expected.endpoints) && raw.feature_units==fit.feature_units && raw.dataset_id==text(a,"training_dataset_id"),"original TRAIN content differs before weights");
  const auto scaler=load_scaler(p->path+".scaler.pt",settings.model,raw.schema_id);
  torch::serialize::InputArchive scaler_envelope; scaler_envelope.load_from(p->path+".scaler.pt",torch::kCPU);
  require(text(scaler_envelope,"fit_dataset_id")==raw.dataset_id && scaler.identity()==text(a,"preprocessing_id") &&
      text(a,"scaler_fit_dataset_id")==raw.dataset_id,"frozen original TRAIN scaler differs before weights");
  // Metadata/byte/geometry/TRAIN association precedes CUDA weight loading.
  p->cp=load_checkpoint(p->path,settings.model.device); auto resolved=p->cp.settings; resolved.model.device=settings.model.device;
  require(settings_text(resolved)==settings_text(settings) && p->cp.source_fingerprint==o.expected_core_source_fingerprint &&
      p->cp.training_policy_id==policy && p->cp.attempted_steps==updates && p->cp.completed_steps==updates &&
      p->cp.dataset_id==raw.dataset_id && p->cp.schema_id==raw.schema_id && p->cp.scaler_fit_dataset_id==raw.dataset_id &&
      p->cp.scaler.identity()==scaler.identity(),"ordinary candidate checkpoint differs");
  p->parameters=named(p->cp.model); p->buffers=named(p->cp.model,true); p->scaler=scaler.identity();
  require(same(read_named(state,"model_parameters"),p->parameters) && same(read_named(state,"model_buffers"),p->buffers),"saved live model state differs");
  const auto initial=read_named(state,"initial_model_parameters"), initial_buffers=read_named(state,"initial_model_buffers");
  require(initial.size()==p->parameters.size() && initial.contains("visible_difference_projection.weight") &&
      torch::equal(initial.at("visible_difference_projection.weight"),torch::zeros({64,48},torch::kFloat32)) &&
      same(initial_buffers,p->buffers),"saved independent zero branch/buffer witness differs");
  int64_t count=0,common=0;
  for(const auto &[name,value]:p->parameters) { count+=value.numel(); require(initial.contains(name) && initial.at(name).sizes()==value.sizes() &&
      initial.at(name).scalar_type()==value.scalar_type(),"initial named geometry differs"); if(name!="visible_difference_projection.weight") common+=value.numel(); }
  require(count==228877 && common==225805 && (updates!=0 || same(initial,p->parameters)),"exact candidate/initial parameter state required");
  torch::serialize::InputArchive opt,ss; state.read("optimizer_state",opt); state.read("scaler",ss);
  require(FrozenScaler::load(ss).identity()==scaler.identity() && integer(opt,"parameter_count")==static_cast<int64_t>(p->parameters.size()),"live scaler/optimizer associations differ");
  int64_t active=0,index=0;
  for(const auto &[name,value]:p->parameters) {
    torch::serialize::InputArchive entry; opt.read("parameter_"+std::to_string(index++),entry); auto has=tensor(entry,"has_state");
    require(text(entry,"parameter_name")==name && torch::equal(tensor(entry,"parameter_shape"),torch::tensor(value.sizes().vec(),torch::kInt64)) &&
        has.device().is_cpu() && has.scalar_type()==torch::kBool && has.dim()==0,"named optimizer identity/shape differs");
    if(has.item<bool>()) { ++active; require(updates>0 && integer(entry,"step")==updates,"absolute AdamW step differs");
      for(const auto &key:{"exp_avg","exp_avg_sq"}) { auto moment=tensor(entry,key); require(moment.device().is_cpu() && moment.scalar_type()==value.scalar_type() && moment.sizes()==value.sizes() && torch::isfinite(moment).all().item<bool>(),"finite named AdamW moment required"); } }
  }
  require(integer(opt,"active_state_count")==active && (updates>0 || active==0),"complete optimizer active count differs");
  auto counters=tensor(state,"loss_trace_counters"),losses=tensor(state,"loss_trace_values");
  require(counters.scalar_type()==torch::kInt64 && counters.sizes()==torch::IntArrayRef({updates,3}) && losses.scalar_type()==torch::kFloat64 &&
      losses.sizes()==torch::IntArrayRef({updates,2}) && torch::isfinite(losses).all().item<bool>(),"complete finite live trace required");
  for(int64_t i=0;i<updates;++i) require(counters[i][0].item<int64_t>()==i+1 && counters[i][1].item<int64_t>()==i+1 && counters[i][2].item<int64_t>()>0,"unskipped absolute trace prefix required");
  for(const auto &key:{"context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates"})
    require(integer(a,key)==integer(state,key),"live context totals differ");
  for(const auto &key:a.keys()) { if(key=="artifact_kind") continue; auto v=tensor(a,key); if(v.scalar_type()==torch::kUInt8) p->fields.emplace(key,embedding::archive::tensor_text(v)); }
  p->fit.training_observations.data=cpu(fit.training_observations.data);
  p->fit.training_observations.feature_mask=cpu(fit.training_observations.feature_mask);
  p->cp.model->eval(); for(auto &v:p->cp.model->parameters()) v.requires_grad_(false); immutable(p); return p;
}
} // namespace

ev::CurveSnapshot make_visible_difference_snapshot(const VisibleDifferenceSnapshotOptions &o,const ev::ProviderFitInput &fit) {
  require(torch::cuda::is_available(),"CUDA snapshot requires available CUDA"); const RuntimeIsolation guard; at::set_num_threads(1); torch::NoGradGuard no_grad;
  const auto parent=admit(o,fit); ev::CurveSnapshot out;
  out.features.provenance="RPB-v13 immutable CUDA visible-first-difference native32; original timing TRAIN scaler; no inference augmentation";
  out.features.audit_fields=parent->fields; auto &fields=out.features.audit_fields;
  fields["protocol_id"]=kVisibleDifferenceProtocol;
  fields["original_training_protocol_id"]=fit.protocol_id;
  fields["checkpoint_path"]=parent->path; fields["snapshot_loader_source_fingerprint"]=VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID;
  fields["parent_checkpoint_path"]=parent->path;
  fields["parent_writer_source_fingerprint"]=parent->cp.source_fingerprint;
  fields["parent_training_producer_source_fingerprint"]=o.expected_training_producer_source_fingerprint;
  fields["source_fingerprint_scope"]="parent=core_writer_and_training_producer;loader=new_visible_difference_adapter";
  fields["original_encoder_attempted"]=std::to_string(o.expected_completed_updates);
  fields["original_encoder_completed"]=std::to_string(o.expected_completed_updates);
  fields["training_schema_id"]=parent->cp.schema_id;
  fields["encoder_updates"]="0"; fields["decoder_updates"]="0"; fields["head_refits"]="0";
  fields["no_optimizer_created"]="true"; fields["inference_device"]=parent->cp.settings.model.device.str();
  fields["snapshot_policy"]="immutable_CUDA_model;original_TRAIN_scaler;encoder_decoder_updates0;no_optimizer_load";
  fields["attempted_steps"]=std::to_string(o.expected_completed_updates); fields["completed_steps"]=std::to_string(o.expected_completed_updates);
  for(const auto &[suffix,id]:parent->content) fields["parent_content_id"+suffix]=id;
  out.features.surfaces.emplace("curve_global",ev::SurfaceDescription{ev::SurfaceKind::global,"at least one observed channel",{}});
  out.features.extract=[parent,provenance=out.features.provenance](const embedding::Batch &batch) {
    const RuntimeIsolation isolate; at::set_num_threads(1); torch::NoGradGuard no_grad; immutable(parent);
    auto &c=parent->cp; const auto raw=c.scaler.transform(input(batch,c.settings.model,parent->fit),c.settings.model);
    const auto encoded=c.model->encode(raw); const auto z=compact_reconstruction_export(encoded,c.settings.model);
    require(raw.data.is_cuda() && z.is_cuda() && z.scalar_type()==torch::kFloat32 && z.sizes()==torch::IntArrayRef({batch.data.size(0),32}) && torch::isfinite(z).all().item<bool>(),"finite exact CUDA BD32 required");
    ev::FeatureMap result{{"curve_global",{cpu(z),cpu(encoded.sample_valid_mask),provenance}}}; immutable(parent); return result;
  };
  out.reconstruct=[parent](const embedding::Batch &batch,const torch::Tensor &hidden) {
    const RuntimeIsolation isolate; at::set_num_threads(1); torch::NoGradGuard no_grad; immutable(parent);
    auto &c=parent->cp; const auto raw=c.scaler.transform(input(batch,c.settings.model,parent->fit),c.settings.model); const auto result=c.model->forward(raw,hidden);
    require(raw.data.is_cuda() && result.reconstruction.is_cuda() && torch::isfinite(result.reconstruction).all().item<bool>(),"finite CUDA original-Q reconstruction required");
    ev::CurveReconstruction out{cpu(result.reconstruction),cpu(raw.data),cpu(result.eligible_channels)}; immutable(parent); return out;
  };
  out.features.save_assets=[parent,fields](const std::string &directory) {
    const RuntimeIsolation isolate; immutable(parent); torch::serialize::OutputArchive a; text(a,"artifact_kind",kVisibleDifferenceSnapshotArtifact);
    for(const auto &[key,value]:fields) text(a,key,value);
    a.write("temporal_difference_input_value",torch::tensor(int64_t(1)),true); a.write("original_encoder_attempted_value",torch::tensor(parent->cp.attempted_steps),true);
    a.write("original_encoder_completed_value",torch::tensor(parent->cp.completed_steps),true); write_named(a,"model_parameters",parent->parameters); write_named(a,"model_buffers",parent->buffers);
    torch::serialize::OutputArchive scaler; parent->cp.scaler.save(scaler); a.write("scaler",scaler);
    const auto path=(fs::path(directory)/kVisibleDifferenceSnapshotAuditFile).string(); const auto fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    require(fd>=0,"exclusive new snapshot asset required"); ::close(fd); embedding::archive::save_archive(path,a); immutable(parent);
  };
  return out;
}

ev::CurveTrainerFactory make_visible_difference_trainer(const Settings &s,const VisibleDifferenceOptions &options,VisibleDifferenceScope scope) {
  configuration(s,scope); require(options.enabled,"explicit visible-difference initialization option required");
  const RuntimeIsolation isolate; auto factory=make_learning_curve_trainer(s,ContextDeletionOptions{true,ContextDeletionRecipe::coordinate15_v1},
      LearningCurveStateWitnessOptions{true},TrainingSourceGainOptions{},PooledContextInitializationOptions{},options);
  return [factory,scope](const ev::ProviderFitInput &fit) {
    fit_contract(fit,scope); const RuntimeIsolation guard; at::set_num_threads(1); auto captured=fit;
    captured.training_observations.data=cpu(fit.training_observations.data); captured.training_observations.feature_mask=cpu(fit.training_observations.feature_mask);
    auto base=factory(captured); auto latest=std::make_shared<ev::CurveProgress>(); auto failed=std::make_shared<bool>(false);
    auto saved=std::make_shared<std::map<std::string,std::pair<int64_t,std::map<std::string,std::string>>>>();
    const auto writer=base.audit_fields.at("core_writer_source_fingerprint"),producer=base.audit_fields.at("training_producer_source_fingerprint");
    ev::CurveTrainer out; out.audit_fields=base.audit_fields; out.audit_fields["snapshot_adapter_source_fingerprint"]=VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID;
    out.train_to=[base,scope,latest,failed,saved](int64_t n) {
      require(!*failed,"failed trainer cannot advance"); budget(scope,n); const RuntimeIsolation isolate; at::set_num_threads(1);
      try { for(const auto &[path,value]:*saved) require(bindings(path)==value.second,"earlier point bytes changed");
        auto progress=base.train_to(n); require(progress.attempted==n && progress.completed==n && progress.sampled_rows==n*8 && progress.losses.size()==static_cast<size_t>(n),"complete unskipped state required");
        require(progress.losses.size()>=latest->losses.size(),"trace prefix shrank");
        for(size_t i=0;i<latest->losses.size();++i) { const auto &a=latest->losses[i],&b=progress.losses[i]; require(a.attempted==b.attempted && a.completed==b.completed && a.target_cells==b.target_cells && a.loss==b.loss && a.gradient_norm==b.gradient_norm,"previous trace prefix changed"); }
        *latest=progress; return progress;
      } catch(...) { *failed=true; throw; }
    };
    out.save_checkpoint=[base,scope,latest,failed,saved](const std::string &path) {
      require(!*failed,"failed trainer cannot save"); budget(scope,latest->completed); const RuntimeIsolation isolate; at::set_num_threads(1);
      base.save_checkpoint(path); require(saved->emplace(path_key(path),std::make_pair(latest->completed,bindings(path))).second,"point already recorded");
    };
    out.snapshot=[captured,scope,writer,producer,failed,saved](const std::string &path) {
      require(!*failed,"failed trainer cannot serve"); const auto found=saved->find(path_key(path)); require(found!=saved->end() && bindings(path)==found->second.second,"this trainer's immutable saved point required");
      return make_visible_difference_snapshot({path,found->second.first,scope,writer,producer},captured);
    };
    return out;
  };
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
