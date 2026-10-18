// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../LICENSE.
#pragma once
#include "cuwacunu/artifacts/v1/sidecar.h"
#include "cuwacunu/artifacts/v1/tensor_state.h"
#include "cuwacunu/networks/direct_embedding_control/v1/control.h"
#include <cerrno>
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/syscall.h>
#include <unistd.h>

// This control owns its publication schema. The evaluated shared sidecar v1
// remains unchanged and does not acquire an unsupported Dynamics Model alias.
namespace cuwacunu::networks::direct_embedding_control::v1::artifact {
namespace av=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;
inline constexpr std::string_view component_name="Direct Embedding Control";

struct Context {
  av::Component encoder;
  std::string encoder_archive_sha256;
  fs::path repository_root;
  std::string protocol_sha256,population_sha256,coordinates_sha256;
  int64_t fit_cutoff_ms{0},cadence_ms{0},history_spacing_ticks{0};
  std::vector<int64_t> channel_ids;
  std::vector<std::string> feature_names;
  std::string embedding_export_layout,availability_policy;
};

inline void require(bool value,const std::string& message) {
  TORCH_CHECK(value,"[direct embedding control artifact v1] ",message);
}
inline void validate_context(const Context& context) {
  av::validate_component(context.encoder);
  require(context.encoder.module_ref.component==av::encoder_component&&context.encoder.parents.empty(),"verified Encoder parent required");
  for(const auto* value:{&context.encoder_archive_sha256,&context.protocol_sha256,&context.population_sha256,
      &context.coordinates_sha256})require(av::is_sha256(*value),"explicit archive/protocol/population/coordinate pins required");
  require(context.fit_cutoff_ms>0&&context.cadence_ms>0&&context.history_spacing_ticks>0&&
    !context.channel_ids.empty()&&!context.feature_names.empty()&&!context.embedding_export_layout.empty()&&
    !context.availability_policy.empty(),"explicit fit cutoff, chronology, semantic order and coordinate policies required");
  require(std::set<int64_t>(context.channel_ids.begin(),context.channel_ids.end()).size()==context.channel_ids.size(),"duplicate semantic channel");
  require(std::set<std::string>(context.feature_names.begin(),context.feature_names.end()).size()==context.feature_names.size(),"duplicate semantic feature");
  for(const auto& value:context.feature_names)require(!value.empty(),"empty semantic feature");
}
inline std::string context_sha256(const Context& context) {
  validate_context(context);
  std::vector<std::string> fields{context.encoder.model_ref,context.encoder_archive_sha256,context.protocol_sha256,
    context.population_sha256,context.coordinates_sha256,std::to_string(context.fit_cutoff_ms),std::to_string(context.cadence_ms),
    std::to_string(context.history_spacing_ticks),context.embedding_export_layout,context.availability_policy,
    std::to_string(context.channel_ids.size())};
  for(auto value:context.channel_ids)fields.push_back(std::to_string(value));
  fields.push_back(std::to_string(context.feature_names.size()));
  for(const auto& value:context.feature_names)fields.push_back(value);
  return av::digest_fields("DirectEmbeddingControlContext-v1",fields);
}
inline void validate_component(const av::Component& component) {
  const auto& module=component.module_ref;
  require(module.component==component_name&&module.implementation_version=="v1","unsupported control component/version");
  require(av::is_sha256(module.source_sha256)&&av::is_sha256(component.state_sha256)&&
    av::is_sha256(component.inference_config_sha256)&&av::is_sha256(component.model_ref),"invalid control digest");
  require(component.parents.size()==1&&component.parents[0].role=="embedding"&&
    component.parents[0].component==av::encoder_component&&av::is_sha256(component.parents[0].model_ref),
    "control requires one exact embedding -> Encoder parent");
  require(av::derive_model_ref(component)==component.model_ref,"control ModelRef does not match its content");
}
struct Manifest {
  std::string archive_kind,archive_sha256,resume_identity_sha256;
  av::Component component;
  bool operator==(const Manifest&) const=default;
};
inline void validate_manifest(const Manifest& manifest) {
  require(manifest.archive_kind=="training"||manifest.archive_kind=="inference","unsupported archive kind");
  require(av::is_sha256(manifest.archive_sha256),"invalid archive digest");
  require(manifest.archive_kind=="training"?av::is_sha256(manifest.resume_identity_sha256):
    manifest.resume_identity_sha256.empty(),"resume digest required only for training archives");
  artifact::validate_component(manifest.component);
}
inline std::string serialize_component(const av::Component& c) {
  using av::detail::quote;
  artifact::validate_component(c);const auto& p=c.parents[0];
  return "{\"module_ref\":{\"component\":"+quote(c.module_ref.component)+",\"implementation_version\":"+
    quote(c.module_ref.implementation_version)+",\"source_sha256\":"+quote(c.module_ref.source_sha256)+
    "},\"weights_ref\":{\"state_sha256\":"+quote(c.state_sha256)+"},\"inference_config_sha256\":"+
    quote(c.inference_config_sha256)+",\"model_ref\":"+quote(c.model_ref)+",\"parents\":[{\"role\":"+
    quote(p.role)+",\"component\":"+quote(p.component)+",\"model_ref\":"+quote(p.model_ref)+"}]}";
}
inline std::string serialize_manifest(const Manifest& manifest) {
  validate_manifest(manifest);using av::detail::quote;
  std::string out="{\"schema\":\"direct_embedding_control_artifact_sidecar_v1\",\"archive_kind\":"+
    quote(manifest.archive_kind)+",\"archive_sha256\":"+quote(manifest.archive_sha256);
  if(manifest.archive_kind=="training")out+=",\"resume_identity_sha256\":"+quote(manifest.resume_identity_sha256);
  return out+",\"component\":"+serialize_component(manifest.component)+"}\n";
}
inline Manifest parse_manifest(std::string_view bytes) {
  using namespace av::detail;
  const auto root=JsonReader(bytes).read();keys(root,{"schema","archive_kind","archive_sha256","component"},{"resume_identity_sha256"});
  require(str(root.object.at("schema"))=="direct_embedding_control_artifact_sidecar_v1","unsupported sidecar schema");
  Manifest result;result.archive_kind=str(root.object.at("archive_kind"));result.archive_sha256=str(root.object.at("archive_sha256"));
  if(root.object.count("resume_identity_sha256"))result.resume_identity_sha256=str(root.object.at("resume_identity_sha256"));
  require(result.archive_kind=="training"||!root.object.count("resume_identity_sha256"),"inference sidecar contains resume state");
  const auto& entry=root.object.at("component");keys(entry,{"module_ref","weights_ref","inference_config_sha256","model_ref","parents"});
  const auto& module=entry.object.at("module_ref");keys(module,{"component","implementation_version","source_sha256"});
  const auto& weights=entry.object.at("weights_ref");keys(weights,{"state_sha256"});
  result.component={{str(module.object.at("component")),str(module.object.at("implementation_version")),str(module.object.at("source_sha256"))},
    str(weights.object.at("state_sha256")),str(entry.object.at("inference_config_sha256")),str(entry.object.at("model_ref")),{}};
  for(const auto& parent:arr(entry.object.at("parents"))) {
    keys(parent,{"role","component","model_ref"});result.component.parents.push_back({str(parent.object.at("role")),
      str(parent.object.at("component")),str(parent.object.at("model_ref"))});
  }
  validate_manifest(result);return result;
}
inline std::string publish_bytes(const fs::path& destination,const std::string& bytes) {
  require(!fs::exists(destination),"immutable publication already exists "+destination.string());
  require(fs::is_directory(destination.parent_path().empty()?fs::path("."):destination.parent_path()),"publication directory must exist");
  fs::path temporary;
  for(unsigned j=0;j<100;++j) {
    temporary=fs::path(destination.string()+".publishing-"+std::to_string(::getpid())+"-"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(j));
    if(fs::create_directory(temporary))break;
    temporary.clear();
  }
  require(!temporary.empty(),"cannot reserve private publication directory");
  try {
    const auto staged=temporary/"record.json";
    {std::ofstream file(staged,std::ios::binary|std::ios::trunc);require(bool(file),"cannot stage publication");
      file.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));file.close();require(bool(file),"publication write failed");}
    fs::permissions(staged,fs::perms::owner_read|fs::perms::group_read|fs::perms::others_read);
    fs::create_hard_link(staged,destination);fs::remove_all(temporary);
  } catch(...) {std::error_code ignored;fs::remove_all(temporary,ignored);throw;}
  return av::sha256_bytes(bytes);
}
inline std::string publish_sidecar(const fs::path& archive,const Manifest& manifest) {
  validate_manifest(manifest);require(av::sha256_file(archive)==manifest.archive_sha256,"archive changed before sidecar publication");
  return publish_bytes(av::sidecar_path(archive),serialize_manifest(manifest));
}
struct VerifiedArchive {std::string bytes,sidecar_sha256;Manifest manifest;};
inline VerifiedArchive read_verified(const fs::path& archive,const std::string& sidecar_pin,const std::string& model_pin,
    const std::string& kind) {
  require(av::is_sha256(sidecar_pin)&&av::is_sha256(model_pin),"trusted sidecar and ModelRef pins are required");
  const auto sidecar=av::detail::read_bytes(av::sidecar_path(archive),262144);
  require(av::sha256_bytes(sidecar)==sidecar_pin,"pinned sidecar digest mismatch");
  auto manifest=parse_manifest(sidecar);require(manifest.archive_kind==kind,"wrong training/inference sidecar kind");
  require(manifest.component.model_ref==model_pin,"pinned fitted control ModelRef mismatch");
  auto bytes=av::detail::read_bytes(archive,uintmax_t{2}*1024*1024*1024);
  require(av::sha256_bytes(bytes)==manifest.archive_sha256,"raw archive digest mismatch");
  return {std::move(bytes),sidecar_pin,std::move(manifest)};
}

namespace checked_detail {
inline void validate_selection(const Config& config,const Identity& identity,const Context& context) {
  validate_context(context);validate(config);validate(identity);
  require(config.fit_cutoff_ms==context.fit_cutoff_ms&&config.spacing_ticks==context.history_spacing_ticks,
    "configured fit cutoff/history spacing differs from selected context");
  require(identity.encoder_model_ref==context.encoder.model_ref,"configured Encoder parent differs from verified selection");
  require(av::sha256_bytes(identity.coordinate_id)==context.coordinates_sha256,"configured coordinates differ from selected identity");
  require(identity.numeric_card==context.protocol_sha256,"trainer numeric protocol differs from comparison registration");
  require(av::is_sha256(identity.source)&&av::is_sha256(identity.pair_ledger),"checked publication requires exact source and pair-ledger SHA256 pins");
  require(context.population_sha256==av::digest_fields("latent-state-control-population-v1",{
    identity.source,identity.pair_ledger,std::to_string(config.fit_cutoff_ms)}),"trainer source/pair/cutoff differs from comparison population");
}
inline av::ModuleRef module(const Context& context) {
  const auto source=av::source_closure_sha256(context.repository_root,{
    "code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/control.h",
    "code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/artifact.h",
    "code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/sidecar.h",
    "code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/tensor_state.h",
    "code/operations/forecasting/evaluation/versions/v1/include/cuwacunu/evaluation/latent_dynamics/v1/baselines.h",
    "code/operations/forecasting/evaluation/versions/v1/src/baselines.cpp"});
#ifdef CUWACUNU_DIRECT_CONTROL_V1_SOURCE_SHA256
  require(source==CUWACUNU_DIRECT_CONTROL_V1_SOURCE_SHA256,
    "current supported source differs from this executable's compiled source identity; rebuild before publication or consumption");
#else
  require(false,"compiled control source identity is required; use the version-owned build target");
#endif
  return {std::string(component_name),"v1",source};
}
inline std::string mapping_digest(const Config& config,const Identity& identity,const Context& context) {
  validate_selection(config,identity,context);
  return av::digest_fields("DirectEmbeddingControlInferenceMapping-v1",{
    detail::canonical(config),detail::canonical(identity),context_sha256(context)});
}
inline av::Component component(const Trainer& trainer,const Context& context) {
  validate_selection(trainer.config,trainer.identity,context);trainer.validate_runtime();
  require(trainer.interval_ms==context.cadence_ms,"fitted candle cadence differs from selected context");
  av::Component result{module(context),trainer.model_state_sha256(),mapping_digest(trainer.config,trainer.identity,context),{},
    {{"embedding","Encoder",context.encoder.model_ref}}};
  result.model_ref=av::derive_model_ref(result);artifact::validate_component(result);return result;
}
inline std::string resume_digest(const Config& config,const Identity& identity,const std::string& archive_sha256) {
  // Raw bytes bind every optimizer slot, local initialization/sampling engine,
  // continuation population and progress counter in addition to their recipe.
  return av::digest_fields("DirectEmbeddingControlExactResume-v1",{
    detail::canonical(config),detail::canonical(identity),archive_sha256});
}
inline std::unique_ptr<Trainer> inspect(const VerifiedArchive& verified,const Config& config,const Identity& identity,
    const Context& context,const std::string& kind) {
  validate_selection(config,identity,context);
  std::istringstream stream(verified.bytes,std::ios::in|std::ios::binary);
  auto trainer=kind=="training"?Trainer::load_training(stream,config,identity):Trainer::load_inference(stream,config,identity);
  trainer->validate_runtime();
  require(verified.manifest.component==component(*trainer,context),"control source/state/configuration/coordinate/Encoder identity mismatch");
  if(kind=="training")require(verified.manifest.resume_identity_sha256==resume_digest(config,identity,verified.manifest.archive_sha256),
    "resume recipe or complete optimizer/RNG/sampler/population/progress identity mismatch");
  return trainer;
}
} // namespace checked_detail

inline Manifest save(const Trainer& trainer,const fs::path& path,const Context& context,const std::string& kind) {
  checked_detail::validate_selection(trainer.config,trainer.identity,context);trainer.validate_runtime();
  require(kind=="training"||kind=="inference","unsupported publication kind");
  require(!fs::exists(path)&&!fs::exists(av::sidecar_path(path)),"immutable archive/sidecar pair already exists");
  const auto original_component=checked_detail::component(trainer,context);
  if(kind=="training")trainer.save_training(path.string());else trainer.save_inference(path.string());
  VerifiedArchive verified{av::detail::read_bytes(path,uintmax_t{2}*1024*1024*1024),{}, {}};
  verified.manifest={kind,av::sha256_bytes(verified.bytes),{},original_component};
  if(kind=="training")verified.manifest.resume_identity_sha256=checked_detail::resume_digest(trainer.config,trainer.identity,verified.manifest.archive_sha256);
  // Deserialize the immutable bytes and recompute state before publication.
  (void)checked_detail::inspect(verified,trainer.config,trainer.identity,context,kind);
  require(original_component==checked_detail::component(trainer,context),"control changed during publication");
  publish_sidecar(path,verified.manifest);return verified.manifest;
}
inline Manifest save_training(const Trainer& trainer,const fs::path& path,const Context& context) {
  return save(trainer,path,context,"training");
}
inline Manifest save_inference(const Trainer& trainer,const fs::path& path,const Context& context) {
  return save(trainer,path,context,"inference");
}

class CheckedInference {
  std::unique_ptr<Trainer> trainer_;
  av::Component component_;
  std::string sidecar_sha256_;
  Context context_;
  CheckedInference(std::unique_ptr<Trainer> trainer,av::Component component,std::string sidecar,Context context)
    :trainer_(std::move(trainer)),component_(std::move(component)),sidecar_sha256_(std::move(sidecar)),context_(std::move(context)){}
 public:
  CheckedInference(const CheckedInference&)=delete;
  CheckedInference& operator=(const CheckedInference&)=delete;
  CheckedInference(CheckedInference&&)=default;
  CheckedInference& operator=(CheckedInference&&)=default;
  template<class Bank,class Pairs> auto predict(const Bank& bank,const Pairs& pairs) const {
    torch::NoGradGuard guard;return trainer_->predict(bank,pairs);
  }
  const av::Component& component() const{return component_;}
  const std::string& sidecar_sha256() const{return sidecar_sha256_;}
  const Context& context() const{return context_;}
  Config config() const{return trainer_->config;}
  friend CheckedInference load_inference(const fs::path&,const std::string&,const std::string&,const Config&,const Identity&,const Context&);
};
inline CheckedInference load_inference(const fs::path& path,const std::string& sidecar_pin,const std::string& model_pin,
    const Config& config,const Identity& identity,const Context& context) {
  auto verified=read_verified(path,sidecar_pin,model_pin,"inference");
  auto trainer=checked_detail::inspect(verified,config,identity,context,"inference");
  return CheckedInference(std::move(trainer),verified.manifest.component,sidecar_pin,context);
}
inline std::unique_ptr<Trainer> load_training(const fs::path& path,const std::string& sidecar_pin,const std::string& model_pin,
    const Config& config,const Identity& identity,const Context& context) {
  auto verified=read_verified(path,sidecar_pin,model_pin,"training");
  return checked_detail::inspect(verified,config,identity,context,"training");
}

struct ComparisonSelection {
  std::string role,model_ref,sidecar_sha256,output_sha256;
  bool operator==(const ComparisonSelection&) const=default;
};
struct ComparisonReceipt {
  std::string protocol_sha256,population_sha256,coordinates_sha256,encoder_model_ref,bundle_ref;
  std::vector<ComparisonSelection> controls;
  bool operator==(const ComparisonReceipt&) const=default;
};
inline std::string derive_bundle_ref(const ComparisonReceipt& receipt) {
  std::vector<std::string> fields{receipt.protocol_sha256,receipt.population_sha256,receipt.coordinates_sha256,
    receipt.encoder_model_ref,std::to_string(receipt.controls.size())};
  for(const auto& item:receipt.controls) {
    fields.push_back(item.role);fields.push_back(std::string(component_name));fields.push_back(item.model_ref);
    fields.push_back(item.sidecar_sha256);fields.push_back(item.output_sha256);
  }
  return av::digest_fields("DirectEmbeddingControlComparisonBundleRef-v1",fields);
}
inline void validate_comparison(const ComparisonReceipt& receipt) {
  for(const auto* value:{&receipt.protocol_sha256,&receipt.population_sha256,&receipt.coordinates_sha256,
      &receipt.encoder_model_ref,&receipt.bundle_ref})require(av::is_sha256(*value),"comparison requires every trusted identity pin");
  require(!receipt.controls.empty()&&receipt.controls.size()<=16,"comparison requires one to sixteen fitted controls");
  std::string previous;
  for(const auto& item:receipt.controls) {
    require(!item.role.empty()&&item.role.size()<=256&&(previous.empty()||previous<item.role),"comparison roles must be unique and ordered");
    require(av::is_sha256(item.model_ref)&&av::is_sha256(item.sidecar_sha256)&&av::is_sha256(item.output_sha256),"comparison control pins are required");
    previous=item.role;
  }
  require(receipt.bundle_ref==derive_bundle_ref(receipt),"comparison BundleRef differs from its pinned controls/outputs");
}
inline ComparisonReceipt make_comparison(std::string protocol,std::string population,std::string coordinates,
    std::string encoder,std::vector<ComparisonSelection> controls) {
  std::sort(controls.begin(),controls.end(),[](const auto& a,const auto& b){return a.role<b.role;});
  ComparisonReceipt receipt{std::move(protocol),std::move(population),std::move(coordinates),std::move(encoder),{},std::move(controls)};
  receipt.bundle_ref=derive_bundle_ref(receipt);validate_comparison(receipt);return receipt;
}
inline std::string serialize_comparison(const ComparisonReceipt& receipt) {
  validate_comparison(receipt);using av::detail::quote;
  std::string out="{\"schema\":\"direct_embedding_control_comparison_v1\",\"protocol_sha256\":"+quote(receipt.protocol_sha256)+
    ",\"population_sha256\":"+quote(receipt.population_sha256)+",\"coordinates_sha256\":"+quote(receipt.coordinates_sha256)+
    ",\"encoder_model_ref\":"+quote(receipt.encoder_model_ref)+",\"bundle_ref\":"+quote(receipt.bundle_ref)+",\"controls\":[";
  for(size_t j=0;j<receipt.controls.size();++j) {
    if(j)out+=',';
    const auto& item=receipt.controls[j];out+="{\"role\":"+quote(item.role)+",\"model_ref\":"+quote(item.model_ref)+
      ",\"sidecar_sha256\":"+quote(item.sidecar_sha256)+",\"output_sha256\":"+quote(item.output_sha256)+"}";
  }
  return out+"]}\n";
}
inline ComparisonReceipt parse_comparison(std::string_view bytes) {
  using namespace av::detail;
  const auto root=JsonReader(bytes).read();keys(root,{"schema","protocol_sha256","population_sha256","coordinates_sha256","encoder_model_ref","bundle_ref","controls"});
  require(str(root.object.at("schema"))=="direct_embedding_control_comparison_v1","unsupported comparison schema");
  ComparisonReceipt receipt{str(root.object.at("protocol_sha256")),str(root.object.at("population_sha256")),
    str(root.object.at("coordinates_sha256")),str(root.object.at("encoder_model_ref")),str(root.object.at("bundle_ref")),{}};
  for(const auto& item:arr(root.object.at("controls"))) {
    keys(item,{"role","model_ref","sidecar_sha256","output_sha256"});receipt.controls.push_back({str(item.object.at("role")),
      str(item.object.at("model_ref")),str(item.object.at("sidecar_sha256")),str(item.object.at("output_sha256"))});
  }
  validate_comparison(receipt);return receipt;
}
inline std::string publish_comparison(const fs::path& path,const ComparisonReceipt& receipt) {
  return publish_bytes(path,serialize_comparison(receipt));
}
inline ComparisonReceipt read_comparison(const fs::path& path,const std::string& receipt_pin,const std::string& protocol,
    const std::string& population,const std::string& coordinates,const std::string& encoder) {
  require(av::is_sha256(receipt_pin),"trusted comparison receipt pin is required");
  const auto bytes=av::detail::read_bytes(path,262144);require(av::sha256_bytes(bytes)==receipt_pin,"comparison receipt digest mismatch");
  auto receipt=parse_comparison(bytes);
  require(receipt.protocol_sha256==protocol&&receipt.population_sha256==population&&receipt.coordinates_sha256==coordinates&&
    receipt.encoder_model_ref==encoder,"comparison protocol/population/coordinate/Encoder mismatch");return receipt;
}
inline const ComparisonSelection& selected_comparison(const ComparisonReceipt& receipt,const std::string& role,
    const CheckedInference& fitted) {
  validate_comparison(receipt);const auto& context=fitted.context();
  require(receipt.protocol_sha256==context.protocol_sha256&&receipt.population_sha256==context.population_sha256&&
    receipt.coordinates_sha256==context.coordinates_sha256&&receipt.encoder_model_ref==context.encoder.model_ref,
    "comparison is incompatible with the fitted control context");
  const auto selected=std::find_if(receipt.controls.begin(),receipt.controls.end(),[&](const auto& item){return item.role==role;});
  require(selected!=receipt.controls.end(),"comparison omitted selected fitted control role");
  require(selected->model_ref==fitted.component().model_ref&&selected->sidecar_sha256==fitted.sidecar_sha256(),
    "comparison attempted fitted control/sidecar substitution");
  return *selected;
}
inline void verify_comparison(const ComparisonReceipt& receipt,const std::string& role,const CheckedInference& fitted,
    const fs::path& published_output) {
  const auto& selected=selected_comparison(receipt,role,fitted);
  require(selected.output_sha256==av::sha256_file(published_output),"comparison output bytes changed");
}
inline cuwacunu::evaluation::latent_dynamics::v1::PredictionSet load_comparison_predictions(
    const ComparisonReceipt& receipt,const std::string& role,const CheckedInference& fitted,const fs::path& published_output) {
  const auto& selected=selected_comparison(receipt,role,fitted);
  const auto bytes=av::detail::read_bytes(published_output,uintmax_t{2}*1024*1024*1024);
  require(selected.output_sha256==av::sha256_bytes(bytes),"comparison output bytes changed");
  // The baseline reader accepts a pathname. This sealed anonymous file supplies
  // exactly the verified output bytes; the original path is never read again.
  const int fd=static_cast<int>(::syscall(SYS_memfd_create,"control-comparison-output-v1",MFD_CLOEXEC|MFD_ALLOW_SEALING));
  require(fd>=0,"cannot stage checked comparison output");
  try {
    size_t offset=0;
    while(offset<bytes.size()) {
      const auto count=::write(fd,bytes.data()+offset,bytes.size()-offset);
      if(count<0&&errno==EINTR)continue;
      require(count>0,"cannot write checked comparison output");offset+=static_cast<size_t>(count);
    }
    require(::fcntl(fd,F_ADD_SEALS,F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL)==0,"cannot seal checked comparison output");
    auto result=cuwacunu::evaluation::latent_dynamics::v1::load_predictions("/proc/self/fd/"+std::to_string(fd));
    require(av::sha256_bytes(result.coordinate_id)==receipt.coordinates_sha256,"published output coordinate identity differs");
    const auto config=fitted.config();
    require(result.fit_cutoff_ms==config.fit_cutoff_ms&&result.history_size==config.history_size&&
      result.spacing_ticks==config.spacing_ticks&&result.method=="residual-mlp-k"+std::to_string(config.history_size)&&
      result.values.dim()==2&&result.values.size(1)==config.width,"published output metadata differs from fitted control configuration");
    ::close(fd);return result;
  } catch(...) {::close(fd);throw;}
}
inline std::string prediction_output_sha256(const torch::Tensor& predictions,const torch::Tensor& origin_ids,
    const std::vector<int64_t>& horizons,const std::string& coordinates) {
  require(av::is_sha256(coordinates)&&!horizons.empty(),"output requires coordinate and horizon identities");
  require(predictions.defined()&&predictions.device().is_cpu()&&predictions.scalar_type()==torch::kFloat64&&
    predictions.dim()==3&&predictions.size(0)>0&&predictions.size(1)==static_cast<int64_t>(horizons.size())&&predictions.size(2)>0&&
    torch::isfinite(predictions).all().item<bool>(),"output must be finite CPU float64 [origin,horizon,coordinate]");
  require(origin_ids.defined()&&origin_ids.device().is_cpu()&&origin_ids.scalar_type()==torch::kInt64&&origin_ids.dim()==1&&
    origin_ids.numel()==predictions.size(0),"output requires exact origin IDs");
  std::set<int64_t> unique;for(int64_t j=0;j<origin_ids.numel();++j)require(unique.insert(origin_ids[j].item<int64_t>()).second,"duplicate output origin");
  std::vector<std::string> fields{av::tensor_state_sha256({{"predictions",predictions},{"origin_ids",origin_ids}}),coordinates};
  int64_t previous=0;for(auto horizon:horizons) {require(horizon>previous,"output horizons must be positive and increasing");fields.push_back(std::to_string(horizon));previous=horizon;}
  return av::digest_fields("DirectEmbeddingControlPredictionOutput-v1",fields);
}
} // namespace cuwacunu::networks::direct_embedding_control::v1::artifact
