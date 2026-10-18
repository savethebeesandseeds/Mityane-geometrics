// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/artifacts/v1/sidecar.h"
#include "cuwacunu/artifacts/v1/tensor_state.h"
#include "cuwacunu/networks/predictive_training/v1/training.h"
#include "cuwacunu/networks/predictive_training/v1/serving.h"
#include <cerrno>
#include <iterator>
#include <linux/memfd.h>
#include <sys/syscall.h>

// Additive checked loading. The version-owned training/archive arithmetic stays
// in training.h; all hashes describe the original saved bytes and mappings.
namespace cuwacunu::networks::predictive_training::v1::artifact {
namespace av = cuwacunu::artifacts::v1;
namespace fs = std::filesystem;

struct Context {
  // Supplied by the independently pinned, verified Encoder selection. The raw
  // archive identity also checks the legacy archive's encoder_checkpoint field.
  av::Component encoder;
  std::string encoder_archive_sha256;
  fs::path repository_root;
  int64_t cadence_ms{0}, history_spacing_ticks{0};
  std::vector<int64_t> channel_ids;
  std::vector<std::string> feature_names;
  std::string embedding_export_layout, availability_policy;
};

namespace checked_detail {
inline void require(bool value, const std::string& message) {
  TORCH_CHECK(value, "[predictive artifact v1] ", message);
}
inline std::string name(std::string_view text) { return std::string(text); }
inline void validate_context(const Context& context, const Identity& identity, bool check_archive_identity=true) {
  validate(identity);
  require(context.encoder.module_ref.component == av::encoder_component,
          "Encoder parent must be the verified Encoder component");
  av::validate_parents({context.encoder});
  require(av::is_sha256(context.encoder_archive_sha256) &&
          (!check_archive_identity || context.encoder_archive_sha256 == identity.encoder_checkpoint),
          "Encoder archive identity differs from the verified Encoder parent");
  require(context.cadence_ms > 0 && context.history_spacing_ticks > 0 &&
          !context.channel_ids.empty() && !context.feature_names.empty() &&
          !context.embedding_export_layout.empty() && !context.availability_policy.empty(),
          "explicit cadence, history spacing, semantic order, export and availability are required");
  require(std::set<int64_t>(context.channel_ids.begin(),context.channel_ids.end()).size()==context.channel_ids.size(),
          "duplicate semantic channel ID");
  require(std::set<std::string>(context.feature_names.begin(),context.feature_names.end()).size()==context.feature_names.size(),
          "duplicate semantic feature name");
  for (const auto& field : context.feature_names) require(!field.empty(), "empty semantic feature name");
}

// Archive inspection instantiates the historical modules. Their constructors
// consume Torch RNG even in inference, so preserve every existing stream.
class RngGuard {
  torch::Tensor cpu_;
  std::vector<torch::Tensor> cuda_;
  bool restore_{true};
 public:
  RngGuard() : cpu_(at::detail::getDefaultCPUGenerator().get_state()) {
    for (size_t j=0;j<torch::cuda::device_count();++j)
      cuda_.push_back(at::cuda::detail::getDefaultCUDAGenerator(j).get_state());
  }
  RngGuard(const RngGuard&) = delete;
  ~RngGuard() {
    if (!restore_) return;
    auto cpu=at::detail::getDefaultCPUGenerator();cpu.set_state(cpu_);
    for (size_t j=0;j<cuda_.size();++j) {
      auto generator=at::cuda::detail::getDefaultCUDAGenerator(j);generator.set_state(cuda_[j]);
    }
  }
  void keep_current() { restore_=false; }
};

// Existing exact-resume reader accepts a pathname. A sealed anonymous file
// gives it precisely the bytes already hashed, without a mutable path race.
class ImmutableArchive {
  int fd_{-1};
 public:
  explicit ImmutableArchive(const std::string& bytes) {
    fd_=static_cast<int>(::syscall(SYS_memfd_create,"predictive-verified-v1",MFD_CLOEXEC|MFD_ALLOW_SEALING));
    require(fd_>=0,"cannot create immutable verified archive staging");
    try {
      size_t offset=0;
      while (offset<bytes.size()) {
        const auto count=::write(fd_,bytes.data()+offset,bytes.size()-offset);
        if (count<0 && errno==EINTR) continue;
        require(count>0,"cannot write verified archive staging");offset+=static_cast<size_t>(count);
      }
      require(::fcntl(fd_,F_ADD_SEALS,F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL)==0,
              "cannot seal verified archive staging");
    } catch (...) { ::close(fd_);fd_=-1;throw; }
  }
  ImmutableArchive(const ImmutableArchive&)=delete;
  ~ImmutableArchive() { if(fd_>=0)::close(fd_); }
  std::string path() const { return "/proc/self/fd/"+std::to_string(fd_); }
};

inline Config parse_config(const std::string& text) {
  require(text.size()<=65536,"oversized archived configuration");
  std::istringstream stream(text);std::string tag,device;std::getline(stream,tag);
  require(tag=="predictive-v1","unsupported archived predictive configuration");
  Config config;int dtype=0;int64_t arm=0;size_t count=0;
  stream>>config.state.embedding_width>>config.state.state_width>>config.state.identity_initialization>>device>>dtype;
  require(!stream.fail(),"malformed archived State Representation Network configuration");
  config.state.device=torch::Device(device);config.state.dtype=static_cast<torch::ScalarType>(dtype);
  config.dynamics.state_width=config.state.state_width;config.dynamics.device=config.state.device;
  config.dynamics.dtype=config.state.dtype;
  stream>>config.dynamics.history_length>>config.dynamics.residual>>count;
  require(!stream.fail() && count>0 && count<=4096,"malformed archived Dynamics Model horizons");
  config.dynamics.horizons.resize(count);for(auto& horizon:config.dynamics.horizons)stream>>horizon;
  stream>>arm>>config.lambda_prediction>>config.lambda_variance>>config.lambda_covariance
    >>config.variance_floor>>config.covariance_epsilon>>config.ema_decay>>config.learning_rate
    >>config.beta1>>config.beta2>>config.optimizer_epsilon>>config.weight_decay>>config.clip_norm>>config.sampler_seed;
  require(!stream.fail(),"malformed archived training configuration");config.arm=static_cast<Arm>(arm);
  require(detail::canonical(config)==text,"noncanonical or unsupported archived configuration");return config;
}
inline Identity parse_identity(const std::string& text) {
  require(text.size()<=200000,"oversized archived identity");std::istringstream stream(text);Identity identity;
  for(auto* field:{&identity.encoder_checkpoint,&identity.encoder_scaler,&identity.embedding_transform,
      &identity.reference_scale,&identity.source,&identity.split,&identity.pair_ledger,
      &identity.sampler_recipe,&identity.loss_recipe,&identity.serving_layout}) {
    size_t length=0;char colon=0,newline=0;stream>>length;stream.get(colon);
    require(!stream.fail() && colon==':' && length>0 && length<=16384,"malformed archived identity length");
    field->resize(length);stream.read(field->data(),static_cast<std::streamsize>(length));stream.get(newline);
    require(!stream.fail() && newline=='\n',"malformed archived identity field");
  }
  require(detail::canonical(identity)==text,"noncanonical archived identity");return identity;
}
struct Inspected {
  std::unique_ptr<Trainer> trainer;
  std::string archive_kind;
};
inline Inspected inspect(const std::string& bytes,const torch::Device& device,const std::string& expected_kind) {
  RngGuard preserve_rng;std::istringstream stream(bytes,std::ios::in|std::ios::binary);
  torch::serialize::InputArchive archive;archive.load_from(stream,device);
  const auto kind=detail::read_text(archive,"artifact_kind");
  require((expected_kind=="training" && kind==training_kind) ||
          (expected_kind=="inference" && kind==inference_kind),"wrong training/inference archive kind");
  auto config=parse_config(detail::read_text(archive,"config"));
  require(config.state.device==device,"archive execution device differs from the selected device");
  const auto identity=parse_identity(detail::read_text(archive,"identity"));
  auto mean=detail::read_tensor(archive,"embedding_mean"),scale=detail::read_tensor(archive,"embedding_scale");
  auto trainer=std::make_unique<Trainer>(config,identity,mean,scale,detail::read_tensor(archive,"reference_scale"));
  trainer->progress=detail::read_progress(archive);
  torch::serialize::InputArchive online,target,dynamics;
  archive.read("online",online);archive.read("target",target);archive.read("dynamics",dynamics);
  trainer->online->load(online);trainer->target->load(target);trainer->dynamics->load(dynamics);
  require(torch::equal(mean,trainer->online->embedding_mean) && torch::equal(scale,trainer->online->embedding_scale),
          "duplicate frozen State Representation Network transform fields disagree");
  trainer->validate_runtime();return {std::move(trainer),expected_kind};
}

inline av::ModuleRef module(const Context& context,std::string_view component) {
  const bool dynamics=component==av::dynamics_component;
  const fs::path header=dynamics?
    "code/networks/latent_dynamics/versions/v1/include/cuwacunu/networks/latent_dynamics/v1/dynamics.h":
    "code/networks/state_representation/versions/v1/include/cuwacunu/networks/state_representation/v1/state.h";
  return {name(component),"v1",av::source_closure_sha256(context.repository_root,{header})};
}
inline std::vector<av::ModuleRef> current_modules(const Context& context) {
  return {module(context,av::state_representation_component),module(context,av::target_state_representation_component),
          module(context,av::dynamics_component)};
}
inline void verify_historical_modules(const Context& context,const std::vector<av::ModuleRef>& historical_modules) {
  require(historical_modules.size()==3,"legacy adoption requires three independently verified historical ModuleRefs");
  std::set<std::string> names;
  for(const auto& historical:historical_modules) {
    require(names.insert(historical.component).second,"duplicate historical ModuleRef "+historical.component);
    require(historical.component==av::state_representation_component ||
            historical.component==av::target_state_representation_component || historical.component==av::dynamics_component,
            "unexpected historical ModuleRef "+historical.component);
    require(historical==module(context,historical.component),
            historical.component+" historical implementation differs from supported version/source bytes");
  }
}
inline std::string mapping_digest(const Config& config,const Identity& identity,const Context& context,
                                  std::string_view component) {
  // Archived provenance may name another byte packaging of the same exact
  // Encoder mapping. Inference parents bind its ModelRef; adoption and resume
  // separately require their independently supplied raw archive identities.
  validate(config);validate_context(context,identity,false);
  std::vector<std::string> fields{name(component),std::to_string(config.state.embedding_width),
    std::to_string(config.state.state_width),std::to_string(static_cast<int>(config.state.dtype)),
    config.state.device.str(),identity.embedding_transform,identity.serving_layout,
    std::to_string(context.cadence_ms),context.embedding_export_layout,context.availability_policy,
    std::to_string(context.channel_ids.size())};
  for(auto id:context.channel_ids)fields.push_back(std::to_string(id));
  fields.push_back(std::to_string(context.feature_names.size()));
  for(const auto& feature:context.feature_names)fields.push_back(feature);
  if(component==av::dynamics_component) {
    fields.push_back(std::to_string(config.dynamics.history_length));
    fields.push_back(std::to_string(context.history_spacing_ticks));fields.push_back("oldest-to-newest");
    fields.push_back(config.dynamics.residual?"last-state-residual":"direct-affine");
    fields.push_back(std::to_string(config.dynamics.horizons.size()));
    for(auto horizon:config.dynamics.horizons)fields.push_back(std::to_string(horizon));
  }
  return av::digest_fields("predictive-inference-mapping-v1",fields);
}
inline std::string resume_digest(const Config& config,const Identity& identity) {
  return av::digest_fields("predictive-exact-resume-v1",{detail::canonical(config),detail::canonical(identity)});
}
inline std::vector<av::Component> components(const Trainer& trainer,const Context& context) {
  validate_context(context,trainer.identity,false);trainer.validate_runtime();
  const auto parent=av::ParentRef{"embedding",name(av::encoder_component),context.encoder.model_ref};
  auto online=av::make_component(module(context,av::state_representation_component),
    av::module_state_sha256(*trainer.online),mapping_digest(trainer.config,trainer.identity,context,av::state_representation_component),{parent});
  auto target=av::make_component(module(context,av::target_state_representation_component),
    av::module_state_sha256(*trainer.target),mapping_digest(trainer.config,trainer.identity,context,av::target_state_representation_component),{parent});
  auto dynamics=av::make_component(module(context,av::dynamics_component),
    av::module_state_sha256(*trainer.dynamics),mapping_digest(trainer.config,trainer.identity,context,av::dynamics_component),
    {{"input_history",name(av::state_representation_component),online.model_ref},
     {"prediction_target",name(av::target_state_representation_component),target.model_ref}});
  return {online,target,dynamics};
}
inline void validate_components(const av::Manifest& manifest,const Trainer& trainer,
                                const Config& expected,const Identity& identity,const Context& context) {
  const auto actual=components(trainer,context);
  require(manifest.components.size()==actual.size(),"joint predictive archive requires exactly three component entries");
  for(const auto& component:actual) {
    const auto found=std::find_if(manifest.components.begin(),manifest.components.end(),[&](const auto& candidate){
      return candidate.module_ref.component==component.module_ref.component;
    });
    require(found!=manifest.components.end(),"missing "+component.module_ref.component+" entry");
    require(found->model_ref==component.model_ref,
            component.module_ref.component+" source/state/configuration/parent identity mismatch: expected "+
            found->model_ref+", actual "+component.model_ref);
    require(found->inference_config_sha256==mapping_digest(expected,identity,context,found->module_ref.component),
            found->module_ref.component+" selected inference configuration mismatch");
  }
  auto selection=manifest.components;selection.push_back(context.encoder);av::validate_parents(selection);
}
} // namespace checked_detail

// No module handles leave this object. All tensors belong to the independently
// deserialized archive, and returned folding/reference tensors are detached copies.
class CheckedInference {
  mutable sr::AffineState online_{nullptr},target_{nullptr};
  mutable ld::AffineDynamics dynamics_{nullptr};
  torch::Tensor reference_;
  std::vector<av::Component> components_;
 public:
  CheckedInference(const CheckedInference&)=delete;
  CheckedInference& operator=(const CheckedInference&)=delete;
  CheckedInference(CheckedInference&&)=default;
  CheckedInference& operator=(CheckedInference&&)=default;
  torch::Tensor forecast(const torch::Tensor& history) const {
    torch::NoGradGuard guard;return dynamics_->forward(online_->forward(history));
  }
  torch::Tensor state_history(const torch::Tensor& history) const {
    torch::NoGradGuard guard;return online_->forward(history);
  }
  torch::Tensor target_states(const torch::Tensor& embeddings) const {
    torch::NoGradGuard guard;return target_->forward(embeddings);
  }
  FoldedAffine fold_forecast() const { return fold_state_forecast(online_,dynamics_); }
  FoldedAffine fold_history() const { return fold_state_history(online_,dynamics_->config.history_length); }
  // Training/evaluation provenance only; no forward/folding mapping uses this
  // scale. Evaluators continue to pin the complete inference archive bytes.
  torch::Tensor reference_scale() const { return reference_.detach().clone(); }
  const std::vector<av::Component>& components() const { return components_; }
 private:
  CheckedInference(std::unique_ptr<Trainer> trainer,std::vector<av::Component> components)
    :online_(trainer->online),target_(trainer->target),dynamics_(trainer->dynamics),
     reference_(trainer->reference_scale.detach().clone()),components_(std::move(components)) {
    for(auto* model:{static_cast<torch::nn::Module*>(online_.get()),static_cast<torch::nn::Module*>(target_.get()),
                    static_cast<torch::nn::Module*>(dynamics_.get())}) {
      model->eval();for(auto& parameter:model->parameters())parameter.set_requires_grad(false);
    }
  }
  friend CheckedInference load_inference(const fs::path&,const std::string&,const Config&,const Identity&,const Context&);
};

inline av::Manifest adopt(const fs::path& path,const Config& expected,const Identity& identity,
                          const Context& context,const std::string& archive_kind,
                          const std::vector<av::ModuleRef>& historical_modules) {
  checked_detail::validate_context(context,identity);validate(expected);
  // These pins come from independently established historical source evidence.
  // Never derive them from current files when adopting an existing archive.
  checked_detail::verify_historical_modules(context,historical_modules);
  checked_detail::require(!fs::exists(path.string()+".manifest.json"),"immutable sidecar already exists");
  std::ifstream file(path,std::ios::binary);checked_detail::require(file.good(),"cannot read archive for additive adoption");
  const std::string bytes{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
  checked_detail::require(!file.bad(),"archive read failed");
  auto inspected=checked_detail::inspect(bytes,expected.state.device,archive_kind);
  checked_detail::require(detail::canonical(inspected.trainer->config)==detail::canonical(expected) &&
      detail::canonical(inspected.trainer->identity)==detail::canonical(identity),
      "legacy adoption requires the exact independently verified historical configuration and identity");
  if(archive_kind=="training") {
    checked_detail::RngGuard preserve_rng;checked_detail::ImmutableArchive immutable(bytes);
    inspected.trainer=Trainer::load_training(immutable.path(),expected,identity);
  }
  av::Manifest manifest;manifest.archive_kind=archive_kind;manifest.archive_sha256=av::sha256_bytes(bytes);
  manifest.components=checked_detail::components(*inspected.trainer,context);
  if(archive_kind=="training")manifest.resume_identity_sha256=checked_detail::resume_digest(expected,identity);
  checked_detail::validate_components(manifest,*inspected.trainer,expected,identity,context);
  // Inspection and digesting can take time. The published identities must
  // still be the independently pinned historical sources, even if files were
  // edited between the initial check and construction of these components.
  checked_detail::verify_historical_modules(context,historical_modules);
  for(const auto& component:manifest.components) {
    const auto historical=std::find_if(historical_modules.begin(),historical_modules.end(),[&](const auto& candidate){
      return candidate.component==component.module_ref.component;
    });
    checked_detail::require(historical!=historical_modules.end() && component.module_ref==*historical,
                            component.module_ref.component+" source changed during additive adoption");
  }
  av::publish_sidecar(path,manifest);return manifest;
}
inline av::Manifest save_training(const Trainer& trainer,const fs::path& path,const Context& context) {
  checked_detail::validate_context(context,trainer.identity);
  checked_detail::require(!fs::exists(path.string()+".manifest.json"),"immutable sidecar already exists");
  const auto saved_modules=checked_detail::current_modules(context);
  trainer.save_training(path.string());return adopt(path,trainer.config,trainer.identity,context,"training",saved_modules);
}
inline av::Manifest save_inference(const Trainer& trainer,const fs::path& path,const Context& context) {
  checked_detail::validate_context(context,trainer.identity);
  checked_detail::require(!fs::exists(path.string()+".manifest.json"),"immutable sidecar already exists");
  const auto saved_modules=checked_detail::current_modules(context);
  trainer.save_inference(path.string());return adopt(path,trainer.config,trainer.identity,context,"inference",saved_modules);
}
inline std::unique_ptr<Trainer> load_training(const fs::path& path,const std::string& sidecar_sha256,
    const Config& expected,const Identity& identity,const Context& context) {
  checked_detail::validate_context(context,identity);validate(expected);
  const auto verified=av::read_verified(path,sidecar_sha256,"training");
  checked_detail::require(verified.manifest.resume_identity_sha256==checked_detail::resume_digest(expected,identity),
                         "exact-resume training configuration or immutable identity differs");
  auto inspected=checked_detail::inspect(verified.bytes,expected.state.device,"training");
  checked_detail::validate_components(verified.manifest,*inspected.trainer,expected,identity,context);
  checked_detail::RngGuard restore_on_failure;checked_detail::ImmutableArchive immutable(verified.bytes);
  auto trainer=Trainer::load_training(immutable.path(),expected,identity);
  checked_detail::validate_components(verified.manifest,*trainer,expected,identity,context);
  restore_on_failure.keep_current();return trainer;
}
inline CheckedInference load_inference(const fs::path& path,const std::string& sidecar_sha256,
    const Config& expected,const Identity& identity,const Context& context) {
  checked_detail::validate_context(context,identity);validate(expected);
  const auto verified=av::read_verified(path,sidecar_sha256,"inference");
  auto inspected=checked_detail::inspect(verified.bytes,expected.state.device,"inference");
  checked_detail::validate_components(verified.manifest,*inspected.trainer,expected,identity,context);
  return CheckedInference(std::move(inspected.trainer),verified.manifest.components);
}
} // namespace cuwacunu::networks::predictive_training::v1::artifact
