// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../LICENSE.
#pragma once
#include "cuwacunu/latent_market/ledger_io.h"
#include "cuwacunu/networks/predictive_training/v1/artifact.h"
#include <memory>

namespace cuwacunu::latent_market::forecast_issuance::v1 {
namespace lm = cuwacunu::latent_market::v1;
namespace av = cuwacunu::artifacts::v1;
namespace pt = cuwacunu::networks::predictive_training::v1;
namespace pa = cuwacunu::networks::predictive_training::v1::artifact;

inline constexpr const char* policy_sha256 = "18fcaeab478e0b66dfe2fca132cc4c5d8ae3de368eb4a74d7428fdf9bfef2805";
inline constexpr const char* stage_kind = "joint-frozen-single-stage";

namespace detail {
inline void require(bool value, const std::string& message) {
  TORCH_CHECK(value, "[forecast issuance v1] ", message);
}
inline const av::Component& component(const av::Manifest& manifest, std::string_view name) {
  const auto found=std::find_if(manifest.components.begin(),manifest.components.end(),[&](const auto& item) {
    return item.module_ref.component==name;
  });
  require(found!=manifest.components.end(),"missing component "+std::string(name));return *found;
}
inline void verified(const av::VerifiedArtifact& artifact, size_t count) {
  av::validate_manifest(artifact.manifest);
  require(artifact.manifest.archive_kind=="inference" && artifact.manifest.components.size()==count,
          "issuance requires complete checked inference artifacts");
  require(av::sha256_bytes(artifact.bytes)==artifact.manifest.archive_sha256,
          "verified archive bytes no longer match their manifest");
  // This version accepts the canonical sidecars emitted by the checked
  // publisher. The byte pin must bind the actual complete manifest as well.
  require(av::sha256_bytes(av::serialize_manifest(artifact.manifest))==artifact.sidecar_sha256,
          "verified sidecar pin differs from canonical manifest bytes");
}
struct Provenance {pt::Config config;pt::Identity identity;};
inline Provenance provenance(const av::VerifiedArtifact& artifact) {
  std::istringstream input(artifact.bytes,std::ios::in|std::ios::binary);
  torch::serialize::InputArchive archive;archive.load_from(input,torch::Device(torch::kCPU));
  require(pt::detail::read_text(archive,"artifact_kind")==pt::inference_kind,"wrong predictive archive type");
  return {pa::checked_detail::parse_config(pt::detail::read_text(archive,"config")),
          pa::checked_detail::parse_identity(pt::detail::read_text(archive,"identity"))};
}
}

struct FitCutoffs {
  int64_t encoder_ms{0},online_ms{0},target_ms{0},dynamics_ms{0};
  bool operator==(const FitCutoffs&) const = default;
};

struct FrozenContract {
  std::string policy_sha256{v1::policy_sha256},stage_kind{v1::stage_kind};
  lm::TimelineIdentity timeline;
  int64_t interval_ms{0},window_ticks{0},history_steps{0},spacing_ticks{0};
  size_t embedding_width{0},state_width{0};
  std::vector<int64_t> horizons_ticks;
  lm::SupportPolicy input_support;
  FitCutoffs fits;
  std::string encoder_archive_sha256,encoder_sidecar_sha256;
  std::string predictive_archive_sha256,predictive_sidecar_sha256;
  std::string timeline_sha256,pairs_sha256,embedding_transform_sha256,reference_scale_sha256;
  av::Component encoder,online,target,dynamics;
  std::string target_coordinate_id,bundle_ref;
};

inline std::string target_coordinate_id(const std::string& timeline_id,const std::string& archive_sha256) {
  return timeline_id+":Target State Representation Network:"+archive_sha256;
}
// Authenticate the original complete captured snapshot once when binding the
// immutable model. Per-origin issuance never serializes or reads future data.
inline std::string timeline_digest(const lm::EmbeddingTimeline& timeline) {
  std::ostringstream output(std::ios::out|std::ios::binary);lm::write_timeline(output,timeline);
  detail::require(bool(output),"cannot hash captured timeline");return av::sha256_bytes(output.str());
}
inline std::vector<av::BundleSelection> bundle_selection(const FrozenContract& c) {
  return {{c.encoder_sidecar_sha256,c.encoder.module_ref.component,c.encoder.model_ref},
    {c.predictive_sidecar_sha256,c.online.module_ref.component,c.online.model_ref},
    {c.predictive_sidecar_sha256,c.target.module_ref.component,c.target.model_ref},
    {c.predictive_sidecar_sha256,c.dynamics.module_ref.component,c.dynamics.model_ref}};
}
inline void validate_contract(const FrozenContract& c) {
  using detail::require;
  require(c.policy_sha256==policy_sha256 && c.stage_kind==stage_kind,
          "unregistered policy, cross-fit or fold pooling is unsupported");
  lm::validate_identity(c.timeline);lm::validate_support_policy(c.input_support);
  require(c.interval_ms>0 && c.window_ticks>0 && c.history_steps>0 && c.spacing_ticks>0 &&
          c.embedding_width>0 && c.state_width>0 && !c.horizons_ticks.empty(),"positive exact geometry required");
  require(c.embedding_width<=static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
          c.state_width<=static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
          c.horizons_ticks.size()<=static_cast<size_t>(std::numeric_limits<int64_t>::max())/c.state_width,
          "forecast dimension overflow");
  lm::temporal_detail::nonnegative_product(c.history_steps-1,c.spacing_ticks);
  lm::temporal_detail::nonnegative_product(c.window_ticks-1,c.interval_ms);
  std::set<int64_t> horizons;
  for(auto h:c.horizons_ticks)require(h>0 && horizons.insert(h).second,"positive unique horizons required");
  require(c.fits.encoder_ms>=0 && c.fits.online_ms>=c.fits.encoder_ms &&
          c.fits.online_ms==c.fits.target_ms && c.fits.online_ms==c.fits.dynamics_ms &&
          c.fits.encoder_ms==c.timeline.encoder_fit_cutoff_ms,
          "joint-frozen single-stage requires one common state/target/Dynamics cutoff after Encoder fit");
  require(av::is_sha256(c.encoder_archive_sha256) && av::is_sha256(c.predictive_archive_sha256) &&
          av::is_sha256(c.encoder_sidecar_sha256) && av::is_sha256(c.predictive_sidecar_sha256) &&
          av::is_sha256(c.timeline_sha256) && av::is_sha256(c.pairs_sha256) &&
          av::is_sha256(c.embedding_transform_sha256) && av::is_sha256(c.reference_scale_sha256),
          "all archive and sidecar pins are required");
  require(c.encoder.module_ref.component==av::encoder_component &&
          c.online.module_ref.component==av::state_representation_component &&
          c.target.module_ref.component==av::target_state_representation_component &&
          c.dynamics.module_ref.component==av::dynamics_component,"role-correct complete components required");
  av::validate_parents({c.encoder,c.online,c.target,c.dynamics});
  require(c.timeline.checkpoint_hash==c.encoder_archive_sha256,"captured timeline has a different Encoder archive");
  require(c.target_coordinate_id==target_coordinate_id(c.timeline.artifact_id,c.predictive_archive_sha256),
          "exact frozen target coordinates required");
  require(c.bundle_ref==av::bundle_ref(bundle_selection(c)),"complete compatible BundleRef mismatch");
}

inline FrozenContract make_contract(const lm::EmbeddingTimeline& timeline,const lm::PairSpec& spec,
    const FitCutoffs& fits,const av::VerifiedArtifact& encoder,const av::VerifiedArtifact& predictive) {
  detail::verified(encoder,1);detail::verified(predictive,3);lm::validate_pair_spec(spec);
  const auto provenance=detail::provenance(predictive);
  detail::require(timeline_digest(timeline)==provenance.identity.source,
                  "captured timeline bytes differ from immutable predictive source provenance");
  FrozenContract c;c.timeline=timeline.identity;c.interval_ms=timeline.interval_ms;c.window_ticks=timeline.window_ticks;
  c.embedding_width=timeline.embedding_width;c.state_width=provenance.config.state.state_width;c.history_steps=spec.history_steps;
  c.spacing_ticks=spec.spacing_ticks;c.horizons_ticks=spec.horizons_ticks;c.input_support=spec.input_support;c.fits=fits;
  c.encoder_archive_sha256=encoder.manifest.archive_sha256;c.encoder_sidecar_sha256=encoder.sidecar_sha256;
  c.predictive_archive_sha256=predictive.manifest.archive_sha256;c.predictive_sidecar_sha256=predictive.sidecar_sha256;
  c.timeline_sha256=provenance.identity.source;c.pairs_sha256=provenance.identity.pair_ledger;
  c.embedding_transform_sha256=provenance.identity.embedding_transform;c.reference_scale_sha256=provenance.identity.reference_scale;
  c.encoder=detail::component(encoder.manifest,av::encoder_component);
  c.online=detail::component(predictive.manifest,av::state_representation_component);
  c.target=detail::component(predictive.manifest,av::target_state_representation_component);
  c.dynamics=detail::component(predictive.manifest,av::dynamics_component);
  c.target_coordinate_id=target_coordinate_id(c.timeline.artifact_id,c.predictive_archive_sha256);
  c.bundle_ref=av::bundle_ref(bundle_selection(c));validate_contract(c);return c;
}

inline detail::Provenance validate_provenance(const FrozenContract& c,const av::VerifiedArtifact& predictive) {
  const auto provenance=detail::provenance(predictive);const auto& id=provenance.identity;const auto& config=provenance.config;
  detail::require(id.split=="fit:"+std::to_string(c.fits.online_ms)+";ledger:"+id.pair_ledger,
                  "component fit cutoff differs from immutable predictive archive provenance");
  detail::require(id.encoder_checkpoint==c.encoder_archive_sha256 && id.encoder_scaler==c.timeline.scaler_hash && id.source==c.timeline_sha256 &&
                  id.pair_ledger==c.pairs_sha256 && id.embedding_transform==c.embedding_transform_sha256 &&
                  id.reference_scale==c.reference_scale_sha256,"archive source/parent/frozen transform provenance differs");
  detail::require(config.state.embedding_width==static_cast<int64_t>(c.embedding_width) &&
                  config.state.state_width==static_cast<int64_t>(c.state_width) &&
                  config.dynamics.history_length==c.history_steps && config.dynamics.horizons==c.horizons_ticks,
                  "archive history, horizon ordering or state dimensions differ from issuance contract");
  const pa::Context context{c.encoder,c.encoder_archive_sha256,{},c.interval_ms,c.spacing_ticks,
    c.timeline.channel_ids,c.timeline.feature_names,c.timeline.export_layout,c.timeline.availability_policy};
  for(const auto* component:{&c.online,&c.target,&c.dynamics})
    detail::require(component->inference_config_sha256==pa::checked_detail::mapping_digest(config,id,context,component->module_ref.component),
                    "contract cadence, history spacing, semantics, export or availability differ from checked mapping");
  return provenance;
}

inline std::string contract_digest(const FrozenContract& c) {
  validate_contract(c);
  std::vector<std::string> fields{c.policy_sha256,c.stage_kind,c.timeline.artifact_id,c.timeline.dataset_id,
    c.timeline.encoder_family,c.timeline.encoder_version,c.timeline.checkpoint_hash,c.timeline.scaler_hash,
    c.timeline.encoder_config_hash,c.timeline.source_hash,c.timeline.cache_hash,c.timeline.export_layout,
    c.timeline.availability_policy,c.timeline.source_closure_hash,c.timeline.executable_hash,
    std::to_string(c.timeline.encoder_fit_cutoff_ms),std::to_string(c.timeline.channel_ids.size())};
  for(auto id:c.timeline.channel_ids)fields.push_back(std::to_string(id));
  fields.push_back(std::to_string(c.timeline.feature_names.size()));
  for(const auto& feature:c.timeline.feature_names)fields.push_back(feature);
  for(auto number:{c.interval_ms,c.window_ticks,c.history_steps,c.spacing_ticks,
      static_cast<int64_t>(c.embedding_width),static_cast<int64_t>(c.state_width),
      c.fits.encoder_ms,c.fits.online_ms,c.fits.target_ms,c.fits.dynamics_ms})fields.push_back(std::to_string(number));
  fields.push_back(std::to_string(c.horizons_ticks.size()));
  for(auto h:c.horizons_ticks)fields.push_back(std::to_string(h));
  fields.push_back(c.input_support.require_complete_window?"complete":"partial");
  for(auto fraction:{c.input_support.minimum_observed_fraction,c.input_support.minimum_channel_fraction,
      c.input_support.minimum_feature_fraction}){std::ostringstream text;text<<std::setprecision(17)<<fraction;fields.push_back(text.str());}
  fields.push_back(std::to_string(c.input_support.maximum_missing_run_ticks));
  fields.push_back(std::to_string(c.input_support.maximum_endpoint_staleness_ticks));
  for(const auto* field:{&c.encoder_archive_sha256,&c.encoder_sidecar_sha256,&c.predictive_archive_sha256,
      &c.predictive_sidecar_sha256,&c.timeline_sha256,&c.pairs_sha256,&c.embedding_transform_sha256,
      &c.reference_scale_sha256,&c.encoder.model_ref,&c.online.model_ref,&c.target.model_ref,&c.dynamics.model_ref,
      &c.target_coordinate_id,&c.bundle_ref})fields.push_back(*field);
  return av::digest_fields("forecast-issuance-frozen-contract-v1",fields);
}

enum Reason : uint64_t {
  no_issue=0,encoder_not_before_origin=1ull<<0,online_not_before_origin=1ull<<1,
  target_not_before_origin=1ull<<2,dynamics_not_before_origin=1ull<<3,
  decision_before_origin=1ull<<4,ledger_input_invalid=1ull<<5,missing_input=1ull<<6,
  invalid_input=1ull<<7,input_support=1ull<<8,input_timestamp_gap=1ull<<9,
  input_role_boundary=1ull<<10,input_unavailable=1ull<<11,input_identity_mismatch=1ull<<12,
  input_shape_mismatch=1ull<<13,input_nonfinite=1ull<<14,origin_invalid=1ull<<15,
  prediction_nonfinite=1ull<<16
};
inline std::vector<std::string> reason_names(uint64_t bits) {
  const std::pair<Reason,const char*> names[]={{encoder_not_before_origin,"encoder_not_before_origin"},
    {online_not_before_origin,"online_not_before_origin"},{target_not_before_origin,"target_not_before_origin"},
    {dynamics_not_before_origin,"dynamics_not_before_origin"},{decision_before_origin,"decision_before_origin"},
    {ledger_input_invalid,"ledger_input_invalid"},{missing_input,"missing_input"},{invalid_input,"invalid_input"},
    {input_support,"input_support"},{input_timestamp_gap,"input_timestamp_gap"},
    {input_role_boundary,"input_role_boundary"},{input_unavailable,"input_unavailable"},
    {input_identity_mismatch,"input_identity_mismatch"},{input_shape_mismatch,"input_shape_mismatch"},
    {input_nonfinite,"input_nonfinite"},{origin_invalid,"origin_invalid"},
    {prediction_nonfinite,"prediction_nonfinite"}};
  std::vector<std::string> result;for(const auto& [bit,name]:names)if(bits&bit)result.emplace_back(name);return result;
}
struct InputAssociation {
  int64_t endpoint_ms{0},available_ms{0};
  std::string window_id;
  std::optional<size_t> row_index;
  uint64_t reasons{no_issue};
  lm::SupportSummary support;
};
struct IssuedOrigin {
  lm::PairOrigin origin;
  torch::Tensor values; // CPU F64 [Q,S]; finite zeros for every invalid origin.
  bool valid{false};
  uint64_t reasons{no_issue},original_input_reasons{0};
  std::vector<InputAssociation> history; // Every required endpoint, oldest first.
};

class Issuer {
  FrozenContract contract_;
  std::shared_ptr<const pa::CheckedInference> serving_;
  torch::TensorOptions options_;
 public:
  Issuer(FrozenContract contract,std::shared_ptr<const pa::CheckedInference> serving,
         const av::VerifiedArtifact& encoder,const av::VerifiedArtifact& predictive,
         const lm::EmbeddingTimeline& captured_timeline,
         const std::filesystem::path& repository_root=std::filesystem::current_path())
    :contract_(std::move(contract)),serving_(std::move(serving)) {
    using detail::require;validate_contract(contract_);detail::verified(encoder,1);detail::verified(predictive,3);
    require(captured_timeline.identity==contract_.timeline && captured_timeline.interval_ms==contract_.interval_ms &&
            captured_timeline.window_ticks==contract_.window_ticks && captured_timeline.embedding_width==contract_.embedding_width &&
            timeline_digest(captured_timeline)==contract_.timeline_sha256,
            "original captured timeline proof differs from exact issuance contract");
    const auto provenance=validate_provenance(contract_,predictive);
    const pa::Context context{contract_.encoder,contract_.encoder_archive_sha256,repository_root,
      contract_.interval_ms,contract_.spacing_ticks,contract_.timeline.channel_ids,contract_.timeline.feature_names,
      contract_.timeline.export_layout,contract_.timeline.availability_policy};
    // Authenticate the actual archive's tensors against its claimed component
    // tuple, rather than trusting a rehashed manifest with substituted bytes.
    const auto inspected=pa::checked_detail::inspect(predictive.bytes,provenance.config.state.device,"inference");
    pa::checked_detail::validate_components(predictive.manifest,*inspected.trainer,provenance.config,provenance.identity,context);
    require(bool(serving_),"an owning checked immutable inference handle is required");
    require(encoder.sidecar_sha256==contract_.encoder_sidecar_sha256 &&
            encoder.manifest.archive_sha256==contract_.encoder_archive_sha256 &&
            predictive.sidecar_sha256==contract_.predictive_sidecar_sha256 &&
            predictive.manifest.archive_sha256==contract_.predictive_archive_sha256,"checked artifact pins differ from issuance selection");
    require(detail::component(encoder.manifest,av::encoder_component)==contract_.encoder,
            "verified Encoder differs from selected component");
    for(const auto* c:{&contract_.online,&contract_.target,&contract_.dynamics}) {
      require(detail::component(predictive.manifest,c->module_ref.component)==*c,
              "verified predictive component differs from issuance selection");
      const auto& actual=serving_->components();
      const auto found=std::find(actual.begin(),actual.end(),*c);
      require(actual.size()==3 && found!=actual.end(),"checked serving tuple differs from selected frozen parents");
    }
    // Read-only folded copies verify serving dimensions without exposing handles
    // or consulting any observed origin/target. Runtime dtype/device remain the
    // checked model's own properties.
    const auto folded=serving_->fold_forecast();
    require(folded.history_length==contract_.history_steps &&
            folded.embedding_width==static_cast<int64_t>(contract_.embedding_width) &&
            folded.weight.size(0)==static_cast<int64_t>(contract_.horizons_ticks.size()*contract_.state_width),
            "serving dimensions differ from issuance geometry");
    options_=folded.weight.options();
  }
  const FrozenContract& contract() const {return contract_;}

  IssuedOrigin issue(const lm::EmbeddingTimeline& timeline,const lm::PairOrigin& origin,
                     const std::vector<lm::RoleInterval>& roles,uint64_t original_input_reasons=0) const {
    using detail::require;
    require(timeline.identity==contract_.timeline && timeline.interval_ms==contract_.interval_ms &&
            timeline.window_ticks==contract_.window_ticks && timeline.embedding_width==contract_.embedding_width,
            "input capture identity/geometry differs from exact issuance contract");
    constexpr uint64_t input_only=lm::missing_input|lm::invalid_input|lm::input_support|lm::input_timestamp_gap|
      lm::input_role_boundary|lm::input_unavailable|lm::decision_before_origin;
    require((original_input_reasons&~input_only)==0,"future/target-side or unknown ledger reasons cannot enter issuance");
    lm::validate_roles(roles);
    IssuedOrigin out;out.origin=origin;out.original_input_reasons=original_input_reasons;
    out.values=torch::zeros({static_cast<int64_t>(contract_.horizons_ticks.size()),static_cast<int64_t>(contract_.state_width)},torch::kFloat64);
    if(origin.pair_id.empty() || origin.dependence_group.empty() || origin.endpoint_ms<0 || origin.decision_ms<0 ||
       !lm::temporal_detail::valid_role(origin.role) || origin.role==lm::Role::unassigned)out.reasons|=origin_invalid;
    if(original_input_reasons)out.reasons|=ledger_input_invalid;
    if(origin.decision_ms<origin.endpoint_ms)out.reasons|=decision_before_origin;
    if(contract_.fits.encoder_ms>=origin.endpoint_ms)out.reasons|=encoder_not_before_origin;
    if(contract_.fits.online_ms>=origin.endpoint_ms)out.reasons|=online_not_before_origin;
    if(contract_.fits.target_ms>=origin.endpoint_ms)out.reasons|=target_not_before_origin;
    if(contract_.fits.dynamics_ms>=origin.endpoint_ms)out.reasons|=dynamics_not_before_origin;
    std::vector<double> values;
    for(int64_t k=contract_.history_steps;k>0;--k) {
      InputAssociation a;
      try {a.endpoint_ms=lm::shift_timestamp(origin.endpoint_ms,-lm::temporal_detail::nonnegative_product(k-1,contract_.spacing_ticks),contract_.interval_ms);}
      catch(const std::exception&) {a.endpoint_ms=-1;a.reasons|=input_timestamp_gap;}
      a.window_id=lm::expected_window_id(contract_.timeline,a.endpoint_ms);
      // Search by exact endpoint only. Other rows' values, masks, eligibility,
      // target identities and availability never enter this path.
      const auto found=std::lower_bound(timeline.rows.begin(),timeline.rows.end(),a.endpoint_ms,
        [](const auto& row,int64_t endpoint){return row.endpoint_ms<endpoint;});
      if(a.endpoint_ms<0 || found==timeline.rows.end() || found->endpoint_ms!=a.endpoint_ms)a.reasons|=missing_input;
      else {
        a.row_index=static_cast<size_t>(found-timeline.rows.begin());a.available_ms=found->available_ms;
        if(found->window_id!=a.window_id)a.reasons|=input_identity_mismatch;
        if(!found->sample_valid)a.reasons|=invalid_input;
        if(found->available_ms<found->endpoint_ms || found->available_ms>origin.decision_ms)a.reasons|=input_unavailable;
        if(found->role!=origin.role)a.reasons|=input_role_boundary;
        const auto& s=found->support;
        const bool shape=s.channels==timeline.identity.channel_ids.size() && s.features==timeline.identity.feature_names.size() &&
          s.timestamps_ms.size()==static_cast<size_t>(timeline.window_ticks) &&
          s.observed.size()==lm::temporal_detail::product(s.timestamps_ms.size(),lm::temporal_detail::product(s.channels,s.features)) && !s.timestamps_ms.empty();
        if(!shape)a.reasons|=input_shape_mismatch;
        else {
          if(std::any_of(s.observed.begin(),s.observed.end(),[](auto mask){return mask>1;}))a.reasons|=input_support;
          a.support=lm::summarize_support(timeline,*found,contract_.input_support);
          if(!a.support.eligible)a.reasons|=input_support;
          if(!a.support.regular_timestamps)a.reasons|=input_timestamp_gap;
          if(!lm::contained_in_role(roles,origin.role,s.timestamps_ms.front(),found->endpoint_ms))a.reasons|=input_role_boundary;
        }
        if(found->values.size()!=contract_.embedding_width)a.reasons|=input_shape_mismatch;
        if(std::any_of(found->values.begin(),found->values.end(),[](double value){return !std::isfinite(value);}))a.reasons|=input_nonfinite;
        if(a.reasons==no_issue)values.insert(values.end(),found->values.begin(),found->values.end());
      }
      out.reasons|=a.reasons;out.history.push_back(std::move(a));
    }
    if(out.reasons)return out;
    const auto input=torch::tensor(values,torch::kFloat64).reshape({1,contract_.history_steps,static_cast<int64_t>(contract_.embedding_width)}).to(options_);
    const auto prediction=serving_->forecast(input);
    require(prediction.dim()==3 && prediction.size(0)==1 && prediction.size(1)==static_cast<int64_t>(contract_.horizons_ticks.size()) &&
            prediction.size(2)==static_cast<int64_t>(contract_.state_width),"checked prediction geometry differs");
    if(!torch::isfinite(prediction).all().item<bool>()){out.reasons|=prediction_nonfinite;return out;}
    out.values=prediction[0].detach().to(torch::kCPU,torch::kFloat64).clone();out.valid=true;return out;
  }
};

// Freeze the wrapper and every local implementation dependency together when
// capturing a new study. Existing evaluated dependency bytes are never edited.
inline std::vector<std::filesystem::path> source_files() {
  return {"code/operations/forecasting/issuance/versions/v1/include/cuwacunu/latent_market/forecast_issuance/v1/issuance.h",
    "code/operations/forecasting/issuance/versions/v1/include/cuwacunu/latent_market/forecast_issuance/v1/publication.h",
    "code/data/market/versions/v1/include/cuwacunu/latent_market/timeline.h",
    "code/data/market/versions/v1/include/cuwacunu/latent_market/arrays.h",
    "code/data/market/versions/v1/include/cuwacunu/latent_market/ledger_io.h",
    "code/data/market/versions/v1/include/cuwacunu/latent_market/risk_targets.h",
    "code/operations/forecasting/evaluation/versions/v1/include/cuwacunu/evaluation/latent_dynamics/v1/baselines.h",
    "code/operations/forecasting/evaluation/versions/v1/src/baselines.cpp",
    "code/operations/forecasting/predictive_training/versions/v1/include/cuwacunu/networks/predictive_training/v1/artifact.h",
    "code/operations/forecasting/predictive_training/versions/v1/include/cuwacunu/networks/predictive_training/v1/training.h",
    "code/operations/forecasting/predictive_training/versions/v1/include/cuwacunu/networks/predictive_training/v1/serving.h",
    "code/embeddings/state_representation/versions/v1/include/cuwacunu/networks/state_representation/v1/state.h",
    "code/operations/forecasting/latent_dynamics/versions/v1/include/cuwacunu/networks/latent_dynamics/v1/dynamics.h",
    "code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/sidecar.h",
    "code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/tensor_state.h"};
}
inline std::string source_closure_sha256(const std::filesystem::path& root) {
  return av::source_closure_sha256(root,source_files());
}
} // namespace cuwacunu::latent_market::forecast_issuance::v1
