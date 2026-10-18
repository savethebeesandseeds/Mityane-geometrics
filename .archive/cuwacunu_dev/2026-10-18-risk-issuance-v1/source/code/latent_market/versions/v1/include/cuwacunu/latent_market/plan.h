// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/config/document.h"
#include "cuwacunu/latent_market/artifacts.h"
#include "cuwacunu/latent_market/timeline.h"
#include <array>
#include <set>

namespace cuwacunu::latent_market::v1 {
struct PreparationPlan {
  std::string experiment_id;
  std::filesystem::path file,card,manifest,source_closure,origins,settings,checkpoint;
  std::array<std::filesystem::path,3> sources;
  std::vector<ArtifactPin> pins,source_pins;
  std::vector<RoleInterval> roles;
  PairSpec pairs;
  int64_t encoder_fit_cutoff_ms{},latency_ms{};
  std::string plan_hash,checkpoint_hash,settings_hash,card_hash,origins_hash,manifest_hash,closure_hash;
  std::string encoder_sidecar_sha256,encoder_model_ref;
};
inline PreparationPlan read_plan(const std::filesystem::path& path) {
  PreparationPlan p;p.file=path;p.plan_hash=sha256_file(path);
  const auto doc=config::Document::read(path);doc.reject_unknown_blocks({"LATENT_MARKET_V1"});
  const auto& b=doc.block("LATENT_MARKET_V1");
  b.reject_unknown_keys({"EXPERIMENT_ID","CARD","CARD_SHA256","PINS","PINS_SHA256","SOURCE_CLOSURE","SOURCE_CLOSURE_SHA256",
    "ORIGINS","ORIGINS_SHA256","ENCODER_SETTINGS","ENCODER_SETTINGS_SHA256","ENCODER_CHECKPOINT","ENCODER_CHECKPOINT_SHA256",
    "ENCODER_SIDECAR_SHA256","ENCODER_MODEL_REF",
    "SOURCE_0","SOURCE_1","SOURCE_2","ENCODER_FIT_CUTOFF_MS","AVAILABILITY_LATENCY_MS","HISTORY_STEPS","SPACING_TICKS",
    "HORIZONS","FUTURE_LABEL_HORIZONS","TRAIN_BEGIN_MS","TRAIN_END_MS","VALIDATION_BEGIN_MS","VALIDATION_END_MS"});
  p.experiment_id=b.string("EXPERIMENT_ID");
  if(p.experiment_id.empty())throw std::runtime_error("An exact prospective experiment identity is required");
  const auto pinned=[&](const char* field,const char* hash_field,std::string& hash) {
    auto file=doc.resolve_path(b.string(field));hash=b.string(hash_field);verify_sha256(file,hash);return file;
  };
  p.card=pinned("CARD","CARD_SHA256",p.card_hash);
  p.manifest=pinned("PINS","PINS_SHA256",p.manifest_hash);p.pins=read_pins(p.manifest);
  p.source_closure=pinned("SOURCE_CLOSURE","SOURCE_CLOSURE_SHA256",p.closure_hash);p.source_pins=read_pins(p.source_closure);
  p.origins=pinned("ORIGINS","ORIGINS_SHA256",p.origins_hash);
  p.settings=pinned("ENCODER_SETTINGS","ENCODER_SETTINGS_SHA256",p.settings_hash);
  p.checkpoint=pinned("ENCODER_CHECKPOINT","ENCODER_CHECKPOINT_SHA256",p.checkpoint_hash);
  p.encoder_sidecar_sha256=b.string("ENCODER_SIDECAR_SHA256");
  p.encoder_model_ref=b.string("ENCODER_MODEL_REF");
  verify_sha256(std::filesystem::path(p.checkpoint.string()+".manifest.json"),p.encoder_sidecar_sha256);
  for(size_t c=0;c<3;++c){p.sources[c]=doc.resolve_path(b.string("SOURCE_"+std::to_string(c)));require_pinned(p.pins,p.sources[c]);}
  p.encoder_fit_cutoff_ms=b.integer("ENCODER_FIT_CUTOFF_MS");p.latency_ms=b.integer("AVAILABILITY_LATENCY_MS");
  p.pairs.history_steps=b.integer("HISTORY_STEPS");p.pairs.spacing_ticks=b.integer("SPACING_TICKS");
  p.pairs.horizons_ticks=b.integers("HORIZONS");p.pairs.future_label_ticks=b.integers("FUTURE_LABEL_HORIZONS");
  validate_pair_spec(p.pairs);
  if(p.pairs.future_label_ticks.empty())throw std::runtime_error("An explicit future-risk horizon panel is required");
  p.roles={{Role::train,b.integer("TRAIN_BEGIN_MS"),b.integer("TRAIN_END_MS")},
           {Role::validation,b.integer("VALIDATION_BEGIN_MS"),b.integer("VALIDATION_END_MS")}};
  validate_roles(p.roles);
  if(p.latency_ms<0 || p.encoder_fit_cutoff_ms<0 || p.pairs.history_steps<1 || p.pairs.spacing_ticks<1 ||
     p.roles[0].begin_ms>p.roles[0].end_ms || p.roles[0].end_ms>=p.roles[1].begin_ms || p.roles[1].begin_ms>p.roles[1].end_ms)
    throw std::runtime_error("Ordered disjoint TRAIN/VALIDATION roles and nonnegative latency required");
  return p;
}
inline std::vector<PairOrigin> read_origins(const std::filesystem::path& path) {
  std::ifstream in(path);if(!in)throw std::runtime_error("Cannot open exact origin ledger");
  std::vector<PairOrigin> out;std::string line;std::set<std::string> ids;
  while(std::getline(in,line)) {
    if(line.empty() || line[0]=='#')continue;
    std::istringstream row(line);PairOrigin p;std::string role,extra;
    if(!(row>>std::quoted(p.pair_id)>>p.endpoint_ms>>p.decision_ms>>role>>std::quoted(p.dependence_group)) || (row>>extra))
      throw std::runtime_error("Origin row requires pair_id endpoint_ms decision_ms train|validation dependence_group");
    if(role=="train")p.role=Role::train;else if(role=="validation")p.role=Role::validation;
    else throw std::runtime_error("Preparation admits declared development roles only; TEST needs a separate confirmation protocol");
    if(!ids.insert(p.pair_id).second || p.dependence_group.empty())throw std::runtime_error("Unique pair IDs and declared temporal blocks required");
    out.push_back(p);
  }
  if(!in.eof() || out.empty())throw std::runtime_error("Empty/unreadable origin ledger");
  return out;
}
inline void verify_plan(const PreparationPlan& p) {
  verify_sha256(p.file,p.plan_hash);verify_sha256(p.card,p.card_hash);verify_sha256(p.origins,p.origins_hash);
  verify_sha256(p.settings,p.settings_hash);verify_sha256(p.checkpoint,p.checkpoint_hash);
  verify_sha256(std::filesystem::path(p.checkpoint.string()+".manifest.json"),p.encoder_sidecar_sha256);
  verify_sha256(p.manifest,p.manifest_hash);verify_sha256(p.source_closure,p.closure_hash);
  verify_pins(p.pins);verify_pins(p.source_pins);
}
} // namespace cuwacunu::latent_market::v1
