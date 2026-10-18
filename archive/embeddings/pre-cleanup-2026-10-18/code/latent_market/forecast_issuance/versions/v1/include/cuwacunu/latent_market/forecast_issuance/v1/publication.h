// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "issuance.h"
#include "cuwacunu/latent_market/arrays.h"
#include "cuwacunu/latent_market/ledger_io.h"
#include <linux/memfd.h>
#include <sys/syscall.h>

namespace cuwacunu::latent_market::forecast_issuance::v1 {
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace fs=std::filesystem;
namespace publication_detail {
using Fields=std::map<std::string,std::string>;
inline Fields fields(const FrozenContract& c) {
  return {{"schema","forecast-issuance-publication-v1"},{"contract_sha256",contract_digest(c)},
    {"policy_sha256",c.policy_sha256},{"stage_kind",c.stage_kind},{"timeline_sha256",c.timeline_sha256},
    {"pairs_sha256",c.pairs_sha256},{"bundle_ref",c.bundle_ref},{"target_coordinate_id",c.target_coordinate_id},
    {"encoder_archive_sha256",c.encoder_archive_sha256},{"encoder_sidecar_sha256",c.encoder_sidecar_sha256},
    {"predictive_archive_sha256",c.predictive_archive_sha256},{"predictive_sidecar_sha256",c.predictive_sidecar_sha256},
    {"encoder_model_ref",c.encoder.model_ref},{"online_model_ref",c.online.model_ref},
    {"target_model_ref",c.target.model_ref},{"dynamics_model_ref",c.dynamics.model_ref},
    {"embedding_transform_sha256",c.embedding_transform_sha256},{"reference_scale_sha256",c.reference_scale_sha256},
    {"encoder_fit_ms",std::to_string(c.fits.encoder_ms)},{"online_fit_ms",std::to_string(c.fits.online_ms)},
    {"target_fit_ms",std::to_string(c.fits.target_ms)},{"dynamics_fit_ms",std::to_string(c.fits.dynamics_ms)},
    {"history_steps",std::to_string(c.history_steps)},{"spacing_ticks",std::to_string(c.spacing_ticks)},
    {"state_width",std::to_string(c.state_width)}};
}
inline std::string serialize(const Fields& fields) {
  std::string out="{";bool first=true;
  for(const auto& [key,value]:fields){if(!first)out+=',';first=false;out+=av::detail::quote(key)+":"+av::detail::quote(value);}
  return out+"}\n";
}
inline std::string csv(const std::string& value) {
  std::string out="\"";for(char ch:value){if(ch=='\"')out+='\"';out+=ch;}return out+'\"';
}
inline std::string ledger(const FrozenContract& c,const lm::DynamicsPairs& pairs,
                          const std::vector<IssuedOrigin>& issued) {
  detail::require(issued.size()==pairs.rows.size(),"publication must retain every original origin");
  std::ostringstream out;
  out<<"pair_id,origin_ms,decision_ms,horizon_ticks,valid,issuance_reasons,original_input_reasons,target_eligible,target_ledger_reasons,encoder_fit_ms,online_fit_ms,target_fit_ms,dynamics_fit_ms,target_coordinate_id,bundle_ref,input_endpoints_ms,input_available_ms,input_window_ids,input_reasons\n";
  for(size_t n=0;n<issued.size();++n) {
    const auto& row=issued[n];const auto& original=pairs.rows[n];
    detail::require(row.origin.pair_id==original.pair_id && row.origin.endpoint_ms==original.endpoint_ms &&
      row.origin.decision_ms==original.decision_ms && row.origin.role==original.role &&
      row.origin.dependence_group==original.dependence_group && row.original_input_reasons==original.input_reasons &&
      original.futures.size()==c.horizons_ticks.size(),"original publication association mismatch");
    std::string endpoints,available,ids,reasons;
    for(const auto& input:row.history) {
      if(!endpoints.empty()){endpoints+=';';available+=';';ids+=';';reasons+=';';}
      endpoints+=std::to_string(input.endpoint_ms);available+=std::to_string(input.available_ms);ids+=input.window_id;
      reasons+=std::to_string(input.reasons);
    }
    for(size_t q=0;q<original.futures.size();++q) {
      const auto& future=original.futures[q];
      detail::require(future.horizon_ticks==c.horizons_ticks[q],"publication horizon ordering differs");
      // Target fields are copied after issuance, solely for separate scoring.
      out<<csv(row.origin.pair_id)<<','<<row.origin.endpoint_ms<<','<<row.origin.decision_ms<<','<<future.horizon_ticks<<','
        <<row.valid<<','<<row.reasons<<','<<row.original_input_reasons<<','<<future.eligible<<','<<future.reasons<<','
        <<c.fits.encoder_ms<<','<<c.fits.online_ms<<','<<c.fits.target_ms<<','<<c.fits.dynamics_ms<<','
        <<csv(c.target_coordinate_id)<<','<<csv(c.bundle_ref)<<','<<csv(endpoints)<<','<<csv(available)<<','<<csv(ids)<<','<<csv(reasons)<<'\n';
    }
  }
  return out.str();
}
inline void validate_prediction(const FrozenContract& c,const ev::PredictionSet& p,size_t count) {
  detail::require(p.coordinate_id==c.target_coordinate_id && p.fit_cutoff_ms==c.fits.dynamics_ms &&
    p.history_size==c.history_steps && p.spacing_ticks==c.spacing_ticks && p.method=="state-causal-forecast-issuance-v1" &&
    p.values.dim()==2 && p.values.size(0)==static_cast<int64_t>(count) &&
    p.values.size(1)==static_cast<int64_t>(c.state_width) && p.associations.size()==count,
    "prediction metadata differs from the exact frozen issuance contract");
  detail::require(torch::isfinite(p.values).all().item<bool>(),"publication predictions must be finite");
}
inline ev::PredictionSet decode(const std::string& bytes) {
  const int fd=static_cast<int>(::syscall(SYS_memfd_create,"forecast-issuance",MFD_CLOEXEC|MFD_ALLOW_SEALING));
  detail::require(fd>=0,"cannot create checked issuance memory file");
  try {
    size_t offset=0;while(offset<bytes.size()) {
      const auto written=::write(fd,bytes.data()+offset,bytes.size()-offset);
      if(written<0 && errno==EINTR)continue;
      detail::require(written>0,"cannot write checked issuance memory file");offset+=static_cast<size_t>(written);
    }
    detail::require(::fcntl(fd,F_ADD_SEALS,F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL)==0,"cannot seal checked issuance bytes");
    auto result=ev::load_predictions("/proc/self/fd/"+std::to_string(fd));::close(fd);return result;
  } catch(...) {::close(fd);throw;}
}
inline bool same_association(const ev::Association& a,const ev::Association& b) {
  return a.pair_id==b.pair_id && a.origin_row_id==b.origin_row_id && a.target_row_id==b.target_row_id &&
    a.horizon_ticks==b.horizon_ticks && a.decision_ms==b.decision_ms && a.history_row_ids==b.history_row_ids &&
    a.input_eligible==b.input_eligible && a.target_eligible==b.target_eligible &&
    a.ledger_input_reasons==b.ledger_input_reasons && a.ledger_target_reasons==b.ledger_target_reasons &&
    a.sample_weight==b.sample_weight && a.actual_overlap_fraction==b.actual_overlap_fraction && a.dependence_group==b.dependence_group;
}
inline ev::Association association(const lm::DynamicsPair& p,const lm::HorizonAssociation& f,size_t offset) {
  ev::Association a;a.pair_id=offset+1;a.origin_row_id=p.history.back().row_index?*p.history.back().row_index+1:-1;
  a.target_row_id=f.target.row_index?*f.target.row_index+1:-1;a.horizon_ticks=f.horizon_ticks;a.decision_ms=p.decision_ms;
  a.input_eligible=p.input_reasons==lm::no_issue;
  a.target_eligible=f.target.reasons==lm::no_issue && !(f.reasons&lm::raw_envelope_role_boundary);
  a.ledger_input_reasons=p.input_reasons;a.ledger_target_reasons=f.target.reasons|(f.reasons&lm::raw_envelope_role_boundary);
  a.actual_overlap_fraction=f.overlap.mask_aware_shared_fraction.value_or(f.overlap.geometric_fraction);
  a.dependence_group=p.dependence_group;
  for(const auto& h:p.history)a.history_row_ids.push_back(h.row_index?*h.row_index+1:-1);
  return a;
}
}

inline int64_t evaluation_reasons(uint64_t reasons) {
  int64_t result=static_cast<int64_t>(reasons<<16);
  if(reasons&15)result|=ev::fit_not_available;
  if(reasons&~uint64_t{15})result|=ev::ledger_input_invalid;
  return result;
}

// The manifest pin authenticates both exact prediction bytes and the complete
// origin/horizon ledger. Publish last; partial leaves have no accepted manifest.
inline std::string publish_issuance(const fs::path& directory,const FrozenContract& c,
    const lm::DynamicsPairs& pairs,const std::vector<IssuedOrigin>& issued,const ev::PredictionSet& prediction) {
  auto fields=publication_detail::fields(c);
  const size_t count=issued.size()*c.horizons_ticks.size();
  publication_detail::validate_prediction(c,prediction,count);
  detail::require(issued.size()==pairs.rows.size(),"every original origin is required");
  for(size_t n=0;n<issued.size();++n)for(size_t q=0;q<c.horizons_ticks.size();++q) {
    const size_t offset=n*c.horizons_ticks.size()+q;
    detail::require(!pairs.rows[n].history.empty() && pairs.rows[n].futures.size()==c.horizons_ticks.size(),"complete original association geometry required");
    detail::require(prediction.valid[offset].item<bool>()==issued[n].valid &&
      prediction.reasons[offset].item<int64_t>()==evaluation_reasons(issued[n].reasons) &&
      publication_detail::same_association(prediction.associations[offset],
        publication_detail::association(pairs.rows[n],pairs.rows[n].futures[q],offset)) &&
      torch::equal(prediction.values[offset],issued[n].values[q]),"prediction differs from issued origin/horizon");
  }
  const auto ledger=publication_detail::ledger(c,pairs,issued);
  // Own immutable prediction publication, then verify the actual saved bytes
  // against the issued values and every retained scoring association.
  ev::save_predictions((directory/"state-predictions.pt").string(),prediction);
  const auto prediction_bytes=av::detail::read_bytes(directory/"state-predictions.pt",uintmax_t{2}<<30);
  const auto persisted=publication_detail::decode(prediction_bytes);
  publication_detail::validate_prediction(c,persisted,count);
  detail::require(torch::equal(persisted.values,prediction.values) && torch::equal(persisted.valid,prediction.valid) &&
    torch::equal(persisted.reasons,prediction.reasons),"saved prediction tensors differ from issued values");
  for(size_t i=0;i<count;++i)detail::require(publication_detail::same_association(persisted.associations[i],prediction.associations[i]),
    "saved prediction association differs from original ledger");
  lm::ledger_detail::publish(directory/"state-issuance.csv",[&](auto& stream){stream<<ledger;});
  fields["prediction_sha256"]=av::sha256_bytes(prediction_bytes);
  fields["ledger_sha256"]=av::sha256_bytes(ledger);
  fields["origin_count"]=std::to_string(issued.size());fields["prediction_count"]=std::to_string(count);
  const auto manifest=publication_detail::serialize(fields);
  lm::ledger_detail::publish(directory/"state-issuance.json",[&](auto& stream){stream<<manifest;});
  return av::sha256_bytes(manifest);
}

inline ev::PredictionSet load_issued_predictions(const fs::path& directory,const std::string& manifest_pin,
                                                const FrozenContract& c) {
  detail::require(av::is_sha256(manifest_pin),"a trusted issuance manifest pin is required");
  const auto manifest=av::detail::read_bytes(directory/"state-issuance.json",262144);
  detail::require(av::sha256_bytes(manifest)==manifest_pin,"issuance manifest digest mismatch");
  const auto root=av::detail::JsonReader(manifest).read();
  auto expected=publication_detail::fields(c);
  detail::require(root.kind==av::detail::Json::Kind::object && root.object.size()==expected.size()+4,
                  "unexpected issuance manifest fields");
  for(const auto& [key,value]:expected)
    detail::require(root.object.contains(key) && av::detail::str(root.object.at(key))==value,
                    "issuance contract field differs: "+key);
  for(const auto* key:{"prediction_sha256","ledger_sha256","origin_count","prediction_count"})
    detail::require(root.object.contains(key),"missing issuance publication field");
  auto value=[&](const char* key){return av::detail::str(root.object.at(key));};
  auto integer=[&](const char* key){const auto text=value(key);size_t end=0;
    const auto result=std::stoull(text,&end);detail::require(end==text.size()&&text==std::to_string(result),"noncanonical publication count");return result;};
  const auto origins=integer("origin_count"),count=integer("prediction_count");
  detail::require(count/c.horizons_ticks.size()==origins && count%c.horizons_ticks.size()==0,"publication counts differ");
  const auto ledger=av::detail::read_bytes(directory/"state-issuance.csv",uintmax_t{1}<<30);
  detail::require(av::sha256_bytes(ledger)==value("ledger_sha256"),"issuance ledger digest mismatch");
  const auto bytes=av::detail::read_bytes(directory/"state-predictions.pt",uintmax_t{2}<<30);
  detail::require(av::sha256_bytes(bytes)==value("prediction_sha256"),"issued prediction digest mismatch");
  // Deserialize the exact authenticated bytes, never reopen a mutable pathname.
  auto result=publication_detail::decode(bytes);
  publication_detail::validate_prediction(c,result,count);return result;
}
}
