// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/latent_market/forecast_issuance/v1/publication.h"
#include "cuwacunu/networks/direct_embedding_control/v1/artifact.h"

namespace cuwacunu::latent_market::risk_issuance::v1 {
namespace fi=cuwacunu::latent_market::forecast_issuance::v1;
namespace lm=cuwacunu::latent_market::v1;
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace dc=cuwacunu::networks::direct_embedding_control::v1;
namespace da=dc::artifact;
namespace av=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;
inline std::string canonical_pairs_sha256(const lm::DynamicsPairs& pairs) {
  std::ostringstream stream(std::ios::out|std::ios::binary);lm::write_pairs(stream,pairs);
  TORCH_CHECK(bool(stream),"[risk native control] cannot authenticate original pairs");return av::sha256_bytes(stream.str());
}
struct NativeIssued {torch::Tensor values;bool valid{false};int64_t reasons{0};};
class NativeForecast {
  ev::PredictionSet predictions_;
  std::vector<lm::DynamicsPair> originals_;
  std::string contract_,timeline_,pairs_,encoder_,model_,sidecar_,comparison_,output_,identity_;
  int64_t fit_{};
  std::vector<int64_t> horizons_;
  NativeForecast()=default;
  static ev::PredictionSet clone(const ev::PredictionSet& p) {
    auto result=p;result.values=p.values.detach().clone();result.valid=p.valid.detach().clone();result.reasons=p.reasons.detach().clone();return result;
  }
 public:
  NativeForecast(const NativeForecast&)=delete;
  NativeForecast& operator=(const NativeForecast&)=delete;
  NativeForecast(NativeForecast&&)=default;
  NativeForecast& operator=(NativeForecast&&)=default;
  static NativeForecast load(const fi::FrozenContract& c,const lm::EmbeddingTimeline& timeline,const lm::DynamicsPairs& pairs,
      const fs::path& archive,const std::string& sidecar_pin,const std::string& model_pin,
      const dc::Config& config,const dc::Identity& id,const da::Context& context,
      const fs::path& comparison,const std::string& comparison_pin,const std::string& role,const fs::path& output) {
    fi::validate_contract(c);lm::validate_pairs(timeline,pairs);
    TORCH_CHECK(fi::timeline_digest(timeline)==c.timeline_sha256&&canonical_pairs_sha256(pairs)==c.pairs_sha256,
      "[risk native control] original capture/pair bytes differ from frozen tuple");
    TORCH_CHECK(context.encoder==c.encoder&&context.encoder_archive_sha256==c.encoder_archive_sha256&&
      id.encoder_model_ref==c.encoder.model_ref,"[risk native control] wrong Encoder parent");
    TORCH_CHECK(config.fit_cutoff_ms>=c.fits.encoder_ms&&config.fit_cutoff_ms<=c.fits.online_ms&&
      config.width==static_cast<int64_t>(c.embedding_width)&&config.history_size==c.history_steps&&
      config.spacing_ticks==c.spacing_ticks&&config.horizons==c.horizons_ticks&&context.cadence_ms==c.interval_ms&&
      context.channel_ids==c.timeline.channel_ids&&context.feature_names==c.timeline.feature_names&&
      context.embedding_export_layout==c.timeline.export_layout&&context.availability_policy==c.timeline.availability_policy,
      "[risk native control] control geometry, semantic contract or fit stage differs");
    const auto native=c.timeline.artifact_id+":native-e";
    TORCH_CHECK(id.coordinate_id==native&&id.source==c.timeline_sha256&&id.pair_ledger==c.pairs_sha256&&
      context.coordinates_sha256==av::sha256_bytes(native),"[risk native control] wrong native coordinates/source/pairs");
    const auto checked=da::load_inference(archive,sidecar_pin,model_pin,config,id,context);
    const auto selection=da::read_comparison(comparison,comparison_pin,context.protocol_sha256,context.population_sha256,
      context.coordinates_sha256,c.encoder.model_ref);
    auto values=da::load_comparison_predictions(selection,role,checked,output);
    const auto recomputed=checked.predict(lm::array_bank(timeline),lm::associations(timeline,pairs));
    TORCH_CHECK(torch::equal(values.values,recomputed.values)&&torch::equal(values.valid,recomputed.valid)&&
      torch::equal(values.reasons,recomputed.reasons),"[risk native control] published values do not come from selected frozen control");
    NativeForecast result;result.predictions_=clone(values);result.originals_=pairs.rows;
    result.contract_=fi::contract_digest(c);result.timeline_=c.timeline_sha256;result.pairs_=c.pairs_sha256;
    result.encoder_=c.encoder.model_ref;result.model_=checked.component().model_ref;result.sidecar_=checked.sidecar_sha256();
    result.comparison_=comparison_pin;result.output_=da::selected_comparison(selection,role,checked).output_sha256;result.fit_=config.fit_cutoff_ms;
    result.horizons_=c.horizons_ticks;
    result.identity_=av::digest_fields("risk-issued-native-control-v1",{result.contract_,result.timeline_,result.pairs_,
      result.encoder_,result.model_,result.sidecar_,result.comparison_,result.output_,std::to_string(result.fit_),native});
    result.verify(c,timeline,pairs);return result;
  }
  void verify(const fi::FrozenContract& c,const lm::EmbeddingTimeline& timeline,const lm::DynamicsPairs& pairs)const {
    fi::validate_contract(c);lm::validate_pairs(timeline,pairs);
    TORCH_CHECK(contract_==fi::contract_digest(c)&&timeline_==fi::timeline_digest(timeline)&&timeline_==c.timeline_sha256&&
      pairs_==canonical_pairs_sha256(pairs)&&pairs_==c.pairs_sha256&&encoder_==c.encoder.model_ref&&
      predictions_.coordinate_id==c.timeline.artifact_id+":native-e"&&fit_==predictions_.fit_cutoff_ms&&
      fit_>=c.fits.encoder_ms&&fit_<=c.fits.online_ms&&horizons_==c.horizons_ticks,
      "[risk native control] frozen capture/coordinate/parent/fit selection changed");
    const auto expected=lm::associations(timeline,pairs);
    TORCH_CHECK(expected.size()==predictions_.associations.size()&&pairs.rows.size()==originals_.size()&&
      predictions_.values.sizes()==torch::IntArrayRef({static_cast<int64_t>(expected.size()),static_cast<int64_t>(c.embedding_width)})&&
      predictions_.valid.sizes()==torch::IntArrayRef({static_cast<int64_t>(expected.size())})&&
      predictions_.reasons.sizes()==torch::IntArrayRef({static_cast<int64_t>(expected.size())}),
      "[risk native control] complete original output population required");
    for(size_t i=0;i<expected.size();++i) {
      TORCH_CHECK(fi::publication_detail::same_association(expected[i],predictions_.associations[i]),
        "[risk native control] prediction association differs from original capture");
      const auto& origin=pairs.rows[i/c.horizons_ticks.size()];
      if(predictions_.valid[i].item<bool>())TORCH_CHECK(fit_<origin.endpoint_ms&&
        predictions_.reasons[i].item<int64_t>()==0&&torch::isfinite(predictions_.values[i]).all().item<bool>(),
        "[risk native control] retrospective or invalid issued control row");
    }
  }
  ev::PredictionSet snapshot()const{return clone(predictions_);}
  NativeIssued at(const lm::PairOrigin& origin)const {
    const auto found=std::find_if(originals_.begin(),originals_.end(),[&](const auto& row){return row.pair_id==origin.pair_id;});
    TORCH_CHECK(found!=originals_.end()&&found->endpoint_ms==origin.endpoint_ms&&found->decision_ms==origin.decision_ms&&
      found->role==origin.role&&found->dependence_group==origin.dependence_group,"[risk native control] wrong original origin");
    const auto index=static_cast<int64_t>(found-originals_.begin())*static_cast<int64_t>(horizons_.size());
    NativeIssued result;result.values=predictions_.values.narrow(0,index,horizons_.size()).detach().clone();result.valid=true;
    for(size_t q=0;q<horizons_.size();++q){result.valid=result.valid&&predictions_.valid[index+q].item<bool>();result.reasons|=predictions_.reasons[index+q].item<int64_t>();}
    return result;
  }
  const std::string& identity()const{return identity_;}
  const std::string& encoder_model_ref()const{return encoder_;}
  const std::string& timeline_sha256()const{return timeline_;}
  const std::string& pairs_sha256()const{return pairs_;}
  const std::string& coordinate_id()const{return predictions_.coordinate_id;}
  const std::string& model_ref()const{return model_;}
  const std::string& sidecar_sha256()const{return sidecar_;}
  const std::string& comparison_sha256()const{return comparison_;}
  const std::string& output_sha256()const{return output_;}
  const std::vector<int64_t>& horizons_ticks()const{return horizons_;}
  int64_t fit_cutoff_ms()const{return fit_;}
};
}
