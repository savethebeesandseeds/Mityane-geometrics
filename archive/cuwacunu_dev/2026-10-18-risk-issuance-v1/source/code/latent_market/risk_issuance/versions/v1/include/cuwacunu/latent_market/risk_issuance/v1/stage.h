// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "native_control.h"
#include "source_identity.h"
#include "cuwacunu/evaluation/latent_dynamics/v1/risk_readouts.h"

namespace cuwacunu::latent_market::risk_issuance::v1 {
namespace pa=cuwacunu::networks::predictive_training::v1::artifact;
inline constexpr const char* policy_sha256="a7f52e1bc77f613041f56c8ce1ce7bfded42046dd4a9d2c31dfe3c7c2277220f";
namespace stage_detail {
inline void require(bool value,const std::string& message){TORCH_CHECK(value,"[risk issuance v1] ",message);}
inline void shape(const torch::Tensor& value,torch::ScalarType dtype,torch::IntArrayRef sizes,const char* message) {
  require(value.defined()&&value.device().is_cpu()&&value.scalar_type()==dtype&&value.sizes()==sizes,message);
}
inline ev::ObservableOrigins clone(const ev::ObservableOrigins& value) {
  return {value.row_ids.detach().clone(),value.roles.detach().clone(),value.decision_ms.detach().clone()};
}
inline ev::FeatureArm clone(const ev::FeatureArm& value) {
  auto out=value;out.values=value.values.detach().clone();out.valid=value.valid.detach().clone();
  out.origin_row_ids=value.origin_row_ids.detach().clone();out.available_ms=value.available_ms.detach().clone();
  out.support_begin_ms=value.support_begin_ms.detach().clone();out.support_end_ms=value.support_end_ms.detach().clone();return out;
}
inline ev::RiskReadoutBundle clone(const ev::RiskReadoutBundle& value) {
  auto out=value;for(auto& arm:out.arms)for(auto& head:arm.heads){head.input_mean=head.input_mean.detach().clone();
    head.input_scale=head.input_scale.detach().clone();head.weights=head.weights.detach().clone();}return out;
}
}
struct StageContract {
  std::string policy_sha256{v1::policy_sha256};
  int64_t train_begin_ms{},train_end_ms{},head_fit_ms{},evaluation_begin_ms{},evaluation_end_ms{},scoring_cutoff_ms{};
};
inline void validate_stage(const StageContract& s) {
  stage_detail::require(s.policy_sha256==policy_sha256&&s.train_begin_ms>=0&&s.train_begin_ms<=s.train_end_ms&&
    s.train_end_ms<=s.head_fit_ms&&s.head_fit_ms<s.evaluation_begin_ms&&s.evaluation_begin_ms<=s.evaluation_end_ms&&
    s.evaluation_end_ms<=s.scoring_cutoff_ms,"unregistered or nonchronological head stage");
}
inline std::string stage_digest(const StageContract& s) {
  validate_stage(s);return av::digest_fields("risk-head-stage-v1",{s.policy_sha256,std::to_string(s.train_begin_ms),
    std::to_string(s.train_end_ms),std::to_string(s.head_fit_ms),std::to_string(s.evaluation_begin_ms),
    std::to_string(s.evaluation_end_ms),std::to_string(s.scoring_cutoff_ms)});
}
enum class HeadRole:int64_t{excluded=0,train=1,evaluation=2};
enum StageReason:uint64_t {
  no_stage_issue=0,outside_head_stage=1ull<<0,input_envelope_outside_stage=1ull<<1,
  upstream_unavailable=1ull<<2,native_unavailable=1ull<<3,producer_not_before_origin=1ull<<4,
  common_input_invalid=1ull<<5,not_head_train=1ull<<6,train_decision_after_fit=1ull<<7,
  invalid_label=1ull<<8,label_not_mature=1ull<<9,label_support_outside_train=1ull<<10,
  noncausal_label_metadata=1ull<<11,not_head_evaluation=1ull<<12,head_not_before_origin=1ull<<13,
  feature_not_available_at_head_fit=1ull<<14
};
struct BankRow {
  int64_t row_id{};std::string pair_id,dependence_group;
  int64_t endpoint_ms{},decision_ms{},input_begin_ms{},input_available_ms{};
  lm::Role capture_role{lm::Role::unassigned};HeadRole head_role{HeadRole::excluded};
  uint64_t reasons{},upstream_reasons{},native_reasons{};bool common_eligible{false};
};
struct TrainingSelection {
  std::string target_identity,selected_identity;
  std::vector<std::string> names,units;
  std::vector<int64_t> common_training_rows;
  std::vector<std::vector<uint64_t>> reasons;
  std::vector<std::vector<int64_t>> selected_origin_ids;
  std::vector<std::vector<int64_t>> label_available_ms,label_support_begin_ms,label_support_end_ms;
};
class FittedStage;
class HeadPublication;
class FeatureBank {
  StageContract stage_;fi::FrozenContract upstream_;
  std::string identity_,issuance_pin_,native_identity_,native_model_,native_sidecar_,native_comparison_,native_output_;
  int64_t native_fit_{};
  std::vector<BankRow> rows_;ev::ObservableOrigins origins_;std::vector<ev::FeatureArm> arms_;
  FeatureBank()=default;
  friend class FittedStage;
 public:
  static FeatureBank build(const StageContract& stage,const lm::EmbeddingTimeline& timeline,const lm::DynamicsPairs& pairs,
      const fi::Issuer& issuer,std::shared_ptr<const pa::CheckedInference> serving,const fs::path& issuance_directory,
      const std::string& issuance_manifest_pin,const NativeForecast& native) {
    using stage_detail::require;verify_runtime_source();validate_stage(stage);const auto& c=issuer.contract();fi::validate_contract(c);
    require(bool(serving),"checked immutable state serving handle required");
    require(fi::timeline_digest(timeline)==c.timeline_sha256&&canonical_pairs_sha256(pairs)==c.pairs_sha256,
      "original captured source/pair bytes differ from upstream contract");lm::validate_pairs(timeline,pairs);
    require(c.history_steps==4&&!pairs.rows.empty(),"engineering panel requires exact K4 history and origins");
    require(stage.train_begin_ms>std::max({c.fits.encoder_ms,c.fits.online_ms,c.fits.target_ms,c.fits.dynamics_ms,native.fit_cutoff_ms()}),
      "every upstream producer must fit strictly before head TRAIN begins");
    require(serving->components().size()==3,"complete checked serving tuple required");
    for(const auto* component:{&c.online,&c.target,&c.dynamics})
      require(std::find(serving->components().begin(),serving->components().end(),*component)!=serving->components().end(),
        "wrong checked state/target/Dynamics parent tuple");
    native.verify(c,timeline,pairs);const auto published=fi::load_issued_predictions(issuance_directory,issuance_manifest_pin,c);
    require(published.method=="state-causal-forecast-issuance-v1","retrospective forecast arrays are not admitted");
    FeatureBank bank;bank.stage_=stage;bank.upstream_=c;bank.issuance_pin_=issuance_manifest_pin;
    bank.native_identity_=native.identity();bank.native_model_=native.model_ref();bank.native_sidecar_=native.sidecar_sha256();
    bank.native_comparison_=native.comparison_sha256();bank.native_output_=native.output_sha256();bank.native_fit_=native.fit_cutoff_ms();
    bank.identity_=av::digest_fields("risk-private-feature-bank-v1",{stage_digest(stage),fi::contract_digest(c),issuance_manifest_pin,
      bank.native_identity_,bank.native_model_,bank.native_sidecar_,bank.native_comparison_,bank.native_output_,std::to_string(bank.native_fit_)});
    const auto count=static_cast<int64_t>(pairs.rows.size()),e=static_cast<int64_t>(c.embedding_width),z=static_cast<int64_t>(c.state_width);
    const auto q=static_cast<int64_t>(c.horizons_ticks.size());
    bank.origins_={torch::arange(1,count+1,torch::kInt64),torch::full({count},static_cast<int64_t>(ev::Role::validation),torch::kInt64),torch::zeros({count},torch::kInt64)};
    const std::vector<std::string> names{"current_e","history_e","current_z","history_z","e_history+forecast_e","e_history+forecast_z","z_history+forecast_z"};
    const std::vector<int64_t> widths{e,4*e,z,4*z,4*e+q*e,4*e+q*z,4*z+q*z};
    for(size_t k=0;k<names.size();++k) {
      ev::FeatureArm arm;arm.name=names[k];arm.identity=av::digest_fields("risk-frozen-feature-arm-v1",{bank.identity_,names[k]});
      arm.information_contract="issued-only;frozen-native/online/target-coordinates;K4;exact-original-origin-bank;"+names[k];
      arm.artifact_fit_cutoff_ms=std::max({c.fits.encoder_ms,c.fits.online_ms,c.fits.target_ms,c.fits.dynamics_ms,bank.native_fit_});
      arm.values=torch::zeros({count,widths[k]},torch::kFloat64);arm.valid=torch::zeros({count},torch::kBool);
      arm.origin_row_ids=bank.origins_.row_ids.detach().clone();arm.available_ms=torch::zeros({count},torch::kInt64);
      arm.support_begin_ms=torch::zeros({count},torch::kInt64);arm.support_end_ms=torch::zeros({count},torch::kInt64);bank.arms_.push_back(std::move(arm));
    }
    const auto options=serving->fold_forecast().weight.options();
    for(size_t n=0;n<pairs.rows.size();++n) {
      const auto& p=pairs.rows[n];const lm::PairOrigin origin{p.pair_id,p.endpoint_ms,p.decision_ms,p.role,p.dependence_group};
      const auto causal=issuer.issue(timeline,origin,pairs.roles,p.input_reasons);const auto direct=native.at(origin);
      for(size_t h=0;h<c.horizons_ticks.size();++h) {
        const auto offset=n*c.horizons_ticks.size()+h;
        require(fi::publication_detail::same_association(published.associations.at(offset),fi::publication_detail::association(p,p.futures.at(h),offset))&&
          published.valid[static_cast<int64_t>(offset)].item<bool>()==causal.valid&&
          published.reasons[static_cast<int64_t>(offset)].item<int64_t>()==fi::evaluation_reasons(causal.reasons)&&
          torch::equal(published.values[static_cast<int64_t>(offset)],causal.values[static_cast<int64_t>(h)]),
          "authenticated published forecast differs from exact origin-only issuance");
      }
      BankRow row;row.row_id=static_cast<int64_t>(n)+1;row.pair_id=p.pair_id;row.dependence_group=p.dependence_group;
      row.endpoint_ms=p.endpoint_ms;row.decision_ms=p.decision_ms;row.capture_role=p.role;row.upstream_reasons=causal.reasons;
      row.native_reasons=static_cast<uint64_t>(direct.reasons);row.input_begin_ms=p.raw_begin_ms;row.input_available_ms=p.endpoint_ms;
      if(p.endpoint_ms>=stage.train_begin_ms&&p.endpoint_ms<=stage.train_end_ms)row.head_role=HeadRole::train;
      else if(p.endpoint_ms>=stage.evaluation_begin_ms&&p.endpoint_ms<=stage.evaluation_end_ms)row.head_role=HeadRole::evaluation;
      else row.reasons|=outside_head_stage;
      const auto begin=row.head_role==HeadRole::train?stage.train_begin_ms:stage.evaluation_begin_ms;
      const auto end=row.head_role==HeadRole::train?stage.train_end_ms:stage.evaluation_end_ms;
      if(row.head_role!=HeadRole::excluded&&(p.raw_begin_ms<begin||p.endpoint_ms>end))row.reasons|=input_envelope_outside_stage;
      for(const auto& input:causal.history)row.input_available_ms=std::max(row.input_available_ms,input.available_ms);
      if(!causal.valid)row.reasons|=upstream_unavailable;
      if(!direct.valid)row.reasons|=native_unavailable;
      if(std::max({c.fits.encoder_ms,c.fits.online_ms,c.fits.target_ms,c.fits.dynamics_ms,bank.native_fit_})>=p.endpoint_ms)
        row.reasons|=producer_not_before_origin;
      row.common_eligible=row.reasons==0;
      bank.origins_.decision_ms[static_cast<int64_t>(n)]=p.decision_ms;
      if(row.head_role==HeadRole::train)bank.origins_.roles[static_cast<int64_t>(n)]=static_cast<int64_t>(ev::Role::train);
      for(auto& arm:bank.arms_) {
        arm.valid[static_cast<int64_t>(n)]=row.common_eligible;arm.available_ms[static_cast<int64_t>(n)]=row.input_available_ms;
        arm.support_begin_ms[static_cast<int64_t>(n)]=row.input_begin_ms;arm.support_end_ms[static_cast<int64_t>(n)]=p.endpoint_ms;
      }
      if(row.common_eligible) {
        std::vector<double> flattened;
        for(const auto& input:causal.history){require(input.row_index.has_value(),"issued input disappeared");const auto& values=timeline.rows[*input.row_index].values;flattened.insert(flattened.end(),values.begin(),values.end());}
        const auto eh=torch::tensor(flattened,torch::kFloat64).reshape({4,e});
        const auto zh=serving->state_history(eh.to(options)).detach().to(torch::kCPU,torch::kFloat64).clone();
        require(zh.sizes()==torch::IntArrayRef({4,z})&&torch::isfinite(zh).all().item<bool>(),"checked online-state history invalid");
        stage_detail::shape(direct.values,torch::kFloat64,{q,e},"checked native forecast geometry differs");
        require(torch::isfinite(direct.values).all().item<bool>(),"nonfinite checked native forecast");
        const std::vector<torch::Tensor> features{eh[3],eh.reshape({-1}),zh[3],zh.reshape({-1}),
          torch::cat({eh.reshape({-1}),direct.values.reshape({-1})}),torch::cat({eh.reshape({-1}),causal.values.reshape({-1})}),
          torch::cat({zh.reshape({-1}),causal.values.reshape({-1})})};
        for(size_t k=0;k<features.size();++k)bank.arms_[k].values[static_cast<int64_t>(n)].copy_(features[k]);
      }
      bank.rows_.push_back(row);
    }
    return bank;
  }
  const StageContract& contract()const{return stage_;}
  const std::string& identity()const{return identity_;}
  std::string upstream_contract_sha256()const{return fi::contract_digest(upstream_);}
  fi::FrozenContract upstream_snapshot()const{return upstream_;}
  const std::string& issuance_manifest_sha256()const{return issuance_pin_;}
  const std::string& native_identity()const{return native_identity_;}
  const std::string& native_model_ref()const{return native_model_;}
  const std::string& native_sidecar_sha256()const{return native_sidecar_;}
  const std::string& native_comparison_sha256()const{return native_comparison_;}
  const std::string& native_output_sha256()const{return native_output_;}
  int64_t native_fit_cutoff_ms()const{return native_fit_;}
  std::vector<BankRow> rows_snapshot()const{return rows_;}
  ev::ObservableOrigins origins_snapshot()const{return stage_detail::clone(origins_);}
  std::vector<ev::FeatureArm> arms_snapshot()const{std::vector<ev::FeatureArm> result;for(const auto& arm:arms_)result.push_back(stage_detail::clone(arm));return result;}
};

class FittedStage {
  StageContract stage_;std::string bank_identity_;ev::RiskReadoutBundle model_;TrainingSelection selection_;
  FittedStage()=default;friend class HeadPublication;
  static FittedStage from_checked_snapshot(const FeatureBank& bank,const ev::RiskReadoutBundle& model,const TrainingSelection& selection) {
    using stage_detail::require;const auto& s=bank.stage_;validate_stage(s);
    require(model.recipe.fit_cutoff_ms==s.head_fit_ms&&model.recipe.train_begin_ms==s.train_begin_ms&&model.recipe.train_end_ms==s.train_end_ms&&
      model.recipe.ridge_penalty==1.&&model.recipe.std_floor==1e-8&&model.recipe.denominator_epsilon==1e-12,"head recipe differs from registered fixed numerical recipe");
    require(!model.target_identity.empty()&&model.target_identity==selection.target_identity&&model.names==selection.names&&model.units==selection.units&&
      !model.names.empty()&&model.names.size()==model.units.size()&&model.common_training_rows==selection.common_training_rows&&
      model.common_training_rows.size()==model.names.size()&&model.arms.size()==bank.arms_.size()&&
      selection.reasons.size()==bank.rows_.size()&&selection.selected_origin_ids.size()==model.names.size()&&av::is_sha256(selection.selected_identity),
      "restored head target/population identity differs");
    require(selection.label_available_ms.size()==bank.rows_.size()&&selection.label_support_begin_ms.size()==bank.rows_.size()&&
      selection.label_support_end_ms.size()==bank.rows_.size(),"restored selected-label maturity witness missing");
    bool support=false;
    for(size_t j=0;j<model.names.size();++j) {
      std::vector<int64_t> selected;
      for(size_t i=0;i<bank.rows_.size();++i) {
        require(selection.reasons[i].size()==model.names.size(),"restored selected-label ledger geometry differs");
        require(selection.label_available_ms[i].size()==model.names.size()&&selection.label_support_begin_ms[i].size()==model.names.size()&&
          selection.label_support_end_ms[i].size()==model.names.size(),"restored selected-label maturity geometry differs");
        if(selection.reasons[i][j])continue;
        const auto& row=bank.rows_[i];
        require(row.head_role==HeadRole::train&&row.common_eligible&&row.decision_ms<=s.head_fit_ms&&row.input_available_ms<=s.head_fit_ms,
          "restored selected label has an ineligible head TRAIN origin");selected.push_back(row.row_id);
        require(selection.label_support_begin_ms[i][j]>=row.endpoint_ms&&selection.label_support_begin_ms[i][j]>=s.train_begin_ms&&
          selection.label_support_end_ms[i][j]>=selection.label_support_begin_ms[i][j]&&selection.label_support_end_ms[i][j]<=s.train_end_ms&&
          selection.label_available_ms[i][j]>=selection.label_support_end_ms[i][j]&&selection.label_available_ms[i][j]<=s.head_fit_ms,
          "restored selected label lacks mature complete TRAIN support");
      }
      require(selected==selection.selected_origin_ids[j]&&static_cast<int64_t>(selected.size())==model.common_training_rows[j],"restored common TRAIN counts/identities differ");
      support=support||selected.size()>=2;
    }
    require(support,"no observable has mature common TRAIN support");
    for(size_t k=0;k<model.arms.size();++k) {
      const auto& m=model.arms[k];const auto& arm=bank.arms_[k];
      require(m.name==arm.name&&m.identity==arm.identity&&m.information_contract==arm.information_contract&&
        m.input_width==arm.values.size(1)&&m.artifact_fit_cutoff_ms==arm.artifact_fit_cutoff_ms&&m.heads.size()==model.names.size(),"restored head feature identity differs");
      for(size_t j=0;j<m.heads.size();++j) {
        const auto& h=m.heads[j];require(h.training_rows==model.common_training_rows[j]&&std::isfinite(h.target_mean)&&std::isfinite(h.target_scale)&&h.target_scale>=1e-8,
          "restored target normalization/count invalid");
        for(const auto* t:{&h.input_mean,&h.input_scale,&h.weights}){stage_detail::shape(*t,torch::kFloat64,{m.input_width},"restored head tensor geometry invalid");require(torch::isfinite(*t).all().item<bool>(),"restored head tensor nonfinite");}
        require(h.input_scale.ge(1e-8).all().item<bool>(),"restored input normalization invalid");
      }
    }
    FittedStage out;out.stage_=s;out.bank_identity_=bank.identity_;out.model_=stage_detail::clone(model);out.selection_=selection;return out;
  }
 public:
  static FittedStage fit(const FeatureBank& bank,const ev::ObservableTargets& targets) {
    using stage_detail::require;verify_runtime_source();const auto count=static_cast<int64_t>(bank.rows_.size()),labels=static_cast<int64_t>(targets.names.size());
    require(!targets.identity.empty()&&labels>0&&targets.units.size()==targets.names.size(),"declared observable identity/names/units required");
    require(std::set<std::string>(targets.names.begin(),targets.names.end()).size()==targets.names.size(),"observable names must be unique");
    for(size_t j=0;j<targets.names.size();++j)require(!targets.names[j].empty()&&!targets.units[j].empty(),"observable names and units must be nonempty");
    stage_detail::shape(targets.values,torch::kFloat64,{count,labels},"observable values shape differs");
    stage_detail::shape(targets.valid,torch::kBool,{count,labels},"observable validity shape differs");
    for(const auto* t:{&targets.available_ms,&targets.support_begin_ms,&targets.support_end_ms})stage_detail::shape(*t,torch::kInt64,{count,labels},"observable metadata shape differs");
    stage_detail::shape(targets.origins.row_ids,torch::kInt64,{count},"target origin IDs shape differs");
    stage_detail::shape(targets.origins.decision_ms,torch::kInt64,{count},"target origin decisions shape differs");
    require(torch::equal(targets.origins.row_ids,bank.origins_.row_ids)&&torch::equal(targets.origins.decision_ms,bank.origins_.decision_ms),"target original origins differ from authenticated feature bank");
    // Only shape/origin identity checks above touch the full table. Evaluation
    // masks, availability, label values and feature values are never inspected.
    std::vector<int64_t> train;
    for(size_t i=0;i<bank.rows_.size();++i)if(bank.rows_[i].head_role==HeadRole::train)train.push_back(static_cast<int64_t>(i));
    require(!train.empty(),"no head TRAIN origins");const auto n=static_cast<int64_t>(train.size());const auto& s=bank.stage_;
    ev::ObservableTargets fit;fit.identity=targets.identity;fit.names=targets.names;fit.units=targets.units;
    fit.origins={torch::zeros({n},torch::kInt64),torch::full({n},static_cast<int64_t>(ev::Role::train),torch::kInt64),torch::zeros({n},torch::kInt64)};
    fit.values=torch::zeros({n,labels},torch::kFloat64);fit.valid=torch::zeros({n,labels},torch::kBool);
    fit.available_ms=torch::full({n,labels},s.head_fit_ms,torch::kInt64);fit.support_begin_ms=torch::full({n,labels},s.train_begin_ms,torch::kInt64);fit.support_end_ms=fit.support_begin_ms.clone();
    TrainingSelection selected;selected.target_identity=targets.identity;selected.names=targets.names;selected.units=targets.units;
    selected.reasons.resize(bank.rows_.size(),std::vector<uint64_t>(static_cast<size_t>(labels),not_head_train));
    selected.common_training_rows.assign(static_cast<size_t>(labels),0);selected.selected_origin_ids.resize(static_cast<size_t>(labels));
    selected.label_available_ms.resize(bank.rows_.size(),std::vector<int64_t>(static_cast<size_t>(labels),-1));
    selected.label_support_begin_ms=selected.label_available_ms;selected.label_support_end_ms=selected.label_available_ms;
    std::vector<ev::FeatureArm> features;
    for(const auto& arm:bank.arms_) {
      auto view=arm;view.values=torch::zeros({n,arm.values.size(1)},torch::kFloat64);view.valid=torch::zeros({n},torch::kBool);
      view.origin_row_ids=torch::zeros({n},torch::kInt64);view.available_ms=torch::full({n},s.train_begin_ms,torch::kInt64);
      view.support_begin_ms=torch::full({n},s.train_begin_ms,torch::kInt64);view.support_end_ms=view.support_begin_ms.clone();features.push_back(std::move(view));
    }
    std::vector<std::string> identity_fields{targets.identity,std::to_string(labels)};
    for(size_t j=0;j<targets.names.size();++j){identity_fields.push_back(targets.names[j]);identity_fields.push_back(targets.units[j]);}
    for(size_t p=0;p<train.size();++p) {
      const auto i=train[p];const auto& row=bank.rows_[static_cast<size_t>(i)];const auto index=static_cast<int64_t>(p);
      fit.origins.row_ids[index]=row.row_id;fit.origins.decision_ms[index]=row.decision_ms;
      uint64_t input_why=row.common_eligible?uint64_t{0}:static_cast<uint64_t>(common_input_invalid);
      if(row.decision_ms>s.head_fit_ms)input_why|=train_decision_after_fit;
      if(row.input_available_ms>s.head_fit_ms)input_why|=feature_not_available_at_head_fit;
      for(size_t k=0;k<features.size();++k) {
        auto& view=features[k];view.origin_row_ids[index]=row.row_id;
        if(!input_why){view.values[index].copy_(bank.arms_[k].values[i]);view.valid[index]=true;view.available_ms[index]=row.input_available_ms;view.support_begin_ms[index]=row.input_begin_ms;view.support_end_ms[index]=row.endpoint_ms;}
      }
      for(int64_t j=0;j<labels;++j) {
        auto why=input_why;
        if(!targets.valid[i][j].item<bool>())why|=invalid_label;
        const auto available=targets.available_ms[i][j].item<int64_t>(),begin=targets.support_begin_ms[i][j].item<int64_t>(),end=targets.support_end_ms[i][j].item<int64_t>();
        if(available>s.head_fit_ms)why|=label_not_mature;
        if(begin<s.train_begin_ms||end>s.train_end_ms)why|=label_support_outside_train;
        if(begin<row.endpoint_ms||end<begin||available<end)why|=noncausal_label_metadata;
        selected.reasons[static_cast<size_t>(i)][static_cast<size_t>(j)]=why;if(why)continue;
        // Read the target value only after every input/mask/maturity/support
        // condition has admitted this specific TRAIN origin/observable.
        const double value=targets.values[i][j].item<double>();require(std::isfinite(value),"nonfinite admitted mature TRAIN label");
        fit.values[index][j]=value;fit.valid[index][j]=true;fit.available_ms[index][j]=available;fit.support_begin_ms[index][j]=begin;fit.support_end_ms[index][j]=end;
        ++selected.common_training_rows[static_cast<size_t>(j)];selected.selected_origin_ids[static_cast<size_t>(j)].push_back(row.row_id);
        selected.label_available_ms[static_cast<size_t>(i)][static_cast<size_t>(j)]=available;
        selected.label_support_begin_ms[static_cast<size_t>(i)][static_cast<size_t>(j)]=begin;
        selected.label_support_end_ms[static_cast<size_t>(i)][static_cast<size_t>(j)]=end;
        identity_fields.push_back(std::to_string(row.row_id));identity_fields.push_back(std::to_string(j));identity_fields.push_back(std::to_string(begin));identity_fields.push_back(std::to_string(end));identity_fields.push_back(std::to_string(available));
        identity_fields.push_back(std::to_string(std::bit_cast<uint64_t>(value)));
      }
    }
    require(std::any_of(selected.common_training_rows.begin(),selected.common_training_rows.end(),[](auto value){return value>=2;}),"no observable has sufficient mature common TRAIN support");
    selected.selected_identity=av::digest_fields("risk-selected-mature-train-labels-v1",identity_fields);
    ev::UtilityRecipe recipe{s.head_fit_ms,s.train_begin_ms,s.train_end_ms,1.,1e-8,1e-12};
    const auto model=ev::fit_risk_readouts(fit,features,recipe);return from_checked_snapshot(bank,model,selected);
  }
  std::vector<ev::RiskPrediction> predict(const FeatureBank& bank)const {
    using stage_detail::require;verify_runtime_source();require(bank.identity_==bank_identity_&&stage_digest(bank.stage_)==stage_digest(stage_),"head cannot consume another bank, tuple, coordinate fold or stage");
    auto origins=bank.origins_snapshot();auto features=bank.arms_snapshot();
    for(size_t i=0;i<bank.rows_.size();++i)if(bank.rows_[i].head_role!=HeadRole::evaluation||bank.rows_[i].endpoint_ms<=stage_.head_fit_ms)
      for(auto& arm:features)arm.valid[static_cast<int64_t>(i)]=false;
    auto result=ev::predict_risk_readouts(origins,features,model_);
    for(auto& prediction:result)for(size_t i=0;i<bank.rows_.size();++i) {
      const auto& row=bank.rows_[i];uint64_t reasons=row.reasons;
      if(row.head_role!=HeadRole::evaluation)reasons|=not_head_evaluation;
      if(row.endpoint_ms<=stage_.head_fit_ms)reasons|=head_not_before_origin;
      if(reasons)for(int64_t j=0;j<prediction.values.size(1);++j){prediction.values[static_cast<int64_t>(i)][j]=0.;prediction.valid[static_cast<int64_t>(i)][j]=false;prediction.reasons[static_cast<int64_t>(i)][j]=prediction.reasons[static_cast<int64_t>(i)][j].item<int64_t>()|static_cast<int64_t>(reasons<<16);}
    }
    return result;
  }
  const StageContract& contract()const{return stage_;}
  const std::string& bank_identity()const{return bank_identity_;}
  TrainingSelection selection_snapshot()const{return selection_;}
  ev::RiskReadoutBundle model_snapshot()const{return stage_detail::clone(model_);}
};
} // namespace cuwacunu::latent_market::risk_issuance::v1
