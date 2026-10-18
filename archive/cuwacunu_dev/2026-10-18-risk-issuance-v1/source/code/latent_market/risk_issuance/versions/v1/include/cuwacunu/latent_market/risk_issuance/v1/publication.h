// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "stage.h"
#include "source_identity.h"
#include "cuwacunu/artifacts/v1/tensor_state.h"

namespace cuwacunu::latent_market::risk_issuance::v1 {
namespace publication_detail {
using Fields=std::map<std::string,std::string>;
inline void require(bool test,const std::string& message){TORCH_CHECK(test,"[risk head publication] ",message);}
inline std::string scalar(double number){std::ostringstream out;out<<std::setprecision(17)<<number;return out.str();}
inline std::string model_semantics(const ev::RiskReadoutBundle& model) {
  std::vector<std::string> fields{model.target_identity,std::to_string(model.recipe.fit_cutoff_ms),
    std::to_string(model.recipe.train_begin_ms),std::to_string(model.recipe.train_end_ms),
    scalar(model.recipe.ridge_penalty),scalar(model.recipe.std_floor),scalar(model.recipe.denominator_epsilon),
    std::to_string(model.names.size()),std::to_string(model.arms.size())};
  for(size_t j=0;j<model.names.size();++j){fields.push_back(model.names[j]);fields.push_back(model.units[j]);fields.push_back(std::to_string(model.common_training_rows[j]));}
  for(const auto& arm:model.arms){fields.push_back(arm.name);fields.push_back(arm.identity);fields.push_back(arm.information_contract);
    fields.push_back(std::to_string(arm.input_width));fields.push_back(std::to_string(arm.artifact_fit_cutoff_ms));}
  return av::digest_fields("risk-stage-head-semantics-v1",fields);
}
inline std::string model_state(const ev::RiskReadoutBundle& model) {
  std::vector<av::NamedTensor> tensors;
  for(size_t k=0;k<model.arms.size();++k)for(size_t j=0;j<model.arms[k].heads.size();++j) {
    const auto& head=model.arms[k].heads[j];const auto prefix="arm."+std::to_string(k)+".observable."+std::to_string(j)+".";
    tensors.emplace_back(prefix+"input_mean",head.input_mean);tensors.emplace_back(prefix+"input_scale",head.input_scale);
    tensors.emplace_back(prefix+"weights",head.weights);
    tensors.emplace_back(prefix+"target_reference",torch::tensor(std::vector<double>{head.target_mean,head.target_scale},torch::kFloat64));
    tensors.emplace_back(prefix+"training_rows",torch::tensor(std::vector<int64_t>{head.training_rows},torch::kInt64));
  }
  return av::tensor_state_sha256(std::move(tensors));
}
inline Fields fields(const FeatureBank& bank,const ev::RiskReadoutBundle& model,const TrainingSelection& selected) {
  const auto& stage=bank.contract();
  Fields fields{{"schema","risk-issuance-publication-v1"},{"policy_sha256",stage.policy_sha256},
    {"source_closure_sha256",source_identity()},{"stage_sha256",stage_digest(stage)},
    {"bank_identity",bank.identity()},{"upstream_contract_sha256",bank.upstream_contract_sha256()},
    {"issuance_manifest_sha256",bank.issuance_manifest_sha256()},
    {"native_identity",bank.native_identity()},{"native_model_ref",bank.native_model_ref()},
    {"native_sidecar_sha256",bank.native_sidecar_sha256()},{"native_comparison_sha256",bank.native_comparison_sha256()},
    {"native_output_sha256",bank.native_output_sha256()},
    {"native_fit_ms",std::to_string(bank.native_fit_cutoff_ms())},
    {"train_begin_ms",std::to_string(stage.train_begin_ms)},{"train_end_ms",std::to_string(stage.train_end_ms)},
    {"head_fit_ms",std::to_string(stage.head_fit_ms)},{"evaluation_begin_ms",std::to_string(stage.evaluation_begin_ms)},
    {"evaluation_end_ms",std::to_string(stage.evaluation_end_ms)},{"scoring_cutoff_ms",std::to_string(stage.scoring_cutoff_ms)},
    {"target_identity",model.target_identity},{"selected_training_identity",selected.selected_identity},
    {"origin_count",std::to_string(bank.rows_snapshot().size())},{"arm_count",std::to_string(model.arms.size())},
    {"observable_count",std::to_string(model.names.size())},{"model_semantics_sha256",model_semantics(model)},
    {"model_state_sha256",model_state(model)}};
  for(const auto& [key,value]:fi::publication_detail::fields(bank.upstream_snapshot()))fields["upstream_"+key]=value;
  fields["head_model_ref"]=av::digest_fields("risk-stage-head-model-v1",{fields.at("policy_sha256"),fields.at("source_closure_sha256"),
    fields.at("stage_sha256"),fields.at("bank_identity"),fields.at("selected_training_identity"),
    fields.at("model_semantics_sha256"),fields.at("model_state_sha256")});
  return fields;
}
inline std::string origins_csv(const FeatureBank& bank) {
  std::ostringstream out;out<<"row_id,pair_id,origin_ms,decision_ms,capture_role,head_role,input_begin_ms,input_available_ms,common_eligible,reasons,upstream_reasons,native_reasons,dependence_group\n";
  for(const auto& row:bank.rows_snapshot())out<<row.row_id<<','<<fi::publication_detail::csv(row.pair_id)<<','<<row.endpoint_ms<<','<<row.decision_ms<<','
    <<static_cast<int64_t>(row.capture_role)<<','<<static_cast<int64_t>(row.head_role)<<','<<row.input_begin_ms<<','<<row.input_available_ms<<','
    <<row.common_eligible<<','<<row.reasons<<','<<row.upstream_reasons<<','<<row.native_reasons<<','<<fi::publication_detail::csv(row.dependence_group)<<'\n';
  return out.str();
}
inline std::string selection_csv(const FeatureBank& bank,const TrainingSelection& selected) {
  const auto rows=bank.rows_snapshot();require(selected.reasons.size()==rows.size(),"selection origin count differs");
  std::ostringstream out;out<<"origin_index,row_id,observable_index,selected,reasons,label_available_ms,label_support_begin_ms,label_support_end_ms\n";
  for(size_t i=0;i<rows.size();++i){require(selected.reasons[i].size()==selected.names.size(),"selection observable count differs");
    for(size_t j=0;j<selected.names.size();++j)out<<i<<','<<rows[i].row_id<<','<<j<<','<<(selected.reasons[i][j]==0)<<','<<selected.reasons[i][j]<<','
      <<selected.label_available_ms.at(i).at(j)<<','<<selected.label_support_begin_ms.at(i).at(j)<<','<<selected.label_support_end_ms.at(i).at(j)<<'\n';}
  return out.str();
}
inline uint64_t number(const std::string& value) {
  size_t end=0;const auto result=std::stoull(value,&end);
  require(end==value.size() && value==std::to_string(result),"noncanonical selection number");return result;
}
inline int64_t signed_number(const std::string& value) {
  size_t end=0;const auto result=std::stoll(value,&end);
  require(end==value.size() && value==std::to_string(result),"noncanonical selection metadata");return result;
}
inline TrainingSelection parse_selection(const std::string& csv,const FeatureBank& bank,
    const ev::RiskReadoutBundle& model,const std::string& identity) {
  TrainingSelection selected;selected.target_identity=model.target_identity;selected.selected_identity=identity;
  selected.names=model.names;selected.units=model.units;selected.common_training_rows=model.common_training_rows;
  const auto rows=bank.rows_snapshot();selected.reasons.resize(rows.size(),std::vector<uint64_t>(model.names.size()));
  selected.label_available_ms.resize(rows.size(),std::vector<int64_t>(model.names.size(),-1));
  selected.label_support_begin_ms=selected.label_available_ms;selected.label_support_end_ms=selected.label_available_ms;
  selected.selected_origin_ids.resize(model.names.size());std::istringstream input(csv);std::string line;
  require(bool(std::getline(input,line)) && line=="origin_index,row_id,observable_index,selected,reasons,label_available_ms,label_support_begin_ms,label_support_end_ms","wrong selection ledger schema");
  for(size_t i=0;i<rows.size();++i)for(size_t j=0;j<model.names.size();++j) {
    require(bool(std::getline(input,line)),"missing selection association");std::istringstream record(line);
    std::vector<std::string> fields;std::string field;while(std::getline(record,field,','))fields.push_back(field);
    require(fields.size()==8 && number(fields[0])==i && number(fields[1])==static_cast<uint64_t>(rows[i].row_id) &&
      number(fields[2])==j,"selection origin/observable association differs");
    const auto reasons=number(fields[4]);require(number(fields[3])==static_cast<uint64_t>(reasons==0),"selection validity/reasons differ");
    selected.reasons[i][j]=reasons;if(!reasons)selected.selected_origin_ids[j].push_back(rows[i].row_id);
    selected.label_available_ms[i][j]=signed_number(fields[5]);selected.label_support_begin_ms[i][j]=signed_number(fields[6]);
    selected.label_support_end_ms[i][j]=signed_number(fields[7]);
  }
  require(!std::getline(input,line),"extra selection associations");
  for(size_t j=0;j<model.names.size();++j)require(selected.selected_origin_ids[j].size()==static_cast<size_t>(model.common_training_rows[j]),"selected TRAIN population differs");
  require(selection_csv(bank,selected)==csv,"noncanonical selection ledger");return selected;
}
template<class Loader> auto decode(const std::string& bytes,Loader loader) {
  const int fd=static_cast<int>(::syscall(SYS_memfd_create,"risk-head-stage",MFD_CLOEXEC|MFD_ALLOW_SEALING));
  require(fd>=0,"cannot create authenticated head memory file");
  try {
    size_t at=0;while(at<bytes.size()){const auto count=::write(fd,bytes.data()+at,bytes.size()-at);if(count<0 && errno==EINTR)continue;
      require(count>0,"cannot write authenticated head bytes");at+=static_cast<size_t>(count);}
    require(::fcntl(fd,F_ADD_SEALS,F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL)==0,"cannot seal authenticated head bytes");
    auto result=loader("/proc/self/fd/"+std::to_string(fd));::close(fd);return result;
  }catch(...){::close(fd);throw;}
}
inline void same_predictions(const std::vector<ev::RiskPrediction>& saved,const std::vector<ev::RiskPrediction>& expected) {
  require(saved.size()==expected.size(),"saved prediction arm count differs");
  for(size_t k=0;k<saved.size();++k){const auto& a=saved[k];const auto& b=expected[k];
    require(a.arm_name==b.arm_name && a.feature_identity==b.feature_identity && a.target_identity==b.target_identity &&
      a.fit_cutoff_ms==b.fit_cutoff_ms && torch::equal(a.origins.row_ids,b.origins.row_ids) &&
      torch::equal(a.origins.roles,b.origins.roles) && torch::equal(a.origins.decision_ms,b.origins.decision_ms) &&
      torch::equal(a.values,b.values) && torch::equal(a.valid,b.valid) && torch::equal(a.reasons,b.reasons),
      "saved predictions differ from the exact frozen head and original associations");}
}
inline std::string predictions_csv(const FeatureBank& bank,const std::vector<ev::RiskPrediction>& predictions) {
  const auto rows=bank.rows_snapshot();std::ostringstream out;out<<std::setprecision(17);
  out<<"row_id,pair_id,origin_ms,decision_ms,arm,observable_index,head_fit_ms,valid,reasons,prediction\n";
  for(const auto& prediction:predictions)for(size_t i=0;i<rows.size();++i)for(int64_t j=0;j<prediction.values.size(1);++j)
    out<<rows[i].row_id<<','<<fi::publication_detail::csv(rows[i].pair_id)<<','<<rows[i].endpoint_ms<<','<<rows[i].decision_ms<<','
      <<fi::publication_detail::csv(prediction.arm_name)<<','<<j<<','<<prediction.fit_cutoff_ms<<','
      <<prediction.valid[i][j].item<bool>()<<','<<prediction.reasons[i][j].item<int64_t>()<<','<<prediction.values[i][j].item<double>()<<'\n';
  return out.str();
}
inline void publish(const std::filesystem::path& file,const std::string& bytes) {
  lm::ledger_detail::publish(file,[&](auto& stream){stream<<bytes;});
}
}

class HeadPublication {
 public:
  struct Loaded {FittedStage head;std::vector<ev::RiskPrediction> predictions;std::string head_model_ref;};
  static std::string publish(const std::filesystem::path& directory,const FeatureBank& bank,const FittedStage& head) {
    using namespace publication_detail;verify_runtime_source();require(head.bank_identity()==bank.identity(),"head belongs to another frozen bank");
    const auto model=head.model_snapshot();const auto selection=head.selection_snapshot();const auto predictions=head.predict(bank);
    ev::save_risk_readouts((directory/"readouts.pt").string(),model);ev::save_risk_predictions((directory/"predictions.pt").string(),predictions);
    const auto model_bytes=av::detail::read_bytes(directory/"readouts.pt",uintmax_t{2}<<30);
    const auto restored=decode(model_bytes,[](const auto& path){return ev::load_risk_readouts(path);});
    require(model_semantics(restored)==model_semantics(model) && model_state(restored)==model_state(model),"saved head differs from fitted immutable state");
    const auto prediction_bytes=av::detail::read_bytes(directory/"predictions.pt",uintmax_t{2}<<30);
    const auto saved=decode(prediction_bytes,[](const auto& path){return ev::load_risk_predictions(path);});same_predictions(saved,predictions);
    auto manifest=fields(bank,model,selection);
    const std::vector<std::pair<std::string,std::string>> ledgers{{"origins.csv",origins_csv(bank)},
      {"training-selection.csv",selection_csv(bank,selection)},{"prediction-ledger.csv",predictions_csv(bank,predictions)}};
    for(const auto& [name,bytes]:ledgers){publication_detail::publish(directory/name,bytes);manifest[name+"_sha256"]=av::sha256_bytes(bytes);}
    manifest["readouts_sha256"]=av::sha256_bytes(model_bytes);manifest["predictions_sha256"]=av::sha256_bytes(prediction_bytes);
    const auto bytes=fi::publication_detail::serialize(manifest);publication_detail::publish(directory/"head.json",bytes);return av::sha256_bytes(bytes);
  }
  static Loaded load(const std::filesystem::path& directory,const std::string& manifest_pin,const FeatureBank& bank) {
    using namespace publication_detail;verify_runtime_source();require(av::is_sha256(manifest_pin),"a trusted head manifest pin is required");
    const auto bytes=av::detail::read_bytes(directory/"head.json",262144);require(av::sha256_bytes(bytes)==manifest_pin,"head manifest digest mismatch");
    const auto json=av::detail::JsonReader(bytes).read();require(json.kind==av::detail::Json::Kind::object,"head manifest must be an object");
    Fields manifest;for(const auto& [key,value]:json.object)manifest.emplace(key,av::detail::str(value));
    const auto read=[&](const std::string& name,const std::string& field,uintmax_t bound){
      require(manifest.contains(field) && av::is_sha256(manifest.at(field)),"missing saved head/output pin");
      const auto data=av::detail::read_bytes(directory/name,bound);require(av::sha256_bytes(data)==manifest.at(field),"saved head/output digest mismatch: "+name);return data;};
    const auto model_bytes=read("readouts.pt","readouts_sha256",uintmax_t{2}<<30);
    const auto prediction_bytes=read("predictions.pt","predictions_sha256",uintmax_t{2}<<30);
    const auto origins=read("origins.csv","origins.csv_sha256",uintmax_t{1}<<30);
    const auto selection_bytes=read("training-selection.csv","training-selection.csv_sha256",uintmax_t{1}<<30);
    const auto ledger=read("prediction-ledger.csv","prediction-ledger.csv_sha256",uintmax_t{1}<<30);
    require(origins==origins_csv(bank),"original head origin associations differ");
    const auto model=decode(model_bytes,[](const auto& path){return ev::load_risk_readouts(path);});
    require(manifest.contains("selected_training_identity"),"missing selected TRAIN identity");
    const auto selection=parse_selection(selection_bytes,bank,model,manifest.at("selected_training_identity"));
    auto expected=fields(bank,model,selection);for(const auto* field:{"readouts_sha256","predictions_sha256","origins.csv_sha256","training-selection.csv_sha256","prediction-ledger.csv_sha256"})expected[field]=manifest.at(field);
    require(manifest==expected,"head state, policy, stage, source, parent, coordinates or selected population differs");
    auto head=FittedStage::from_checked_snapshot(bank,model,selection);
    const auto predictions=decode(prediction_bytes,[](const auto& path){return ev::load_risk_predictions(path);});
    same_predictions(predictions,head.predict(bank));require(ledger==predictions_csv(bank,predictions),"prediction ledger differs from checked outputs");
    return {std::move(head),predictions,manifest.at("head_model_ref")};
  }
};
}
