// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Artificial three-stage engineering fixture; no market execution path.
#include "cuwacunu/latent_market/risk_issuance/v1/publication.h"
#include "../tests/fixture.h"
#include <iostream>

namespace ri=cuwacunu::latent_market::risk_issuance::v1;
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace av=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;
namespace {
ri::FeatureBank bank(ri::test_fixture::Fixture& fixture) {
  const auto issuer=fixture.issuer();
  return ri::FeatureBank::build(fixture.stages(),fixture.timeline,fixture.pairs,issuer,fixture.serving,
    fixture.issuance_directory,fixture.issuance_pin,*fixture.native);
}
std::map<std::string,std::string> inventory(const fs::path& directory) {
  std::map<std::string,std::string> result;
  for(const auto& entry:fs::recursive_directory_iterator(directory))if(entry.is_regular_file())
    result.emplace(entry.path().lexically_relative(directory).generic_string(),av::sha256_file(entry.path()));
  return result;
}
void target_audit(const fs::path& file,const ri::FeatureBank& features,const ev::ObservableTargets& targets,
                  const ri::TrainingSelection& selected) {
  const auto rows=features.rows_snapshot();std::ostringstream out;
  out<<"row_id,observable_index,target_valid,available_ms,support_begin_ms,support_end_ms,training_selected,training_exclusion_reasons\n";
  for(size_t i=0;i<rows.size();++i)for(size_t j=0;j<targets.names.size();++j)
    out<<rows[i].row_id<<','<<j<<','<<targets.valid[i][j].item<bool>()<<','<<targets.available_ms[i][j].item<int64_t>()<<','
      <<targets.support_begin_ms[i][j].item<int64_t>()<<','<<targets.support_end_ms[i][j].item<int64_t>()<<','
      <<(selected.reasons[i][j]==0)<<','<<selected.reasons[i][j]<<'\n';
  ri::publication_detail::publish(file,out.str());
}
}
int main(int argc,char** argv)try {
  TORCH_CHECK(fs::exists("/.dockerenv"),"Run the synthetic fixture in the managed development container");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  TORCH_CHECK(argc>=2,"usage: risk_issuance_v1_pipeline synthetic-engineering NEW_LEAF | verify-synthetic INPUTS PUBLISHED PIN");
  const std::string mode=argv[1];
  TORCH_CHECK(mode=="synthetic-engineering" || mode=="verify-synthetic",
    "Market execution remains closed; this runner admits explicit synthetic engineering only");
  ri::verify_runtime_source();
  if(mode=="verify-synthetic") {
    TORCH_CHECK(argc==5,"verify-synthetic requires original inputs, published head and trusted manifest pin");
    ri::test_fixture::Fixture fixture(argv[2],false);const auto features=bank(fixture);
    const auto checked=ri::HeadPublication::load(argv[3],argv[4],features);
    std::cout<<"PASS checked synthetic head reload: "<<checked.head_model_ref<<'\n';return 0;
  }
  TORCH_CHECK(argc==3,"synthetic-engineering requires one fresh output leaf");
  const fs::path directory=argv[2];TORCH_CHECK(!fs::exists(directory) && fs::is_directory(directory.parent_path()),"A fresh leaf inside an existing directory is required");
  TORCH_CHECK(fs::create_directory(directory),"Cannot create synthetic output leaf");
  ri::test_fixture::Fixture fixture(directory/"inputs",true);
  const auto upstream_before=inventory(directory/"inputs");
  const auto features=bank(fixture);const auto targets=fixture.targets(features);
  const auto head=ri::FittedStage::fit(features,targets);const auto selection=head.selection_snapshot();
  TORCH_CHECK(fs::create_directory(directory/"published"),"Cannot create synthetic head publication leaf");
  const auto pin=ri::HeadPublication::publish(directory/"published",features,head);
  const auto loaded=ri::HeadPublication::load(directory/"published",pin,features);
  ri::publication_detail::same_predictions(loaded.predictions,head.predict(features));
  auto scoring=targets;scoring.values=targets.values.clone();scoring.valid=targets.valid.clone();
  const auto rows=features.rows_snapshot();const auto& stage=features.contract();
  for(size_t i=0;i<rows.size();++i)for(size_t j=0;j<targets.names.size();++j) {
    const bool evaluation=rows[i].head_role==ri::HeadRole::evaluation;
    const bool support=targets.support_begin_ms[i][j].item<int64_t>()>=stage.evaluation_begin_ms &&
      targets.support_end_ms[i][j].item<int64_t>()<=stage.evaluation_end_ms;
    if(!evaluation || !support)scoring.valid[i][j]=false;
  }
  std::vector<int64_t> evaluation_indices;for(size_t i=0;i<rows.size();++i)
    if(rows[i].head_role==ri::HeadRole::evaluation)evaluation_indices.push_back(static_cast<int64_t>(i));
  const auto evaluation_index=torch::tensor(evaluation_indices,torch::kInt64);
  const auto slice_origins=[&](const ev::ObservableOrigins& original){return ev::ObservableOrigins{
    original.row_ids.index_select(0,evaluation_index),original.roles.index_select(0,evaluation_index),original.decision_ms.index_select(0,evaluation_index)};};
  scoring.origins=slice_origins(features.origins_snapshot());
  scoring.values=scoring.values.index_select(0,evaluation_index);scoring.valid=scoring.valid.index_select(0,evaluation_index);
  scoring.available_ms=scoring.available_ms.index_select(0,evaluation_index);
  scoring.support_begin_ms=scoring.support_begin_ms.index_select(0,evaluation_index);scoring.support_end_ms=scoring.support_end_ms.index_select(0,evaluation_index);
  auto scoring_predictions=loaded.predictions;for(auto& prediction:scoring_predictions) {
    prediction.origins=slice_origins(prediction.origins);prediction.values=prediction.values.index_select(0,evaluation_index);
    prediction.valid=prediction.valid.index_select(0,evaluation_index);prediction.reasons=prediction.reasons.index_select(0,evaluation_index);
  }
  const auto scores=ev::score_risk_readouts(scoring,scoring_predictions,loaded.head.model_snapshot(),ev::Role::validation,stage.scoring_cutoff_ms);
  std::ostringstream report;report<<std::setprecision(17)<<"arm,observable,unit,total,eligible_targets,valid_inputs,common_scored,coverage,mse\n";
  for(const auto& score:scores)report<<ri::fi::publication_detail::csv(score.arm_name)<<','<<ri::fi::publication_detail::csv(score.observable)<<','
    <<ri::fi::publication_detail::csv(score.unit)<<','<<score.total<<','<<score.eligible_targets<<','<<score.valid_inputs<<','<<score.common_scored<<','
    <<score.coverage<<','<<(score.supported?ri::publication_detail::scalar(score.mse):"N/A")<<'\n';
  ri::publication_detail::publish(directory/"scores.csv",report.str());target_audit(directory/"target-audit.csv",features,targets,selection);
  TORCH_CHECK(upstream_before==inventory(directory/"inputs"),"Head fitting or serving changed an upstream artifact");
  std::string counts;for(const auto count:selection.common_training_rows){if(!counts.empty())counts+=';';counts+=std::to_string(count);}
  ri::publication_detail::Fields receipt{{"schema","synthetic-staged-risk-head-run-v1"},{"scope","synthetic-engineering"},
    {"policy_sha256",stage.policy_sha256},{"source_closure_sha256",ri::source_identity()},
    {"head_manifest_sha256",pin},{"head_model_ref",loaded.head_model_ref},{"bank_identity",features.identity()},
    {"origin_count",std::to_string(rows.size())},{"arm_count",std::to_string(loaded.predictions.size())},
    {"observable_count",std::to_string(targets.names.size())},{"common_training_rows",counts},
    {"encoder_calls","0"},{"upstream_updates_during_head_stage","0"},{"market_fitting","false"},
    {"scores_sha256",av::sha256_file(directory/"scores.csv")},{"target_audit_sha256",av::sha256_file(directory/"target-audit.csv")},
    {"binary_sha256",av::sha256_file("/proc/self/exe")}};
  ri::publication_detail::publish(directory/"run.json",ri::fi::publication_detail::serialize(receipt));
  std::cout<<"PASS staged risk-head pipeline: checked issued TRAIN features, mature common labels, frozen head publication, later label-free predictions and common-observable scoring; synthetic engineering only\n";
  std::cout<<"Evidence preserved at "<<directory.string()<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
