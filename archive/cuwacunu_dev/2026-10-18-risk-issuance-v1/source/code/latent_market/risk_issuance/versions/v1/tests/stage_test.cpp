// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include "cuwacunu/latent_market/risk_issuance/v1/publication.h"
#include "fixture.h"
#include <iostream>

namespace ri=cuwacunu::latent_market::risk_issuance::v1;
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace av=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;
namespace {
size_t checks=0;
void check(bool value,const std::string& why){++checks;TORCH_CHECK(value,"[risk stage test] ",why);}
template<class F>void rejects(F&& action,const std::string& why){bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}check(rejected,why);}
struct Temporary {
  fs::path path;
  Temporary(){auto pattern=(fs::temp_directory_path()/"risk-stage-v1-XXXXXX").string();std::vector<char> bytes(pattern.begin(),pattern.end());bytes.push_back(0);const auto made=::mkdtemp(bytes.data());TORCH_CHECK(made,"temporary directory");path=made;}
  ~Temporary(){std::error_code ignored;fs::remove_all(path,ignored);}
};
ri::FeatureBank bank(ri::test_fixture::Fixture& f,const ri::StageContract& stage) {
  const auto issuer=f.issuer();return ri::FeatureBank::build(stage,f.timeline,f.pairs,issuer,f.serving,f.issuance_directory,f.issuance_pin,*f.native);
}
ev::ObservableTargets clone(const ev::ObservableTargets& t) {
  auto v=t;v.origins=ri::stage_detail::clone(t.origins);v.values=t.values.clone();v.valid=t.valid.clone();v.available_ms=t.available_ms.clone();v.support_begin_ms=t.support_begin_ms.clone();v.support_end_ms=t.support_end_ms.clone();return v;
}
bool numeric_equal(const ev::RiskReadoutBundle& x,const ev::RiskReadoutBundle& y) {
  if(x.arms.size()!=y.arms.size()||x.common_training_rows!=y.common_training_rows)return false;
  for(size_t k=0;k<x.arms.size();++k)for(size_t j=0;j<x.arms[k].heads.size();++j){const auto& a=x.arms[k].heads[j];const auto& b=y.arms[k].heads[j];
    if(a.training_rows!=b.training_rows||a.target_mean!=b.target_mean||a.target_scale!=b.target_scale||
      !torch::equal(a.input_mean,b.input_mean)||!torch::equal(a.input_scale,b.input_scale)||!torch::equal(a.weights,b.weights))return false;}
  return true;
}
void fitting_and_privacy(ri::test_fixture::Fixture& fixture,const ri::FeatureBank& features) {
  const auto targets=fixture.targets(features);const auto head=ri::FittedStage::fit(features,targets);const auto selected=head.selection_snapshot();
  const auto model=head.model_snapshot();const auto rows=features.rows_snapshot();
  check(model.arms.size()==7&&model.names.size()==6,"seven frozen feature arms and six observable heads");
  check(std::all_of(model.common_training_rows.begin(),model.common_training_rows.end(),[](auto n){return n>=2;}),"each observable has common mature TRAIN support");
  for(const auto& arm:model.arms)for(size_t j=0;j<arm.heads.size();++j)check(arm.heads[j].training_rows==model.common_training_rows[j],"compared arms use identical per-observable populations");
  auto poisoned=clone(targets);
  for(size_t i=0;i<rows.size();++i)for(size_t j=0;j<targets.names.size();++j) {
    if(rows[i].head_role!=ri::HeadRole::train){poisoned.values[i][j]=NAN;poisoned.valid[i][j]=true;poisoned.available_ms[i][j]=INT64_MIN;poisoned.support_begin_ms[i][j]=INT64_MAX;poisoned.support_end_ms[i][j]=INT64_MIN;}
    else if(selected.reasons[i][j])poisoned.values[i][j]=NAN;
  }
  const auto unchanged=ri::FittedStage::fit(features,poisoned);
  check(numeric_equal(model,unchanged.model_snapshot()),"evaluation poison and excluded TRAIN values cannot enter fitted parameters/transforms");
  check(selected.selected_identity==unchanged.selection_snapshot().selected_identity,"selected TRAIN identity excludes held-out and excluded values");
  const auto prediction=head.predict(features);ri::publication_detail::same_predictions(prediction,unchanged.predict(features));
  check(true,"prediction receives no future target values, masks or availability");
  auto editable_model=head.model_snapshot();editable_model.arms[0].heads[0].weights.fill_(NAN);editable_model.arms[0].heads[0].input_mean.fill_(NAN);
  ri::publication_detail::same_predictions(prediction,head.predict(features));check(true,"head snapshots cannot mutate privately owned fitted tensors");
  auto editable_arms=features.arms_snapshot();editable_arms[0].values.fill_(NAN);editable_arms[1].valid.fill_(false);editable_arms[2].available_ms.fill_(INT64_MAX);
  auto editable_origins=features.origins_snapshot();editable_origins.row_ids.fill_(-1);
  check(numeric_equal(model,ri::FittedStage::fit(features,targets).model_snapshot()),"feature/origin snapshots cannot mutate private input bank");
  bool have_invalid=false,have_valid=false;
  for(size_t i=0;i<rows.size();++i){const bool valid=prediction[0].valid[static_cast<int64_t>(i)].any().item<bool>();have_valid=have_valid||valid;have_invalid=have_invalid||!valid;
    if(rows[i].head_role!=ri::HeadRole::evaluation||rows[i].endpoint_ms<=features.contract().head_fit_ms)
      check(!valid&&prediction[0].values[static_cast<int64_t>(i)].count_nonzero().item<int64_t>()==0,"unavailable origin remains zero and invalid despite delayed decision");}
  check(have_valid&&have_invalid&&prediction[0].values.size(0)==static_cast<int64_t>(rows.size()),"all requested origins retained with valid and invalid associations");
  auto late=features.contract();late.head_fit_ms=late.evaluation_begin_ms;
  rejects([&]{bank(fixture,late);},"head fit equality with evaluation origin cannot be legalized by decision latency");
  auto early=features.contract();early.train_begin_ms=fixture.contract.fits.online_ms;
  rejects([&]{bank(fixture,early);},"upstream fit equality with head TRAIN begin rejected");
  auto wrong=clone(targets);wrong.origins.row_ids[0]=-1;
  rejects([&]{ri::FittedStage::fit(features,wrong);},"wrong original label associations rejected");
  auto unsupported=clone(targets);unsupported.valid.fill_(false);unsupported.values.fill_(NAN);
  rejects([&]{ri::FittedStage::fit(features,unsupported);},"no mature common support cannot publish a fitted stage");
  const auto chosen=selected.selected_origin_ids[0].front()-1;auto exact=clone(targets);exact.available_ms[chosen][0]=features.contract().head_fit_ms;
  const auto at_cutoff=ri::FittedStage::fit(features,exact).selection_snapshot();
  check(at_cutoff.reasons[static_cast<size_t>(chosen)][0]==0,"label maturity exactly at cutoff admitted");
  auto immature=clone(exact);immature.available_ms[chosen][0]=features.contract().head_fit_ms+1;immature.values[chosen][0]=NAN;
  const auto excluded=ri::FittedStage::fit(features,immature).selection_snapshot();
  check(excluded.reasons[static_cast<size_t>(chosen)][0]&ri::label_not_mature,"later label maturity excludes unread NaN value");
  auto crossing=clone(targets);crossing.support_end_ms[chosen][0]=features.contract().train_end_ms+1;crossing.available_ms[chosen][0]=features.contract().head_fit_ms;crossing.values[chosen][0]=NAN;
  check(ri::FittedStage::fit(features,crossing).selection_snapshot().reasons[static_cast<size_t>(chosen)][0]&ri::label_support_outside_train,"stage-crossing target support excludes unread NaN value");
}
void publication(Temporary& temp,ri::test_fixture::Fixture& fixture,const ri::FeatureBank& features) {
  const auto head=ri::FittedStage::fit(features,fixture.targets(features));const auto directory=temp.path/"published";fs::create_directory(directory);
  const auto pin=ri::HeadPublication::publish(directory,features,head);const auto loaded=ri::HeadPublication::load(directory,pin,features);
  check(numeric_equal(head.model_snapshot(),loaded.head.model_snapshot()),"checked head archive round-trip preserves every parameter and frozen transform");
  ri::publication_detail::same_predictions(head.predict(features),loaded.predictions);check(true,"checked same-byte predictions round-trip");
  rejects([&]{ri::HeadPublication::publish(directory,features,head);},"immutable publication refuses overwrite");
  rejects([&]{ri::HeadPublication::load(directory,av::sha256_bytes("wrong pin"),features);},"wrong head manifest pin rejected");
  for(const auto& file:{"readouts.pt","predictions.pt","origins.csv","training-selection.csv","prediction-ledger.csv","head.json"}) {
    const auto copy=temp.path/(std::string("corrupt-")+file);fs::copy(directory,copy,fs::copy_options::recursive);
    {std::ofstream output(copy/file,std::ios::app|std::ios::binary);output<<'!';}
    rejects([&]{ri::HeadPublication::load(copy,pin,features);},std::string("changed ")+file+" rejected");
  }
  ri::test_fixture::Fixture alternate(temp.path/"alternative-capture",true,.25);const auto other=bank(alternate,alternate.stages());
  const auto other_head=ri::FittedStage::fit(other,alternate.targets(other));
  check(numeric_equal(head.model_snapshot(),other_head.model_snapshot()),"independently authenticated changed evaluation features leave fitted numerical heads unchanged");
  check(features.identity()!=other.identity(),"changed capture provenance retains a distinct exact bank identity");
  rejects([&]{head.predict(other);},"one head cannot consume another same-width frozen coordinate bank");
  rejects([&]{ri::HeadPublication::load(directory,pin,other);},"checked head rejects wrong upstream bank before serving");
}
}
int main()try {
  torch::set_num_threads(1);torch::set_num_interop_threads(1);Temporary temporary;
  ri::test_fixture::Fixture fixture(temporary.path/"inputs",true);const auto features=bank(fixture,fixture.stages());
  fitting_and_privacy(fixture,features);publication(temporary,fixture,features);
  std::cout<<checks<<" risk issuance v1 synthetic checks passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
