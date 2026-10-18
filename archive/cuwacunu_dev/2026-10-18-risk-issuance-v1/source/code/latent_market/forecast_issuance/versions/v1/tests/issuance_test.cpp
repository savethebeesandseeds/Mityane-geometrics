// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Artificial chronology/identity witnesses; no market fitting or quality claim.
#include "cuwacunu/latent_market/forecast_issuance/v1/publication.h"
#include <iostream>

namespace fi=cuwacunu::latent_market::forecast_issuance::v1;
namespace lm=cuwacunu::latent_market::v1;
namespace pt=cuwacunu::networks::predictive_training::v1;
namespace pa=pt::artifact;
namespace av=cuwacunu::artifacts::v1;
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace fs=std::filesystem;
namespace {
size_t checks=0;
void check(bool value,const std::string& message){++checks;TORCH_CHECK(value,"[issuance test] ",message);}
template<class F>void rejects(F&& f,const std::string& message){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected,message);}
struct Temporary {
  fs::path path;
  Temporary(){auto pattern=(fs::temp_directory_path()/"forecast-issuance-v1-XXXXXX").string();std::vector<char> bytes(pattern.begin(),pattern.end());bytes.push_back('\0');const auto created=::mkdtemp(bytes.data());TORCH_CHECK(created,"temporary fixture directory");path=created;}
  ~Temporary(){std::error_code ignored;fs::remove_all(path,ignored);}
};
struct Fixture {
  Temporary temporary;
  pt::Config config;
  pt::Identity identity;
  pa::Context context;
  av::VerifiedArtifact encoder,predictive;
  std::shared_ptr<const pa::CheckedInference> serving;
  lm::EmbeddingTimeline timeline;
  lm::PairSpec spec;
  std::vector<lm::RoleInterval> roles{{lm::Role::train,0,2000}};
  lm::PairOrigin origin{"origin-1200",1200,1202,lm::Role::train,"synthetic-block"};
  fi::FrozenContract contract;
  Fixture() {
    const auto encoder_path=temporary.path/"encoder.pt";
    {std::ofstream out(encoder_path,std::ios::binary);out<<"artificial Encoder archive; no market values";}
    const auto encoder_sha=av::sha256_file(encoder_path);
    context.encoder=av::make_component({std::string(av::encoder_component),"v3",av::sha256_bytes("artificial Encoder source")},
      av::sha256_bytes("artificial Encoder weights"),av::sha256_bytes("artificial Encoder mapping"));
    av::Manifest encoder_manifest{"inference",encoder_sha,"",{context.encoder}};
    const auto encoder_pin=av::publish_sidecar(encoder_path,encoder_manifest);encoder=av::read_verified(encoder_path,encoder_pin,"inference");
    auto& id=timeline.identity;id.artifact_id="artificial-issuance-timeline-v1";id.dataset_id="artificial chronology";
    id.encoder_family="artificial";id.encoder_version="v3";id.checkpoint_hash=encoder_sha;id.scaler_hash=av::sha256_bytes("fixed scaler");
    id.encoder_config_hash=av::sha256_bytes("fixed config");id.source_hash=av::sha256_bytes("fixed source");id.cache_hash=av::sha256_bytes("fixed cache");
    id.export_layout="native3";id.availability_policy="SIMULATED endpoint+2ms";id.source_closure_hash=av::sha256_bytes("capture closure");
    id.executable_hash=av::sha256_bytes("capture executable");id.encoder_fit_cutoff_ms=900;id.channel_ids={7};id.feature_names={"close"};
    timeline.interval_ms=10;timeline.window_ticks=3;timeline.embedding_width=3;
    for(int64_t endpoint=1000;endpoint<=1400;endpoint+=10) {
      lm::EmbeddingRow row;row.window_id=lm::expected_window_id(id,endpoint);row.endpoint_ms=endpoint;row.available_ms=endpoint+2;
      row.role=lm::Role::train;row.sample_valid=true;row.values={std::sin(endpoint*.01),std::cos(endpoint*.03),endpoint*.001};
      row.support={{endpoint-20,endpoint-10,endpoint},1,1,{1,1,1}};timeline.rows.push_back(row);
    }
    lm::validate_timeline(timeline);spec.history_steps=4;spec.spacing_ticks=1;spec.horizons_ticks={1,3};
    config.state.embedding_width=3;config.state.state_width=2;config.state.identity_initialization=false;
    config.state.device=torch::Device(torch::kCPU);config.state.dtype=torch::kFloat64;
    config.dynamics.state_width=2;config.dynamics.history_length=4;config.dynamics.horizons=spec.horizons_ticks;
    config.dynamics.device=config.state.device;config.dynamics.dtype=config.state.dtype;config.sampler_seed=418;
    identity={encoder_sha,id.scaler_hash,av::sha256_bytes("TRAIN transform"),av::sha256_bytes("TRAIN reference"),
      fi::timeline_digest(timeline),"",av::sha256_bytes("saved artificial pairs"),"artificial sampler","artificial recipe","frozen-target-state"};
    identity.split="fit:1150;ledger:"+identity.pair_ledger;
    context.encoder_archive_sha256=encoder_sha;context.repository_root=fs::current_path();context.cadence_ms=10;context.history_spacing_ticks=1;
    context.channel_ids=id.channel_ids;context.feature_names=id.feature_names;context.embedding_export_layout=id.export_layout;context.availability_policy=id.availability_policy;
    torch::manual_seed(781);pt::Trainer trainer(config,identity);
    const auto predictive_path=temporary.path/"predictive.pt";pa::save_inference(trainer,predictive_path,context);
    const auto predictive_pin=av::sha256_file(av::sidecar_path(predictive_path));predictive=av::read_verified(predictive_path,predictive_pin,"inference");
    serving=std::make_shared<const pa::CheckedInference>(pa::load_inference(predictive_path,predictive_pin,config,identity,context));
    contract=fi::make_contract(timeline,spec,{900,1150,1150,1150},encoder,predictive);
  }
  fi::Issuer issuer()const{return {contract,serving,encoder,predictive,timeline};}
};

void timing_and_inputs(Fixture& f) {
  const auto issuer=f.issuer();const auto accepted=issuer.issue(f.timeline,f.origin,f.roles);
  check(accepted.valid && accepted.reasons==0 && accepted.history.size()==4,"all exact histories issue after fitting");
  std::vector<double> flat;for(size_t n=17;n<=20;++n)flat.insert(flat.end(),f.timeline.rows[n].values.begin(),f.timeline.rows[n].values.end());
  const auto expected=f.serving->forecast(torch::tensor(flat,torch::kFloat64).reshape({1,4,3}))[0];
  check(torch::equal(accepted.values,expected),"issued values equal the exact checked frozen model");
  check(accepted.history.back().available_ms==f.origin.decision_ms,"availability equal to decision is admitted");
  auto origin=f.origin;origin.endpoint_ms=1150;
  auto issue=issuer.issue(f.timeline,origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::online_not_before_origin) && (issue.reasons&fi::target_not_before_origin) &&
    (issue.reasons&fi::dynamics_not_before_origin),"all joint component fit equalities are unavailable despite later decision");
  origin.endpoint_ms=1140;issue=issuer.issue(f.timeline,origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::dynamics_not_before_origin),"fit after origin but before delayed decision remains unavailable");
  origin.endpoint_ms=900;issue=issuer.issue(f.timeline,origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::encoder_not_before_origin),"Encoder fit equality is unavailable");
  origin=f.origin;origin.decision_ms=1199;issue=issuer.issue(f.timeline,origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::decision_before_origin),"decision before origin retained");
  auto altered=f.timeline;altered.rows[20].available_ms=f.origin.decision_ms+1;
  issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::input_unavailable) && issue.history.size()==4,"late input availability retained with every history slot");
  altered=f.timeline;altered.rows.erase(altered.rows.begin()+18);issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::missing_input) && !issue.history[1].row_index && issue.history[1].endpoint_ms==1180,
    "missing input retains exact endpoint without neighboring substitution");
  altered=f.timeline;altered.rows[17].sample_valid=false;altered.rows[18].support.observed[0]=0;altered.rows[19].available_ms+=1000;
  issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::invalid_input) && (issue.reasons&fi::input_support) &&
    (issue.reasons&fi::input_unavailable),"all invalid input reasons are retained together");
  check(torch::isfinite(issue.values).all().item<bool>() && issue.values.count_nonzero().item<int64_t>()==0,"invalid predictions retain finite zero storage");
  altered=f.timeline;altered.rows[17].support.timestamps_ms[0]-=1;issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::input_timestamp_gap),"actual input support grid independently checked");
  altered=f.timeline;altered.rows[18].role=lm::Role::validation;issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::input_role_boundary),"input role independently checked");
  issue=issuer.issue(f.timeline,f.origin,{{lm::Role::train,1160,2000}});
  check(!issue.valid && (issue.reasons&fi::input_role_boundary),"full input support must stay inside role");
  altered=f.timeline;altered.rows[19].window_id="substituted window";issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::input_identity_mismatch),"input semantic window identity independently checked");
  altered=f.timeline;altered.rows[19].support.observed.pop_back();issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::input_shape_mismatch),"malformed input support safely retained");
  altered=f.timeline;altered.rows[19].values[0]=std::numeric_limits<double>::quiet_NaN();issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::input_nonfinite),"nonfinite actual input retained without serving");
  issue=issuer.issue(f.timeline,f.origin,f.roles,lm::input_support);
  check(!issue.valid && issue.original_input_reasons==lm::input_support && (issue.reasons&fi::ledger_input_invalid),"original input-only ledger reasons preserved");
  altered=f.timeline;altered.rows[19].support.observed[1]=2;issue=issuer.issue(altered,f.origin,f.roles);
  check(!issue.valid && (issue.reasons&fi::input_support),"nonbinary actual support refused");
  check(!fi::reason_names(issue.reasons).empty(),"issuance reasons have stable serializable names");
}

void target_blindness(Fixture& f) {
  const auto issuer=f.issuer();const auto before=issuer.issue(f.timeline,f.origin,f.roles);auto future=f.timeline;
  for(auto& row:future.rows)if(row.endpoint_ms>f.origin.endpoint_ms) {
    row.values.assign(3,std::numeric_limits<double>::quiet_NaN());row.sample_valid=false;
    row.support.observed.clear();row.support.timestamps_ms.clear();row.available_ms=std::numeric_limits<int64_t>::max();row.window_id="future-only changed";
  }
  const auto changed=issuer.issue(future,f.origin,f.roles);
  check(changed.valid && changed.reasons==before.reasons && torch::equal(changed.values,before.values),
        "future values, masks, eligibility, identities, availability and malformed support cannot affect serving");
  future=f.timeline;future.rows.erase(future.rows.begin()+21,future.rows.end());
  check(torch::equal(issuer.issue(future,f.origin,f.roles).values,before.values),"absent future targets do not suppress issued forecasts");
  auto pairs=lm::build_pairs(f.timeline,{f.origin},f.spec,f.roles);
  check(!lm::targets_available_by(f.timeline,pairs.rows.front(),1202) && before.valid,"unmatured targets prevent fit/scoring support independently of issuance");
  check(lm::targets_available_by(f.timeline,pairs.rows.front(),1232),"mature full-support targets become separately available");
  auto editable=before.values;editable.fill_(9e9);
  check(torch::equal(issuer.issue(f.timeline,f.origin,f.roles).values,changed.values),"returned forecast mutation cannot alter frozen parent state");
}

void identity_rejections(Fixture& f) {
  auto changed_timeline=f.timeline;changed_timeline.identity.encoder_fit_cutoff_ms=800;
  rejects([&]{fi::make_contract(changed_timeline,f.spec,{800,1150,1150,1150},f.encoder,f.predictive);},
          "Encoder fit cannot be relabeled earlier than authenticated captured timeline");
  changed_timeline=f.timeline;changed_timeline.identity.dataset_id+="-other";
  rejects([&]{fi::make_contract(changed_timeline,f.spec,f.contract.fits,f.encoder,f.predictive);},
          "capture dataset metadata must match immutable archive source bytes");
  auto changed=f.contract;changed.fits.online_ms=1100;
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"independent representation fold rejected");
  changed=f.contract;changed.fits.online_ms=changed.fits.target_ms=changed.fits.dynamics_ms=1100;
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"lying about common fit cutoff rejected by immutable archive split");
  changed=f.contract;changed.stage_kind="dynamics-cross-fit";
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"unregistered Dynamics cross-fit rejected");
  changed=f.contract;changed.policy_sha256=av::sha256_bytes("unregistered changed policy");
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"changed policy rejected");
  changed=f.contract;changed.target_coordinate_id="same dimensions but independent target fold";
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"coordinate pooling rejected");
  changed=f.contract;changed.bundle_ref=av::sha256_bytes("other bundle");
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"changed BundleRef rejected");
  changed=f.contract;changed.horizons_ticks={3,1};
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"same-width horizon-order substitution rejected");
  changed=f.contract;changed.predictive_sidecar_sha256=av::sha256_bytes("other valid sidecar");changed.bundle_ref=av::bundle_ref(fi::bundle_selection(changed));
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"different sidecar cannot be substituted under an internally consistent BundleRef");
  changed=f.contract;changed.online.state_sha256=av::sha256_bytes("different online state");changed.online.model_ref=av::derive_model_ref(changed.online);
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"stale Dynamics online parent rejected");
  changed=f.contract;changed.target.state_sha256=av::sha256_bytes("different target state");changed.target.model_ref=av::derive_model_ref(changed.target);
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"stale Dynamics target parent rejected");
  auto encoder=f.encoder;encoder.bytes+='!';
  rejects([&]{fi::Issuer i(f.contract,f.serving,encoder,f.predictive,f.timeline);},"changed verified Encoder archive bytes rejected");
  auto predictive=f.predictive;predictive.bytes+='!';
  rejects([&]{fi::Issuer i(f.contract,f.serving,f.encoder,predictive,f.timeline);},"changed verified predictive archive bytes rejected");
  predictive=f.predictive;predictive.sidecar_sha256=av::sha256_bytes("changed sidecar bytes");
  rejects([&]{fi::Issuer i(f.contract,f.serving,f.encoder,predictive,f.timeline);},"changed verified sidecar pin rejected");
  predictive=f.predictive;predictive.manifest.archive_kind="training";predictive.manifest.resume_identity_sha256=av::sha256_bytes("resume");
  rejects([&]{fi::Issuer i(f.contract,f.serving,f.encoder,predictive,f.timeline);},"training artifact cannot enter inference issuance");
  auto timeline=f.timeline;timeline.identity.dataset_id="changed capture";
  rejects([&]{f.issuer().issue(timeline,f.origin,f.roles);},"changed timeline identity rejected");
  changed=f.contract;changed.timeline_sha256=av::sha256_bytes("another captured timeline");
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"archive source provenance cannot be substituted");
  changed=f.contract;changed.pairs_sha256=av::sha256_bytes("another pair ledger");
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"archive ledger provenance cannot be substituted");
  changed=f.contract;changed.embedding_transform_sha256=av::sha256_bytes("another transform");
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"frozen transform provenance cannot be substituted");
  changed=f.contract;changed.reference_scale_sha256=av::sha256_bytes("another reference scale");
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"frozen reference provenance cannot be substituted");
  changed=f.contract;changed.timeline.encoder_fit_cutoff_ms=800;changed.fits.encoder_ms=800;
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"forged contract cannot relabel captured Encoder fit");
  changed=f.contract;changed.timeline.dataset_id+="-other";
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"forged contract cannot relabel capture metadata");
  changed_timeline=f.timeline;changed_timeline.identity.encoder_fit_cutoff_ms=800;
  changed=f.contract;changed.timeline=changed_timeline.identity;changed.fits.encoder_ms=800;
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,changed_timeline);},"forged contract and capture together still fail immutable source hash");
  changed=f.contract;changed.spacing_ticks=2;
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"history spacing must match the checked mapping");
  changed=f.contract;changed.interval_ms=20;
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"cadence must match the checked mapping");
  changed=f.contract;changed.timeline.channel_ids={8};
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"semantic channel IDs must match the checked mapping");
  changed=f.contract;changed.timeline.feature_names={"volume"};
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"semantic feature names must match the checked mapping");
  changed=f.contract;changed.timeline.export_layout="same width but different layout";
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"export layout must match the checked mapping");
  changed=f.contract;changed.timeline.availability_policy="different availability rule";
  rejects([&]{fi::Issuer i(changed,f.serving,f.encoder,f.predictive,f.timeline);},"availability policy must match the checked mapping");
  rejects([&]{f.issuer().issue(f.timeline,f.origin,f.roles,lm::invalid_target);},"target-side ledger reasons cannot leak into serving");
  torch::manual_seed(99);pt::Trainer other(f.config,f.identity);
  {torch::NoGradGuard guard;other.online->affine->weight.add_(.5);}
  const auto other_path=f.temporary.path/"different-predictive.pt";pa::save_inference(other,other_path,f.context);
  auto spoofed=av::read_verified(other_path,av::sha256_file(av::sidecar_path(other_path)),"inference");
  spoofed.manifest.components=f.predictive.manifest.components;
  spoofed.sidecar_sha256=av::sha256_bytes(av::serialize_manifest(spoofed.manifest));
  auto spoofed_contract=fi::make_contract(f.timeline,f.spec,f.contract.fits,f.encoder,spoofed);
  rejects([&]{fi::Issuer i(spoofed_contract,f.serving,f.encoder,spoofed,f.timeline);},"actual changed archive tensors cannot be hidden behind original serving ModelRefs");
  check(av::is_sha256(fi::contract_digest(f.contract)) && av::is_sha256(fi::source_closure_sha256(fs::current_path())),"complete contract and source closure digests available");
  changed=f.contract;changed.timeline.dataset_id+="-other";
  check(fi::contract_digest(changed)!=fi::contract_digest(f.contract),"contract digest binds complete capture metadata");
  changed=f.contract;changed.input_support.maximum_missing_run_ticks=0;
  check(fi::contract_digest(changed)!=fi::contract_digest(f.contract),"contract digest binds actual support admission recipe");
  const auto shared=f.serving;f.serving.reset();
  fi::Issuer owning(f.contract,shared,f.encoder,f.predictive,f.timeline);
  check(owning.issue(f.timeline,f.origin,f.roles).valid,"issuer owns guarded inference lifetime");f.serving=shared;
}

void publication_roundtrip(Fixture& f) {
  const auto directory=f.temporary.path/"publication";fs::create_directory(directory);
  const std::vector<lm::PairOrigin> origins{f.origin,{"early",1150,1202,lm::Role::train,"synthetic-block"},
    {"without-targets",1400,1402,lm::Role::train,"synthetic-block"},
    {"late-input",1200,1201,lm::Role::train,"synthetic-block"}};
  const auto pairs=lm::build_pairs(f.timeline,origins,f.spec,f.roles);const auto issuer=f.issuer();
  std::vector<fi::IssuedOrigin> issued;
  for(size_t n=0;n<origins.size();++n)issued.push_back(issuer.issue(f.timeline,origins[n],f.roles,pairs.rows[n].input_reasons));
  ev::PredictionSet prediction;prediction.coordinate_id=f.contract.target_coordinate_id;prediction.method="state-causal-forecast-issuance-v1";
  prediction.fit_cutoff_ms=f.contract.fits.dynamics_ms;prediction.history_size=f.spec.history_steps;prediction.spacing_ticks=f.spec.spacing_ticks;
  prediction.associations=lm::associations(f.timeline,pairs);const auto count=static_cast<int64_t>(prediction.associations.size());
  prediction.values=torch::zeros({count,static_cast<int64_t>(f.contract.state_width)},torch::kFloat64);
  prediction.valid=torch::zeros({count},torch::kBool);prediction.reasons=torch::zeros({count},torch::kInt64);
  for(size_t n=0;n<issued.size();++n)for(size_t q=0;q<f.spec.horizons_ticks.size();++q) {
    const auto offset=static_cast<int64_t>(n*f.spec.horizons_ticks.size()+q);prediction.values[offset].copy_(issued[n].values[q]);
    prediction.valid[offset]=issued[n].valid;prediction.reasons[offset]=fi::evaluation_reasons(issued[n].reasons);
  }
  const auto pin=fi::publish_issuance(directory,f.contract,pairs,issued,prediction);
  const auto loaded=fi::load_issued_predictions(directory,pin,f.contract);
  check(torch::equal(loaded.values,prediction.values) && torch::equal(loaded.valid,prediction.valid) &&
        torch::equal(loaded.reasons,prediction.reasons) && loaded.associations.size()==prediction.associations.size(),
        "issued forecasts and every invalid origin/horizon round-trip through pinned publication");
  check(!issued[1].valid && issued[2].valid && !issued[3].valid && loaded.associations.size()==8,
        "publication retains unavailable fit, absent future targets and unavailable input associations");
  rejects([&]{fi::publish_issuance(directory,f.contract,pairs,issued,prediction);},"issuance publication cannot overwrite an existing ledger or manifest");
  rejects([&]{fi::load_issued_predictions(directory,av::sha256_bytes("stale manifest"),f.contract);},"stale issuance manifest pin rejected");
  auto contract=f.contract;contract.timeline.dataset_id+="-different";
  rejects([&]{fi::load_issued_predictions(directory,pin,contract);},"stale complete frozen issuance contract rejected");
  const auto corrupt=[&](const std::string& file,const std::string& suffix) {
    const auto clone=f.temporary.path/("publication-"+suffix);fs::copy(directory,clone,fs::copy_options::recursive);
    {std::ofstream out(clone/file,std::ios::app|std::ios::binary);out<<'!';}
    rejects([&]{fi::load_issued_predictions(clone,pin,f.contract);},"changed publication "+file+" rejected");
  };
  corrupt("state-predictions.pt","prediction");corrupt("state-issuance.csv","ledger");corrupt("state-issuance.json","manifest");
}
}
int main()try {
  torch::set_num_threads(1);torch::set_num_interop_threads(1);Fixture fixture;
  timing_and_inputs(fixture);target_blindness(fixture);identity_rejections(fixture);publication_roundtrip(fixture);
  std::cout<<checks<<" forecast issuance v1 synthetic checks passed; no market fit.\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
