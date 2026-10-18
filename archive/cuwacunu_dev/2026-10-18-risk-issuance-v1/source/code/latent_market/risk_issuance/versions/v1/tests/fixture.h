// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/latent_market/risk_issuance/v1/stage.h"
#include <fstream>

namespace cuwacunu::latent_market::risk_issuance::v1::test_fixture {
namespace pt=cuwacunu::networks::predictive_training::v1;
namespace pa=pt::artifact;
inline constexpr int64_t base=1800000000000LL,interval=60000;
inline int64_t at(int64_t tick){return base+tick*interval;}
inline std::string sidecar_pin(const fs::path& path){return av::sha256_file(av::sidecar_path(path));}
inline void write(const fs::path& path,const std::string& text){lm::ledger_detail::publish(path,[&](auto& out){out<<text;});}
struct Fixture {
  fs::path directory,issuance_directory;
  lm::EmbeddingTimeline timeline;
  lm::DynamicsPairs pairs;
  fi::FrozenContract contract;
  std::shared_ptr<const pa::CheckedInference> serving;
  std::unique_ptr<NativeForecast> native;
  std::string issuance_pin;
  av::VerifiedArtifact encoder,predictive;
  pt::Config config;
  pt::Identity identity;
  pa::Context context;
  dc::Config native_config;
  dc::Identity native_identity;
  da::Context native_context;
  lm::FutureRiskTargets future_risk;

  explicit Fixture(fs::path root,bool create=true,double evaluation_shift=0):directory(fs::absolute(root)),issuance_directory(directory/"issued") {
    if(create){TORCH_CHECK(fs::create_directories(directory),"synthetic fixture leaf already exists");generate(evaluation_shift);}
    else verify_inventory();
    load();
  }
  fi::Issuer issuer()const{return {contract,serving,encoder,predictive,timeline};}
  StageContract stages()const {
    StageContract stage;stage.train_begin_ms=at(120);stage.train_end_ms=at(239);stage.head_fit_ms=at(239)+5;
    stage.evaluation_begin_ms=at(240);stage.evaluation_end_ms=at(319);stage.scoring_cutoff_ms=at(319)+5;return stage;
  }
  void configure() {
    context={encoder.manifest.components.front(),encoder.manifest.archive_sha256,fs::current_path(),interval,1,
      timeline.identity.channel_ids,timeline.identity.feature_names,timeline.identity.export_layout,timeline.identity.availability_policy};
    config.state.embedding_width=4;config.state.state_width=4;config.state.identity_initialization=true;
    TORCH_CHECK(torch::cuda::is_available(),"synthetic predictive parent requires existing managed CUDA");
    config.state.device=torch::Device(torch::kCUDA,0);config.state.dtype=torch::kFloat64;
    config.dynamics.state_width=4;config.dynamics.history_length=4;config.dynamics.horizons={1,4};
    config.dynamics.device=config.state.device;config.dynamics.dtype=config.state.dtype;config.dynamics.residual=true;
    config.sampler_seed=29;config.clip_norm=1;config.variance_floor=.1;
    const auto transform=av::sha256_file(directory/"transform.txt");
    identity={timeline.identity.checkpoint_hash,timeline.identity.scaler_hash,transform,transform,
      fi::timeline_digest(timeline),"fit:"+std::to_string(at(119)+5)+";ledger:"+canonical_pairs_sha256(pairs),
      canonical_pairs_sha256(pairs),"artificial-fixed-minibatch-v1","SYNTHETIC two-update parent fixture","frozen-target-state"};
    native_config.width=4;native_config.history_size=4;native_config.hidden_width=8;native_config.horizons={1,4};
    native_config.updates=2;native_config.microbatch=4;native_config.repetition_seed=17;
    native_config.fit_cutoff_ms=at(119)+5;native_config.train_begin_ms=at(1);native_config.train_end_ms=at(119);
    const auto protocol=av::sha256_file(directory/"synthetic-card.txt");
    native_identity={timeline.identity.artifact_id+":native-e",identity.source,"fit:"+std::to_string(at(119)+5),identity.pair_ledger,context.encoder.model_ref,protocol};
    native_context={context.encoder,context.encoder_archive_sha256,fs::current_path(),protocol,
      av::digest_fields("latent-state-control-population-v1",{native_identity.source,native_identity.pair_ledger,std::to_string(native_config.fit_cutoff_ms)}),
      av::sha256_bytes(native_identity.coordinate_id),native_config.fit_cutoff_ms,interval,1,
      timeline.identity.channel_ids,timeline.identity.feature_names,timeline.identity.export_layout,timeline.identity.availability_policy};
  }
  void generate(double evaluation_shift) {
    write(directory/"SYNTHETIC.txt","SYNTHETIC-RISK-ISSUANCE-V1; artificial staged fixture; no market or Encoder execution.\n");
    write(directory/"synthetic-card.txt","SYNTHETIC staged risk fixture H4/K4 D4 h1/4 R4; parent2updates and checked direct MLP2updates.\n");
    write(directory/"encoder.pt","SYNTHETIC Encoder ancestry text; no Encoder implementation call.\n");
    const auto ec=av::make_component({"Encoder","v3",av::sha256_bytes("synthetic Encoder source")},av::sha256_bytes("synthetic weights"),av::sha256_bytes("synthetic mapping"));
    av::publish_sidecar(directory/"encoder.pt",{"inference",av::sha256_file(directory/"encoder.pt"),"",{ec}});
    encoder=av::read_verified(directory/"encoder.pt",sidecar_pin(directory/"encoder.pt"),"inference");
    auto& id=timeline.identity;id.artifact_id="synthetic-risk-stage-v1";id.dataset_id="SYNTHETIC-RISK-ISSUANCE-V1";
    id.encoder_family="synthetic-fixture-no-encoder-execution";id.encoder_version="v3";id.checkpoint_hash=encoder.manifest.archive_sha256;
    id.scaler_hash=av::sha256_bytes("synthetic frozen scaler");id.encoder_config_hash=av::sha256_bytes("synthetic D4/H4 config");
    id.source_hash=av::sha256_bytes("synthetic finite close source-v1");id.cache_hash=id.source_hash;
    id.export_layout="synthetic-native4";id.availability_policy="SIMULATED endpoint+5ms";
    id.source_closure_hash=av::sha256_bytes("synthetic risk fixture source closure");id.executable_hash=av::sha256_file("/proc/self/exe");
    id.encoder_fit_cutoff_ms=at(0);id.channel_ids={0,1,2};id.feature_names={"close"};
    timeline.interval_ms=interval;timeline.window_ticks=4;timeline.embedding_width=4;
    for(int64_t tick=4;tick<=319;++tick){lm::EmbeddingRow row;row.window_id=lm::expected_window_id(id,at(tick));
      row.endpoint_ms=at(tick);row.available_ms=at(tick)+5;row.role=tick<=119?lm::Role::train:lm::Role::validation;row.sample_valid=true;
      row.support.channels=3;row.support.features=1;row.support.observed.assign(12,1);
      for(auto past=tick-3;past<=tick;++past)row.support.timestamps_ms.push_back(at(past));
      for(int64_t k=0;k<4;++k)row.values.push_back(std::sin(tick*(.04+k*.013))+.2*std::cos(tick*.031+k)+(tick>=240?evaluation_shift:0));
      timeline.rows.push_back(row);}
    lm::PairSpec spec;spec.history_steps=4;spec.spacing_ticks=1;spec.horizons_ticks={1,4};spec.future_label_ticks={4};
    const std::vector<lm::RoleInterval> roles{{lm::Role::train,at(1),at(119)},{lm::Role::validation,at(120),at(319)}};
    std::vector<lm::PairOrigin> origins;for(int64_t tick=7;tick<=318;++tick)origins.push_back({"risk-origin-"+std::to_string(tick),at(tick),at(tick)+5,tick<=119?lm::Role::train:lm::Role::validation,"synthetic-block-"+std::to_string(tick/16)});
    pairs=lm::build_pairs(timeline,origins,spec,roles);lm::publish_timeline(directory/"timeline.bin",timeline);lm::publish_pairs(directory/"pairs.bin",pairs);
    std::vector<lm::AssetCloseSeries> closes;for(int64_t channel=0;channel<3;++channel){lm::AssetCloseSeries s;s.channel_id=channel;
      for(int64_t tick=1;tick<=319;++tick)s.rows.push_back({at(tick),at(tick)+5,100*std::exp(.001*std::sin(tick*.13+channel)+.00001*tick*(channel+1)),true});
      closes.push_back(s);}
    future_risk=lm::build_future_risk_targets(id.artifact_id,origins,closes,interval,{4},roles,id.source_hash);
    lm::publish_risk_targets(directory/"future-risk.bin",future_risk);
    const auto bank=lm::array_bank(timeline);std::vector<int64_t> fit_indices;for(size_t j=0;j<timeline.rows.size();++j)
      if(timeline.rows[j].role==lm::Role::train&&timeline.rows[j].support.timestamps_ms.front()>=at(1))fit_indices.push_back(j);
    const auto train=bank.values.index_select(0,torch::tensor(fit_indices,torch::kInt64));
    const auto mean=train.mean(0),scale=train.var(0,false).sqrt().clamp_min(1e-8),reference=torch::ones({4},torch::kFloat64);
    std::ostringstream transform;transform<<std::setprecision(17);for(const auto& tensor:{mean,scale,reference}){for(int64_t j=0;j<tensor.numel();++j)transform<<tensor[j].item<double>()<<' ';transform<<'\n';}
    write(directory/"transform.txt",transform.str());configure();torch::manual_seed(3319);pt::Trainer trainer(config,identity,mean,scale,reference);
    const auto index=[&](int64_t tick){return tick-4;};
    auto history=torch::zeros({4,4,4},torch::kFloat64),future=torch::zeros({4,2,4},torch::kFloat64);
    for(int64_t j=0;j<4;++j){const auto tick=20+j*10;for(int64_t k=0;k<4;++k)history[j][k].copy_(bank.values[index(tick-3+k)]);
      future[j][0].copy_(bank.values[index(tick+1)]);future[j][1].copy_(bank.values[index(tick+4)]);}
    const auto pool=train.narrow(0,0,8);const auto pool_ids=torch::arange(1,9,torch::kInt64);
    for(int j=0;j<2;++j)TORCH_CHECK(trainer.step({history.to(config.state.device),future.to(config.state.device),
      torch::ones({4,2},torch::TensorOptions().dtype(torch::kBool).device(config.state.device)),
      {pool.to(config.state.device),pool_ids.to(config.state.device)}}).updated,"synthetic parent update skipped");
    TORCH_CHECK(trainer.progress.completed_updates==2,"synthetic parent budget differs");
    pa::save_training(trainer,directory/"parent-training.pt",context);pa::save_inference(trainer,directory/"predictive.pt",context);
    predictive=av::read_verified(directory/"predictive.pt",sidecar_pin(directory/"predictive.pt"),"inference");
    serving=std::make_shared<const pa::CheckedInference>(pa::load_inference(directory/"predictive.pt",predictive.sidecar_sha256,config,identity,context));
    contract=fi::make_contract(timeline,pairs.spec,{at(0),at(119)+5,at(119)+5,at(119)+5},encoder,predictive);
    fs::create_directory(issuance_directory);const auto issue=issuer();std::vector<fi::IssuedOrigin> issued;
    auto prediction=ev::PredictionSet{};prediction.coordinate_id=contract.target_coordinate_id;prediction.method="state-causal-forecast-issuance-v1";
    prediction.fit_cutoff_ms=at(119)+5;prediction.history_size=4;prediction.spacing_ticks=1;prediction.associations=lm::associations(timeline,pairs);
    const auto n=prediction.associations.size();prediction.values=torch::zeros({static_cast<int64_t>(n),4},torch::kFloat64);
    prediction.valid=torch::zeros({static_cast<int64_t>(n)},torch::kBool);prediction.reasons=torch::zeros({static_cast<int64_t>(n)},torch::kInt64);
    for(size_t j=0;j<pairs.rows.size();++j){const auto& p=pairs.rows[j];issued.push_back(issue.issue(timeline,{p.pair_id,p.endpoint_ms,p.decision_ms,p.role,p.dependence_group},pairs.roles,p.input_reasons));
      for(size_t q=0;q<2;++q){prediction.values[j*2+q].copy_(issued.back().values[q]);prediction.valid[j*2+q]=issued.back().valid;prediction.reasons[j*2+q]=fi::evaluation_reasons(issued.back().reasons);}}
    issuance_pin=fi::publish_issuance(issuance_directory,contract,pairs,issued,prediction);
    dc::Trainer direct(native_config,native_identity,bank,lm::associations(timeline,pairs));direct.train();
    da::save_training(direct,directory/"native-training.pt",native_context);const auto dm=da::save_inference(direct,directory/"native.pt",native_context);
    const auto checked=da::load_inference(directory/"native.pt",sidecar_pin(directory/"native.pt"),dm.component.model_ref,native_config,native_identity,native_context);
    ev::save_predictions((directory/"native-predictions.pt").string(),checked.predict(bank,lm::associations(timeline,pairs)));
    const auto comparison=da::make_comparison(native_context.protocol_sha256,native_context.population_sha256,native_context.coordinates_sha256,context.encoder.model_ref,
      {{"residual-mlp",dm.component.model_ref,sidecar_pin(directory/"native.pt"),av::sha256_file(directory/"native-predictions.pt")}});
    da::publish_comparison(directory/"native-comparison.json",comparison);
    std::ostringstream inventory;inventory<<"# SYNTHETIC-RISK-ISSUANCE-V1 immutable fixture pins\n";
    for(const auto& entry:fs::recursive_directory_iterator(directory))if(entry.is_regular_file())inventory<<av::sha256_file(entry.path())<<"  "<<fs::relative(entry.path(),directory).generic_string()<<'\n';
    write(directory/"fixture.sha256",inventory.str());
  }
  void verify_inventory()const {
    std::ifstream marker(directory/"SYNTHETIC.txt");std::string text;std::getline(marker,text);
    TORCH_CHECK(text=="SYNTHETIC-RISK-ISSUANCE-V1; artificial staged fixture; no market or Encoder execution.","not an explicitly synthetic fixture");
    std::ifstream inventory(directory/"fixture.sha256");TORCH_CHECK(bool(inventory),"missing synthetic fixture inventory");
    std::getline(inventory,text);TORCH_CHECK(text=="# SYNTHETIC-RISK-ISSUANCE-V1 immutable fixture pins","wrong fixture inventory schema");
    std::set<std::string> seen;
    while(std::getline(inventory,text)) {
      TORCH_CHECK(text.size()>66&&text.substr(64,2)=="  ","malformed fixture pin");
      const auto hash=text.substr(0,64),name=text.substr(66);const fs::path relative(name);
      TORCH_CHECK(av::is_sha256(hash)&&!relative.is_absolute()&&relative.lexically_normal()==relative&&
        *relative.begin()!=".."&&seen.insert(name).second,"noncanonical fixture pin");
      TORCH_CHECK(av::sha256_file(directory/relative)==hash,"changed synthetic fixture input: ",name);
    }
    TORCH_CHECK(!seen.empty(),"empty synthetic fixture inventory");
  }
  void load() {
    verify_inventory();timeline=lm::load_timeline(directory/"timeline.bin");pairs=lm::load_pairs(directory/"pairs.bin",timeline);
    future_risk=lm::load_risk_targets(directory/"future-risk.bin");
    encoder=av::read_verified(directory/"encoder.pt",sidecar_pin(directory/"encoder.pt"),"inference");configure();
    predictive=av::read_verified(directory/"predictive.pt",sidecar_pin(directory/"predictive.pt"),"inference");
    serving=std::make_shared<const pa::CheckedInference>(pa::load_inference(directory/"predictive.pt",predictive.sidecar_sha256,config,identity,context));
    contract=fi::make_contract(timeline,pairs.spec,{at(0),at(119)+5,at(119)+5,at(119)+5},encoder,predictive);
    issuance_pin=av::sha256_file(issuance_directory/"state-issuance.json");(void)fi::load_issued_predictions(issuance_directory,issuance_pin,contract);
    const auto dm=da::parse_manifest(av::detail::read_bytes(av::sidecar_path(directory/"native.pt"),262144));
    native=std::make_unique<NativeForecast>(NativeForecast::load(contract,timeline,pairs,directory/"native.pt",sidecar_pin(directory/"native.pt"),dm.component.model_ref,
      native_config,native_identity,native_context,directory/"native-comparison.json",av::sha256_file(directory/"native-comparison.json"),"residual-mlp",directory/"native-predictions.pt"));
  }
  ev::ObservableTargets targets(const FeatureBank&)const {
    ev::ObservableTargets result;result.identity=av::sha256_file(directory/"future-risk.bin")+":"+future_risk.formula_id;
    const auto n=static_cast<int64_t>(pairs.rows.size());result.origins.row_ids=torch::arange(1,n+1,torch::kInt64);
    result.origins.roles=torch::zeros({n},torch::kInt64);result.origins.decision_ms=torch::zeros({n},torch::kInt64);
    result.values=torch::zeros({n,6},torch::kFloat64);result.valid=torch::zeros({n,6},torch::kBool);
    result.available_ms=torch::full({n,6},INT64_MAX,torch::kInt64);result.support_begin_ms=torch::zeros({n,6},torch::kInt64);result.support_end_ms=torch::zeros({n,6},torch::kInt64);
    for(int64_t channel=0;channel<3;++channel)for(const auto& label:{std::string("dispersion"),std::string("downside")}){
      result.names.push_back(label+":asset="+std::to_string(channel)+":h=4");result.units.push_back(label=="dispersion"?future_risk.dispersion_units:future_risk.downside_units);}
    std::map<std::string,int64_t> ids;for(int64_t j=0;j<n;++j){const auto& p=pairs.rows[j];ids[p.pair_id]=j;
      result.origins.roles[j]=static_cast<int64_t>(p.endpoint_ms>=at(120)&&p.endpoint_ms<=at(239)?ev::Role::train:ev::Role::validation);
      result.origins.decision_ms[j]=p.decision_ms;for(int64_t k=0;k<6;++k){result.support_begin_ms[j][k]=p.endpoint_ms;result.support_end_ms[j][k]=p.endpoint_ms+4*interval;}}
    for(const auto& row:future_risk.rows){const auto j=ids.at(row.pair_id),k=row.channel_id*2;
      result.values[j][k]=row.dispersion_1m;result.values[j][k+1]=row.downside;
      for(int64_t q=0;q<2;++q){result.valid[j][k+q]=row.eligible;result.available_ms[j][k+q]=row.available_ms.value_or(INT64_MAX);}}
    return result;
  }
};
}
