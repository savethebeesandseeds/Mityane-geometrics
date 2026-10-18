// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include "cuwacunu/config/document.h"
#include "cuwacunu/latent_market/artifacts.h"
#include "cuwacunu/latent_market/ledger_io.h"
#include "cuwacunu/latent_market/arrays.h"
#include "cuwacunu/networks/predictive_training/v1/training.h"
#include "cuwacunu/networks/predictive_training/v1/serving.h"
#include "cuwacunu/networks/predictive_training/v1/artifact.h"
#include "cuwacunu/latent_market/forecast_issuance/v1/publication.h"
#include <iostream>
#include <set>

namespace lm=cuwacunu::latent_market::v1;
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace pt=cuwacunu::networks::predictive_training::v1;
namespace av=cuwacunu::artifacts::v1;
namespace fi=cuwacunu::latent_market::forecast_issuance::v1;
namespace sr=cuwacunu::networks::state_representation::v1;
namespace fs=std::filesystem;

namespace {
void verify_capture_binding(const fs::path& receipt,const std::string& timeline_sha256,
    const std::string& encoder_sidecar,const std::string& encoder_model_ref) {
  std::ifstream in(receipt);std::string line;
  TORCH_CHECK(bool(in)&&bool(std::getline(in,line)) &&
      (line=="latent-market-preparation-v1" || line=="latent-market-synthetic-capture-v1"),
      "A typed capture receipt is required to establish historical Encoder lineage");
  std::map<std::string,std::string> fields;
  while(std::getline(in,line)) {
    const auto space=line.find(' ');
    TORCH_CHECK(space!=std::string::npos && space>0 && space+1<line.size() &&
        fields.emplace(line.substr(0,space),line.substr(space+1)).second,"Malformed or duplicate capture receipt field");
  }
  TORCH_CHECK(in.eof(),"Capture receipt read failed");
  for(const auto& key:{"timeline_sha256","encoder_sidecar_sha256","encoder_model_ref"})
    TORCH_CHECK(fields.contains(key),"Capture receipt missing required binding: ",key);
  TORCH_CHECK(fields.at("timeline_sha256")==timeline_sha256 &&
      fields.at("encoder_sidecar_sha256")==encoder_sidecar &&
      fields.at("encoder_model_ref")==encoder_model_ref,
      "Captured embeddings require their original Encoder ModelRef and sidecar; archive hash alone is insufficient");
}
ev::PredictionSet role_subset(const ev::PredictionSet& p,const lm::DynamicsPairs& pairs,lm::Role role) {
  auto out=p;out.associations.clear();std::vector<int64_t> indices;
  for(size_t n=0;n<p.associations.size();++n)if(pairs.rows.at((p.associations[n].pair_id-1)/pairs.spec.horizons_ticks.size()).role==role){indices.push_back(n);out.associations.push_back(p.associations[n]);}
  const auto select=torch::tensor(indices,torch::kInt64);
  out.values=p.values.index_select(0,select);out.valid=p.valid.index_select(0,select);out.reasons=p.reasons.index_select(0,select);return out;
}
void scores(std::ofstream& out,const ev::ArrayBank& bank,const lm::DynamicsPairs& pairs,const ev::PredictionSet& prediction,const torch::Tensor& reference,int64_t cutoff) {
  for(auto role:{lm::Role::train,lm::Role::validation}) {
    const auto p=role_subset(prediction,pairs,role);
    if(p.associations.empty())continue;
    const auto report=ev::score_predictions(bank,p,{reference,torch::ones_like(reference),1e-12},cutoff);
    const auto line=[&](const ev::Score& s,const std::string& support,const std::string& overlap){
      out<<std::quoted(bank.coordinate_id)<<','<<std::quoted(p.method)<<','<<p.history_size<<','<<lm::role_name(role)<<','<<s.horizon_ticks<<','<<support<<','<<overlap<<','
         <<s.total<<','<<s.input_valid<<','<<s.target_valid<<','<<s.scored<<','<<s.coverage<<',';
      if(s.supported)out<<s.normalized_mse<<','<<s.native_mae<<','<<s.persistence_normalized_mse;else out<<"N/A,N/A,N/A";
      out<<',';if(s.skill_supported)out<<s.skill;else out<<"N/A";out<<'\n';
    };
    for(const auto& s:report.horizons)line(s,"all","all");
    for(const auto& s:report.strata)line(s.score,s.support,s.overlap);
  }
}
void baseline_suite(const fs::path& output,const ev::ArrayBank& bank,const std::vector<ev::Association>& associations,
    const lm::DynamicsPairs& pairs,const torch::Tensor& reference,int64_t fit,int64_t begin,int64_t end,double ridge,std::ofstream& report,int64_t scoring_cutoff) {
  for(int64_t k:{1,4}) {
    if(k>pairs.spec.history_steps)continue;
    ev::BaselineRecipe recipe;recipe.fit_cutoff_ms=fit;recipe.train_begin_ms=begin;recipe.train_end_ms=end;
    recipe.history_size=k;recipe.spacing_ticks=pairs.spec.spacing_ticks;recipe.ridge_penalty=ridge;
    const auto fitted=ev::fit_baselines(bank,associations,recipe);
    ev::save_model((output/("baseline-k"+std::to_string(k)+".pt")).string(),fitted);
    for(const auto& p:ev::predict_baselines(bank,associations,fitted)) {
      const auto name=p.method.starts_with("ridge-k")?p.method:p.method+"-k"+std::to_string(k);
      ev::save_predictions((output/(name+".pt")).string(),p);
      scores(report,bank,pairs,p,reference,scoring_cutoff);
    }
  }
}
}

int main(int argc,char** argv) try {
  TORCH_CHECK(argc==3 || argc==5,"usage: latent_market_v1_study STUDY.conf NEW_OUTPUT_LEAF [TRAINING_CHECKPOINT SIDECAR_SHA256]");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  const fs::path settings=argv[1],output=argv[2];const auto settings_hash=lm::sha256_file(settings);
  const auto doc=cuwacunu::config::Document::read(settings);doc.reject_unknown_blocks({"STATE_STUDY_V1"});const auto& b=doc.block("STATE_STUDY_V1");
  b.reject_unknown_keys({"CARD","CARD_SHA256","SOURCE_CLOSURE","SOURCE_CLOSURE_SHA256","TIMELINE","TIMELINE_SHA256","PAIRS","PAIRS_SHA256","FIT_CUTOFF_MS","SCORING_CUTOFF_MS","CHECKPOINT_EVERY",
    "STATE_WIDTH","IDENTITY_INITIALIZATION","RESIDUAL_DYNAMICS","ARM","UPDATES","MICROBATCH","N_COV","SEED","LEARNING_RATE","BETA1","BETA2","OPTIMIZER_EPSILON",
    "WEIGHT_DECAY","CLIP_NORM","LAMBDA_PREDICTION","LAMBDA_VARIANCE","LAMBDA_COVARIANCE","VARIANCE_FLOOR","COVARIANCE_EPSILON","EMA_DECAY",
    "TRANSFORM_SCALE_FLOOR","REFERENCE_SCALE_FLOOR","RIDGE_PENALTY","DEVICE",
    "ENCODER_CHECKPOINT","ENCODER_SIDECAR_SHA256","ENCODER_MODEL_REF","CAPTURE_RECEIPT","CAPTURE_RECEIPT_SHA256"});
  const auto card=doc.resolve_path(b.string("CARD"));const auto card_hash=b.string("CARD_SHA256");lm::verify_sha256(card,card_hash);
  const auto closure=doc.resolve_path(b.string("SOURCE_CLOSURE"));const auto closure_hash=b.string("SOURCE_CLOSURE_SHA256");lm::verify_sha256(closure,closure_hash);
  const auto source_pins=lm::read_pins(closure);lm::require_pinned(source_pins,__FILE__);const auto binary_hash=lm::sha256_file("/proc/self/exe");
  const auto timeline_path=doc.resolve_path(b.string("TIMELINE")),pairs_path=doc.resolve_path(b.string("PAIRS"));
  const auto timeline_hash=b.string("TIMELINE_SHA256"),pairs_hash=b.string("PAIRS_SHA256");lm::verify_sha256(timeline_path,timeline_hash);lm::verify_sha256(pairs_path,pairs_hash);
  const auto timeline=lm::load_timeline(timeline_path);const auto pairs=lm::load_pairs(pairs_path);lm::validate_pairs(timeline,pairs);
  const auto encoder_path=doc.resolve_path(b.string("ENCODER_CHECKPOINT"));
  const auto encoder_pin=b.string("ENCODER_SIDECAR_SHA256");
  const auto capture_receipt=doc.resolve_path(b.string("CAPTURE_RECEIPT"));
  lm::verify_sha256(capture_receipt,b.string("CAPTURE_RECEIPT_SHA256"));
  verify_capture_binding(capture_receipt,timeline_hash,encoder_pin,b.string("ENCODER_MODEL_REF"));
  const auto encoder=av::read_verified(encoder_path,encoder_pin,"inference");
  TORCH_CHECK(encoder.manifest.components.size()==1 &&
      encoder.manifest.components.front().module_ref.component==av::encoder_component &&
      encoder.manifest.components.front().model_ref==b.string("ENCODER_MODEL_REF") &&
      encoder.manifest.archive_sha256==timeline.identity.checkpoint_hash,
      "Saved embeddings require their exact selected Encoder artifact and ModelRef");
  pt::artifact::Context context{encoder.manifest.components.front(),encoder.manifest.archive_sha256,
      fs::current_path(),timeline.interval_ms,pairs.spec.spacing_ticks,timeline.identity.channel_ids,
      timeline.identity.feature_names,timeline.identity.export_layout,timeline.identity.availability_policy};
  for(const auto& row:timeline.rows)TORCH_CHECK(row.role!=lm::Role::test,"TEST opening requires the later separately registered confirmation runner");
  TORCH_CHECK(!timeline.rows.empty() && pairs.spec.history_steps>=4,"The initial study requires K4 history and nonempty native arrays");
  TORCH_CHECK(*std::max_element(pairs.spec.horizons_ticks.begin(),pairs.spec.horizons_ticks.end())>=timeline.window_ticks,"A declared nonoverlapping horizon is mandatory");
  const int64_t fit=b.integer("FIT_CUTOFF_MS"),scoring_cutoff=b.integer("SCORING_CUTOFF_MS"),updates=b.integer("UPDATES"),microbatch=b.integer("MICROBATCH"),n_cov=b.integer("N_COV"),seed=b.integer("SEED");
  const auto checkpoint_every=b.integer("CHECKPOINT_EVERY");TORCH_CHECK(checkpoint_every>0,"Declare a positive immutable checkpoint cadence");
  TORCH_CHECK(fit>=timeline.identity.encoder_fit_cutoff_ms && updates>0 && microbatch>0 && n_cov>=2 && seed>=0 && scoring_cutoff>=fit,"Invalid cutoff/budget/population recipe");
  for(const auto& r:pairs.roles)if(r.role!=lm::Role::train)TORCH_CHECK(r.begin_ms>fit,"Fitted artifacts must precede held-out origins");
  const auto bank=lm::array_bank(timeline);const auto assoc=lm::associations(timeline,pairs);
  const auto population=lm::training_window_indices(timeline,pairs.roles,fit);
  TORCH_CHECK(n_cov<=population.numel(),"Regularizer pool must contain distinct eligible TRAIN origins");
  const auto train=bank.values.index_select(0,population);
  const auto floor=b.number("TRANSFORM_SCALE_FLOOR"),reference_floor=b.number("REFERENCE_SCALE_FLOOR");
  TORCH_CHECK(floor>0 && reference_floor>0,"Positive predeclared frozen scale floors required");
  const auto mean=train.mean(0),scale=torch::sqrt(train.var(0,false)).clamp_min(floor);
  pt::Config c;c.state.embedding_width=timeline.embedding_width;c.state.state_width=b.integer("STATE_WIDTH");
  c.state.identity_initialization=b.boolean("IDENTITY_INITIALIZATION");c.state.device=torch::Device(b.string("DEVICE"));c.state.dtype=torch::kFloat32;
  if(c.state.device.is_cuda() && c.state.device.index()<0)c.state.device=torch::Device(torch::kCUDA,0);
  TORCH_CHECK(c.state.device.is_cuda() && torch::cuda::is_available(),"State learning must use actual CUDA; no CPU fallback");
  c.dynamics.state_width=c.state.state_width;c.dynamics.history_length=pairs.spec.history_steps;c.dynamics.horizons=pairs.spec.horizons_ticks;
  c.dynamics.residual=b.boolean("RESIDUAL_DYNAMICS");c.dynamics.device=c.state.device;c.dynamics.dtype=c.state.dtype;
  const auto arm=b.string("ARM");
  if(arm=="frozen_state")c.arm=pt::Arm::frozen_state;else if(arm=="blocked_prediction")c.arm=pt::Arm::blocked_prediction;else if(arm=="full")c.arm=pt::Arm::full;else TORCH_CHECK(false,"Unknown state-learning arm");
  c.learning_rate=b.number("LEARNING_RATE");c.beta1=b.number("BETA1");c.beta2=b.number("BETA2");c.optimizer_epsilon=b.number("OPTIMIZER_EPSILON");
  c.weight_decay=b.number("WEIGHT_DECAY");c.clip_norm=b.number("CLIP_NORM");c.lambda_prediction=b.number("LAMBDA_PREDICTION");
  c.lambda_variance=b.number("LAMBDA_VARIANCE");c.lambda_covariance=b.number("LAMBDA_COVARIANCE");c.variance_floor=b.number("VARIANCE_FLOOR");
  c.covariance_epsilon=b.number("COVARIANCE_EPSILON");c.ema_decay=b.number("EMA_DECAY");c.sampler_seed=seed;pt::validate(c);
  const auto options=torch::TensorOptions().device(c.state.device).dtype(c.state.dtype);
  torch::manual_seed(seed);sr::AffineState initial(c.state,mean,scale);torch::Tensor reference;
  {torch::NoGradGuard guard;reference=torch::sqrt(initial->forward(train.to(options)).var(0,false)).clamp_min(reference_floor).cpu().to(torch::kFloat64);}
  lm::new_leaf(output);fs::create_directory(output/"e-controls");fs::create_directory(output/"state-controls");
  std::ofstream transform(output/"frozen-transform.txt");transform<<std::setprecision(17)<<"unique-complete-mature-TRAIN-window-equal-weight-population-moments-v1\n";
  for(const auto& t:{mean,scale,reference}){for(int64_t n=0;n<t.numel();++n)transform<<t[n].item<double>()<<' ';transform<<'\n';}
  transform<<"TRAIN-window-endpoints ";for(int64_t j=0;j<population.numel();++j)transform<<timeline.rows[population[j].item<int64_t>()].endpoint_ms<<' ';transform<<'\n';
  transform.close();TORCH_CHECK(transform.good(),"Transform write failed");
  const auto transform_hash=lm::sha256_file(output/"frozen-transform.txt");
  pt::Identity identity{timeline.identity.checkpoint_hash,timeline.identity.scaler_hash,transform_hash,transform_hash,
    timeline_hash,"fit:"+std::to_string(fit)+";ledger:"+pairs_hash,pairs_hash,
    "replacement-pair-sampling;unique-origin-rejection-pool;n_cov:"+std::to_string(n_cov)+";B:"+std::to_string(microbatch)+";seed:"+std::to_string(seed),
    "card:"+card_hash+";settings:"+settings_hash+";source-closure:"+closure_hash+";binary:"+binary_hash,"online-state-history-to-frozen-target-state;native-e-unchanged"};
  torch::manual_seed(seed);
  const auto resume_archive_hash=argc==5?av::read_verified(argv[3],argv[4],"training").manifest.archive_sha256:"none";
  auto trainer=argc==5?pt::artifact::load_training(argv[3],argv[4],c,identity,context):
      std::make_unique<pt::Trainer>(c,identity,mean,scale,reference);
  TORCH_CHECK(trainer->progress.completed_updates<=updates,"Resume has already exceeded the declared budget");
  std::vector<int64_t> training_pairs;
  for(size_t n=0;n<pairs.rows.size();++n) {
    const auto& p=pairs.rows[n];bool eligible=false;
    for(const auto& f:p.futures)if(f.eligible && f.target.row_index && timeline.rows[*f.target.row_index].available_ms<=fit)eligible=true;
    if(p.role==lm::Role::train && p.decision_ms<=fit && p.input_reasons==lm::no_issue && eligible)training_pairs.push_back(n);
  }
  TORCH_CHECK(!training_pairs.empty(),"No eligible TRAIN transition targets matured by the declared fit cutoff");
  std::ofstream losses(output/"updates.csv");losses<<std::setprecision(17)<<"attempt,update,prediction,variance,covariance,total,n_cov,rank_bound\n";
  while(trainer->progress.completed_updates<updates) {
    auto history=torch::zeros({microbatch,pairs.spec.history_steps,static_cast<int64_t>(timeline.embedding_width)},torch::kFloat64);
    auto future=torch::zeros({microbatch,static_cast<int64_t>(pairs.spec.horizons_ticks.size()),static_cast<int64_t>(timeline.embedding_width)},torch::kFloat64);
    auto valid=torch::zeros({microbatch,static_cast<int64_t>(pairs.spec.horizons_ticks.size())},torch::kBool);
    const auto chosen=trainer->sample_indices(training_pairs.size(),microbatch);
    for(int64_t j=0;j<microbatch;++j) {
      const auto& p=pairs.rows[training_pairs[chosen[j]]];
      for(size_t k=0;k<p.history.size();++k)history[j][k].copy_(bank.values[*p.history[k].row_index]);
      for(size_t q=0;q<p.futures.size();++q){const auto& f=p.futures[q];if(f.eligible && f.target.row_index && timeline.rows[*f.target.row_index].available_ms<=fit){future[j][q].copy_(bank.values[*f.target.row_index]);valid[j][q]=true;}}
    }
    std::set<int64_t> pool_indices;while(pool_indices.size()<static_cast<size_t>(n_cov))pool_indices.insert(trainer->sample_indices(population.numel(),1)[0]);
    std::vector<int64_t> pool_rows,pool_origins;for(auto i:pool_indices){const auto n=population[i].item<int64_t>();pool_rows.push_back(n);pool_origins.push_back(timeline.rows[n].endpoint_ms);}
    const auto receipt=trainer->step({history.to(options),future.to(options),valid.to(c.state.device),
      {bank.values.index_select(0,torch::tensor(pool_rows,torch::kInt64)).to(options),torch::tensor(pool_origins,torch::kInt64)}});
    TORCH_CHECK(receipt.updated,"An admitted all-valid training batch unexpectedly skipped; state preserved");
    losses<<trainer->progress.attempted_batches<<','<<trainer->progress.completed_updates<<','<<receipt.prediction_loss<<','<<receipt.variance_loss<<','<<receipt.covariance_loss<<','<<receipt.total_loss<<','<<receipt.n_cov<<','<<receipt.rank_upper_bound<<'\n';
    if(trainer->progress.completed_updates%checkpoint_every==0)
      pt::artifact::save_training(*trainer,output/("training-"+std::to_string(trainer->progress.completed_updates)+".pt"),context);
  }
  pt::artifact::save_training(*trainer,output/"training.pt",context);
  const auto inference_manifest=pt::artifact::save_inference(*trainer,output/"inference.pt",context);
  const auto inference_sidecar=av::sha256_file(av::sidecar_path(output/"inference.pt"));
  const auto serving=std::make_shared<const pt::artifact::CheckedInference>(
    pt::artifact::load_inference(output/"inference.pt",inference_sidecar,c,identity,context));
  const auto predictive_verified=av::read_verified(output/"inference.pt",inference_sidecar,"inference");
  const auto issuance_contract=fi::make_contract(timeline,pairs.spec,
    {timeline.identity.encoder_fit_cutoff_ms,fit,fit,fit},encoder,predictive_verified);
  TORCH_CHECK(issuance_contract.timeline_sha256==timeline_hash && issuance_contract.pairs_sha256==pairs_hash,
              "Issuance archive sources differ from the loaded exact timeline and pair ledger");
  const fi::Issuer issuer(issuance_contract,serving,encoder,predictive_verified,timeline);
  std::vector<av::BundleSelection> selection{{encoder_pin,std::string(av::encoder_component),context.encoder.model_ref}};
  for(const auto& component:inference_manifest.components)
    selection.push_back({inference_sidecar,component.module_ref.component,component.model_ref});
  const auto bundle=av::bundle_ref(selection);
  const auto checkpoint_hash=lm::sha256_file(output/"inference.pt");
  torch::NoGradGuard no_grad;trainer->online->eval();trainer->target->eval();trainer->dynamics->eval();
  auto state_bank=bank;state_bank.coordinate_id=lm::target_state_coordinate_id(timeline.identity.artifact_id,checkpoint_hash);
  std::vector<torch::Tensor> online_parts,target_parts;
  for(int64_t n=0;n<bank.values.size(0);n+=256) {
    const auto e=bank.values.narrow(0,n,std::min<int64_t>(256,bank.values.size(0)-n)).to(options);
    online_parts.push_back(serving->state_history(e).cpu().to(torch::kFloat64));target_parts.push_back(serving->target_states(e).cpu().to(torch::kFloat64));
  }
  const auto online=torch::cat(online_parts);state_bank.values=torch::cat(target_parts);
  torch::serialize::OutputArchive states;states.write("artifact_kind",torch::IValue("latent_market_state_bank_v1"));
  states.write("online",online,true);states.write("target",state_bank.values,true);states.write("valid",bank.valid,true);
  std::vector<int64_t> endpoints;for(const auto& r:timeline.rows)endpoints.push_back(r.endpoint_ms);
  states.write("endpoint_ms",torch::tensor(endpoints,torch::kInt64),true);states.write("timeline_sha256",torch::IValue(timeline_hash));
  states.write("checkpoint_sha256",torch::IValue(checkpoint_hash));states.write("target_coordinate_id",torch::IValue(state_bank.coordinate_id));
  auto prediction=ev::PredictionSet{};prediction.coordinate_id=state_bank.coordinate_id;prediction.method="state-"+arm;
  prediction.method="state-causal-forecast-issuance-v1";
  prediction.fit_cutoff_ms=fit;prediction.history_size=pairs.spec.history_steps;prediction.spacing_ticks=pairs.spec.spacing_ticks;prediction.associations=assoc;
  prediction.values=torch::zeros({static_cast<int64_t>(assoc.size()),c.state.state_width},torch::kFloat64);
  prediction.valid=torch::zeros({static_cast<int64_t>(assoc.size())},torch::kBool);prediction.reasons=torch::zeros({static_cast<int64_t>(assoc.size())},torch::kInt64);
  auto fitting_forecasts=torch::zeros({static_cast<int64_t>(pairs.rows.size()),static_cast<int64_t>(pairs.spec.horizons_ticks.size()),c.state.state_width},torch::kFloat64);
  auto fitting_valid=torch::zeros({static_cast<int64_t>(pairs.rows.size())},torch::kBool);
  std::vector<fi::IssuedOrigin> issued;issued.reserve(pairs.rows.size());
  for(size_t n=0;n<pairs.rows.size();++n) {
    const auto& p=pairs.rows[n];const auto offset=n*p.futures.size();
    issued.push_back(issuer.issue(timeline,{p.pair_id,p.endpoint_ms,p.decision_ms,p.role,p.dependence_group},pairs.roles,p.input_reasons));
    const auto& causal=issued.back();
    for(size_t q=0;q<p.futures.size();++q) {
      prediction.values[offset+q].copy_(causal.values[q]);prediction.valid[offset+q]=causal.valid;
      // The high bits retain all issuance-specific reasons, alongside the
      // evaluation vocabulary used by existing scoring consumers.
      prediction.reasons[offset+q]=fi::evaluation_reasons(causal.reasons);
    }
    if(p.input_reasons)continue;
    // origin_inputs exposes no future values, masks, target eligibility or errors.
    const auto input=lm::origin_inputs(timeline,p);
    const auto e=torch::tensor(input.flattened_values,torch::kFloat64).reshape({1,pairs.spec.history_steps,static_cast<int64_t>(timeline.embedding_width)}).to(options);
    const auto forecasts=serving->forecast(e)[0].cpu().to(torch::kFloat64);
    // Retrospective frozen-predictor features for fitting TRAIN risk heads.
    // These are never relabeled as issued out-of-sample forecasts or surprise.
    fitting_forecasts[n].copy_(forecasts);fitting_valid[n]=true;
  }
  states.write("fitting_forecasts",fitting_forecasts,true);states.write("fitting_forecast_valid",fitting_valid,true);
  states.write("fit_cutoff_ms",torch::tensor(std::vector<int64_t>{fit},torch::kInt64),true);
  states.write("pair_ledger_sha256",torch::IValue(pairs_hash));
  pt::detail::publish(states,(output/"state-bank.pt").string());
  const auto issuance_manifest_hash=fi::publish_issuance(output,issuance_contract,pairs,issued,prediction);
  prediction=fi::load_issued_predictions(output,issuance_manifest_hash,issuance_contract);
  std::ofstream report(output/"scores.csv");report<<std::setprecision(17)<<"coordinates,method,K,role,horizon,support,overlap,total,input_valid,target_valid,scored,coverage,normalized_mse,native_mae,persistence_mse,skill\n";
  int64_t train_begin=0,train_end=0;for(const auto& r:pairs.roles)if(r.role==lm::Role::train){train_begin=r.begin_ms;train_end=r.end_ms;}
  baseline_suite(output/"e-controls",bank,assoc,pairs,scale,fit,train_begin,train_end,b.number("RIDGE_PENALTY"),report,scoring_cutoff);
  baseline_suite(output/"state-controls",state_bank,assoc,pairs,reference,fit,train_begin,train_end,b.number("RIDGE_PENALTY"),report,scoring_cutoff);
  scores(report,state_bank,pairs,prediction,reference,scoring_cutoff);
  auto diagnostic=state_bank;diagnostic.values=state_bank.values.index_select(0,population);diagnostic.valid=state_bank.valid.index_select(0,population);diagnostic.rows.clear();
  for(int64_t j=0;j<population.numel();++j)diagnostic.rows.push_back(state_bank.rows[population[j].item<int64_t>()]);
  const auto diversity=ev::diversity(diagnostic,reference);
  std::ofstream receipt(output/"study.txt");receipt<<std::setprecision(17)<<"latent-market-state-study-v1\narm "<<arm<<"\nsettings_sha256 "<<settings_hash<<"\ncard_sha256 "<<card_hash
    <<"\ntimeline_sha256 "<<timeline_hash<<"\npairs_sha256 "<<pairs_hash<<"\ninference_sha256 "<<checkpoint_hash<<"\nbinary_sha256 "<<lm::sha256_file("/proc/self/exe")
    <<"\ninference_sidecar_sha256 "<<inference_sidecar<<"\nencoder_model_ref "<<context.encoder.model_ref<<"\nbundle_ref "<<bundle
    <<"\nissuance_policy_sha256 "<<fi::policy_sha256<<"\nissuance_manifest_sha256 "<<issuance_manifest_hash
    <<"\nissuance_contract_sha256 "<<fi::contract_digest(issuance_contract)
    <<"\nissuance_ledger_sha256 "<<lm::sha256_file(output/"state-issuance.csv")
    <<"\nforecast_scope issued-valid-only;retrospective-fitting-forecasts-remain-diagnostic"
    <<"\nresume_archive_sha256 "<<resume_archive_hash
    <<"\ncompleted_updates "<<trainer->progress.completed_updates<<"\nencoder_calls 0\nencoder_updates 0\ntransform_train_windows "<<population.numel()<<"\nregularizer_distinct_origins "<<n_cov
    <<"\nsource_closure_sha256 "<<closure_hash<<"\ndiagnostic_population unique-complete-mature-TRAIN-windows\ndiagnostic_windows "<<population.numel()
    <<"\ntarget_state_effective_rank "<<diversity.effective_rank<<"\nfull_state_factor_fraction "<<diversity.largest_factor_fraction
    <<"\nquality_gate no-automatic-advancement;retention-and-common-future-risk-utility-required\n";
  lm::verify_sha256(settings,settings_hash);lm::verify_sha256(card,card_hash);lm::verify_sha256(timeline_path,timeline_hash);lm::verify_sha256(pairs_path,pairs_hash);
  lm::verify_sha256(closure,closure_hash);lm::verify_pins(source_pins);
  lm::verify_sha256(capture_receipt,b.string("CAPTURE_RECEIPT_SHA256"));
  TORCH_CHECK(losses.good() && report.good() && receipt.good(),"Study artifact write failed");
  std::cout<<"Saved "<<arm<<" checkpoint, fixed state bank, predictions and matched controls. No automatic quality or deployment claim.\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
