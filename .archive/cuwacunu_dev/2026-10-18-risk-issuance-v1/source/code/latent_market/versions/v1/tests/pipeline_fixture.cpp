// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Synthetic engineering fixture only. It executes no Encoder and establishes
// no market, forecasting, representation-quality or investment result.
#include "cuwacunu/latent_market/artifacts.h"
#include "cuwacunu/latent_market/ledger_io.h"
#include "cuwacunu/artifacts/v1/sidecar.h"
#include <iostream>
#include <set>

namespace lm=cuwacunu::latent_market::v1;
namespace fs=std::filesystem;
namespace av=cuwacunu::artifacts::v1;

namespace {
void checked_write(const fs::path& path,const std::string& text) {
  lm::ledger_detail::publish(path,[&](std::ostream& out){out<<text;});
}

void source_closure(const fs::path& destination) {
  const fs::path root="/cuwacunu";
  std::set<fs::path> sources;
  for (const auto* relative:{"code/latent_market/versions/v1","code/networks","code/artifacts/versions/v1",
                             "code/evaluation/latent_dynamics/versions/v1","code/config/include"}) {
    const auto directory=root/relative;
    if (!fs::is_directory(directory)) throw std::runtime_error("Fixture source tree missing: "+directory.string());
    for (const auto& entry:fs::recursive_directory_iterator(directory))
      if (entry.is_regular_file() && !entry.is_symlink()) sources.insert(entry.path());
  }
  sources.insert(root/"Makefile");
  std::ostringstream manifest;
  manifest<<"# Synthetic engineering fixture source superset, not a production dependency closure.\n";
  for (const auto& path:sources) manifest<<lm::sha256_file(path)<<"  "<<path.generic_string()<<'\n';
  checked_write(destination,manifest.str());
  lm::verify_pins(lm::read_pins(destination));
}
} // namespace

int main(int argc,char** argv) try {
  if (argc!=2) throw std::runtime_error("usage: latent_market_v1_pipeline_fixture NEW_OUTPUT_LEAF");
  const fs::path leaf=fs::absolute(argv[1]).lexically_normal();
  lm::new_leaf(leaf);
  // Published files deliberately remain on any later failure for inspection.
  checked_write(leaf/"sha-empty.txt","");checked_write(leaf/"sha-abc.txt","abc");
  lm::verify_sha256(leaf/"sha-empty.txt","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  lm::verify_sha256(leaf/"sha-abc.txt","ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  source_closure(leaf/"source-closure.sha256");
  checked_write(leaf/"fixture-checkpoint.txt","SYNTHETIC D4 deterministic array generator; no actual Encoder checkpoint or call.\n");
  checked_write(leaf/"fixture-scaler.txt","SYNTHETIC identity source scaler; no calibration or market data.\n");
  checked_write(leaf/"fixture-config.txt","SYNTHETIC fixture-v1 H4 C1 F1 D4, interval 60000ms; all support observed.\n");
  // Explicit synthetic parent evidence only: the generator has no learned
  // Encoder state and these text bytes never enter the selected Encoder loader.
  const auto encoder_component=av::make_component({"Encoder","v1",
      av::source_closure_sha256("/cuwacunu",{"code/latent_market/versions/v1/tests/pipeline_fixture.cpp"})},
      lm::sha256_file(leaf/"fixture-scaler.txt"),lm::sha256_file(leaf/"fixture-config.txt"));
  const av::Manifest encoder_manifest{"inference",lm::sha256_file(leaf/"fixture-checkpoint.txt"),{},{encoder_component}};
  const auto encoder_sidecar=av::publish_sidecar(leaf/"fixture-checkpoint.txt",encoder_manifest);

  constexpr std::int64_t interval=60000,base=1800000000000ll;
  const auto at=[](std::int64_t tick){return base+tick*interval;};
  const std::vector<lm::RoleInterval> roles={{lm::Role::train,at(0),at(119)},
                                          {lm::Role::validation,at(120),at(179)}};
  const auto fit=at(119)+5,scoring=at(179)+5;
  std::vector<lm::AssetCloseSeries> closes={{101,{}}};
  std::ostringstream raw;raw<<std::setprecision(17)<<"endpoint_ms\tavailable_ms\tchannel_id\tclose\tobserved\n";
  for (std::int64_t tick=0;tick<180;++tick) {
    const double close=100*std::exp(0.001*std::sin(0.13*tick)+0.0001*tick);
    closes[0].rows.push_back({at(tick),at(tick)+5,close,true});
    raw<<at(tick)<<'\t'<<at(tick)+5<<"\t101\t"<<close<<"\t1\n";
  }
  checked_write(leaf/"synthetic-source.tsv",raw.str());
  checked_write(leaf/"synthetic-cache.tsv",raw.str());

  lm::EmbeddingTimeline timeline;
  auto& id=timeline.identity;
  id.artifact_id="synthetic-latent-market-backbone-v1";id.dataset_id="SYNTHETIC-LATENT-PIPELINE-V1";
  id.encoder_family="synthetic-fixture-no-encoder-execution";id.encoder_version="v1";
  id.checkpoint_hash=lm::sha256_file(leaf/"fixture-checkpoint.txt");
  id.scaler_hash=lm::sha256_file(leaf/"fixture-scaler.txt");
  id.encoder_config_hash=lm::sha256_file(leaf/"fixture-config.txt");
  id.source_hash=lm::sha256_file(leaf/"synthetic-source.tsv");id.cache_hash=lm::sha256_file(leaf/"synthetic-cache.tsv");
  id.export_layout="synthetic-complete-D4";id.availability_policy="SIMULATED endpoint+5ms; no live latency measurement";
  id.source_closure_hash=lm::sha256_file(leaf/"source-closure.sha256");id.executable_hash=lm::sha256_file("/proc/self/exe");
  id.encoder_fit_cutoff_ms=at(0);id.channel_ids={101};id.feature_names={"close"};
  timeline.interval_ms=interval;timeline.window_ticks=4;timeline.embedding_width=4;
  for (std::int64_t tick=3;tick<180;++tick) {
    lm::EmbeddingRow row;
    row.window_id=lm::expected_window_id(id,at(tick));row.endpoint_ms=at(tick);row.available_ms=at(tick)+5;
    row.role=tick<120?lm::Role::train:lm::Role::validation;row.sample_valid=true;
    row.support.channels=1;row.support.features=1;row.support.observed.assign(4,1);
    for (std::int64_t past=tick-3;past<=tick;++past) row.support.timestamps_ms.push_back(at(past));
    row.values={std::sin(tick*0.11),std::cos(tick*0.071)+0.1*std::sin(tick*0.2),
                static_cast<double>(tick%13-6)/7.0,0.4*std::sin(tick*0.17)+0.6*std::cos(tick*0.037)};
    timeline.rows.push_back(std::move(row));
  }
  lm::PairSpec spec;spec.history_steps=4;spec.spacing_ticks=1;
  spec.horizons_ticks={1,4};spec.future_label_ticks={4};
  std::vector<lm::PairOrigin> origins;
  // Include deliberate incomplete/boundary associations as an archive witness;
  // valid populations are 110 TRAIN and 50 VALIDATION origins.
  for (std::int64_t tick=5;tick<=119;++tick)
    origins.push_back({"synthetic-origin-"+std::to_string(tick),at(tick),at(tick)+5,lm::Role::train,
                       "synthetic-train-block-"+std::to_string(tick/16)});
  for (std::int64_t tick=123;tick<=178;++tick)
    origins.push_back({"synthetic-origin-"+std::to_string(tick),at(tick),at(tick)+5,lm::Role::validation,
                       "synthetic-validation-block-"+std::to_string((tick-120)/16)});
  const auto pairs=lm::build_pairs(timeline,origins,spec,roles);
  const auto risk=lm::build_future_risk_targets(id.artifact_id,origins,closes,interval,{4},roles,id.source_hash);
  lm::validate_pairs(timeline,pairs);lm::validate_risk_targets(risk);
  lm::publish_timeline(leaf/"timeline.bin",timeline);lm::publish_pairs(leaf/"pairs.bin",pairs);
  checked_write(leaf/"preparation.txt",std::string("latent-market-synthetic-capture-v1\n")+
      "timeline_sha256 "+lm::sha256_file(leaf/"timeline.bin")+"\nencoder_sidecar_sha256 "+encoder_sidecar+
      "\nencoder_model_ref "+encoder_component.model_ref+"\n");
  lm::publish_risk_targets(leaf/"future-risk.bin",risk);
  if (lm::load_timeline(leaf/"timeline.bin")!=timeline || lm::load_pairs(leaf/"pairs.bin",timeline)!=pairs ||
      lm::load_risk_targets(leaf/"future-risk.bin")!=risk) throw std::runtime_error("Synthetic fixture roundtrip failed");
  std::size_t train_count=0,validation_count=0;
  for (const auto& row:pairs.rows) if (std::all_of(row.futures.begin(),row.futures.end(),[](const auto& f){return f.eligible;}))
    (row.role==lm::Role::train?train_count:validation_count)++;
  if (train_count!=110 || validation_count!=50) throw std::runtime_error("Synthetic complete population count mismatch");
  std::ostringstream card;
  card<<"SYNTHETIC-LATENT-PIPELINE-V1 ENGINEERING CARD\n"
      <<"No market inputs, actual Encoder execution, historical replay or market quality claim.\n"
      <<"Deterministic synthetic native-e D4, H4/C1/F1, cadence 60000ms, all close masks observed.\n"
      <<"Chronological TRAIN ticks 0..119; VALIDATION ticks 120..179; strict full-window role support.\n"
      <<"K4 s1 direct horizons 1,4; future-risk R4. Valid origins TRAIN 110, VALIDATION 50.\n"
      <<"Purged associations remain in pairs.bin. Availability endpoint+5ms is simulated.\n"
      <<"Frozen synthetic source arrays; affine same-width G initialized to identity, exact target copy, linear A.\n"
      <<"Full state-learning arm, CUDA float32, 2 successful updates, B4, n_cov8 distinct TRAIN origins.\n"
      <<"Seed 17; AdamW LR 0.001 beta1 0.9 beta2 0.999 epsilon 1e-8 decay 0; clip norm 1.\n"
      <<"Prediction weight 1, variance weight 1, covariance weight 1, variance floor 0.1, covariance epsilon 1e-4.\n"
      <<"EMA decay 0.99 on successful updates; transform/reference scale floors 1e-6; ridge penalty 0.01.\n"
      <<"Equal weighting of complete mature unique TRAIN windows; no validation transform fitting.\n"
      <<"Future-risk dispersion is centered population one-minute log-return deviation; downside uses origin close.\n"
      <<"Acceptance: end-to-end artifact construction, finite optimization and fixed-checkpoint arrays only.\n"
      <<"No statistical gate, promotion, portfolio, deployment, external publication or generalization claim.\n"
      <<"fit_cutoff_ms "<<fit<<"\nscoring_cutoff_ms "<<scoring<<'\n';
  checked_write(leaf/"synthetic-card.txt",card.str());
  std::ostringstream conf;
  conf<<"# Synthetic engineering fixture only; exact data/card/source bytes are independently pinned.\nSTATE_STUDY_V1 {\n"
      <<"  CARD = synthetic-card.txt;\n  CARD_SHA256 = "<<lm::sha256_file(leaf/"synthetic-card.txt")<<";\n"
      <<"  TIMELINE = timeline.bin;\n  TIMELINE_SHA256 = "<<lm::sha256_file(leaf/"timeline.bin")<<";\n"
      <<"  PAIRS = pairs.bin;\n  PAIRS_SHA256 = "<<lm::sha256_file(leaf/"pairs.bin")<<";\n"
      <<"  ENCODER_CHECKPOINT = fixture-checkpoint.txt;\n  ENCODER_SIDECAR_SHA256 = "<<encoder_sidecar<<";\n"
      <<"  ENCODER_MODEL_REF = "<<encoder_component.model_ref<<";\n"
      <<"  CAPTURE_RECEIPT = preparation.txt;\n  CAPTURE_RECEIPT_SHA256 = "<<lm::sha256_file(leaf/"preparation.txt")<<";\n"
      <<"  SOURCE_CLOSURE = source-closure.sha256;\n  SOURCE_CLOSURE_SHA256 = "<<lm::sha256_file(leaf/"source-closure.sha256")<<";\n"
      <<"  FIT_CUTOFF_MS = "<<fit<<";\n  SCORING_CUTOFF_MS = "<<scoring<<";\n"
      <<"  STATE_WIDTH = 4;\n  IDENTITY_INITIALIZATION = true;\n  RESIDUAL_DYNAMICS = true;\n"
      <<"  ARM = full;\n  UPDATES = 2;\n  CHECKPOINT_EVERY = 1;\n  MICROBATCH = 4;\n  N_COV = 8;\n  SEED = 17;\n"
      <<"  LEARNING_RATE = 0.001;\n  BETA1 = 0.9;\n  BETA2 = 0.999;\n  OPTIMIZER_EPSILON = 1e-8;\n"
      <<"  WEIGHT_DECAY = 0;\n  CLIP_NORM = 1;\n  LAMBDA_PREDICTION = 1;\n  LAMBDA_VARIANCE = 1;\n"
      <<"  LAMBDA_COVARIANCE = 1;\n  VARIANCE_FLOOR = 0.1;\n  COVARIANCE_EPSILON = 1e-4;\n"
      <<"  EMA_DECAY = 0.99;\n  TRANSFORM_SCALE_FLOOR = 1e-6;\n  REFERENCE_SCALE_FLOOR = 1e-6;\n"
      <<"  RIDGE_PENALTY = 0.01;\n  DEVICE = cuda;\n}\n";
  checked_write(leaf/"study.conf",conf.str());
  std::ostringstream pins;
  for (const auto* file:{"timeline.bin","pairs.bin","future-risk.bin","synthetic-card.txt","study.conf",
                         "source-closure.sha256","synthetic-source.tsv","synthetic-cache.tsv",
                         "fixture-checkpoint.txt","fixture-scaler.txt","fixture-config.txt"})
    pins<<lm::sha256_file(leaf/file)<<"  "<<file<<'\n';
  checked_write(leaf/"artifacts.sha256",pins.str());lm::verify_pins(lm::read_pins(leaf/"artifacts.sha256"));
  std::cout<<"Synthetic pipeline fixture admitted: "<<train_count<<" TRAIN / "<<validation_count
           <<" VALIDATION complete origins; "<<pairs.rows.size()<<" requested origins retained.\n"
           <<"Settings: "<<(leaf/"study.conf").string()<<"\nNo market or representation-quality claim.\n";
  return 0;
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
