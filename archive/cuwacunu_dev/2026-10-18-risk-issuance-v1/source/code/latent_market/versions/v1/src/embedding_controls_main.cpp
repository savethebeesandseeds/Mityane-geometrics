// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Shared-study fitted controls consume admitted saved arrays; no Encoder calls.
#include "cuwacunu/config/document.h"
#include "cuwacunu/latent_market/artifacts.h"
#include "cuwacunu/latent_market/ledger_io.h"
#include "cuwacunu/latent_market/arrays.h"
#include "cuwacunu/networks/direct_embedding_control/v1/artifact.h"
#include <iostream>

namespace lm=cuwacunu::latent_market::v1;
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace dc=cuwacunu::networks::direct_embedding_control::v1;
namespace ca=dc::artifact;
namespace av=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;

namespace {
std::map<std::string,std::string> fields(const fs::path& path,const std::string& type) {
  std::ifstream stream(path);std::string line;
  TORCH_CHECK(bool(stream)&&bool(std::getline(stream,line))&&line==type,"Wrong receipt type: ",path.string());
  std::map<std::string,std::string> result;
  while(std::getline(stream,line)) {
    const auto at=line.find(' ');
    TORCH_CHECK(at!=std::string::npos&&at>0&&at+1<line.size()&&
      result.emplace(line.substr(0,at),line.substr(at+1)).second,"Malformed or duplicate receipt field");
  }
  TORCH_CHECK(stream.eof(),"Receipt read failed");return result;
}
void equal_field(const std::map<std::string,std::string>& receipt,const std::string& name,const std::string& expected) {
  TORCH_CHECK(receipt.contains(name)&&receipt.at(name)==expected,"Receipt binding mismatch: ",name);
}
}

int main(int argc,char** argv) try {
  TORCH_CHECK(argc==3,"usage: latent_market_v1_embedding_controls PINNED.conf NEW_OUTPUT_LEAF");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  const fs::path settings=argv[1],output=argv[2];
  const auto settings_hash=av::sha256_file(settings);
  const auto document=cuwacunu::config::Document::read(settings);
  document.reject_unknown_blocks({"DIRECT_EMBEDDING_CONTROL_STUDY_V1"});
  const auto& b=document.block("DIRECT_EMBEDDING_CONTROL_STUDY_V1");
  b.reject_unknown_keys({"SCOPE","CARD","CARD_SHA256","AMENDMENT_1","AMENDMENT_1_SHA256","AMENDMENT_2","AMENDMENT_2_SHA256",
    "SOURCE_CLOSURE","SOURCE_CLOSURE_SHA256","TIMELINE","TIMELINE_SHA256","PAIRS","PAIRS_SHA256",
    "ENCODER_CHECKPOINT","ENCODER_SIDECAR_SHA256","ENCODER_MODEL_REF","CAPTURE_RECEIPT","CAPTURE_RECEIPT_SHA256",
    "FIT_ADMISSION","FIT_ADMISSION_SHA256","FIT_CUTOFF_MS","HIDDEN_WIDTH","UPDATES","MICROBATCH","SEEDS"});
  std::map<std::string,fs::path> paths;std::map<std::string,std::string> hashes;
  for(const auto* key:{"CARD","SOURCE_CLOSURE","TIMELINE","PAIRS","CAPTURE_RECEIPT","FIT_ADMISSION"}) {
    paths[key]=document.resolve_path(b.string(key));hashes[key]=b.string(std::string(key)+"_SHA256");
    lm::verify_sha256(paths[key],hashes[key]);
  }
  std::vector<std::string> protocol_fields{hashes.at("CARD")};
  const auto scope=b.string("SCOPE");
  TORCH_CHECK(scope=="synthetic-engineering",
    "Market fitting remains closed pending the separately verified campaign execution closure. This runner verifies artificial shared-study publication only.");
  TORCH_CHECK(!b.contains("AMENDMENT_1")&&!b.contains("AMENDMENT_2")&&
    !b.contains("AMENDMENT_1_SHA256")&&!b.contains("AMENDMENT_2_SHA256"),"Synthetic card is separate from market registration");
  const auto protocol_hash=av::digest_fields("latent-state-control-study-protocol-v1",protocol_fields);
  const auto pins=lm::read_pins(paths.at("SOURCE_CLOSURE"));lm::verify_pins(pins);lm::require_pinned(pins,__FILE__);
  const auto timeline=lm::load_timeline(paths.at("TIMELINE"));const auto pairs=lm::load_pairs(paths.at("PAIRS"));
  lm::validate_pairs(timeline,pairs);
  for(const auto& row:timeline.rows)TORCH_CHECK(row.role!=lm::Role::test,"TEST is not admitted by this runner");
  const bool artificial=timeline.identity.dataset_id.starts_with("SYNTHETIC-");
  TORCH_CHECK(artificial&&timeline.identity.encoder_family=="synthetic-fixture-no-encoder-execution",
    "Engineering fixture requires an explicitly artificial parent/population; market arrays are not admitted");
  const auto encoder_path=document.resolve_path(b.string("ENCODER_CHECKPOINT"));
  const auto encoder_pin=b.string("ENCODER_SIDECAR_SHA256");
  const auto encoder=av::read_verified(encoder_path,encoder_pin,"inference");
  TORCH_CHECK(encoder.manifest.components.size()==1&&encoder.manifest.components.front().module_ref.component==av::encoder_component&&
    encoder.manifest.components.front().model_ref==b.string("ENCODER_MODEL_REF")&&
    encoder.manifest.archive_sha256==timeline.identity.checkpoint_hash,"Saved inputs require their exact Encoder parent");
  const auto capture=fields(paths.at("CAPTURE_RECEIPT"),artificial?"latent-market-synthetic-capture-v1":"latent-market-preparation-v1");
  equal_field(capture,"timeline_sha256",hashes.at("TIMELINE"));equal_field(capture,"encoder_sidecar_sha256",encoder_pin);
  equal_field(capture,"encoder_model_ref",b.string("ENCODER_MODEL_REF"));
  const auto fit=b.integer("FIT_CUTOFF_MS");
  TORCH_CHECK(fit>=timeline.identity.encoder_fit_cutoff_ms,"Control cannot be fitted before its Encoder parent");
  const auto admission=fields(paths.at("FIT_ADMISSION"),"latent-market-control-fit-admission-v1");
  equal_field(admission,"scope",scope);equal_field(admission,"protocol_sha256",protocol_hash);
  equal_field(admission,"timeline_sha256",hashes.at("TIMELINE"));equal_field(admission,"pairs_sha256",hashes.at("PAIRS"));
  equal_field(admission,"source_closure_sha256",hashes.at("SOURCE_CLOSURE"));
  equal_field(admission,"encoder_model_ref",b.string("ENCODER_MODEL_REF"));equal_field(admission,"fit_cutoff_ms",std::to_string(fit));
  equal_field(admission,"current_chain_and_control_witnesses","synthetic-fixture-only");
  const auto bank=lm::array_bank(timeline);const auto associations=lm::associations(timeline,pairs);
  const auto population=av::digest_fields("latent-state-control-population-v1",{hashes.at("TIMELINE"),hashes.at("PAIRS"),std::to_string(fit)});
  dc::Config config;config.width=timeline.embedding_width;config.history_size=pairs.spec.history_steps;
  config.spacing_ticks=pairs.spec.spacing_ticks;config.horizons=pairs.spec.horizons_ticks;
  config.hidden_width=b.integer("HIDDEN_WIDTH");config.updates=b.integer("UPDATES");config.microbatch=b.integer("MICROBATCH");
  config.learning_rate=0.001;config.beta1=0.9;config.beta2=0.999;config.optimizer_epsilon=1e-8;
  config.weight_decay=0.0001;config.clip_norm=1.0;config.std_floor=1e-8;
  config.parameter_budget=18000;config.require_complete_support=true;
  config.fit_cutoff_ms=fit;
  for(const auto& role:pairs.roles) {
    if(role.role==lm::Role::train){config.train_begin_ms=role.begin_ms;config.train_end_ms=role.end_ms;}
    else TORCH_CHECK(role.begin_ms>fit,"Control fit must precede held-out origins");
  }
  const auto seeds=b.integers("SEEDS");TORCH_CHECK(!seeds.empty()&&seeds.size()<=3,"One to three declared control repetitions required");
  TORCH_CHECK(std::set<int64_t>(seeds.begin(),seeds.end()).size()==seeds.size(),"Duplicate repetition seed");
  ca::Context context{encoder.manifest.components.front(),encoder.manifest.archive_sha256,fs::current_path(),
    protocol_hash,population,av::sha256_bytes(bank.coordinate_id),fit,timeline.interval_ms,pairs.spec.spacing_ticks,
    timeline.identity.channel_ids,timeline.identity.feature_names,timeline.identity.export_layout,timeline.identity.availability_policy};
  dc::Identity identity{bank.coordinate_id,hashes.at("TIMELINE"),
    "fit:"+std::to_string(fit)+";pairs:"+hashes.at("PAIRS"),hashes.at("PAIRS"),context.encoder.model_ref,protocol_hash};
  lm::new_leaf(output);std::vector<ca::ComparisonSelection> selections;
  for(const auto seed:seeds) {
    config.repetition_seed=seed;const auto role="residual-mlp-seed-"+std::to_string(seed);
    const auto directory=output/role;fs::create_directory(directory);
    dc::Trainer trainer(config,identity,bank,associations);trainer.train();
    ca::save_training(trainer,directory/"training.pt",context);
    const auto manifest=ca::save_inference(trainer,directory/"inference.pt",context);
    const auto sidecar=av::sha256_file(av::sidecar_path(directory/"inference.pt"));
    const auto checked=ca::load_inference(directory/"inference.pt",sidecar,manifest.component.model_ref,config,identity,context);
    ev::save_predictions((directory/"predictions.pt").string(),checked.predict(bank,associations));
    selections.push_back({role,manifest.component.model_ref,sidecar,av::sha256_file(directory/"predictions.pt")});
  }
  const auto comparison=ca::make_comparison(protocol_hash,population,context.coordinates_sha256,context.encoder.model_ref,selections);
  const auto comparison_pin=ca::publish_comparison(output/"comparison.json",comparison);
  const auto reread=ca::read_comparison(output/"comparison.json",comparison_pin,protocol_hash,population,
    context.coordinates_sha256,context.encoder.model_ref);
  for(const auto& selected:reread.controls) {
    const auto directory=output/selected.role;
    const auto seed=std::stoll(selected.role.substr(std::string("residual-mlp-seed-").size()));config.repetition_seed=seed;
    const auto checked=ca::load_inference(directory/"inference.pt",selected.sidecar_sha256,selected.model_ref,config,identity,context);
    const auto verified_predictions=ca::load_comparison_predictions(reread,selected.role,checked,directory/"predictions.pt");
    TORCH_CHECK(verified_predictions.associations.size()==associations.size(),"Published predictions lost requested associations");
  }
  for(const auto& [key,path]:paths) {
    lm::verify_sha256(path,hashes.at(key));
  }
  lm::verify_pins(pins);
  lm::verify_sha256(settings,settings_hash);
  ca::publish_bytes(output/"study-controls.txt","latent-market-direct-embedding-control-study-v1\nscope "+scope+
    "\nprotocol_sha256 "+protocol_hash+"\npopulation_sha256 "+population+"\nsettings_sha256 "+settings_hash+
    "\nsource_closure_sha256 "+hashes.at("SOURCE_CLOSURE")+"\nbinary_sha256 "+av::sha256_file("/proc/self/exe")+
    "\nencoder_model_ref "+context.encoder.model_ref+"\ncomparison_sha256 "+comparison_pin+"\nbundle_ref "+comparison.bundle_ref+
    "\nencoder_calls 0\nencoder_updates 0\nTEST_opened false\npoint_skill_decision open\n");
  std::cout<<"PASS shared-study control publication: "<<seeds.size()<<" repetitions; native-e, checked artifacts and comparison pins; point skill remains open.\n";
  return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
