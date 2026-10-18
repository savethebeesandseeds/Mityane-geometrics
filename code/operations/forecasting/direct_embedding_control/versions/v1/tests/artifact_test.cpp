// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Artificial publication/consumption witnesses; no market fit or quality claim.
#include "cuwacunu/networks/direct_embedding_control/v1/artifact.h"
#include <cstring>
#include <iostream>

namespace dc=cuwacunu::networks::direct_embedding_control::v1;
namespace da=dc::artifact;
namespace av=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;
namespace ev=cuwacunu::evaluation::latent_dynamics::v1;
namespace {
void check(bool value,const std::string& text) {TORCH_CHECK(value,"[control artifact test] ",text);}
template<class F> void rejects(F&& action,const std::string& text) {
  bool caught=false;try{action();}catch(const std::exception&){caught=true;}check(caught,text);
}
bool exact(const torch::Tensor& a,const torch::Tensor& b) {
  if(!a.defined()||!b.defined())return a.defined()==b.defined();
  const auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
  return x.scalar_type()==y.scalar_type()&&x.sizes()==y.sizes()&&
    (!x.numel()||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.nbytes())==0);
}
struct Temporary {
  fs::path path;
  Temporary() {
    auto pattern=(fs::temp_directory_path()/"direct-control-artifact-v1-XXXXXX").string();
    std::vector<char> bytes(pattern.begin(),pattern.end());bytes.push_back('\0');
    const auto created=::mkdtemp(bytes.data());check(created,"private temporary directory");path=created;
  }
  ~Temporary(){std::error_code ignored;fs::remove_all(path,ignored);}
};
da::Context context() {
  da::Context result;
  result.encoder=av::make_component({"Encoder","v3",av::sha256_bytes("artificial Encoder source")},
    av::sha256_bytes("artificial Encoder state"),av::sha256_bytes("artificial Encoder mapping"));
  result.encoder_archive_sha256=av::sha256_bytes("artificial Encoder archive");
  result.repository_root=fs::current_path();result.protocol_sha256=av::sha256_bytes("artificial protocol");
  result.population_sha256=av::sha256_bytes("artificial TRAIN population");
  result.coordinates_sha256=av::sha256_bytes("artificial native3 coordinates");
  result.fit_cutoff_ms=1700000000000;result.cadence_ms=60000;result.history_spacing_ticks=1;
  result.channel_ids={0,1,2};result.feature_names={"close","volume"};
  result.embedding_export_layout="artificial native3";result.availability_policy="completed-candle cutoff";return result;
}
ev::ArrayBank bank() {
  ev::ArrayBank result;result.coordinate_id="artificial-native3";result.interval_ms=10;result.window_ticks=4;
  std::vector<double> values;
  for(const auto start:{0,100})for(int64_t j=start;j<start+40;++j) {
    ev::ArrayRow row;row.row_id=j;row.endpoint_ms=j*10;row.available_ms=j*10+2;
    row.support_begin_ms=(j-3)*10;row.support_end_ms=j*10;row.role=start==0?ev::Role::train:ev::Role::validation;
    result.rows.push_back(row);
    for(int64_t coordinate=0;coordinate<3;++coordinate)values.push_back(std::sin(j*.13+coordinate)+.01*j*(coordinate+1));
  }
  result.values=torch::tensor(values,torch::kFloat64).reshape({80,3});result.valid=torch::ones({80},torch::kBool);return result;
}
std::vector<ev::Association> pairs(int64_t begin,int64_t end) {
  std::vector<ev::Association> result;
  for(int64_t j=begin;j<end;++j)for(int64_t horizon:{1,4}) {
    ev::Association pair;pair.pair_id=j*10+horizon;pair.origin_row_id=j;pair.target_row_id=j+horizon;
    pair.horizon_ticks=horizon;pair.decision_ms=j*10+3;pair.history_row_ids={j-1,j};
    pair.dependence_group="artificial-block";result.push_back(pair);
  }
  return result;
}
dc::Config config() {
  dc::Config c;c.width=3;c.history_size=2;c.hidden_width=5;c.horizons={1,4};c.updates=6;c.microbatch=4;
  c.repetition_seed=7;c.fit_cutoff_ms=400;c.train_begin_ms=0;c.train_end_ms=390;c.parameter_budget=1000;return c;
}
dc::Identity identity(const da::Context& context) {
  dc::Identity result;result.coordinate_id="artificial-native3";result.source=av::sha256_bytes("artificial source");result.split="artificial TRAIN";
  result.pair_ledger=av::sha256_bytes("artificial immutable pair ledger");result.encoder_model_ref=context.encoder.model_ref;
  result.numeric_card=context.protocol_sha256;return result;
}
da::Context fitted_context() {
  auto result=context();result.fit_cutoff_ms=400;result.cadence_ms=10;
  result.coordinates_sha256=av::sha256_bytes("artificial-native3");
  const auto id=identity(result);result.population_sha256=av::digest_fields("latent-state-control-population-v1",{
    id.source,id.pair_ledger,std::to_string(result.fit_cutoff_ms)});return result;
}
std::string pin(const fs::path& archive) {return av::sha256_file(av::sidecar_path(archive));}
void same_resume(const dc::Trainer& a,const dc::Trainer& b) {
  check(a.model_state_sha256()==b.model_state_sha256(),"resumed parameters/transforms/fit metadata are exact");
  check(a.horizons.size()==b.horizons.size(),"same resumed horizon count");
  for(size_t j=0;j<a.horizons.size();++j) {
    const auto& x=a.horizons[j];const auto& y=b.horizons[j];
    check(x.completed_updates==y.completed_updates&&x.sampler_draws==y.sampler_draws,"exact horizon progress and sampler count");
    std::ostringstream xs,ys;xs<<x.sampler;ys<<y.sampler;check(xs.str()==ys.str(),"exact serialized local sampler state");
    const auto xp=x.model->parameters(),yp=y.model->parameters();
    check(x.optimizer->state().size()==y.optimizer->state().size(),"same optimizer state population");
    for(size_t k=0;k<xp.size();++k) {
      const auto xi=x.optimizer->state().find(xp[k].unsafeGetTensorImpl()),yi=y.optimizer->state().find(yp[k].unsafeGetTensorImpl());
      check((xi==x.optimizer->state().end())==(yi==y.optimizer->state().end()),"same populated parameter slots");
      if(xi==x.optimizer->state().end())continue;
      const auto* xx=dynamic_cast<const torch::optim::AdamWParamState*>(xi->second.get());
      const auto* yy=dynamic_cast<const torch::optim::AdamWParamState*>(yi->second.get());
      check(xx&&yy&&xx->step()==yy->step()&&exact(xx->exp_avg(),yy->exp_avg())&&exact(xx->exp_avg_sq(),yy->exp_avg_sq()),
        "exact resumed AdamW steps and first/second moments");
    }
  }
}
void fitted_publication_and_rejection(const fs::path& directory) {
  const auto context=fitted_context();const auto cfg=config();const auto id=identity(context);const auto data=bank();
  const auto train=pairs(6,31),evaluation=pairs(106,131);dc::Trainer trainer(cfg,id,data,train);
  trainer.step(1);trainer.step(4);
  const auto training=directory/"training.pt";const auto training_manifest=da::save_training(trainer,training,context);
  auto resumed=da::load_training(training,pin(training),training_manifest.component.model_ref,cfg,id,context);
  same_resume(trainer,*resumed);trainer.step(1);resumed->step(1);trainer.step(4);resumed->step(4);same_resume(trainer,*resumed);
  auto changed_cfg=cfg;changed_cfg.learning_rate*=2;
  rejects([&]{da::load_training(training,pin(training),training_manifest.component.model_ref,changed_cfg,id,context);},"resume rejects changed optimizer recipe");
  const auto inference=directory/"inference.pt";const auto manifest=da::save_inference(trainer,inference,context);
  const auto frozen=da::load_inference(inference,pin(inference),manifest.component.model_ref,cfg,id,context);
  const auto prediction=frozen.predict(data,evaluation);
  check(exact(prediction.values,trainer.predict(data,evaluation).values),"checked inference reproduces published prediction bytes");
  {torch::NoGradGuard guard;trainer.horizons[0].model->parameters()[0].add_(.1);}
  check(exact(prediction.values,frozen.predict(data,evaluation).values),"checked inference owns isolated frozen weights");
  rejects([&]{da::load_training(inference,pin(inference),manifest.component.model_ref,cfg,id,context);},"inference archive cannot resume");
  rejects([&]{da::load_inference(training,pin(training),training_manifest.component.model_ref,cfg,id,context);},"training archive cannot serve as inference");
  const auto false_kind=directory/"false-inference-kind.pt";fs::copy_file(training,false_kind);
  auto false_kind_manifest=training_manifest;false_kind_manifest.archive_kind="inference";false_kind_manifest.resume_identity_sha256.clear();
  da::publish_sidecar(false_kind,false_kind_manifest);
  rejects([&]{da::load_inference(false_kind,pin(false_kind),false_kind_manifest.component.model_ref,cfg,id,context);},"underlying training archive cannot masquerade as inference with relabeled sidecar");
  rejects([&]{da::load_inference(inference,"",manifest.component.model_ref,cfg,id,context);},"missing sidecar pin rejected");
  rejects([&]{da::load_inference(inference,pin(inference),"",cfg,id,context);},"missing fitted ModelRef pin rejected");
  rejects([&]{da::load_inference(inference,pin(inference),av::sha256_bytes("other fitted control"),cfg,id,context);},"substituted fitted control rejected");
  const auto missing=directory/"missing-sidecar.pt";fs::copy_file(inference,missing);
  rejects([&]{da::load_inference(missing,pin(inference),manifest.component.model_ref,cfg,id,context);},"missing adjacent sidecar rejected");
  changed_cfg=cfg;changed_cfg.hidden_width++;
  rejects([&]{da::load_inference(inference,pin(inference),manifest.component.model_ref,changed_cfg,id,context);},"wrong architecture rejected");
  auto changed_context=context;changed_context.fit_cutoff_ms++;
  rejects([&]{da::load_inference(inference,pin(inference),manifest.component.model_ref,cfg,id,changed_context);},"changed fit cutoff rejected");
  changed_context=context;std::reverse(changed_context.channel_ids.begin(),changed_context.channel_ids.end());
  rejects([&]{da::load_inference(inference,pin(inference),manifest.component.model_ref,cfg,id,changed_context);},"semantic coordinate reorder rejected");
  changed_context=context;changed_context.protocol_sha256=av::sha256_bytes("different protocol registration");
  rejects([&]{da::load_inference(inference,pin(inference),manifest.component.model_ref,cfg,id,changed_context);},"trainer cannot load under another protocol registration");
  const auto wrong_protocol=directory/"wrong-protocol.pt";
  rejects([&]{da::save_inference(trainer,wrong_protocol,changed_context);},"trainer cannot publish under another protocol registration");
  check(!fs::exists(wrong_protocol)&&!fs::exists(av::sidecar_path(wrong_protocol)),"protocol mismatch publishes no bytes");
  changed_context=context;changed_context.population_sha256=av::sha256_bytes("different population");
  rejects([&]{da::load_inference(inference,pin(inference),manifest.component.model_ref,cfg,id,changed_context);},"trainer cannot load under another population");
  const auto wrong_population=directory/"wrong-population.pt";
  rejects([&]{da::save_inference(trainer,wrong_population,changed_context);},"trainer cannot publish under another population");
  check(!fs::exists(wrong_population),"population mismatch publishes no bytes");
  changed_context=context;changed_context.encoder=av::make_component(context.encoder.module_ref,av::sha256_bytes("different Encoder weights"),context.encoder.inference_config_sha256);
  auto changed_id=id;changed_id.encoder_model_ref=changed_context.encoder.model_ref;
  rejects([&]{da::load_inference(inference,pin(inference),manifest.component.model_ref,cfg,changed_id,changed_context);},"stale/substituted Encoder parent rejected");
  auto variant=[&](const std::string& name,da::Manifest altered) {
    const auto path=directory/(name+".pt");fs::copy_file(inference,path);da::publish_sidecar(path,altered);return path;
  };
  auto altered=manifest;altered.component.state_sha256=av::sha256_bytes("false control state");
  altered.component.model_ref=av::derive_model_ref(altered.component);auto path=variant("false-state",altered);
  rejects([&]{da::load_inference(path,pin(path),altered.component.model_ref,cfg,id,context);},"complete state digest independently recomputed");
  altered=manifest;altered.component.inference_config_sha256=av::sha256_bytes("false mapping");
  altered.component.model_ref=av::derive_model_ref(altered.component);path=variant("false-config",altered);
  rejects([&]{da::load_inference(path,pin(path),altered.component.model_ref,cfg,id,context);},"mapping digest independently recomputed");
  altered=manifest;altered.component.module_ref.source_sha256=av::sha256_bytes("false source");
  altered.component.model_ref=av::derive_model_ref(altered.component);path=variant("false-source",altered);
  rejects([&]{da::load_inference(path,pin(path),altered.component.model_ref,cfg,id,context);},"module source digest independently recomputed");
  // Copy bytes into a private fixture; never edit live/evaluated source.
  const auto copied_root=directory/"source-copy";
  for(const auto& relative:{
      "code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/control.h",
      "code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/artifact.h",
      "code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/sidecar.h",
      "code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/tensor_state.h",
      "code/operations/forecasting/evaluation/versions/v1/include/cuwacunu/evaluation/latent_dynamics/v1/baselines.h",
      "code/operations/forecasting/evaluation/versions/v1/src/baselines.cpp"}) {
    fs::create_directories((copied_root/relative).parent_path());fs::copy_file(context.repository_root/relative,copied_root/relative);
  }
  changed_context=context;changed_context.repository_root=copied_root;
  (void)da::load_inference(inference,pin(inference),manifest.component.model_ref,cfg,id,changed_context);
  {std::ofstream file(copied_root/"code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/control.h",std::ios::app);
    file<<"\n// artificial post-compile source drift\n";}
  rejects([&]{da::load_inference(inference,pin(inference),manifest.component.model_ref,cfg,id,changed_context);},"compiled executable rejects changed runtime source closure");
  const auto stale_source_publication=directory/"stale-source-publication.pt";
  rejects([&]{da::save_inference(trainer,stale_source_publication,changed_context);},"stale executable cannot publish changed source identity");
  check(!fs::exists(stale_source_publication),"source drift publishes no archive");
  const auto old_scale=trainer.horizons[0].model->input_scale.clone();
  trainer.horizons[0].model->input_scale.add_(.1);
  rejects([&]{da::save_inference(trainer,directory/"changed-transform.pt",context);},"runtime frozen transform mutation rejected");
  trainer.horizons[0].model->input_scale.copy_(old_scale);
  path=variant("changed-bytes",manifest);{std::ofstream file(path,std::ios::binary|std::ios::app);file<<"changed";}
  rejects([&]{da::load_inference(path,pin(path),manifest.component.model_ref,cfg,id,context);},"changed raw archive rejected");
  rejects([&]{da::save_inference(trainer,inference,context);},"immutable fitted archive cannot be replaced");
  const auto output=directory/"prediction.pt";ev::save_predictions(output.string(),prediction);
  const auto receipt=da::make_comparison(context.protocol_sha256,context.population_sha256,context.coordinates_sha256,context.encoder.model_ref,
    {{"residual-mlp.seed7",manifest.component.model_ref,pin(inference),av::sha256_file(output)}});
  da::verify_comparison(receipt,"residual-mlp.seed7",frozen,output);
  const auto checked_output=da::load_comparison_predictions(receipt,"residual-mlp.seed7",frozen,output);
  check(exact(checked_output.values,prediction.values)&&exact(checked_output.valid,prediction.valid),"scoring receives exactly the verified immutable output bytes");
  auto wrong_metadata=prediction;wrong_metadata.fit_cutoff_ms++;
  const auto wrong_output=directory/"wrong-output-metadata.pt";ev::save_predictions(wrong_output.string(),wrong_metadata);
  const auto wrong_output_receipt=da::make_comparison(context.protocol_sha256,context.population_sha256,context.coordinates_sha256,context.encoder.model_ref,
    {{"residual-mlp.seed7",manifest.component.model_ref,pin(inference),av::sha256_file(wrong_output)}});
  rejects([&]{da::load_comparison_predictions(wrong_output_receipt,"residual-mlp.seed7",frozen,wrong_output);},"even pinned output metadata must match fitted control configuration");
  rejects([&]{da::verify_comparison(receipt,"missing-role",frozen,output);},"missing comparison control role rejected");
  auto substituted=receipt;substituted.controls[0].model_ref=av::sha256_bytes("substituted control");substituted.bundle_ref=da::derive_bundle_ref(substituted);
  rejects([&]{da::verify_comparison(substituted,"residual-mlp.seed7",frozen,output);},"comparison rejects independently valid substituted control identity");
  {std::ofstream file(output,std::ios::binary|std::ios::app);file<<"changed output";}
  rejects([&]{da::verify_comparison(receipt,"residual-mlp.seed7",frozen,output);},"comparison rejects changed frozen output bytes before scoring");
  rejects([&]{da::load_comparison_predictions(receipt,"residual-mlp.seed7",frozen,output);},"checked output loader rejects changed bytes");
}
void manifest_and_receipt_grammar(const fs::path& directory) {
  const auto c=context();
  av::Component component{{std::string(da::component_name),"v1",av::sha256_bytes("artificial control source")},
    av::sha256_bytes("complete artificial control state"),da::context_sha256(c),{},
    {{"embedding","Encoder",c.encoder.model_ref}}};
  component.model_ref=av::derive_model_ref(component);da::validate_component(component);
  da::Manifest manifest{"inference",av::sha256_bytes("artificial raw archive"),{},component};
  const auto json=da::serialize_manifest(manifest);check(da::parse_manifest(json)==manifest,"control sidecar round trip");
  rejects([&]{av::parse_manifest(json);},"shared historical parser does not silently accept control schema");
  auto altered=manifest;altered.component.module_ref.component="Dynamics Model";
  altered.component.model_ref=av::derive_model_ref(altered.component);
  rejects([&]{da::serialize_manifest(altered);},"control never impersonates Dynamics Model");
  altered=manifest;altered.component.parents[0].role="input_history";altered.component.model_ref=av::derive_model_ref(altered.component);
  rejects([&]{da::serialize_manifest(altered);},"wrong Encoder parent role rejected");
  altered=manifest;altered.resume_identity_sha256=av::sha256_bytes("unwanted resume");
  rejects([&]{da::serialize_manifest(altered);},"inference sidecar cannot contain training resume identity");
  altered=manifest;altered.archive_kind="training";
  rejects([&]{da::serialize_manifest(altered);},"training sidecar requires resume identity");
  auto duplicate=json;duplicate.insert(1,"\"archive_kind\":\"inference\",");
  rejects([&]{da::parse_manifest(duplicate);},"duplicate JSON binding rejected");
  auto unknown=json;unknown.insert(1,"\"unexpected\":\"value\",");
  rejects([&]{da::parse_manifest(unknown);},"unknown JSON binding rejected");
  rejects([&]{da::parse_manifest(json+"trailing");},"trailing sidecar bytes rejected");
  const auto output=torch::arange(18,torch::kFloat64).reshape({3,2,3});const auto origins=torch::tensor({10,20,30},torch::kInt64);
  const auto output_sha=da::prediction_output_sha256(output,origins,{1,9},c.coordinates_sha256);
  check(output_sha!=da::prediction_output_sha256(output,origins,{1,10},c.coordinates_sha256),"output identity includes horizon association");
  auto reordered=torch::tensor({30,20,10},torch::kInt64);
  check(output_sha!=da::prediction_output_sha256(output,reordered,{1,9},c.coordinates_sha256),"output identity includes origin order");
  rejects([&]{da::prediction_output_sha256(output,torch::tensor({10,10,30},torch::kInt64),{1,9},c.coordinates_sha256);},"duplicate origins rejected");
  const auto receipt=da::make_comparison(c.protocol_sha256,c.population_sha256,c.coordinates_sha256,c.encoder.model_ref,
    {{"residual-mlp.seed7",component.model_ref,av::sha256_bytes(json),output_sha}});
  check(da::parse_comparison(da::serialize_comparison(receipt))==receipt,"comparison receipt round trip");
  const auto path=directory/"comparison.json";const auto pin=da::publish_comparison(path,receipt);
  check(da::read_comparison(path,pin,c.protocol_sha256,c.population_sha256,c.coordinates_sha256,c.encoder.model_ref)==receipt,
    "pinned immutable comparison admitted");
  rejects([&]{da::publish_comparison(path,receipt);},"immutable comparison cannot be replaced");
  rejects([&]{da::read_comparison(path,"",c.protocol_sha256,c.population_sha256,c.coordinates_sha256,c.encoder.model_ref);},"missing comparison pin rejected");
  rejects([&]{da::read_comparison(path,pin,av::sha256_bytes("changed protocol"),c.population_sha256,c.coordinates_sha256,c.encoder.model_ref);},"stale protocol rejected");
  auto changed=receipt;changed.controls[0].model_ref=av::sha256_bytes("substituted fitted control");
  rejects([&]{da::serialize_comparison(changed);},"control substitution changes immutable BundleRef");
  changed=receipt;changed.controls[0].output_sha256.clear();changed.bundle_ref=da::derive_bundle_ref(changed);
  rejects([&]{da::serialize_comparison(changed);},"missing control output pin rejected");
  auto semantic=c;std::reverse(semantic.channel_ids.begin(),semantic.channel_ids.end());
  check(da::context_sha256(c)!=da::context_sha256(semantic),"context binds semantic channel ordering");
  semantic=c;semantic.fit_cutoff_ms++;
  check(da::context_sha256(c)!=da::context_sha256(semantic),"context binds TRAIN fit cutoff");
}
} // namespace
int main() try {
  torch::set_num_threads(1);Temporary directory;
  const auto rng_before=at::detail::getDefaultCPUGenerator().get_state();
  manifest_and_receipt_grammar(directory.path);
  fitted_publication_and_rejection(directory.path);
  check(exact(rng_before,at::detail::getDefaultCPUGenerator().get_state()),"artifact APIs preserve global CPU RNG");
  std::cout<<"PASS direct control artifacts v1: strict control sidecar, exact AdamW/local sampler/progress resume, isolated inference, immutable fitted/comparison/BundleRef/output identities and substitution rejection; artificial engineering only\n";
  return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
