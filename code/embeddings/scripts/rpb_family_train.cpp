// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Bounded portable RPB example: numerical work stays in C++; no quality scoring.
#include "embedding/encoders/raw_patch_bottleneck_mae/family/training.h"
#include "embedding/data/observation_windows.h"
#include "cuwacunu/artifacts/v1/sidecar.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>

namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
namespace family=rpb::family;
namespace ow=embedding::data::observation_windows;
namespace digest=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;
namespace {
std::string quote(const std::string &value){
  std::ostringstream out;out<<'"';for(const unsigned char c:value){
    if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;
  }out<<'"';return out.str();
}
void write_text(const fs::path &path,const std::string &text){
  TORCH_CHECK(!fs::exists(path),"[RPB family train] output exists: ",path.string());
  std::ofstream out(path,std::ios::binary);TORCH_CHECK(out,"[RPB family train] open output");out<<text<<'\n';out.close();
  TORCH_CHECK(out,"[RPB family train] write output");
}
std::string artifact(const fs::path &path){
  std::ostringstream out;out<<"{\"path\":"<<quote(path.generic_string())<<",\"bytes\":"<<fs::file_size(path)
    <<",\"sha256\":"<<quote(digest::sha256_file(path))<<'}';return out.str();
}
rpb::Input crop(const ow::Dataset &data,const family::Config &config){
  TORCH_CHECK(std::abs(data.sampling_interval-config.sampling_interval)<=1e-9*std::max(data.sampling_interval,config.sampling_interval),
      "[RPB family train] dataset/config sampling interval mismatch");
  return rpb::prepare_endpoint_history(data.values,data.observed,data.channel_ids,data.timestamps,data.cutoffs,
      rpb::configurable_spectral_input_config(config));
}
std::string schema_identity(const family::Config &config){
  std::ostringstream out;out<<std::setprecision(17)<<"observation-window-f64-bool-retained-endpoint-v1;C="<<config.channel_count
    <<";H="<<config.history_length<<";F="<<config.input_width<<";interval="<<config.sampling_interval<<";ids=";
  for(const auto id:rpb::resolved_channel_ids(config))out<<id<<',';
  return digest::sha256_bytes(out.str());
}
void help(){
  std::cout<<"rpb_family_train --settings FILE --dataset FILE --output NEWLEAF --steps TOTAL --max-attempts TOTAL --batch-size N\n"
    <<"  --initialization-seed I --row-seed I --mask-seed I --context-seed I [--resume TRAINING_STATE --expected-scaler ID] [--save-every N]\n"
    <<"Configurable RPB-v22, CUDA float32, fixed AdamW lr=.001 wd=.0001 clip=1, context deletion none.\n"
    <<"Dataset must be observation_window_dataset_v1 with caller-declared TRAIN role. Its real membership is not authenticated here.\n"
    <<"Each endpoint crops its retained H ticks before fitting/use. Within-window attention is bidirectional, not prefix-causal.\n"
    <<"Fresh fit visits TRAIN rows once in source order/batch chunks. Resume never refits and requires an independently supplied scaler identity.\n"
    <<"Rows use mt19937_64(counter_seed(row_seed,attempt)) modulo row count with replacement; masking uses the existing patch masker.\n"
    <<"Steps/attempt limits are TOTAL ceilings and may increase on resume. All other recipe/input/executable bytes and seeds must match.\n";
}
int execute(int argc,char **argv){
  if(argc==2&&std::string(argv[1])=="--help"){help();return 0;}
  TORCH_CHECK(argc>1&&(argc-1)%2==0,"[RPB family train] use --help; options require values");
  const std::set<std::string> allowed{"--settings","--dataset","--output","--steps","--max-attempts","--batch-size",
    "--initialization-seed","--row-seed","--mask-seed","--context-seed","--resume","--expected-scaler","--save-every"};
  std::map<std::string,std::string> options;
  for(int i=1;i<argc;i+=2)TORCH_CHECK(allowed.count(argv[i])&&options.emplace(argv[i],argv[i+1]).second,"[RPB family train] unknown or duplicate option");
  for(const auto key:{"--settings","--dataset","--output","--steps","--max-attempts","--batch-size","--initialization-seed","--row-seed","--mask-seed","--context-seed"})
    TORCH_CHECK(options.count(key),"[RPB family train] missing option: ",key);
  const bool resuming=options.count("--resume");
  TORCH_CHECK(resuming==bool(options.count("--expected-scaler")),"[RPB family train] --resume and independent --expected-scaler are required together");
  const auto integer=[&](const std::string &key,int64_t fallback)->int64_t{
    if(!options.count(key))return fallback;
    size_t consumed=0;const auto n=std::stoll(options.at(key),&consumed);
    TORCH_CHECK(consumed==options.at(key).size()&&n>=0,"[RPB family train] nonnegative integer required: ",key);return n;
  };
  const int64_t target=integer("--steps",0),limit=integer("--max-attempts",0),batch_size=integer("--batch-size",0),
    initialization_seed=integer("--initialization-seed",0),row_seed=integer("--row-seed",0),mask_seed=integer("--mask-seed",0),
    context_seed=integer("--context-seed",0),save_every=integer("--save-every",0);
  TORCH_CHECK(target>0&&limit>=target&&limit<std::numeric_limits<int64_t>::max()-1&&batch_size>0,
      "[RPB family train] positive target/batch and bounded attempt ceiling >= target required");
  const fs::path settings=options.at("--settings"),input_path=options.at("--dataset"),output=options.at("--output");
  TORCH_CHECK(fs::is_regular_file(settings)&&fs::is_regular_file(input_path)&&!fs::exists(output),"[RPB family train] inputs must exist and output must be new");
  const auto settings_sha=digest::sha256_file(settings),input_sha=digest::sha256_file(input_path);
  const auto executable=fs::canonical("/proc/self/exe");const auto executable_sha=digest::sha256_file(executable);
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  const auto config=family::read_settings(settings.string());const auto dataset=ow::read(input_path);
  TORCH_CHECK(dataset.role=="TRAIN","[RPB family train] only caller-declared TRAIN may fit/train");
  TORCH_CHECK(batch_size<=std::numeric_limits<int64_t>::max()/dataset.values[0].numel(),"[RPB family train] batch cell overflow");
  TORCH_CHECK(torch::cuda::is_available(),"[RPB family train] actual CUDA required");
  TORCH_CHECK(digest::sha256_file(settings)==settings_sha&&digest::sha256_file(input_path)==input_sha,"[RPB family train] inputs changed while reading");
  const auto source_binding=dataset.source_identity+";archive-sha256="+input_sha;
  std::ostringstream recipe;recipe<<"rpb-family-train-example-v1;executable="<<executable_sha<<";settings="<<settings_sha<<";dataset="<<input_sha
    <<";config="<<family::config_identity(config)<<";initialization-seed="<<initialization_seed<<";batch="<<batch_size
    <<";fit=all-TRAIN-source-order-batch-chunks-streaming-Chan-v1;rows=mt19937_64-counter-seed-mod-N-with-replacement-v1"
    <<";mask=original-patch-mt19937_64-shuffle-v1;context=none;row-seed="<<row_seed<<";mask-seed="<<mask_seed<<";context-seed="<<context_seed
    <<";AdamW-lr=.001;wd=.0001;betas=.9,.999;eps=1e-8;amsgrad=off;clip=1;attempt-boundary-v1";
  const family::TrainingIdentity identity{source_binding,dataset.split_identity,recipe.str()};
  family::TrainingProgress progress{0,0,row_seed,mask_seed,context_seed};family::Model model{nullptr};family::FrozenScaler scaler;
  std::unique_ptr<torch::optim::AdamW> optimizer;std::string resume_sha;int64_t fit_rows=0;
  if(resuming){
    const fs::path resume=options.at("--resume");TORCH_CHECK(fs::is_regular_file(resume),"[RPB family train] resume state required");
    resume_sha=digest::sha256_file(resume);
    auto restored=family::load_training_state(resume.string(),config,identity,options.at("--expected-scaler"),progress);
    TORCH_CHECK(digest::sha256_file(resume)==resume_sha,"[RPB family train] resume input changed while reading");
    model=restored.model;scaler=restored.scaler;optimizer=std::move(restored.optimizer);progress=restored.progress;
  }else{
    family::PreprocessingFitter fitter(config);
    for(int64_t begin=0;begin<dataset.rows();){
      const auto count=std::min(batch_size,dataset.rows()-begin);std::vector<int64_t> rows;
      for(int64_t i=0;i<count;++i)rows.push_back(begin+i);
      fitter.add(crop(ow::subset(dataset,rows),config));begin+=count;
    }
    scaler=fitter.freeze();fit_rows=dataset.rows();model=family::fresh(config,scaler,initialization_seed);optimizer=family::make_optimizer(model);
  }
  TORCH_CHECK(progress.completed_updates<=target&&progress.attempted_batches<=limit,"[RPB family train] total budgets precede saved progress");
  const auto initial_updates=progress.completed_updates,initial_attempts=progress.attempted_batches;
  const auto starting_state=family::model_identity(model),starting_optimizer=family::optimizer_identity(model,*optimizer);
  if(!output.parent_path().empty())fs::create_directories(output.parent_path());
  TORCH_CHECK(fs::create_directory(output),"[RPB family train] exclusive new output leaf required");
  if(!resuming){
    family::save_inference((output/"inference-0.pt").string(),model,scaler,0);
    torch::serialize::OutputArchive archive,fit;archive.write("artifact_kind",ow::text_tensor("rpb_family_example_preprocessing_v1"),true);
    archive.write("source_identity",ow::text_tensor(source_binding),true);archive.write("split_identity",ow::text_tensor(dataset.split_identity),true);
    archive.write("recipe_identity",ow::text_tensor(identity.recipe_identity),true);archive.write("fit_rows",torch::tensor(fit_rows,torch::kInt64),true);
    scaler.save(fit);archive.write("scaler",fit);rpb::configurable_checkpoint_detail::publish_checkpoint(archive,(output/"preprocessing.pt").string());
    std::ostringstream receipt;receipt<<"{\"kind\":\"rpb_family_preprocessing_fit_v1\",\"scaler_identity\":"<<quote(scaler.identity())
      <<",\"fit_rows\":"<<fit_rows<<",\"source_identity\":"<<quote(source_binding)<<",\"split_identity\":"<<quote(dataset.split_identity)
      <<",\"recipe_identity\":"<<quote(identity.recipe_identity)<<",\"initialization_seed\":"<<initialization_seed<<",\"artifact\":"<<artifact(output/"preprocessing.pt")<<'}';
    write_text(output/"preprocessing-fit.json",receipt.str());
  }
  std::ofstream events(output/"steps.jsonl",std::ios::binary);TORCH_CHECK(events,"[RPB family train] event log open");events<<std::setprecision(17);int64_t skipped=0;
  while(progress.completed_updates<target&&progress.attempted_batches<limit){
    const auto attempt=progress.attempted_batches;std::mt19937_64 generator(rpb::training_counter_seed(row_seed,attempt));std::vector<int64_t> rows;
    for(int64_t i=0;i<batch_size;++i)rows.push_back(static_cast<int64_t>(generator()%static_cast<uint64_t>(dataset.rows())));
    const auto batch=ow::subset(dataset,rows);const auto result=family::train_step(model,scaler,*optimizer,progress,crop(batch,config));
    if(!result.updated)++skipped;
    events<<"{\"attempt\":"<<attempt<<",\"updated\":"<<(result.updated?"true":"false")<<",\"completed_updates\":"<<progress.completed_updates
      <<",\"loss\":"<<result.loss<<",\"gradient_norm\":"<<result.gradient_norm<<",\"eligible_examples\":"<<result.eligible_examples
      <<",\"target_cells\":"<<result.target_cells<<",\"hidden_sha256\":"<<quote(family::tensor_identity(result.hidden))
      <<",\"gradients_sha256\":"<<quote(result.gradients_sha256)<<",\"source_row_ids\":[";
    for(int64_t i=0;i<batch_size;++i){if(i)events<<',';events<<batch.row_ids[i].item<int64_t>();}events<<"]}\n";
    TORCH_CHECK(events,"[RPB family train] event log write");
    if(result.updated&&save_every>0&&progress.completed_updates%save_every==0)
      family::save_training_state((output/("training-"+std::to_string(progress.completed_updates)+".pt")).string(),model,scaler,*optimizer,identity,progress);
  }
  events.close();TORCH_CHECK(events,"[RPB family train] event log close");
  const auto training_path=output/"training-final.pt",inference_path=output/"inference-final.pt",embeddings_path=output/"embeddings.pt";
  family::save_training_state(training_path.string(),model,scaler,*optimizer,identity,progress);
  family::save_inference(inference_path.string(),model,scaler,progress.completed_updates);
  auto loaded=family::load_inference(inference_path.string(),config.device,&config);
  TORCH_CHECK(loaded.training_updates==progress.completed_updates&&family::model_identity(model)==family::model_identity(loaded.model)&&scaler.identity()==loaded.scaler.identity(),
      "[RPB family train] inference reload state self-check failed");
  std::vector<torch::Tensor> native,valid,channel_valid,observation_counts,patch_counts,ids;
  for(int64_t begin=0;begin<dataset.rows();){
    const auto count=std::min(batch_size,dataset.rows()-begin);std::vector<int64_t> rows;for(int64_t i=0;i<count;++i)rows.push_back(begin+i);
    const auto raw=crop(ow::subset(dataset,rows),config);const auto live=family::encode(model,scaler,raw),reloaded=family::encode(loaded.model,loaded.scaler,raw);
    TORCH_CHECK(torch::equal(live.z_contextual_global,reloaded.z_contextual_global)&&torch::equal(live.sample_valid_mask,reloaded.sample_valid_mask),
      "[RPB family train] inference reload embedding self-check failed");
    native.push_back(live.z_contextual_global.to(torch::kCPU));valid.push_back(live.sample_valid_mask.to(torch::kCPU));
    channel_valid.push_back(live.channel_valid_mask.to(torch::kCPU));observation_counts.push_back(live.visible_observation_counts.to(torch::kCPU));
    patch_counts.push_back(live.visible_patch_counts.to(torch::kCPU));ids.push_back(live.channel_ids.to(torch::kCPU));begin+=count;
  }
  const auto exported=torch::cat(native,0),sample_support=torch::cat(valid,0),channel_support=torch::cat(channel_valid,0),
    observation_support=torch::cat(observation_counts,0),patch_support=torch::cat(patch_counts,0),semantic_ids=torch::cat(ids,0);
  TORCH_CHECK(exported.scalar_type()==torch::kFloat32&&exported.sizes()==torch::IntArrayRef({dataset.rows(),config.export_width})&&
    sample_support.scalar_type()==torch::kBool&&sample_support.sizes()==torch::IntArrayRef({dataset.rows()})&&
    channel_support.scalar_type()==torch::kBool&&channel_support.sizes()==torch::IntArrayRef({dataset.rows(),config.channel_count})&&
    observation_support.scalar_type()==torch::kInt64&&observation_support.sizes()==channel_support.sizes()&&
    patch_support.scalar_type()==torch::kInt64&&patch_support.sizes()==channel_support.sizes()&&
    semantic_ids.scalar_type()==torch::kInt64&&semantic_ids.sizes()==channel_support.sizes()&&
    torch::isfinite(exported).all().item<bool>()&&observation_support.ge(0).all().item<bool>()&&patch_support.ge(0).all().item<bool>(),
      "[RPB family train] export tensor schema mismatch");
  torch::serialize::OutputArchive features;
  features.write("artifact_kind",ow::text_tensor("rpb_family_endpoint_embeddings_v1"),true);
  features.write("model_tag",ow::text_tensor(rpb::kContentOrderedRelationModelTag),true);
  features.write("architecture_id",ow::text_tensor(rpb::kContentOrderedRelationArchitectureId),true);
  features.write("native_layout",ow::text_tensor(rpb::kContentOrderedRelationLayout),true);
  features.write("output_selector",ow::text_tensor("z_contextual_global"),true);
  features.write("source_identity",ow::text_tensor(source_binding),true);features.write("split_identity",ow::text_tensor(dataset.split_identity),true);
  features.write("role",ow::text_tensor(dataset.role),true);features.write("recipe_identity",ow::text_tensor(identity.recipe_identity),true);
  features.write("schema_identity",ow::text_tensor(schema_identity(config)),true);features.write("config_identity",ow::text_tensor(family::config_identity(config)),true);
  features.write("scaler_identity",ow::text_tensor(scaler.identity()),true);features.write("inference_sha256",ow::text_tensor(digest::sha256_file(inference_path)),true);
  features.write("native",exported,true);features.write("sample_valid_mask",sample_support,true);features.write("channel_valid_mask",channel_support,true);
  features.write("visible_observation_counts",observation_support,true);features.write("visible_patch_counts",patch_support,true);
  features.write("channel_ids",semantic_ids,true);features.write("row_ids",dataset.row_ids,true);features.write("endpoints",dataset.cutoffs,true);
  features.write("sampling_interval",torch::tensor(dataset.sampling_interval,torch::kFloat64),true);
  rpb::configurable_checkpoint_detail::publish_checkpoint(features,embeddings_path.string());
  // Read the actual published archive: a matching in-memory feature result alone
  // is not evidence that its output schema and storage round-tripped.
  torch::serialize::InputArchive published;published.load_from(embeddings_path.string(),torch::kCPU);
  const std::vector<std::pair<const char*,std::string>> expected_text{
    {"artifact_kind","rpb_family_endpoint_embeddings_v1"},{"model_tag",rpb::kContentOrderedRelationModelTag},
    {"architecture_id",rpb::kContentOrderedRelationArchitectureId},{"native_layout",rpb::kContentOrderedRelationLayout},
    {"output_selector","z_contextual_global"},{"source_identity",source_binding},{"split_identity",dataset.split_identity},
    {"role",dataset.role},{"recipe_identity",identity.recipe_identity},{"schema_identity",schema_identity(config)},
    {"config_identity",family::config_identity(config)},{"scaler_identity",scaler.identity()},
    {"inference_sha256",digest::sha256_file(inference_path)}};
  for(const auto &[key,value]:expected_text)TORCH_CHECK(rpb::configurable_checkpoint_detail::read_text(published,key)==value,
      "[RPB family train] published embedding text mismatch: ",key);
  const std::vector<std::pair<const char*,torch::Tensor>> expected_tensors{
    {"native",exported},{"sample_valid_mask",sample_support},{"channel_valid_mask",channel_support},
    {"visible_observation_counts",observation_support},{"visible_patch_counts",patch_support},{"channel_ids",semantic_ids},
    {"row_ids",dataset.row_ids},{"endpoints",dataset.cutoffs},{"sampling_interval",torch::tensor(dataset.sampling_interval,torch::kFloat64)}};
  for(const auto &[key,value]:expected_tensors){
    torch::Tensor actual;published.read(key,actual,true);
    TORCH_CHECK(actual.device().is_cpu()&&actual.scalar_type()==value.scalar_type()&&actual.sizes()==value.sizes()&&torch::equal(actual,value),
      "[RPB family train] published embedding tensor shape/dtype/value mismatch: ",key);
  }
  TORCH_CHECK(digest::sha256_file(settings)==settings_sha&&digest::sha256_file(input_path)==input_sha&&digest::sha256_file(executable)==executable_sha,
      "[RPB family train] bound input/executable changed during run");
  if(resuming)TORCH_CHECK(digest::sha256_file(options.at("--resume"))==resume_sha,"[RPB family train] original resume input changed");
  const bool passed=progress.completed_updates==target;const auto parameters=family::parameter_inventory(model);std::ostringstream report;report<<std::setprecision(17)
    <<"{\"kind\":\"rpb_family_training_example_run_v1\",\"status\":"<<quote(passed?"passed":"attempt_budget_exhausted")<<",\"resume\":"<<(resuming?"true":"false")
    <<",\"settings\":"<<artifact(settings)<<",\"dataset\":"<<artifact(input_path)<<",\"executable\":"<<artifact(executable)
    <<",\"source_identity\":"<<quote(dataset.source_identity)<<",\"split_identity\":"<<quote(dataset.split_identity)<<",\"declared_role\":\"TRAIN\",\"split_membership_independently_verified\":false"
    <<",\"recipe_identity\":"<<quote(identity.recipe_identity)<<",\"target_total_updates\":"<<target<<",\"maximum_total_attempts\":"<<limit<<",\"batch_size\":"<<batch_size
    <<",\"initialization_seed\":"<<initialization_seed<<",\"row_seed\":"<<row_seed<<",\"mask_seed\":"<<mask_seed<<",\"context_seed\":"<<context_seed
    <<",\"context_deletion\":\"none\",\"learning_rate\":0.001,\"weight_decay\":0.0001,\"clip_norm\":1,\"completed_updates\":"<<progress.completed_updates
    <<",\"attempted_batches\":"<<progress.attempted_batches<<",\"starting_completed_updates\":"<<initial_updates<<",\"starting_attempted_batches\":"<<initial_attempts
    <<",\"physical_training_updates_this_invocation\":"<<progress.completed_updates-initial_updates<<",\"skipped_attempts_this_invocation\":"<<skipped
    <<",\"preprocessing_fitted\":"<<(!resuming?"true":"false")<<",\"preprocessing_fit_rows\":"<<fit_rows<<",\"native_width\":"<<config.export_width
    <<",\"parameter_count\":"<<parameters.total<<",\"trainable_parameter_count\":"<<parameters.trainable<<",\"final_state_identity\":"<<quote(family::model_identity(model))
    <<",\"starting_state_identity\":"<<quote(starting_state)<<",\"starting_optimizer_identity\":"<<quote(starting_optimizer)
    <<",\"optimizer_identity\":"<<quote(family::optimizer_identity(model,*optimizer))<<",\"scaler_identity\":"<<quote(scaler.identity())<<",\"embeddings_identity\":"<<quote(family::tensor_identity(exported))
    <<",\"final_inference_reload_verified\":true,\"embedding_export_reload_verified\":true,\"quality_evaluated\":false,\"artifacts\":{\"training_state\":"<<artifact(training_path)<<",\"inference\":"<<artifact(inference_path)
    <<",\"embeddings\":"<<artifact(embeddings_path)<<",\"steps\":"<<artifact(output/"steps.jsonl");
  if(!resuming)report<<",\"preprocessing\":"<<artifact(output/"preprocessing.pt")<<",\"fit_receipt\":"<<artifact(output/"preprocessing-fit.json")<<",\"initial_inference\":"<<artifact(output/"inference-0.pt");
  if(resuming)report<<",\"resume_input\":"<<artifact(options.at("--resume"));
  report<<"}}";
  write_text(output/"run.json",report.str());std::cout<<report.str()<<std::endl;return passed?0:2;
}
}
int main(int argc,char **argv){try{return execute(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<std::endl;return 1;}}
