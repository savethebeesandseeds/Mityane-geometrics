// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Portable single-device example: no market reader, inherited checkpoint or quality scoring.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v18.h"
#include "embedding/data/observation_windows.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace v18=embedding::encoders::causal_patch_bottleneck_mae::v18;
namespace tr=cp::training;
namespace ow=embedding::data::observation_windows;
namespace fs=std::filesystem;
namespace {
std::string quote(const std::string &value){
  std::ostringstream out;out<<'"';for(const unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<c;
    else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<c;}out<<'"';return out.str();
}
void write_text(const fs::path &path,const std::string &text){
  TORCH_CHECK(!fs::exists(path),"[CPB v18 train] existing output cannot be replaced: ",path.string());
  std::ofstream stream(path,std::ios::binary);TORCH_CHECK(stream,"[CPB v18 train] open output");stream<<text<<'\n';stream.close();
  TORCH_CHECK(stream,"[CPB v18 train] write output");
}
std::string artifact(const fs::path &path){
  std::ostringstream out;out<<"{\"path\":"<<quote(path.generic_string())<<",\"bytes\":"<<fs::file_size(path)
    <<",\"sha256\":"<<quote(cp::boundary::sha256_file(path))<<'}';return out.str();
}
cp::RawHistory raw(const ow::Dataset &dataset){return{dataset.values,dataset.observed,dataset.channel_ids,dataset.timestamps,dataset.sampling_interval};}
std::string integers(const torch::Tensor &input){
  TORCH_CHECK(input.device().is_cpu()&&input.scalar_type()==torch::kInt64,"[CPB v18 train] CPU I64 mask metadata required");
  const auto values=input.contiguous().reshape({-1});const auto *data=values.const_data_ptr<int64_t>();
  std::ostringstream out;out<<'[';for(int64_t i=0;i<values.numel();++i){if(i)out<<',';out<<data[i];}out<<']';return out.str();
}
void help(){
  std::cout<<"cpb_v18_train --mask-mode contiguous|scattered --settings FILE --dataset FILE --output NEWLEAF --steps TOTAL --max-attempts TOTAL --batch-size N\n"
    <<"  --row-seed I --mask-seed I --context-seed I (--seed I | --resume TRAINING_STATE)\n"
    <<"  [--learning-rate .001] [--weight-decay .0001] [--clip-norm 1] [--save-every 0]\n"
    <<"Single-device configurable-family example; DATASET must be a typed observation_window_dataset_v1 archive declaring role TRAIN.\n"
    <<"--steps is the TOTAL completed-update target, not additional steps. --max-attempts is a TOTAL attempt ceiling.\n"
    <<"Resume requires identical data/settings/executable bytes, batching, options and stream seeds. Stop budgets may increase.\n"
    <<"Fresh fitting uses all declared TRAIN rows once in source order, in --batch-size chunks; no fit on resume.\n"
    <<"Rows use mt19937_64(counter_seed(row_seed,attempt)) modulo row count, with replacement. Context deletion is none.\n"
    <<"The dataset declaration does not independently prove real TRAIN membership. No scheduler/accumulation/distributed training.\n";
}
int execute(int argc,char **argv){
  if(argc==2&&std::string(argv[1])=="--help"){help();return 0;}
  TORCH_CHECK(argc>1&&(argc-1)%2==0,"[CPB v18 train] use --help; options require values");
  const std::set<std::string> allowed{"--settings","--dataset","--output","--steps","--max-attempts","--batch-size","--seed","--resume",
    "--row-seed","--mask-seed","--context-seed","--mask-mode","--learning-rate","--weight-decay","--clip-norm","--save-every"};
  std::map<std::string,std::string> options;
  for(int i=1;i<argc;i+=2)TORCH_CHECK(allowed.count(argv[i])&&options.emplace(argv[i],argv[i+1]).second,"[CPB v18 train] unknown or duplicate option");
  for(const auto key:{"--settings","--dataset","--output","--steps","--max-attempts","--batch-size","--row-seed","--mask-seed","--context-seed","--mask-mode"})
    TORCH_CHECK(options.count(key),"[CPB v18 train] required option: ",key);
  TORCH_CHECK(bool(options.count("--seed"))!=bool(options.count("--resume")),"[CPB v18 train] choose fresh --seed OR typed --resume");
  const auto integer=[&](const std::string &key,int64_t fallback)->int64_t{
    if(!options.count(key))return fallback;
    size_t consumed=0;const auto value=std::stoll(options.at(key),&consumed);
    TORCH_CHECK(consumed==options.at(key).size()&&value>=0,"[CPB v18 train] nonnegative integer required: ",key);return value;
  };
  const auto number=[&](const std::string &key,double fallback){
    if(!options.count(key))return fallback;
    size_t consumed=0;const auto value=std::stod(options.at(key),&consumed);
    TORCH_CHECK(consumed==options.at(key).size()&&std::isfinite(value),"[CPB v18 train] finite number required: ",key);return value;
  };
  const int64_t target=integer("--steps",0),maximum_attempts=integer("--max-attempts",0),batch_size=integer("--batch-size",0);
  const int64_t row_seed=integer("--row-seed",0),mask_seed=integer("--mask-seed",0),context_seed=integer("--context-seed",0),save_every=integer("--save-every",0);
  TORCH_CHECK(target>0&&maximum_attempts>=target&&maximum_attempts<std::numeric_limits<int64_t>::max()-1&&batch_size>0,
      "[CPB v18 train] positive target/batch, attempt ceiling>=target and incrementable counters required");
  TORCH_CHECK(options.at("--mask-mode")=="contiguous"||options.at("--mask-mode")=="scattered","[CPB v18 train] exact mask mode required");
  const v18::Recipe recipe{options.at("--mask-mode")=="contiguous"?v18::MaskMode::contiguous:v18::MaskMode::scattered};v18::validate_recipe(recipe);
  const tr::OptimizerConfig optimizer_options{number("--learning-rate",.001),number("--weight-decay",.0001),number("--clip-norm",1.)};
  tr::validate_optimizer_config(optimizer_options);
  const fs::path settings=options.at("--settings"),dataset_path=options.at("--dataset"),output=options.at("--output");
  TORCH_CHECK(fs::is_regular_file(settings)&&fs::is_regular_file(dataset_path)&&!fs::exists(output),"[CPB v18 train] settings/dataset must exist and output must be new");
  const auto settings_sha=cp::boundary::sha256_file(settings),dataset_sha=cp::boundary::sha256_file(dataset_path);
  const auto executable_path=fs::canonical("/proc/self/exe");
  const auto executable_sha=cp::boundary::sha256_file(executable_path);
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  const auto config=cp::read_settings(settings);const auto dataset=ow::read(dataset_path);
  TORCH_CHECK(dataset.role=="TRAIN","[CPB v18 train] only a caller-declared TRAIN dataset may train/fit");
  TORCH_CHECK(batch_size<=std::numeric_limits<int64_t>::max()/dataset.values[0].numel(),"[CPB v18 train] batch cell count overflow");
  TORCH_CHECK(cp::boundary::sha256_file(settings)==settings_sha&&cp::boundary::sha256_file(dataset_path)==dataset_sha,"[CPB v18 train] input changed while reading");
  TORCH_CHECK(torch::cuda::is_available(),"[CPB v18 train] actual CUDA required");
  const auto source_binding=dataset.source_identity+";archive-sha256="+dataset_sha;
  std::ostringstream recipe_stream;recipe_stream<<std::setprecision(17)
      <<"cpb-v18-portable-train-v1;executable="<<executable_sha<<";settings="<<settings_sha<<";dataset="<<dataset_sha
      <<";batch="<<batch_size<<";rows=mt19937_64-counter-seed-mod-N-with-replacement-v1;context=none;calibration=all-TRAIN-source-order-batch-chunks-Chan-v1"
      <<";row-seed="<<row_seed<<";mask-seed="<<mask_seed<<";context-seed="<<context_seed
      <<";learning-recipe="<<v18::recipe_identity(recipe)<<";AdamW-lr="<<optimizer_options.learning_rate<<";wd="<<optimizer_options.weight_decay<<";global-clip="<<optimizer_options.clip_norm;
  const tr::TrainingIdentity identity{source_binding,dataset.split_identity,recipe_stream.str()};
  const bool resuming=options.count("--resume");cp::Loaded live;std::unique_ptr<torch::optim::AdamW> optimizer;
  tr::TrainingProgress progress{0,0,row_seed,mask_seed,context_seed};int64_t fit_windows=0;std::string resume_sha;
  if(resuming){
    const fs::path resume=options.at("--resume");TORCH_CHECK(fs::is_regular_file(resume),"[CPB v18 train] resume state must exist");resume_sha=cp::boundary::sha256_file(resume);
    auto restored=v18::load_training_state(resume.string(),config,identity,recipe,optimizer_options);
    TORCH_CHECK(cp::boundary::sha256_file(resume)==resume_sha,"[CPB v18 train] resume artifact changed while reading");
    progress=restored.progress;TORCH_CHECK(progress.row_seed==row_seed&&progress.mask_seed==mask_seed&&progress.context_seed==context_seed,"[CPB v18 train] exact saved stream seeds required");
    live={restored.config,restored.model,restored.scaler,0,progress.completed_updates,restored.model->initialization_seed(),restored.model->initial_state_identity()};optimizer=std::move(restored.optimizer);
  }else{
    tr::PreprocessingFitter fitter(config,{source_binding,dataset.split_identity,"all-TRAIN-source-order-batch-chunks-Chan-v1;batch="+std::to_string(batch_size)});
    for(int64_t begin=0;begin<dataset.rows();){
      std::vector<int64_t> rows;const auto count=std::min(batch_size,dataset.rows()-begin);for(int64_t offset=0;offset<count;++offset)rows.push_back(begin+offset);
      const auto batch=ow::subset(dataset,rows);fitter.add(raw(batch),batch.cutoffs);begin+=count;
    }
    auto fitted=fitter.freeze();fit_windows=fitted.windows;
    live=cp::fresh(config,fitted.scaler,fitted.bank,integer("--seed",0));optimizer=tr::make_optimizer(live.model,optimizer_options);
  }
  TORCH_CHECK(progress.completed_updates<=target,"[CPB v18 train] target precedes saved update count");
  const auto starting_updates=progress.completed_updates,starting_attempts=progress.attempted_batches;
  TORCH_CHECK(fs::create_directories(output),"[CPB v18 train] exclusive output leaf required");
  if(!resuming){
    v18::save_checkpoint((output/"inference-0.pt").string(),live.model,live.scaler,0,recipe);
    torch::serialize::OutputArchive archive,scaler,bank;archive.write("artifact_kind",ow::text_tensor("cpb_v18_example_preprocessing_v1"),true);
    archive.write("source_identity",ow::text_tensor(source_binding),true);archive.write("split_identity",ow::text_tensor(dataset.split_identity),true);
    archive.write("recipe_identity",ow::text_tensor(identity.recipe_identity),true);archive.write("fit_windows",torch::tensor(fit_windows,torch::kInt64),true);
    live.scaler.save(scaler);live.model->normalizer().save(bank);archive.write("scaler",scaler);archive.write("bank",bank);
    cp::checkpoint_detail::d::publish_checkpoint(archive,(output/"preprocessing.pt").string());
  }
  std::ofstream events(output/"steps.jsonl",std::ios::binary);TORCH_CHECK(events,"[CPB v18 train] open event log");events<<std::setprecision(17);
  int64_t skipped=0;
  while(progress.completed_updates<target&&progress.attempted_batches<maximum_attempts){
    std::mt19937_64 generator(tr::training_counter_seed(progress.row_seed,progress.attempted_batches));std::vector<int64_t> rows;
    for(int64_t b=0;b<batch_size;++b)rows.push_back(static_cast<int64_t>(generator()%static_cast<uint64_t>(dataset.rows())));
    const auto batch=ow::subset(dataset,rows);const auto result=v18::train_step(live.model,live.scaler,*optimizer,progress,raw(batch),batch.cutoffs,recipe,optimizer_options);
    if(!result.updated)++skipped;
    events<<"{\"attempt\":"<<result.attempt_index<<",\"updated\":"<<(result.updated?"true":"false")<<",\"completed_updates\":"<<progress.completed_updates
      <<",\"loss\":"<<result.total_loss<<",\"gaussian_nll\":"<<result.gaussian_nll<<",\"temporal_loss\":"<<result.temporal_loss<<",\"gradient_norm\":"<<result.gradient_norm
      <<",\"target_cells\":"<<result.target_cells<<",\"selected_patch_count\":"<<result.selected_patch_count
      <<",\"unsupported_channels\":"<<result.unsupported_channels<<",\"no_legal_span_channels\":"<<result.no_legal_span_channels
      <<",\"insufficient_support_channels\":"<<result.insufficient_support_channels
      <<",\"temporal_pair_counts\":["<<result.temporal_pair_counts[0]<<','<<result.temporal_pair_counts[1]<<','<<result.temporal_pair_counts[2]<<']'
      <<",\"mask_starts\":"<<integers(result.selected_starts)<<",\"legal_span_counts\":"<<integers(result.legal_span_counts)<<",\"mask_quotas\":"<<integers(result.quotas)
      <<",\"mask_counter_seed\":"<<quote(std::to_string(tr::training_counter_seed(progress.mask_seed,result.attempt_index)))<<",\"source_row_ids\":[";
    for(size_t i=0;i<rows.size();++i){if(i)events<<',';events<<batch.row_ids[static_cast<int64_t>(i)].item<int64_t>();}events<<"]}\n";
    TORCH_CHECK(events,"[CPB v18 train] write event log");
    if(result.updated&&save_every>0&&progress.completed_updates%save_every==0)
      v18::save_training_state((output/("training-"+std::to_string(progress.completed_updates)+".pt")).string(),live.model,live.scaler,*optimizer,identity,progress,recipe,optimizer_options);
  }
  events.close();TORCH_CHECK(events,"[CPB v18 train] finish event log");
  // Exhausted attempts still publish a valid typed partial state and an honest
  // failure receipt. They never silently report reaching the update target.
  const auto training_path=output/"training-final.pt",inference_path=output/"inference-final.pt";
  v18::save_training_state(training_path.string(),live.model,live.scaler,*optimizer,identity,progress,recipe,optimizer_options);
  live.model->eval();live.model->clear_gradients();v18::save_checkpoint(inference_path.string(),live.model,live.scaler,progress.completed_updates,recipe);
  auto restored_export=v18::load_checkpoint(inference_path.string(),config.level.device,&config,recipe).numerical;
  TORCH_CHECK(restored_export.causal_updates==progress.completed_updates&&restored_export.model->state_identity()==live.model->state_identity()&&
      restored_export.scaler.identity()==live.scaler.identity()&&restored_export.model->normalizer().identity()==live.model->normalizer().identity(),"[CPB v18 train] final exported checkpoint self-check failed");
  TORCH_CHECK(cp::boundary::sha256_file(settings)==settings_sha&&cp::boundary::sha256_file(dataset_path)==dataset_sha&&cp::boundary::sha256_file(executable_path)==executable_sha,
      "[CPB v18 train] bound input/executable bytes changed during run");
  if(resuming)TORCH_CHECK(cp::boundary::sha256_file(options.at("--resume"))==resume_sha,"[CPB v18 train] original resume artifact changed");
  const bool passed=progress.completed_updates==target;std::ostringstream report;report<<std::setprecision(17)
      <<"{\"kind\":\"cpb_v18_training_recipe_run_v1\",\"status\":"<<quote(passed?"passed":"attempt_budget_exhausted")
      <<",\"resume\":"<<(resuming?"true":"false")<<",\"settings\":"<<artifact(settings)<<",\"dataset\":"<<artifact(dataset_path)
      <<",\"executable\":"<<artifact(executable_path)<<",\"source_identity\":"<<quote(dataset.source_identity)<<",\"split_identity\":"<<quote(dataset.split_identity)
      <<",\"declared_role\":\"TRAIN\",\"split_membership_independently_verified\":false,\"recipe_identity\":"<<quote(identity.recipe_identity)
      <<",\"target_total_updates\":"<<target<<",\"maximum_total_attempts\":"<<maximum_attempts<<",\"batch_size\":"<<batch_size<<",\"save_every_updates\":"<<save_every
      <<",\"initialization_seed\":"<<live.model->initialization_seed()<<",\"row_seed\":"<<progress.row_seed<<",\"mask_seed\":"<<progress.mask_seed<<",\"context_seed\":"<<progress.context_seed
      <<",\"context_deletion\":\"none\",\"learning_rate\":"<<optimizer_options.learning_rate<<",\"weight_decay\":"<<optimizer_options.weight_decay<<",\"clip_norm\":"<<optimizer_options.clip_norm
      <<",\"completed_updates\":"<<progress.completed_updates<<",\"attempted_batches\":"<<progress.attempted_batches
      <<",\"starting_completed_updates\":"<<starting_updates<<",\"starting_attempted_batches\":"<<starting_attempts
      <<",\"physical_training_updates_this_invocation\":"<<progress.completed_updates-starting_updates<<",\"skipped_attempts_this_invocation\":"<<skipped
      <<",\"preprocessing_fitted\":"<<(!resuming?"true":"false")<<",\"preprocessing_fit_windows\":"<<fit_windows
      <<",\"mask_mode\":"<<quote(options.at("--mask-mode"))<<",\"temporal_weight\":0.1"<<",\"native_width\":"<<cp::dimensions(config).native_width
      <<",\"initial_state_identity\":"<<quote(live.model->initial_state_identity())<<",\"final_state_identity\":"<<quote(live.model->state_identity())
      <<",\"optimizer_identity\":"<<quote(tr::optimizer_identity(live.model,*optimizer))<<",\"scaler_identity\":"<<quote(live.scaler.identity())
      <<",\"bank_identity\":"<<quote(live.model->normalizer().identity())<<",\"final_inference_reload_verified\":true,\"quality_evaluated\":false,\"artifacts\":{\"training_state\":"<<artifact(training_path)
      <<",\"inference\":"<<artifact(inference_path)<<",\"steps\":"<<artifact(output/"steps.jsonl");
  if(!resuming)report<<",\"preprocessing\":"<<artifact(output/"preprocessing.pt")<<",\"initial_inference\":"<<artifact(output/"inference-0.pt");
  if(resuming)report<<",\"resume_input\":"<<artifact(options.at("--resume"));
  report<<"}}";write_text(output/"run.json",report.str());std::cout<<report.str()<<std::endl;return passed?0:2;
}
}
int main(int argc,char **argv){try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
