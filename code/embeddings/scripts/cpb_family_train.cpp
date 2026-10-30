// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Portable single-device example: no market reader, inherited checkpoint or quality scoring.
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include "embedding/data/observation_windows.h"
#include <torch/cuda.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <vector>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
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
  TORCH_CHECK(!fs::exists(path),"[CPB family train] existing output cannot be replaced: ",path.string());
  std::ofstream stream(path,std::ios::binary);TORCH_CHECK(stream,"[CPB family train] open output");stream<<text<<'\n';stream.close();
  TORCH_CHECK(stream,"[CPB family train] write output");
}
std::string artifact(const fs::path &path){
  std::ostringstream out;out<<"{\"path\":"<<quote(path.generic_string())<<",\"bytes\":"<<fs::file_size(path)
    <<",\"sha256\":"<<quote(cp::boundary::sha256_file(path))<<'}';return out.str();
}
cp::RawHistory raw(const ow::Dataset &dataset){return{dataset.values,dataset.observed,dataset.channel_ids,dataset.timestamps,dataset.sampling_interval};}
// Historical coordinate15_v1 draw. Ratio, stream, canonical-rank ordinal and
// two-group repair match the V16 study sampler. This is not the family row RNG.
uint64_t context_mixed(uint64_t x){
  x+=0x9e3779b97f4a7c15ULL;
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;
  return x^(x>>31);
}
uint64_t context_counter_seed(int64_t seed,int64_t attempt,uint64_t stream){
  return context_mixed(static_cast<uint64_t>(seed)^context_mixed(static_cast<uint64_t>(attempt))^stream);
}
torch::Tensor coordinate15_deleted(const tr::MaskBatch &masks,const torch::Tensor &ids,
    const cp::implementation::Config &config,int64_t seed,int64_t attempt){
  constexpr double ratio=.15;
  constexpr uint64_t stream=0x6374782d64726f70ULL;
  TORCH_CHECK(seed>=0&&attempt>=0,"[CPB family train] context15 nonnegative absolute counters required");
  TORCH_CHECK(masks.visible.defined()&&masks.visible.scalar_type()==torch::kBool&&masks.visible.dim()==4,
      "[CPB family train] context15 blocked: visible must be logical B,C,H,F bool");
  const auto B=masks.visible.size(0);
  const auto C=config.channel_count,H=config.history_length,F=config.input_width,P=config.patch_length;
  TORCH_CHECK(P>0&&H>0&&H%P==0&&F>0&&C>0,"[CPB family train] context15 blocked: history must be a positive multiple of patch length");
  TORCH_CHECK(masks.visible.size(1)==C&&masks.visible.size(2)==H&&masks.visible.size(3)==F,
      "[CPB family train] context15 blocked: visible must be logical B,C,H,F bool");
  TORCH_CHECK(masks.eligible_channels.defined()&&masks.eligible_channels.scalar_type()==torch::kBool&&
      masks.eligible_channels.dim()==2&&masks.eligible_channels.size(0)==B&&masks.eligible_channels.size(1)==C,
      "[CPB family train] context15 blocked: eligible channels must be bool B,C");
  const auto K=H/P,PF=P*F;
  (void)cp::implementation::channel_indices(ids,config,B,torch::kCPU);
  const auto rows=(ids.dim()==1?ids.unsqueeze(0).expand({B,C}):ids).to(torch::kCPU);
  const auto order=rows.argsort(int64_t{1}).contiguous();
  const auto canonical=order.accessor<int64_t,2>();
  auto visible=masks.visible.to(torch::kCPU).reshape({B,C,K,PF}).contiguous();
  auto requested=torch::zeros_like(visible);
  auto support=visible.accessor<bool,4>();
  auto requests=requested.accessor<bool,4>();
  auto eligible=masks.eligible_channels.to(torch::kCPU).contiguous();
  auto original_eligible=eligible.accessor<bool,2>();
  const auto base=context_counter_seed(seed,attempt,stream);
  for(int64_t b=0;b<B;++b){
    for(int64_t rank=0;rank<C;++rank){
      const auto c=canonical[b][rank];
      for(int64_t k=0;k<K;++k){
        for(int64_t j=0;j<PF;++j){
          const auto ordinal=static_cast<uint64_t>((b*C+rank)*H*F+k*PF+j);
          const double uniform=static_cast<double>(context_mixed(base+ordinal)>>11)*0x1.0p-53;
          requests[b][c][k][j]=support[b][c][k][j]&&uniform<ratio;
        }
      }
    }
  }
  auto deleted=requested.clone();auto erasures=deleted.accessor<bool,4>();
  for(int64_t b=0;b<B;++b){
    for(int64_t rank=0;rank<C;++rank){
      const auto c=canonical[b][rank];
      if(!original_eligible[b][c])continue;
      int64_t retained_groups=0;
      for(int64_t k=0;k<K;++k){
        bool retained=false;
        for(int64_t j=0;j<PF;++j)retained|=support[b][c][k][j]&&!erasures[b][c][k][j];
        retained_groups+=retained;
      }
      for(int64_t k=0;k<K&&retained_groups<2;++k){
        bool retained=false;int64_t first=-1;
        for(int64_t j=0;j<PF;++j){
          if(support[b][c][k][j]&&first<0)first=j;
          retained|=support[b][c][k][j]&&!erasures[b][c][k][j];
        }
        if(!retained&&first>=0){
          TORCH_CHECK(erasures[b][c][k][first],"[CPB family train] context15 repair cannot invent support");
          erasures[b][c][k][first]=false;++retained_groups;
        }
      }
      TORCH_CHECK(retained_groups>=2,"[CPB family train] context15 original eligible support cannot be repaired");
    }
  }
  return deleted.reshape({B,C,H,F}).to(masks.visible.device());
}
uint64_t context15_fnv(const torch::Tensor &deleted){
  const auto cpu=deleted.to(torch::kCPU).contiguous();
  const auto *bytes=cpu.data_ptr<bool>();
  uint64_t hash=14695981039346656037ULL;
  for(int64_t i=0;i<cpu.numel();++i){hash^=bytes[i]?1ULL:0ULL;hash*=1099511628211ULL;}
  return hash;
}
void context15_self_check(const cp::implementation::Config &loaded){
  // Ids {0,1,2} are the withdrawn instrument binding this historical rule was measured on.
  // They are not timeframe channels. The digest lock stays on that recorded geometry.
  TORCH_CHECK(loaded.channel_count==3&&loaded.history_length==128&&loaded.input_width==9&&loaded.patch_length==8&&
      loaded.mask_ratio==0.25&&loaded.channel_ids==std::vector<int64_t>({0,1,2}),
      "[CPB family train] context15 self-check requires the withdrawn C3/H128/F9/P8 instrument binding");
  auto level=cp::implementation::default_configurable_spectral_config();
  level.channel_ids={0,1,2};
  constexpr int64_t batch=2,mask_seed=11011;
  auto observed=torch::ones({batch,level.channel_count,level.history_length,level.input_width},torch::kBool);
  const auto masks=cp::implementation::configurable_spectral_make_training_mask(observed,level,mask_seed);
  const struct {const char *order;std::vector<int64_t> ids;} orders[]{{"fwd",{0,1,2}},{"rev",{2,1,0}}};
  const int64_t seeds[]{951101,42},attempts[]{0,1,16,2047};
  uint64_t first_fwd=0,first_rev=0;bool have_fwd=false,have_rev=false;
  for(const auto &order:orders){
    auto ids=torch::tensor(order.ids,torch::kInt64);
    for(const auto seed:seeds)for(const auto attempt:attempts){
      const auto deleted=coordinate15_deleted(masks,ids,level,seed,attempt);
      const auto again=coordinate15_deleted(masks,ids,level,seed,attempt);
      TORCH_CHECK(torch::equal(deleted,again),"[CPB family train] context15 self-check is not repeatable");
      TORCH_CHECK(deleted.scalar_type()==torch::kBool&&deleted.sizes()==masks.visible.sizes(),
          "[CPB family train] context15 self-check changed the visible shape");
      TORCH_CHECK(!deleted.logical_and(masks.visible.logical_not()).any().item<bool>(),
          "[CPB family train] context15 self-check left the visible support");
      const auto hash=context15_fnv(deleted);
      if(!have_fwd&&std::string(order.order)=="fwd"&&seed==951101&&attempt==0){first_fwd=hash;have_fwd=true;}
      if(!have_rev&&std::string(order.order)=="rev"&&seed==951101&&attempt==0){first_rev=hash;have_rev=true;}
      std::ostringstream line;line<<"CONTEXT15_DIGEST order="<<order.order<<" seed="<<seed<<" attempt="<<attempt
        <<" fnv="<<std::hex<<std::setw(16)<<std::setfill('0')<<hash<<std::dec;
      std::cout<<line.str()<<'\n';
    }
  }
  TORCH_CHECK(have_fwd&&have_rev&&first_fwd!=first_rev,"[CPB family train] context15 self-check did not observe canonical channel rank");
  std::cout<<"CONTEXT15_SELF_CHECK passed\n";
}
void help(){
  std::cout<<"cpb_family_train --settings FILE --dataset FILE --output NEWLEAF --steps TOTAL --max-attempts TOTAL --batch-size N\n"
    <<"  --row-seed I --mask-seed I --context-seed I (--seed I | --resume TRAINING_STATE)\n"
    <<"  [--learning-rate .001] [--weight-decay .0001] [--clip-norm 1] [--save-every 0] [--temporal-weight 0.1] [--context-deletion coordinate15]\n"
    <<"Single-device configurable-family example; DATASET must be a typed observation_window_dataset_v1 archive declaring role TRAIN.\n"
    <<"--steps is the TOTAL completed-update target, not additional steps. --max-attempts is a TOTAL attempt ceiling.\n"
    <<"Resume requires identical data/settings/executable bytes, batching, options and stream seeds. Stop budgets may increase.\n"
    <<"Fresh fitting uses all declared TRAIN rows once in source order, in --batch-size chunks; no fit on resume.\n"
    <<"Rows use mt19937_64(counter_seed(row_seed,attempt)) modulo row count, with replacement. Context deletion defaults to none.\n"
    <<"--context-deletion coordinate15 uses the historical 0.15 canonical-rank rule. No other rate is accepted.\n"
    <<"The dataset declaration does not independently prove real TRAIN membership. No scheduler/accumulation/distributed training.\n";
}
int execute(int argc,char **argv){
  if(argc==2&&std::string(argv[1])=="--help"){help();return 0;}
  TORCH_CHECK(argc>1&&(argc-1)%2==0,"[CPB family train] use --help; options require values");
  const std::set<std::string> allowed{"--settings","--dataset","--output","--steps","--max-attempts","--batch-size","--seed","--resume",
    "--row-seed","--mask-seed","--context-seed","--learning-rate","--weight-decay","--clip-norm","--save-every",
    "--temporal-weight","--context-deletion","--context-deletion-self-check"};
  std::map<std::string,std::string> options;
  for(int i=1;i<argc;i+=2)TORCH_CHECK(allowed.count(argv[i])&&options.emplace(argv[i],argv[i+1]).second,"[CPB family train] unknown or duplicate option");
  const bool use_context15=options.count("--context-deletion");
  if(use_context15)TORCH_CHECK(options.at("--context-deletion")=="coordinate15","[CPB family train] --context-deletion accepts only coordinate15");
  const std::string context_label=use_context15?"coordinate15-v1-ratio-0.15-stream-6374782d64726f70":"none";
  if(options.count("--context-deletion-self-check")){
    TORCH_CHECK(options.at("--context-deletion-self-check")=="1"&&use_context15&&options.count("--settings"),
        "[CPB family train] context15 self-check requires --settings and --context-deletion coordinate15");
    context15_self_check(cp::read_withdrawn_instrument_settings(options.at("--settings")).level);return 0;
  }
  for(const auto key:{"--settings","--dataset","--output","--steps","--max-attempts","--batch-size","--row-seed","--mask-seed","--context-seed"})
    TORCH_CHECK(options.count(key),"[CPB family train] required option: ",key);
  TORCH_CHECK(bool(options.count("--seed"))!=bool(options.count("--resume")),"[CPB family train] choose fresh --seed OR typed --resume");
  const auto integer=[&](const std::string &key,int64_t fallback)->int64_t{
    if(!options.count(key))return fallback;
    size_t consumed=0;const auto value=std::stoll(options.at(key),&consumed);
    TORCH_CHECK(consumed==options.at(key).size()&&value>=0,"[CPB family train] nonnegative integer required: ",key);return value;
  };
  const auto number=[&](const std::string &key,double fallback){
    if(!options.count(key))return fallback;
    size_t consumed=0;const auto value=std::stod(options.at(key),&consumed);
    TORCH_CHECK(consumed==options.at(key).size()&&std::isfinite(value),"[CPB family train] finite number required: ",key);return value;
  };
  const int64_t target=integer("--steps",0),maximum_attempts=integer("--max-attempts",0),batch_size=integer("--batch-size",0);
  const int64_t row_seed=integer("--row-seed",0),mask_seed=integer("--mask-seed",0),context_seed=integer("--context-seed",0),save_every=integer("--save-every",0);
  TORCH_CHECK(target>0&&maximum_attempts>=target&&maximum_attempts<std::numeric_limits<int64_t>::max()-1&&batch_size>0,
      "[CPB family train] positive target/batch, attempt ceiling>=target and incrementable counters required");
  const tr::OptimizerConfig optimizer_options{number("--learning-rate",.001),number("--weight-decay",.0001),number("--clip-norm",1.)};
  const double temporal_weight=number("--temporal-weight",0.1);
  TORCH_CHECK(std::isfinite(temporal_weight)&&temporal_weight>=0,"[CPB family train] --temporal-weight must be finite and nonnegative");
  tr::validate_optimizer_config(optimizer_options);
  const fs::path settings=options.at("--settings"),dataset_path=options.at("--dataset"),output=options.at("--output");
  TORCH_CHECK(fs::is_regular_file(settings)&&fs::is_regular_file(dataset_path)&&!fs::exists(output),"[CPB family train] settings/dataset must exist and output must be new");
  const auto settings_sha=cp::boundary::sha256_file(settings),dataset_sha=cp::boundary::sha256_file(dataset_path);
  const auto executable_path=fs::canonical("/proc/self/exe");
  const auto executable_sha=cp::boundary::sha256_file(executable_path);
  const auto config=cp::read_settings(settings);const auto dataset=ow::read(dataset_path);
  TORCH_CHECK(dataset.role=="TRAIN","[CPB family train] only a caller-declared TRAIN dataset may train/fit");
  TORCH_CHECK(batch_size<=std::numeric_limits<int64_t>::max()/dataset.values[0].numel(),"[CPB family train] batch cell count overflow");
  TORCH_CHECK(cp::boundary::sha256_file(settings)==settings_sha&&cp::boundary::sha256_file(dataset_path)==dataset_sha,"[CPB family train] input changed while reading");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);TORCH_CHECK(torch::cuda::is_available(),"[CPB family train] actual CUDA required");
  const auto source_binding=dataset.source_identity+";archive-sha256="+dataset_sha;
  std::ostringstream recipe_stream;recipe_stream<<std::setprecision(17)
      <<"cpb-family-train-example-v1;executable="<<executable_sha<<";settings="<<settings_sha<<";dataset="<<dataset_sha
      <<";batch="<<batch_size<<";rows=mt19937_64-counter-seed-mod-N-with-replacement-v1;context="<<context_label
      <<";calibration=all-TRAIN-source-order-batch-chunks-Chan-v1"
      <<";row-seed="<<row_seed<<";mask-seed="<<mask_seed<<";context-seed="<<context_seed
      <<";AdamW-lr="<<optimizer_options.learning_rate<<";wd="<<optimizer_options.weight_decay<<";global-clip="<<optimizer_options.clip_norm
      <<";temporal-weight="<<temporal_weight;
  const tr::TrainingIdentity identity{source_binding,dataset.split_identity,recipe_stream.str()};
  const bool resuming=options.count("--resume");cp::Loaded live;std::unique_ptr<torch::optim::AdamW> optimizer;
  tr::TrainingProgress progress{0,0,row_seed,mask_seed,context_seed};int64_t fit_windows=0;std::string resume_sha;
  if(resuming){
    const fs::path resume=options.at("--resume");TORCH_CHECK(fs::is_regular_file(resume),"[CPB family train] resume state must exist");resume_sha=cp::boundary::sha256_file(resume);
    auto restored=tr::load_training_state(resume.string(),config,identity,optimizer_options);
    TORCH_CHECK(cp::boundary::sha256_file(resume)==resume_sha,"[CPB family train] resume artifact changed while reading");
    progress=restored.progress;TORCH_CHECK(progress.row_seed==row_seed&&progress.mask_seed==mask_seed&&progress.context_seed==context_seed,"[CPB family train] exact saved stream seeds required");
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
  TORCH_CHECK(progress.completed_updates<=target,"[CPB family train] target precedes saved update count");
  const auto starting_updates=progress.completed_updates,starting_attempts=progress.attempted_batches;
  TORCH_CHECK(fs::create_directories(output),"[CPB family train] exclusive output leaf required");
  if(!resuming){
    cp::save_checkpoint((output/"inference-0.pt").string(),live.model,live.scaler,0);
    torch::serialize::OutputArchive archive,scaler,bank;archive.write("artifact_kind",ow::text_tensor("cpb_family_example_preprocessing_v1"),true);
    archive.write("source_identity",ow::text_tensor(source_binding),true);archive.write("split_identity",ow::text_tensor(dataset.split_identity),true);
    archive.write("recipe_identity",ow::text_tensor(identity.recipe_identity),true);archive.write("fit_windows",torch::tensor(fit_windows,torch::kInt64),true);
    live.scaler.save(scaler);live.model->normalizer().save(bank);archive.write("scaler",scaler);archive.write("bank",bank);
    cp::checkpoint_detail::d::publish_checkpoint(archive,(output/"preprocessing.pt").string());
  }
  std::ofstream events(output/"steps.jsonl",std::ios::binary);TORCH_CHECK(events,"[CPB family train] open event log");events<<std::setprecision(17);
  int64_t skipped=0;
  while(progress.completed_updates<target&&progress.attempted_batches<maximum_attempts){
    std::mt19937_64 generator(tr::training_counter_seed(progress.row_seed,progress.attempted_batches));std::vector<int64_t> rows;
    for(int64_t b=0;b<batch_size;++b)rows.push_back(static_cast<int64_t>(generator()%static_cast<uint64_t>(dataset.rows())));
    const auto batch=ow::subset(dataset,rows);
    const auto history=raw(batch);
    torch::Tensor context_deleted;
    if(use_context15){
      const auto masks=tr::make_mask(history,batch.cutoffs,live.model->config(),progress.mask_seed,progress.attempted_batches);
      const auto retained=cp::boundary::retain(history,batch.cutoffs,live.model->config());
      context_deleted=coordinate15_deleted(masks,retained.raw.channel_ids,live.model->config().level,progress.context_seed,progress.attempted_batches);
    }
    const auto result=use_context15
        ?tr::train_step(live.model,live.scaler,*optimizer,progress,history,batch.cutoffs,optimizer_options,context_deleted,temporal_weight)
        :tr::train_step(live.model,live.scaler,*optimizer,progress,history,batch.cutoffs,optimizer_options,{},temporal_weight);
    if(!result.updated)++skipped;
    events<<"{\"attempt\":"<<result.attempt_index<<",\"updated\":"<<(result.updated?"true":"false")<<",\"completed_updates\":"<<progress.completed_updates
      <<",\"loss\":"<<result.total_loss<<",\"gaussian_nll\":"<<result.gaussian_nll<<",\"temporal_loss\":"<<result.temporal_loss<<",\"gradient_norm\":"<<result.gradient_norm<<",\"source_row_ids\":[";
    for(size_t i=0;i<rows.size();++i){if(i)events<<',';events<<batch.row_ids[static_cast<int64_t>(i)].item<int64_t>();}events<<"]}\n";
    TORCH_CHECK(events,"[CPB family train] write event log");
    if(result.updated&&save_every>0&&progress.completed_updates%save_every==0)
      tr::save_training_state((output/("training-"+std::to_string(progress.completed_updates)+".pt")).string(),live.model,live.scaler,*optimizer,identity,progress,optimizer_options);
  }
  events.close();TORCH_CHECK(events,"[CPB family train] finish event log");
  // Exhausted attempts still publish a valid typed partial state and an honest
  // failure receipt. They never silently report reaching the update target.
  const auto training_path=output/"training-final.pt",inference_path=output/"inference-final.pt";
  tr::save_training_state(training_path.string(),live.model,live.scaler,*optimizer,identity,progress,optimizer_options);
  live.model->eval();live.model->clear_gradients();cp::save_checkpoint(inference_path.string(),live.model,live.scaler,progress.completed_updates);
  auto restored_export=cp::load_checkpoint(inference_path.string(),config.level.device,&config);
  TORCH_CHECK(restored_export.causal_updates==progress.completed_updates&&restored_export.model->state_identity()==live.model->state_identity()&&
      restored_export.scaler.identity()==live.scaler.identity()&&restored_export.model->normalizer().identity()==live.model->normalizer().identity(),"[CPB family train] final exported checkpoint self-check failed");
  TORCH_CHECK(cp::boundary::sha256_file(settings)==settings_sha&&cp::boundary::sha256_file(dataset_path)==dataset_sha&&cp::boundary::sha256_file(executable_path)==executable_sha,
      "[CPB family train] bound input/executable bytes changed during run");
  if(resuming)TORCH_CHECK(cp::boundary::sha256_file(options.at("--resume"))==resume_sha,"[CPB family train] original resume artifact changed");
  const bool passed=progress.completed_updates==target;std::ostringstream report;report<<std::setprecision(17)
      <<"{\"kind\":\"cpb_family_training_example_run_v1\",\"status\":"<<quote(passed?"passed":"attempt_budget_exhausted")
      <<",\"resume\":"<<(resuming?"true":"false")<<",\"settings\":"<<artifact(settings)<<",\"dataset\":"<<artifact(dataset_path)
      <<",\"executable\":"<<artifact(executable_path)<<",\"source_identity\":"<<quote(dataset.source_identity)<<",\"split_identity\":"<<quote(dataset.split_identity)
      <<",\"declared_role\":\"TRAIN\",\"split_membership_independently_verified\":false,\"recipe_identity\":"<<quote(identity.recipe_identity)
      <<",\"target_total_updates\":"<<target<<",\"maximum_total_attempts\":"<<maximum_attempts<<",\"batch_size\":"<<batch_size<<",\"save_every_updates\":"<<save_every
      <<",\"initialization_seed\":"<<live.model->initialization_seed()<<",\"row_seed\":"<<progress.row_seed<<",\"mask_seed\":"<<progress.mask_seed<<",\"context_seed\":"<<progress.context_seed
      <<",\"context_deletion\":"<<quote(context_label)<<",\"learning_rate\":"<<optimizer_options.learning_rate<<",\"weight_decay\":"<<optimizer_options.weight_decay<<",\"clip_norm\":"<<optimizer_options.clip_norm
      <<",\"completed_updates\":"<<progress.completed_updates<<",\"attempted_batches\":"<<progress.attempted_batches
      <<",\"starting_completed_updates\":"<<starting_updates<<",\"starting_attempted_batches\":"<<starting_attempts
      <<",\"physical_training_updates_this_invocation\":"<<progress.completed_updates-starting_updates<<",\"skipped_attempts_this_invocation\":"<<skipped
      <<",\"preprocessing_fitted\":"<<(!resuming?"true":"false")<<",\"preprocessing_fit_windows\":"<<fit_windows
      <<",\"native_width\":"<<cp::dimensions(config).native_width<<",\"final_state_identity\":"<<quote(live.model->state_identity())
      <<",\"optimizer_identity\":"<<quote(tr::optimizer_identity(live.model,*optimizer))<<",\"scaler_identity\":"<<quote(live.scaler.identity())
      <<",\"bank_identity\":"<<quote(live.model->normalizer().identity())<<",\"final_inference_reload_verified\":true,\"quality_evaluated\":false,\"artifacts\":{\"training_state\":"<<artifact(training_path)
      <<",\"inference\":"<<artifact(inference_path)<<",\"steps\":"<<artifact(output/"steps.jsonl");
  if(!resuming)report<<",\"preprocessing\":"<<artifact(output/"preprocessing.pt")<<",\"initial_inference\":"<<artifact(output/"inference-0.pt");
  if(resuming)report<<",\"resume_input\":"<<artifact(options.at("--resume"));
  report<<"}}";write_text(output/"run.json",report.str());std::cout<<report.str()<<std::endl;return passed?0:2;
}
}
int main(int argc,char **argv){try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
