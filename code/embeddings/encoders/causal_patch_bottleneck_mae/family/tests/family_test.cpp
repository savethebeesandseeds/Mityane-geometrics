// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Bounded synthetic engineering fixtures. No market data, inherited weights or scores.
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v16.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace old=embedding::encoders::causal_patch_bottleneck_mae::v16;
namespace tr=cp::training;
namespace fs=std::filesystem;
namespace {
int64_t checks=0,updates=0;
void check(bool condition,const char *message){++checks;TORCH_CHECK(condition,"[CPB family fixture] ",message);}
template<class F>void refuses(F call,const char *message){bool caught=false;try{call();}catch(const std::exception&){caught=true;}check(caught,message);}
bool bits(const torch::Tensor &a,const torch::Tensor &b){return cp::boundary::same_bits(a,b);}
cp::RawHistory history(const cp::Config &config){
  const auto &c=config.level;const int64_t B=4,N=c.history_length+7;
  auto values=torch::zeros({B,c.channel_count,N,c.input_width},torch::kFloat64);
  auto support=torch::ones(values.sizes(),torch::kBool);
  auto x=values.accessor<double,4>();auto m=support.accessor<bool,4>();
  for(int64_t b=0;b<B;++b)for(int64_t j=0;j<c.channel_count;++j)
    for(int64_t t=0;t<N;++t)for(int64_t f=0;f<c.input_width;++f){
      x[b][j][t][f]=10.+j*.7+f*.2+b*.3+.03*t+.6*std::sin(t*.27+f*.31+j*.13+b*.22);
      m[b][j][t][f]=(t+f+j+b)%29!=0;
    }
  return{values,support,torch::tensor(cp::implementation::resolved_channel_ids(c),torch::kInt64),
      torch::arange(N,torch::kFloat64)*c.sampling_interval,c.sampling_interval};
}
torch::Tensor cuts(const cp::Config &c){return torch::full({4},(c.level.history_length-1)*c.level.sampling_interval,torch::kFloat64);}
cp::RawHistory future_poison(const cp::RawHistory &raw,const cp::Config &c){
  auto out=raw;out.data=raw.data.clone();out.observed=raw.observed.clone();
  out.data.narrow(2,c.level.history_length,7).fill_(std::numeric_limits<double>::quiet_NaN());
  out.observed.narrow(2,c.level.history_length,7).logical_not_();return out;
}
std::pair<int64_t,int64_t> values(const cp::Encoder &model){
  int64_t learned=0,fixed=0;
  for(const auto &item:model->named_parameters()) {
    (item.value().requires_grad()?learned:fixed)+=item.value().numel();
  }
  return{learned,fixed};
}
void same_model(const cp::Encoder &a,const cp::Encoder &b){
  const auto ap=a->named_parameters(),bp=b->named_parameters();
  check(ap.size()==bp.size(),"same full parameter inventory");
  for(const auto &p:ap)check(bp.contains(p.key())&&bits(p.value(),bp[p.key()]),"exact parameter equality");
  const auto ab=a->named_buffers(),bb=b->named_buffers();check(ab.size()==bb.size(),"same buffer inventory");
  for(const auto &p:ab)check(bb.contains(p.key())&&bits(p.value(),bb[p.key()]),"exact buffer equality");
  check(a->state_identity()==b->state_identity(),"complete model identity equality");
}
void same_optimizer(const cp::Encoder &a,const torch::optim::AdamW &oa,const cp::Encoder &b,const torch::optim::AdamW &ob){
  check(tr::optimizer_identity(a,oa)==tr::optimizer_identity(b,ob),"exact AdamW moment content/order/steps");
}
void fixture(const fs::path &configs,const fs::path &output,const std::string &name,std::ostream &report){
  auto config=cp::read_settings(configs/(name+".conf"));const auto &c=config.level;
  const auto raw=history(config),poison=future_poison(raw,config);
  const auto cutoffs=cuts(config);
  cp::training::PreprocessingIdentity fit_identity{"synthetic-engineering-no-market-data","TRAIN-only-fixture", "rounded-parent-population-and-relative-bank-v1"};
  cp::training::PreprocessingFitter fitter(config,fit_identity);fitter.add(raw,cutoffs);auto stats=fitter.freeze();
  refuses([&]{fitter.add(raw,cutoffs);},"fitter cannot mutate frozen preprocessing");
  refuses([&]{(void)fitter.freeze();},"preprocessing freezes exactly once");
  cp::training::PreprocessingFitter poison_fit(config,fit_identity);poison_fit.add(poison,cutoffs);const auto same_stats=poison_fit.freeze();
  check(stats.scaler.identity()==same_stats.scaler.identity()&&stats.bank.identity()==same_stats.bank.identity(),"fit crops before reading poisoned future values/masks");
  const auto permutation=torch::arange(c.channel_count-1,-1,-1,torch::kInt64);
  const cp::RawHistory permuted{raw.data.index_select(1,permutation),raw.observed.index_select(1,permutation),raw.channel_ids.index_select(0,permutation),raw.timestamps,raw.sampling_interval};
  cp::training::PreprocessingFitter permutation_fit(config,fit_identity);permutation_fit.add(permuted,cutoffs);const auto reordered_stats=permutation_fit.freeze();
  check(stats.scaler.identity()==reordered_stats.scaler.identity()&&stats.bank.identity()==reordered_stats.bank.identity(),"timeframe channel mapping preserves preprocessing");
  auto fresh=cp::fresh(config,stats.scaler,stats.bank,221000+c.channel_count);
  auto repeated=cp::fresh(config,stats.scaler,stats.bank,221000+c.channel_count);same_model(fresh.model,repeated.model);
  check(fresh.source_updates==0&&fresh.causal_updates==0,"fresh means no pretrained ancestry or updates");
  const auto scal_id=stats.scaler.identity(),bank_id=stats.bank.identity(),initial_state=fresh.model->state_identity();
  std::map<std::string,torch::Tensor> fixed;
  for(const auto &p:fresh.model->named_parameters())if(!p.value().requires_grad())fixed.emplace(p.key(),p.value().detach().clone());
  const auto encoding=fresh.model->encode_at(raw,cutoffs,stats.scaler);
  check(encoding.native.sizes()==torch::IntArrayRef({4,c.export_width+config.change_width}),"configured native width");
  check(encoding.native.is_cuda()&&torch::isfinite(encoding.native).all().item<bool>()&&encoding.valid.all().item<bool>(),"finite valid CUDA forward");
  const auto poisoned=fresh.model->encode_at(poison,cutoffs,stats.scaler);
  check(bits(encoding.native,poisoned.native)&&bits(encoding.valid,poisoned.valid),"future poison invariant encoding");
  const auto trace=fresh.model->trace_at(raw,cutoffs,stats.scaler);
  check(bits(trace.encoding.native,encoding.native),"partial-last-patch diagnostic trace works");
  if(trace.patch_valid.any().item<bool>())check(trace.token_end_times.masked_select(trace.patch_valid).le(cutoffs[0].item<double>()).all().item<bool>(),"partial patch endpoint does not exceed cutoff");
  const auto permuted_encoding=fresh.model->encode_at(permuted,cutoffs,stats.scaler);
  check(torch::allclose(encoding.native,permuted_encoding.native,1e-6,1e-6),"timeframe channel permutation respects canonical output");
  const auto dir=output/name;check(fs::create_directory(dir),"exclusive shape output");
  const auto inference=dir/"inference-0.pt",training=dir/"training-1.pt";
  cp::save_checkpoint(inference.string(),fresh.model,stats.scaler,0);
  const auto restored=cp::load_checkpoint(inference.string(),c.device,&config);same_model(fresh.model,restored.model);
  refuses([&]{(void)old::load_checkpoint(inference.string(),c.device,&config);},"family archive is not a historical V16 archive");
  const tr::TrainingIdentity identity{"synthetic-engineering-no-market-data","TRAIN-only-fixture","family-gaussian-temporal-v1/configured-adamw-fixture"};
  tr::TrainingProgress progress{0,0,22001,22002,22003};tr::OptimizerConfig recipe;
  recipe.learning_rate=.0005;recipe.weight_decay=.0002;recipe.clip_norm=.7;
  auto optimizer=tr::make_optimizer(fresh.model,recipe);
  tr::train_step(fresh.model,stats.scaler,*optimizer,progress,raw,cutoffs,recipe);++updates;
  check(progress.completed_updates==1&&progress.attempted_batches==1&&fresh.model->causal_updates()==1,"one genuine training update counted");
  check(initial_state!=fresh.model->state_identity(),"learned weights actually change");
  tr::save_training_state(training.string(),fresh.model,stats.scaler,*optimizer,identity,progress,recipe);
  auto resumed=tr::load_training_state(training.string(),config,identity,recipe);same_model(fresh.model,resumed.model);
  same_optimizer(fresh.model,*optimizer,resumed.model,*resumed.optimizer);
  check(resumed.progress.completed_updates==progress.completed_updates&&resumed.progress.attempted_batches==progress.attempted_batches,"resume update and attempt counters");
  const std::vector<int64_t> seeds{progress.row_seed,progress.mask_seed,progress.context_seed};
  const std::vector<int64_t> resumed_seeds{resumed.progress.row_seed,resumed.progress.mask_seed,resumed.progress.context_seed};
  for(size_t i=0;i<seeds.size();++i)check(seeds[i]==resumed_seeds[i]&&tr::training_counter_seed(seeds[i],progress.attempted_batches)==tr::training_counter_seed(resumed_seeds[i],resumed.progress.attempted_batches),"next counter-based stream preserved");
  const auto next=tr::train_step(fresh.model,stats.scaler,*optimizer,progress,raw,cutoffs,recipe);++updates;
  const auto next_resumed=tr::train_step(resumed.model,resumed.scaler,*resumed.optimizer,resumed.progress,raw,cutoffs,recipe);++updates;
  check(next.updated&&next_resumed.updated&&next.total_loss==next_resumed.total_loss&&next.gaussian_nll==next_resumed.gaussian_nll&&next.temporal_loss==next_resumed.temporal_loss,"identical next-step loss components after resume");
  same_model(fresh.model,resumed.model);same_optimizer(fresh.model,*optimizer,resumed.model,*resumed.optimizer);
  check(progress.completed_updates==2&&resumed.progress.completed_updates==2,"both continuation branches complete the next update");
  fresh.model->eval();resumed.model->eval();fresh.model->clear_gradients();resumed.model->clear_gradients();
  check(bits(fresh.model->encode_at(raw,cutoffs,stats.scaler).native,resumed.model->encode_at(raw,cutoffs,resumed.scaler).native),"identical next-step resumed encoding");
  for(const auto &p:fixed)check(bits(fresh.model->named_parameters()[p.first],p.second),"literal analytic prior unchanged by training");
  check(stats.scaler.identity()==scal_id&&fresh.model->normalizer().identity()==bank_id&&resumed.scaler.identity()==scal_id,"preprocessing stays frozen through both training branches");
  auto wrong=config;wrong.change_width+=1;
  refuses([&]{(void)cp::load_checkpoint(inference.string(),c.device,&wrong);},"checkpoint cannot reshape weights");
  refuses([&]{(void)tr::load_training_state(training.string(),wrong,identity,recipe);},"resume refuses wrong configuration");
  for(int kind=0;kind<3;++kind){auto bad=identity;if(kind==0)bad.input_source_identity+="wrong";else if(kind==1)bad.split_identity+="wrong";else bad.recipe_identity+="wrong";
    refuses([&]{(void)tr::load_training_state(training.string(),config,bad,recipe);},"resume refuses wrong data/split/recipe identity");}
  auto wrong_recipe=recipe;wrong_recipe.clip_norm=1.;
  refuses([&]{(void)tr::load_training_state(training.string(),config,identity,wrong_recipe);},"resume refuses different optimizer/clip recipe");
  refuses([&]{(void)tr::load_training_state(inference.string(),config,identity,recipe);},"inference artifact cannot resume training");
  // A batch without observed patches must count an attempt, not invent an update.
  auto empty=raw;empty.observed=torch::zeros_like(raw.observed);const auto before_empty=fresh.model->state_identity();
  tr::train_step(fresh.model,stats.scaler,*optimizer,progress,empty,cutoffs,recipe);
  check(progress.completed_updates==2&&progress.attempted_batches==3&&fresh.model->state_identity()==before_empty,"unsupported batch abstains without training");
  const auto counts=values(fresh.model);
  report<<"{\"name\":\""<<name<<"\",\"C\":"<<c.channel_count<<",\"H\":"<<c.history_length<<",\"F\":"<<c.input_width<<",\"P\":"<<c.patch_length
      <<",\"W\":"<<c.encoder_width<<",\"D\":"<<c.export_width<<",\"layers\":"<<c.num_layers<<",\"change\":"<<config.change_width
      <<",\"native\":"<<c.export_width+config.change_width<<",\"learned_values\":"<<counts.first<<",\"fixed_values\":"<<counts.second
      <<",\"fresh_forward_train_export_exact_next_step_resume\":true,\"physical_fixture_updates\":3}";
}
void invalid_configurations(const fs::path &configs){
  const auto valid=cp::read_settings(configs/"paired.conf");
  for(int kind=0;kind<6;++kind){auto c=valid;
    if(kind==0)c.level.history_length=16;else if(kind==1)c.level.num_heads=3;else if(kind==2)c.level.export_width=4;
    else if(kind==3)c.change_width=0;else if(kind==4)c.level.channel_ids={11,11};else c.level.input_width=std::numeric_limits<int64_t>::max();
    refuses([&]{cp::training::PreprocessingFitter fitter(c,{"synthetic","TRAIN","fixture"});},"invalid or overflowing configuration refused before allocation");}
}
void benchmark_constructor_parity(){
  const auto config=old::read_settings("code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v16/config/v16.conf");
  const auto raw=history(config);const auto cutoffs=cuts(config);
  tr::PreprocessingFitter fitter(config,{"synthetic-benchmark-constructor-only","TRAIN-only-fixture","family-rounded-moments-v1"});
  fitter.add(raw,cutoffs);const auto stats=fitter.freeze();
  auto current=cp::fresh(config,stats.scaler,stats.bank,222000);
  const auto counts=cp::parameter_counts(config);
  auto historical=old::fresh(config,stats.scaler,stats.bank,222000);
  const old::RawHistory historical_input{raw.data,raw.observed,raw.channel_ids,raw.timestamps,raw.sampling_interval};
  check(counts.learned_tensors==75&&counts.learned_values==293672&&counts.fixed_values==3888,"derived benchmark counts agree with historical fixture");
  const auto before=historical.model->named_parameters();
  for(const auto &item:current.model->named_parameters())check(before.contains(item.key())&&bits(item.value(),before[item.key()]),"family constructor matches historical numerical tensors at benchmark configuration");
  check(bits(current.model->encode_at(raw,cutoffs,stats.scaler).native,historical.model->encode_at(historical_input,cutoffs,stats.scaler).native),"benchmark numerical forward unchanged");
}
void high_offset_preprocessing(const fs::path &configs){
  const auto config=cp::read_settings(configs/"single.conf");const auto &c=config.level;
  // A rare one-F32-ULP change on a high offset defeats E[x²]-E[x]².
  // Centered F64 moments must retain its genuine nonzero variation.
  const int64_t B=1024,H=c.history_length;const double base=std::ldexp(1.,60),ulp=std::ldexp(1.,37);
  auto values=torch::full({B,1,H,1},base,torch::kFloat64);values[0][0][0][0]=base+ulp;
  // 11 is the torch ordinal of 4h, matching single.conf.
  const cp::RawHistory raw{values,torch::ones(values.sizes(),torch::kBool),torch::tensor({11},torch::kInt64),torch::arange(H,torch::kFloat64)*c.sampling_interval,c.sampling_interval};
  const auto cutoffs=torch::full({B},(H-1)*c.sampling_interval,torch::kFloat64);
  tr::PreprocessingFitter fitter(config,{"synthetic-high-offset-one-ulp","TRAIN-only-fixture","centered-window-moments-v1"});
  fitter.add(raw,cutoffs);const auto stats=fitter.freeze();
  const auto rounded=values.to(torch::kFloat32).to(torch::kFloat64);const auto reference=(rounded-rounded.mean()).square().mean().sqrt();
  check(reference.item<double>()>c.scale_floor,"high-offset fixture has genuine variation");
  check(torch::allclose(stats.scaler.scale,reference.expand_as(stats.scaler.scale),1e-9,0.),"centered scaler preserves rare high-offset F32 variation");
}
}
int main(int argc,char **argv){try{
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  check(argc==5&&std::string(argv[1])=="--config-directory"&&std::string(argv[3])=="--output","usage: --config-directory DIR --output NEWLEAF");
  check(torch::cuda::is_available(),"actual CUDA required");
  const fs::path configs=argv[2],output=argv[4];check(!fs::exists(output)&&fs::create_directories(output),"output must be new");
  invalid_configurations(configs);benchmark_constructor_parity();high_offset_preprocessing(configs);std::ostringstream report;
  report<<"{\"kind\":\"cpb_configurable_family_engineering_v1\",\"status\":\"passed\",\"synthetic_only\":true,\"pretrained_checkpoints_read\":0,\"quality_evaluated\":false,\"fixtures\":[";
  bool first=true;for(const auto name:{"single","paired","multi"}){if(!first)report<<',';fixture(configs,output,name,report);first=false;}
  report<<"],\"checks\":"<<checks<<",\"physical_training_updates\":"<<updates<<",\"exact_next_step_resume_cases\":3}";
  std::ofstream out(output/"summary.json");out<<report.str()<<'\n';out.close();TORCH_CHECK(bool(out),"write receipt");
  std::cout<<report.str()<<std::endl;return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
