// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Synthetic mechanism checks only. No research checkpoints/data or quality scores.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v17.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace tr=cp::training;
namespace next=embedding::encoders::causal_patch_bottleneck_mae::v17;
namespace fs=std::filesystem;
namespace {
int64_t assertions=0,updates=0,gradient_tensors=0,refusals=0,fits=0;
void check(bool value,const char *why){++assertions;TORCH_CHECK(value,"[cpb-v17 engineering] ",why);}
template<class F>void refuses(F operation,const char *why){bool rejected=false;try{operation();}catch(const std::exception &){rejected=true;}
  check(rejected,why);++refusals;}
bool bits(const torch::Tensor &a,const torch::Tensor &b){
  return a.defined()==b.defined()&&(!a.defined()||cp::boundary::same_bits(a,b));
}
cp::RawHistory history(const cp::Config &config,int64_t phase=0){
  const auto &c=config.level;const int64_t B=2,N=c.history_length+9;
  auto data=torch::zeros({B,c.channel_count,N,c.input_width},torch::kFloat64);
  auto observed=torch::ones(data.sizes(),torch::kBool);
  auto x=data.accessor<double,4>();auto m=observed.accessor<bool,4>();
  for(int64_t b=0;b<B;++b)for(int64_t j=0;j<c.channel_count;++j)
    for(int64_t t=0;t<N;++t)for(int64_t f=0;f<c.input_width;++f){
      x[b][j][t][f]=20.+2*j+.2*f+.3*b+.021*t+.009*phase+
          .7*std::sin(.19*t+.31*f+.27*j+.17*b+.03*phase);
      m[b][j][t][f]=(t+3*f+j+b)%37!=0;
    }
  return{data,observed,torch::tensor(cp::implementation::resolved_channel_ids(c),torch::kInt64),
      torch::arange(N,torch::kFloat64)*c.sampling_interval,c.sampling_interval};
}
torch::Tensor cuts(const cp::Config &config){return torch::full({2},(config.level.history_length-1)*config.level.sampling_interval,torch::kFloat64);}
void same_model(const cp::Encoder &a,const cp::Encoder &b){
  const auto ap=a->named_parameters(),bp=b->named_parameters();check(ap.size()==bp.size(),"same parameter inventory");
  for(const auto &p:ap)check(bp.contains(p.key())&&bits(p.value(),bp[p.key()]),"exact learned/fixed parameter tensor");
  const auto ab=a->named_buffers(),bb=b->named_buffers();check(ab.size()==bb.size(),"same buffer inventory");
  for(const auto &p:ab)check(bb.contains(p.key())&&bits(p.value(),bb[p.key()]),"exact buffer tensor");
  check(a->state_identity()==b->state_identity(),"same complete numerical state identity");
}
void same_progress(const tr::TrainingProgress &a,const tr::TrainingProgress &b){
  check(a.completed_updates==b.completed_updates&&a.attempted_batches==b.attempted_batches&&
      a.row_seed==b.row_seed&&a.mask_seed==b.mask_seed&&a.context_seed==b.context_seed,"exact progress/counter seeds");
}
void fixture(const fs::path &settings,const fs::path &output,int64_t seed){
  const auto config=next::read_settings(settings);const auto raw=history(config);const auto cutoffs=cuts(config);
  next::PreprocessingFitter fitter(config,{"synthetic-v17-engineering","TRAIN-only-two-windows","centered-Chan-and-bank-v1"});
  fitter.add(raw,cutoffs);const auto preprocessing=fitter.freeze();++fits;
  const next::Recipe control{.1},candidate{.03};const tr::OptimizerConfig options{};
  auto original=cp::fresh(config,preprocessing.scaler,preprocessing.bank,seed);
  auto unchanged=next::fresh(config,preprocessing.scaler,preprocessing.bank,seed);
  same_model(original.model,unchanged.model);
  const auto masks=tr::make_mask(raw,cutoffs,config,7712,0);
  original.model->train();unchanged.model->train();
  auto base_trace=original.model->forward_at(raw,cutoffs,masks.hidden,original.scaler);
  auto same_trace=next::forward_at(unchanged.model,raw,cutoffs,masks.hidden,unchanged.scaler,control);
  check(bits(base_trace.output.loss,same_trace.output.loss)&&bits(base_trace.nll.loss,same_trace.nll.loss)&&
      bits(base_trace.objective.temporal.loss,same_trace.objective.temporal.loss),"weight0.1 forward/loss exactly matches unchanged family");
  base_trace.output.loss.backward();same_trace.output.loss.backward();
  const auto left=original.model->named_parameters(),right=unchanged.model->named_parameters();
  for(const auto &p:left){check(bits(p.value().grad(),right[p.key()].grad()),"weight0.1 named gradient exactly matches unchanged family");
    gradient_tensors+=p.value().grad().defined();}
  original.model->clear_gradients();unchanged.model->clear_gradients();
  auto original_optimizer=tr::make_optimizer(original.model,options),unchanged_optimizer=next::make_optimizer(unchanged.model,options);
  tr::TrainingProgress original_progress{0,0,7711,7712,7713},unchanged_progress=original_progress;
  const auto old_step=tr::train_step(original.model,original.scaler,*original_optimizer,original_progress,raw,cutoffs,options);
  const auto same_step=next::train_step(unchanged.model,unchanged.scaler,*unchanged_optimizer,unchanged_progress,raw,cutoffs,control,options);
  updates+=2;check(old_step.updated&&same_step.updated&&old_step.total_loss==same_step.total_loss&&
      old_step.gradient_norm==same_step.gradient_norm,"weight0.1 step loss and clipping exactly match");
  same_model(original.model,unchanged.model);same_progress(original_progress,unchanged_progress);
  check(tr::optimizer_identity(original.model,*original_optimizer)==tr::optimizer_identity(unchanged.model,*unchanged_optimizer),
      "weight0.1 AdamW moment content/order/steps exactly match");

  auto prefix=next::fresh(config,preprocessing.scaler,preprocessing.bank,seed+1);
  auto prefix_optimizer=next::make_optimizer(prefix.model,options);tr::TrainingProgress prefix_progress{0,0,8811,8812,8813};
  const tr::TrainingIdentity identity{"synthetic-v17-engineering","TRAIN-only-two-windows","phase-by-absolute-attempt;no-context-deletion"};
  auto prefix_step=next::train_step(prefix.model,prefix.scaler,*prefix_optimizer,prefix_progress,raw,cutoffs,candidate,options);
  check(prefix_step.updated,"candidate has actual gradient update");++updates;
  const auto training=output/"candidate-training-1.pt",inference=output/"candidate-inference-1.pt";
  next::save_training_state(training.string(),prefix.model,prefix.scaler,*prefix_optimizer,identity,prefix_progress,candidate,options);
  prefix.model->eval();prefix.model->clear_gradients();next::save_checkpoint(inference.string(),prefix.model,prefix.scaler,1,candidate);
  auto loaded=next::load_checkpoint(inference.string(),config.level.device,&config,candidate);
  same_model(prefix.model,loaded.numerical.model);check(loaded.recipe.temporal_weight==.03,"typed inference carries actual coefficient");
  check(bits(prefix.model->encode_at(raw,cutoffs,prefix.scaler).native,
      loaded.numerical.model->encode_at(raw,cutoffs,loaded.numerical.scaler).native),"typed inference export/load exact native output");
  auto poison=raw;poison.data=raw.data.clone();poison.observed=raw.observed.clone();
  poison.data.narrow(2,config.level.history_length,9).fill_(std::numeric_limits<double>::quiet_NaN());
  poison.observed.narrow(2,config.level.history_length,9).logical_not_();
  check(bits(prefix.model->encode_at(raw,cutoffs,prefix.scaler).native,
      prefix.model->encode_at(poison,cutoffs,prefix.scaler).native),"future values/masks cannot affect V17 inference");
  refuses([&]{next::load_checkpoint(inference.string(),config.level.device,&config,control);},"wrong inference recipe rejected");
  refuses([&]{next::load_training_state(training.string(),config,identity,control,options);},"wrong training recipe rejected");
  refuses([&]{cp::load_checkpoint(inference.string(),config.level.device,&config);},"raw family loader rejects V17 outer kind");
  refuses([&]{tr::load_training_state(training.string(),config,identity,options);},"raw family training loader rejects V17 outer kind");
  refuses([&]{next::load_training_state(inference.string(),config,identity,candidate,options);},"inference cannot resume training");
  refuses([&]{next::load_checkpoint(training.string(),config.level.device,&config,candidate);},"training cannot masquerade as inference");
  auto wrong_identity=identity;wrong_identity.recipe_identity+=";different-sampling";
  refuses([&]{next::load_training_state(training.string(),config,wrong_identity,candidate,options);},"wrong caller recipe rejected");
  auto wrong_options=options;wrong_options.learning_rate*=.5;
  refuses([&]{next::load_training_state(training.string(),config,identity,candidate,wrong_options);},"wrong optimizer options rejected");
  refuses([&]{next::save_checkpoint(inference.string(),prefix.model,prefix.scaler,1,candidate);},"existing artifact cannot be overwritten");
  const auto family_path=output/"family-inference.pt";
  original.model->eval();original.model->clear_gradients();cp::save_checkpoint(family_path.string(),original.model,original.scaler,1);
  refuses([&]{next::load_checkpoint(family_path.string(),config.level.device,&config,candidate);},"V17 loader rejects raw family kind");

  auto resumed=next::load_training_state(training.string(),config,identity,candidate,options);
  same_model(prefix.model,resumed.model);same_progress(prefix_progress,resumed.progress);
  check(tr::optimizer_identity(prefix.model,*prefix_optimizer)==tr::optimizer_identity(resumed.model,*resumed.optimizer),"resumed actual AdamW content exact");
  auto direct=next::fresh(config,preprocessing.scaler,preprocessing.bank,seed+1);
  auto direct_optimizer=next::make_optimizer(direct.model,options);tr::TrainingProgress direct_progress{0,0,8811,8812,8813};
  for(int64_t step=0;step<3;++step){const auto batch=history(config,step);
    const auto result=next::train_step(direct.model,direct.scaler,*direct_optimizer,direct_progress,batch,cutoffs,candidate,options);
    check(result.updated,"uninterrupted candidate update succeeds");++updates;
    if(step>0){const auto continued=next::train_step(resumed.model,resumed.scaler,*resumed.optimizer,resumed.progress,batch,cutoffs,candidate,options);
      check(continued.updated&&continued.total_loss==result.total_loss&&continued.gradient_norm==result.gradient_norm,
          "resumed next-step objective and gradient norm exact");++updates;}
  }
  same_model(direct.model,resumed.model);same_progress(direct_progress,resumed.progress);
  check(tr::optimizer_identity(direct.model,*direct_optimizer)==tr::optimizer_identity(resumed.model,*resumed.optimizer),"candidate three-step resumed/uninterrupted AdamW exact");
  check(direct.scaler.identity()==resumed.scaler.identity()&&direct.model->normalizer().identity()==resumed.model->normalizer().identity(),
      "resume preserves frozen fitted state without refit");
  check(cp::dimensions(config).native_width==loaded.numerical.model->encode_at(raw,cutoffs,loaded.numerical.scaler).native.size(1),
      "native width remains derived from supplied configuration");
}
int execute(int argc,char **argv){
  TORCH_CHECK(argc==7,"usage: --settings FILE --alternative-settings FILE --output NEWLEAF");std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2)TORCH_CHECK(args.emplace(argv[i],argv[i+1]).second,"unique options required");
  TORCH_CHECK(args.count("--settings")&&args.count("--alternative-settings")&&args.count("--output"),"explicit options required");
  const fs::path output=args.at("--output");TORCH_CHECK(!fs::exists(output)&&fs::exists(output.parent_path()),"new output leaf in existing parent required");
  TORCH_CHECK(torch::cuda::is_available(),"actual CUDA required");torch::set_num_threads(1);torch::set_num_interop_threads(1);
  fs::create_directory(output);
  refuses([]{next::validate_recipe({-.1});},"negative weight rejected");
  refuses([]{next::validate_recipe({std::numeric_limits<double>::quiet_NaN()});},"NaN weight rejected");
  refuses([]{next::validate_recipe({std::numeric_limits<double>::infinity()});},"infinite weight rejected");
  for(const auto &label:{"reference","alternative"})fs::create_directory(output/label);
  fixture(args.at("--settings"),output/"reference",179101);
  fixture(args.at("--alternative-settings"),output/"alternative",179102);
  std::ostringstream result;result<<"{\"kind\":\"cpb_v17_recipe_engineering_v1\",\"status\":\"passed\",\"assertions\":"<<assertions
      <<",\"named_gradient_tensors_compared\":"<<gradient_tensors<<",\"physical_synthetic_updates\":"<<updates
      <<",\"preprocessing_fits\":"<<fits<<",\"refusals\":"<<refusals<<",\"quality_evaluated\":false}";
  std::ofstream report(output/"summary.json",std::ios::binary);report<<result.str()<<'\n';report.close();
  TORCH_CHECK(report,"write engineering result");std::cout<<result.str()<<'\n';return 0;
}
}
int main(int argc,char **argv){try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
