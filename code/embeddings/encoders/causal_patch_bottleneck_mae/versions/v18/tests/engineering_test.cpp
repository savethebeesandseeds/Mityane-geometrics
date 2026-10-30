// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Synthetic mechanism checks only. No research checkpoints/data or quality scores.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v18.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace tr=cp::training;
namespace next=embedding::encoders::causal_patch_bottleneck_mae::v18;
namespace fs=std::filesystem;
namespace {
int64_t assertions=0,updates=0,gradient_tensors=0,refusals=0,fits=0;
void check(bool value,const char *why){++assertions;TORCH_CHECK(value,"[cpb-v18 engineering] ",why);}
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
      (torch::arange(N,torch::kFloat64)*c.sampling_interval).view({1,1,N}).expand({B,c.channel_count,N}).contiguous(),c.sampling_interval};
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
void same_mask(const tr::MaskBatch &a,const tr::MaskBatch &b){
  check(bits(a.hidden,b.hidden)&&bits(a.visible,b.visible)&&bits(a.target,b.target)&&bits(a.eligible_channels,b.eligible_channels),
      "scattered mask exactly matches preserved family mask");
}
void mask_geometry(const cp::Config &config){
  const auto raw=history(config);const auto cutoffs=cuts(config);const auto &c=config.level;
  const next::Recipe control{next::MaskMode::scattered},candidate{next::MaskMode::contiguous};
  const auto masks=next::make_mask(raw,cutoffs,config,7712,0,candidate);
  const auto repeated=next::make_mask(raw,cutoffs,config,7712,0,candidate);
  same_mask(next::make_mask(raw,cutoffs,config,7712,0,control),tr::make_mask(raw,cutoffs,config,7712,0));
  same_mask(masks,repeated);check(bits(masks.selected_starts,repeated.selected_starts),"span start is deterministic by absolute attempt");
  const auto K=cp::dimensions(config).patch_count;
  const auto selected=cp::implementation::configurable_spectral_pad_observed(masks.hidden,c).reshape({2,c.channel_count,K,-1}).any(-1);
  for(int64_t b=0;b<2;++b)for(int64_t j=0;j<c.channel_count;++j){
    const auto quota=std::clamp<int64_t>(static_cast<int64_t>(std::floor(c.mask_ratio*K+.5)),1,K-2);
    const auto first=masks.selected_starts[b][j].item<int64_t>();
    check(first>=0&&first+quota<=K&&masks.quotas[b][j].item<int64_t>()==quota,"dense row uses exact configured quota within original grid");
    check(masks.legal_span_counts[b][j].item<int64_t>()==K-quota+1,"all dense nonwrapping spans enumerated");
    for(int64_t k=0;k<K;++k)check(selected[b][j][k].item<bool>()==(k>=first&&k<first+quota),"one physical contiguous selected grid span");
  }
  check(masks.unsupported_channels==0&&masks.selected_patch_count==selected.sum().item<int64_t>(),"mask counter coverage and selected patches are explicit");
  auto poison=raw;poison.data=raw.data.clone();poison.observed=raw.observed.clone();
  poison.data.narrow(2,c.history_length,9).fill_(std::numeric_limits<double>::quiet_NaN());
  poison.observed.narrow(2,c.history_length,9).logical_not_();
  same_mask(masks,next::make_mask(poison,cutoffs,config,7712,0,candidate));
  check(next::mask_detail::legal_starts({true,true,false,true,true,true},2)==std::vector<int64_t>({0,3,4}),"holes preserve physical starts rather than compressed observed ranks");
  check(next::mask_detail::legal_starts({true,false,true,false,true,false},2).empty(),"no physical span when support has only isolated patches");
  {
    auto sparse_config=config;sparse_config.level.history_length=std::max<int64_t>(17,12*c.patch_length);
    auto sparse=history(sparse_config);sparse.observed.zero_();const auto sparse_cuts=cuts(sparse_config);
    const auto P=sparse_config.level.patch_length;
    // Six observed but mutually nonadjacent patches have q=2 and no legal span.
    for(int64_t k=0;k<12;k+=2)sparse.observed[0][0].narrow(0,k*P,P).fill_(true);
    const auto blocked=next::make_mask(sparse,sparse_cuts,sparse_config,113,0,candidate);
    check(blocked.no_legal_span_channels==1&&!blocked.hidden.any().item<bool>()&&!blocked.eligible_channels.any().item<bool>(),"no legal span abstains without fallback or hidden targets");
    check(blocked.insufficient_support_channels==2*c.channel_count-1,"empty channels are explicitly unsupported");
    const auto scattered=next::make_mask(sparse,sparse_cuts,sparse_config,113,0,control);
    check(scattered.eligible_channels[0][0].item<bool>()&&scattered.selected_patch_count==2,"scattered control retains legal sparse support independently");
    for(int64_t M=0;M<3;++M){
      sparse.observed.zero_();for(int64_t k=0;k<M;++k)sparse.observed[0][0].narrow(0,k*P,P).fill_(true);
      const auto unsupported=next::make_mask(sparse,sparse_cuts,sparse_config,113,M,candidate);
      check(!unsupported.hidden.any().item<bool>()&&!unsupported.eligible_channels.any().item<bool>()&&unsupported.insufficient_support_channels==2*c.channel_count,
          "fewer than three observed patches abstains with no artificial mask");
    }
  }
  {
    auto partial_config=config;partial_config.level.history_length=33;partial_config.level.patch_length=6;
    auto partial=history(partial_config);partial.observed.fill_(true);const auto partial_cuts=cuts(partial_config);
    bool covered=false;
    for(int64_t attempt=0;attempt<128&&!covered;++attempt){
      const auto choice=next::make_mask(partial,partial_cuts,partial_config,117,attempt,candidate);
      if(choice.selected_starts[0][0].item<int64_t>()!=4)continue;
      covered=true;check(choice.quotas[0][0].item<int64_t>()==2&&choice.hidden.size(2)==33,"partial final patch retains logical H and same rounded quota");
      check(choice.hidden[0][0].narrow(0,24,9).all().item<bool>()&&!choice.hidden[0][0].narrow(0,0,24).any().item<bool>(),"physical span may include observed partial final patch, cropped at H");
    }
    check(covered,"deterministic fixture covers the partial-final-span endpoint");
  }
}
void fixture(const fs::path &settings,const fs::path &output,int64_t seed){
  const auto config=next::read_settings(settings);const auto raw=history(config);const auto cutoffs=cuts(config);
  mask_geometry(config);
  next::PreprocessingFitter fitter(config,{"synthetic-v18-engineering","TRAIN-only-two-windows","centered-Chan-and-bank-v1"});
  fitter.add(raw,cutoffs);const auto preprocessing=fitter.freeze();++fits;
  const next::Recipe control{next::MaskMode::scattered},candidate{next::MaskMode::contiguous};const tr::OptimizerConfig options{};
  auto original=cp::fresh(config,preprocessing.scaler,preprocessing.bank,seed);
  auto unchanged=next::fresh(config,preprocessing.scaler,preprocessing.bank,seed);
  same_model(original.model,unchanged.model);
  const auto masks=tr::make_mask(raw,cutoffs,config,7712,0);
  same_mask(masks,next::make_mask(raw,cutoffs,config,7712,0,control));
  original.model->train();unchanged.model->train();
  auto base_trace=original.model->forward_at(raw,cutoffs,masks.hidden,original.scaler);
  auto same_trace=next::forward_at(unchanged.model,raw,cutoffs,masks.hidden,unchanged.scaler,control);
  check(bits(base_trace.output.loss,same_trace.output.loss)&&bits(base_trace.nll.loss,same_trace.nll.loss)&&
      bits(base_trace.objective.temporal.loss,same_trace.objective.temporal.loss),"scattered-control forward/loss exactly matches unchanged family");
  base_trace.output.loss.backward();same_trace.output.loss.backward();
  const auto left=original.model->named_parameters(),right=unchanged.model->named_parameters();
  for(const auto &p:left){check(bits(p.value().grad(),right[p.key()].grad()),"scattered-control named gradient exactly matches unchanged family");
    gradient_tensors+=p.value().grad().defined();}
  original.model->clear_gradients();unchanged.model->clear_gradients();
  auto original_optimizer=tr::make_optimizer(original.model,options),unchanged_optimizer=next::make_optimizer(unchanged.model,options);
  tr::TrainingProgress original_progress{0,0,7711,7712,7713},unchanged_progress=original_progress;
  const auto old_step=tr::train_step(original.model,original.scaler,*original_optimizer,original_progress,raw,cutoffs,options);
  const auto same_step=next::train_step(unchanged.model,unchanged.scaler,*unchanged_optimizer,unchanged_progress,raw,cutoffs,control,options);
  updates+=2;check(old_step.updated&&same_step.updated&&old_step.total_loss==same_step.total_loss&&
      old_step.gradient_norm==same_step.gradient_norm,"scattered-control step loss and clipping exactly match");
  same_model(original.model,unchanged.model);same_progress(original_progress,unchanged_progress);
  check(tr::optimizer_identity(original.model,*original_optimizer)==tr::optimizer_identity(unchanged.model,*unchanged_optimizer),
      "scattered-control AdamW moment content/order/steps exactly match");

  auto prefix=next::fresh(config,preprocessing.scaler,preprocessing.bank,seed+1);
  auto prefix_optimizer=next::make_optimizer(prefix.model,options);tr::TrainingProgress prefix_progress{0,0,8811,8812,8813};
  const tr::TrainingIdentity identity{"synthetic-v18-engineering","TRAIN-only-two-windows","phase-by-absolute-attempt;no-context-deletion"};
  auto prefix_step=next::train_step(prefix.model,prefix.scaler,*prefix_optimizer,prefix_progress,raw,cutoffs,candidate,options);
  check(prefix_step.updated,"candidate has actual gradient update");++updates;
  check(prefix_step.selected_patch_count>0&&prefix_step.unsupported_channels==0,"candidate step reports actual query geometry");
  const auto prefix_mask=next::make_mask(raw,cutoffs,config,8812,0,candidate);
  const std::array<int64_t,3> lags{{1,4,16}};
  for(size_t index=0;index<3;++index){const auto lag=lags[index],length=config.level.history_length-lag;
    const auto count=(prefix_mask.target.narrow(2,lag,length)&prefix_mask.target.narrow(2,0,length)).sum().item<int64_t>();
    check(prefix_step.temporal_pair_counts[index]==count,"candidate step reports actual1/4/16 pair counts without expanding support");
  }
  const auto training=output/"candidate-training-1.pt",inference=output/"candidate-inference-1.pt";
  next::save_training_state(training.string(),prefix.model,prefix.scaler,*prefix_optimizer,identity,prefix_progress,candidate,options);
  prefix.model->eval();prefix.model->clear_gradients();next::save_checkpoint(inference.string(),prefix.model,prefix.scaler,1,candidate);
  auto loaded=next::load_checkpoint(inference.string(),config.level.device,&config,candidate);
  same_model(prefix.model,loaded.numerical.model);check(loaded.recipe.mask_mode==next::MaskMode::contiguous,"typed inference carries actual mask geometry");
  check(bits(prefix.model->encode_at(raw,cutoffs,prefix.scaler).native,
      loaded.numerical.model->encode_at(raw,cutoffs,loaded.numerical.scaler).native),"typed inference export/load exact native output");
  auto poison=raw;poison.data=raw.data.clone();poison.observed=raw.observed.clone();
  poison.data.narrow(2,config.level.history_length,9).fill_(std::numeric_limits<double>::quiet_NaN());
  poison.observed.narrow(2,config.level.history_length,9).logical_not_();
  check(bits(prefix.model->encode_at(raw,cutoffs,prefix.scaler).native,
      prefix.model->encode_at(poison,cutoffs,prefix.scaler).native),"future values/masks cannot affect V18 inference");
  refuses([&]{next::load_checkpoint(inference.string(),config.level.device,&config,control);},"wrong inference recipe rejected");
  refuses([&]{next::load_training_state(training.string(),config,identity,control,options);},"wrong training recipe rejected");
  const auto legacy_kind=output/"wrong-legacy-outer-kind.pt";
  {torch::serialize::OutputArchive archive;
    next::artifact_detail::write_recipe(archive,candidate,"cpb_v17_weighted_temporal_inference_v1");
    next::artifact_detail::d::publish_checkpoint(archive,legacy_kind.string());}
  refuses([&]{next::load_checkpoint(legacy_kind.string(),config.level.device,&config,candidate);},"V17 outer kind rejected despite V18-like header");
  const auto wrong_type=output/"wrong-mode-dtype.pt";
  {torch::serialize::OutputArchive archive;
    const auto text=[&](const char *key,const std::string &value){archive.write(key,next::artifact_detail::d::text_tensor(value),true);};
    text("artifact_kind",next::kInferenceKind);text("model_tag",next::kModelTag);text("recipe_kind",next::kRecipeKind);
    text("recipe_binding",next::recipe_identity(candidate));text("implementation_payload_codec",next::kPayloadCodec);text("implementation_payload_meaning",next::kPayloadMeaning);
    archive.write("mask_mode_i64",torch::tensor(1.,torch::kFloat64),true);archive.write("temporal_weight_f64",torch::tensor(.1,torch::kFloat64),true);
    next::artifact_detail::d::publish_checkpoint(archive,wrong_type.string());}
  refuses([&]{next::load_checkpoint(wrong_type.string(),config.level.device,&config,candidate);},"wrong mask-mode dtype rejected before any numerical payload");
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
  refuses([]{next::validate_recipe({static_cast<next::MaskMode>(-1)});},"negative mask enum rejected");
  refuses([]{next::validate_recipe({static_cast<next::MaskMode>(2)});},"unknown mask enum rejected");
  refuses([]{next::validate_recipe({static_cast<next::MaskMode>(std::numeric_limits<int64_t>::max())});},"large unknown mask enum rejected");
  for(const auto &label:{"reference","alternative"})fs::create_directory(output/label);
  fixture(args.at("--settings"),output/"reference",179101);
  fixture(args.at("--alternative-settings"),output/"alternative",179102);
  std::ostringstream result;result<<"{\"kind\":\"cpb_v18_recipe_engineering_v1\",\"status\":\"passed\",\"assertions\":"<<assertions
      <<",\"named_gradient_tensors_compared\":"<<gradient_tensors<<",\"physical_synthetic_updates\":"<<updates
      <<",\"preprocessing_fits\":"<<fits<<",\"refusals\":"<<refusals<<",\"quality_evaluated\":false}";
  std::ofstream report(output/"summary.json",std::ios::binary);report<<result.str()<<'\n';report.close();
  TORCH_CHECK(report,"write engineering result");std::cout<<result.str()<<'\n';return 0;
}
}
int main(int argc,char **argv){try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
