// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Artificial mechanism/serialization checks, not research-quality evaluation.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v19.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace tr=cp::training;
namespace next=embedding::encoders::causal_patch_bottleneck_mae::v19;
namespace fs=std::filesystem;
namespace {
int64_t assertions=0,updates=0,gradient_tensors=0,refusals=0,fits=0,teacher_forwards=0;
void check(bool condition,const char *why){++assertions;TORCH_CHECK(condition,"[cpb-v19 engineering] ",why);}
template<class F>void refuses(F action,const char *why){bool caught=false;try{action();}catch(const std::exception&){caught=true;}check(caught,why);++refusals;}
bool bits(const torch::Tensor &a,const torch::Tensor &b){return a.defined()==b.defined()&&(!a.defined()||cp::boundary::same_bits(a,b));}
cp::RawHistory history(const cp::Config &config,int64_t phase=0){
  const auto &c=config.level;const int64_t B=2,N=c.history_length+9;
  auto data=torch::zeros({B,c.channel_count,N,c.input_width},torch::kFloat64),observed=torch::ones(data.sizes(),torch::kBool);
  auto x=data.accessor<double,4>();auto m=observed.accessor<bool,4>();
  for(int64_t b=0;b<B;++b)for(int64_t j=0;j<c.channel_count;++j)for(int64_t t=0;t<N;++t)for(int64_t f=0;f<c.input_width;++f){
    x[b][j][t][f]=20.+2*j+.2*f+.3*b+.021*t+.009*phase+.7*std::sin(.19*t+.31*f+.27*j+.17*b+.03*phase);
    m[b][j][t][f]=(t+3*f+j+b)%37!=0;
  }
  return{data,observed,torch::tensor(cp::implementation::resolved_channel_ids(c),torch::kInt64),(torch::arange(N,torch::kFloat64)*c.sampling_interval).view({1,1,N}).expand({B,c.channel_count,N}).contiguous(),c.sampling_interval};
}
torch::Tensor cuts(const cp::Config &c){return torch::full({2},(c.level.history_length-1)*c.level.sampling_interval,torch::kFloat64);}
void same_model(const cp::Encoder &a,const cp::Encoder &b){
  const auto x=a->named_parameters(),y=b->named_parameters();check(x.size()==y.size(),"same parameter inventory");
  for(const auto &p:x)check(y.contains(p.key())&&bits(p.value(),y[p.key()]),"exact model parameter tensor");
  const auto xb=a->named_buffers(),yb=b->named_buffers();check(xb.size()==yb.size(),"same buffer inventory");
  for(const auto &p:xb)check(yb.contains(p.key())&&bits(p.value(),yb[p.key()]),"exact model buffer tensor");
  check(a->state_identity()==b->state_identity(),"same complete numerical state identity");
}
void same_bundle(const next::TrainingBundle &a,const next::TrainingBundle &b){
  same_model(a.model,b.model);same_model(a.teacher,b.teacher);
  check(next::predictor_identity(a.predictor)==next::predictor_identity(b.predictor),"exact predictor state identity");
  for(const auto &p:a.predictor->named_parameters())check(bits(p.value(),b.predictor->named_parameters()[p.key()]),"exact predictor weight tensor");
  check(a.teacher_updates==b.teacher_updates&&a.predictor_seed==b.predictor_seed&&a.initial_teacher_identity==b.initial_teacher_identity&&a.initial_predictor_identity==b.initial_predictor_identity,"exact teacher/predictor metadata");
}
void same_progress(const tr::TrainingProgress &a,const tr::TrainingProgress &b){
  check(a.completed_updates==b.completed_updates&&a.attempted_batches==b.attempted_batches&&a.row_seed==b.row_seed&&a.mask_seed==b.mask_seed&&a.context_seed==b.context_seed,"exact progress and stochastic counter seeds");
}
void arithmetic(){
  auto p=torch::tensor({{3.f,4.f},{0.f,0.f},{0.f,0.f}},torch::kFloat32).to(torch::kCUDA).requires_grad_(true);
  auto t=torch::tensor({{6.f,8.f},{1.f,0.f},{0.f,0.f}},torch::kFloat32).to(torch::kCUDA).requires_grad_(true);
  const auto support=torch::tensor({true,true,false},torch::kBool).to(torch::kCUDA);
  const auto out=next::auxiliary_loss(p,t,support);check(out.rows==2&&out.loss.scalar_type()==torch::kFloat64&&std::abs(out.loss.item<double>()-1.)<1e-14,"manual normalized-dot arithmetic includes a zero predictor row without a gate");
  out.loss.backward();check(p.grad().defined()&&torch::isfinite(p.grad()).all().item<bool>()&&!t.grad().defined(),"finite zero-norm predictor derivative and detached teacher");
  check(p.grad()[2].eq(0).all().item<bool>(),"unsupported row has zero auxiliary gradient");
  const auto d=next::dispersion(p,support);check(d.supported&&d.rows==2&&d.near_zero_rows==1&&d.norm_min==0&&d.norm_max==5,"eligible population counts and pre-normalization norms");
  check(bits(d.coordinate_std,torch::tensor({1.5,2.},torch::kFloat64).to(torch::kCUDA))&&d.std_mean==1.75&&d.std_min==1.5&&d.std_max==2.,"manual coordinate population standard deviations");
  const auto single=next::dispersion(p,torch::tensor({true,false,false},torch::kBool).to(torch::kCUDA));check(single.rows==1&&single.coordinate_std.eq(0).all().item<bool>(),"one eligible row has population standard deviation zero");
  p.mutable_grad()=torch::Tensor();const auto empty_support=torch::zeros({3},support.options());const auto empty=next::auxiliary_loss(p,t,empty_support);empty.loss.backward();
  check(empty.rows==0&&empty.loss.item<double>()==0&&p.grad().eq(0).all().item<bool>(),"empty auxiliary remains a graph-connected zero");
  const auto none=next::dispersion(p,empty_support);check(!none.supported&&none.rows==0&&none.near_zero_rows==0&&none.std_mean==0&&none.norm_min==0&&none.coordinate_std.eq(0).all().item<bool>(),"empty diagnostics are finite explicit zeros");
}
void fixture(const fs::path &settings,const fs::path &directory,int64_t seed){
  const auto config=next::read_settings(settings);const auto raw=history(config);const auto cutoffs=cuts(config);const next::Recipe recipe{};const tr::OptimizerConfig options{};
  next::PreprocessingFitter fitter(config,{"synthetic-v19-engineering","TRAIN-two-distinct-windows","centered-Chan-and-bank-v1"});fitter.add(raw,cutoffs);const auto prep=fitter.freeze();++fits;
  auto base=cp::fresh(config,prep.scaler,prep.bank,seed);auto bundle=next::fresh(config,prep.scaler,prep.bank,seed,seed+4000);
  same_model(base.model,bundle.model);same_model(bundle.model,bundle.teacher);
  check(bundle.predictor->width()==cp::dimensions(config).shape_width&&bundle.predictor->first->weight.size(0)==2*cp::dimensions(config).shape_width,"predictor dimensions derive from supported configuration");
  const auto mask=tr::make_mask(raw,cutoffs,config,4412,0),candidate_mask=next::make_mask(raw,cutoffs,config,4412,0);
  check(bits(mask.hidden,candidate_mask.hidden)&&bits(mask.target,candidate_mask.target)&&bits(mask.visible,candidate_mask.visible)&&bits(mask.eligible_channels,candidate_mask.eligible_channels),"scattered masks match unchanged family exactly");
  base.model->train();bundle.model->train();const auto baseline=base.model->forward_at(raw,cutoffs,mask.hidden,base.scaler);
  const auto trace=next::forward_at(bundle,raw,cutoffs,mask.hidden);++teacher_forwards;
  check(bits(baseline.output.loss,trace.base.output.loss)&&bits(baseline.nll.loss,trace.base.nll.loss)&&bits(baseline.objective.temporal.loss,trace.base.objective.temporal.loss),"base objective and forward exactly match unchanged family");
  check(bits(trace.total_loss,trace.base.output.loss+.1*trace.auxiliary.loss)&&trace.auxiliary.rows==2,"combined objective decomposition and actual auxiliary support");
  trace.auxiliary.loss.backward();bool encoder_reached=false,predictor_reached=false;
  for(const auto &p:bundle.model->named_parameters()){
    const auto g=p.value().grad();if(g.defined()){++gradient_tensors;check(torch::isfinite(g).all().item<bool>(),"finite auxiliary student gradient");if(g.abs().sum().item<double>()>0)encoder_reached=true;}
    if(p.key().find("decoder")!=std::string::npos||p.key().find("variance_")!=std::string::npos||p.key().find("change_projection")!=std::string::npos)
      check(!g.defined()||g.eq(0).all().item<bool>(),"auxiliary alone does not reach waveform/variance decoder or change suffix");
  }
  for(const auto &p:bundle.predictor->parameters()){check(p.grad().defined()&&torch::isfinite(p.grad()).all().item<bool>(),"finite actual predictor auxiliary gradient");predictor_reached=predictor_reached||p.grad().abs().sum().item<double>()>0;++gradient_tensors;}
  for(const auto &p:bundle.teacher->parameters())check(!p.grad().defined(),"teacher has no auxiliary gradients");
  check(encoder_reached&&predictor_reached,"auxiliary reaches learned encoder and training-only predictor");bundle.model->clear_gradients();bundle.predictor->zero_grad(true);
  refuses([&]{next::update_teacher(bundle);},"EMA before optimizer update is refused");
  auto optimizer=next::make_optimizer(bundle,options);tr::TrainingProgress progress{0,0,4411,4412,4413};
  const auto named_teacher=bundle.teacher->named_parameters();std::map<std::string,torch::Tensor> before;
  for(const auto &p:named_teacher)before.emplace(p.key(),p.value().detach().clone());
  const auto step=next::train_step(bundle,*optimizer,progress,raw,cutoffs);check(step.updated,"actual joint student/predictor optimizer update");++updates;teacher_forwards+=step.teacher_forwards;
  check(step.teacher_updates==1&&bundle.teacher_updates==1&&step.teacher_forwards==1&&step.auxiliary_rows==2,"one full-view teacher forward and one post-optimizer EMA per update");
  check(step.student_dispersion.rows==2&&step.teacher_dispersion.rows==2&&step.predictor_dispersion.rows==2,"actual eligible diagnostic population recorded");
  const auto student=bundle.model->named_parameters(),teacher=bundle.teacher->named_parameters();
  for(const auto &p:student){auto expected=before.at(p.key()).clone();if(p.value().requires_grad())expected.mul_(.99).add_(p.value(),.01);
    check(bits(expected,teacher[p.key()]),"exact F32 EMA formula uses post-optimizer student and preserves fixed priors");}
  refuses([&]{next::update_teacher(bundle);},"duplicate EMA for the same optimizer update is refused");
  const auto training=directory/"training-1.pt",inference=directory/"inference-1.pt";
  const tr::TrainingIdentity identity{"synthetic-v19-engineering","TRAIN-two-distinct-windows","source-order-fitting;absolute-counter-batches;no-context-deletion"};
  next::save_training_state(training.string(),bundle,*optimizer,identity,progress);bundle.model->eval();bundle.model->clear_gradients();next::save_checkpoint(inference.string(),bundle.model,bundle.scaler,1);
  auto loaded=next::load_checkpoint(inference.string(),config.level.device,&config);same_model(bundle.model,loaded.numerical.model);
  check(bits(bundle.model->encode_at(raw,cutoffs,bundle.scaler).native,loaded.numerical.model->encode_at(raw,cutoffs,loaded.numerical.scaler).native),"student-only typed inference load preserves exact native output");
  auto poison=raw;poison.data=raw.data.clone();poison.observed=raw.observed.clone();poison.data.narrow(2,config.level.history_length,9).fill_(std::numeric_limits<double>::quiet_NaN());poison.observed.narrow(2,config.level.history_length,9).logical_not_();
  check(bits(bundle.model->encode_at(raw,cutoffs,bundle.scaler).native,bundle.model->encode_at(poison,cutoffs,bundle.scaler).native),"student inference ignores future values and support");
  check(bits(bundle.teacher->encode_at(raw,cutoffs,bundle.scaler).native,bundle.teacher->encode_at(poison,cutoffs,bundle.scaler).native),"full observed teacher respects the same cutoff");
  teacher_forwards+=2;
  bundle.model->train();const auto ordinary=next::forward_at(bundle,raw,cutoffs,mask.hidden),poisoned=next::forward_at(bundle,poison,cutoffs,mask.hidden);teacher_forwards+=2;
  check(bits(ordinary.total_loss,poisoned.total_loss)&&bits(ordinary.teacher_learned,poisoned.teacher_learned)&&bits(ordinary.student_learned,poisoned.student_learned),"both training paths crop before reading future values/support");bundle.model->clear_gradients();
  auto resumed=next::load_training_state(training.string(),config,identity);same_bundle(bundle,resumed.bundle);same_progress(progress,resumed.progress);
  check(next::optimizer_identity(bundle,*optimizer)==next::optimizer_identity(resumed.bundle,*resumed.optimizer),"serialized combined AdamW content/order/steps exact");
  auto direct=next::fresh(config,prep.scaler,prep.bank,seed,seed+4000);auto direct_optimizer=next::make_optimizer(direct);tr::TrainingProgress direct_progress{0,0,4411,4412,4413};
  for(int64_t phase=0;phase<3;++phase){const auto batch=history(config,phase);const auto result=next::train_step(direct,*direct_optimizer,direct_progress,batch,cutoffs);check(result.updated,"uninterrupted candidate update succeeds");++updates;teacher_forwards+=result.teacher_forwards;
    if(phase>0){const auto continued=next::train_step(resumed.bundle,*resumed.optimizer,resumed.progress,batch,cutoffs);check(continued.updated&&continued.total_loss==result.total_loss&&continued.auxiliary_loss==result.auxiliary_loss&&continued.gradient_norm==result.gradient_norm,"resumed next-step objective and global gradient norm exact");++updates;teacher_forwards+=continued.teacher_forwards;}}
  same_bundle(direct,resumed.bundle);same_progress(direct_progress,resumed.progress);check(next::optimizer_identity(direct,*direct_optimizer)==next::optimizer_identity(resumed.bundle,*resumed.optimizer),"three-step full teacher/predictor/AdamW resume exact");
  // Sparse support performs no optimizer/EMA/teacher forward and does not refit.
  auto absent=raw;absent.observed=torch::zeros_like(raw.observed);const auto state_before=direct.model->state_identity(),teacher_before=direct.teacher->state_identity(),pred_before=next::predictor_identity(direct.predictor),opt_before=next::optimizer_identity(direct,*direct_optimizer);
  const auto skipped=next::train_step(direct,*direct_optimizer,direct_progress,absent,cutoffs);
  check(!skipped.updated&&skipped.teacher_forwards==0&&skipped.teacher_updates==3&&direct_progress.attempted_batches==4&&direct_progress.completed_updates==3,"unsupported batch increments attempt only and skips teacher/optimizer");
  check(state_before==direct.model->state_identity()&&teacher_before==direct.teacher->state_identity()&&pred_before==next::predictor_identity(direct.predictor)&&opt_before==next::optimizer_identity(direct,*direct_optimizer),"unsupported batch leaves all trainable and optimizer state unchanged");
  const auto skipped_state=directory/"training-after-skipped-attempt.pt";next::save_training_state(skipped_state.string(),direct,*direct_optimizer,identity,direct_progress);
  auto resumed_skip=next::load_training_state(skipped_state.string(),config,identity);same_bundle(direct,resumed_skip.bundle);same_progress(direct_progress,resumed_skip.progress);
  refuses([&]{next::load_training_state(inference.string(),config,identity);},"inference artifact cannot resume training");
  refuses([&]{next::load_checkpoint(training.string(),config.level.device,&config);},"training artifact cannot masquerade as inference");
  refuses([&]{cp::load_checkpoint(inference.string(),config.level.device,&config);},"family loader rejects new typed inference kind");
  refuses([&]{tr::load_training_state(training.string(),config,identity);},"family loader rejects new typed training kind");
  auto wrong=identity;wrong.input_source_identity+="different";refuses([&]{next::load_training_state(training.string(),config,wrong);},"wrong source identity is refused");
  wrong=identity;wrong.split_identity+="different";refuses([&]{next::load_training_state(training.string(),config,wrong);},"wrong TRAIN split identity is refused");
  wrong=identity;wrong.recipe_identity+="different";refuses([&]{next::load_training_state(training.string(),config,wrong);},"wrong caller training recipe is refused");
  auto wrong_options=options;wrong_options.learning_rate*=.5;refuses([&]{next::load_training_state(training.string(),config,identity,{},wrong_options);},"wrong optimizer options are refused");
  auto wrong_config=config;wrong_config.change_width+=1;refuses([&]{next::load_training_state(training.string(),wrong_config,identity);},"wrong configured latent width is refused");
  refuses([&]{next::load_checkpoint(inference.string(),config.level.device,&config,next::Recipe{.2,.99,1e-6});},"wrong auxiliary recipe is refused");
  refuses([&]{next::load_training_state(training.string(),config,identity,next::Recipe{.1,.9,1e-6});},"wrong teacher decay is refused");
  const auto old_kind=directory/"wrong-legacy-kind.pt";{torch::serialize::OutputArchive a;next::artifact_detail::write_recipe(a,{},"cpb_v18_mask_geometry_inference_v1");next::artifact_detail::d::publish_checkpoint(a,old_kind.string());}
  refuses([&]{next::load_checkpoint(old_kind.string(),config.level.device,&config);},"legacy outer kind is refused before reading payload");
  refuses([&]{next::save_checkpoint(inference.string(),bundle.model,bundle.scaler,1);},"existing inference artifact cannot be overwritten");
  auto wrong_seed=bundle;wrong_seed.predictor_seed+=1;const auto wrong_seed_path=directory/"wrong-predictor-seed.pt";
  next::save_training_state(wrong_seed_path.string(),wrong_seed,*optimizer,identity,progress);
  refuses([&]{next::load_training_state(wrong_seed_path.string(),config,identity);},"wrong predictor seed cannot match bound initial predictor state");
  auto wrong_count=bundle;wrong_count.teacher_updates+=1;refuses([&]{next::validate_teacher(wrong_count);},"teacher and optimizer update count mismatch is refused");
  auto teacher_parameter=bundle.teacher->parameters().front();teacher_parameter.mutable_grad()=torch::ones_like(teacher_parameter);
  refuses([&]{next::validate_teacher(bundle);},"unexpected teacher gradient is refused");teacher_parameter.mutable_grad()=torch::Tensor();
  auto reversed=next::learned_parameters(bundle);std::reverse(reversed.begin(),reversed.end());torch::optim::AdamW wrong_order(reversed,tr::adamw_options());
  refuses([&]{next::validate_optimizer(bundle,wrong_order,progress);},"optimizer parameter order substitution is refused");
  auto with_teacher=next::learned_parameters(bundle);for(const auto &p:bundle.teacher->parameters())if(p.requires_grad())with_teacher.push_back(p);torch::optim::AdamW teacher_optimizer(with_teacher,tr::adamw_options());
  refuses([&]{next::validate_optimizer(bundle,teacher_optimizer,progress);},"teacher cannot enter optimizer groups");
  // Separate tiny construction demonstrates predictor seed is a portable choice,
  // while this fixture/study explicitly uses encoder seed plus4000.
  auto independent_seed=next::fresh(config,prep.scaler,prep.bank,seed,7);check(independent_seed.predictor_seed==7&&independent_seed.model->initial_state_identity()==bundle.model->initial_state_identity(),"explicit predictor seed does not constrain encoder construction");
  refuses([&]{next::fresh(config,prep.scaler,prep.bank,seed,-1);},"negative predictor seed is refused");
}
int execute(int argc,char **argv){
  TORCH_CHECK(argc==7,"usage: --settings FILE --alternative-settings FILE --output NEWLEAF");std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2)TORCH_CHECK(args.emplace(argv[i],argv[i+1]).second,"unique explicit options required");
  TORCH_CHECK(args.count("--settings")&&args.count("--alternative-settings")&&args.count("--output"),"settings and output required");
  const fs::path output=args.at("--output");TORCH_CHECK(!fs::exists(output)&&fs::exists(output.parent_path()),"new output leaf in existing parent required");
  TORCH_CHECK(torch::cuda::is_available(),"actual CUDA required");torch::set_num_threads(1);torch::set_num_interop_threads(1);fs::create_directory(output);
  arithmetic();for(const auto *label:{"reference","alternative"})fs::create_directory(output/label);
  fixture(args.at("--settings"),output/"reference",199101);fixture(args.at("--alternative-settings"),output/"alternative",199102);
  std::ostringstream out;out<<"{\"kind\":\"cpb_v19_teacher_engineering_v1\",\"status\":\"passed\",\"assertions\":"<<assertions<<",\"named_auxiliary_gradient_tensors\":"<<gradient_tensors<<",\"physical_synthetic_updates\":"<<updates<<",\"preprocessing_fits\":"<<fits<<",\"teacher_forwards\":"<<teacher_forwards<<",\"refusals\":"<<refusals<<",\"quality_evaluated\":false}";
  std::ofstream report(output/"summary.json",std::ios::binary);report<<out.str()<<'\n';report.close();TORCH_CHECK(report,"write engineering summary");std::cout<<out.str()<<'\n';return 0;
}
}
int main(int argc,char **argv){try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
