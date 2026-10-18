// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// One separately declared causal trajectory from fresh initialization. No readout fitting.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v4.h"
#include "include/market_data.h"
#include "include/context_deletion.h"
#include "include/calibration.h"
#include <torch/cuda.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <unistd.h>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v4;
namespace impl=cp::implementation;
namespace tr=cp::training;
namespace md=causal_fresh_v4::market;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
constexpr int64_t batch=8,updates=2048,max_attempts=4096,row_seed=943002,mask_seed=943003,context_seed=943004,query_seed=943005;
void check(bool yes,const std::string &why){TORCH_CHECK(yes,"[CPB-v4 train] ",why);}
double elapsed(const Clock::time_point &start){return std::chrono::duration<double>(Clock::now()-start).count();}
std::string quote(const std::string &s){std::ostringstream out;out<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}return out.str()+'"';}
void write(const fs::path &path,const std::string &contents){check(!fs::exists(path),"preserve existing output: "+path.string());std::ofstream out(path,std::ios::binary);check(bool(out),"create output");out<<contents<<'\n';out.close();check(bool(out),"save output");}
void publish(torch::serialize::OutputArchive &archive,const fs::path &path){check(!fs::exists(path),"exclusive archive");archive.save_to(path.string());}
std::string vector_json(const torch::Tensor &tensor){const auto x=tensor.detach().cpu().to(torch::kFloat64).flatten().contiguous();std::ostringstream out;out<<std::setprecision(17)<<'[';for(int64_t i=0;i<x.numel();++i){if(i)out<<',';out<<x[i].item<double>();}return out.str()+']';}
struct Pin{fs::path path;std::string hash;};
void verify(const std::vector<Pin> &pins){for(const auto &p:pins)check(cp::boundary::sha256_file(p.path)==p.hash,"frozen input/source changed: "+p.path.string());}
struct Args {
  fs::path settings,shared,output;std::vector<fs::path> caches;
  Args(int argc,char **argv){for(int i=1;i<argc;i+=2){check(i+1<argc,"flag/value pairs required");const std::string flag=argv[i];const fs::path value=argv[i+1];
      if(flag=="--cache")caches.push_back(value);
      else if(flag=="--settings"){check(settings.empty(),"duplicate settings");settings=value;}
      else if(flag=="--shared"){check(shared.empty(),"duplicate shared");shared=value;}
      else if(flag=="--output"){check(output.empty(),"duplicate output");output=value;}else check(false,"unknown argument: "+flag);}
    check(!settings.empty()&&!shared.empty()&&!output.empty()&&caches.size()==3,"--settings --shared --cache BTC --cache ETH --cache SOL --output NEWLEAF required");
    settings=fs::canonical(settings);shared=fs::canonical(shared);output=fs::absolute(output).lexically_normal();
    for(auto &p:caches)p=fs::canonical(p);check(!fs::exists(output)&&fs::exists(output.parent_path()),"exclusive leaf in existing attempt required");
  }
};
struct Selections {
  std::vector<int64_t> train,probe_train,probe_validation;
  explicit Selections(const fs::path &shared){
    train=md::anchors(md::train_begin,md::train_end,256,4096,false);
    probe_train=md::anchors(md::train_begin,md::train_end,256,112,true);
    probe_validation=md::anchors(md::validation_begin,md::validation_end,256,32,true);
    torch::serialize::InputArchive archive;archive.load_from((shared/"selections.pt").string(),torch::kCPU);
    const auto read=[&](const char *key,const std::vector<int64_t> &rows){torch::Tensor actual;archive.read(key,actual,true);check(actual.device().is_cpu()&&actual.scalar_type()==torch::kInt64&&torch::equal(actual,torch::tensor(rows,torch::kInt64)),"exact frozen selection: "+std::string(key));};
    read("train_anchors",train);read("probe_train_anchors",probe_train);read("probe_validation_anchors",probe_validation);
    read("roles_and_strides",{md::train_begin,md::train_end,md::validation_begin,md::validation_end,md::unused_begin,60000,256,128,256*60000});
  }
};
struct QueryPlan{
  torch::Tensor hidden,target,eligible;
  QueryPlan(const fs::path &shared,const std::string &role,const std::vector<int64_t> &rows){torch::serialize::InputArchive archive;archive.load_from((shared/("query-plan-"+role+".pt")).string(),torch::kCPU);
    torch::Tensor ids;archive.read("row_ids",ids,true);check(ids.scalar_type()==torch::kInt64&&torch::equal(ids,torch::tensor(rows,torch::kInt64)),"fixed query row association");
    archive.read("hidden_recent128",hidden,true);archive.read("target_recent128",target,true);archive.read("eligible_channels",eligible,true);
    for(const auto &x:{hidden,target})check(x.device().is_cpu()&&x.scalar_type()==torch::kBool&&x.sizes()==torch::IntArrayRef({static_cast<int64_t>(rows.size()),3,128,9}),"fixed query mask");
    check(eligible.device().is_cpu()&&eligible.scalar_type()==torch::kBool&&eligible.sizes()==torch::IntArrayRef({static_cast<int64_t>(rows.size()),3}),"fixed query eligibility");
  }
};
struct SavedTensor{std::string name;torch::Tensor value;};
struct Frozen {
  std::vector<SavedTensor> prior,buffers;cp::FrozenScaler scaler;decltype(std::declval<cp::Encoder>()->normalizer().clone()) normalizer;
  Frozen(const cp::Encoder &model,const cp::FrozenScaler &s):scaler{s.mean.clone(),s.scale.clone(),s.count.clone(),s.channel_ids.clone(),s.scale_floor,s.floor_applied.clone()},normalizer(model->normalizer().clone()){
    for(const auto &p:model->named_parameters())if(!p.value().requires_grad())prior.push_back({p.key(),p.value().detach().cpu().clone()});
    for(const auto &p:model->named_buffers())buffers.push_back({p.key(),p.value().detach().cpu().clone()});
    check(!prior.empty(),"literal fixed prior is present");
  }
  void validate(const cp::Encoder &model,const cp::FrozenScaler &s)const{
    model->core()->validate_state();model->core()->validate_fixed_prior();
    for(const auto &p:prior){const auto x=model->named_parameters()[p.name];check(!x.requires_grad()&&!x.grad().defined()&&cp::boundary::same_bits(x,p.value),"fixed literal prior bytes/gradient");}
    check(model->named_buffers().size()==buffers.size(),"same buffer registration");for(const auto &p:buffers)check(cp::boundary::same_bits(model->named_buffers()[p.name],p.value),"immutable registered buffer");
    const auto &n=model->normalizer();check(n.identity()==normalizer.identity()&&n.source_identity==normalizer.source_identity&&n.input_scaler_identity==normalizer.input_scaler_identity&&n.frozen,"fixed bank metadata");
    for(const auto &[a,b]:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{n.counts,normalizer.counts},{n.sum,normalizer.sum},{n.sumsq,normalizer.sumsq},{n.mean,normalizer.mean},{n.scale,normalizer.scale},{n.support,normalizer.support}})check(cp::boundary::same_bits(a,b),"fixed normalizer tensor");
    check(s.identity()==scaler.identity()&&s.scale_floor==scaler.scale_floor,"fixed input scaler metadata");
    for(const auto &[a,b]:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{s.mean,scaler.mean},{s.scale,scaler.scale},{s.count,scaler.count},{s.channel_ids,scaler.channel_ids},{s.floor_applied,scaler.floor_applied}})check(cp::boundary::same_bits(a,b),"fixed scaler tensor");
  }
};
struct QueryCapture {double mae{0},nll{0},seconds{0};int64_t valid{0},cells{0},calls{0};torch::Tensor row_mae,row_nll,row_cells;};
QueryCapture capture(md::Dataset &data,cp::Encoder &model,const cp::FrozenScaler &scaler,const std::vector<int64_t> &rows,
    const QueryPlan &plan,int64_t begin,int64_t end,const fs::path &witness){
  torch::NoGradGuard no_grad;model->eval();QueryCapture result;std::vector<torch::Tensor> maes,nlls,counts;
  const auto start=Clock::now();
  for(size_t offset=0;offset<rows.size();offset+=batch){const auto anchors=md::slice(rows,offset,batch);const auto raw=data.load(anchors,begin,end);
    const auto hidden=plan.hidden.narrow(0,offset,anchors.size());
    auto query_config=model->config().level;query_config.mask_ratio=1./16.;
    const auto generated=impl::configurable_spectral_mask_from_hidden(raw.observed,hidden,query_config);
    check(torch::equal(generated.hidden.reshape({static_cast<int64_t>(anchors.size()),3,16,72}).select(-1,0).sum(-1),generated.eligible_channels.to(torch::kInt64)),"saved fixed query hides exactly one of sixteen patches per eligible channel");
    check(torch::equal(hidden,generated.hidden)&&torch::equal(plan.target.narrow(0,offset,anchors.size()),generated.target)&&torch::equal(plan.eligible.narrow(0,offset,anchors.size()),generated.eligible_channels),"unchanged saved query recipe/support");
    const auto trace=model->forward_at(raw,torch::tensor(anchors,torch::kFloat64),hidden,scaler);++result.calls;
    const auto support=trace.target_mask,eligible=trace.nll.eligible_channels;
    check(torch::equal(support.cpu(),plan.target.narrow(0,offset,anchors.size()))&&torch::equal(eligible.cpu(),plan.eligible.narrow(0,offset,anchors.size())),"actual query target/eligibility parity");
    const auto error=torch::where(support,trace.distribution.mean.to(torch::kFloat64)-trace.targets.to(torch::kFloat64),torch::zeros_like(trace.targets,trace.targets.options().dtype(torch::kFloat64))).abs();
    const auto per_channel=error.sum(std::vector<int64_t>{2,3})/trace.nll.target_counts.clamp_min(1).to(torch::kFloat64);
    const auto per_example=torch::where(eligible,per_channel,torch::zeros_like(per_channel)).sum(1)/eligible.sum(1).clamp_min(1).to(torch::kFloat64);
    const auto valid=trace.nll.eligible_examples;
    check(per_example.is_cuda()&&trace.nll.loss.is_cuda()&&trace.nll.loss.scalar_type()==torch::kFloat64&&torch::isfinite(per_example).all().item<bool>(),"finite actual CUDA standardized query MAE and separate NLL");
    result.mae+=per_example.masked_select(valid).sum().item<double>();result.nll+=trace.nll.example_loss.masked_select(valid).sum().item<double>();result.valid+=valid.sum().item<int64_t>();result.cells+=trace.nll.target_cell_count;
    maes.push_back(per_example.cpu());nlls.push_back(trace.nll.example_loss.cpu());counts.push_back(trace.nll.target_counts.sum(1).cpu());
    if(offset==0){torch::serialize::OutputArchive archive;archive.write("row_ids",torch::tensor(anchors,torch::kInt64),true);archive.write("raw",raw.data,true);archive.write("observed",raw.observed,true);archive.write("timestamps",raw.timestamps,true);archive.write("hidden",hidden,true);archive.write("visible",trace.visible.cpu(),true);archive.write("target_mask",support.cpu(),true);archive.write("targets",trace.targets.cpu(),true);archive.write("mean",trace.distribution.mean.cpu(),true);archive.write("variance",trace.distribution.variance.cpu(),true);archive.write("native",trace.output.encoding.z_contextual_global.cpu(),true);archive.write("row_mae",per_example.cpu(),true);archive.write("row_nll",trace.nll.example_loss.cpu(),true);publish(archive,witness);}
  }
  torch::cuda::synchronize();result.seconds=elapsed(start);check(result.valid==static_cast<int64_t>(rows.size()),"complete eligible query population");result.mae/=result.valid;result.nll/=result.valid;
  result.row_mae=torch::cat(maes);result.row_nll=torch::cat(nlls);result.row_cells=torch::cat(counts);return result;
}
std::string query_json(const QueryCapture &q){std::ostringstream out;out<<std::setprecision(17)<<"{\"standardized_fixed_query_MAE\":"<<q.mae<<",\"Gaussian_NLL_F64\":"<<q.nll<<",\"valid_rows\":"<<q.valid<<",\"target_cells\":"<<q.cells<<",\"CUDA_encoder_decoder_calls\":"<<q.calls<<",\"seconds_including_loader_witness_IO\":"<<q.seconds<<",\"row_mae\":"<<vector_json(q.row_mae)<<",\"row_nll\":"<<vector_json(q.row_nll)<<",\"row_target_cells\":"<<vector_json(q.row_cells)<<'}';return out.str();}
std::string category(const std::string &name){
  if(name.find("variance_projection")!=std::string::npos)return"variance_matrix";if(name.find("variance_bias")!=std::string::npos)return"variance_bias";
  if(name.find("change_projection")!=std::string::npos)return"change_projection";if(name.find("change_decoder")!=std::string::npos)return"change_wave_decoder";
  if(name.find("relation_projection")!=std::string::npos)return"parent_content_relation";
  if(name.find("decoder")!=std::string::npos)return"parent_decoder";if(name.find("global")!=std::string::npos)return"parent_global_pool";
  return"parent_backbone_projection";
}
struct ParameterEvidence{int64_t values{0},gradient_steps{0},moved_steps{0};bool trainable{false};torch::Tensor initial;};
using Evidence=std::map<std::string,ParameterEvidence>;
Evidence initial_evidence(const cp::Encoder &model){Evidence out;for(const auto &p:model->named_parameters()){check(p.value().is_cuda()&&p.value().scalar_type()==torch::kFloat32,"actual CUDA F32 parameter");out.emplace(p.key(),ParameterEvidence{p.value().numel(),0,0,p.value().requires_grad(),p.value().detach().cpu().clone()});}return out;}
struct ResumeSnapshot {
  std::vector<SavedTensor> parameters,buffers,moments;std::vector<int64_t> steps;
  ResumeSnapshot(const cp::Encoder &model,const torch::optim::AdamW &optimizer){
    for(const auto &p:model->named_parameters()){parameters.push_back({p.key(),p.value().detach().cpu().clone()});const auto it=optimizer.state().find(p.value().unsafeGetTensorImpl());if(it==optimizer.state().end()){steps.push_back(-1);continue;}
      const auto *state=dynamic_cast<const torch::optim::AdamWParamState*>(it->second.get());check(state,"typed AdamW state");steps.push_back(state->step());moments.push_back({p.key()+".mean",state->exp_avg().detach().cpu().clone()});moments.push_back({p.key()+".variance",state->exp_avg_sq().detach().cpu().clone()});}
    for(const auto &p:model->named_buffers())buffers.push_back({p.key(),p.value().detach().cpu().clone()});
  }
  void compare(const ResumeSnapshot &other)const{check(steps==other.steps,"typed resume exact optimizer steps");const auto compare=[](const auto &a,const auto &b){check(a.size()==b.size(),"resume state inventory");for(size_t i=0;i<a.size();++i)check(a[i].name==b[i].name&&cp::boundary::same_bits(a[i].value,b[i].value),"typed resume exact values/moments/order");};compare(parameters,other.parameters);compare(buffers,other.buffers);compare(moments,other.moments);}
};
int run(int argc,char **argv){
  check(fs::exists("/.dockerenv")&&torch::cuda::is_available(),"existing CUDA container required");torch::set_num_threads(1);torch::set_num_interop_threads(1);Args args(argc,argv);
  const auto config=cp::read_settings(args.settings);check(config.level.device==torch::Device(torch::kCUDA,0)&&config.level.dropout==0&&config.level.mask_ratio==.25,"frozen CUDA0 zero-dropout mask25 geometry");
  std::vector<Pin> pins{{args.settings,cp::boundary::sha256_file(args.settings)},
    {args.shared/"selections.pt","10df626e83cc652edfdce2f0cb2eae72ccf802a3a0081cf5db8db71a7ef8bdb2"},
    {args.shared/"query-plan-train.pt","1853a271a32bd6635c122b36dfd6373247815178cff1525e95d852c5aafc137e"},
    {args.shared/"query-plan-validation.pt","5a98906368f3b56508d81b254d598594573545babd3c8e8ff67979a30b12018d"}};
  const std::array<const char*,3> cache_hashes{"67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58","0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa","364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7"};
  for(size_t i=0;i<3;++i)pins.push_back({args.caches[i],cache_hashes[i]});verify(pins);
  Selections selections(args.shared);QueryPlan query_train(args.shared,"train",selections.probe_train),query_validation(args.shared,"validation",selections.probe_validation);md::Dataset data(args.caches);
  check(fs::create_directory(args.output),"exclusive new fresh run leaf");fs::create_directory(args.output/"checkpoints");fs::create_directory(args.output/"witnesses");
  const std::string calibration_source="CPB-v4.market-01;fresh-TRAIN-only-CACHE88:67745d6a/0aad9d11/364fbf2c;selection10df626e;4096windows;recent128;IDs0,1,2;no-encoder-calibration";
  auto calibration=causal_fresh_v4::calibration::fit(data,config,selections.train,args.output/"calibration",calibration_source);
  for(const auto &name:{"input-scaler.pt","bank-standardizer.pt","first-TRAIN-batch-witness.pt","summary.json"}){const auto path=args.output/"calibration"/name;pins.push_back({path,cp::boundary::sha256_file(path)});}
  auto loaded=cp::fresh(config,calibration.scaler,calibration.bank,943001);auto model=loaded.model;auto scaler=loaded.scaler;
  check(loaded.source_updates==0&&loaded.causal_updates==0&&model->source_updates()==0&&model->initialization_seed()==943001,"fresh factory has zero inherited updates and declared seed");
  int64_t constructor_parameter_tensors=0,constructor_values=0,constructor_buffer_tensors=0,constructor_learned_values=0,constructor_fixed_values=0;std::string direct_state_identity;
  {
    torch::NoGradGuard guard;torch::manual_seed(943001);impl::LocationScaleDecoderModel direct(config,calibration.bank);
    const auto actual=model->core()->named_parameters(),reference=direct->named_parameters();check(actual.size()==reference.size(),"fresh constructor parameter inventory");
    for(const auto &p:reference){const auto *q=actual.find(p.key());check(q&&q->requires_grad()==p.value().requires_grad()&&cp::boundary::same_bits(*q,p.value()),"all initial tensors equal direct seeded constructor, no imported weights");++constructor_parameter_tensors;constructor_values+=p.value().numel();if(p.value().requires_grad())constructor_learned_values+=p.value().numel();else constructor_fixed_values+=p.value().numel();}
    const auto actual_buffers=model->core()->named_buffers(),reference_buffers=direct->named_buffers();check(actual_buffers.size()==reference_buffers.size(),"fresh constructor buffer inventory");
    for(const auto &p:reference_buffers){const auto *q=actual_buffers.find(p.key());check(q&&cp::boundary::same_bits(*q,p.value()),"fresh constructor buffer bytes");++constructor_buffer_tensors;}
    direct->validate_fixed_prior();direct_state_identity=cp::boundary::state_identity(direct);check(direct_state_identity==model->state_identity(),"full fresh seeded state identity");
    check(constructor_parameter_tensors==76&&constructor_values==297560&&constructor_learned_values==293672&&constructor_fixed_values==3888,"fresh learned/fixed parameter partition");
  }
  Frozen frozen(model,scaler);frozen.validate(model,scaler);
  std::ostringstream bindings;bindings<<"[";for(size_t i=0;i<pins.size();++i){if(i)bindings<<',';bindings<<"{\"path\":"<<quote(pins[i].path.string())<<",\"sha256\":"<<quote(pins[i].hash)<<'}';}bindings<<']';write(args.output/"input-bindings.json",bindings.str());
  const tr::TrainingIdentity identity{"MKT-HIST-2;CACHE88:67745d6a/0aad9d11/364fbf2c;genuine-F64;C3H128F9;IDs0,1,2",
      "selection-sha256:10df626e83cc652edfdce2f0cb2eae72ccf802a3a0081cf5db8db71a7ef8bdb2;same4096TRAIN112probeTRAIN32VAL;past-only",
      std::string("CPB-v4.market-01;fresh-constructor-seed943001;source-updates0;no-import;all-learned-causal;variance_input_gradient_stop-v1;encoder-partial-NLL-gradient-through-live-mean-residual;conditional-variance-head-live;new-TRAIN-scaler:")+scaler.identity()+";new-TRAIN-bank:"+model->normalizer().identity()+";fixed-literal-prior;numeric-Gaussian-NLL-F64-unchanged;not-full-joint-NLL-gradient;B8;2048;mask25/context15;AdamW.001/.0001;clip1;row943002/mask943003/context943004;fixed-saved-query-masks-no-query-RNG;no-label-loss"};
  tr::TrainingProgress progress{0,0,row_seed,mask_seed,context_seed};auto optimizer=tr::make_optimizer(model);tr::validate_optimizer(model,*optimizer,progress);Evidence evidence=initial_evidence(model);
  double checkpoint_seconds=0,loader_mask_seconds=0,compute_seconds=0,evidence_seconds=0;auto io_start=Clock::now();model->eval();model->clear_gradients();const auto initial_checkpoint=args.output/"checkpoints/inference-0.pt";cp::save_checkpoint(initial_checkpoint.string(),model,scaler,0);checkpoint_seconds+=elapsed(io_start);
  std::ostringstream initialization;initialization<<"{\"kind\":\"cpb_v4_fresh_initialization_v1\",\"status\":\"passed\",\"initialization_kind\":"<<quote(cp::kInitializationKind)<<",\"initialization_seed\":943001,\"pretrained_ancestor\":\"none\",\"inherited_learned_parameters\":0,\"fresh_learned_parameter_values\":"<<constructor_learned_values<<",\"analytic_fixed_prior_values\":"<<constructor_fixed_values<<",\"source_updates\":0,\"causal_updates\":0,\"old_model_artifacts_opened\":0,\"old_normalization_artifacts_opened\":0,\"all_initial_parameters_equal_direct_seeded_constructor\":true,\"parameter_tensors_checked\":"<<constructor_parameter_tensors<<",\"parameter_values_checked\":"<<constructor_values<<",\"buffer_tensors_checked\":"<<constructor_buffer_tensors<<",\"direct_constructor_state_identity\":"<<quote(direct_state_identity)<<",\"initial_state_identity\":"<<quote(model->initial_state_identity())<<",\"fresh_input_scaler_identity\":"<<quote(scaler.identity())<<",\"fresh_bank_identity\":"<<quote(model->normalizer().identity())<<",\"initial_checkpoint\":{\"path\":"<<quote(initial_checkpoint.string())<<",\"bytes\":"<<fs::file_size(initial_checkpoint)<<",\"sha256\":"<<quote(cp::boundary::sha256_file(initial_checkpoint))<<"}}";write(args.output/"fresh-initialization.json",initialization.str());
  const auto initial_train=capture(data,model,scaler,selections.probe_train,query_train,md::train_begin,md::train_end,args.output/"witnesses/initial-train-query.pt");
  const auto initial_validation=capture(data,model,scaler,selections.probe_validation,query_validation,md::validation_begin,md::validation_end,args.output/"witnesses/initial-validation-query.pt");
  write(args.output/"query-initial.json","{\"train\":"+query_json(initial_train)+",\"validation\":"+query_json(initial_validation)+"}");
  std::ofstream log(args.output/"training.csv");check(bool(log),"new training log");log<<"attempt,update,Gaussian_NLL_F64,eligible_examples,target_cells,nonzero_gradient_tensors,changed_tensors,row_seed,mask_seed,anchors\n"<<std::setprecision(17);
  model->train();const auto trajectory_start=Clock::now();bool resume_verified=false;int64_t trained_cells=0;double first_loss=0,last_loss=0;
  while(progress.completed_updates<updates&&progress.attempted_batches<max_attempts){
    auto stage=Clock::now();const auto attempt=progress.attempted_batches,row_stream=tr::training_counter_seed(row_seed,attempt),mask_stream=tr::training_counter_seed(mask_seed,attempt);
    std::mt19937_64 random(row_stream);std::uniform_int_distribution<int64_t> pick(0,selections.train.size()-1);std::vector<int64_t> rows;for(int64_t i=0;i<batch;++i)rows.push_back(selections.train[pick(random)]);
    const auto raw=data.load(rows,md::train_begin,md::train_end);const auto masks=impl::configurable_spectral_make_training_mask(raw.observed,config.level,mask_stream);
    if(!masks.eligible_channels.any().item<bool>()){++progress.attempted_batches;loader_mask_seconds+=elapsed(stage);log<<attempt<<','<<progress.completed_updates<<",,0,0,0,0,"<<row_stream<<','<<mask_stream<<",\"\"\n";continue;}
    const auto context=impl::context_deletion::make_plan(masks,raw.channel_ids,config.level,context_seed,attempt,impl::ContextDeletionRecipe::coordinate15_v1);loader_mask_seconds+=elapsed(stage);
    stage=Clock::now();std::vector<SavedTensor> before;for(const auto &p:model->named_parameters())if(p.value().requires_grad())before.push_back({p.key(),p.value().detach().clone()});torch::cuda::synchronize();evidence_seconds+=elapsed(stage);
    optimizer->zero_grad(true);torch::cuda::synchronize();stage=Clock::now();
    const auto trace=model->forward_at(raw,torch::tensor(rows,torch::kFloat64),masks.hidden,scaler,context.deleted);
    check(trace.nll.loss.is_cuda()&&trace.nll.loss.scalar_type()==torch::kFloat64&&torch::isfinite(trace.nll.loss).item<bool>(),"finite actual CUDA F64 Gaussian loss");trace.nll.loss.backward();
    for(const auto &p:model->named_parameters())if(p.value().grad().defined())check(p.value().requires_grad()&&p.value().grad().is_cuda()&&torch::isfinite(p.value().grad()).all().item<bool>(),"finite learned CUDA gradients before optimizer mutation");
    torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer->step();torch::cuda::synchronize();compute_seconds+=elapsed(stage);
    stage=Clock::now();check(torch::equal(trace.visible.cpu(),context.visible)&&torch::equal(trace.target_mask.cpu(),masks.target)&&!(trace.visible&trace.target_mask).any().item<bool>(),"target/context separation with exact final mask");
    int64_t gradient_tensors=0,changed_tensors=0;const auto named=model->named_parameters();
    for(const auto &p:named){const auto gradient=p.value().grad();if(gradient.defined()){check(p.value().requires_grad()&&gradient.is_cuda()&&torch::isfinite(gradient).all().item<bool>(),"finite actual CUDA gradient");if(gradient.abs().sum().item<double>()>0){++evidence.at(p.key()).gradient_steps;++gradient_tensors;}}else check(!p.value().requires_grad()||p.value().is_cuda(),"CUDA inactive parameter");}
    for(const auto &p:before)if(!torch::equal(p.value,named[p.name].detach())){++evidence.at(p.name).moved_steps;++changed_tensors;}
    ++progress.completed_updates;++progress.attempted_batches;model->set_causal_updates(progress.completed_updates);
    check(model->causal_updates()==progress.completed_updates,"live causal update counter matches completed CUDA steps");
    last_loss=trace.nll.loss.item<double>();if(progress.completed_updates==1)first_loss=last_loss;trained_cells+=trace.nll.target_cell_count;
    check(gradient_tensors>0&&changed_tensors>0,"actual CUDA learning at every successful update");evidence_seconds+=elapsed(stage);
    stage=Clock::now();log<<attempt<<','<<progress.completed_updates<<','<<last_loss<<','<<trace.nll.eligible_example_count<<','<<trace.nll.target_cell_count<<','<<gradient_tensors<<','<<changed_tensors<<','<<row_stream<<','<<mask_stream<<",\"";
    for(size_t i=0;i<rows.size();++i){if(i)log<<';';log<<rows[i];}log<<"\"\n";checkpoint_seconds+=elapsed(stage);
    optimizer->zero_grad(true);
    if(progress.completed_updates%128==0){frozen.validate(model,scaler);tr::validate_optimizer(model,*optimizer,progress);std::cout<<"update="<<progress.completed_updates<<" attempts="<<progress.attempted_batches<<" loss="<<last_loss<<std::endl;}
    if(progress.completed_updates==512||progress.completed_updates==1024||progress.completed_updates==2048){
      stage=Clock::now();check(model->causal_updates()==progress.completed_updates,"live counter agrees before typed training save");log.flush();const auto path=args.output/"checkpoints"/("training-"+std::to_string(progress.completed_updates)+".pt");tr::save_training_state(path.string(),model,scaler,*optimizer,identity,progress);
      if(progress.completed_updates==512){const ResumeSnapshot original(model,*optimizer);auto resumed=tr::load_training_state(path.string(),config,identity);original.compare(ResumeSnapshot(resumed.model,*resumed.optimizer));
        check(resumed.progress.completed_updates==progress.completed_updates&&resumed.progress.attempted_batches==progress.attempted_batches&&resumed.progress.row_seed==row_seed&&resumed.progress.mask_seed==mask_seed&&resumed.progress.context_seed==context_seed,"exact typed resume counters/seed streams");
        model=resumed.model;scaler=resumed.scaler;optimizer=std::move(resumed.optimizer);progress=resumed.progress;check(model->causal_updates()==progress.completed_updates,"resumed live counter agrees with saved progress");model->train();frozen.validate(model,scaler);resume_verified=true;
        write(args.output/"typed-resume-512.json","{\"status\":\"passed\",\"completed_updates\":512,\"same_parameters_buffers_moments_steps_order\":true,\"same_attempt_counter_and_three_seeds\":true,\"continued_actual_trajectory_from_saved_state\":true,\"repeated_training_updates\":0}");}
      checkpoint_seconds+=elapsed(stage);
    }
  }
  torch::cuda::synchronize();const auto trajectory_seconds=elapsed(trajectory_start);log.close();check(bool(log),"persist training log");check(progress.completed_updates==2048&&resume_verified,"complete one declared causal trajectory with actual typed resume");optimizer->zero_grad(true);frozen.validate(model,scaler);tr::validate_optimizer(model,*optimizer,progress);
  check(model->causal_updates()==progress.completed_updates&&progress.completed_updates==2048,"final live counter agrees before inference save and query evidence");
  io_start=Clock::now();model->eval();model->clear_gradients();cp::save_checkpoint((args.output/"checkpoints/inference-2048.pt").string(),model,scaler,2048);checkpoint_seconds+=elapsed(io_start);
  const auto final_train=capture(data,model,scaler,selections.probe_train,query_train,md::train_begin,md::train_end,args.output/"witnesses/final-train-query.pt");
  const auto final_validation=capture(data,model,scaler,selections.probe_validation,query_validation,md::validation_begin,md::validation_end,args.output/"witnesses/final-validation-query.pt");
  check(model->causal_updates()==progress.completed_updates,"final query evidence retains exact live update counter");write(args.output/"query-final.json","{\"train\":"+query_json(final_train)+",\"validation\":"+query_json(final_validation)+"}");frozen.validate(model,scaler);verify(pins);
  struct Group{int64_t tensors{0},values{0},gradient_tensors{0},moved_tensors{0},changed_final_tensors{0};};std::map<std::string,Group> groups;int64_t total_values=0,updated_values=0,fixed_values=0;std::ostringstream parameter_json;parameter_json<<'[';bool first=true;
  for(const auto &[name,e]:evidence){const auto current=model->named_parameters()[name];const bool changed=!cp::boundary::same_bits(current,e.initial);total_values+=e.values;if(e.trainable){auto &g=groups[category(name)];++g.tensors;g.values+=e.values;g.gradient_tensors+=e.gradient_steps>0;g.moved_tensors+=e.moved_steps>0;g.changed_final_tensors+=changed;if(e.moved_steps)updated_values+=e.values;}else{fixed_values+=e.values;check(!changed&&e.gradient_steps==0&&e.moved_steps==0,"fixed prior never learns or drifts");}
    if(!first)parameter_json<<',';first=false;parameter_json<<"{\"name\":"<<quote(name)<<",\"group\":"<<quote(category(name))<<",\"values\":"<<e.values<<",\"requires_grad\":"<<(e.trainable?"true":"false")<<",\"nonzero_CUDA_gradient_steps\":"<<e.gradient_steps<<",\"weight_movement_steps\":"<<e.moved_steps<<",\"changed_from_initial\":"<<(changed?"true":"false")<<'}';}
  parameter_json<<']';write(args.output/"parameter-learning.json",parameter_json.str());
  std::ostringstream group_json;group_json<<'[';first=true;for(const auto &[name,g]:groups){if(!first)group_json<<',';first=false;group_json<<"{\"group\":"<<quote(name)<<",\"tensors\":"<<g.tensors<<",\"values\":"<<g.values<<",\"nonzero_gradient_tensors\":"<<g.gradient_tensors<<",\"moved_tensors\":"<<g.moved_tensors<<",\"changed_final_tensors\":"<<g.changed_final_tensors<<'}';}
  group_json<<']';
  for(const auto &required:{"parent_backbone_projection","parent_content_relation","parent_decoder","parent_global_pool","change_projection","change_wave_decoder","variance_matrix","variance_bias"})check(groups.count(required)&&groups.at(required).gradient_tensors>0&&groups.at(required).moved_tensors>0,"required model category actually learns on CUDA: "+std::string(required));
  std::ostringstream summary;summary<<std::setprecision(17)<<"{\"kind\":\"causal_fresh_v4_training_summary_v1\",\"status\":\"passed\",\"model\":\"CPB-v4.market-01\",\"gradient_policy\":"<<quote(cp::kGradientPolicy)<<",\"encoder_gradient_kind\":\"partial-NLL-gradient;live-mean-residual;variance-input-stopped\",\"conditional_variance_head_trainable\":true,\"initialization_kind\":\"fresh-seeded-constructor-no-import-v1\",\"initialization_seed\":943001,\"source_updates\":0,\"new_causal_updates\":"<<progress.completed_updates<<",\"attempted_batches\":"<<progress.attempted_batches<<",\"batch_size\":8,\"TRAIN_windows\":4096,\"probe_TRAIN_windows\":112,\"probe_VALIDATION_windows\":32,\"native_width\":36,\"device\":\"cuda:0\",\"parameters\":"<<total_values<<",\"parameters_with_recorded_updates\":"<<updated_values<<",\"fixed_prior_parameters\":"<<fixed_values<<",\"parameter_groups\":"<<group_json.str()<<",\"first_training_NLL\":"<<first_loss<<",\"last_training_NLL\":"<<last_loss<<",\"training_target_cells\":"<<trained_cells
    <<",\"GPU_step_seconds_synchronized\":"<<compute_seconds<<",\"GPU_step_timing_scope\":\"forward_at+F64_NLL+backward+clip+AdamW;synchronized_wall;includes_in_step_contract_checks;excludes_loader_masks_checkpoints_evidence_logging\",\"loader_mask_seconds\":"<<loader_mask_seconds<<",\"checkpoint_resume_logging_seconds\":"<<checkpoint_seconds<<",\"parameter_gradient_evidence_seconds\":"<<evidence_seconds<<",\"training_trajectory_wall_seconds\":"<<trajectory_seconds
    <<",\"fixed_query_initial\":{\"train\":"<<query_json(initial_train)<<",\"validation\":"<<query_json(initial_validation)<<"},\"fixed_query_final\":{\"train\":"<<query_json(final_train)<<",\"validation\":"<<query_json(final_validation)<<"},\"raw_load_calls\":"<<data.calls<<",\"direct_F64_record_cells_checked\":"<<data.record_cells<<",\"genuine_F64_differs_from_F32_cells\":"<<data.genuine_f64_cells<<",\"typed_resume_512_exact\":true,\"fixed_prior_scaler_normalizer_unchanged\":true,\"new_scaler_fits\":1,\"new_normalizer_fits\":1,\"calibration_encoder_calls\":0,\"calibration_validation_rows\":0,\"calibration_seconds\":"<<calibration.seconds<<",\"calibration_TRAIN_loader_calls\":"<<calibration.loader_calls<<",\"fresh_scaler_identity\":"<<quote(scaler.identity())<<",\"fresh_bank_identity\":"<<quote(model->normalizer().identity())<<",\"old_model_artifacts_opened\":0,\"old_normalization_artifacts_opened\":0,\"saved_quality_features_reused\":false,\"fixed_query_source\":\"unchanged saved masks; source seed942005; query943005 not consumed\",\"CPU_encoder_calls\":0,\"old_encoder_calls\":0,\"head_fits\":0,\"training_replays\":0}";
  write(args.output/"training-summary.json",summary.str());std::cout<<"CPB_V4_TRAINING="<<args.output<<'\n';return 0;
}
} // namespace
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
