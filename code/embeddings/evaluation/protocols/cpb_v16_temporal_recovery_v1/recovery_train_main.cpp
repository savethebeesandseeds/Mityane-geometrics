// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Operational recovery of the interrupted candidate946103 trajectory, logical513..2048.
// The original runner, model, recipe and partial outputs remain immutable.
// Physical work after the last saved512 state is repeated; its exact count is unknown.
#ifndef CPB_OBJECTIVE_CANDIDATE
#error "This recovery source must be built only for the V16 candidate"
#endif
#include "../cpb_v16_temporal_v1/include/model_binding.h"
#include "../cpb_v16_temporal_v1/include/market_data.h"
#include "../cpb_v16_temporal_v1/include/context_deletion.h"
#include <torch/cuda.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <unistd.h>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::CPB_OBJECTIVE_VERSION;
namespace impl=cp::implementation;
namespace tr=cp::training;
namespace md=cpb_v16_temporal::market;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
constexpr int64_t batch=8,updates=2048,max_attempts=4096,resume_update=512;
void check(bool yes,const std::string &why){TORCH_CHECK(yes,"[CPB paired-objective train] ",why);}
double elapsed(const Clock::time_point &start){return std::chrono::duration<double>(Clock::now()-start).count();}
std::string quote(const std::string &s){std::ostringstream out;out<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}return out.str()+'"';}
void write(const fs::path &path,const std::string &contents){check(!fs::exists(path),"preserve existing output: "+path.string());std::ofstream out(path,std::ios::binary);check(bool(out),"create output");out<<contents<<'\n';out.close();check(bool(out),"save output");}
void publish(torch::serialize::OutputArchive &archive,const fs::path &path){check(!fs::exists(path),"exclusive archive");archive.save_to(path.string());}
std::string vector_json(const torch::Tensor &tensor){const auto x=tensor.detach().cpu().to(torch::kFloat64).flatten().contiguous();std::ostringstream out;out<<std::setprecision(17)<<'[';for(int64_t i=0;i<x.numel();++i){if(i)out<<',';out<<x[i].item<double>();}return out.str()+']';}
struct Pin{fs::path path;std::string hash;};
void verify(const std::vector<Pin> &pins){for(const auto &p:pins)check(cp::boundary::sha256_file(p.path)==p.hash,"frozen input/source changed: "+p.path.string());}
struct Args {
  fs::path settings,shared,output,resume_state,source_output;std::vector<fs::path> caches;int64_t seed{-1};bool validate_only{false};
  Args(int argc,char **argv){for(int i=1;i<argc;i+=2){check(i+1<argc,"flag/value pairs required");const std::string flag=argv[i];const fs::path value=argv[i+1];
      if(flag=="--cache")caches.push_back(value);
      else if(flag=="--settings"){check(settings.empty(),"duplicate settings");settings=value;}
      else if(flag=="--shared"){check(shared.empty(),"duplicate shared");shared=value;}
      else if(flag=="--output"){check(output.empty(),"duplicate output");output=value;}
      else if(flag=="--resume-state"){check(resume_state.empty(),"duplicate resume state");resume_state=value;}
      else if(flag=="--source-output"){check(source_output.empty(),"duplicate source output");source_output=value;}
      else if(flag=="--validate-only"){check(!validate_only&&value=="true","one --validate-only true allowed");validate_only=true;}
      else if(flag=="--seed"){check(seed<0&&value.string().find_first_not_of("0123456789")==std::string::npos,"one decimal seed");seed=std::stoll(value.string());}
      else check(false,"unknown argument: "+flag);}
    check(seed==946103&&!settings.empty()&&!shared.empty()&&!output.empty()&&!resume_state.empty()&&!source_output.empty()&&caches.size()==3,"candidate946103 --settings --shared --cache BTC --cache ETH --cache SOL --resume-state ORIGINAL512 --source-output ORIGINALTRAINING --output NEWLEAF required");
    settings=fs::canonical(settings);shared=fs::canonical(shared);output=fs::absolute(output).lexically_normal();
    resume_state=fs::canonical(resume_state);source_output=fs::canonical(source_output);
    const auto required_source=fs::canonical("/mityane/.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/training");
    check(source_output==required_source&&resume_state==source_output/"checkpoints/training-512.pt","only the retained original candidate946103 typed512 state is admitted");
    for(auto &p:caches)p=fs::canonical(p);check(!fs::exists(output)&&fs::exists(output.parent_path()),"exclusive leaf in existing attempt required");
    const auto canonical_parent=fs::canonical(output.parent_path());check(canonical_parent==output.parent_path(),"canonical output parent required");
    const auto rel=output.lexically_relative(source_output);check(rel.empty()||*rel.begin()=="..","recovery output must not be inside the preserved original leaf");
  }
};
std::string read_text(const fs::path &path){std::ifstream in(path,std::ios::binary);check(bool(in),"read retained metadata "+path.string());std::ostringstream out;out<<in.rdbuf();check(in.good()||in.eof(),"complete retained metadata read");return out.str();}
// These flat fields are from the producer's exact JSON serialization. No tensor
// payload is read here; full input hashes are independently admitted by the bridge.
std::string flat_token(const std::string &json,const std::string &key){
  const auto marker=quote(key)+":";const auto at=json.find(marker);check(at!=std::string::npos&&json.find(marker,at+marker.size())==std::string::npos,"one retained metadata field: "+key);
  const auto begin=at+marker.size();check(begin<json.size(),"retained metadata value exists");
  if(json[begin]=='"'){const auto end=json.find('"',begin+1);check(end!=std::string::npos,"closed retained metadata string");const auto value=json.substr(begin+1,end-begin-1);check(value.find('\\')==std::string::npos&&value.find_first_of("\r\n\t")==std::string::npos,"unescaped retained identity string");return value;}
  const auto end=json.find_first_of(",}\r\n",begin);check(end!=std::string::npos,"terminated retained metadata token");return json.substr(begin,end-begin);
}
int64_t decimal(const std::string &value){check(!value.empty()&&value.find_first_not_of("0123456789")==std::string::npos,"unsigned decimal integer required");size_t used=0;const auto result=std::stoll(value,&used);check(used==value.size(),"complete decimal integer");return result;}
uint64_t unsigned_decimal(const std::string &value){check(!value.empty()&&value.find_first_not_of("0123456789")==std::string::npos,"unsigned decimal stream without sign required");size_t used=0;const auto result=std::stoull(value,&used,10);check(used==value.size()&&result<=std::numeric_limits<uint64_t>::max(),"complete bounded uint64 stream required");return static_cast<uint64_t>(result);}
double finite_number(const std::string &value){size_t used=0;const auto result=std::stod(value,&used);check(used==value.size()&&std::isfinite(result),"complete finite scalar required");return result;}
std::vector<std::string> csv_row(const std::string &line){std::vector<std::string> cells;std::string cell;bool quoted=false;for(char ch:line){if(ch=='"')quoted=!quoted;else if(ch==','&&!quoted){cells.push_back(cell);cell.clear();}else cell+=ch;}check(!quoted,"complete prefix CSV quote");cells.push_back(cell);return cells;}
struct Prefix {double first_loss{0};int64_t target_cells{0};};
Prefix read_prefix(const fs::path &path,int64_t row_seed,int64_t mask_seed){
  std::ifstream in(path);check(bool(in),"retained prefix CSV");std::string line;check(bool(std::getline(in,line)),"prefix CSV header");if(!line.empty()&&line.back()=='\r')line.pop_back();
  check(line=="attempt,update,training_objective_F64,eligible_examples,target_cells,nonzero_gradient_tensors,changed_tensors,row_seed,mask_seed,anchors","original prefix CSV schema");Prefix prefix;
  for(int64_t update=1;update<=resume_update;++update){check(bool(std::getline(in,line)),"all512 original complete prefix rows required");if(!line.empty()&&line.back()=='\r')line.pop_back();const auto cells=csv_row(line);check(cells.size()==10,"original prefix ten columns");const auto attempt=decimal(cells[0]);check(attempt==update-1&&decimal(cells[1])==update,"original512 prefix has no skipped attempts");
    const auto loss=finite_number(cells[2]);if(update==1)prefix.first_loss=loss;check(decimal(cells[3])==batch&&decimal(cells[4])>0&&decimal(cells[5])>0&&decimal(cells[6])>0,"original prefix eligible CUDA learning evidence");prefix.target_cells+=decimal(cells[4]);
    check(unsigned_decimal(cells[7])==static_cast<uint64_t>(tr::training_counter_seed(row_seed,attempt))&&unsigned_decimal(cells[8])==static_cast<uint64_t>(tr::training_counter_seed(mask_seed,attempt)),"original prefix counter streams");
  }
  return prefix; // Original post512 rows do not enter logical-prefix aggregates.
}
std::map<int64_t,std::string> complete_rows(const fs::path &path,size_t columns){
  std::ifstream in(path);check(bool(in),"retained replay CSV");std::string line;check(bool(std::getline(in,line)),"retained replay header");std::map<int64_t,std::string> result;
  while(std::getline(in,line)){if(in.eof())break; // An unterminated last row is preserved, not treated as complete.
    if(!line.empty()&&line.back()=='\r')line.pop_back();const auto cells=csv_row(line);check(cells.size()==columns,"complete original replay CSV schema");const auto update=decimal(cells[1]);check(result.emplace(update,line).second,"one original row per successful update");}
  return result;
}
void compare_replay(const std::map<int64_t,std::string> &original,int64_t update,const std::string &row,const char *scope){const auto old=original.find(update);if(old!=original.end())check(old->second==row,std::string("bit-exact textual replay mismatch; halt without rescue: ")+scope+" update"+std::to_string(update));}
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
#ifdef CPB_OBJECTIVE_CANDIDATE
    check(trace.objective.temporal.lags.size()==3&&trace.objective.temporal.lags[2].lag==16&&trace.objective.temporal.lags[2].pair_count==0,"unchanged one-patch fixed query has unsupported lag16; no target expansion");
#endif
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
std::string query_json(const QueryCapture &q){std::ostringstream out;out<<std::setprecision(17)<<"{\"standardized_fixed_query_MAE\":"<<q.mae<<",\"Gaussian_NLL_F64\":"<<q.nll<<",\"valid_rows\":"<<q.valid<<",\"target_cells\":"<<q.cells<<",\"CUDA_encoder_decoder_calls\":"<<q.calls<<",\"seconds_including_loader_witness_IO\":"<<q.seconds<<",\"row_mae\":"<<vector_json(q.row_mae)<<",\"row_nll\":"<<vector_json(q.row_nll)<<",\"row_target_cells\":"<<vector_json(q.row_cells);
#ifdef CPB_OBJECTIVE_CANDIDATE
  out<<",\"temporal_lag16\":{\"pair_count\":0,\"supported\":false,\"loss\":null,\"reason\":\"unchanged single hidden8-step patch\"}";
#endif
  out<<'}';return out.str();}
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
  const int64_t seed=args.seed,row_seed=seed+1000,mask_seed=seed+2000,context_seed=seed+3000;
  const auto config=cp::read_settings(args.settings);check(config.level.device==torch::Device(torch::kCUDA,0)&&config.level.dropout==0&&config.level.mask_ratio==.25,"frozen CUDA0 zero-dropout mask25 geometry");
  std::vector<Pin> pins{{args.settings,cp::boundary::sha256_file(args.settings)},
    {args.shared/"selections.pt","10df626e83cc652edfdce2f0cb2eae72ccf802a3a0081cf5db8db71a7ef8bdb2"},
    {args.shared/"query-plan-train.pt","1853a271a32bd6635c122b36dfd6373247815178cff1525e95d852c5aafc137e"},
    {args.shared/"query-plan-validation.pt","5a98906368f3b56508d81b254d598594573545babd3c8e8ff67979a30b12018d"}};
  const std::array<const char*,3> cache_hashes{"67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58","0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa","364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7"};
  for(size_t i=0;i<3;++i)pins.push_back({args.caches[i],cache_hashes[i]});verify(pins);
  for(const auto &name:{"fresh-initialization.json","input-bindings.json","query-initial.json","training.csv","temporal-training.csv","typed-resume-512.json","calibration/input-scaler.pt","calibration/bank-standardizer.pt","calibration/first-TRAIN-batch-witness.pt","calibration/summary.json","checkpoints/inference-0.pt","checkpoints/training-512.pt","witnesses/initial-train-query.pt","witnesses/initial-validation-query.pt","witnesses/objective-update-1.pt","witnesses/temporal-objective-step-1.pt"}){
    const auto path=args.source_output/name;check(fs::is_regular_file(path),"all original partial artifacts retained");pins.push_back({path,cp::boundary::sha256_file(path)});
  }
  const auto calibration_metadata=read_text(args.source_output/"calibration/summary.json"),initialization_metadata=read_text(args.source_output/"fresh-initialization.json"),original_resume_metadata=read_text(args.source_output/"typed-resume-512.json");
  check(flat_token(calibration_metadata,"status")=="passed"&&decimal(flat_token(calibration_metadata,"input_scaler_fits"))==1&&decimal(flat_token(calibration_metadata,"bank_normalizer_fits"))==1&&decimal(flat_token(calibration_metadata,"validation_rows_used"))==0&&decimal(flat_token(calibration_metadata,"encoder_calls"))==0,"retained original TRAIN-only calibration proof");
  check(flat_token(initialization_metadata,"status")=="passed"&&decimal(flat_token(initialization_metadata,"initialization_seed"))==seed&&decimal(flat_token(initialization_metadata,"source_updates"))==0&&flat_token(initialization_metadata,"pretrained_ancestor")=="none"&&flat_token(initialization_metadata,"all_initial_parameters_equal_direct_seeded_constructor")=="true","retained original fresh constructor proof");
  check(flat_token(original_resume_metadata,"status")=="passed"&&decimal(flat_token(original_resume_metadata,"completed_updates"))==resume_update&&flat_token(original_resume_metadata,"same_parameters_buffers_moments_steps_order")=="true"&&flat_token(original_resume_metadata,"same_attempt_counter_and_three_seeds")=="true","retained original exact typed512 proof");
  const auto scaler_id=flat_token(calibration_metadata,"input_scaler_identity"),bank_id=flat_token(calibration_metadata,"bank_identity");
  check(flat_token(initialization_metadata,"fresh_input_scaler_identity")==scaler_id&&flat_token(initialization_metadata,"fresh_bank_identity")==bank_id,"original initialization/calibration identities agree");
  const tr::TrainingIdentity identity{"MKT-HIST-2;CACHE88:67745d6a/0aad9d11/364fbf2c;genuine-F64;C3H128F9;IDs0,1,2",
      "selection-sha256:10df626e83cc652edfdce2f0cb2eae72ccf802a3a0081cf5db8db71a7ef8bdb2;same4096TRAIN112probeTRAIN32VAL;past-only",
      std::string("CPB-v16-temporal-paired;fresh-constructor-runtime-seed;source-updates0;no-import;all-learned-causal;variance_input_gradient_stop-v1;branch-policy-bound-by-compiled-model;conditional-variance-head-live;new-TRAIN-scaler:")+scaler_id+";new-TRAIN-bank:"+bank_id+";fixed-literal-prior;separate-Gaussian-reporting-NLL;train-output-loss;B8;2048;mask25/context15;AdamW.001/.0001;clip1;runtime-init-plus1000/2000/3000-paired-streams;fixed-saved-query-masks-no-query-RNG;no-label-loss"};
  const auto prefix=read_prefix(args.source_output/"training.csv",row_seed,mask_seed);
  const auto original_main_rows=complete_rows(args.source_output/"training.csv",10),original_temporal_rows=complete_rows(args.source_output/"temporal-training.csv",15);
  check(!original_main_rows.empty()&&original_main_rows.rbegin()->first==553&&!original_temporal_rows.empty()&&original_temporal_rows.rbegin()->first==565,"exact retained interruption boundaries553/565; preserve incomplete tails");
  auto initial_query_json=read_text(args.source_output/"query-initial.json");const auto first_json=initial_query_json.find_first_not_of(" \r\n\t"),last_json=initial_query_json.find_last_not_of(" \r\n\t");check(first_json!=std::string::npos&&initial_query_json[first_json]=='{'&&initial_query_json[last_json]=='}',"retained original initial query JSON object");initial_query_json=initial_query_json.substr(first_json,last_json-first_json+1);
  check(fs::create_directory(args.output),"exclusive new recovery output leaf");
  const auto state_load_started=Clock::now();auto loaded=tr::load_training_state(args.resume_state.string(),config,identity);auto model=loaded.model;auto scaler=loaded.scaler;auto optimizer=std::move(loaded.optimizer);auto progress=loaded.progress;
  check(progress.completed_updates==resume_update&&progress.attempted_batches==resume_update&&progress.row_seed==row_seed&&progress.mask_seed==mask_seed&&progress.context_seed==context_seed&&model->causal_updates()==resume_update,"restore exact512 counter and three absolute streams");
  check(model->source_updates()==0&&model->initialization_seed()==seed&&model->initial_state_identity()==flat_token(initialization_metadata,"initial_state_identity")&&scaler.identity()==scaler_id&&model->normalizer().identity()==bank_id,"restore original fresh trajectory model/scaler/bank identity");
  Frozen frozen(model,scaler);frozen.validate(model,scaler);tr::validate_optimizer(model,*optimizer,progress);
  std::ostringstream bindings;bindings<<"{\"kind\":\"cpb_v16_operational_recovery_bindings_v1\",\"status\":\"passed\",\"mode\":"<<quote(args.validate_only?"validate-only":"suffix-training")<<",\"source_output\":"<<quote(args.source_output.string())<<",\"resume_state\":"<<quote(args.resume_state.string())<<",\"resume_checkpoint_update\":512,\"resumed_attempt_counter\":512,\"row_seed\":"<<row_seed<<",\"mask_seed\":"<<mask_seed<<",\"context_seed\":"<<context_seed<<",\"model_state_identity_at512\":"<<quote(model->state_identity())<<",\"optimizer_state_identity_at512\":"<<quote(tr::optimizer_identity(model,*optimizer))<<",\"inputs\":[";
  for(size_t i=0;i<pins.size();++i){if(i)bindings<<',';bindings<<"{\"path\":"<<quote(pins[i].path.string())<<",\"bytes\":"<<fs::file_size(pins[i].path)<<",\"sha256\":"<<quote(pins[i].hash)<<'}';}bindings<<"]}";write(args.output/"recovery-bindings.json",bindings.str());
  if(args.validate_only){
    const ResumeSnapshot before(model,*optimizer);const auto roundtrip_path=args.output/"validation-training-512.pt";tr::save_training_state(roundtrip_path.string(),model,scaler,*optimizer,identity,progress);auto roundtrip=tr::load_training_state(roundtrip_path.string(),config,identity);before.compare(ResumeSnapshot(roundtrip.model,*roundtrip.optimizer));frozen.validate(roundtrip.model,roundtrip.scaler);
    check(roundtrip.progress.completed_updates==progress.completed_updates&&roundtrip.progress.attempted_batches==progress.attempted_batches&&roundtrip.progress.row_seed==row_seed&&roundtrip.progress.mask_seed==mask_seed&&roundtrip.progress.context_seed==context_seed,"roundtrip exact counters/seed streams");verify(pins);
    write(args.output/"validation.json","{\"kind\":\"cpb_v16_operational_recovery_validation_v1\",\"status\":\"passed\",\"passed\":true,\"restored_update\":512,\"restored_attempts\":512,\"parameters_buffers_moments_steps_order_bitwise_equal\":true,\"scaler_bank_and_three_streams_equal\":true,\"encoder_forwards\":0,\"optimizer_steps\":0,\"scientific_recipe_changed\":false}");std::cout<<"CPB_V16_RECOVERY_VALIDATION="<<args.output<<'\n';return 0;
  }
  fs::create_directory(args.output/"checkpoints");fs::create_directory(args.output/"witnesses");
  Selections selections(args.shared);QueryPlan query_train(args.shared,"train",selections.probe_train),query_validation(args.shared,"validation",selections.probe_validation);md::Dataset data(args.caches);
  Evidence evidence=initial_evidence(model);double checkpoint_seconds=elapsed(state_load_started),loader_mask_seconds=0,compute_seconds=0,evidence_seconds=0;auto io_start=Clock::now();
  std::ofstream log(args.output/"training.csv");check(bool(log),"new suffix training log");log<<"attempt,update,training_objective_F64,eligible_examples,target_cells,nonzero_gradient_tensors,changed_tensors,row_seed,mask_seed,anchors\n"<<std::setprecision(17);
#ifdef CPB_OBJECTIVE_CANDIDATE
  std::ofstream temporal_log(args.output/"temporal-training.csv");check(bool(temporal_log),"new suffix temporal support/component log");
  temporal_log<<"attempt,update,base_nll,temporal_loss,weighted_temporal,total_loss,lag1_loss,lag4_loss,lag16_loss,lag1_pairs,lag4_pairs,lag16_pairs,lag1_supported,lag4_supported,lag16_supported\n"<<std::setprecision(17);
#endif
  model->train();const auto trajectory_start=Clock::now();const bool resume_verified=true;int64_t trained_cells=prefix.target_cells;double first_loss=prefix.first_loss,last_loss=0;
  while(progress.completed_updates<updates&&progress.attempted_batches<max_attempts){
    auto stage=Clock::now();const auto attempt=progress.attempted_batches,row_stream=tr::training_counter_seed(row_seed,attempt),mask_stream=tr::training_counter_seed(mask_seed,attempt);
    std::mt19937_64 random(row_stream);std::uniform_int_distribution<int64_t> pick(0,selections.train.size()-1);std::vector<int64_t> rows;for(int64_t i=0;i<batch;++i)rows.push_back(selections.train[pick(random)]);
    const auto raw=data.load(rows,md::train_begin,md::train_end);const auto masks=impl::configurable_spectral_make_training_mask(raw.observed,config.level,mask_stream);
    if(!masks.eligible_channels.any().item<bool>()){++progress.attempted_batches;loader_mask_seconds+=elapsed(stage);log<<attempt<<','<<progress.completed_updates<<",,0,0,0,0,"<<row_stream<<','<<mask_stream<<",\"\"\n";continue;}
    const auto context=impl::context_deletion::make_plan(masks,raw.channel_ids,config.level,context_seed,attempt,impl::ContextDeletionRecipe::coordinate15_v1);loader_mask_seconds+=elapsed(stage);
    stage=Clock::now();std::vector<SavedTensor> before;for(const auto &p:model->named_parameters())if(p.value().requires_grad())before.push_back({p.key(),p.value().detach().clone()});torch::cuda::synchronize();evidence_seconds+=elapsed(stage);
    optimizer->zero_grad(true);torch::cuda::synchronize();stage=Clock::now();
    const auto trace=model->forward_at(raw,torch::tensor(rows,torch::kFloat64),masks.hidden,scaler,context.deleted);
    check(trace.nll.loss.is_cuda()&&trace.nll.loss.scalar_type()==torch::kFloat64&&torch::isfinite(trace.nll.loss).item<bool>(),"finite actual CUDA F64 reporting Gaussian loss");
    check(trace.output.loss.is_cuda()&&trace.output.loss.scalar_type()==torch::kFloat64&&torch::isfinite(trace.output.loss).item<bool>(),"finite actual CUDA F64 declared training objective");
    torch::cuda::synchronize();compute_seconds+=elapsed(stage);
    if(progress.completed_updates==0||progress.completed_updates==2047){
      const auto witness_start=Clock::now();
      torch::serialize::OutputArchive witness;
      const auto save=[&](const char *name,const torch::Tensor &value){witness.write(name,value.detach().cpu(),true);};
      save("mean",trace.distribution.mean);save("variance",trace.distribution.variance);save("targets",trace.targets);
      save("target_mask",trace.target_mask);save("visible",trace.visible);save("eligible_channels",trace.output.eligible_channels);
      save("reporting_nll",trace.nll.loss);save("training_objective",trace.output.loss);
#ifdef CPB_OBJECTIVE_CANDIDATE
      save("temporal_huber",trace.objective.temporal.loss);
      trace.distribution.mean.retain_grad();trace.distribution.variance.retain_grad();
#endif
      witness.write("update",torch::tensor(progress.completed_updates+1,torch::kInt64),true);
      witness.write("attempt",torch::tensor(progress.attempted_batches,torch::kInt64),true);
      publish(witness,args.output/"witnesses"/("objective-update-"+std::to_string(progress.completed_updates+1)+".pt"));
      torch::cuda::synchronize();evidence_seconds+=elapsed(witness_start);
    }
    torch::cuda::synchronize();stage=Clock::now();
    trace.output.loss.backward();
#ifdef CPB_OBJECTIVE_CANDIDATE
    if(progress.completed_updates==0||progress.completed_updates==2047){
      torch::cuda::synchronize();compute_seconds+=elapsed(stage);
      const auto witness_start=Clock::now();torch::serialize::OutputArchive witness;
      const auto save=[&](const char *name,const torch::Tensor &value){witness.write(name,value.detach().cpu(),true);};
      save("mean",trace.distribution.mean);save("variance",trace.distribution.variance);save("targets",trace.targets);save("target_mask",trace.target_mask);save("visible",trace.visible);
      check(trace.distribution.mean.grad().defined()&&trace.distribution.variance.grad().defined(),"actual retained total output gradients before clipping");
      save("mean_total_gradient",trace.distribution.mean.grad());save("variance_total_gradient",trace.distribution.variance.grad());
      std::vector<torch::Tensor> lag_masks,lag_counts,lag_channel_losses,lag_example_losses,lag_losses,lag_eligible_channels,lag_eligible_examples;
      std::vector<int64_t> lags;std::vector<uint8_t> lag_supported;
      const auto B=trace.target_mask.size(0),C=trace.target_mask.size(1),H=trace.target_mask.size(2),F=trace.target_mask.size(3);
      for(const auto &lag:trace.objective.temporal.lags){
        lags.push_back(lag.lag);auto padded=torch::zeros({B,C,H,F},trace.target_mask.options());
        if(H>lag.lag)padded.narrow(2,lag.lag,H-lag.lag).copy_(lag.pair_mask);
        lag_masks.push_back(padded);lag_counts.push_back(lag.pair_counts);lag_channel_losses.push_back(lag.channel_loss);lag_example_losses.push_back(lag.example_loss);lag_losses.push_back(lag.loss);
        lag_eligible_channels.push_back(lag.eligible_channels);lag_eligible_examples.push_back(lag.eligible_examples);lag_supported.push_back(lag.eligible_example_count>0);
      }
      save("lags",torch::tensor(lags,torch::kInt64));save("pair_masks",torch::stack(lag_masks));save("pair_counts",torch::stack(lag_counts));
      save("lag_channel_losses",torch::stack(lag_channel_losses));save("lag_example_losses",torch::stack(lag_example_losses));save("lag_losses",torch::stack(lag_losses));
      save("lag_eligible_channels",torch::stack(lag_eligible_channels));save("lag_eligible_examples",torch::stack(lag_eligible_examples));save("lag_supported",torch::tensor(lag_supported,torch::kUInt8).to(torch::kBool));
      save("temporal_loss",trace.objective.temporal.loss);save("base_nll",trace.nll.loss);save("total_loss",trace.output.loss);
      save("coefficient",torch::tensor(.1,torch::kFloat64));save("endpoint_scale",torch::tensor(std::sqrt(2.),torch::kFloat64));save("huber_delta",torch::tensor(1.,torch::kFloat64));
      save("update",torch::tensor(progress.completed_updates+1,torch::kInt64));save("attempt",torch::tensor(progress.attempted_batches,torch::kInt64));
      publish(witness,args.output/"witnesses"/("temporal-objective-step-"+std::to_string(progress.completed_updates+1)+".pt"));
      torch::cuda::synchronize();evidence_seconds+=elapsed(witness_start);stage=Clock::now();
    }
#endif
    for(const auto &p:model->named_parameters())if(p.value().grad().defined())check(p.value().requires_grad()&&p.value().grad().is_cuda()&&torch::isfinite(p.value().grad()).all().item<bool>(),"finite learned CUDA gradients before optimizer mutation");
    torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer->step();torch::cuda::synchronize();compute_seconds+=elapsed(stage);
    stage=Clock::now();check(torch::equal(trace.visible.cpu(),context.visible)&&torch::equal(trace.target_mask.cpu(),masks.target)&&!(trace.visible&trace.target_mask).any().item<bool>(),"target/context separation with exact final mask");
    int64_t gradient_tensors=0,changed_tensors=0;const auto named=model->named_parameters();
    for(const auto &p:named){const auto gradient=p.value().grad();if(gradient.defined()){check(p.value().requires_grad()&&gradient.is_cuda()&&torch::isfinite(gradient).all().item<bool>(),"finite actual CUDA gradient");if(gradient.abs().sum().item<double>()>0){++evidence.at(p.key()).gradient_steps;++gradient_tensors;}}else check(!p.value().requires_grad()||p.value().is_cuda(),"CUDA inactive parameter");}
    for(const auto &p:before)if(!torch::equal(p.value,named[p.name].detach())){++evidence.at(p.name).moved_steps;++changed_tensors;}
    ++progress.completed_updates;++progress.attempted_batches;model->set_causal_updates(progress.completed_updates);
#ifdef CPB_OBJECTIVE_CANDIDATE
    std::ostringstream temporal_row;temporal_row<<std::setprecision(17)<<attempt<<','<<progress.completed_updates<<','<<trace.nll.loss.item<double>()<<','<<trace.objective.temporal.loss.item<double>()<<','<<.1*trace.objective.temporal.loss.item<double>()<<','<<trace.output.loss.item<double>();
    for(const auto &lag:trace.objective.temporal.lags)temporal_row<<','<<lag.loss.item<double>();
    for(const auto &lag:trace.objective.temporal.lags)temporal_row<<','<<lag.pair_count;
    for(const auto &lag:trace.objective.temporal.lags)temporal_row<<','<<(lag.eligible_example_count>0?1:0);
    compare_replay(original_temporal_rows,progress.completed_updates,temporal_row.str(),"temporal-training.csv");temporal_log<<temporal_row.str()<<'\n';
#endif
    check(model->causal_updates()==progress.completed_updates,"live causal update counter matches completed CUDA steps");
    last_loss=trace.output.loss.item<double>();if(progress.completed_updates==1)first_loss=last_loss;trained_cells+=trace.nll.target_cell_count;
    check(gradient_tensors>0&&changed_tensors>0,"actual CUDA learning at every successful update");evidence_seconds+=elapsed(stage);
    stage=Clock::now();std::ostringstream main_row;main_row<<std::setprecision(17)<<attempt<<','<<progress.completed_updates<<','<<last_loss<<','<<trace.nll.eligible_example_count<<','<<trace.nll.target_cell_count<<','<<gradient_tensors<<','<<changed_tensors<<','<<row_stream<<','<<mask_stream<<",\"";
    for(size_t i=0;i<rows.size();++i){if(i)main_row<<';';main_row<<rows[i];}main_row<<'"';compare_replay(original_main_rows,progress.completed_updates,main_row.str(),"training.csv");log<<main_row.str()<<'\n';checkpoint_seconds+=elapsed(stage);
    optimizer->zero_grad(true);
    if(progress.completed_updates%128==0){frozen.validate(model,scaler);tr::validate_optimizer(model,*optimizer,progress);std::cout<<"update="<<progress.completed_updates<<" attempts="<<progress.attempted_batches<<" loss="<<last_loss<<std::endl;}
    if(progress.completed_updates==1024||progress.completed_updates==2048){
      stage=Clock::now();check(model->causal_updates()==progress.completed_updates,"live counter agrees before typed training save");log.flush();const auto path=args.output/"checkpoints"/("training-"+std::to_string(progress.completed_updates)+".pt");tr::save_training_state(path.string(),model,scaler,*optimizer,identity,progress);
      temporal_log.flush();check(bool(temporal_log),"flush suffix temporal CSV at typed checkpoint");
      checkpoint_seconds+=elapsed(stage);
    }
  }
  torch::cuda::synchronize();const auto trajectory_seconds=elapsed(trajectory_start);log.close();check(bool(log),"persist training log");check(progress.completed_updates==2048&&resume_verified,"complete one declared causal trajectory with actual typed resume");optimizer->zero_grad(true);frozen.validate(model,scaler);tr::validate_optimizer(model,*optimizer,progress);
#ifdef CPB_OBJECTIVE_CANDIDATE
  temporal_log.close();check(bool(temporal_log),"persist all actual temporal support/component rows");
#endif
  check(model->causal_updates()==progress.completed_updates&&progress.completed_updates==2048,"final live counter agrees before inference save and query evidence");
  io_start=Clock::now();model->eval();model->clear_gradients();cp::save_checkpoint((args.output/"checkpoints/inference-2048.pt").string(),model,scaler,2048);checkpoint_seconds+=elapsed(io_start);
  const auto final_train=capture(data,model,scaler,selections.probe_train,query_train,md::train_begin,md::train_end,args.output/"witnesses/final-train-query.pt");
  const auto final_validation=capture(data,model,scaler,selections.probe_validation,query_validation,md::validation_begin,md::validation_end,args.output/"witnesses/final-validation-query.pt");
  check(model->causal_updates()==progress.completed_updates,"final query evidence retains exact live update counter");write(args.output/"query-final.json","{\"train\":"+query_json(final_train)+",\"validation\":"+query_json(final_validation)+"}");frozen.validate(model,scaler);verify(pins);
  struct Group{int64_t tensors{0},values{0},gradient_tensors{0},moved_tensors{0},changed_boundary_tensors{0};};std::map<std::string,Group> groups;int64_t total_values=0,updated_values=0,fixed_values=0;std::ostringstream parameter_json;parameter_json<<"{\"kind\":\"cpb_v16_operational_recovery_parameter_evidence_v1\",\"status\":\"passed\",\"evidence_scope\":\"logical-updates513..2048-only\",\"whole_trajectory_per_parameter_counters_complete\":false,\"entries\":[";bool first=true;
  for(const auto &[name,e]:evidence){const auto current=model->named_parameters()[name];const bool changed=!cp::boundary::same_bits(current,e.initial);total_values+=e.values;if(e.trainable){auto &g=groups[category(name)];++g.tensors;g.values+=e.values;g.gradient_tensors+=e.gradient_steps>0;g.moved_tensors+=e.moved_steps>0;g.changed_boundary_tensors+=changed;if(e.moved_steps)updated_values+=e.values;}else{fixed_values+=e.values;check(!changed&&e.gradient_steps==0&&e.moved_steps==0,"fixed prior never learns or drifts in recovery segment");}
    if(!first)parameter_json<<',';first=false;parameter_json<<"{\"name\":"<<quote(name)<<",\"group\":"<<quote(category(name))<<",\"values\":"<<e.values<<",\"requires_grad\":"<<(e.trainable?"true":"false")<<",\"nonzero_CUDA_gradient_steps\":null,\"weight_movement_steps\":null,\"changed_from_initial\":null,\"suffix_nonzero_CUDA_gradient_steps\":"<<e.gradient_steps<<",\"suffix_weight_movement_steps\":"<<e.moved_steps<<",\"changed_from_recovery_boundary\":"<<(changed?"true":"false")<<'}';}
  parameter_json<<"]}";write(args.output/"parameter-learning.json",parameter_json.str());
  std::ostringstream group_json;group_json<<'[';first=true;for(const auto &[name,g]:groups){if(!first)group_json<<',';first=false;group_json<<"{\"group\":"<<quote(name)<<",\"evidence_scope\":\"logical-updates513..2048-only\",\"tensors\":"<<g.tensors<<",\"values\":"<<g.values<<",\"nonzero_gradient_tensors\":"<<g.gradient_tensors<<",\"moved_tensors\":"<<g.moved_tensors<<",\"changed_final_tensors\":null,\"changed_from_recovery_boundary_tensors\":"<<g.changed_boundary_tensors<<'}';}group_json<<']';
  for(const auto &required:{"parent_backbone_projection","parent_content_relation","parent_decoder","parent_global_pool","change_projection","change_wave_decoder","variance_matrix","variance_bias"})check(groups.count(required)&&groups.at(required).gradient_tensors>0&&groups.at(required).moved_tensors>0,"required model category actually learns on CUDA in recovery segment: "+std::string(required));
  check(total_values==297560&&fixed_values==3888&&progress.completed_updates-resume_update==1536,"unchanged parameter partition and full declared recovery suffix");
  std::ostringstream summary;summary<<std::setprecision(17)<<"{\"kind\":\"cpb_v16_temporal_paired_training_summary_v1\",\"recovery_summary_kind\":\"cpb_v16_temporal_operational_recovery_summary_v1\",\"status\":\"passed\",\"model\":\"CPB-v16-temporal-paired\",\"arm\":"<<quote(cpb_v16_temporal::arm)<<",\"row_seed\":"<<row_seed<<",\"mask_seed\":"<<mask_seed<<",\"context_seed\":"<<context_seed<<",\"gradient_policy\":"<<quote(cp::kGradientPolicy)<<",\"encoder_gradient_kind\":\"compiled-branch-objective-see-gradient-policy\",\"conditional_variance_head_trainable\":true,\"initialization_kind\":\"fresh-seeded-constructor-no-import-v1\",\"initialization_seed\":"<<seed<<",\"source_updates\":0,\"new_causal_updates\":"<<progress.completed_updates<<",\"attempted_batches\":"<<progress.attempted_batches<<",\"batch_size\":8,\"TRAIN_windows\":4096,\"probe_TRAIN_windows\":112,\"probe_VALIDATION_windows\":32,\"native_width\":36,\"device\":\"cuda:0\",\"parameters\":"<<total_values<<",\"parameters_with_recorded_updates\":"<<updated_values<<",\"parameter_evidence_scope\":\"recovery-segment513..2048;full-prefix-per-parameter-counters-unavailable\",\"fixed_prior_parameters\":"<<fixed_values<<",\"parameter_groups\":"<<group_json.str()<<",\"first_training_objective\":"<<first_loss<<",\"last_training_objective\":"<<last_loss<<",\"training_target_cells\":"<<trained_cells
    <<",\"GPU_step_seconds_synchronized\":null,\"GPU_step_timing_scope\":\"whole-trajectory-unavailable-after-external-interruption;see-recovery_segment\",\"loader_mask_seconds\":null,\"checkpoint_resume_logging_seconds\":null,\"parameter_gradient_evidence_seconds\":null,\"training_trajectory_wall_seconds\":null"
    <<",\"fixed_query_initial\":"<<initial_query_json<<",\"fixed_query_initial_source\":"<<quote((args.source_output/"query-initial.json").string())<<",\"initial_query_encoder_replays\":0,\"fixed_query_final\":{\"train\":"<<query_json(final_train)<<",\"validation\":"<<query_json(final_validation)<<"},\"raw_load_calls\":null,\"direct_F64_record_cells_checked\":null,\"genuine_F64_differs_from_F32_cells\":null,\"typed_resume_512_exact\":true,\"typed_resume_512_original_proof_source\":"<<quote((args.source_output/"typed-resume-512.json").string())<<",\"fixed_prior_scaler_normalizer_unchanged\":true,\"new_scaler_fits\":1,\"new_normalizer_fits\":1,\"recovery_new_scaler_fits\":0,\"recovery_new_normalizer_fits\":0,\"calibration_encoder_calls\":0,\"calibration_validation_rows\":0,\"calibration_seconds\":"<<finite_number(flat_token(calibration_metadata,"seconds_including_loader_witness_IO"))<<",\"calibration_TRAIN_loader_calls\":"<<decimal(flat_token(calibration_metadata,"loader_calls"))<<",\"calibration_scope\":\"retained-original-calibration-metadata;not-refitted\",\"fresh_scaler_identity\":"<<quote(scaler.identity())<<",\"fresh_bank_identity\":"<<quote(model->normalizer().identity())<<",\"old_model_artifacts_opened\":0,\"old_normalization_artifacts_opened\":0,\"recovery_own_trajectory_training_states_opened\":1,\"saved_quality_features_reused\":false,\"fixed_query_source\":\"unchanged saved masks; source seed942005; no new query RNG consumed\",\"CPU_encoder_calls\":0,\"old_encoder_calls\":0,\"head_fits\":0,\"training_replays\":null"
    <<",\"operational_recovery\":{\"external_process_interruption\":true,\"resume_checkpoint_update\":512,\"suffix_first_update\":513,\"suffix_last_update\":2048,\"suffix_successful_updates\":1536,\"original_completed_updates_exact\":null,\"physical_repeated_updates_exact\":null,\"observed_original_complete_temporal_last_update\":565,\"whole_trajectory_cost_complete\":false,\"whole_trajectory_per_parameter_counters_complete\":false,\"scientific_recipe_changed\":false,\"all_complete_original_main_overlap_rows_equal\":true,\"all_complete_original_temporal_overlap_rows_equal\":true,\"overlap_comparison\":\"exact-producer-CSV-lexemes;no-tolerance-or-rescue\"}"
    <<",\"recovery_segment\":{\"scope\":\"actual-logical-updates513..2048-plus-final-queries;excludes-original-prefix-and-discarded-interrupted-work\",\"GPU_step_seconds_synchronized\":"<<compute_seconds<<",\"GPU_step_timing_scope\":\"forward_at+declared_F64_objective+backward+clip+AdamW;synchronized_wall;includes_in_step_contract_checks;excludes_loader_masks_checkpoints_evidence_logging\",\"loader_mask_seconds\":"<<loader_mask_seconds<<",\"checkpoint_resume_logging_seconds\":"<<checkpoint_seconds<<",\"parameter_gradient_evidence_seconds\":"<<evidence_seconds<<",\"training_trajectory_wall_seconds\":"<<trajectory_seconds<<",\"raw_load_calls\":"<<data.calls<<",\"direct_F64_record_cells_checked\":"<<data.record_cells<<",\"genuine_F64_differs_from_F32_cells\":"<<data.genuine_f64_cells<<"}}";
  write(args.output/"training-summary.json",summary.str());std::cout<<"CPB_V16_RECOVERY_TRAINING="<<args.output<<'\n';return 0;
}
} // namespace
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
