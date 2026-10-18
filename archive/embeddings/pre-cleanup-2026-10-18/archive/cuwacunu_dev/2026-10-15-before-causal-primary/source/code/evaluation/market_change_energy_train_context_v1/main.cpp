// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Saved TRAIN prediction arithmetic only. No encoder, fitting, or held-out input.
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <spawn.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
extern char **environ;

namespace {
namespace fs=std::filesystem;
using Tensor=torch::Tensor;
constexpr int64_t rows=112,tasks=4;
constexpr double audit_tolerance=1e-12;
constexpr std::array<int64_t,3> repetitions{2701,2802,2903};
constexpr std::array<const char*,4> task_names{"past_volatility","past_trend","past_activity","past_relation"};
constexpr std::array<const char*,4> task_units{"logreturn","logreturn/tick","log1p(base_volume)","correlation"};
struct Pin{fs::path path;std::string hash;uintmax_t bytes;};
std::vector<Pin> opened;
int64_t audit_checks=0;double maximum_normalized_difference=0;
void require(bool condition,const std::string &why){TORCH_CHECK(condition,"[energy saved TRAIN context] ",why);}
std::string quote(const std::string &value){std::ostringstream out;out<<'"';for(unsigned char c:value){
  if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
  else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}return out.str()+'"';}
std::string number(double value){if(!std::isfinite(value))return "null";std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();}
std::string array(const std::vector<std::string> &values){std::string out="[";for(size_t j=0;j<values.size();++j){if(j)out+=',';out+=values[j];}return out+']';}
std::string object(const std::vector<std::pair<std::string,std::string>> &values){std::string out="{";for(size_t j=0;j<values.size();++j){if(j)out+=',';out+=quote(values[j].first)+':'+values[j].second;}return out+'}';}
std::string tensor_json(const Tensor &value){const auto x=value.to(torch::kFloat64).contiguous();if(x.dim()==0)return number(x.item<double>());std::vector<std::string> parts;for(int64_t j=0;j<x.size(0);++j)parts.push_back(tensor_json(x[j]));return array(parts);}

std::string sha256(const fs::path &path){
  require(fs::is_regular_file(path),"missing input "+path.string());int fds[2];require(::pipe(fds)==0,"checksum pipe");
  posix_spawn_file_actions_t actions;require(::posix_spawn_file_actions_init(&actions)==0,"checksum actions");
  ::posix_spawn_file_actions_adddup2(&actions,fds[1],STDOUT_FILENO);::posix_spawn_file_actions_addclose(&actions,fds[0]);::posix_spawn_file_actions_addclose(&actions,fds[1]);
  std::string name=path.string();char command[]="sha256sum",separator[]="--";char *argv[]={command,separator,name.data(),nullptr};pid_t pid;
  const auto rc=::posix_spawnp(&pid,command,&actions,nullptr,argv,environ);::posix_spawn_file_actions_destroy(&actions);::close(fds[1]);
  if(rc){::close(fds[0]);require(false,"checksum spawn");}
  std::string output;char data[512];ssize_t n;while((n=::read(fds[0],data,sizeof data))>0)output.append(data,n);::close(fds[0]);
  int status=0;require(::waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0&&output.size()>=64,"checksum child");
  const auto hash=output.substr(0,64);require(hash.find_first_not_of("0123456789abcdef")==std::string::npos,"checksum syntax");return hash;
}
void open(torch::serialize::InputArchive &archive,const fs::path &path,const std::string &expected={}){
  const auto canonical=fs::canonical(path);require(canonical.filename()=="predictions_train.pt"||canonical==fs::path("/cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/train-targets.pt"),"only declared TRAIN tensor basenames permitted");
  const auto hash=sha256(canonical);require(expected.empty()||hash==expected,"pinned historical TRAIN archive changed: "+canonical.string());
  for(const auto &pin:opened)require(pin.path!=canonical,"duplicate archive access");
  opened.push_back({canonical,hash,fs::file_size(canonical)});archive.load_from(canonical.string(),torch::kCPU);
}
Tensor read(torch::serialize::InputArchive &archive,const char *name){Tensor out;archive.read(name,out,true);require(out.defined()&&out.device().is_cpu()&&!out.requires_grad(),std::string(name)+" detached CPU tensor required");return out;}
void shape(const Tensor &value,at::IntArrayRef sizes,torch::ScalarType type,const char *name){require(value.sizes()==sizes&&value.scalar_type()==type,std::string(name)+" shape/type mismatch");}
void finite(const Tensor &value,const char *name){require(torch::isfinite(value).all().item<bool>(),std::string(name)+" nonfinite storage");}
void exact(const Tensor &a,const Tensor &b,const char *why){require(torch::equal(a,b),why);++audit_checks;}
void close(double actual,double expected,const char *why){require(std::isfinite(actual)&&std::isfinite(expected),std::string(why)+" undefined arithmetic");const double difference=std::abs(actual-expected)/(1.+std::abs(expected));maximum_normalized_difference=std::max(maximum_normalized_difference,difference);require(difference<=audit_tolerance,std::string(why)+" scalar/tensor mismatch");++audit_checks;}

struct Targets{Tensor values,valid,row_ids,group_ids;};
Targets targets(){torch::serialize::InputArchive archive;open(archive,"/cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/train-targets.pt","521d8662f498bce27ad26a41c4723bbebb9d2ab6b924cfbcce7132c25b7e43e6");
  Targets out{read(archive,"targets"),read(archive,"valid"),read(archive,"row_ids"),read(archive,"group_ids")};
  shape(out.values,{rows,tasks},torch::kFloat64,"TRAIN targets");shape(out.valid,{rows,tasks},torch::kBool,"TRAIN target support");shape(out.row_ids,{rows},torch::kInt64,"TRAIN row ids");shape(out.group_ids,{rows},torch::kInt64,"TRAIN group ids");finite(out.values,"TRAIN targets");
  require(out.valid.all().item<bool>(),"full TRAIN target coverage required");require(out.row_ids.slice(0,1).gt(out.row_ids.slice(0,0,rows-1)).all().item<bool>(),"fixed halves require strict chronological row ids");return out;}
struct Predictions{Tensor ridge,neural,valid;};
Predictions predictions(const fs::path &path,const Targets &target,const std::string &expected={}){
  torch::serialize::InputArchive archive;open(archive,path,expected);
  Predictions out{read(archive,"ridge_predictions"),read(archive,"neural_predictions"),read(archive,"valid")};
  shape(out.ridge,{rows,tasks},torch::kFloat64,"ridge predictions");shape(out.neural,{rows,tasks},torch::kFloat64,"neural predictions");shape(out.valid,{rows,tasks},torch::kBool,"prediction support");finite(out.ridge,"ridge predictions");finite(out.neural,"neural predictions");
  exact(read(archive,"targets"),target.values,"saved TRAIN targets differ");exact(read(archive,"target_valid"),target.valid,"saved TRAIN target support differs");exact(read(archive,"row_ids"),target.row_ids,"saved TRAIN row association differs");exact(read(archive,"group_ids"),target.group_ids,"saved TRAIN group association differs");require(out.valid.all().item<bool>(),"full112-row prediction coverage required");return out;
}
struct Method{std::string label,key;int64_t width;std::array<Predictions,3> repetitions;};
Method method(const std::string &label,const std::string &key,int64_t width,const fs::path &run,const Targets &target,const std::array<std::string,3> &expected={}){
  Method out{label,key,width,{}};for(size_t j=0;j<repetitions.size();++j){out.repetitions[j]=predictions(run/"readouts/models"/key/("rep-"+std::to_string(repetitions[j]))/"predictions_train.pt",target,expected[j]);if(j){exact(out.repetitions[j].ridge,out.repetitions[0].ridge,"deterministic ridge differs across saved repetitions");exact(out.repetitions[j].valid,out.repetitions[0].valid,"repetition scoring support differs");}}return out;
}
struct Score{double mae,rmse,r2,sse,sst;int64_t count;};
Score score(const Tensor &prediction,const Tensor &target,int64_t begin,int64_t end){
  const auto p=prediction.slice(0,begin,end).contiguous(),y=target.slice(0,begin,end).contiguous(),error=p-y;
  const auto sse=error.square().sum().item<double>(),sst=(y-y.mean()).square().sum().item<double>();
  require(end-begin>=2&&sst>0,"R² requires nonconstant declared TRAIN block");
  Score out{error.abs().mean().item<double>(),std::sqrt(sse/(end-begin)),1.-sse/sst,sse,sst,end-begin};
  const auto pv=p.accessor<double,1>(),yv=y.accessor<double,1>();double sum=0,squares=0,absolute=0,mean=0;
  for(int64_t j=0;j<p.numel();++j)mean+=yv[j];
  mean/=p.numel();
  for(int64_t j=0;j<p.numel();++j){const auto e=pv[j]-yv[j],d=yv[j]-mean;sum+=e*e;squares+=d*d;absolute+=std::abs(e);}
  close(out.sse,sum,"SSE");close(out.sst,squares,"SST");close(out.mae,absolute/p.numel(),"MAE");close(out.rmse,std::sqrt(sum/p.numel()),"RMSE");close(out.r2,1.-sum/squares,"R²");return out;
}
std::string score_json(const Score &s){return object({{"total",std::to_string(s.count)},{"target_eligible",std::to_string(s.count)},{"valid",std::to_string(s.count)},{"coverage","1"},{"r2",number(s.r2)},{"mae",number(s.mae)},{"rmse",number(s.rmse)},{"sse",number(s.sse)},{"sst",number(s.sst)}});}
std::string task_context(const Method &method,const Targets &target,int64_t task,int64_t begin,int64_t end){
  const auto y=target.values.select(1,task);const auto linear=score(method.repetitions[0].ridge.select(1,task),y,begin,end);
  std::vector<std::string> each;double r2=0,mae=0,rmse=0;
  for(size_t j=0;j<repetitions.size();++j){const auto s=score(method.repetitions[j].neural.select(1,task),y,begin,end);r2+=s.r2/3.;mae+=s.mae/3.;rmse+=s.rmse/3.;each.push_back(object({{"repetition",std::to_string(repetitions[j])},{"score",score_json(s)}}));}
  return object({{"task",quote(task_names[task])},{"unit",quote(task_units[task])},{"row_start_inclusive",std::to_string(begin)},{"row_end_exclusive",std::to_string(end)},{"ridge",score_json(linear)},{"neural_each",array(each)},{"neural_mean_repetition_scores",object({{"r2",number(r2)},{"mae",number(mae)},{"rmse",number(rmse)}})}});
}
std::string method_context(const Method &method,const Targets &target){std::vector<std::string> contexts;
  for(const auto &[label,bounds]:std::vector<std::pair<std::string,std::pair<int64_t,int64_t>>>{{"all112",{0,112}},{"first56",{0,56}},{"last56",{56,112}}}){
    std::vector<std::string> scores;for(int64_t task=0;task<tasks;++task)scores.push_back(task_context(method,target,task,bounds.first,bounds.second));
    contexts.push_back(object({{"context",quote(label)},{"rows",std::to_string(bounds.second-bounds.first)},{"tasks",array(scores)}}));}
  return object({{"method",quote(method.label)},{"archive_method",quote(method.key)},{"size",std::to_string(method.width)},{"contexts",array(contexts)}});
}
void write_new(const fs::path &path,const std::string &text){const auto fd=::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);require(fd>=0,"exclusive output create "+path.string());
  size_t offset=0;while(offset<text.size()){const auto n=::write(fd,text.data()+offset,text.size()-offset);if(n<=0){::close(fd);require(false,"output write "+path.string());}offset+=n;}require(::fsync(fd)==0,"output sync");require(::close(fd)==0,"output close");}
} // namespace

int main(int argc,char **argv)try{
  require(argc==3,"usage: market_change_energy_train_context V25_RUN NEW_OUTPUT_LEAF");require(fs::is_regular_file("/.dockerenv")&&fs::is_directory("/cuwacunu"),"managed cuwacunu container required");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard no_grad;
  const fs::path run=fs::canonical(argv[1]);require(run.parent_path()==fs::path("/cuwacunu/output/runs/market-change-energy")&&run.filename().string().starts_with("run-"),"exact declared v25 run tree required");
  const fs::path output=fs::weakly_canonical(fs::absolute(argv[2]));require(output.parent_path()==fs::path("/cuwacunu/output/runs/market-change-energy-train-context")&&!fs::exists(output),"exclusive dedicated TRAIN context output leaf required");
  const auto started=std::chrono::steady_clock::now();const auto target=targets();
  std::vector<Method> methods;
  methods.push_back(method("RPB-v22.market-02","base_trained",32,"/cuwacunu/output/runs/market-capacity-study/run-base-gryiAV",target,{"02d61e4f687767cc6fd21185ada2001c69d1daa5fb16e86e3d0a451814c870da","446751110104217a8dea4db35648b7b4617a0f7ed8c4ab5108ef0fd829ab4f43","098b2b86d05b5150511b4e1ccbf11be2b21d4d9e0d7bd22ac5a897d74306c6b8"}));
  methods.push_back(method("RPB-v24.market-01.alt-01","trained_v24_native36",36,"/cuwacunu/output/runs/market-separate-changes-native36/run-4SauBf",target,{"efc721e51a66590e9a67380665415e55078a7f9d8d710f7e1f69e54e4f78ff53","b8dcc6caebabc1cf4a0c1501a673113204c39561a8bf38ff9f889da7bb5ea996","ae5a345877bf85b4058d3c4512ebcb9e8d721f35c783065bc55f132621097741"}));
  methods.push_back(method("RPB-v25.market-01.init","initialized_v25",36,run,target));methods.push_back(method("RPB-v25.market-01","trained_v25",36,run,target));
  require(opened.size()==13,"exact13 TRAIN-only tensor input count");const auto loaded=std::chrono::steady_clock::now();
  std::vector<std::string> details;for(const auto &m:methods)details.push_back(method_context(m,target));
  std::vector<std::string> inputs,witnesses;for(const auto &pin:opened){require(fs::file_size(pin.path)==pin.bytes&&sha256(pin.path)==pin.hash,"TRAIN input mutated during arithmetic");inputs.push_back(object({{"path",quote(fs::relative(pin.path,"/cuwacunu").generic_string())},{"sha256",quote(pin.hash)},{"bytes",std::to_string(pin.bytes)},{"role",quote("saved TRAIN prediction arithmetic only")}}));}
  for(const auto &m:methods){std::vector<std::string> neural;for(const auto &p:m.repetitions)neural.push_back(tensor_json(p.neural));witnesses.push_back(object({{"method",quote(m.label)},{"size",std::to_string(m.width)},{"ridge_predictions",tensor_json(m.repetitions[0].ridge)},{"neural_predictions_each",array(neural)}}));}
  const auto report=object({{"protocol",quote("market-change-energy-train-context-v1")},{"dataset_id",quote("MKT-HIST-2")},{"dataset_recipe",quote("BTC/ETH/SOL January2024 minute history; recent128 past-only targets")},{"designed_complexity_level","4"},{"complexity_scale_max","5"},{"view",quote("saved chronological TRAIN only; all112 and fixed first56/last56")},{"TRAIN_rows","112"},{"batch_size_during_original_encoder_training","8"},{"new_encoder_calls","0"},{"new_encoder_updates","0"},{"new_head_fits","0"},{"new_scaler_fits","0"},{"held_out_tensor_inputs","0"},{"device",quote("CPU saved-tensor arithmetic in existing managed GPU-enabled container")},{"opened_tensor_archives","13"},{"input_hashes_verified_before_and_after","true"},{"three_saved_deterministic_ridge_predictions_exact","true"},{"row_group_target_association_and_full_support_exact","true"},{"scalar_arithmetic_checks",std::to_string(audit_checks)},{"scalar_arithmetic_absolute_relative_tolerance",number(audit_tolerance)},{"scalar_arithmetic_maximum_normalized_difference",number(maximum_normalized_difference)},{"load_and_verification_seconds",number(std::chrono::duration<double>(loaded-started).count())},{"load_verify_arithmetic_and_hash_seconds_excluding_output",number(std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count())},{"opened_inputs",array(inputs)},{"methods",array(details)}});
  const auto witness=object({{"protocol",quote("market-change-energy-train-context-v1")},{"view",quote("saved TRAIN only")},{"target_values",tensor_json(target.values)},{"row_ids",tensor_json(target.row_ids)},{"group_ids",tensor_json(target.group_ids)},{"methods",array(witnesses)}});
  require(fs::create_directories(output),"exclusive output leaf create");write_new(output/"context.json",report+"\n");write_new(output/"opened-inputs.json",array(inputs)+"\n");write_new(output/"numerical-witness.json",witness+"\n");
  std::cout<<"TRAIN_CONTEXT_REPORT="<<(output/"context.json").string()<<"\nTRAIN_ARCHIVES_OPENED="<<opened.size()<<"\nSCALAR_ARITHMETIC_CHECKS="<<audit_checks<<"\nENCODER_CALLS=0 HEAD_FITS=0 ENCODER_UPDATES=0 HELD_OUT_TENSORS=0\n";return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
