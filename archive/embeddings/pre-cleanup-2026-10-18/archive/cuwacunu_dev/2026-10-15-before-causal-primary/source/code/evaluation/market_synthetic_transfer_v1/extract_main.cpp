// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// One declared frozen feature extraction, with no optimizer, decoder or head calls.
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_settings.h"
#include <torch/cuda.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;

namespace {
namespace fs=std::filesystem;
namespace r=embedding::encoders::raw_patch_bottleneck_mae;
using Clock=std::chrono::steady_clock;
using Pin=std::pair<fs::path,std::string>;
constexpr std::array<int64_t,3> masters{941101,941102,941103};
constexpr std::array<const char*,3> scaler_ids{
  "rpb-scaler-v1-fnv1a64-ac9d9673f1e24751",
  "rpb-scaler-v1-fnv1a64-5f51eb3918f161a4",
  "rpb-scaler-v1-fnv1a64-3a642694962518e9"};
constexpr const char *checkpoint_hash="a3939b13df8b1704fc3621876cf3adc6efaddd98951147ff43ec6408978fa98c";
void check(bool yes,const std::string &why){TORCH_CHECK(yes,"[market synthetic transfer] ",why);}
std::string json(const std::string &value){
  std::ostringstream out;out<<'"';for(unsigned char c:value){
    if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<c;
  }out<<'"';return out.str();
}
std::string sha256(const fs::path &path){
  check(fs::is_regular_file(path),"missing provenance file: "+path.string());
  int fds[2];check(::pipe(fds)==0,"checksum pipe");posix_spawn_file_actions_t actions;
  check(::posix_spawn_file_actions_init(&actions)==0,"checksum actions");
  ::posix_spawn_file_actions_adddup2(&actions,fds[1],STDOUT_FILENO);
  ::posix_spawn_file_actions_addclose(&actions,fds[0]);::posix_spawn_file_actions_addclose(&actions,fds[1]);
  std::string name=path.string();char executable[]="sha256sum",separator[]="--";
  char *args[]={executable,separator,name.data(),nullptr};pid_t pid;
  const int rc=::posix_spawnp(&pid,executable,&actions,nullptr,args,environ);
  ::posix_spawn_file_actions_destroy(&actions);::close(fds[1]);
  if(rc){::close(fds[0]);check(false,"checksum spawn");}
  std::string result;char buffer[512];ssize_t n;
  while((n=::read(fds[0],buffer,sizeof buffer))>0)result.append(buffer,n);
  ::close(fds[0]);int status=0;
  check(::waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0&&result.size()>=64,"checksum failed");
  const auto hash=result.substr(0,64);
  check(hash.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid checksum");return hash;
}
void write_text(const fs::path &path,const std::string &value){
  r::configurable_checkpoint_detail::StagedCheckpoint staged(path.string());
  std::ofstream f(staged.path(),std::ios::binary);check(bool(f),"open new text");
  f<<value;f.close();check(bool(f),"write new text");staged.publish();
}
std::vector<Pin> pin_sources(const fs::path &inputs,const fs::path &checkpoint,const fs::path &settings,const fs::path &card){
  std::vector<fs::path> files{checkpoint,settings,card,"Makefile","code/scripts/task.sh",
    "code/evaluation/market_synthetic_transfer_v1/extract_main.cpp",
    "code/evaluation/market_synthetic_transfer_v1/transfer.mk",
    fs::path("/proc")/std::to_string(::getpid())/"exe"};
  for(const auto &root:std::vector<fs::path>{inputs,"code/config/include","code/encoders/rpb_v19/include",
      "code/encoders/rpb_spectral/include","code/encoders/rpb_content/include","code/encoders/rpb_changes/include","code/encoders/rpb_separate/include"}){
    check(fs::is_directory(root),"missing source/input directory");
    for(const auto &entry:fs::recursive_directory_iterator(root)){
      check(!entry.is_symlink(),"symlink in frozen source/input bank");
      if(entry.is_regular_file())files.push_back(entry.path());
    }
  }
  std::sort(files.begin(),files.end());files.erase(std::unique(files.begin(),files.end()),files.end());
  std::vector<Pin> pins;for(const auto &file:files)pins.emplace_back(file,sha256(file));return pins;
}
std::string ledger(const std::vector<Pin> &pins){
  std::string value;for(const auto &[path,hash]:pins)value+=hash+"  "+path.generic_string()+"\n";return value;
}
void verify(const std::vector<Pin> &pins){for(const auto &[path,hash]:pins)check(sha256(path)==hash,"frozen source changed: "+path.string());}

struct FrozenValue{std::string name;torch::Tensor value;bool requires_grad;};
struct FrozenModel{
  std::vector<FrozenValue> parameters,buffers;
  explicit FrozenModel(const r::SeparatedChangeModel &model){
    for(const auto &v:model->named_parameters()){
      check(v.value().device()==torch::Device(torch::kCUDA,0)&&!v.value().grad().defined(),"actual CUDA0 parameter without gradient");
      parameters.push_back({v.key(),v.value().detach().to(torch::kCPU).clone(),v.value().requires_grad()});
    }
    for(const auto &v:model->named_buffers()){
      check(v.value().device()==torch::Device(torch::kCUDA,0),"actual CUDA0 buffer");
      buffers.push_back({v.key(),v.value().detach().to(torch::kCPU).clone(),v.value().requires_grad()});
    }
    check(!parameters.empty(),"empty model");
  }
  void verify(const r::SeparatedChangeModel &model)const{
    check(!model->is_training(),"model must remain in eval mode");
    const auto p=model->named_parameters(),b=model->named_buffers();
    check(p.size()==parameters.size()&&b.size()==buffers.size(),"model state count changed");
    const auto compare=[](const auto &named,const auto &saved){for(const auto &v:saved){
      const auto *now=named.find(v.name);
      check(now&&now->device()==torch::Device(torch::kCUDA,0)&&now->scalar_type()==v.value.scalar_type()&&
        !now->grad().defined()&&now->requires_grad()==v.requires_grad&&
        torch::equal(now->detach().to(torch::kCPU),v.value),"model state/gradient changed: "+v.name);
    }};compare(p,parameters);compare(b,buffers);
  }
};

// Label tensors are deliberately absent from the encoder-facing record.
struct Observations{torch::Tensor values,mask;};
Observations observations(const fs::path &file,int64_t rows){
  torch::serialize::InputArchive a;a.load_from(file.string(),torch::kCPU);Observations out;
  a.read("observed",out.values,true);a.read("feature_mask",out.mask,true);
  check(out.values.device().is_cpu()&&out.values.scalar_type()==torch::kFloat64&&
    out.values.sizes()==torch::IntArrayRef({rows,3,128,9}),"exact synthetic BCHF float64 observations");
  check(out.mask.device().is_cpu()&&out.mask.scalar_type()==torch::kBool&&out.mask.sizes()==out.values.sizes(),"exact synthetic support");
  check(torch::isfinite(out.values.masked_select(out.mask)).all().item<bool>(),"finite visible observations");return out;
}
r::Input input(const Observations &raw,int64_t offset,int64_t count){
  return {raw.values.narrow(0,offset,count),raw.mask.narrow(0,offset,count),torch::tensor({0,1,2},torch::kInt64),
    torch::full({count},127.*60000.,torch::kFloat64),60000.};
}
void copy_scoring_metadata(const fs::path &file,const std::string &role,int64_t rows,torch::serialize::OutputArchive &output){
  // This happens only after the role's frozen embeddings have been extracted.
  torch::serialize::InputArchive a;a.load_from(file.string(),torch::kCPU);
  torch::Tensor labels,ids,ids_json;
  a.read("labels_scoring_only",labels,true);a.read("source_ids",ids,true);a.read("source_ids_json",ids_json,true);
  check(labels.device().is_cpu()&&labels.scalar_type()==torch::kInt64&&labels.sizes()==torch::IntArrayRef({rows}),"scoring-only labels schema");
  check(ids.device().is_cpu()&&ids.scalar_type()==torch::kInt64&&ids.sizes()==torch::IntArrayRef({rows}),"source IDs schema");
  check(ids_json.device().is_cpu()&&ids_json.scalar_type()==torch::kUInt8&&ids_json.dim()==1,"source JSON byte schema");
  output.write(role+"_labels",labels,true);output.write(role+"_source_ids",ids,true);output.write(role+"_source_ids_json",ids_json,true);
}
struct Capture{int64_t calls{0},rows{0},valid{0},masked_cells{0};double seconds{0};};
Capture extract(const fs::path &file,const std::string &role,int64_t rows,r::SeparatedChangeModel &model,
    const r::SeparatedChangeConfig &config,const r::FrozenScaler &scaler,torch::serialize::OutputArchive &output){
  const auto raw=observations(file,rows);std::vector<torch::Tensor> values,valid;Capture result;
  torch::cuda::synchronize();const auto start=Clock::now();
  for(int64_t offset=0;offset<rows;offset+=8){
    const auto batch=std::min<int64_t>(8,rows-offset);
    const auto raw_input=input(raw,offset,batch);
    const auto normalized=r::transform_configurable_spectral_input(scaler,raw_input,config.level);
    check(normalized.data.is_cuda()&&normalized.observed.is_cuda()&&normalized.channel_ids.is_cuda()&&normalized.endpoints.is_cuda(),"actual CUDA values, mask, IDs and endpoints");
    check(normalized.data.scalar_type()==torch::kFloat32&&normalized.observed.scalar_type()==torch::kBool&&
      !normalized.data.masked_select(normalized.observed.logical_not()).ne(0).any().item<bool>(),"masked values must normalize to exact zero");
    // A masked sentinel is removed by preprocessing; no extra model forward is needed.
    if(offset==0){
      auto poisoned=raw_input;poisoned.data=torch::where(raw_input.observed,raw_input.data,torch::full_like(raw_input.data,123456789.));
      const auto safe=r::transform_configurable_spectral_input(scaler,poisoned,config.level);
      check(torch::equal(safe.data,normalized.data)&&torch::equal(safe.observed,normalized.observed),"masked sentinel changed encoder-facing input");
    }
    const auto encoded=model->encode(normalized);const auto native=r::separated_change_native(encoded,config);
    check(native.is_cuda()&&!native.requires_grad()&&native.sizes()==torch::IntArrayRef({batch,36})&&
      torch::isfinite(native).all().item<bool>(),"finite frozen CUDA native36 export");
    check(encoded.sample_valid_mask.is_cuda()&&encoded.sample_valid_mask.scalar_type()==torch::kBool&&
      encoded.sample_valid_mask.sizes()==torch::IntArrayRef({batch}),"CUDA sample validity");
    values.push_back(native.to(torch::kCPU,torch::kFloat64));valid.push_back(encoded.sample_valid_mask.to(torch::kCPU));
    result.valid+=encoded.sample_valid_mask.sum().item<int64_t>();result.masked_cells+=normalized.observed.logical_not().sum().item<int64_t>();
    ++result.calls;result.rows+=batch;
  }
  torch::cuda::synchronize();result.seconds=std::chrono::duration<double>(Clock::now()-start).count();
  output.write(role+"_values",torch::cat(values),true);output.write(role+"_valid",torch::cat(valid),true);
  copy_scoring_metadata(file,role,rows,output);return result;
}

int execute(int argc,char **argv){
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  check(argc==11,"usage: extract --inputs ROOT --output NEWLEAF --card FILE --checkpoint FILE --settings FILE");
  std::map<std::string,fs::path> args;
  for(int i=1;i<argc;i+=2)check(args.emplace(argv[i],argv[i+1]).second,"duplicate argument");
  for(const auto &key:{"--inputs","--output","--card","--checkpoint","--settings"})check(args.count(key),"missing argument");
  const auto inputs=args.at("--inputs"),out=args.at("--output"),card=args.at("--card"),checkpoint=args.at("--checkpoint"),settings=args.at("--settings");
  check(fs::is_regular_file(inputs/"manifest.json")&&!fs::exists(out),"input manifest/new output leaf required");
  check(sha256(checkpoint)==checkpoint_hash,"exact final native36 market checkpoint required");
  check(torch::cuda::is_available(),"actual CUDA device required");torch::NoGradGuard guard;
  const auto config=r::read_separated_change_settings(settings);const auto &c=config.level;
  check(c.device==torch::Device(torch::kCUDA,0)&&c.channel_count==3&&c.history_length==128&&c.input_width==9&&c.export_width==32&&
    config.change_width==4&&c.sampling_interval==60000.&&r::resolved_channel_ids(c)==std::vector<int64_t>({0,1,2}),"unchanged declared checkpoint geometry");
  auto loaded=r::load_separated_change_checkpoint(checkpoint.string(),c.device,&config);
  check(loaded.training_updates==2048,"exact final trained checkpoint update count");loaded.model->eval();
  const FrozenModel frozen(loaded.model);const auto market_scaler_identity=loaded.scaler.identity();
  const auto source_pins=pin_sources(inputs,checkpoint,settings,card);
  check(fs::create_directories(out),"exclusive new output directory");write_text(out/"sources-before.sha256",ledger(source_pins));
  std::ostringstream report;report<<std::setprecision(17)<<"{\"kind\":\"market_synthetic_transfer_v1\",\"checkpoint_sha256\":"<<json(checkpoint_hash)
    <<",\"settings_sha256\":"<<json(sha256(settings))<<",\"card_sha256\":"<<json(sha256(card))
    <<",\"binary_sha256\":"<<json(sha256(fs::path("/proc")/std::to_string(::getpid())/"exe"))
    <<",\"encoder_updates\":0,\"decoder_calls\":0,\"head_calls\":0,\"CPU_encoder_calls\":0,\"batch_size\":8,\"native_width\":36,\"device\":\"cuda:0\""
    <<",\"parameter_tensor_count\":"<<frozen.parameters.size()<<",\"buffer_tensor_count\":"<<frozen.buffers.size()
    <<",\"endpoint\":7620000,\"sampling_interval\":60000,\"resampling\":false,\"runs\":[";
  bool first=true;int64_t total_calls=0,total_rows=0;double total_seconds=0;
  for(const std::string mode:{"synthetic_train","market_scaler"})for(size_t m=0;m<masters.size();++m){
    const auto name="master-"+std::to_string(masters[m]);
    const auto bank=inputs/name,directory=out/mode/name;
    check(fs::create_directories(directory),"exclusive mode/master output");r::FrozenScaler scaler=loaded.scaler;
    std::string scaler_origin=checkpoint.string();
    if(mode=="synthetic_train"){
      const auto witness=bank/"scaler-witness.pt";torch::serialize::InputArchive a;a.load_from(witness.string(),torch::kCPU);
      scaler=r::FrozenScaler::load(a);scaler.validate(r::configurable_spectral_input_config(c));
      check(scaler.identity()==scaler_ids[m],"exact saved synthetic TRAIN scaler witness");scaler_origin=witness.string();
    }
    const auto scaler_identity=scaler.identity();torch::serialize::OutputArchive scaler_out;scaler.save(scaler_out);
    r::configurable_checkpoint_detail::publish_checkpoint(scaler_out,(directory/"scaler.pt").string());
    torch::serialize::OutputArchive features;std::vector<Capture> captures;
    const std::array<const char*,3> roles{"training","validation","deleted"},files{"training.pt","validation.pt","validation-deleted.pt"};
    for(size_t j=0;j<roles.size();++j){
      captures.push_back(extract(bank/files[j],roles[j],j==0?512:256,loaded.model,config,scaler,features));
      frozen.verify(loaded.model);check(scaler.identity()==scaler_identity&&loaded.scaler.identity()==market_scaler_identity,"frozen scaler identity");
    }
    r::configurable_checkpoint_detail::publish_checkpoint(features,(directory/"features.pt").string());
    if(!first)report<<',';first=false;
    report<<"{\"mode\":"<<json(mode)<<",\"master\":"<<masters[m]<<",\"scaler_identity\":"<<json(scaler_identity)
      <<",\"scaler_origin\":"<<json(scaler_origin)<<",\"scaler_origin_sha256\":"<<json(sha256(scaler_origin))
      <<",\"features_sha256\":"<<json(sha256(directory/"features.pt"))<<",\"roles\":[";
    for(size_t j=0;j<captures.size();++j){const auto &v=captures[j];if(j)report<<',';
      report<<"{\"role\":"<<json(roles[j])<<",\"rows\":"<<v.rows<<",\"valid\":"<<v.valid<<",\"masked_cells\":"<<v.masked_cells
        <<",\"CUDA_encoder_calls\":"<<v.calls<<",\"inference_seconds\":"<<v.seconds<<'}';total_calls+=v.calls;total_rows+=v.rows;total_seconds+=v.seconds;
    }report<<"]}";
  }
  frozen.verify(loaded.model);verify(source_pins);write_text(out/"sources-after.sha256",ledger(source_pins));
  report<<"],\"total_CUDA_encoder_calls\":"<<total_calls<<",\"total_rows\":"<<total_rows<<",\"inference_seconds\":"<<total_seconds
    <<",\"all_parameters_and_buffers_bitwise_unchanged\":true,\"all_parameter_gradients_undefined\":true,\"masked_sentinel_input_invariance\":true"
    <<",\"source_hashes_unchanged\":true,\"scaler_refits\":0,\"labels_passed_to_encoder\":false}\n";
  write_text(out/"summary.json",report.str());std::cout<<"FROZEN_TRANSFER_DIRECTORY="<<out.string()<<'\n';return 0;
}
} // namespace
int main(int argc,char **argv)try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
