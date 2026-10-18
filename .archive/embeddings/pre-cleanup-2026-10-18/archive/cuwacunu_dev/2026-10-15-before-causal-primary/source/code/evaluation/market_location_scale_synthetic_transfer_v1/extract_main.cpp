// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// One declared frozen feature extraction, with no optimizer, decoder or head calls.
#include "embedding/encoders/raw_patch_bottleneck_mae/location_scale_decoder_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/location_scale_decoder_settings.h"
#include <torch/cuda.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <cstring>
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
constexpr const char *checkpoint_hash="c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb";
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
    "code/evaluation/market_location_scale_synthetic_transfer_v1/extract_main.cpp",
    "code/evaluation/market_location_scale_synthetic_transfer_v1/transfer.mk",
    "code/evaluation/market_location_scale_synthetic_transfer_v1/read_main.cpp",
    "code/evaluation/market_location_scale_synthetic_transfer_v1/reader.mk",
    fs::path("/proc")/std::to_string(::getpid())/"exe"};
  for(const auto &root:std::vector<fs::path>{inputs,"code/config/include","code/encoders/rpb_v19/include",
      "code/encoders/rpb_spectral/include","code/encoders/rpb_content/include","code/encoders/rpb_changes/include","code/encoders/rpb_separate/include","code/encoders/rpb_conditioned/include","code/encoders/rpb_relative/include","code/encoders/rpb_location_scale/include"}){
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

// JSON witnesses carry original floating values with round-trip precision.
std::string tensor_json(const torch::Tensor &value){
  const auto t=value.detach().to(torch::kCPU).contiguous();std::ostringstream out;out<<std::setprecision(17);
  const auto scalar=[&](int64_t i){
    if(t.scalar_type()==torch::kBool)out<<(t.const_data_ptr<bool>()[i]?"true":"false");
    else if(t.scalar_type()==torch::kInt64)out<<t.const_data_ptr<int64_t>()[i];
    else{
      check(t.scalar_type()==torch::kFloat32||t.scalar_type()==torch::kFloat64,"JSON witness float dtype");
      const double v=t.scalar_type()==torch::kFloat32?double(t.const_data_ptr<float>()[i]):t.const_data_ptr<double>()[i];
      if(std::isnan(v))out<<json("NaN");else if(std::isinf(v))out<<json(v>0?"+Infinity":"-Infinity");else out<<v;
    }
  };
  std::function<void(int64_t,int64_t)> emit=[&](int64_t dim,int64_t offset){
    if(dim==t.dim()){scalar(offset);return;}
    out<<'[';for(int64_t i=0;i<t.size(dim);++i){if(i)out<<',';emit(dim+1,offset+i*t.stride(dim));}out<<']';
  };emit(0,0);return out.str();
}
std::string vector_json(const std::vector<int64_t> &v){return tensor_json(torch::tensor(v,torch::kInt64));}
std::string content_fingerprint(const torch::Tensor &tensor){
  const auto t=tensor.detach().to(torch::kCPU).contiguous();uint64_t h=14695981039346656037ULL;
  const auto *p=static_cast<const unsigned char *>(t.const_data_ptr());
  for(size_t i=0;i<t.nbytes();++i){h^=p[i];h*=1099511628211ULL;}
  std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return out.str();
}
bool bitwise_equal(const torch::Tensor &a,const torch::Tensor &b){
  if(a.scalar_type()!=b.scalar_type()||a.sizes()!=b.sizes())return false;
  const auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
  return x.nbytes()==0||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.nbytes())==0;
}
std::string scaler_json(const r::FrozenScaler &s){
  std::ostringstream o;o<<std::setprecision(17)<<"{\"identity\":"<<json(s.identity())<<",\"mean\":"<<tensor_json(s.mean)
    <<",\"scale\":"<<tensor_json(s.scale)<<",\"count\":"<<tensor_json(s.count)<<",\"channel_ids\":"<<tensor_json(s.channel_ids)
    <<",\"floor_applied\":"<<tensor_json(s.floor_applied)<<",\"scale_floor\":"<<s.scale_floor<<'}';return o.str();
}
r::FrozenScaler clone_scaler(const r::FrozenScaler &s){
  return {s.mean.clone(),s.scale.clone(),s.count.clone(),s.channel_ids.clone(),s.scale_floor,s.floor_applied.clone()};
}
void same_scaler(const r::FrozenScaler &a,const r::FrozenScaler &b){
  check(a.scale_floor==b.scale_floor&&bitwise_equal(a.mean,b.mean)&&bitwise_equal(a.scale,b.scale)&&bitwise_equal(a.count,b.count)&&
    bitwise_equal(a.channel_ids,b.channel_ids)&&bitwise_equal(a.floor_applied,b.floor_applied)&&a.identity()==b.identity(),"frozen scaler tensors/metadata changed");
}
std::string normalizer_json(const r::RelativeBankStandardizer &n){
  std::ostringstream o;o<<std::setprecision(17)<<"{\"identity\":"<<json(n.identity());
  for(const auto &[name,t]:std::vector<std::pair<std::string,torch::Tensor>>{{"counts",n.counts},{"sum",n.sum},{"sumsq",n.sumsq},{"mean",n.mean},{"scale",n.scale},{"support",n.support}})
    o<<','<<json(name)<<':'<<tensor_json(t);
  o<<",\"input_geometry\":"<<vector_json(n.input_geometry)<<",\"sampling_interval\":"<<n.sampling_interval<<",\"floor\":"<<n.floor
    <<",\"minimum_support\":"<<n.minimum_support<<",\"frozen\":"<<(n.frozen?"true":"false");
  for(const auto &[name,text]:std::vector<std::pair<std::string,std::string>>{{"source_identity",n.source_identity},{"input_scaler_identity",n.input_scaler_identity},
      {"layout_identity",n.layout_identity},{"recipe_identity",n.recipe_identity},{"bank_family",n.bank_family},{"raw_contract",n.raw_contract},{"increment_recipe",n.increment_recipe}})
    o<<','<<json(name)<<':'<<json(text);
  o<<'}';return o.str();
}
struct FrozenValue{std::string name;torch::Tensor value;bool requires_grad;};
struct FrozenModel{
  std::vector<FrozenValue> parameters,buffers;r::RelativeBankStandardizer normalizer;
  explicit FrozenModel(const r::LocationScaleDecoderModel &model):normalizer(model->normalizer().clone()){
    for(const auto &v:model->named_parameters()){
      check(v.value().device()==torch::Device(torch::kCUDA,0)&&v.value().scalar_type()==torch::kFloat32&&!v.value().grad().defined(),"actual CUDA0 F32 parameter without gradient");
      parameters.push_back({v.key(),v.value().detach().to(torch::kCPU).clone(),v.value().requires_grad()});
    }
    for(const auto &v:model->named_buffers()){
      check(v.value().device()==torch::Device(torch::kCUDA,0),"actual CUDA0 buffer");
      buffers.push_back({v.key(),v.value().detach().to(torch::kCPU).clone(),v.value().requires_grad()});
    }
    int64_t count=0;for(const auto &p:parameters)count+=p.value.numel();
    check(parameters.size()==76&&buffers.empty()&&count==297560,"exact V30 parameter/buffer contract");
  }
  void verify(const r::LocationScaleDecoderModel &model)const{
    for(const auto &m:model->modules())check(!m->is_training(),"every module must remain in eval mode");
    const auto p=model->named_parameters(),b=model->named_buffers();
    check(p.size()==parameters.size()&&b.size()==buffers.size(),"model state count changed");
    const auto compare=[](const auto &named,const auto &saved){for(const auto &v:saved){
      const auto *now=named.find(v.name);
      check(now&&now->device()==torch::Device(torch::kCUDA,0)&&now->scalar_type()==v.value.scalar_type()&&now->sizes()==v.value.sizes()&&
        !now->grad().defined()&&now->requires_grad()==v.requires_grad&&bitwise_equal(*now,v.value),"model state/gradient changed: "+v.name);
    }};compare(p,parameters);compare(b,buffers);
    const auto &n=model->normalizer();
    for(const auto &[a,z]:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{n.counts,normalizer.counts},{n.sum,normalizer.sum},{n.sumsq,normalizer.sumsq},
        {n.mean,normalizer.mean},{n.scale,normalizer.scale},{n.support,normalizer.support}})
      check(a.device()==z.device()&&a.scalar_type()==z.scalar_type()&&a.sizes()==z.sizes()&&!a.requires_grad()&&!a.grad().defined()&&bitwise_equal(a,z),"unregistered normalizer tensor changed");
    check(normalizer_json(n)==normalizer_json(normalizer),"unregistered normalizer metadata changed");
    model->validate_state();model->validate_fixed_prior();
  }
};
std::string named_state_json(const torch::OrderedDict<std::string,torch::Tensor> &named){
  std::ostringstream o;o<<'[';bool first=true;
  for(const auto &v:named){if(!first)o<<',';first=false;const auto &t=v.value();
    o<<"{\"name\":"<<json(v.key())<<",\"shape\":"<<vector_json(t.sizes().vec())<<",\"dtype\":"<<json(c10::toString(t.scalar_type()))
      <<",\"device\":"<<json(t.device().str())<<",\"requires_grad\":"<<(t.requires_grad()?"true":"false")
      <<",\"gradient_defined\":"<<(t.grad().defined()?"true":"false")<<",\"content_fnv1a64\":"<<json(content_fingerprint(t))<<'}';
  }o<<']';return o.str();
}
std::string state_json(const r::LocationScaleDecoderModel &model,const r::FrozenScaler &checkpoint_scaler){
  return "{\"parameters\":"+named_state_json(model->named_parameters())+",\"buffers\":"+named_state_json(model->named_buffers())+
    ",\"normalizer\":"+normalizer_json(model->normalizer())+",\"checkpoint_scaler\":"+scaler_json(checkpoint_scaler)+"}";
}

// Labels and source groups are deliberately absent from the encoder-facing record.
struct Observations{torch::Tensor values,mask;};
Observations observations(const fs::path &file,int64_t rows){
  torch::serialize::InputArchive a;a.load_from(file.string(),torch::kCPU);Observations out;
  a.read("observed",out.values,true);a.read("feature_mask",out.mask,true);
  check(out.values.device().is_cpu()&&out.values.scalar_type()==torch::kFloat64&&out.values.sizes()==torch::IntArrayRef({rows,3,128,9}),"exact original synthetic F64 BCHF observations");
  check(out.mask.device().is_cpu()&&out.mask.scalar_type()==torch::kBool&&out.mask.sizes()==out.values.sizes(),"exact synthetic support");
  check(torch::isfinite(out.values.masked_select(out.mask)).all().item<bool>(),"finite visible observations");return out;
}
r::Input input(const Observations &raw,int64_t offset,int64_t count){
  return {raw.values.narrow(0,offset,count),raw.mask.narrow(0,offset,count),torch::tensor({0,1,2},torch::kInt64),
    torch::full({count},127.*60000.,torch::kFloat64),60000.};
}
void copy_scoring_metadata(const fs::path &file,const std::string &role,int64_t rows,torch::serialize::OutputArchive &output){
  // Open scoring-only fields only after this role's embeddings were extracted.
  torch::serialize::InputArchive a;a.load_from(file.string(),torch::kCPU);torch::Tensor labels,ids,ids_json;
  a.read("labels_scoring_only",labels,true);a.read("source_ids",ids,true);a.read("source_ids_json",ids_json,true);
  check(labels.device().is_cpu()&&labels.scalar_type()==torch::kInt64&&labels.sizes()==torch::IntArrayRef({rows}),"scoring-only labels schema");
  check(ids.device().is_cpu()&&ids.scalar_type()==torch::kInt64&&ids.sizes()==torch::IntArrayRef({rows}),"source IDs schema");
  check(ids_json.device().is_cpu()&&ids_json.scalar_type()==torch::kUInt8&&ids_json.dim()==1,"source JSON byte schema");
  output.write(role+"_labels",labels,true);output.write(role+"_source_ids",ids,true);output.write(role+"_source_ids_json",ids_json,true);
}
struct Timings{double full_api{0},external{0},parent{0},io{0},state{0};};
double elapsed(const Clock::time_point &start){return std::chrono::duration<double>(Clock::now()-start).count();}
struct Evidence{
  std::vector<std::string> receipts,cases,parents,states,external_initial,prefixes;Timings time;
  int64_t parent_calls{0};
};
std::string key_json(const std::string &mode,int64_t master,const std::string &role,int64_t offset){
  return "\"mode\":"+json(mode)+",\"master\":"+std::to_string(master)+",\"role\":"+json(role)+",\"offset\":"+std::to_string(offset);
}
void write_array(const fs::path &file,const char *kind,const char *key,const std::vector<std::string> &values){
  std::ostringstream o;o<<"{\"kind\":"<<json(kind)<<','<<json(key)<<":[";
  for(size_t i=0;i<values.size();++i){if(i)o<<',';o<<values[i];}o<<"]}\n";write_text(file,o.str());
}
void check_parent_input(const r::Input &prepared,const r::Input &raw){
  check(prepared.data.is_cuda()&&prepared.data.scalar_type()==torch::kFloat32&&prepared.observed.is_cuda()&&prepared.channel_ids.is_cuda()&&prepared.endpoints.is_cuda(),"actual CUDA parent-facing values/masks/metadata");
  check(torch::isfinite(prepared.data).all().item<bool>()&&!prepared.data.masked_select(prepared.observed.logical_not()).ne(0).any().item<bool>(),"finite parent input with exact hidden zero");
  check(torch::equal(prepared.observed.to(torch::kCPU),raw.observed)&&torch::equal(prepared.channel_ids.to(torch::kCPU),raw.channel_ids)&&
    torch::equal(prepared.endpoints.to(torch::kCPU),raw.endpoints)&&prepared.sampling_interval==raw.sampling_interval,"mask/metadata preserved");
}
std::string bounded_case(const r::Input &raw,const r::Input &legacy,const r::Input &prepared,const r::FrozenScaler &scaler,
    const std::string &mode,int64_t master,const std::string &role){
  const auto data=raw.data.contiguous(),mask=raw.observed.contiguous();
  const auto rounded=torch::where(raw.observed,raw.data,torch::zeros_like(raw.data)).to(torch::kFloat32).contiguous();
  const auto old=legacy.data.to(torch::kCPU).contiguous(),now=prepared.data.to(torch::kCPU).contiguous();
  std::set<int64_t> chosen;
  for(int64_t b=0;b<8;++b)for(int64_t c=0;c<3;++c)for(const int64_t t:{0,127})for(int64_t f=0;f<9;++f)chosen.insert(((b*3+c)*128+t)*9+f);
  int64_t hidden=0;for(int64_t i=0;i<mask.numel()&&hidden<32;++i)if(!mask.const_data_ptr<bool>()[i]){chosen.insert(i);++hidden;}
  std::ostringstream o;o<<std::setprecision(17)<<'{'<<key_json(mode,master,role,0)<<",\"B\":8,\"C\":3,\"H\":128,\"F\":9,\"scaler_identity\":"<<json(scaler.identity())
    <<",\"mean64\":"<<tensor_json(scaler.mean)<<",\"scale64\":"<<tensor_json(scaler.scale)<<",\"channel_ids\":"<<tensor_json(raw.channel_ids)
    <<",\"endpoints\":"<<tensor_json(raw.endpoints)<<",\"sampling_interval\":"<<raw.sampling_interval<<",\"observed_flat\":"<<tensor_json(mask.flatten())<<",\"cells\":[";
  bool first=true;for(const auto i:chosen){if(!first)o<<',';first=false;const int64_t f=i%9,t=i/9%128,c=i/(9*128)%3,b=i/(9*128*3);
    o<<"{\"flat\":"<<i<<",\"b\":"<<b<<",\"c\":"<<c<<",\"t\":"<<t<<",\"f\":"<<f
      <<",\"raw64\":"<<tensor_json(data.flatten()[i])<<",\"observed\":"<<(mask.const_data_ptr<bool>()[i]?"true":"false")
      <<",\"rounded32\":"<<rounded.const_data_ptr<float>()[i]<<",\"legacy_scaled32\":"<<old.const_data_ptr<float>()[i]<<",\"v30_scaled32\":"<<now.const_data_ptr<float>()[i]<<'}';
  }
  o<<"],\"poison_parent_exact\":true,\"poison_bank_values_exact\":true,\"poison_bank_valid_exact\":true,\"poison_conditioned_exact\":true}";return o.str();
}
struct Capture{int64_t calls{0},rows{0},valid{0},masked_cells{0};double seconds{0};};
Capture extract(const fs::path &file,const std::string &mode,int64_t master,const std::string &role,int64_t rows,r::LocationScaleDecoderModel &model,
    const r::LocationScaleDecoderConfig &config,const r::FrozenScaler &scaler,torch::serialize::OutputArchive &output,Evidence &e){
  auto start=Clock::now();const auto raw=observations(file,rows);e.time.io+=elapsed(start);
  std::vector<torch::Tensor> values,valid;Capture result;
  for(int64_t offset=0;offset<rows;offset+=8){
    const auto raw_input=input(raw,offset,8);torch::cuda::synchronize();start=Clock::now();
    const auto prepared=r::relative_change_parent_input(raw_input,scaler,config.level);
    const auto legacy=r::transform_configurable_spectral_input(scaler,raw_input,config.level);check_parent_input(prepared,raw_input);check_parent_input(legacy,raw_input);
    const auto rounded=torch::where(raw_input.observed,raw_input.data,torch::zeros_like(raw_input.data)).to(torch::kFloat32).to(torch::kFloat64);
    const auto raw_delta=(raw_input.data.masked_select(raw_input.observed)-rounded.masked_select(raw_input.observed)).abs();
    const auto norm_delta=(legacy.data.to(torch::kCPU,torch::kFloat64)-prepared.data.to(torch::kCPU,torch::kFloat64)).abs().masked_select(raw_input.observed);
    if(offset==0){
      auto poison=raw_input;poison.data=raw_input.data.clone();
      auto flat=poison.data.flatten();const auto hidden=raw_input.observed.logical_not().flatten();
      auto indices=hidden.nonzero().flatten();const auto count=indices.numel();
      auto bad=torch::full({count},123456789.,torch::kFloat64);
      for(int64_t i=0;i<count;++i){if(i%3==0)bad[i]=std::numeric_limits<double>::quiet_NaN();else if(i%3==1)bad[i]=std::numeric_limits<double>::infinity();}
      flat.index_put_({indices},bad);const auto safe=r::relative_change_parent_input(poison,scaler,config.level);
      check(torch::equal(safe.data,prepared.data)&&torch::equal(safe.observed,prepared.observed)&&torch::equal(safe.channel_ids,prepared.channel_ids)&&
        torch::equal(safe.endpoints,prepared.endpoints)&&safe.sampling_interval==prepared.sampling_interval,"hidden poison changed V30 parent preprocessing");
      const auto a=r::relative_change_targets(raw_input,config.level),b=r::relative_change_targets(poison,config.level);
      check(torch::equal(a.values,b.values)&&torch::equal(a.valid,b.valid)&&torch::equal(model->normalizer().transform(a),model->normalizer().transform(b)),"hidden poison changed relative/conditioned bank");
      e.cases.push_back(bounded_case(raw_input,legacy,prepared,scaler,mode,master,role));
    }
    std::ostringstream receipt;receipt<<std::setprecision(17)<<'{'<<key_json(mode,master,role,offset)<<",\"rows\":8,\"observed_cells\":"<<raw_input.observed.sum().item<int64_t>()
      <<",\"raw_rounding_difference_cells\":"<<raw_delta.ne(0).sum().item<int64_t>()<<",\"max_raw_rounding_delta\":"<<(raw_delta.numel()?raw_delta.max().item<double>():0.)
      <<",\"normalized_difference_cells\":"<<norm_delta.ne(0).sum().item<int64_t>()<<",\"max_normalized_delta\":"<<(norm_delta.numel()?norm_delta.max().item<double>():0.)
      <<",\"hidden_zero\":true,\"mask_metadata_exact\":true,\"poison_checks\":"<<(offset==0?"true":"false")<<'}';e.receipts.push_back(receipt.str());
    torch::cuda::synchronize();e.time.external+=elapsed(start);
    // This is the sole full-model encode for the batch. Its API includes preprocessing/transfers.
    torch::cuda::synchronize();start=Clock::now();const auto encoded=model->encode(raw_input,scaler);const auto native=r::location_scale_decoder_native(encoded,config);
    torch::cuda::synchronize();const auto seconds=elapsed(start);e.time.full_api+=seconds;result.seconds+=seconds;
    check(native.is_cuda()&&!native.requires_grad()&&native.sizes()==torch::IntArrayRef({8,36})&&torch::isfinite(native).all().item<bool>(),"finite frozen CUDA native36");
    check(encoded.sample_valid_mask.is_cuda()&&encoded.sample_valid_mask.scalar_type()==torch::kBool&&encoded.sample_valid_mask.sizes()==torch::IntArrayRef({8}),"CUDA sample validity");
    if(role=="training"&&offset==0){
      torch::cuda::synchronize();start=Clock::now();const auto parent=model->encoder->level->encode(prepared).z_contextual_global;
      check(bitwise_equal(parent,native.narrow(1,0,32)),"same loaded parent/same V30 preprocessing prefix must match exactly");
      torch::cuda::synchronize();e.time.parent+=elapsed(start);++e.parent_calls;
      start=Clock::now();e.parents.push_back("{"+key_json(mode,master,role,offset)+",\"rows\":8,\"full_prefix32\":"+tensor_json(native.narrow(1,0,32))+
        ",\"parent_native32\":"+tensor_json(parent)+",\"exact_equal\":true}");e.time.io+=elapsed(start);
    }
    start=Clock::now();values.push_back(native.to(torch::kCPU,torch::kFloat64));valid.push_back(encoded.sample_valid_mask.to(torch::kCPU));
    result.valid+=encoded.sample_valid_mask.sum().item<int64_t>();result.masked_cells+=raw_input.observed.logical_not().sum().item<int64_t>();
    ++result.calls;result.rows+=8;e.time.io+=elapsed(start);
  }
  start=Clock::now();output.write(role+"_values",torch::cat(values),true);output.write(role+"_valid",torch::cat(valid),true);
  copy_scoring_metadata(file,role,rows,output);e.time.io+=elapsed(start);return result;
}
constexpr const char *old_root="output/runs/market-synthetic-transfer/run-oujlisd9/features";
const std::array<std::array<const char*,3>,2> old_hashes{{
  {"6f8903c256b013b92527a6e6ba07e114beb1b2be86649a44393e65dc1e4644c1","8f4f11ef9aff267c459ca0ec26c7ea64b09e313c4b0336acb3698ae0ff3decd6","4a9d0a890c6ffefc380d2961a963e8bf27fc0c9258b825ecabab545af8f5c6b6"},
  {"e98b21adc447934b21d6495abce10fa24895ac45e5a28326ceae6fcc30bf5d8f","182be75aa83acf685d6e481cf0ab7891b62ec2392e1afaa922715a7699ac551f","8eba65d1b8afc3a634418892a7eb361a406c83c3af9bf11385eb6c52b8d6d0db"}}};
void compare_saved_prefix(const fs::path &current,const fs::path &old,const std::string &mode,int64_t master,Evidence &e){
  torch::serialize::InputArchive a,b;a.load_from(current.string(),torch::kCPU);b.load_from(old.string(),torch::kCPU);
  for(const std::string role:{"training","validation","deleted"}){
    torch::Tensor now,before,valid,old_valid;
    a.read(role+"_values",now,true);b.read(role+"_values",before,true);a.read(role+"_valid",valid,true);b.read(role+"_valid",old_valid,true);
    check(now.scalar_type()==torch::kFloat64&&before.scalar_type()==torch::kFloat64&&now.sizes()==before.sizes()&&now.size(1)==36,"saved native36 comparison schema");
    for(const std::string key:{"_source_ids","_source_ids_json","_labels"}){torch::Tensor x,y;a.read(role+key,x,true);b.read(role+key,y,true);check(torch::equal(x,y),"old/new saved row association changed");}
    const auto delta=(now.narrow(1,0,32)-before.narrow(1,0,32)).abs();std::ostringstream o;o<<std::setprecision(17)<<"{\"mode\":"<<json(mode)<<",\"master\":"<<master
      <<",\"role\":"<<json(role)<<",\"rows\":"<<now.size(0)<<",\"old_features_sha256\":"<<json(sha256(old))<<",\"new_features_sha256\":"<<json(sha256(current))
      <<",\"row_associations_exact\":true,\"validity_exact\":"<<(torch::equal(valid,old_valid)?"true":"false")<<",\"changed_cells\":"<<delta.ne(0).sum().item<int64_t>()
      <<",\"changed_rows\":"<<delta.ne(0).any(1).sum().item<int64_t>()<<",\"max_absolute_delta\":"<<delta.max().item<double>()<<",\"exact_equal\":"<<(delta.eq(0).all().item<bool>()?"true":"false")
      <<",\"descriptive_only\":true}";e.prefixes.push_back(o.str());
  }
}

int execute(int argc,char **argv){
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  check(argc==11,"usage: extract --inputs ROOT --output NEWLEAF --card FILE --checkpoint FILE --settings FILE");std::map<std::string,fs::path> args;
  for(int i=1;i<argc;i+=2)check(args.emplace(argv[i],argv[i+1]).second,"duplicate argument");
  for(const auto &key:{"--inputs","--output","--card","--checkpoint","--settings"})check(args.count(key),"missing argument");
  const auto inputs=args.at("--inputs"),out=args.at("--output"),card=args.at("--card"),checkpoint=args.at("--checkpoint"),settings=args.at("--settings");
  check(fs::is_regular_file(inputs/"manifest.json")&&!fs::exists(out),"input manifest/new output leaf required");
  check(sha256(checkpoint)==checkpoint_hash,"exact final V30 checkpoint required");
  check(sha256(settings)=="43320396447ca80be5fde4324d17c3a9ab129e116bf184455fcb6261aeb6923c"&&
    sha256(card)=="c3cfe0a04b174734599b2b01363c90c3f9b81fd5964fb30ffd4ce8a21976260e"&&
    sha256(inputs/"manifest.json")=="fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb","sealed settings/card/input manifest required");
  check(torch::cuda::is_available(),"actual CUDA device required");torch::NoGradGuard guard;
  const auto config=r::read_location_scale_decoder_settings(settings);const auto &c=config.level;
  check(c.device==torch::Device(torch::kCUDA,0)&&c.channel_count==3&&c.history_length==128&&c.input_width==9&&c.export_width==32&&config.change_width==4&&
    c.sampling_interval==60000.&&r::resolved_channel_ids(c)==std::vector<int64_t>({0,1,2}),"unchanged declared checkpoint geometry");
  Evidence e;const auto admission_started=Clock::now();
  auto loaded=r::load_location_scale_decoder_checkpoint(checkpoint.string(),c.device,&config);
  check(loaded.training_updates==2048,"exact final trained checkpoint update count");loaded.model->eval();
  check(loaded.scaler.identity()=="rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969"&&loaded.model->normalizer().identity()==
    "rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476","exact saved market scaler and bank normalizer");
  const FrozenModel frozen(loaded.model);const auto checkpoint_scaler=clone_scaler(loaded.scaler);frozen.verify(loaded.model);
  auto source_pins=pin_sources(inputs,checkpoint,settings,card);
  size_t mode_index=0;for(const std::string mode:{"synthetic_train","market_scaler"}){
    for(size_t m=0;m<masters.size();++m){const auto path=fs::path(old_root)/mode/("master-"+std::to_string(masters[m]))/"features.pt";
      check(sha256(path)==old_hashes[mode_index][m],"exact saved old native36 archive");source_pins.emplace_back(path,old_hashes[mode_index][m]);}
    ++mode_index;
  }
  check(fs::create_directories(out),"exclusive new output directory");write_text(out/"sources-before.sha256",ledger(source_pins));
  const auto initial_state=state_json(loaded.model,loaded.scaler);e.time.state+=elapsed(admission_started);
  std::ostringstream report;report<<std::setprecision(17)
    <<"{\"kind\":\"market_location_scale_synthetic_transfer_v1\",\"protocol\":\"market-location-scale-synthetic-transfer-v1\",\"model\":\"RPB-v30.market-01\",\"checkpoint_sha256\":"<<json(checkpoint_hash)
    <<",\"settings_sha256\":"<<json(sha256(settings))<<",\"card_sha256\":"<<json(sha256(card))
    <<",\"binary_sha256\":"<<json(sha256(fs::path("/proc")/std::to_string(::getpid())/"exe"))
    <<",\"encoder_updates\":0,\"decoder_calls\":0,\"head_calls\":0,\"CPU_encoder_calls\":0,\"batch_size\":8,\"native_width\":36,\"device\":\"cuda:0\""
    <<",\"parameter_tensor_count\":"<<frozen.parameters.size()<<",\"buffer_tensor_count\":"<<frozen.buffers.size()
    <<",\"checkpoint_scaler_identity\":"<<json(loaded.scaler.identity())<<",\"bank_normalizer_identity\":"<<json(loaded.model->normalizer().identity())
    <<",\"bank_normalizer_fits\":0,\"calibration_encoder_calls\":0,\"calibration_encoder_updates\":0,\"old_encoder_calls\":0"
    <<",\"precision_contract\":\"F32((F64(F32(sanitize(raw64)))-mean64)/scale64); hidden0\",\"old_transfer_precision\":\"F32((raw64-mean64)/scale64); hidden0\""
    <<",\"endpoint\":7620000,\"sampling_interval\":60000,\"resampling\":false,\"runs\":[";
  bool first=true;int64_t total_calls=0,total_rows=0;r::FrozenScaler last_scaler=checkpoint_scaler;
  for(const std::string mode:{"synthetic_train","market_scaler"})for(size_t m=0;m<masters.size();++m){
    const auto name="master-"+std::to_string(masters[m]);const auto bank=inputs/name,directory=out/mode/name;
    check(fs::create_directories(directory),"exclusive mode/master output");auto scaler=clone_scaler(loaded.scaler);std::string scaler_origin=checkpoint.string();
    auto start=Clock::now();
    if(mode=="synthetic_train"){
      const auto witness=bank/"scaler-witness.pt";torch::serialize::InputArchive a;a.load_from(witness.string(),torch::kCPU);
      scaler=r::FrozenScaler::load(a);scaler.validate(r::configurable_spectral_input_config(c));check(scaler.identity()==scaler_ids[m],"exact saved synthetic TRAIN scaler witness");scaler_origin=witness.string();
    }
    const auto original_scaler=clone_scaler(scaler);last_scaler=original_scaler;
    e.external_initial.push_back("{\"mode\":"+json(mode)+",\"master\":"+std::to_string(masters[m])+",\"external_scaler\":"+scaler_json(scaler)+"}");
    torch::serialize::OutputArchive scaler_out;scaler.save(scaler_out);r::configurable_checkpoint_detail::publish_checkpoint(scaler_out,(directory/"scaler.pt").string());e.time.io+=elapsed(start);
    torch::serialize::OutputArchive features;std::vector<Capture> captures;
    const std::array<const char*,3> roles{"training","validation","deleted"},files{"training.pt","validation.pt","validation-deleted.pt"};
    for(size_t j=0;j<roles.size();++j){
      captures.push_back(extract(bank/files[j],mode,masters[m],roles[j],j==0?512:256,loaded.model,config,scaler,features,e));
      start=Clock::now();frozen.verify(loaded.model);same_scaler(scaler,original_scaler);same_scaler(loaded.scaler,checkpoint_scaler);
      auto snapshot=state_json(loaded.model,loaded.scaler);snapshot.pop_back();
      e.states.push_back("{"+key_json(mode,masters[m],roles[j],0)+",\"exact_equal\":true,\"external_scaler\":"+scaler_json(scaler)+","+snapshot.substr(1)+"}");e.time.state+=elapsed(start);
    }
    start=Clock::now();r::configurable_checkpoint_detail::publish_checkpoint(features,(directory/"features.pt").string());e.time.io+=elapsed(start);
    start=Clock::now();compare_saved_prefix(directory/"features.pt",fs::path(old_root)/mode/name/"features.pt",mode,masters[m],e);e.time.io+=elapsed(start);
    if(!first){report<<',';}
    first=false;report<<"{\"mode\":"<<json(mode)<<",\"master\":"<<masters[m]<<",\"scaler_identity\":"<<json(scaler.identity())
      <<",\"scaler_origin\":"<<json(scaler_origin)<<",\"scaler_origin_sha256\":"<<json(sha256(scaler_origin))
      <<",\"features_sha256\":"<<json(sha256(directory/"features.pt"))<<",\"roles\":[";
    for(size_t j=0;j<captures.size();++j){const auto &v=captures[j];if(j)report<<',';
      report<<"{\"role\":"<<json(roles[j])<<",\"rows\":"<<v.rows<<",\"valid\":"<<v.valid<<",\"masked_cells\":"<<v.masked_cells
        <<",\"CUDA_encoder_calls\":"<<v.calls<<",\"inference_seconds\":"<<v.seconds<<",\"full_API_inference_seconds\":"<<v.seconds<<'}';total_calls+=v.calls;total_rows+=v.rows;
    }report<<"]}";
  }
  check(total_calls==768&&total_rows==6144&&e.parent_calls==6&&e.receipts.size()==768&&e.cases.size()==18&&e.parents.size()==6&&e.states.size()==18&&e.prefixes.size()==18,"declared extraction/witness counts");
  auto start=Clock::now();frozen.verify(loaded.model);same_scaler(loaded.scaler,checkpoint_scaler);verify(source_pins);
  auto exit_state=state_json(loaded.model,loaded.scaler);exit_state.pop_back();
  e.states.push_back("{\"mode\":\"exit\",\"master\":0,\"role\":\"exit\",\"exact_equal\":true,\"external_scaler\":"+scaler_json(last_scaler)+","+exit_state.substr(1)+"}");e.time.state+=elapsed(start);
  start=Clock::now();write_text(out/"sources-after.sha256",ledger(source_pins));
  write_array(out/"preprocessing-receipts.json","v30_transfer_preprocessing_receipts_v1","receipts",e.receipts);
  write_array(out/"preprocessing-witnesses.json","v30_transfer_preprocessing_witnesses_v1","cases",e.cases);
  write_array(out/"parent-parity.json","v30_transfer_parent_parity_v1","cases",e.parents);
  write_array(out/"saved-prefix-comparison.json","v30_transfer_saved_prefix_comparison_v1","cases",e.prefixes);
  std::ostringstream state;state<<"{\"kind\":\"v30_transfer_frozen_state_v1\",\"initial\":"<<initial_state<<",\"external_initial\":[";
  for(size_t i=0;i<e.external_initial.size();++i){if(i)state<<',';state<<e.external_initial[i];}state<<"],\"checks\":[";
  for(size_t i=0;i<e.states.size();++i){if(i)state<<',';state<<e.states[i];}state<<"]}\n";write_text(out/"frozen-state.json",state.str());e.time.io+=elapsed(start);
  report<<"],\"full_model_encode_calls\":"<<total_calls<<",\"total_CUDA_encoder_calls\":"<<total_calls<<",\"total_rows\":"<<total_rows
    <<",\"parent_correctness_calls\":"<<e.parent_calls<<",\"parent_correctness_rows\":48,\"total_parent_forwards\":774,\"preprocessing_receipt_count\":768,\"preprocessing_witness_case_count\":18"
    <<",\"full_API_inference_seconds\":"<<e.time.full_api<<",\"inference_seconds\":"<<e.time.full_api<<",\"external_preprocessing_seconds\":"<<e.time.external
    <<",\"parent_correctness_seconds\":"<<e.time.parent<<",\"feature_and_receipt_io_seconds\":"<<e.time.io<<",\"state_verification_seconds\":"<<e.time.state
    <<",\"all_parameters_and_buffers_bitwise_unchanged\":true,\"all_parameter_gradients_undefined\":true,\"normalizer_unchanged\":true,\"scalers_unchanged\":true"
    <<",\"masked_sentinel_input_invariance\":true,\"hidden_poison_preprocessing_checks\":18,\"same_preprocessing_parent_prefix_exact\":true"
    <<",\"source_hashes_unchanged\":true,\"scaler_refits\":0,\"labels_passed_to_encoder\":false}\n";
  write_text(out/"summary.json",report.str());std::cout<<"FROZEN_TRANSFER_DIRECTORY="<<out.string()<<'\n';return 0;
}
} // namespace
int main(int argc,char **argv)try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}