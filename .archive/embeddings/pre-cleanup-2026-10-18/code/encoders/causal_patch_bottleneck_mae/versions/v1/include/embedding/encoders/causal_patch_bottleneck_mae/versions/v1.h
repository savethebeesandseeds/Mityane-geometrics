// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// The single public CPB-v1 entry. Version-private code is owned by this family.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v1/location_scale_decoder_checkpoint.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v1/location_scale_decoder_settings.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v1/location_scale_decoder_training_state.h"
#include <array>
#include <cstring>
#include <iterator>

namespace embedding::encoders::causal_patch_bottleneck_mae::v1 {
namespace implementation=embedding::encoders::causal_patch_bottleneck_mae::detail::v1;
using Config=implementation::LocationScaleDecoderConfig;
using FrozenScaler=implementation::FrozenScaler;
using BankStandardizer=implementation::RelativeBankStandardizer;
inline constexpr const char *kModelTag="CPB-v1";
inline constexpr const char *kCheckpointKind="cpb_v1_inference_checkpoint_v1";
inline constexpr const char *kCausalPolicy=
    "crop-before-validation;rolling-last-H;masked-left-pad;original-patch-end-causal-attention;prefix-only-banks-pooling-v1";
inline constexpr const char *kSourceCheckpointSha256="c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb";
inline constexpr const char *kSourceConfigSha256="43320396447ca80be5fde4324d17c3a9ab129e116bf184455fcb6261aeb6923c";
inline constexpr const char *kInputScalerIdentity="rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969";
inline constexpr const char *kBankIdentity="rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476";

struct RawHistory {
  torch::Tensor data,observed,channel_ids,timestamps;
  double sampling_interval{60000.};
};
struct Encoding {torch::Tensor native,valid,cutoffs;};
struct CausalTrace : implementation::TemporalExecutionTrace {
  Encoding encoding;
  torch::Tensor retained_raw,retained_observed,retained_timestamps,normalized_parent,token_end_times;
};
inline Config read_settings(const std::filesystem::path &path){return implementation::read_location_scale_decoder_settings(path);}
inline auto input_config(const Config &c){return implementation::configurable_spectral_input_config(c.level);}

namespace boundary {
struct Window {implementation::Input raw;torch::Tensor times,cutoffs;};
inline Window retain(const RawHistory &history,const torch::Tensor &requested,const Config &config){
  const auto &c=config.level;implementation::validate_location_scale_decoder_config(config);
  TORCH_CHECK(history.data.defined() && history.data.device().is_cpu() && history.data.scalar_type()==torch::kFloat64 &&
      history.data.dim()==4 && history.data.size(0)>0 && history.data.size(1)==c.channel_count &&
      history.data.size(2)>0 && history.data.size(3)==c.input_width,"[cpb-v1] CPU raw F64 history B,C,N,F required");
  TORCH_CHECK(history.observed.defined() && history.observed.device().is_cpu() &&
      history.observed.scalar_type()==torch::kBool && history.observed.sizes()==history.data.sizes(),"[cpb-v1] matching CPU bool history mask required");
  const auto B=history.data.size(0),N=history.data.size(2),H=c.history_length;
  TORCH_CHECK(history.channel_ids.defined() && history.channel_ids.device().is_cpu() &&
      history.channel_ids.scalar_type()==torch::kInt64,"[cpb-v1] CPU semantic channel IDs required");
  (void)implementation::channel_indices(history.channel_ids,c,B,torch::kCPU);
  TORCH_CHECK(std::isfinite(history.sampling_interval) && history.sampling_interval==c.sampling_interval,
      "[cpb-v1] declared sampling grid must match config");
  TORCH_CHECK(history.timestamps.defined() && history.timestamps.device().is_cpu() &&
      history.timestamps.scalar_type()==torch::kFloat64 && ((history.timestamps.dim()==1 && history.timestamps.size(0)==N) ||
      (history.timestamps.dim()==2 && history.timestamps.sizes()==torch::IntArrayRef({B,N}))),"[cpb-v1] CPU F64 timestamp grid N or B,N required");
  TORCH_CHECK(requested.defined() && requested.device().is_cpu() && requested.scalar_type()==torch::kFloat64 &&
      requested.sizes()==torch::IntArrayRef({B}) && torch::isfinite(requested).all().item<bool>(),"[cpb-v1] explicit CPU F64 cutoff per example required");
  const auto times=(history.timestamps.dim()==1?history.timestamps.unsqueeze(0).expand({B,N}):history.timestamps).contiguous();
  TORCH_CHECK(torch::isfinite(times).all().item<bool>(),"[cpb-v1] finite declared timestamp schema required");
  const double tolerance=c.sampling_interval*1e-9;
  std::vector<torch::Tensor> values,masks,window_times;
  for(int64_t b=0;b<B;++b){
    const auto grid=times[b];
    if(N>1){const auto delta=grid.narrow(0,1,N-1)-grid.narrow(0,0,N-1);
      TORCH_CHECK(delta.gt(0).all().item<bool>() && torch::allclose(delta,torch::full_like(delta,c.sampling_interval),1e-9,tolerance),
          "[cpb-v1] timestamp grid must be ordered and uniform");}
    const double cutoff=requested[b].item<double>(),first=grid[0].item<double>(),last=grid[N-1].item<double>();
    TORCH_CHECK(cutoff>=first && cutoff<=last,"[cpb-v1] cutoff outside supplied grid");
    const double coordinate=(cutoff-first)/c.sampling_interval;
    TORCH_CHECK(coordinate>=0 && coordinate<=double(N-1),"[cpb-v1] cutoff coordinate outside grid");
    const auto end=static_cast<int64_t>(std::llround(coordinate));
    TORCH_CHECK(end>=0 && end<N && std::abs(grid[end].item<double>()-cutoff)<=tolerance,
        "[cpb-v1] cutoff must coincide with an explicitly supplied timestamp");
    const auto begin=std::max<int64_t>(0,end-H+1),count=end-begin+1,pad=H-count;
    // This slice is the first operation that reads observation values/masks.
    // In particular no finite-value check, cast or support reduction sees suffix cells.
    const auto retained=history.data[b].narrow(1,begin,count);
    const auto observed=history.observed[b].narrow(1,begin,count);
    const auto safe=torch::where(observed,retained,torch::zeros_like(retained));
    values.push_back(torch::cat({torch::zeros({c.channel_count,pad,c.input_width},retained.options()),safe},1));
    masks.push_back(torch::cat({torch::zeros({c.channel_count,pad,c.input_width},observed.options()),observed},1));
    window_times.push_back(torch::arange(1-H,1,torch::TensorOptions().dtype(torch::kFloat64))*c.sampling_interval+cutoff);
  }
  implementation::Input raw{torch::stack(values),torch::stack(masks),history.channel_ids,requested.clone(),history.sampling_interval};
  implementation::validate_relative_change_raw(raw,c);
  return{raw,torch::stack(window_times),requested.clone()};
}

// Standalone SHA256 over checkpoint bytes; Torch/STL are the only dependencies.
inline std::string sha256_file(const std::filesystem::path &path){
  std::ifstream stream(path,std::ios::binary);TORCH_CHECK(stream,"[cpb-v1] cannot open source checkpoint");
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(stream)),{});
  TORCH_CHECK(!stream.bad() && data.size()<=std::numeric_limits<uint64_t>::max()/8,"[cpb-v1] source checkpoint read failed");
  const uint64_t bits=uint64_t(data.size())*8;data.push_back(0x80);
  while(data.size()%64!=56)data.push_back(0);
  for(int shift=56;shift>=0;shift-=8)data.push_back(uint8_t(bits>>shift));
  constexpr std::array<uint32_t,64> k{{0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
      0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
      0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
      0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
      0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
      0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
      0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
      0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U}};
  std::array<uint32_t,8> h{{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U}};
  const auto rotate=[](uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));};
  for(size_t block=0;block<data.size();block+=64){
    std::array<uint32_t,64> w{};for(size_t i=0;i<16;++i){const auto j=block+4*i;w[i]=(uint32_t(data[j])<<24)|(uint32_t(data[j+1])<<16)|(uint32_t(data[j+2])<<8)|uint32_t(data[j+3]);}
    for(size_t i=16;i<64;++i){const auto x=w[i-15],y=w[i-2];w[i]=w[i-16]+(rotate(x,7)^rotate(x,18)^(x>>3))+w[i-7]+(rotate(y,17)^rotate(y,19)^(y>>10));}
    auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
    for(size_t i=0;i<64;++i){const auto t1=z+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+k[i]+w[i];
      const auto t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
  }
  std::ostringstream out;out<<std::hex<<std::setfill('0');for(const auto value:h)out<<std::setw(8)<<value;return out.str();
}
inline bool same_bits(const torch::Tensor &a,const torch::Tensor &b){
  if(a.scalar_type()!=b.scalar_type() || a.sizes()!=b.sizes())return false;
  const auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
  return x.nbytes()==0 || std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.nbytes())==0;
}
inline std::string state_identity(const implementation::LocationScaleDecoderModel &core){
  uint64_t hash=14695981039346656037ULL;
  const auto bytes=[&](const void *pointer,size_t size){const auto *p=static_cast<const unsigned char*>(pointer);for(size_t i=0;i<size;++i){hash^=p[i];hash*=1099511628211ULL;}};
  const auto named=[&](const auto &items){for(const auto &item:items){bytes(item.key().data(),item.key().size());const auto value=item.value().detach().to(torch::kCPU).contiguous();
    const auto type=int64_t(value.scalar_type());bytes(&type,sizeof(type));for(const auto n:value.sizes())bytes(&n,sizeof(n));bytes(value.const_data_ptr(),value.nbytes());}};
  named(core->named_parameters());named(core->named_buffers());const auto normalizer=core->normalizer().identity();bytes(normalizer.data(),normalizer.size());
  std::ostringstream out;out<<"cpb-v1-state-fnv1a64-"<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();
}
} // namespace boundary

class Encoder;
struct EncoderImpl : torch::nn::Module {
  explicit EncoderImpl(implementation::LocationScaleDecoderModel core)
      : core_(register_module("core",core)){
    TORCH_CHECK(core_,"[cpb-v1] owned model state required");imported_model_identity_=boundary::state_identity(core_);eval();
  }
  const Config &config()const{return core_->config();}
  const BankStandardizer &normalizer()const{return core_->normalizer();}
  int64_t source_updates()const{return 2048;}
  int64_t causal_updates()const{return 0;}
  std::string source_checkpoint_sha256()const{return kSourceCheckpointSha256;}
  const std::string &imported_model_identity()const{return imported_model_identity_;}
  std::string state_identity()const{return boundary::state_identity(core_);}
  void validate_frozen()const{
    core_->validate_state();core_->validate_fixed_prior();
    for(const auto &module:modules())TORCH_CHECK(!module->is_training(),"[cpb-v1] frozen eval modules required");
    for(const auto &value:named_parameters())TORCH_CHECK(!value.value().grad().defined(),"[cpb-v1] gradients forbidden");
    TORCH_CHECK(normalizer().identity()==kBankIdentity && state_identity()==imported_model_identity_,"[cpb-v1] imported frozen state changed");
  }
  Encoding encode_at(const RawHistory &raw,const torch::Tensor &cutoffs,const FrozenScaler &scaler){
    torch::NoGradGuard guard;TORCH_CHECK(!is_training(),"[cpb-v1] eval required");
    const auto window=boundary::retain(raw,cutoffs,config());
    const auto encoded=core_->encode(window.raw,scaler);
    return{encoded.z_contextual_global,encoded.sample_valid_mask,window.cutoffs};
  }
  Encoding encode_at(const RawHistory &raw,double cutoff,const FrozenScaler &scaler){
    return encode_at(raw,torch::full({raw.data.size(0)},cutoff,torch::kFloat64),scaler);
  }
  CausalTrace trace_at(const RawHistory &raw,const torch::Tensor &cutoffs,const FrozenScaler &scaler){
    torch::NoGradGuard guard;TORCH_CHECK(!is_training(),"[cpb-v1] eval required");
    const auto window=boundary::retain(raw,cutoffs,config());CausalTrace out;
    const auto encoded=core_->encode(window.raw,scaler,&out);
    out.encoding={encoded.z_contextual_global,encoded.sample_valid_mask,window.cutoffs};
    out.retained_raw=window.raw.data;out.retained_observed=window.raw.observed;out.retained_timestamps=window.times;
    out.normalized_parent=implementation::relative_change_parent_input(window.raw,scaler,config().level).data;
    const auto &c=config().level;
    if(out.patch_rows.numel()){
      const auto batches=torch::floor_divide(out.patch_rows,c.channel_count);
      const auto indices=(out.patch_positions.clamp_min(0)*c.patch_length+c.patch_length-1);
      const auto grid=window.times.to(c.device).index_select(0,batches);
      out.token_end_times=torch::where(out.patch_valid,grid.gather(1,indices),torch::zeros_like(indices).to(torch::kFloat64));
    }else out.token_end_times=torch::zeros(out.patch_positions.sizes(),torch::TensorOptions().device(c.device).dtype(torch::kFloat64));
    return out;
  }
  CausalTrace trace_at(const RawHistory &raw,double cutoff,const FrozenScaler &scaler){
    return trace_at(raw,torch::full({raw.data.size(0)},cutoff,torch::kFloat64),scaler);
  }
private:
  implementation::LocationScaleDecoderModel core_{nullptr};
  std::string imported_model_identity_;
  friend void save_checkpoint(const std::string &,const Encoder &,const FrozenScaler &);
};
TORCH_MODULE(Encoder);
struct Loaded {
  Config config;Encoder model{nullptr};FrozenScaler scaler;
  int64_t source_updates{2048},causal_updates{0};std::string source_checkpoint_sha256{kSourceCheckpointSha256};
};

inline Loaded import_v30_checkpoint(const std::string &path,torch::Device device=torch::Device(torch::kCUDA,0),const Config *expected=nullptr){
  TORCH_CHECK(boundary::sha256_file(path)==kSourceCheckpointSha256,"[cpb-v1] only the pinned V30 data artifact may be imported");
  auto copied=implementation::load_location_scale_decoder_checkpoint(path,device,expected);
  TORCH_CHECK(copied.training_updates==2048 && copied.scaler.identity()==kInputScalerIdentity && copied.model->normalizer().identity()==kBankIdentity,
      "[cpb-v1] source update/scaler/normalizer mismatch");
  // Re-read the archive as tensors, not as an old model. Every tensor copied
  // into the independently owned model must equal the source bytes exactly.
  torch::serialize::InputArchive source;source.load_from(path,device);torch::serialize::InputArchive source_model;source.read("model",source_model);
  const auto check_named=[&](const auto &named,bool buffer){for(const auto &item:named){
    torch::serialize::InputArchive *cursor=&source_model;std::vector<std::unique_ptr<torch::serialize::InputArchive>> levels;
    const auto &name=item.key();size_t begin=0,end;
    while((end=name.find('.',begin))!=std::string::npos){auto child=std::make_unique<torch::serialize::InputArchive>();cursor->read(name.substr(begin,end-begin),*child);cursor=child.get();levels.push_back(std::move(child));begin=end+1;}
    torch::Tensor value;cursor->read(name.substr(begin),value,buffer);TORCH_CHECK(boundary::same_bits(value,item.value()),"[cpb-v1] imported parameter/buffer bytes differ: ",name);
  }};
  check_named(copied.model->named_parameters(),false);check_named(copied.model->named_buffers(),true);
  const auto tensor_equal=[](torch::serialize::InputArchive &archive,const char *key,const torch::Tensor &expected_value){
    torch::Tensor value;archive.read(key,value,true);TORCH_CHECK(boundary::same_bits(value,expected_value),"[cpb-v1] imported scaler/normalizer bytes differ: ",key);};
  torch::serialize::InputArchive source_scaler,source_normalizer,source_bank;
  source.read("scaler",source_scaler);source.read("bank_standardizer",source_normalizer);source_normalizer.read("bank",source_bank);
  for(const auto &[key,value]:std::vector<std::pair<const char*,torch::Tensor>>{{"mean",copied.scaler.mean},{"scale",copied.scaler.scale},
      {"count",copied.scaler.count},{"channel_ids",copied.scaler.channel_ids},{"floor_applied",copied.scaler.floor_applied},
      {"scale_floor",torch::tensor(copied.scaler.scale_floor,torch::kFloat64)}})tensor_equal(source_scaler,key,value);
  const auto &normalizer=copied.model->normalizer();
  for(const auto &[key,value]:std::vector<std::pair<const char*,torch::Tensor>>{{"counts",normalizer.counts},{"sum",normalizer.sum},{"sumsq",normalizer.sumsq},
      {"mean",normalizer.mean},{"scale",normalizer.scale},{"support",normalizer.support},{"input_geometry",torch::tensor(normalizer.input_geometry,torch::kInt64)},
      {"sampling_interval",torch::tensor(normalizer.sampling_interval,torch::kFloat64)},{"floor",torch::tensor(normalizer.floor,torch::kFloat64)},
      {"minimum_support",torch::tensor(normalizer.minimum_support,torch::kInt64)}})tensor_equal(source_bank,key,value);
  namespace d=implementation::configurable_checkpoint_detail;
  for(const auto &[key,value]:std::vector<std::pair<const char*,std::string>>{{"source_identity",normalizer.source_identity},{"input_scaler_identity",normalizer.input_scaler_identity},
      {"layout_identity",normalizer.layout_identity},{"recipe_identity",normalizer.recipe_identity}})
    TORCH_CHECK(d::read_text(source_bank,key)==value,"[cpb-v1] imported normalizer metadata differs: ",key);
  for(const auto &[key,value]:std::vector<std::pair<const char*,std::string>>{{"bank_family",normalizer.bank_family},{"raw_contract",normalizer.raw_contract},{"increment_recipe",normalizer.increment_recipe}})
    TORCH_CHECK(d::read_text(source_normalizer,key)==value,"[cpb-v1] imported bank family metadata differs: ",key);
  TORCH_CHECK(boundary::sha256_file(path)==kSourceCheckpointSha256,"[cpb-v1] source changed during import");
  auto model=Encoder(copied.model);model->validate_frozen();return{copied.config,model,copied.scaler,2048,0,kSourceCheckpointSha256};
}

inline void save_checkpoint(const std::string &path,const Encoder &model,const FrozenScaler &scaler){
  namespace d=implementation::configurable_checkpoint_detail;
  TORCH_CHECK(model && scaler.identity()==kInputScalerIdentity,"[cpb-v1] original checkpoint scaler required");model->validate_frozen();
  torch::serialize::OutputArchive archive,payload;
  archive.write("artifact_kind",d::text_tensor(kCheckpointKind),true);archive.write("model_tag",d::text_tensor(kModelTag),true);
  archive.write("causal_policy",d::text_tensor(kCausalPolicy),true);archive.write("source_checkpoint_sha256",d::text_tensor(kSourceCheckpointSha256),true);
  archive.write("source_config_sha256",d::text_tensor(kSourceConfigSha256),true);
  archive.write("source_updates",torch::tensor(int64_t{2048},torch::kInt64),true);archive.write("causal_updates",torch::tensor(int64_t{0},torch::kInt64),true);
  archive.write("scaler_identity",d::text_tensor(scaler.identity()),true);archive.write("normalizer_identity",d::text_tensor(model->normalizer().identity()),true);
  archive.write("imported_model_identity",d::text_tensor(model->imported_model_identity()),true);
  archive.write("normalizer_source_identity",d::text_tensor(model->normalizer().source_identity),true);
  archive.write("config_integer_v1",torch::tensor(d::integers(model->config().level),torch::kInt64),true);
  archive.write("config_number_v1",torch::tensor(d::numbers(model->config().level),torch::kFloat64),true);
  archive.write("channel_ids",torch::tensor(implementation::resolved_channel_ids(model->config().level),torch::kInt64),true);
  archive.write("change_width",torch::tensor(model->config().change_width,torch::kInt64),true);
  archive.write("native_width",torch::tensor(implementation::location_scale_decoder_native_width(model->config()),torch::kInt64),true);
  implementation::location_scale_decoder_checkpoint_detail::write(payload,model->core_,scaler,2048);
  archive.write("payload",payload);d::publish_checkpoint(archive,path);
}
inline Loaded load_checkpoint(const std::string &path,torch::Device device=torch::Device(torch::kCUDA,0),const Config *expected=nullptr){
  namespace d=implementation::configurable_checkpoint_detail;
  torch::serialize::InputArchive archive;archive.load_from(path,device);
  TORCH_CHECK(d::read_text(archive,"artifact_kind")==kCheckpointKind && d::read_text(archive,"model_tag")==kModelTag &&
      d::read_text(archive,"causal_policy")==kCausalPolicy && d::read_text(archive,"source_checkpoint_sha256")==kSourceCheckpointSha256 &&
      d::read_text(archive,"source_config_sha256")==kSourceConfigSha256 &&
      implementation::separated_change_checkpoint_detail::integer(archive,"source_updates")==2048 &&
      implementation::separated_change_checkpoint_detail::integer(archive,"causal_updates")==0 &&
      d::read_text(archive,"scaler_identity")==kInputScalerIdentity && d::read_text(archive,"normalizer_identity")==kBankIdentity,
      "[cpb-v1] typed causal version/policy/provenance mismatch");
  const auto expected_state=d::read_text(archive,"imported_model_identity");torch::serialize::InputArchive payload;archive.read("payload",payload);
  const std::string scaler_id=kInputScalerIdentity,normalizer_id=kBankIdentity;
  auto copied=implementation::location_scale_decoder_checkpoint_detail::read(payload,device,expected,&scaler_id,nullptr,&normalizer_id);
  const auto &c=copied.config;
  TORCH_CHECK(boundary::same_bits(d::read_vector(archive,"config_integer_v1",torch::kInt64,d::integers(c.level).size()),torch::tensor(d::integers(c.level),torch::kInt64)) &&
      boundary::same_bits(d::read_vector(archive,"config_number_v1",torch::kFloat64,d::numbers(c.level).size()),torch::tensor(d::numbers(c.level),torch::kFloat64)) &&
      boundary::same_bits(d::read_vector(archive,"channel_ids",torch::kInt64,c.level.channel_count),torch::tensor(implementation::resolved_channel_ids(c.level),torch::kInt64)) &&
      implementation::separated_change_checkpoint_detail::integer(archive,"change_width")==c.change_width &&
      implementation::separated_change_checkpoint_detail::integer(archive,"native_width")==implementation::location_scale_decoder_native_width(c) &&
      d::read_text(archive,"normalizer_source_identity")==copied.model->normalizer().source_identity,"[cpb-v1] outer config/normalizer binding mismatch");
  TORCH_CHECK(copied.training_updates==2048,"[cpb-v1] nested source updates mismatch");auto model=Encoder(copied.model);
  TORCH_CHECK(model->state_identity()==expected_state,"[cpb-v1] saved imported-state identity mismatch");model->validate_frozen();
  return{copied.config,model,copied.scaler,2048,0,kSourceCheckpointSha256};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::v1
