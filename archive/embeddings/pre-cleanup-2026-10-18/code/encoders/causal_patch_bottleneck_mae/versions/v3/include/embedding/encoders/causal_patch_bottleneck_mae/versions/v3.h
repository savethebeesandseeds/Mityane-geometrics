// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// The single public CPB-v3 entry. Version-private code is owned by this family.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/location_scale_decoder_checkpoint.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/location_scale_decoder_settings.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/location_scale_decoder_training_state.h"
#include <array>
#include <cstring>
#include <iterator>

namespace embedding::encoders::causal_patch_bottleneck_mae::v3 {
namespace implementation=embedding::encoders::causal_patch_bottleneck_mae::detail::v3;
using Config=implementation::LocationScaleDecoderConfig;
using FrozenScaler=implementation::FrozenScaler;
using BankStandardizer=implementation::RelativeBankStandardizer;
inline constexpr const char *kModelTag="CPB-v3";
inline constexpr const char *kCheckpointKind="cpb_v3_inference_checkpoint_v1";
inline constexpr const char *kCausalPolicy=
    "crop-before-validation;rolling-last-H;masked-left-pad;original-patch-end-causal-attention;prefix-only-banks-pooling-v1";
inline constexpr const char *kInitializationKind="fresh-seeded-constructor-no-import-v1";

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
      history.data.size(2)>0 && history.data.size(3)==c.input_width,"[cpb-v3] CPU raw F64 history B,C,N,F required");
  TORCH_CHECK(history.observed.defined() && history.observed.device().is_cpu() &&
      history.observed.scalar_type()==torch::kBool && history.observed.sizes()==history.data.sizes(),"[cpb-v3] matching CPU bool history mask required");
  const auto B=history.data.size(0),N=history.data.size(2),H=c.history_length;
  TORCH_CHECK(history.channel_ids.defined() && history.channel_ids.device().is_cpu() &&
      history.channel_ids.scalar_type()==torch::kInt64,"[cpb-v3] CPU semantic channel IDs required");
  (void)implementation::channel_indices(history.channel_ids,c,B,torch::kCPU);
  TORCH_CHECK(std::isfinite(history.sampling_interval) && history.sampling_interval==c.sampling_interval,
      "[cpb-v3] declared sampling grid must match config");
  TORCH_CHECK(history.timestamps.defined() && history.timestamps.device().is_cpu() &&
      history.timestamps.scalar_type()==torch::kFloat64 && ((history.timestamps.dim()==1 && history.timestamps.size(0)==N) ||
      (history.timestamps.dim()==2 && history.timestamps.sizes()==torch::IntArrayRef({B,N}))),"[cpb-v3] CPU F64 timestamp grid N or B,N required");
  TORCH_CHECK(requested.defined() && requested.device().is_cpu() && requested.scalar_type()==torch::kFloat64 &&
      requested.sizes()==torch::IntArrayRef({B}) && torch::isfinite(requested).all().item<bool>(),"[cpb-v3] explicit CPU F64 cutoff per example required");
  const auto times=(history.timestamps.dim()==1?history.timestamps.unsqueeze(0).expand({B,N}):history.timestamps).contiguous();
  TORCH_CHECK(torch::isfinite(times).all().item<bool>(),"[cpb-v3] finite declared timestamp schema required");
  const double tolerance=c.sampling_interval*1e-9;
  std::vector<torch::Tensor> values,masks,window_times;
  for(int64_t b=0;b<B;++b){
    const auto grid=times[b];
    if(N>1){const auto delta=grid.narrow(0,1,N-1)-grid.narrow(0,0,N-1);
      TORCH_CHECK(delta.gt(0).all().item<bool>() && torch::allclose(delta,torch::full_like(delta,c.sampling_interval),1e-9,tolerance),
          "[cpb-v3] timestamp grid must be ordered and uniform");}
    const double cutoff=requested[b].item<double>(),first=grid[0].item<double>(),last=grid[N-1].item<double>();
    TORCH_CHECK(cutoff>=first && cutoff<=last,"[cpb-v3] cutoff outside supplied grid");
    const double coordinate=(cutoff-first)/c.sampling_interval;
    TORCH_CHECK(coordinate>=0 && coordinate<=double(N-1),"[cpb-v3] cutoff coordinate outside grid");
    const auto end=static_cast<int64_t>(std::llround(coordinate));
    TORCH_CHECK(end>=0 && end<N && std::abs(grid[end].item<double>()-cutoff)<=tolerance,
        "[cpb-v3] cutoff must coincide with an explicitly supplied timestamp");
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
  std::ifstream stream(path,std::ios::binary);TORCH_CHECK(stream,"[cpb-v3] cannot open source checkpoint");
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(stream)),{});
  TORCH_CHECK(!stream.bad() && data.size()<=std::numeric_limits<uint64_t>::max()/8,"[cpb-v3] source checkpoint read failed");
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
  std::ostringstream out;out<<"cpb-v3-state-fnv1a64-"<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();
}
} // namespace boundary

class Encoder;
struct EncoderImpl : torch::nn::Module {
  explicit EncoderImpl(implementation::LocationScaleDecoderModel core,int64_t seed,const std::string &scaler_id,int64_t updates=0,const std::string &initial_state={})
      :core_(register_module("core",core)),causal_updates_(updates),initialization_seed_(seed),scaler_identity_(scaler_id){
    TORCH_CHECK(core_&&updates>=0&&seed>=0&&!scaler_id.empty(),"[cpb-v3] fresh model/seed/scaler and nonnegative causal updates required");
    normalizer_identity_=core_->normalizer().identity();initial_state_identity_=initial_state.empty()?boundary::state_identity(core_):initial_state;eval();
  }
  const Config &config()const{return core_->config();}
  const BankStandardizer &normalizer()const{return core_->normalizer();}
  int64_t source_updates()const{return 0;}
  int64_t causal_updates()const{return causal_updates_;}
  void set_causal_updates(int64_t count){TORCH_CHECK(count>=0,"[cpb-v3] nonnegative updates required");causal_updates_=count;}
  int64_t initialization_seed()const{return initialization_seed_;}
  const std::string &input_scaler_identity()const{return scaler_identity_;}
  const std::string &initial_state_identity()const{return initial_state_identity_;}
  std::string state_identity()const{return boundary::state_identity(core_);}
  implementation::LocationScaleDecoderModel &core(){return core_;}
  const implementation::LocationScaleDecoderModel &core()const{return core_;}
  void validate_frozen()const{
    core_->validate_state();core_->validate_fixed_prior();
    for(const auto &module:modules())TORCH_CHECK(!module->is_training(),"[cpb-v3] eval modules required");
    for(const auto &value:named_parameters())TORCH_CHECK(!value.value().grad().defined(),"[cpb-v3] clear training gradients before inference/checkpoint");
    TORCH_CHECK(normalizer().identity()==normalizer_identity_,"[cpb-v3] frozen bank changed");
  }
  void clear_gradients(){for(auto &value:parameters())value.mutable_grad()=torch::Tensor();}
  Encoding encode_at(const RawHistory &raw,const torch::Tensor &cutoffs,const FrozenScaler &scaler){
    torch::NoGradGuard guard;TORCH_CHECK(!is_training(),"[cpb-v3] eval required");
    const auto window=boundary::retain(raw,cutoffs,config());const auto out=core_->encode(window.raw,scaler);
    return{out.z_contextual_global,out.sample_valid_mask,window.cutoffs};
  }
  Encoding encode_at(const RawHistory &raw,double cutoff,const FrozenScaler &scaler){return encode_at(raw,torch::full({raw.data.size(0)},cutoff,torch::kFloat64),scaler);}
  CausalTrace trace_at(const RawHistory &raw,const torch::Tensor &cutoffs,const FrozenScaler &scaler){
    torch::NoGradGuard guard;TORCH_CHECK(!is_training(),"[cpb-v3] eval required");
    const auto window=boundary::retain(raw,cutoffs,config());CausalTrace out;const auto encoded=core_->encode(window.raw,scaler,&out);
    out.encoding={encoded.z_contextual_global,encoded.sample_valid_mask,window.cutoffs};
    out.retained_raw=window.raw.data;out.retained_observed=window.raw.observed;out.retained_timestamps=window.times;
    out.normalized_parent=implementation::relative_change_parent_input(window.raw,scaler,config().level).data;
    const auto &c=config().level;
    if(out.patch_rows.numel()){
      const auto batches=torch::floor_divide(out.patch_rows,c.channel_count);
      const auto indices=out.patch_positions.clamp_min(0)*c.patch_length+c.patch_length-1;
      const auto grid=window.times.to(c.device).index_select(0,batches);
      out.token_end_times=torch::where(out.patch_valid,grid.gather(1,indices),torch::zeros_like(indices).to(torch::kFloat64));
    }else out.token_end_times=torch::zeros(out.patch_positions.sizes(),torch::TensorOptions().device(c.device).dtype(torch::kFloat64));
    return out;
  }
  CausalTrace trace_at(const RawHistory &raw,double cutoff,const FrozenScaler &scaler){return trace_at(raw,torch::full({raw.data.size(0)},cutoff,torch::kFloat64),scaler);}
  implementation::LocationScaleForwardTrace forward_at(const RawHistory &raw,const torch::Tensor &cutoffs,
      const torch::Tensor &hidden,const FrozenScaler &scaler,const torch::Tensor &context_deleted={}){
    TORCH_CHECK(is_training()?torch::GradMode::is_enabled():!torch::GradMode::is_enabled(),"[cpb-v3] training needs gradients; eval reconstruction needs NoGradGuard");
    // Observation/target storage is cropped before either masking stream is read.
    const auto retained=boundary::retain(raw,cutoffs,config());const auto &c=config().level;
    TORCH_CHECK(scaler.identity()==scaler_identity_,"[cpb-v3] training must use its own frozen TRAIN scaler");
    const auto target=implementation::relative_change_parent_input(retained.raw,scaler,c);
    const auto masks=implementation::configurable_spectral_mask_from_hidden(retained.raw.observed.to(c.device),hidden.to(c.device),c);
    auto visible=masks.visible;
    if(context_deleted.defined()){
      TORCH_CHECK(context_deleted.scalar_type()==torch::kBool&&context_deleted.sizes()==visible.sizes(),"[cpb-v3] retained-window boolean context deletion required");
      const auto deleted=context_deleted.to(c.device);
      TORCH_CHECK(!(deleted&visible.logical_not()).any().item<bool>(),"[cpb-v3] context deletions must be a subset of query-visible cells");visible=visible&deleted.logical_not();
    }
    const implementation::Input input{retained.raw.data,visible.to(torch::kCPU),retained.raw.channel_ids,retained.raw.endpoints,retained.raw.sampling_interval};
    return core_->forward_with_trace(input,target.data,masks.target,masks.eligible_channels,scaler);
  }
private:implementation::LocationScaleDecoderModel core_{nullptr};int64_t causal_updates_{0},initialization_seed_{0};
  std::string scaler_identity_,normalizer_identity_,initial_state_identity_;
};
TORCH_MODULE(Encoder);
struct Loaded{Config config;Encoder model{nullptr};FrozenScaler scaler;int64_t source_updates{0},causal_updates{0},initialization_seed{0};std::string initial_state_identity;};

// The only initialization API constructs every learned tensor from its seeded
// constructor. No checkpoint path, pretrained model or learned tensor map enters.
inline Loaded fresh(const Config &c,const FrozenScaler &scaler,const BankStandardizer &normalizer,int64_t seed){
  implementation::validate_location_scale_decoder_config(c);scaler.validate(input_config(c));
  const auto scaler_id=scaler.identity();normalizer.validate(c.level,nullptr,&scaler_id);
  TORCH_CHECK(seed>=0&&normalizer.frozen&&c.level.device==torch::Device(torch::kCUDA,0),"[cpb-v3] seed/frozen TRAIN normalizer/CUDA0 required");
  TORCH_CHECK(c.level.channel_count==3&&c.level.history_length==128&&c.level.input_width==9&&c.level.patch_length==8&&c.change_width==4&&c.level.export_width==32,"[cpb-v3] declared fresh native36 geometry required");
  torch::manual_seed(seed);auto core=implementation::LocationScaleDecoderModel(c,normalizer);auto model=Encoder(core,seed,scaler_id);
  int64_t learned_tensors=0,learned_values=0,fixed_values=0;for(const auto &p:model->named_parameters())if(p.value().requires_grad()){++learned_tensors;learned_values+=p.value().numel();}else fixed_values+=p.value().numel();
  TORCH_CHECK(learned_tensors==75&&learned_values==293672&&fixed_values==3888,"[cpb-v3] all75 learned tensors fresh; literal3888 prior only");model->validate_frozen();
  return{c,model,scaler,0,0,seed,model->initial_state_identity()};
}

namespace checkpoint_detail {
namespace d=implementation::configurable_checkpoint_detail;
using implementation::separated_change_checkpoint_detail::integer;
inline constexpr const char *kTrainingPolicy="from-scratch-end-to-end-causal-location-scale-nll-v1;all-learned-except-literal-spectral-prior;fresh-TRAIN-stats;fresh-AdamW";
inline void validate_geometry(torch::serialize::InputArchive &a,const Config &c){
  TORCH_CHECK(boundary::same_bits(d::read_vector(a,"config_integer_v1",torch::kInt64,d::integers(c.level).size()),torch::tensor(d::integers(c.level),torch::kInt64))&&
      boundary::same_bits(d::read_vector(a,"config_number_v1",torch::kFloat64,d::numbers(c.level).size()),torch::tensor(d::numbers(c.level),torch::kFloat64))&&
      boundary::same_bits(d::read_vector(a,"channel_ids",torch::kInt64,c.level.channel_count),torch::tensor(implementation::resolved_channel_ids(c.level),torch::kInt64))&&
      integer(a,"change_width")==c.change_width&&integer(a,"native_width")==implementation::location_scale_decoder_native_width(c),"[cpb-v3] outer geometry mismatch");
}
inline void write(torch::serialize::OutputArchive &a,const Encoder &model,const FrozenScaler &scaler,int64_t updates){
  TORCH_CHECK(model&&updates>=0&&model->causal_updates()==updates&&scaler.identity()==model->input_scaler_identity(),"[cpb-v3] own TRAIN scaler/exact causal count required");
  model->core()->validate_state();model->core()->validate_fixed_prior();const auto scaler_id=scaler.identity();model->normalizer().validate(model->config().level,nullptr,&scaler_id);
  a.write("artifact_kind",d::text_tensor(kCheckpointKind),true);a.write("model_tag",d::text_tensor(kModelTag),true);
  a.write("causal_policy",d::text_tensor(kCausalPolicy),true);a.write("training_policy",d::text_tensor(kTrainingPolicy),true);
  a.write("initialization_kind",d::text_tensor(kInitializationKind),true);a.write("initialization_seed",torch::tensor(model->initialization_seed(),torch::kInt64),true);
  a.write("pretrained_ancestor",d::text_tensor("none"),true);a.write("source_updates",torch::tensor(int64_t{0},torch::kInt64),true);a.write("causal_updates",torch::tensor(updates,torch::kInt64),true);
  a.write("scaler_identity",d::text_tensor(scaler_id),true);a.write("normalizer_identity",d::text_tensor(model->normalizer().identity()),true);
  a.write("initialization_binding",d::text_tensor(std::string(kInitializationKind)+":"+std::to_string(model->initialization_seed())+":"+model->initial_state_identity()+":"+scaler_id+":"+model->normalizer().identity()),true);
  a.write("state_identity",d::text_tensor(model->state_identity()),true);a.write("initial_state_identity",d::text_tensor(model->initial_state_identity()),true);a.write("normalizer_source_identity",d::text_tensor(model->normalizer().source_identity),true);
  const auto &c=model->config();a.write("config_integer_v1",torch::tensor(d::integers(c.level),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(c.level),torch::kFloat64),true);a.write("channel_ids",torch::tensor(implementation::resolved_channel_ids(c.level),torch::kInt64),true);
  a.write("change_width",torch::tensor(c.change_width,torch::kInt64),true);a.write("native_width",torch::tensor(implementation::location_scale_decoder_native_width(c),torch::kInt64),true);
  torch::serialize::OutputArchive payload;implementation::location_scale_decoder_checkpoint_detail::write(payload,model->core(),scaler,updates);a.write("payload",payload);
}
inline Loaded read(torch::serialize::InputArchive &a,torch::Device device,const Config *expected){
  TORCH_CHECK(device==torch::Device(torch::kCUDA,0)&&d::read_text(a,"artifact_kind")==kCheckpointKind&&d::read_text(a,"model_tag")==kModelTag&&
      d::read_text(a,"causal_policy")==kCausalPolicy&&d::read_text(a,"training_policy")==kTrainingPolicy&&d::read_text(a,"initialization_kind")==kInitializationKind&&
      d::read_text(a,"pretrained_ancestor")=="none"&&integer(a,"source_updates")==0&&integer(a,"causal_updates")>=0&&integer(a,"initialization_seed")>=0,"[cpb-v3] typed fresh family/causal policy/zero ancestry required");
  const auto updates=integer(a,"causal_updates"),seed=integer(a,"initialization_seed");const auto state=d::read_text(a,"state_identity"),initial_state=d::read_text(a,"initial_state_identity");
  const auto scaler_id=d::read_text(a,"scaler_identity"),bank_id=d::read_text(a,"normalizer_identity"),bank_source=d::read_text(a,"normalizer_source_identity");torch::serialize::InputArchive payload;a.read("payload",payload);
  TORCH_CHECK(d::read_text(a,"initialization_binding")==std::string(kInitializationKind)+":"+std::to_string(seed)+":"+initial_state+":"+scaler_id+":"+bank_id,"[cpb-v3] fresh initialization/seed/stat binding mismatch");
  auto copied=implementation::location_scale_decoder_checkpoint_detail::read(payload,device,expected,&scaler_id,&bank_source,&bank_id);
  validate_geometry(a,copied.config);TORCH_CHECK(copied.training_updates==updates,"[cpb-v3] nested update mismatch");
  auto model=Encoder(copied.model,seed,scaler_id,updates,initial_state);model->clear_gradients();model->validate_frozen();
  TORCH_CHECK(model->state_identity()==state&&(!initial_state.empty())&&(updates!=0||state==initial_state),"[cpb-v3] model/initial tensor state binding mismatch");
  return{copied.config,model,copied.scaler,0,updates,seed,initial_state};
}
} // namespace checkpoint_detail
inline void save_checkpoint(const std::string &path,const Encoder &model,const FrozenScaler &scaler,int64_t updates){
  TORCH_CHECK(model,"[cpb-v3] model required");model->validate_frozen();torch::serialize::OutputArchive a;checkpoint_detail::write(a,model,scaler,updates);checkpoint_detail::d::publish_checkpoint(a,path);
}
inline Loaded load_checkpoint(const std::string &path,torch::Device device=torch::Device(torch::kCUDA,0),const Config *expected=nullptr){
  torch::serialize::InputArchive a;a.load_from(path,device);return checkpoint_detail::read(a,device,expected);
}
namespace training {
using TrainingIdentity=implementation::consecutive_change_training::TrainingIdentity;
using TrainingProgress=implementation::consecutive_change_training::TrainingProgress;
using implementation::consecutive_change_training::training_counter_seed;
namespace common=implementation::consecutive_change_training::training_state_detail;
namespace d=implementation::configurable_checkpoint_detail;
inline constexpr const char *kTrainingStateKind="cpb_v3_causal_training_state_v1";
inline std::string progress_binding(const TrainingIdentity &i,const TrainingProgress &p){return i.recipe_identity+":"+std::to_string(p.completed_updates)+":"+std::to_string(p.attempted_batches)+":"+std::to_string(p.row_seed)+":"+std::to_string(p.mask_seed)+":"+std::to_string(p.context_seed);}
inline auto adamw_options(){return implementation::consecutive_change_training::content_training_adamw_options();}
inline std::vector<torch::Tensor> learned_parameters(const Encoder &model){
  std::vector<torch::Tensor> out;for(const auto &p:model->named_parameters())if(p.value().requires_grad())out.push_back(p.value());return out;
}
inline std::string parameter_order(const Encoder &model){std::string out;for(const auto &p:model->named_parameters())out+=p.key()+":"+(p.value().requires_grad()?"learned":"fixed")+'\n';return out;}
inline std::unique_ptr<torch::optim::AdamW> make_optimizer(const Encoder &model){
  TORCH_CHECK(model,"[cpb-v3] owned model required");model->core()->validate_state();return std::make_unique<torch::optim::AdamW>(learned_parameters(model),adamw_options());
}
inline std::vector<int64_t> validate_optimizer(const Encoder &model,const torch::optim::AdamW &optimizer,const TrainingProgress &progress){
  common::validate_progress(progress);model->core()->validate_state();model->core()->validate_fixed_prior();
  const auto *defaults=dynamic_cast<const torch::optim::AdamWOptions *>(&optimizer.defaults());
  TORCH_CHECK(defaults&&*defaults==adamw_options()&&optimizer.param_groups().size()==1,"[cpb-v3] fixed single-group AdamW required");
  const auto params=learned_parameters(model);const auto &group=optimizer.param_groups().front();const auto *options=dynamic_cast<const torch::optim::AdamWOptions *>(&group.options());
  TORCH_CHECK(options&&*options==adamw_options()&&params.size()==group.params().size(),"[cpb-v3] learned optimizer group mismatch");
  std::set<void *> known;std::vector<int64_t> steps;int64_t maximum_step=0;
  for(size_t i=0;i<params.size();++i){const auto &p=params[i];auto *key=p.unsafeGetTensorImpl();TORCH_CHECK(group.params()[i].unsafeGetTensorImpl()==key&&known.insert(key).second,"[cpb-v3] parameter order/alias mismatch");
    const auto found=optimizer.state().find(key);if(found==optimizer.state().end()){steps.push_back(-1);continue;}
    const auto *state=dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
    TORCH_CHECK(state&&state->step()>0&&state->step()<=progress.completed_updates&&!state->max_exp_avg_sq().defined(),"[cpb-v3] optimizer step/flags mismatch");
    for(const auto &m:{state->exp_avg(),state->exp_avg_sq()})TORCH_CHECK(m.defined()&&m.device()==p.device()&&m.scalar_type()==p.scalar_type()&&m.sizes()==p.sizes()&&torch::isfinite(m).all().item<bool>(),"[cpb-v3] matching finite CUDA optimizer moments required");
    TORCH_CHECK(state->exp_avg_sq().ge(0).all().item<bool>(),"[cpb-v3] negative second moment");steps.push_back(state->step());maximum_step=std::max(maximum_step,state->step());
  }
  for(const auto &item:optimizer.state())TORCH_CHECK(known.count(item.first),"[cpb-v3] orphan/fixed optimizer state forbidden");
  TORCH_CHECK(maximum_step==progress.completed_updates,"[cpb-v3] completed count and optimizer maximum step disagree");
  TORCH_CHECK(progress.completed_updates==0?optimizer.state().empty():!optimizer.state().empty(),"[cpb-v3] completed updates and initialized optimizer disagree");return steps;
}
// Bind moment CONTENT as well as per-parameter presence/order/steps. Ordinary
// finite shape-correct reset or substituted moments must not silently resume.
inline std::string optimizer_identity(const Encoder &model,const torch::optim::AdamW &optimizer){
  uint64_t hash=14695981039346656037ULL;const auto bytes=[&](const void *p,size_t n){const auto *v=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;++i){hash^=v[i];hash*=1099511628211ULL;}};
  for(const auto &item:model->named_parameters())if(item.value().requires_grad()){
    bytes(item.key().data(),item.key().size());const auto found=optimizer.state().find(item.value().unsafeGetTensorImpl());const uint8_t present=found!=optimizer.state().end();bytes(&present,sizeof(present));if(!present)continue;
    const auto *state=dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());TORCH_CHECK(state,"[cpb-v3] AdamW moments required for identity");const int64_t step=state->step();bytes(&step,sizeof(step));
    for(const auto &moment:{state->exp_avg(),state->exp_avg_sq()}){const auto cpu=moment.detach().to(torch::kCPU).contiguous();bytes(cpu.const_data_ptr(),cpu.nbytes());}
  }
  std::ostringstream out;out<<"cpb-v3-optimizer-fnv1a64-"<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();
}
struct LoadedTrainingState{Config config;Encoder model{nullptr};FrozenScaler scaler;std::unique_ptr<torch::optim::AdamW> optimizer;TrainingIdentity identity;TrainingProgress progress;};
inline void save_training_state(const std::string &path,const Encoder &model,const FrozenScaler &scaler,const torch::optim::AdamW &optimizer,const TrainingIdentity &identity,const TrainingProgress &progress){
  common::validate_identity(identity);TORCH_CHECK(model->causal_updates()==progress.completed_updates,"[cpb-v3] model causal count and training progress disagree");const auto steps=validate_optimizer(model,optimizer,progress);
  torch::serialize::OutputArchive a,e,o;a.write("artifact_kind",d::text_tensor(kTrainingStateKind),true);a.write("model_tag",d::text_tensor(kModelTag),true);
  a.write("counter_contract",d::text_tensor(implementation::consecutive_change_training::kChangeTrainingCounterContract),true);
  a.write("input_source_identity",d::text_tensor(identity.input_source_identity),true);a.write("split_identity",d::text_tensor(identity.split_identity),true);a.write("recipe_identity",d::text_tensor(identity.recipe_identity),true);
  a.write("progress_v1",torch::tensor(std::vector<int64_t>{progress.completed_updates,progress.attempted_batches,progress.row_seed,progress.mask_seed,progress.context_seed},torch::kInt64),true);
  a.write("progress_binding",d::text_tensor(progress_binding(identity,progress)),true);
  a.write("parameter_order",d::text_tensor(parameter_order(model)),true);a.write("optimizer_parameter_keys",d::text_tensor(common::optimizer_parameter_keys(optimizer)),true);a.write("optimizer_steps_v1",torch::tensor(steps,torch::kInt64),true);
  a.write("optimizer_state_identity",d::text_tensor(optimizer_identity(model,optimizer)),true);
  checkpoint_detail::write(e,model,scaler,progress.completed_updates);optimizer.save(o);a.write("encoder",e);a.write("optimizer",o);d::publish_checkpoint(a,path);
}
inline LoadedTrainingState load_training_state(const std::string &path,const Config &expected,const TrainingIdentity &identity){
  common::validate_identity(identity);torch::serialize::InputArchive a;a.load_from(path,expected.level.device);
  TORCH_CHECK(d::read_text(a,"artifact_kind")==kTrainingStateKind&&d::read_text(a,"model_tag")==kModelTag&&d::read_text(a,"counter_contract")==implementation::consecutive_change_training::kChangeTrainingCounterContract,"[cpb-v3] typed causal training/counter kind required");
  TrainingIdentity actual{d::read_text(a,"input_source_identity"),d::read_text(a,"split_identity"),d::read_text(a,"recipe_identity")};common::validate_identity(actual);TORCH_CHECK(common::same_identity(actual,identity),"[cpb-v3] dataset/split/recipe identity mismatch");
  const auto v=d::read_vector(a,"progress_v1",torch::kInt64,5);const auto x=v.accessor<int64_t,1>();TrainingProgress progress{x[0],x[1],x[2],x[3],x[4]};common::validate_progress(progress);
  TORCH_CHECK(d::read_text(a,"progress_binding")==progress_binding(identity,progress),"[cpb-v3] counter/seed binding mismatch");
  torch::serialize::InputArchive e,o;a.read("encoder",e);auto child=checkpoint_detail::read(e,expected.level.device,&expected);
  TORCH_CHECK(child.causal_updates==progress.completed_updates&&d::read_text(a,"parameter_order")==parameter_order(child.model),"[cpb-v3] child update/parameter order mismatch");
  auto optimizer=make_optimizer(child.model);a.read("optimizer",o);common::validate_optimizer_archive(o,d::read_text(a,"optimizer_parameter_keys"),learned_parameters(child.model).size());
  optimizer->load(o);const auto steps=validate_optimizer(child.model,*optimizer,progress);
  TORCH_CHECK(d::read_text(a,"optimizer_state_identity")==optimizer_identity(child.model,*optimizer),"[cpb-v3] changed/reset optimizer moment content");
  TORCH_CHECK(torch::equal(d::read_vector(a,"optimizer_steps_v1",torch::kInt64,steps.size()),torch::tensor(steps,torch::kInt64)),"[cpb-v3] missing/reset/remapped steps");
  child.model->train();child.model->clear_gradients();return{child.config,child.model,child.scaler,std::move(optimizer),actual,progress};
}
} // namespace training
} // namespace embedding::encoders::causal_patch_bottleneck_mae::v3
