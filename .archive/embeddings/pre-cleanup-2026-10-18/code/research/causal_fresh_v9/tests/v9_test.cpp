// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Fresh seeded CUDA initialization, causal learning/resume and bounded causal proof.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v9.h"
#include <torch/cuda.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v9;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
int64_t checks=0,model_calls=0,api_attempts=0,suffix_cases=0,internal_cases=0,retained_cells_checked=0;
void check(bool yes,const std::string &why){++checks;TORCH_CHECK(yes,"[CPB-v9 frozen CUDA proof] ",why);}
template<class F>void rejects(F operation,const std::string &why){bool caught=false;try{operation();}catch(const std::exception&){caught=true;}check(caught,why);}
bool bits(const torch::Tensor &a,const torch::Tensor &b){
  if(!a.defined()||!b.defined())return !a.defined()&&!b.defined();
  if(a.scalar_type()!=b.scalar_type()||a.sizes()!=b.sizes())return false;
  const auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
  return x.nbytes()==0||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.nbytes())==0;
}
std::string quote(const std::string &s){std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c=='\n')o<<"\\n";else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;}o<<'"';return o.str();}
std::string objective_json_fields(){return "\"objective_policy\":"+quote(cp::kObjectivePolicy)+",\"mean_huber_delta\":1,\"loss_coefficients\":[1,1],\"variance_error_detached\":true,\"reporting_nll_backpropagated\":false,\"shared_global_clip\":true,\"optimizer_independence_claimed\":false,\"temporal_pooling_policy\":"+quote(cp::kTemporalPoolingPolicy)+",\"temporal_pooling_partition\":[4,16,32,4,8],\"temporal_pooling_mix\":[0.5,0.5],\"empty_local_bin_policy\":"+quote(cp::kEmptyLocalBinPolicy)+",\"registered_learned_tensors\":76,\"mandatory_identifiable_learned_tensors\":75,\"data_gradient_gauge_exemption\":\"core.encoder.level.base.backbone.pool_score.bias\"";}
std::string tensor_json(const torch::Tensor &value){
  const auto t=value.detach().to(torch::kCPU).contiguous();std::ostringstream o;o<<std::setprecision(17);
  const auto scalar=[&](int64_t i){
    if(t.scalar_type()==torch::kBool)o<<(t.const_data_ptr<bool>()[i]?"true":"false");
    else if(t.scalar_type()==torch::kInt64)o<<t.const_data_ptr<int64_t>()[i];
    else{check(t.scalar_type()==torch::kFloat32||t.scalar_type()==torch::kFloat64,"JSON floating dtype");const double v=t.scalar_type()==torch::kFloat32?double(t.const_data_ptr<float>()[i]):t.const_data_ptr<double>()[i];
      if(std::isnan(v))o<<quote("NaN");else if(std::isinf(v))o<<quote(v>0?"+Infinity":"-Infinity");else o<<v;}
  };
  std::function<void(int64_t,int64_t)> emit=[&](int64_t d,int64_t offset){if(d==t.dim()){scalar(offset);return;}o<<'[';for(int64_t i=0;i<t.size(d);++i){if(i)o<<',';emit(d+1,offset+i*t.stride(d));}o<<']';};emit(0,0);return o.str();
}
void write_new(const fs::path &path,const std::string &text){check(!fs::exists(path),"exclusive evidence file");std::ofstream f(path,std::ios::binary);check(bool(f),"open evidence");f<<text<<'\n';f.close();check(bool(f),"write evidence");}
std::string fingerprint(const torch::Tensor &v){const auto t=v.detach().to(torch::kCPU).contiguous();uint64_t h=14695981039346656037ULL;const auto *p=static_cast<const unsigned char*>(t.const_data_ptr());for(size_t i=0;i<t.nbytes();++i){h^=p[i];h*=1099511628211ULL;}std::ostringstream o;o<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return o.str();}
std::string scaler_metadata(const cp::FrozenScaler &s){std::ostringstream o;o<<std::setprecision(17)<<s.identity()<<';'<<s.scale_floor;return o.str();}
template<class N>std::string normalizer_metadata(const N &n){
  std::ostringstream o;o<<std::setprecision(17)<<n.identity()<<';'<<n.source_identity<<';'<<n.input_scaler_identity<<';'<<n.layout_identity<<';'<<n.recipe_identity
    <<';'<<n.bank_family<<';'<<n.raw_contract<<';'<<n.increment_recipe<<';'<<n.sampling_interval<<';'<<n.floor<<';'<<n.minimum_support<<';'<<n.frozen;
  for(const auto v:n.input_geometry){o<<';'<<v;}
  return o.str();
}
struct SavedTensor{std::string name;torch::Tensor value;torch::Device device;bool requires_grad;};
struct FrozenState{
  std::vector<SavedTensor> tensors;std::string norm_metadata,scale_metadata;int64_t parameter_count{0};
  template<class Model>FrozenState(const Model &model,const cp::FrozenScaler &scaler){
    const auto add=[&](const std::string &name,const torch::Tensor &t){check(!t.grad().defined(),"no state gradients before proof");tensors.push_back({name,t.detach().to(torch::kCPU).clone(),t.device(),t.requires_grad()});};
    const auto params=model->named_parameters(),buffers=model->named_buffers();
    check(params.size()==77,"exact fresh77 parameter tensors");
    for(const auto &v:params){check(v.value().device()==torch::Device(torch::kCUDA,0)&&v.value().scalar_type()==torch::kFloat32,"all actual CUDA0 F32 parameters");parameter_count+=v.value().numel();add("parameter/"+v.key(),v.value());}
    for(const auto &v:buffers)add("buffer/"+v.key(),v.value());
    check(parameter_count==298640,"exact fresh298640 parameter values");
    const auto &n=model->normalizer();
    for(const auto &[key,t]:std::vector<std::pair<std::string,torch::Tensor>>{{"counts",n.counts},{"sum",n.sum},{"sumsq",n.sumsq},{"mean",n.mean},{"scale",n.scale},{"support",n.support}}){check(t.device().is_cpu()&&!t.requires_grad(),"unregistered frozen normalizer CPU state");add("normalizer/"+key,t);}
    for(const auto &[key,t]:std::vector<std::pair<std::string,torch::Tensor>>{{"mean",scaler.mean},{"scale",scaler.scale},{"count",scaler.count},{"channel_ids",scaler.channel_ids},{"floor_applied",scaler.floor_applied}})add("scaler/"+key,t);
    norm_metadata=normalizer_metadata(n);scale_metadata=scaler_metadata(scaler);
  }
  template<class Model>void verify(const Model &model,const cp::FrozenScaler &scaler)const{
    for(const auto &m:model->modules())check(!m->is_training(),"every model module remains eval");
    std::map<std::string,torch::Tensor> now;for(const auto &v:model->named_parameters())now.emplace("parameter/"+v.key(),v.value());for(const auto &v:model->named_buffers())now.emplace("buffer/"+v.key(),v.value());
    const auto &n=model->normalizer();for(const auto &[key,t]:std::vector<std::pair<std::string,torch::Tensor>>{{"counts",n.counts},{"sum",n.sum},{"sumsq",n.sumsq},{"mean",n.mean},{"scale",n.scale},{"support",n.support}})now.emplace("normalizer/"+key,t);
    for(const auto &[key,t]:std::vector<std::pair<std::string,torch::Tensor>>{{"mean",scaler.mean},{"scale",scaler.scale},{"count",scaler.count},{"channel_ids",scaler.channel_ids},{"floor_applied",scaler.floor_applied}})now.emplace("scaler/"+key,t);
    check(now.size()==tensors.size(),"complete immutable state inventory");for(const auto &v:tensors){const auto &t=now.at(v.name);check(t.device()==v.device&&t.requires_grad()==v.requires_grad&&!t.grad().defined()&&bits(t,v.value),"exact state/dtype/device/shape/gradient: "+v.name);}
    check(normalizer_metadata(n)==norm_metadata&&scaler_metadata(scaler)==scale_metadata,"all unregistered normalizer/scaler metadata unchanged");
  }
  std::string json()const{
    std::ostringstream o;o<<"{\"parameter_count\":"<<parameter_count<<",\"normalizer_metadata\":"<<quote(norm_metadata)<<",\"scaler_metadata\":"<<quote(scale_metadata)<<",\"tensors\":[";
    for(size_t i=0;i<tensors.size();++i){if(i)o<<',';const auto &v=tensors[i];o<<"{\"name\":"<<quote(v.name)<<",\"shape\":"<<tensor_json(torch::tensor(v.value.sizes().vec(),torch::kInt64))<<",\"dtype\":"<<quote(c10::toString(v.value.scalar_type()))<<",\"device\":"<<quote(v.device.str())<<",\"requires_grad\":"<<(v.requires_grad?"true":"false")<<",\"gradient_defined\":false,\"content_fnv1a64\":"<<quote(fingerprint(v.value))<<'}';}o<<"]}";return o.str();
  }
};
cp::RawHistory history(int64_t N,int64_t B=3){
  auto data=torch::zeros({B,3,N,9},torch::kFloat64),mask=torch::ones({B,3,N,9},torch::kBool);auto x=data.accessor<double,4>();auto m=mask.accessor<bool,4>();
  for(int64_t b=0;b<B;++b)for(int64_t c=0;c<3;++c)for(int64_t t=0;t<N;++t)for(int64_t f=0;f<9;++f){
    x[b][c][t][f]=10.+c*3.+f*.4+.01*t+.8*std::sin(.11*t+.21*c+.17*f+.23*b)+.2*std::cos(.73*t+.31*f);
    m[b][c][t][f]=!((t>=17&&t<=23)||(t>=61&&t<=70)||((t+3*f+c)%23==0));
    if(b==1&&c==2)m[b][c][t][f]=false;
    if(b==2&&t<81)m[b][c][t][f]=false;
  }
  return {data,mask,torch::tensor({0,1,2},torch::kInt64),torch::arange(N,torch::kFloat64)*60000.,60000.};
}
cp::RawHistory clone(const cp::RawHistory &h){return{h.data.clone(),h.observed.clone(),h.channel_ids.clone(),h.timestamps.clone(),h.sampling_interval};}
cp::RawHistory permuted(const cp::RawHistory &h){const auto p=torch::tensor({2,0,1},torch::kInt64);return{h.data.index_select(1,p),h.observed.index_select(1,p),h.channel_ids.index_select(h.channel_ids.dim()==1?0:1,p),h.timestamps,h.sampling_interval};}
torch::Tensor cutoffs(int64_t index,int64_t B=3){return torch::full({B},index*60000.,torch::kFloat64);}
cp::RawHistory intervention(const cp::RawHistory &base,const torch::Tensor &cuts,int variant){
  auto out=clone(base);auto x=out.data.accessor<double,4>();auto m=out.observed.accessor<bool,4>();
  for(int64_t b=0;b<out.data.size(0);++b)for(int64_t c=0;c<3;++c)for(int64_t t=0;t<out.data.size(2);++t)for(int64_t f=0;f<9;++f){
    const double timestamp=out.timestamps.dim()==1?out.timestamps[t].template item<double>():out.timestamps[b][t].template item<double>();if(timestamp<=cuts[b].template item<double>())continue;
    if(variant==0||variant==8)x[b][c][t][f]=-1234567.+t*33.+c*17.-f;
    if(variant==1||variant==7)x[b][c][t][f]=(t%2?1.:-1.)*std::numeric_limits<double>::max()/4.;
    if(variant==2||variant==6){const auto choice=(b+c+t+f)%3;x[b][c][t][f]=choice==0?std::numeric_limits<double>::quiet_NaN():choice==1?std::numeric_limits<double>::infinity():-std::numeric_limits<double>::infinity();}
    if(variant==3||variant==6)m[b][c][t][f]=true;
    if(variant==4||variant==7)m[b][c][t][f]=false;
    if(variant==5||variant==8)m[b][c][t][f]=(b+7*c+3*t+f)%3==0;
  }return out;
}
template<class Model>auto trace(Model &m,const cp::RawHistory &h,const torch::Tensor &cuts,const cp::FrozenScaler &s){++api_attempts;auto result=m->trace_at(h,cuts,s);++model_calls;return result;}
template<class Model>auto encode(Model &m,const cp::RawHistory &h,const torch::Tensor &cuts,const cp::FrozenScaler &s){++api_attempts;auto result=m->encode_at(h,cuts,s);++model_calls;return result;}
template<class A,class B>void same_encoding(const A &a,const B &b){check(bits(a.native,b.native)&&bits(a.valid,b.valid)&&bits(a.cutoffs,b.cutoffs),"exact same-cutoff native/valid/cutoff invariance");}
template<class A,class B>void same_trace(const A &a,const B &b,bool compare_raw=true){
  same_encoding(a.encoding,b.encoding);
  if(compare_raw)check(bits(a.retained_raw,b.retained_raw),"exact retained raw under suffix intervention");
  check(bits(a.retained_observed,b.retained_observed)&&bits(a.retained_timestamps,b.retained_timestamps),"exact retained mask/time support");
  check(bits(a.normalized_parent,b.normalized_parent),"exact parent input before all learned processing");
  check(bits(a.content_values,b.content_values)&&bits(a.content_counts,b.content_counts)&&bits(a.content_eligible,b.content_eligible),"exact prefix-only content bank");
  check(bits(a.spectral_values,b.spectral_values)&&bits(a.spectral_counts,b.spectral_counts)&&bits(a.spectral_eligible,b.spectral_eligible),"exact prefix-only spectral bank");
  check(bits(a.change_values,b.change_values)&&bits(a.change_valid,b.change_valid)&&bits(a.change_counts,b.change_counts)&&bits(a.conditioned_bank,b.conditioned_bank),"exact prefix-only relative/quarter/conditioned bank");
  check(bits(a.patch_positions,b.patch_positions)&&bits(a.patch_valid,b.patch_valid)&&bits(a.patch_rows,b.patch_rows)&&bits(a.patch_values,b.patch_values)&&bits(a.patch_visibility,b.patch_visibility),"exact causal tokenization/support");
  check(bits(a.temporal_hidden,b.temporal_hidden)&&bits(a.contextual_temporal_hidden,b.contextual_temporal_hidden),"exact both temporal paths under supplied suffix");
  check(bits(a.pool_weights,b.pool_weights)&&bits(a.contextual_pool_weights,b.contextual_pool_weights)&&bits(a.pool_bin_valid,b.pool_bin_valid)&&bits(a.contextual_pool_bin_valid,b.contextual_pool_bin_valid)&&bits(a.token_end_times,b.token_end_times),"exact prefix-only pooling/token times");
  check(bits(a.pool_local_weights,b.pool_local_weights)&&bits(a.contextual_pool_local_weights,b.contextual_pool_local_weights)&&
      bits(a.pool_global_weights,b.pool_global_weights)&&bits(a.contextual_pool_global_weights,b.contextual_pool_global_weights)&&
      bits(a.pool_mixture_weights,b.pool_mixture_weights)&&bits(a.contextual_pool_mixture_weights,b.contextual_pool_mixture_weights),
      "exact actual local/global/per-export mixture traces under future intervention");
}
template<class T>void retained_oracle(const T &out,const cp::RawHistory &h,const torch::Tensor &cuts,const cp::Config &config){
  const auto &c=config.level;const auto B=h.data.size(0),H=c.history_length,F=c.input_width;
  check(out.encoding.native.is_cuda()&&out.encoding.native.scalar_type()==torch::kFloat32&&out.encoding.native.sizes()==torch::IntArrayRef({B,40})&&torch::isfinite(out.encoding.native).all().template item<bool>()&&!out.encoding.native.requires_grad(),"actual finite detached CUDA native40");
  check(out.encoding.valid.is_cuda()&&out.encoding.valid.scalar_type()==torch::kBool&&bits(out.encoding.cutoffs,cuts),"actual CUDA validity and exact cutoff metadata");
  check(out.retained_raw.device().is_cpu()&&out.retained_raw.scalar_type()==torch::kFloat64&&out.retained_raw.sizes()==torch::IntArrayRef({B,3,H,F}),"retained genuine CPU F64 geometry");
  check(out.retained_observed.device().is_cpu()&&out.retained_observed.scalar_type()==torch::kBool&&out.retained_observed.sizes()==out.retained_raw.sizes(),"retained original boolean support");
  auto expected=torch::zeros_like(out.retained_raw),expected_mask=torch::zeros_like(out.retained_observed),times=torch::zeros({B,H},torch::kFloat64);
  for(int64_t b=0;b<B;++b){const auto grid=h.timestamps.dim()==1?h.timestamps:h.timestamps[b];const auto selected=grid.eq(cuts[b]).nonzero().flatten();check(selected.numel()==1,"oracle cutoff exactly on original grid");const auto end=selected[0].template item<int64_t>(),n=std::min<int64_t>(H,end+1),begin=end+1-n;
    expected[b].narrow(1,H-n,n).copy_(h.data[b].narrow(1,begin,n));expected_mask[b].narrow(1,H-n,n).copy_(h.observed[b].narrow(1,begin,n));
    for(int64_t t=0;t<H;++t)times[b][t]=cuts[b].template item<double>()-(H-1-t)*h.sampling_interval;
    if(n<H)check(out.retained_raw[b].narrow(1,0,H-n).eq(0).all().template item<bool>()&&!out.retained_observed[b].narrow(1,0,H-n).any().template item<bool>(),"short prefix left-pad is unavailable exact zero");
  }
  check(bits(out.retained_observed,expected_mask)&&bits(out.retained_timestamps,times),"independent per-example rolling lastH timestamps/support oracle");
  check(bits(torch::where(expected_mask,out.retained_raw,torch::zeros_like(expected)),torch::where(expected_mask,expected,torch::zeros_like(expected))),"original retained visible raw double values exact");
  check(bits(out.encoding.valid.to(torch::kCPU),expected_mask.flatten(1).any(1)),"validity uses only retained support");
  check(out.encoding.native.masked_select(out.encoding.valid.logical_not().unsqueeze(1).expand_as(out.encoding.native)).eq(0).all().template item<bool>(),"empty-support encoding exact zero");
  for(const auto &v:{out.normalized_parent,out.content_values,out.content_counts,out.content_eligible,out.spectral_values,out.spectral_counts,out.spectral_eligible,out.change_values,out.change_valid,out.change_counts,out.conditioned_bank})check(v.is_cuda(),"actual CUDA parent/bank/support traces");
  check(out.normalized_parent.masked_select(out.retained_observed.to(out.normalized_parent.device()).logical_not()).eq(0).all().template item<bool>(),"hidden parent input zero");
  retained_cells_checked+=B*3*H*F;
}template<class T>void token_oracle(const T &out,const torch::Tensor &cuts,const cp::Config &config){
  const auto &c=config.level;const auto positions=out.patch_positions.to(torch::kCPU),valid=out.patch_valid.to(torch::kCPU),rows=out.patch_rows.to(torch::kCPU),times=out.token_end_times.to(torch::kCPU);
  check(positions.dim()==2&&valid.sizes()==positions.sizes()&&rows.sizes()==torch::IntArrayRef({positions.size(0)})&&times.sizes()==positions.sizes(),"original patch/time trace geometry");
  check(out.temporal_hidden.is_cuda()&&out.contextual_temporal_hidden.is_cuda()&&out.pool_weights.is_cuda()&&out.contextual_pool_weights.is_cuda(),"both actual CUDA temporal/pooling paths traced");
  std::set<int64_t> seen_rows;
  for(int64_t r=0;r<positions.size(0);++r){const auto row=rows[r].template item<int64_t>();check(seen_rows.insert(row).second&&row>=0&&row<cuts.size(0)*c.channel_count,"unique packed active original row");int64_t previous=-1;
    for(int64_t p=0;p<positions.size(1);++p){const bool active=valid[r][p].template item<bool>();const auto original=positions[r][p].template item<int64_t>();
      if(!active){check(original==-1&&times[r][p].template item<double>()==0.&&out.pool_weights[r][p].template item<double>()==0.&&out.contextual_pool_weights[r][p].template item<double>()==0.,"padded tokens cannot enter pools");continue;}
      check(original>previous&&original<c.history_length/c.patch_length,"packing preserves original monotone time");previous=original;
      const auto b=row/c.channel_count,ch=row%c.channel_count,start=original*c.patch_length;
      const double end=cuts[b].template item<double>()-(c.history_length-(original+1)*c.patch_length)*c.sampling_interval;
      check(times[r][p].template item<double>()==end&&end<=cuts[b].template item<double>(),"token timestamp is completed retained original patch at or before cutoff");
      check(bits(out.patch_visibility[r][p],out.retained_observed[b][ch].narrow(0,start,c.patch_length).flatten()),"token visibility is original prefix mask");
      check(bits(out.patch_values[r][p],out.normalized_parent[b][ch].narrow(0,start,c.patch_length).flatten()),"token values contain exactly the causal parent patch");
    }
    check(out.pool_bin_valid.sizes()==torch::IntArrayRef({positions.size(0),4})&&out.contextual_pool_bin_valid.sizes()==out.pool_bin_valid.sizes(),"actual four-bin support trace geometry");
    int64_t nonempty=0;const auto K=c.history_length/c.patch_length;
    for(int64_t j=0;j<4;++j){const auto member=out.patch_valid[r].logical_and(out.patch_positions[r].ge(j*(K/4))).logical_and(out.patch_positions[r].lt((j+1)*(K/4)));
      const bool has=member.any().template item<bool>();nonempty+=has;
      check(out.pool_bin_valid[r][j].template item<bool>()==has&&out.contextual_pool_bin_valid[r][j].template item<bool>()==has,"bin validity follows original positions, not packed rank");
      const auto a=out.pool_local_weights[r].masked_select(member).sum().template item<double>(),b=out.contextual_pool_local_weights[r].masked_select(member).sum().template item<double>();
      check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-double(has))<2e-6&&std::abs(b-double(has))<2e-6,"anchored local weights have mass1 or exact empty0 in original bin");
      for(int path=0;path<2;++path){const auto local=path?out.contextual_pool_local_weights:out.pool_local_weights;
        const auto global=path?out.contextual_pool_global_weights:out.pool_global_weights;
        const auto mixture=path?out.contextual_pool_mixture_weights:out.pool_mixture_weights;
        check(mixture.sizes()==torch::IntArrayRef({positions.size(0),4,positions.size(1)})&&global.sizes()==positions.sizes(),"actual per-export/global trace geometry");
        const auto expected=has?.5*torch::where(member,local[r],torch::zeros_like(local[r]))+.5*global[r]:torch::zeros_like(global[r]);
        check(bits(mixture[r][j],expected)&&std::abs(mixture[r][j].sum().template item<double>()-double(has))<2e-6,"truthful half-local half-global per-export weights; empty slice never borrows global support");
      }
    }
    check(std::abs(out.pool_weights[r].sum().template item<double>()-nonempty)<8e-6&&std::abs(out.contextual_pool_weights[r].sum().template item<double>()-nonempty)<8e-6,"true flat weight mass equals number of nonempty bins");
    check(std::abs(out.pool_global_weights[r].sum().template item<double>()-1.)<2e-6&&std::abs(out.contextual_pool_global_weights[r].sum().template item<double>()-1.)<2e-6,"one shared valid-history probability per active row");
  }
}
struct StoredEncoding{torch::Tensor native,valid,cutoffs;};
template<class T>StoredEncoding store(const T &v){return{v.native.detach().to(torch::kCPU).clone(),v.valid.detach().to(torch::kCPU).clone(),v.cutoffs.detach().to(torch::kCPU).clone()};}
std::string bounded_witness(int64_t t,int layout,const cp::RawHistory &h,const torch::Tensor &cuts,const auto &out){
  std::ostringstream o;o<<"{\"cutoff_index\":"<<t<<",\"permuted\":"<<(layout?"true":"false")<<",\"cutoffs\":"<<tensor_json(cuts)
    <<",\"native\":"<<tensor_json(out.encoding.native)<<",\"valid\":"<<tensor_json(out.encoding.valid)<<",\"retained_timestamps\":"<<tensor_json(out.retained_timestamps)
    <<",\"retained_observed_count\":"<<out.retained_observed.sum().template item<int64_t>()<<",\"channel_ids\":"<<tensor_json(h.channel_ids)
    <<",\"content_values_fnv1a64\":"<<quote(fingerprint(out.content_values))<<",\"spectral_values_fnv1a64\":"<<quote(fingerprint(out.spectral_values))
    <<",\"change_values_fnv1a64\":"<<quote(fingerprint(out.change_values))<<",\"patch_positions\":"<<tensor_json(out.patch_positions)
    <<",\"patch_valid\":"<<tensor_json(out.patch_valid)<<",\"token_end_times\":"<<tensor_json(out.token_end_times)<<",\"observed_raw_samples\":[";
  bool first=true;for(int64_t b=0;b<h.data.size(0);++b)for(int64_t ch=0;ch<3;++ch)for(const int64_t f:{0,3,8}){if(!first)o<<',';first=false;
    o<<"{\"b\":"<<b<<",\"c\":"<<ch<<",\"f\":"<<f<<",\"raw64\":"<<tensor_json(h.data[b][ch][t][f])<<",\"observed\":"<<tensor_json(h.observed[b][ch][t][f])<<'}';}
  o<<"]}";return o.str();
}
template<class Model>void exhaustive_suffix(Model &model,const cp::FrozenScaler &scaler,const cp::Config &config,std::vector<std::string> &witnesses){
  const auto base=history(128);std::array<std::vector<StoredEncoding>,2> saved;
  for(int layout=0;layout<2;++layout){const auto h=layout?permuted(base):base;
    for(int64_t t=0;t<128;++t){const auto cuts=cutoffs(t);const auto original=trace(model,h,cuts,scaler);retained_oracle(original,h,cuts,config);token_oracle(original,cuts,config);
      same_encoding(original.encoding,encode(model,h,cuts,scaler));saved[layout].push_back(store(original.encoding));
      cp::RawHistory prefix{h.data.narrow(2,0,t+1),h.observed.narrow(2,0,t+1),h.channel_ids,h.timestamps.narrow(0,0,t+1),h.sampling_interval};
      same_trace(original,trace(model,prefix,cuts,scaler));
      for(int variant=0;variant<9;++variant){const auto changed=intervention(h,cuts,variant);const auto tested=trace(model,changed,cuts,scaler);same_trace(original,tested);++suffix_cases;}
      if(t==0||t==1||t==7||t==8||t==15||t==16||t==63||t==64||t==126||t==127)witnesses.push_back(bounded_witness(t,layout,h,cuts,original));
    }
    // Reverse and interleaved orders must not contaminate any earlier cutoff.
    for(int64_t t=127;t>=0;--t)same_encoding(saved[layout][t],encode(model,h,cutoffs(t),scaler));
    for(int64_t j=0;j<128;++j){const int64_t t=j%2?127-j/2:j/2;same_encoding(saved[layout][t],encode(model,h,cutoffs(t),scaler));}
  }
  for(int64_t t=0;t<128;++t){check(bits(saved[0][t].valid,saved[1][t].valid),"semantic channel permutation preserves support");check(torch::allclose(saved[0][t].native,saved[1][t].native,1e-5,1e-5),"separately declared semantic permutation abs/rel1e-5");}
}
template<class T>std::map<std::pair<int64_t,int64_t>,std::pair<torch::Tensor,torch::Tensor>> earlier_states(const T &out,int64_t boundary){
  const auto rows=out.patch_rows.to(torch::kCPU),positions=out.patch_positions.to(torch::kCPU),valid=out.patch_valid.to(torch::kCPU);
  std::map<std::pair<int64_t,int64_t>,std::pair<torch::Tensor,torch::Tensor>> result;
  for(int64_t r=0;r<positions.size(0);++r)for(int64_t p=0;p<positions.size(1);++p)if(valid[r][p].template item<bool>()&&positions[r][p].template item<int64_t>()<=boundary)
    result.emplace(std::make_pair(rows[r].template item<int64_t>(),positions[r][p].template item<int64_t>()),std::make_pair(out.temporal_hidden[r][p].detach().to(torch::kCPU),out.contextual_temporal_hidden[r][p].detach().to(torch::kCPU)));
  return result;
}
template<class Model>void internal_temporal_proof(Model &model,const cp::FrozenScaler &scaler,const cp::Config &config){
  auto h=history(128);h.observed.fill_(true);
  // Whole missing patches make packed ranks differ across channels.
  h.observed[0][0].narrow(0,8,8).fill_(false);h.observed[0][1].narrow(0,24,16).fill_(false);h.observed[1][2].narrow(0,48,24).fill_(false);
  const auto cuts=cutoffs(127);const auto baseline=trace(model,h,cuts,scaler);token_oracle(baseline,cuts,config);
  for(int64_t boundary=0;boundary<15;++boundary)for(int variant=0;variant<2;++variant){auto changed=clone(h);const auto begin=(boundary+1)*8;
    changed.data.narrow(2,begin,128-begin).mul_(-9.).add_(123.);
    if(variant)for(int64_t b=0;b<3;++b)for(int64_t ch=0;ch<3;++ch){const auto drop=std::min<int64_t>(16,128-begin);if((b+ch)%2==0)changed.observed[b][ch].narrow(0,begin,drop).fill_(false);}
    const auto tested=trace(model,changed,cuts,scaler);const auto expected=earlier_states(baseline,boundary),actual=earlier_states(tested,boundary);
    check(!expected.empty()&&expected.size()==actual.size(),"same completed original temporal query tokens");
    for(const auto &[key,v]:expected){check(actual.count(key)&&bits(v.first,actual.at(key).first),"ordinary temporal state cannot use later original patches");check(bits(v.second,actual.at(key).second),"channel-mixed temporal state cannot use later original patches");}
    ++internal_cases;
  }
}
template<class Model>void longer_and_structural(Model &model,const cp::FrozenScaler &scaler,const cp::Config &config){
  const auto h=history(193);const std::array<std::array<int64_t,3>,5> plans{{{0,7,15},{8,63,127},{127,128,192},{192,160,130},{1,81,191}}};
  for(const auto &plan:plans){const auto cuts=torch::tensor(std::vector<double>{plan[0]*60000.,plan[1]*60000.,plan[2]*60000.},torch::kFloat64);const auto baseline=trace(model,h,cuts,scaler);retained_oracle(baseline,h,cuts,config);token_oracle(baseline,cuts,config);
    for(int variant=0;variant<9;++variant){same_trace(baseline,trace(model,intervention(h,cuts,variant),cuts,scaler));++suffix_cases;}
    auto alternate=clone(h);alternate.timestamps=h.timestamps.unsqueeze(0).expand({3,193}).clone();same_trace(baseline,trace(model,alternate,cuts,scaler));
  }
  const auto cuts=cutoffs(192);const auto baseline=trace(model,h,cuts,scaler);auto discarded=clone(h);
  discarded.data.narrow(2,0,65).fill_(std::numeric_limits<double>::quiet_NaN());discarded.observed.narrow(2,0,65).fill_(true);
  same_trace(baseline,trace(model,discarded,cuts,scaler));
  cp::RawHistory cropped{h.data.narrow(2,65,128),h.observed.narrow(2,65,128),h.channel_ids,h.timestamps.narrow(0,65,128),h.sampling_interval};same_trace(baseline,trace(model,cropped,cuts,scaler));
  auto empty=clone(h);empty.observed.fill_(false);empty.data.fill_(std::numeric_limits<double>::infinity());const auto none=trace(model,empty,cuts,scaler);retained_oracle(none,empty,cuts,config);
  check(!none.encoding.valid.any().template item<bool>()&&none.encoding.native.eq(0).all().template item<bool>(),"no support, including poisoned storage, yields invalid zero");
  auto isolated=clone(h);isolated.observed.fill_(false);for(int64_t b=0;b<3;++b)isolated.observed[b][0][192][3]=true;
  const auto single=trace(model,isolated,cuts,scaler);check(!single.change_valid.any().template item<bool>()&&single.change_values.eq(0).all().template item<bool>(),"isolated visible cells never bridge change gaps");
  auto absent=clone(h);absent.observed.select(1,1).fill_(false);retained_oracle(trace(model,absent,cuts,scaler),absent,cuts,config);
  const auto short_h=history(1);retained_oracle(trace(model,short_h,cutoffs(0),scaler),short_h,cutoffs(0),config);
  auto ids=clone(h);ids.channel_ids=h.channel_ids.unsqueeze(0).expand({3,3}).clone();same_trace(baseline,trace(model,ids,cuts,scaler));
  rejects([&]{encode(model,h,cutoffs(-1),scaler);},"cutoff before provided grid rejected");rejects([&]{encode(model,h,cutoffs(193),scaler);},"cutoff after provided grid rejected");
  rejects([&]{encode(model,h,cuts+30000.,scaler);},"unaligned cutoff rejected");
  auto bad_cut=cuts.clone();bad_cut[1]=std::numeric_limits<double>::quiet_NaN();rejects([&]{encode(model,h,bad_cut,scaler);},"nonfinite cutoff rejected");
  auto bad=clone(h);bad.data=bad.data.to(torch::kFloat32);rejects([&]{encode(model,bad,cuts,scaler);},"raw F32 upcast contract refused");
  bad=clone(h);bad.observed=bad.observed.to(torch::kInt64);rejects([&]{encode(model,bad,cuts,scaler);},"nonbool mask rejected");
  bad=clone(h);bad.timestamps[3]=bad.timestamps[2];rejects([&]{encode(model,bad,cuts,scaler);},"nonuniform/repeated timestamp schema rejected");
  bad=clone(h);bad.channel_ids=torch::tensor({0,0,2},torch::kInt64);rejects([&]{encode(model,bad,cuts,scaler);},"duplicate semantic IDs rejected");
  bad=clone(h);bad.sampling_interval=1.;rejects([&]{encode(model,bad,cuts,scaler);},"incorrect interval rejected");
  for(const double poison:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::max()/4.}){
    auto visible=clone(h);visible.data[0][0][192][0]=poison;visible.observed[0][0][192][0]=true;rejects([&]{encode(model,visible,cuts,scaler);},"observed retained poison/overflow rejected");
  }
  auto hidden=clone(h);hidden.data.masked_fill_(hidden.observed.logical_not(),std::numeric_limits<double>::quiet_NaN());same_trace(baseline,trace(model,hidden,cuts,scaler),false);
  auto changed=clone(h);changed.data[0][0][192][3]+=123.;changed.observed[0][0][192][3]=true;const auto positive=trace(model,changed,cuts,scaler);
  check(!bits(baseline.encoding.native,positive.encoding.native),"legal past-value positive control changes native, excluding constant-output proof");
  auto mask_change=clone(h);mask_change.observed[0][0].fill_(false);const auto positive_mask=trace(model,mask_change,cuts,scaler);
  check(!bits(baseline.retained_observed,positive_mask.retained_observed)&&!bits(baseline.content_counts,positive_mask.content_counts),"legal past-mask positive control changes actual support");
}template<class Model>void external_scaler_and_time_schema(Model &model,const cp::FrozenScaler &scaler,const cp::Config &config){
  const auto h=history(128);const auto cuts=cutoffs(63);const auto baseline=trace(model,h,cuts,scaler);
  auto external=scaler;external.mean=scaler.mean.clone()+.125*scaler.scale;external.scale=scaler.scale.clone();external.count=scaler.count.clone();external.channel_ids=scaler.channel_ids.clone();external.floor_applied=scaler.floor_applied.clone();
  const FrozenState state(model,external);const auto altered=trace(model,h,cuts,external);
  check(!bits(baseline.normalized_parent,altered.normalized_parent),"external fixed scaler actually changes parent preprocessing");
  check(bits(baseline.change_values,altered.change_values)&&bits(baseline.conditioned_bank,altered.conditioned_bank)&&bits(baseline.encoding.native.narrow(1,32,4),altered.encoding.native.narrow(1,32,4)),"external scaler never refits/rebinds market relative bank");
  same_trace(altered,trace(model,intervention(h,cuts,6),cuts,external));++suffix_cases;state.verify(model,external);
  auto shifted=clone(h);shifted.timestamps=h.timestamps.unsqueeze(0).expand({3,128}).clone();auto requested=cuts.clone();
  for(int64_t b=0;b<3;++b){shifted.timestamps[b].add_(b*1000000000.);requested[b]+=b*1000000000.;}
  const auto result=trace(model,shifted,requested,scaler);retained_oracle(result,shifted,requested,config);token_oracle(result,requested,config);
  check(bits(result.encoding.native,baseline.encoding.native)&&bits(result.encoding.valid,baseline.encoding.valid)&&bits(result.encoding.cutoffs,requested),"absolute per-example time schema is reported without adding future content");
}
torch::Tensor text_tensor(const std::string &s){return torch::tensor(std::vector<uint8_t>(s.begin(),s.end()),torch::kUInt8);}
void copy_mutated(torch::serialize::InputArchive &in,torch::serialize::OutputArchive &out,const std::string &prefix,const std::string &target,const torch::Tensor &replacement,int &replaced){
  for(const auto &key:in.keys()){
    c10::IValue value;in.read(key,value);const auto path=prefix+key;
    if(value.isTensor()){if(path==target){++replaced;if(replacement.defined())out.write(key,replacement,true);}else out.write(key,value.toTensor(),true);}
    else if(value.isObject()){torch::serialize::InputArchive a;torch::serialize::OutputArchive b;in.read(key,a);copy_mutated(a,b,path+".",target,replacement,replaced);out.write(key,b);}else out.write(key,value);
  }
}
template<class Model>concept HasUngatedEncode=requires(Model &m,const cp::RawHistory &h){m.encode(h);};
static_assert(!HasUngatedEncode<cp::EncoderImpl>,"public Encoder must not expose an unguarded encode");

namespace tr=cp::training;
cp::RawHistory learning_history(const cp::FrozenScaler &s){
  auto h=history(193,8);h.observed.fill_(true);auto x=h.data.accessor<double,4>();const auto mean=s.mean.accessor<double,2>(),scale=s.scale.accessor<double,2>();
  for(int64_t b=0;b<8;++b)for(int64_t c=0;c<3;++c)for(int64_t t=0;t<193;++t)for(int64_t f=0;f<9;++f)
    x[b][c][t][f]=mean[c][f]+scale[c][f]*(.2*std::sin(.021*t+.27*c+.11*f+.19*b)+.03*std::cos(.061*t+.2*b));
  return h;
}
std::string group_of(const std::string &name){
  if(name=="core.uncertainty_projection")return "uncertainty_projection";
  if(name=="core.encoder.change_projection")return "change_projection";
  if(name=="core.encoder.change_decoder")return "residual_decoder";
  if(name=="core.variance_projection")return "variance_matrix";
  if(name=="core.variance_bias")return "variance_bias";
  if(name.find("relation_projection.weight")!=std::string::npos)return "content_relation";
  if(name.find(".decoder_")!=std::string::npos)return "parent_decoder";
  if(name.find(".block_")!=std::string::npos)return "temporal_attention";
  if(name.find(".global_pool_")!=std::string::npos)return "global_pool";
  return "other_learned";
}
using LearningCounts=std::map<std::string,std::pair<int64_t,int64_t>>;
void temporal_learning_counts(const cp::Encoder &model,const std::map<std::string,torch::Tensor> &before,LearningCounts &counts,bool gradient){
  const auto named=model->named_parameters();const auto w=named["core.encoder.level.base.backbone.export_projection.weight"],b=named["core.encoder.level.base.backbone.export_projection.bias"],p=named["core.encoder.level.base.backbone.pool_positions.weight"];
  for(int64_t bin=0;bin<4;++bin){bool export_event=false,position_event=false;
    if(gradient){check(w.grad().defined()&&b.grad().defined()&&p.grad().defined(),"actual temporal group data gradients");export_event=w.grad().narrow(0,bin*8,8).ne(0).any().item<bool>()&&b.grad().narrow(0,bin*8,8).ne(0).any().item<bool>();const auto g=p.grad().narrow(0,bin*4,4).to(torch::kFloat64);position_event=(g-g.mean()).ne(0).any().item<bool>();}
    else{export_event=!bits(w.narrow(0,bin*8,8),before.at("core.encoder.level.base.backbone.export_projection.weight").narrow(0,bin*8,8))&&!bits(b.narrow(0,bin*8,8),before.at("core.encoder.level.base.backbone.export_projection.bias").narrow(0,bin*8,8));const auto delta=(p.detach().to(torch::kFloat64)-before.at("core.encoder.level.base.backbone.pool_positions.weight").to(torch::kFloat64)).narrow(0,bin*4,4);position_event=(delta-delta.mean()).ne(0).any().item<bool>();}
    auto &e=counts["temporal_export_bin_"+std::to_string(bin)],&q=counts["temporal_position_bin_"+std::to_string(bin)];if(gradient){e.first+=export_event;q.first+=position_event;}else{e.second+=export_event;q.second+=position_event;}
  }
  const auto value=gradient?p.grad().detach().to(torch::kFloat64):(p.detach().to(torch::kFloat64)-before.at("core.encoder.level.base.backbone.pool_positions.weight").to(torch::kFloat64));
  const auto means=value.reshape({4,4}).mean(1);
  const auto signs=torch::tensor(std::vector<double>{1,1,-1,-1,1,-1,1,-1,1,-1,-1,1},means.options()).reshape({3,4});
  const auto coefficients=torch::matmul(signs,means);
  check(torch::isfinite(coefficients).all().item<bool>()&&torch::allclose(coefficients.square().sum(),4.*(means-means.mean()).square().sum(),1e-10,1e-12),"F64 Hadamard full-position between-bin projection identity");
  for(int64_t direction=0;direction<3;++direction){auto &entry=counts["temporal_between_direction_"+std::to_string(direction)];const bool event=coefficients[direction].item<double>()!=0.;if(gradient)entry.first+=event;else entry.second+=event;}
}
double step(cp::Encoder &model,torch::optim::AdamW &optimizer,tr::TrainingProgress &progress,const cp::FrozenScaler &scaler,LearningCounts &counts,bool test_future=false){
  model->train();optimizer.zero_grad();auto h=learning_history(scaler);const auto cuts=cutoffs(127,8);
  const auto retained=cp::boundary::retain(h,cuts,model->config());
  const auto hidden=cp::implementation::configurable_spectral_make_training_mask(retained.raw.observed.to(torch::kCUDA),model->config().level,tr::training_counter_seed(progress.mask_seed,progress.attempted_batches));
  auto context=torch::zeros_like(hidden.visible);context.select(2,0).copy_(hidden.visible.select(2,0));
  auto forward=model->forward_at(h,cuts,hidden.hidden,scaler,context);
  check(forward.output.loss.is_cuda()&&forward.output.loss.scalar_type()==torch::kFloat64&&torch::isfinite(forward.output.loss).item<bool>()&&
      !forward.nll.loss.requires_grad()&&bits(forward.output.loss,forward.objective.total_loss),"actual explicit CUDA F64 objective and reporting-only Gaussian NLL");
  check(bits(forward.target_mask,hidden.target)&&bits(forward.visible,hidden.visible&context.logical_not()),"context deletion changes only visible cells, never query targets");
  check(!(forward.visible&forward.target_mask).any().item<bool>(),"visible input and query targets disjoint");
  if(test_future){auto poison=intervention(h,cuts,6);auto other=model->forward_at(poison,cuts,hidden.hidden,scaler,context);
    check(bits(forward.output.encoding.z_contextual_global,other.output.encoding.z_contextual_global)&&bits(forward.targets,other.targets)&&bits(forward.target_mask,other.target_mask)&&bits(forward.output.loss,other.output.loss),"training boundary crops future values/masks before targets/masks/loss");
  }
  std::map<std::string,torch::Tensor> before;for(const auto &p:model->named_parameters())before.emplace(p.key(),p.value().detach().clone());
  forward.output.loss.backward();bool any=false;
  for(const auto &p:model->named_parameters()){
    check(p.value().is_cuda(),"actual CUDA model parameter");
    if(!p.value().requires_grad()){check(!p.value().grad().defined(),"literal spectral prior has no gradient");continue;}
    const auto g=p.value().grad();if(!g.defined())continue;
    check(g.is_cuda()&&torch::isfinite(g).all().item<bool>(),"finite actual CUDA loss gradient");
    if(g.ne(0).any().item<bool>()){if(p.key()!="core.encoder.level.base.backbone.pool_score.bias")++counts[group_of(p.key())].first;any=true;}
  }
  temporal_learning_counts(model,before,counts,true);
  check(any,"actual nonzero loss gradients");torch::nn::utils::clip_grad_norm_(tr::learned_parameters(model),1.);optimizer.step();
  ++progress.completed_updates;++progress.attempted_batches;model->set_causal_updates(progress.completed_updates);
  for(const auto &p:model->named_parameters()){
    check(torch::isfinite(p.value()).all().item<bool>(),"finite updated CUDA weight");
    if(!p.value().requires_grad())check(bits(before.at(p.key()),p.value()),"literal spectral prior bitwise unchanged");
    else if(p.key()!="core.encoder.level.base.backbone.pool_score.bias"&&!bits(before.at(p.key()),p.value()))++counts[group_of(p.key())].second;
  }
  temporal_learning_counts(model,before,counts,false);
  tr::validate_optimizer(model,optimizer,progress);model->clear_gradients();return forward.output.loss.item<double>();
}
void compare_optimizer(const cp::Encoder &a,const torch::optim::AdamW &oa,const cp::Encoder &b,const torch::optim::AdamW &ob){
  check(a->state_identity()==b->state_identity(),"all model parameters/buffers exact after typed resume");const auto pa=tr::learned_parameters(a),pb=tr::learned_parameters(b);check(pa.size()==pb.size(),"resume learned parameter count");
  for(size_t i=0;i<pa.size();++i){const auto ia=oa.state().find(pa[i].unsafeGetTensorImpl()),ib=ob.state().find(pb[i].unsafeGetTensorImpl());
    check((ia==oa.state().end())==(ib==ob.state().end()),"optimizer initialized-state presence exact");if(ia==oa.state().end())continue;
    const auto *sa=dynamic_cast<const torch::optim::AdamWParamState *>(ia->second.get()),*sb=dynamic_cast<const torch::optim::AdamWParamState *>(ib->second.get());
    check(sa&&sb&&sa->step()==sb->step()&&bits(sa->exp_avg(),sb->exp_avg())&&bits(sa->exp_avg_sq(),sb->exp_avg_sq()),"typed optimizer steps/moments bitwise exact");
  }
}
int64_t mutation_refusals(const fs::path &checkpoint,const fs::path &training,const fs::path &out,const cp::Config &config,const tr::TrainingIdentity &identity){
  int64_t negatives=0;
  const auto attempt=[&](const fs::path &input,const std::string &key,const torch::Tensor &replacement,bool is_training){
    const auto path=out/("refusal-"+std::to_string(++negatives)+".pt");torch::serialize::InputArchive a;torch::serialize::OutputArchive b;a.load_from(input.string(),torch::kCPU);int changed=0;
    copy_mutated(a,b,"",key,replacement,changed);check(changed==1,"single actual typed archive field mutated");b.save_to(path.string());
    rejects([&]{if(is_training)(void)tr::load_training_state(path.string(),config,identity);else(void)cp::load_checkpoint(path.string(),config.level.device,&config);},"wrong typed binding refuses: "+key);
  };
  for(const auto &[key,value]:std::vector<std::pair<std::string,torch::Tensor>>{
    {"artifact_kind",text_tensor("cpb_v1_inference_checkpoint_v1")},{"model_tag",text_tensor("CPB-v1")},{"causal_policy",text_tensor("unmasked")},

    {"training_policy",text_tensor("pretrained-parent")},{"initialization_kind",text_tensor("pretrained-import")},
    {"gradient_policy",text_tensor("variance_input_gradient_stop-v1")},{"payload.gradient_policy",text_tensor("variance_input_gradient_stop-v1")},
    {"native_layout",text_tensor("parent32+shared-change4-native36")},{"payload.native_layout",text_tensor("parent32+shared-change4-native36")},
    {"uncertainty_width",torch::tensor(int64_t{0},torch::kInt64)},{"payload.uncertainty_width",torch::tensor(int64_t{0},torch::kInt64)},
    {"pretrained_ancestor",text_tensor("CPB-v1")},{"source_updates",torch::tensor(int64_t{2048},torch::kInt64)},
    {"initialization_seed",torch::tensor(int64_t{1},torch::kInt64)},{"initial_state_identity",text_tensor("changed-seeded-initial-state")},
    {"scaler_identity",text_tensor("wrong-scaler")},{"normalizer_identity",text_tensor("wrong-bank")},
    {"payload.scaler_identity",text_tensor("wrong-nested-scaler")},{"payload.normalizer_identity",text_tensor("wrong-nested-bank")},
    {"causal_updates",torch::tensor(int64_t{999},torch::kInt64)},{"state_identity",text_tensor("changed-state")}})attempt(checkpoint,key,value,false);
  for(const auto &[key,value]:std::vector<std::pair<std::string,torch::Tensor>>{
    {"artifact_kind",text_tensor("rpb_location_scale_decoder_training_state_v1")},{"model_tag",text_tensor("CPB-v1")},
    {"native_layout",text_tensor("parent32+shared-change4-native36")},{"gradient_policy",text_tensor("variance_input_gradient_stop-v1")},
    {"counter_contract",text_tensor("reset-rng")},{"input_source_identity",text_tensor("wrong-data")},{"split_identity",text_tensor("wrong-split")},
    {"recipe_identity",text_tensor("wrong-recipe")},{"parameter_order",text_tensor("remapped")},{"optimizer_parameter_keys",text_tensor("remapped")},
    {"progress_v1",torch::tensor(std::vector<int64_t>{1,1,999,1103,1104},torch::kInt64)},
    {"optimizer_steps_v1",torch::zeros({int64_t{76}},torch::kInt64)}})attempt(training,key,value,true);
  const std::vector<std::pair<std::string,torch::Tensor>> wrong_objective{
    {"objective_policy",text_tensor("full-joint-Gaussian-NLL-v5")},
    {"mean_huber_delta_f64",torch::tensor(2.,torch::kFloat64)},
    {"loss_coefficients_f64",torch::tensor(std::vector<double>{1.,2.},torch::kFloat64)},
    {"variance_error_detached",torch::tensor(int64_t{0},torch::kInt64)},
    {"direct_gradient_routes",text_tensor("Gaussian-NLL-weights-mean")},
    {"reporting_nll_backpropagated",torch::tensor(int64_t{1},torch::kInt64)}};
  for(const auto &prefix:{std::string{},std::string{"payload."}})for(const auto &[key,value]:wrong_objective){
    attempt(checkpoint,prefix+key,value,false);attempt(checkpoint,prefix+key,torch::Tensor{},false);
  }
  for(const auto &[key,value]:wrong_objective){attempt(training,key,value,true);attempt(training,key,torch::Tensor{},true);}
  const std::vector<std::pair<std::string,torch::Tensor>> wrong_pool{
    {"temporal_pooling_policy",text_tensor("single-global-softmax-v6")},
    {"temporal_pooling_policy",text_tensor("original-position-four-bin-disjoint-export-rows-v1")},
    {"temporal_pooling_policy",torch::tensor(int64_t{1},torch::kInt64)},
    {"temporal_pooling_policy",text_tensor(cp::kTemporalPoolingPolicy).reshape({1,-1})},
    {"temporal_pooling_mix_f64",torch::tensor(std::vector<double>{1.,0.},torch::kFloat64)},
    {"temporal_pooling_mix_f64",torch::tensor(std::vector<float>{.5f,.5f},torch::kFloat32)},
    {"temporal_pooling_mix_f64",torch::tensor(std::vector<double>{.5},torch::kFloat64)},
    {"empty_local_bin_policy",text_tensor("fill-empty-slices-from-global")},
    {"empty_local_bin_policy",torch::tensor(int64_t{1},torch::kInt64)},
    {"empty_local_bin_policy",text_tensor(cp::kEmptyLocalBinPolicy).reshape({1,-1})},
    {"temporal_pooling_partition_v1",torch::tensor(std::vector<int64_t>{4,16,32,8,4},torch::kInt64)},
    {"temporal_pooling_partition_v1",torch::tensor(std::vector<double>{4,16,32,4,8},torch::kFloat64)},
    {"temporal_pooling_partition_v1",torch::tensor(std::vector<int64_t>{4,16,32,4},torch::kInt64)}};
  for(const auto &prefix:{std::string{},std::string{"payload."}})for(const auto &[key,value]:wrong_pool){
    attempt(checkpoint,prefix+key,value,false);attempt(checkpoint,prefix+key,torch::Tensor{},false);
  }
  for(const auto &prefix:{std::string{},std::string{"encoder."},std::string{"encoder.payload."}})for(const auto &[key,value]:wrong_pool){
    attempt(training,prefix+key,value,true);attempt(training,prefix+key,torch::Tensor{},true);
  }
  attempt(checkpoint,"artifact_kind",text_tensor("cpb_v6_inference_checkpoint_v1"),false);
  attempt(checkpoint,"artifact_kind",text_tensor("cpb_v8_inference_checkpoint_v1"),false);
  attempt(checkpoint,"payload.artifact_kind",text_tensor("cpb_v8_location_scale_decoder_payload_v1"),false);
  attempt(training,"artifact_kind",text_tensor("cpb_v8_causal_training_state_v1"),true);
  attempt(checkpoint,"payload.artifact_kind",text_tensor("cpb_v6_location_scale_decoder_payload_v1"),false);
  attempt(training,"artifact_kind",text_tensor("cpb_v6_causal_training_state_v1"),true);
  attempt(checkpoint,"payload.artifact_kind",text_tensor("cpb_v5_location_scale_decoder_payload_v1"),false);
  attempt(training,"artifact_kind",text_tensor("cpb_v5_causal_training_state_v1"),true);
  {torch::serialize::InputArchive input,optimizer,states,state;input.load_from(training.string(),torch::kCPU);input.read("optimizer",optimizer);optimizer.read("state",states);
    const auto keys=states.keys();check(!keys.empty(),"saved actual optimizer moments for negative test");states.read(keys.front(),state);torch::Tensor moment;state.read("exp_avg",moment);
    const auto changed=moment.clone()+1.;attempt(training,"optimizer.state."+keys.front()+".exp_avg",changed,true);
  }
  rejects([&]{(void)tr::load_training_state(checkpoint.string(),config,identity);},"inference cannot resume training");++negatives;
  rejects([&]{(void)cp::load_checkpoint(training.string(),config.level.device,&config);},"training cannot load as inference");++negatives;
  auto wrong=identity;wrong.recipe_identity+="-wrong";rejects([&]{(void)tr::load_training_state(training.string(),config,wrong);},"requested recipe mismatch");++negatives;
  auto wrong_config=config;wrong_config.change_width=8;rejects([&]{(void)tr::load_training_state(training.string(),wrong_config,identity);},"requested geometry mismatch");++negatives;
  return negatives;
}
void proof_stage(cp::Encoder &model,const cp::FrozenScaler &scaler,const cp::Config &config,const fs::path &out,const std::string &name){
  model->eval();model->clear_gradients();const FrozenState frozen(model,scaler);const auto old_suffix=suffix_cases,old_internal=internal_cases;
  std::vector<std::string> witnesses;exhaustive_suffix(model,scaler,config,witnesses);frozen.verify(model,scaler);
  longer_and_structural(model,scaler,config);frozen.verify(model,scaler);internal_temporal_proof(model,scaler,config);frozen.verify(model,scaler);
  external_scaler_and_time_schema(model,scaler,config);frozen.verify(model,scaler);
  check(suffix_cases-old_suffix==2350&&internal_cases-old_internal==30&&witnesses.size()==20,"full causal matrix at declared stage");
  std::ostringstream json;json<<"{\"status\":\"passed\",\"stage\":"<<quote(name)<<",\"all_cutoffs\":128,\"suffix_cases\":2350,\"internal_temporal_cases\":30,\"state_unchanged\":true,\"bounded_witnesses\":[";
  for(size_t i=0;i<witnesses.size();++i){if(i)json<<',';json<<witnesses[i];}json<<"]}";write_new(out/(name+"-causality.json"),json.str());
}
struct FixtureStats{cp::FrozenScaler scaler;cp::BankStandardizer bank;};
FixtureStats fixture_train_statistics(const cp::Config &config){
  // This generated cohort is solely an engineering TRAIN fixture. No old file,
  // checkpoint, scaler, normalizer or quality population is read to fit it.
  const auto train=history(128,32);const auto retained=cp::boundary::retain(train,cutoffs(127,32),config);
  const auto rounded=torch::where(retained.raw.observed,retained.raw.data,torch::zeros_like(retained.raw.data)).to(torch::kFloat32);
  const cp::implementation::Input fit_input{rounded,retained.raw.observed,retained.raw.channel_ids,retained.raw.endpoints,retained.raw.sampling_interval};
  auto scaler=cp::implementation::fit_configurable_spectral_scaler(fit_input,config.level);
  cp::BankStandardizer bank(config.level,"cpb-v9-fixture32-TRAIN-only-no-quality-sources",scaler.identity());
  bank.add(cp::implementation::relative_change_targets(retained.raw,config.level));bank.freeze();const auto id=scaler.identity();bank.validate(config.level,nullptr,&id);
  return{scaler,bank};
}
int execute(int argc,char **argv){
  torch::set_num_threads(1);torch::set_num_interop_threads(1);check(argc==7,"usage: v9_test --settings CONFIG --output NEWLEAF --seed SEED (or --saved-checkpoint FILE)");
  std::map<std::string,fs::path> args;for(int i=1;i<argc;i+=2){const std::string key=argv[i];check(key=="--settings"||key=="--output"||key=="--seed"||key=="--saved-checkpoint"||key=="--final-checkpoint","only fresh or own saved-checkpoint modes are accepted");check(args.emplace(key,argv[i+1]).second,"unique options");}
  for(const auto key:{"--settings","--output"})check(args.count(key),"required CLI option");
  const bool saved=args.count("--saved-checkpoint")||args.count("--final-checkpoint");check(saved!=bool(args.count("--seed")),"choose fresh seed OR own saved checkpoint");
  const auto settings=args.at("--settings"),out=args.at("--output");check(!fs::exists(out)&&fs::is_regular_file(settings),"own settings and exclusive output");check(fs::create_directories(out),"create proof leaf");
  std::cout<<"CPB_V9_PROOF_DIRECTORY="<<out.string()<<std::endl;check(torch::cuda::is_available(),"actual CUDA required");const auto config=cp::read_settings(settings);
  check(config.level.device==torch::Device(torch::kCUDA,0)&&config.level.channel_count==3&&config.level.history_length==128&&config.level.input_width==9&&config.level.patch_length==8&&config.change_width==4&&config.uncertainty_width==4,"prospective C3H128F9P8/native40");
  check(cp::temporal_pooling_partition(config)==std::vector<int64_t>({4,16,32,4,8})&&cp::temporal_pooling_mix()==std::vector<double>({.5,.5})&&std::string(cp::kEmptyLocalBinPolicy)=="zero-whole-export-slice-after-bias-v1","prospective original-position half-mixture partition/mix/empty policy");
  if(saved){
    const auto source=args.at(args.count("--saved-checkpoint")?"--saved-checkpoint":"--final-checkpoint");const auto hash=cp::boundary::sha256_file(source);auto loaded=cp::load_checkpoint(source.string(),config.level.device,&config);
    check(loaded.source_updates==0&&(loaded.causal_updates==0||loaded.causal_updates==2048),"fresh ancestry and declared initial/final counter only");
    if(args.count("--final-checkpoint"))check(loaded.causal_updates==2048,"final alias requires2048 updates");
    const FrozenState before(loaded.model,loaded.scaler);const auto state=loaded.model->state_identity();const auto stage=loaded.causal_updates==0?"initial0":"final2048";
    proof_stage(loaded.model,loaded.scaler,config,out,stage);before.verify(loaded.model,loaded.scaler);check(state==loaded.model->state_identity()&&cp::boundary::sha256_file(source)==hash,"saved artifact/state unchanged");
    std::ostringstream json;json<<"{\"kind\":\"cpb_v9_saved_checkpoint_causal_proof_v1\",\"status\":\"passed\","<<objective_json_fields()<<",\"gradient_policy\":"<<quote(cp::kGradientPolicy)<<",\"model\":\"CPB-v9\",\"checkpoint\":"<<quote(source.string())<<",\"checkpoint_sha256\":"<<quote(hash)
      <<",\"source_updates\":0,\"causal_updates\":"<<loaded.causal_updates<<",\"initialization_seed\":"<<loaded.initialization_seed<<",\"pretrained_ancestor\":\"none\",\"new_training_updates\":0,\"all_cutoffs\":128,\"suffix_cases\":2350,\"internal_temporal_cases\":30,\"state_unchanged\":true,\"future_values_and_masks_invariant\":true,\"CPU_encoder_calls\":0,\"CUDA_encoder_calls\":"<<model_calls<<",\"head_fits\":0,\"checks\":"<<checks<<'}';
    write_new(out/"proof.json",json.str());std::cout<<"CPB_V9_SAVED_CAUSAL_PROOF_PASSED updates="<<loaded.causal_updates<<" checks="<<checks<<std::endl;return 0;
  }
  size_t consumed=0;const auto seed=std::stoll(args.at("--seed").string(),&consumed);check(seed>=0&&consumed==args.at("--seed").string().size(),"nonnegative explicit initialization seed");
  auto statistics=fixture_train_statistics(config);auto loaded=cp::fresh(config,statistics.scaler,statistics.bank,seed);
  check(loaded.source_updates==0&&loaded.causal_updates==0&&loaded.initialization_seed==seed,"zero imported ancestry and correct fresh counter/seed");
  const FrozenState before(loaded.model,loaded.scaler);
  // Independently repeat only construction with the same seed BEFORE loading any
  // checkpoint. This is a constructor parity witness, not encoder training.
  torch::manual_seed(seed);auto direct_core=cp::implementation::LocationScaleDecoderModel(config,statistics.bank);auto direct=cp::Encoder(direct_core,seed,statistics.scaler.identity());before.verify(direct,statistics.scaler);
  check(direct->state_identity()==loaded.model->initial_state_identity()&&direct->state_identity()==loaded.model->state_identity(),"every fresh learned tensor equals independent seeded constructor");
  const auto initial=out/"inference-0.pt";cp::save_checkpoint(initial.string(),loaded.model,loaded.scaler,0);
  auto restored=cp::load_checkpoint(initial.string(),config.level.device,&config);before.verify(restored.model,restored.scaler);
  std::ostringstream init;init<<"{\"kind\":\"cpb_v9_fresh_initialization_proof_v1\",\"status\":\"passed\","<<objective_json_fields()<<",\"gradient_policy\":"<<quote(cp::kGradientPolicy)<<",\"initialization_kind\":"<<quote(cp::kInitializationKind)<<",\"initialization_seed\":"<<seed
    <<",\"source_updates\":0,\"causal_updates\":0,\"pretrained_ancestor\":\"none\",\"old_checkpoints_opened\":0,\"checkpoint_reads_before_first_save\":0,\"learned_tensors_from_seeded_constructor\":76,\"learned_values\":294752,\"literal_fixed_values\":3888,\"all_initial_tensors_equal_independent_constructor\":true,\"first_typed_artifact_exact\":true,\"scaler_fit_source\":\"engineering TRAIN-only generated32\",\"bank_fit_source\":\"same engineering TRAIN-only generated32\",\"scaler_identity\":"<<quote(loaded.scaler.identity())<<",\"bank_identity\":"<<quote(loaded.model->normalizer().identity())<<",\"initial_state_identity\":"<<quote(loaded.model->initial_state_identity())<<",\"initial_checkpoint_sha256\":"<<quote(cp::boundary::sha256_file(initial))<<'}';write_new(out/"initialization-proof.json",init.str());
  proof_stage(loaded.model,loaded.scaler,config,out,"initialized");
  const tr::TrainingIdentity identity{"artificial-engineering-input-v9","TRAIN-only-fixture-v9","from-scratch-causal-all-learned-huber1-mean-detached-error-variance-engineering-v9;delta1;unitweights;shared-global-clip1;reporting-NLL-never-backward"};tr::TrainingProgress progress{0,0,1102,1103,1104};
  auto optimizer=tr::make_optimizer(loaded.model);check(optimizer->state().empty(),"fresh optimizer has no inherited moments");tr::validate_optimizer(loaded.model,*optimizer,progress);LearningCounts counts;
  const auto before_native=encode(loaded.model,history(128),cutoffs(127),loaded.scaler).native.detach().clone();
  const auto initial_positions=loaded.model->named_parameters()["core.encoder.level.base.backbone.pool_positions.weight"].detach().to(torch::kFloat64).clone();
  const double loss1=step(loaded.model,*optimizer,progress,loaded.scaler,counts,true);const auto training=out/"training-1.pt";tr::save_training_state(training.string(),loaded.model,loaded.scaler,*optimizer,identity,progress);
  auto resumed=tr::load_training_state(training.string(),config,identity);compare_optimizer(loaded.model,*optimizer,resumed.model,*resumed.optimizer);
  check(resumed.progress.completed_updates==progress.completed_updates&&resumed.progress.attempted_batches==progress.attempted_batches&&resumed.progress.row_seed==progress.row_seed&&resumed.progress.mask_seed==progress.mask_seed&&resumed.progress.context_seed==progress.context_seed,"all counters/seeds exact");
  for(const auto stream_seed:{progress.row_seed,progress.mask_seed,progress.context_seed})check(tr::training_counter_seed(stream_seed,progress.attempted_batches)==tr::training_counter_seed(stream_seed,resumed.progress.attempted_batches),"next row/mask/context stream exact");
  const double loss2=step(loaded.model,*optimizer,progress,loaded.scaler,counts);LearningCounts resumed_counts;const double resumed_loss=step(resumed.model,*resumed.optimizer,resumed.progress,resumed.scaler,resumed_counts);
  check(loss2==resumed_loss,"exact resumed next CUDA loss");compare_optimizer(loaded.model,*optimizer,resumed.model,*resumed.optimizer);
  for(const auto *name:{"temporal_attention","global_pool","parent_decoder","content_relation","change_projection","residual_decoder","uncertainty_projection","variance_matrix","variance_bias"})check(counts[name].first>0&&counts[name].second>0,"actual CUDA loss gradients and weight movement: "+std::string(name));
  for(int64_t bin=0;bin<4;++bin)for(const auto &prefix:{"temporal_export_bin_","temporal_position_bin_"}){const auto name=std::string(prefix)+std::to_string(bin);check(counts[name].first>0&&counts[name].second>0,"actual data gradients and observed movement for identifiable original bin: "+name);}
  for(int64_t direction=0;direction<3;++direction){const auto name="temporal_between_direction_"+std::to_string(direction);check(counts[name].first>0&&counts[name].second>0,"actual nonzero data-gradient and observed movement for Hadamard direction: "+name);}
  const auto position_change=loaded.model->named_parameters()["core.encoder.level.base.backbone.pool_positions.weight"].detach().to(torch::kFloat64)-initial_positions;
  const auto position_bin_means=position_change.reshape({4,4}).mean(1);check((4.*(position_bin_means-position_bin_means.mean()).square().sum()).item<double>()>0.,"final full-position between-bin projection changed; decay alone is not data-learning proof");
  loaded.model->eval();loaded.model->clear_gradients();loaded.model->validate_frozen();const auto after_native=encode(loaded.model,history(128),cutoffs(127),loaded.scaler).native;
  check(!bits(before_native.narrow(1,0,32),after_native.narrow(1,0,32)),"fresh learned causal32 actually changes after updates");
  const auto &prior=loaded.model->core()->encoder->level->base->grouped_odd_relation_projection->weight;check(!prior.requires_grad()&&!prior.grad().defined(),"literal prior stays frozen");
  const FrozenState now(loaded.model,loaded.scaler);for(const auto &old:before.tensors)if(old.name.rfind("normalizer/",0)==0||old.name.rfind("scaler/",0)==0){bool found=false;for(const auto &value:now.tensors)if(value.name==old.name){check(bits(old.value,value.value),"fixture-fitted scaler/bank remain exact");found=true;}check(found,"immutable tensor retained");}
  proof_stage(loaded.model,loaded.scaler,config,out,"trained");const auto final=out/"inference-2.pt";cp::save_checkpoint(final.string(),loaded.model,loaded.scaler,2);
  auto final_reload=cp::load_checkpoint(final.string(),config.level.device,&config);const FrozenState final_state(loaded.model,loaded.scaler);final_state.verify(final_reload.model,final_reload.scaler);
  const auto negatives=mutation_refusals(final,training,out,config,identity);
  std::ostringstream groups;groups<<'{';bool first=true;for(const auto &[name,value]:counts){if(!first)groups<<',';first=false;groups<<quote(name)<<":{\"nonzero_gradient_parameter_events\":"<<value.first<<",\"weight_movement_parameter_events\":"<<value.second<<'}';}groups<<'}';
  std::ostringstream result;result<<std::setprecision(17)<<"{\"kind\":\"cpb_v9_fresh_cuda_engineering_v1\",\"status\":\"passed\","<<objective_json_fields()<<",\"gradient_policy\":"<<quote(cp::kGradientPolicy)<<",\"source_updates\":0,\"initialization_seed\":"<<seed<<",\"pretrained_ancestor\":\"none\",\"old_checkpoints_opened\":0,\"all_initial_tensors_equal_seeded_constructor\":true,\"first_typed_artifact_exact\":true,\"scaler_and_bank_fitted_fixture_TRAIN_only\":true,\"all_cutoffs_per_stage\":128,\"suffix_cases_per_stage\":2350,\"internal_temporal_cases_per_stage\":30,\"stages\":[\"initialized\",\"trained\"],\"causal_updates_primary_fixture\":2,\"resume_witness_updates\":1,\"quality_updates\":0,\"typed_negative_cases\":"<<negatives
    <<",\"typed_resume_next_loss_model_moments_bitwise_exact\":true,\"CPU_encoder_calls\":0,\"CUDA_encoder_calls\":"<<model_calls<<",\"head_fits\":0,\"checks\":"<<checks<<",\"loss_first\":"<<loss1<<",\"loss_next\":"<<loss2<<",\"all_four_temporal_export_row_groups_learned\":true,\"all_four_pool_position_contrasts_learned\":true,\"all_three_between_bin_position_directions_learned\":true,\"final_between_bin_position_projection_changed\":true,\"weight_decay_counted_as_data_gradient\":false,\"actual_cuda_learning\":"<<groups.str()<<'}';
  write_new(out/"proof.json",result.str());std::cout<<"CPB_V9_ENGINEERING_PASSED checks="<<checks<<" suffix_cases="<<suffix_cases<<" CUDA_encoder_calls="<<model_calls<<std::endl;return 0;
}
} // namespace
int main(int argc,char **argv)try{return execute(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<std::endl;return 1;}
