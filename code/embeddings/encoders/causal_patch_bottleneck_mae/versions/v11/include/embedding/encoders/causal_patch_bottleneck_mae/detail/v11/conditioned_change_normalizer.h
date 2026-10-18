// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v11/conditioned_change_bank.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v11/configurable_spectral_checkpoint.h"
#include <cstring>
#include <iomanip>
#include <sstream>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v11 {
// Fit once from the unchanged FLOAT32 bank. No encoder, input scaler or label
// is called by this accumulator. All sufficient statistics remain CPU F64.
struct BankStandardizer {
  BankStandardizer()=default;
  BankStandardizer(const Config &c,const std::string &source,const std::string &scaler):
      source_identity(source),input_scaler_identity(scaler){
    input_geometry={c.channel_count,c.history_length,c.input_width};
    const auto ids=resolved_channel_ids(c);input_geometry.insert(input_geometry.end(),ids.begin(),ids.end());
    sampling_interval=c.sampling_interval;
    const auto w=consecutive_change_bank_width(c);
    counts=torch::zeros({w},torch::kInt64);sum=torch::zeros({w},torch::kFloat64);sumsq=torch::zeros_like(sum);
    mean=torch::zeros_like(sum);scale=torch::ones_like(sum);support=torch::zeros({w},torch::kBool);validate(c);
  }
  static void validate_targets(const ConditionedChangeTargets &t,int64_t width){
    TORCH_CHECK(t.values.defined() && t.valid.defined() && t.values.dim()==2 && t.values.size(1)==width &&
        t.values.scalar_type()==torch::kFloat32 && t.valid.sizes()==t.values.sizes() &&
        t.valid.scalar_type()==torch::kBool && t.valid.device()==t.values.device() &&
        torch::isfinite(torch::where(t.valid,t.values,torch::zeros_like(t.values))).all().item<bool>(),
        "[rpb-v26 bank] finite eligible F32 values and bool validity required");
  }
  void add(const ConditionedChangeTargets &t){
    TORCH_CHECK(!frozen,"[rpb-v26 bank] frozen TRAIN normalizer cannot refit");validate_targets(t,counts.numel());
    const auto valid=t.valid.detach().to(torch::kCPU);
    const auto safe=torch::where(t.valid,t.values,torch::zeros_like(t.values)).detach().to(torch::kCPU,torch::kFloat64);
    const auto n=valid.sum(0);
    TORCH_CHECK(counts.le(std::numeric_limits<int64_t>::max()-n).all().item<bool>(),"[rpb-v26 bank] count overflow");
    counts+=n;sum+=safe.sum(0);sumsq+=safe.square().sum(0);
    TORCH_CHECK(torch::isfinite(sum).all().item<bool>() && torch::isfinite(sumsq).all().item<bool>(),"[rpb-v26 bank] accumulated overflow");
  }
  void freeze(){
    TORCH_CHECK(!frozen,"[rpb-v26 bank] normalizer freezes exactly once");
    support=counts.ge(kConditionedChangeSupportCount);
    const auto denominator=counts.clamp_min(1).to(torch::kFloat64),average=sum/denominator;
    mean=torch::where(support,average,torch::zeros_like(sum));
    scale=torch::where(support,(sumsq/denominator-average.square()).clamp_min(0).sqrt().clamp_min(floor),torch::ones_like(sum));
    frozen=true;
  }
  void validate(const Config &c,const std::string *expected_source=nullptr,const std::string *expected_scaler=nullptr)const{
    const auto w=consecutive_change_bank_width(c);
    std::vector<int64_t> geometry{c.channel_count,c.history_length,c.input_width};
    const auto ids=resolved_channel_ids(c);geometry.insert(geometry.end(),ids.begin(),ids.end());
    const auto identity_ok=[](const std::string &s){return !s.empty() && s.size()<=4096 && s.find('\0')==std::string::npos;};
    TORCH_CHECK(input_geometry==geometry && sampling_interval==c.sampling_interval,"[rpb-v26 bank] canonical geometry mismatch");
    TORCH_CHECK(identity_ok(source_identity) && identity_ok(input_scaler_identity) &&
        (!expected_source || source_identity==*expected_source) && (!expected_scaler || input_scaler_identity==*expected_scaler),
        "[rpb-v26 bank] TRAIN source/input scaler identity mismatch");
    TORCH_CHECK(layout_identity==kConditionedChangeBankLayout && recipe_identity==kConditionedChangeNormalizationRecipe &&
        floor==kConditionedChangeScaleFloor && minimum_support==kConditionedChangeSupportCount,"[rpb-v26 bank] normalization recipe mismatch");
    for(const auto &t:{counts,sum,sumsq,mean,scale,support})TORCH_CHECK(t.defined() && t.device().is_cpu() &&
        t.dim()==1 && t.numel()==w && !t.requires_grad(),"[rpb-v26 bank] CPU vector contract required");
    TORCH_CHECK(counts.scalar_type()==torch::kInt64 && support.scalar_type()==torch::kBool && counts.ge(0).all().item<bool>(),
        "[rpb-v26 bank] count/support dtype mismatch");
    for(const auto &t:{sum,sumsq,mean,scale})TORCH_CHECK(t.scalar_type()==torch::kFloat64 && torch::isfinite(t).all().item<bool>(),
        "[rpb-v26 bank] finite CPU F64 moments required");
    TORCH_CHECK(sumsq.ge(0).all().item<bool>() && scale.ge(floor).all().item<bool>() &&
        torch::where(counts.eq(0),sum,torch::zeros_like(sum)).eq(0).all().item<bool>() &&
        torch::where(counts.eq(0),sumsq,torch::zeros_like(sumsq)).eq(0).all().item<bool>(),"[rpb-v26 bank] invalid sufficient statistics");
    const auto expected_support=frozen?counts.ge(minimum_support):torch::zeros_like(support);
    const auto denominator=counts.clamp_min(1).to(torch::kFloat64),average=sum/denominator;
    const auto expected_mean=frozen?torch::where(expected_support,average,torch::zeros_like(sum)):torch::zeros_like(sum);
    const auto expected_scale=frozen?torch::where(expected_support,(sumsq/denominator-average.square()).clamp_min(0).sqrt().clamp_min(floor),
        torch::ones_like(sum)):torch::ones_like(sum);
    TORCH_CHECK(torch::equal(support,expected_support) && torch::equal(mean,expected_mean) && torch::equal(scale,expected_scale),
        "[rpb-v26 bank] inconsistent population mean/scale/support");
  }
  torch::Tensor transform(const ConditionedChangeTargets &t)const{
    TORCH_CHECK(frozen,"[rpb-v26 bank] frozen normalizer required");validate_targets(t,counts.numel());
    const auto eligible=t.valid.logical_and(support.to(t.values.device()).unsqueeze(0));
    // Sanitize before subtraction: hidden NaN/Inf must not enter arithmetic.
    const auto safe=torch::where(eligible,t.values,torch::zeros_like(t.values));
    const auto centered=safe-mean.to(t.values.device(),t.values.scalar_type());
    const auto normalized=centered/scale.to(t.values.device(),t.values.scalar_type());
    const auto out=torch::where(eligible,normalized,torch::zeros_like(normalized));
    TORCH_CHECK(torch::isfinite(out).all().item<bool>(),"[rpb-v26 bank] nonfinite conditioned bank");return out;
  }
  BankStandardizer clone()const{
    BankStandardizer out=*this;out.counts=counts.clone();out.sum=sum.clone();out.sumsq=sumsq.clone();
    out.mean=mean.clone();out.scale=scale.clone();out.support=support.clone();return out;
  }
  std::string identity()const{
    TORCH_CHECK(frozen,"[rpb-v26 bank] only frozen normalizers have content identities");
    uint64_t h=14695981039346656037ULL;
    const auto bytes=[&](const void *p,size_t n){const auto *v=static_cast<const uint8_t *>(p);for(size_t i=0;i<n;++i){h^=v[i];h*=1099511628211ULL;}};
    const auto text=[&](const std::string &s){const uint64_t n=s.size();bytes(&n,sizeof(n));bytes(s.data(),s.size());};
    text(source_identity);text(input_scaler_identity);text(layout_identity);text(recipe_identity);
    bytes(input_geometry.data(),input_geometry.size()*sizeof(int64_t));bytes(&sampling_interval,sizeof(sampling_interval));
    bytes(&floor,sizeof(floor));bytes(&minimum_support,sizeof(minimum_support));
    for(const auto &t:{counts,sum,sumsq,mean,scale,support}){
      const auto cpu=t.detach().to(torch::kCPU).contiguous();const int64_t dtype=int64_t(cpu.scalar_type()),n=cpu.numel();
      bytes(&dtype,sizeof(dtype));bytes(&n,sizeof(n));bytes(cpu.const_data_ptr(),cpu.nbytes());
    }
    std::ostringstream out;out<<"rpb-bank-standardizer-v1-fnv1a64-"<<std::hex<<std::setfill('0')<<std::setw(16)<<h;return out.str();
  }
  void save(torch::serialize::OutputArchive &a)const{
    namespace d=configurable_checkpoint_detail;TORCH_CHECK(frozen,"[rpb-v26 bank] frozen archive required");
    a.write("artifact_kind",d::text_tensor("rpb_conditioned_bank_standardizer_v1"),true);
    a.write("source_identity",d::text_tensor(source_identity),true);a.write("input_scaler_identity",d::text_tensor(input_scaler_identity),true);
    a.write("layout_identity",d::text_tensor(layout_identity),true);a.write("recipe_identity",d::text_tensor(recipe_identity),true);
    a.write("normalizer_identity",d::text_tensor(identity()),true);a.write("input_geometry",torch::tensor(input_geometry,torch::kInt64),true);
    a.write("sampling_interval",torch::tensor(sampling_interval,torch::kFloat64),true);a.write("floor",torch::tensor(floor,torch::kFloat64),true);
    a.write("minimum_support",torch::tensor(minimum_support,torch::kInt64),true);
    a.write("counts",counts,true);a.write("sum",sum,true);a.write("sumsq",sumsq,true);
    a.write("mean",mean,true);a.write("scale",scale,true);a.write("support",support,true);
  }
  static BankStandardizer load(torch::serialize::InputArchive &a,const Config &c,const std::string *source=nullptr,const std::string *scaler=nullptr){
    namespace d=configurable_checkpoint_detail;
    TORCH_CHECK(d::read_text(a,"artifact_kind")=="rpb_conditioned_bank_standardizer_v1","[rpb-v26 bank] typed normalizer required");
    BankStandardizer out;out.source_identity=d::read_text(a,"source_identity");out.input_scaler_identity=d::read_text(a,"input_scaler_identity");
    out.layout_identity=d::read_text(a,"layout_identity");out.recipe_identity=d::read_text(a,"recipe_identity");
    const auto id=d::read_text(a,"normalizer_identity");const auto geometry=d::read_vector(a,"input_geometry",torch::kInt64,3+c.channel_count);
    const auto *p=geometry.const_data_ptr<int64_t>();out.input_geometry.assign(p,p+geometry.numel());
    const auto scalar=[&](const char *key){torch::Tensor t;a.read(key,t,true);TORCH_CHECK(t.dim()==0 && t.scalar_type()==torch::kFloat64,
        "[rpb-v26 bank] F64 scalar required: ",key);return t.item<double>();};
    out.sampling_interval=scalar("sampling_interval");out.floor=scalar("floor");
    torch::Tensor minimum;a.read("minimum_support",minimum,true);
    TORCH_CHECK(minimum.dim()==0 && minimum.scalar_type()==torch::kInt64,"[rpb-v26 bank] I64 minimum required");out.minimum_support=minimum.item<int64_t>();
    a.read("counts",out.counts,true);a.read("sum",out.sum,true);a.read("sumsq",out.sumsq,true);
    a.read("mean",out.mean,true);a.read("scale",out.scale,true);a.read("support",out.support,true);
    out.counts=out.counts.to(torch::kCPU);out.sum=out.sum.to(torch::kCPU);out.sumsq=out.sumsq.to(torch::kCPU);
    out.mean=out.mean.to(torch::kCPU);out.scale=out.scale.to(torch::kCPU);out.support=out.support.to(torch::kCPU);
    out.frozen=true;out.validate(c,source,scaler);TORCH_CHECK(out.identity()==id,"[rpb-v26 bank] content identity mismatch");return out;
  }
  torch::Tensor counts,sum,sumsq,mean,scale,support;
  std::string source_identity,input_scaler_identity,layout_identity{kConditionedChangeBankLayout},recipe_identity{kConditionedChangeNormalizationRecipe};
  std::vector<int64_t> input_geometry;double sampling_interval{0},floor{kConditionedChangeScaleFloor};
  int64_t minimum_support{kConditionedChangeSupportCount};bool frozen{false};
};
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v11
