// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_model.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_checkpoint.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
struct EnergyChangeMomentTargets{torch::Tensor values,valid;};
// Values are the existing bank verbatim. Eligibility is determined separately
// from actual adjacent visible endpoints, including legitimately zero energy.
inline EnergyChangeMomentTargets energy_change_moment_targets(const Input &visible,const Config &c){
  const auto bank=consecutive_change_bank(visible,c);
  const auto B=visible.data.size(0),C=c.channel_count,H=c.history_length,F=c.input_width;
  auto mask=visible.observed.to(c.device);
  const auto order=channel_indices(visible.channel_ids,c,B,c.device).argsort(int64_t{1});
  mask=mask.gather(1,order.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F}));
  const auto pairs=mask.narrow(2,1,H-1).logical_and(mask.narrow(2,0,H-1));
  const auto global=pairs.sum(2).ge(kConsecutiveChangeMinimumCount);
  std::vector<torch::Tensor> valid{global,global};
  for(int64_t q=0;q<4;++q){
    const auto begin=q*(H-1)/4,end=(q+1)*(H-1)/4;
    const auto part=begin==end?torch::zeros_like(global):
        global.logical_and(pairs.narrow(2,begin,end-begin).sum(2).ge(kConsecutiveChangeMinimumCount));
    valid.push_back(part);valid.push_back(part);
  }
  return{bank.values.detach(),torch::stack(valid,-1).flatten(1).detach()};
}

// Streaming sufficient statistics reside on CPU F64. No mean is removed and
// no input/encoder/scaler is called here. Fit once on the declared TRAIN bank.
struct MomentRmsScaler {
  MomentRmsScaler()=default;
  MomentRmsScaler(const Config &c,const std::string &source):source_identity(source){
    input_geometry={c.channel_count,c.history_length,c.input_width};
    const auto ids=resolved_channel_ids(c);input_geometry.insert(input_geometry.end(),ids.begin(),ids.end());
    sampling_interval=c.sampling_interval;
    const auto w=consecutive_change_bank_width(c);
    counts=torch::zeros({w},torch::kInt64);sumsq=torch::zeros({w},torch::kFloat64);
    scales=torch::ones({w},torch::kFloat64);support=torch::zeros({w},torch::kBool);
    validate(c);
  }
  void add(const EnergyChangeMomentTargets &t){
    TORCH_CHECK(!frozen,"[rpb-v25 moments] frozen TRAIN normalizer cannot refit");
    validate_targets(t,counts.numel());
    const auto valid=t.valid.detach().to(torch::kCPU);
    const auto safe=torch::where(t.valid,t.values,torch::zeros_like(t.values)).detach().to(torch::kCPU,torch::kFloat64);
    const auto n=valid.sum(0);TORCH_CHECK(counts.le(std::numeric_limits<int64_t>::max()-n).all().item<bool>(),"[rpb-v25 moments] count overflow");
    counts+=n;sumsq+=safe.square().sum(0);
    TORCH_CHECK(torch::isfinite(sumsq).all().item<bool>(),"[rpb-v25 moments] nonfinite accumulated energy");
  }
  void freeze(){
    TORCH_CHECK(!frozen,"[rpb-v25 moments] TRAIN normalizer freezes exactly once");
    support=counts.gt(0);
    scales=torch::where(support,(sumsq/counts.clamp_min(1).to(torch::kFloat64)).sqrt().clamp_min(kEnergyChangeMomentScaleFloor),torch::ones_like(sumsq));
    frozen=true;
  }
  void validate(const Config &c,const std::string *expected_source=nullptr)const{
    const auto w=consecutive_change_bank_width(c);
    std::vector<int64_t> geometry{c.channel_count,c.history_length,c.input_width};
    const auto ids=resolved_channel_ids(c);geometry.insert(geometry.end(),ids.begin(),ids.end());
    TORCH_CHECK(input_geometry==geometry && sampling_interval==c.sampling_interval,"[rpb-v25 moments] canonical input geometry mismatch");
    TORCH_CHECK(!source_identity.empty() && source_identity.size()<=4096 && source_identity.find('\0')==std::string::npos &&
        (!expected_source || source_identity==*expected_source),"[rpb-v25 moments] TRAIN source identity mismatch");
    TORCH_CHECK(layout_identity==kEnergyChangeMomentLayout && floor==kEnergyChangeMomentScaleFloor,"[rpb-v25 moments] layout/floor mismatch");
    for(const auto &t:{counts,sumsq,scales,support})TORCH_CHECK(t.defined() && t.device().is_cpu() && t.dim()==1 && t.numel()==w && !t.requires_grad(),"[rpb-v25 moments] finite CPU vector contract required");
    TORCH_CHECK(counts.scalar_type()==torch::kInt64 && sumsq.scalar_type()==torch::kFloat64 && scales.scalar_type()==torch::kFloat64 &&
        support.scalar_type()==torch::kBool && counts.ge(0).all().item<bool>() && torch::isfinite(sumsq).all().item<bool>() &&
        sumsq.ge(0).all().item<bool>() && torch::isfinite(scales).all().item<bool>() && scales.ge(floor).all().item<bool>() &&
        torch::where(counts.eq(0),sumsq,torch::zeros_like(sumsq)).eq(0).all().item<bool>(),"[rpb-v25 moments] invalid RMS sufficient statistics");
    const auto expected_support=frozen?counts.gt(0):torch::zeros_like(support);
    const auto expected_scale=frozen?torch::where(expected_support,(sumsq/counts.clamp_min(1).to(torch::kFloat64)).sqrt().clamp_min(floor),torch::ones_like(sumsq)):torch::ones_like(sumsq);
    TORCH_CHECK(torch::equal(support,expected_support) && torch::equal(scales,expected_scale),"[rpb-v25 moments] inconsistent support/scale statistics");
  }
  torch::Tensor normalized(const EnergyChangeMomentTargets &t)const{
    TORCH_CHECK(frozen,"[rpb-v25 moments] frozen normalizer required");validate_targets(t,counts.numel());
    return torch::where(t.valid,t.values,torch::zeros_like(t.values))/scales.to(t.values.device(),t.values.scalar_type());
  }
  void save(torch::serialize::OutputArchive &a)const{
    namespace d=configurable_checkpoint_detail;
    TORCH_CHECK(frozen,"[rpb-v25 moments] only frozen normalizers can be archived");
    a.write("artifact_kind",d::text_tensor("rpb_energy_moment_rms_scaler_v1"),true);
    a.write("source_identity",d::text_tensor(source_identity),true);a.write("layout_identity",d::text_tensor(layout_identity),true);
    a.write("input_geometry",torch::tensor(input_geometry,torch::kInt64),true);
    a.write("sampling_interval",torch::tensor(sampling_interval,torch::kFloat64),true);
    a.write("floor",torch::tensor(floor,torch::kFloat64),true);
    a.write("counts",counts,true);a.write("sumsq",sumsq,true);a.write("scales",scales,true);a.write("support",support,true);
  }
  static MomentRmsScaler load(torch::serialize::InputArchive &a,const Config &c,const std::string *expected_source=nullptr){
    namespace d=configurable_checkpoint_detail;
    TORCH_CHECK(d::read_text(a,"artifact_kind")=="rpb_energy_moment_rms_scaler_v1","[rpb-v25 moments] typed RMS normalizer required");
    MomentRmsScaler out;out.source_identity=d::read_text(a,"source_identity");out.layout_identity=d::read_text(a,"layout_identity");
    const auto geometry=d::read_vector(a,"input_geometry",torch::kInt64,3+c.channel_count);
    const auto *geometry_data=geometry.const_data_ptr<int64_t>();out.input_geometry.assign(geometry_data,geometry_data+geometry.numel());
    torch::Tensor interval;a.read("sampling_interval",interval,true);
    TORCH_CHECK(interval.dim()==0 && interval.scalar_type()==torch::kFloat64,"[rpb-v25 moments] scalar F64 sampling interval required");
    out.sampling_interval=interval.item<double>();
    torch::Tensor f;a.read("floor",f,true);
    TORCH_CHECK(f.dim()==0 && f.scalar_type()==torch::kFloat64,"[rpb-v25 moments] scalar F64 floor required");out.floor=f.item<double>();
    a.read("counts",out.counts,true);a.read("sumsq",out.sumsq,true);a.read("scales",out.scales,true);a.read("support",out.support,true);
    out.counts=out.counts.to(torch::kCPU);out.sumsq=out.sumsq.to(torch::kCPU);out.scales=out.scales.to(torch::kCPU);out.support=out.support.to(torch::kCPU);
    out.frozen=true;out.validate(c,expected_source);return out;
  }
  static void validate_targets(const EnergyChangeMomentTargets &t,int64_t width){
    TORCH_CHECK(t.values.defined() && t.valid.defined() && t.values.dim()==2 && t.values.size(1)==width &&
        t.valid.sizes()==t.values.sizes() && t.valid.scalar_type()==torch::kBool && t.valid.device()==t.values.device() &&
        t.values.is_floating_point() && torch::isfinite(torch::where(t.valid,t.values,torch::zeros_like(t.values))).all().item<bool>(),"[rpb-v25 moments] finite eligible moment targets required");
  }
  torch::Tensor counts,sumsq,scales,support;
  std::string source_identity,layout_identity{kEnergyChangeMomentLayout};
  std::vector<int64_t> input_geometry;double sampling_interval{0};
  double floor{kEnergyChangeMomentScaleFloor};bool frozen{false};
};

struct EnergyChangeMomentLoss {
  torch::Tensor loss,signed_loss,energy_loss,eligible_examples,signed_counts,energy_counts;
  int64_t eligible_example_count{0};
};
// Each example averages each supported group separately, then combines 1/2 +
// 1/2. Only examples supporting both groups enter the final example average.
inline EnergyChangeMomentLoss energy_change_balanced_loss(const torch::Tensor &prediction,
    const EnergyChangeMomentTargets &targets,const MomentRmsScaler &scaler){
  TORCH_CHECK(scaler.frozen,"[rpb-v25 moments] frozen TRAIN normalizer required");
  MomentRmsScaler::validate_targets(targets,scaler.counts.numel());
  TORCH_CHECK(prediction.sizes()==targets.values.sizes() && prediction.device()==targets.values.device() &&
      prediction.scalar_type()==targets.values.scalar_type() && torch::isfinite(prediction).all().item<bool>(),"[rpb-v25 moments] finite matching predictions required");
  const auto valid=targets.valid.logical_and(scaler.support.to(prediction.device()).unsqueeze(0));
  const auto safe_target=torch::where(valid,targets.values.detach(),torch::zeros_like(targets.values));
  const auto residual=torch::where(valid,prediction-safe_target,torch::zeros_like(prediction))/scaler.scales.to(prediction.device(),prediction.scalar_type());
  TORCH_CHECK(torch::isfinite(residual).all().item<bool>(),"[rpb-v25 moments] finite normalized residual required");
  const auto magnitude=residual.abs();
  const auto losses=torch::where(magnitude.le(kEnergyChangeHuberDelta),.5*residual.clamp(-1.,1.).square(),magnitude-.5);
  const auto signed_columns=torch::arange(prediction.size(1),torch::TensorOptions().device(prediction.device()).dtype(torch::kInt64)).remainder(2).eq(0).unsqueeze(0);
  const auto sm=valid.logical_and(signed_columns),em=valid.logical_and(signed_columns.logical_not());
  const auto sn=sm.sum(1),en=em.sum(1),eligible=sn.gt(0).logical_and(en.gt(0));
  const auto signed_rows=torch::where(sm,losses,torch::zeros_like(losses)).sum(1)/sn.clamp_min(1).to(prediction.scalar_type());
  const auto energy_rows=torch::where(em,losses,torch::zeros_like(losses)).sum(1)/en.clamp_min(1).to(prediction.scalar_type());
  const auto count=eligible.sum().item<int64_t>();
  const auto signed_loss=torch::where(eligible,signed_rows,torch::zeros_like(signed_rows)).sum()/double(std::max<int64_t>(count,1));
  const auto energy_loss=torch::where(eligible,energy_rows,torch::zeros_like(energy_rows)).sum()/double(std::max<int64_t>(count,1));
  return{kEnergyChangeSignedWeight*signed_loss+kEnergyChangeEnergyWeight*energy_loss,signed_loss,energy_loss,eligible,sn,en,count};
}
inline EnergyChangeMomentLoss energy_change_moment_loss(const EnergyChangeModel &model,
    const Input &visible,const torch::Tensor &native,const MomentRmsScaler &scaler){
  const auto &c=model->config();scaler.validate(c.level);
  TORCH_CHECK(native.dim()==2 && native.size(1)==energy_change_native_width(c) && native.device()==c.level.device &&
      native.scalar_type()==c.level.dtype,"[rpb-v25 moments] configured native required");
  const auto prediction=torch::matmul(native.narrow(1,c.level.export_width,c.change_width),model->moment_decoder.transpose(0,1));
  return energy_change_balanced_loss(prediction,energy_change_moment_targets(visible,c.level),scaler);
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
