// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/conditioned_change_normalizer.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_model.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
struct ConditionedChangeModelImpl:torch::nn::Module {
  ConditionedChangeModelImpl(const ConditionedChangeConfig &c,const BankStandardizer &normalizer):
      config_(c),normalizer_(normalizer.clone()){
    validate_conditioned_change_config(c);normalizer_.validate(c.level);TORCH_CHECK(normalizer_.frozen,"[rpb-v26] frozen calibration required");
    normalizer_identity_=normalizer_.identity();encoder=register_module("encoder",SeparatedChangeModel(c));
    for(const auto &p:named_parameters())parameters_.push_back({p.key(),p.value().sizes().vec(),p.value().requires_grad()});
    validate_state();
  }
  const ConditionedChangeConfig &config()const{return config_;}
  const BankStandardizer &normalizer()const{return normalizer_;}
  EncodeOutput encode(const Input &input){
    const auto values=normalizer_.transform(conditioned_change_targets(input,config_.level));
    EncodeOutput out;{torch::NoGradGuard guard;out=encoder->level->encode(input);}
    const auto changes=torch::matmul(values,encoder->change_projection.transpose(0,1));
    out.z_contextual_global=torch::cat({out.z_contextual_global,
        torch::where(out.sample_valid_mask.unsqueeze(-1),changes,torch::zeros_like(changes))},1);
    TORCH_CHECK(torch::isfinite(out.z_contextual_global).all().item<bool>(),"[rpb-v26] finite native required");return out;
  }
  torch::Tensor decode(const torch::Tensor &native,const torch::Tensor &ids){return encoder->decode(native,ids);}
  ForwardOutput forward(const Input &input,const torch::Tensor &hidden){
    const auto &c=config_.level;validate_input(input,configurable_spectral_input_config(c),true);
    const auto masks=configurable_spectral_mask_from_hidden(input.observed.to(c.device),hidden.to(c.device),c);
    const Input visible{input.data,masks.visible,input.channel_ids,input.endpoints,input.sampling_interval};
    ForwardOutput out;out.encoding=encode(visible);out.reconstruction=decode(out.encoding.z_contextual_global,input.channel_ids);
    const auto loss=hierarchical_huber(out.reconstruction,input.data.to(c.device).detach(),masks.target,masks.eligible_channels,c.huber_delta);
    out.loss=loss.loss;out.target_counts=loss.target_counts;out.eligible_channels=loss.eligible_channels;
    out.eligible_examples=loss.eligible_examples;out.eligible_channel_count=loss.eligible_channel_count;
    out.eligible_example_count=loss.eligible_example_count;out.target_cell_count=loss.target_cell_count;return out;
  }
  void load(torch::serialize::InputArchive &a)override{torch::nn::Module::load(a);validate_state();}
  void validate_fixed_prior()const{encoder->validate_fixed_prior();}
  void validate_state()const{
    normalizer_.validate(config_.level);TORCH_CHECK(normalizer_.frozen && normalizer_.identity()==normalizer_identity_,"[rpb-v26] frozen normalizer changed");
    TORCH_CHECK(encoder,"[rpb-v26] registered encoder required");encoder->validate_state();const auto named=named_parameters();
    TORCH_CHECK(named.size()==parameters_.size(),"[rpb-v26] parameter count mismatch");
    for(const auto &p:parameters_){const auto *v=named.find(p.name);
      TORCH_CHECK(v && v->sizes().vec()==p.shape && v->requires_grad()==p.trainable && v->device()==config_.level.device &&
          v->scalar_type()==config_.level.dtype && torch::isfinite(*v).all().item<bool>(),"[rpb-v26] parameter contract mismatch: ",p.name);}
    for(const auto &p:encoder->named_parameters()){const auto *v=named.find("encoder."+p.key());
      TORCH_CHECK(v && v->unsafeGetTensorImpl()==p.value().unsafeGetTensorImpl(),"[rpb-v26] registered child holder mismatch");}
    TORCH_CHECK(named["encoder.change_projection"].unsafeGetTensorImpl()==encoder->change_projection.unsafeGetTensorImpl() &&
        named["encoder.change_decoder"].unsafeGetTensorImpl()==encoder->change_decoder.unsafeGetTensorImpl(),"[rpb-v26] branch parameter holder mismatch");
    const auto buffers=named_buffers();TORCH_CHECK(buffers.size()==encoder->named_buffers().size(),"[rpb-v26] buffer count mismatch");
    for(const auto &b:encoder->named_buffers()){const auto *v=buffers.find("encoder."+b.key());
      TORCH_CHECK(v && v->unsafeGetTensorImpl()==b.value().unsafeGetTensorImpl(),"[rpb-v26] registered buffer holder mismatch");}
  }
  SeparatedChangeModel encoder{nullptr};
private:
  struct Contract{std::string name;std::vector<int64_t> shape;bool trainable;};
  ConditionedChangeConfig config_;BankStandardizer normalizer_;std::string normalizer_identity_;std::vector<Contract> parameters_;
};
TORCH_MODULE(ConditionedChangeModel);
inline torch::Tensor conditioned_change_native(const EncodeOutput &out,const ConditionedChangeConfig &c){return separated_change_native(out,c);}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
