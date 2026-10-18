// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_config.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_model.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
struct EnergyChangeModelImpl:torch::nn::Module {
  explicit EnergyChangeModelImpl(const EnergyChangeConfig &c):config_(c){
    validate_energy_change_config(c);
    encoder=register_module("encoder",SeparatedChangeModel(c));
    moment_decoder=register_parameter("moment_decoder",torch::zeros(
        {consecutive_change_bank_width(c.level),c.change_width},
        torch::TensorOptions().device(c.level.device).dtype(c.level.dtype)));
    for(const auto &p:named_parameters())parameters_.push_back({p.key(),p.value().sizes().vec(),p.value().requires_grad()});
    validate_state();
  }
  const EnergyChangeConfig &config()const{return config_;}
  EncodeOutput encode(const Input &input){return encoder->encode(input);}
  torch::Tensor decode(const torch::Tensor &native,const torch::Tensor &ids){return encoder->decode(native,ids);}
  ForwardOutput forward(const Input &input,const torch::Tensor &hidden){return encoder->forward(input,hidden);}
  void load(torch::serialize::InputArchive &a)override{torch::nn::Module::load(a);validate_state();}
  void validate_fixed_prior()const{encoder->validate_fixed_prior();}
  void validate_state()const{
    TORCH_CHECK(encoder,"[rpb-v25] registered encoder required");encoder->validate_state();
    const auto named=named_parameters();
    TORCH_CHECK(named.size()==parameters_.size(),"[rpb-v25] parameter count mismatch");
    for(const auto &p:parameters_){const auto *v=named.find(p.name);
      TORCH_CHECK(v && v->sizes().vec()==p.shape && v->requires_grad()==p.trainable &&
          v->device()==config_.level.device && v->scalar_type()==config_.level.dtype && torch::isfinite(*v).all().item<bool>(),
          "[rpb-v25] parameter contract mismatch: ",p.name);}
    const auto *registered_decoder=named.find("moment_decoder");
    TORCH_CHECK(registered_decoder && registered_decoder->unsafeGetTensorImpl()==moment_decoder.unsafeGetTensorImpl(),
        "[rpb-v25] moment decoder holder mismatch");
    for(const auto &p:encoder->named_parameters()){
      const auto *v=named.find("encoder."+p.key());
      TORCH_CHECK(v && v->unsafeGetTensorImpl()==p.value().unsafeGetTensorImpl(),"[rpb-v25] encoder holder mismatch");}
    const auto buffers=named_buffers();
    TORCH_CHECK(buffers.size()==encoder->named_buffers().size(),"[rpb-v25] buffer count mismatch");
    for(const auto &b:encoder->named_buffers()){
      const auto *v=buffers.find("encoder."+b.key());
      TORCH_CHECK(v && v->unsafeGetTensorImpl()==b.value().unsafeGetTensorImpl(),"[rpb-v25] encoder buffer holder mismatch");}
  }
  SeparatedChangeModel encoder{nullptr};torch::Tensor moment_decoder;
private:
  struct Contract{std::string name;std::vector<int64_t> shape;bool trainable;};
  EnergyChangeConfig config_;std::vector<Contract> parameters_;
};
TORCH_MODULE(EnergyChangeModel);
inline torch::Tensor energy_change_native(const EncodeOutput &out,const EnergyChangeConfig &c){return separated_change_native(out,c);}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
