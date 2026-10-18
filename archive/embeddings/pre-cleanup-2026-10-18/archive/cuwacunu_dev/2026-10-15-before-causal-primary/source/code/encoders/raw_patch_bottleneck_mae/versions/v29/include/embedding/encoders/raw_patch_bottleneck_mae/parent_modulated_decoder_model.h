// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_config.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/relative_change_normalizer.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_model.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
struct ParentModulatedDecoderTrace {
  torch::Tensor parent_prediction,canonical_base_residual,gain,scaled_residual,prediction;
};
struct ParentModulatedDecoderModelImpl:torch::nn::Module {
  ParentModulatedDecoderModelImpl(const ParentModulatedDecoderConfig &c,const RelativeBankStandardizer &normalizer):
      config_(c),normalizer_(normalizer.clone()){
    validate_parent_modulated_decoder_config(c);normalizer_.validate(c.level);TORCH_CHECK(normalizer_.frozen,"[rpb-v29] frozen calibration required");
    normalizer_identity_=normalizer_.identity();encoder=register_module("encoder",SeparatedChangeModel(c));
    waveform_gain=register_parameter("waveform_gain",torch::zeros({c.level.channel_count*c.level.input_width,c.level.export_width},
        torch::TensorOptions().device(c.level.device).dtype(c.level.dtype)));
    for(const auto &p:named_parameters())parameters_.push_back({p.key(),p.value().sizes().vec(),p.value().requires_grad()});
    validate_state();
  }
  const ParentModulatedDecoderConfig &config()const{return config_;}
  const RelativeBankStandardizer &normalizer()const{return normalizer_;}
  EncodeOutput encode(const Input &raw,const FrozenScaler &scaler){
    const auto input=relative_change_parent_input(raw,scaler,config_.level);
    const auto values=normalizer_.transform(relative_change_targets(raw,config_.level));
    EncodeOutput out;{torch::NoGradGuard guard;out=encoder->level->encode(input);}
    const auto changes=torch::matmul(values,encoder->change_projection.transpose(0,1));
    out.z_contextual_global=torch::cat({out.z_contextual_global,
        torch::where(out.sample_valid_mask.unsqueeze(-1),changes,torch::zeros_like(changes))},1);
    TORCH_CHECK(torch::isfinite(out.z_contextual_global).all().item<bool>(),"[rpb-v29] finite native required");return out;
  }
  ParentModulatedDecoderTrace decode_with_trace(const torch::Tensor &native,const torch::Tensor &ids){
    const auto &c=config_.level;
    TORCH_CHECK(native.defined() && native.dim()==2 && native.size(1)==parent_modulated_decoder_native_width(config_) &&
        native.device()==c.device && native.scalar_type()==c.dtype && torch::isfinite(native).all().item<bool>(),
        "[rpb-v29] configured model-device finite native required");
    const auto B=native.size(0),C=c.channel_count,H=c.history_length,F=c.input_width;
    const auto parent=native.narrow(1,0,c.export_width).detach();
    ParentModulatedDecoderTrace out;
    {torch::NoGradGuard guard;out.parent_prediction=encoder->level->decode(parent,ids);}
    out.canonical_base_residual=torch::matmul(native.narrow(1,c.export_width,config_.change_width),
        encoder->change_decoder.transpose(0,1)).reshape({B,C,H,F});
    out.gain=1.+torch::matmul(parent,waveform_gain.transpose(0,1)).reshape({B,C,1,F});
    out.scaled_residual=out.canonical_base_residual*out.gain;
    const auto semantic=channel_indices(ids,c,B,c.device);
    out.prediction=out.parent_prediction+out.scaled_residual.gather(1,
        semantic.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F}));
    TORCH_CHECK(torch::isfinite(out.gain).all().item<bool>() && torch::isfinite(out.prediction).all().item<bool>(),
        "[rpb-v29] finite gain and reconstruction required");return out;
  }
  torch::Tensor decode(const torch::Tensor &native,const torch::Tensor &ids){return decode_with_trace(native,ids).prediction;}
  ForwardOutput forward(const Input &raw,const torch::Tensor &hidden,const FrozenScaler &scaler){
    const auto &c=config_.level;const auto input=relative_change_parent_input(raw,scaler,c);
    const auto masks=configurable_spectral_mask_from_hidden(raw.observed.to(c.device),hidden.to(c.device),c);
    const Input visible{raw.data,masks.visible,raw.channel_ids,raw.endpoints,raw.sampling_interval};
    ForwardOutput out;out.encoding=encode(visible,scaler);out.reconstruction=decode(out.encoding.z_contextual_global,raw.channel_ids);
    const auto loss=hierarchical_huber(out.reconstruction,input.data.to(c.device).detach(),masks.target,masks.eligible_channels,c.huber_delta);
    out.loss=loss.loss;out.target_counts=loss.target_counts;out.eligible_channels=loss.eligible_channels;
    out.eligible_examples=loss.eligible_examples;out.eligible_channel_count=loss.eligible_channel_count;
    out.eligible_example_count=loss.eligible_example_count;out.target_cell_count=loss.target_cell_count;return out;
  }
  void load(torch::serialize::InputArchive &a)override{torch::nn::Module::load(a);validate_state();}
  void validate_fixed_prior()const{encoder->validate_fixed_prior();}
  void validate_state()const{
    normalizer_.validate(config_.level);TORCH_CHECK(normalizer_.frozen && normalizer_.identity()==normalizer_identity_,"[rpb-v29] frozen normalizer changed");
    TORCH_CHECK(encoder,"[rpb-v29] registered encoder required");encoder->validate_state();const auto named=named_parameters();
    TORCH_CHECK(named.size()==parameters_.size(),"[rpb-v29] parameter count mismatch");
    for(const auto &p:parameters_){const auto *v=named.find(p.name);
      TORCH_CHECK(v && v->sizes().vec()==p.shape && v->requires_grad()==p.trainable && v->device()==config_.level.device &&
          v->scalar_type()==config_.level.dtype && torch::isfinite(*v).all().item<bool>(),"[rpb-v29] parameter contract mismatch: ",p.name);}
    for(const auto &p:encoder->named_parameters()){const auto *v=named.find("encoder."+p.key());
      TORCH_CHECK(v && v->unsafeGetTensorImpl()==p.value().unsafeGetTensorImpl(),"[rpb-v29] registered child holder mismatch");}
    TORCH_CHECK(named["encoder.change_projection"].unsafeGetTensorImpl()==encoder->change_projection.unsafeGetTensorImpl() &&
        named["encoder.change_decoder"].unsafeGetTensorImpl()==encoder->change_decoder.unsafeGetTensorImpl(),"[rpb-v29] branch parameter holder mismatch");
    TORCH_CHECK(named["waveform_gain"].unsafeGetTensorImpl()==waveform_gain.unsafeGetTensorImpl(),
        "[rpb-v29] registered waveform gain holder mismatch");
    const auto buffers=named_buffers();TORCH_CHECK(buffers.size()==encoder->named_buffers().size(),"[rpb-v29] buffer count mismatch");
    for(const auto &b:encoder->named_buffers()){const auto *v=buffers.find("encoder."+b.key());
      TORCH_CHECK(v && v->unsafeGetTensorImpl()==b.value().unsafeGetTensorImpl(),"[rpb-v29] registered buffer holder mismatch");}
  }
  SeparatedChangeModel encoder{nullptr};torch::Tensor waveform_gain;
private:
  struct Contract{std::string name;std::vector<int64_t> shape;bool trainable;};
  ParentModulatedDecoderConfig config_;RelativeBankStandardizer normalizer_;std::string normalizer_identity_;std::vector<Contract> parameters_;
};
TORCH_MODULE(ParentModulatedDecoderModel);
inline torch::Tensor parent_modulated_decoder_native(const EncodeOutput &out,const ParentModulatedDecoderConfig &c){return separated_change_native(out,c);}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
