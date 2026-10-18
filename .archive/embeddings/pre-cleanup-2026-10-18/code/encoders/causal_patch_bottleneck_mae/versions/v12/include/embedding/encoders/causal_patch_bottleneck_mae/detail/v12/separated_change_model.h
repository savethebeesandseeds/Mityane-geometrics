// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v12/separated_change_config.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v12/consecutive_change_bank.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v12/content_ordered_relation_model.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v12 {
struct SeparatedChangeModelImpl : torch::nn::Module {
  explicit SeparatedChangeModelImpl(const SeparatedChangeConfig &c):config_(c) {
    validate_separated_change_config(c);level=register_module("level",ContentOrderedRelationModel(c.level));
    const auto options=torch::TensorOptions().device(c.level.device).dtype(c.level.dtype);
    const auto bank_width=consecutive_change_bank_width(c.level);
    change_projection=register_parameter("change_projection",torch::empty({c.change_width,bank_width},options));
    { torch::NoGradGuard guard;change_projection.uniform_(-1./std::sqrt(double(bank_width)),1./std::sqrt(double(bank_width))); }
    change_decoder=register_parameter("change_decoder",torch::zeros(
        {c.level.channel_count*c.level.history_length*c.level.input_width,c.change_width},options));
    for(const auto &p:named_parameters())parameters_.push_back({p.key(),p.value().sizes().vec(),p.value().requires_grad()});
    validate_state();
  }
  const SeparatedChangeConfig &config()const{return config_;}
  EncodeOutput encode(const Input &input) {
    const auto bank=consecutive_change_bank(input,config_.level);
    EncodeOutput out=level->encode(input);
    const auto changes=torch::matmul(bank.values,change_projection.transpose(0,1));
    // Concatenation preserves the parent output values and its live training graph.
    out.z_contextual_global=torch::cat({out.z_contextual_global,
        torch::where(out.sample_valid_mask.unsqueeze(-1),changes,torch::zeros_like(changes))},1);
    TORCH_CHECK(torch::isfinite(out.z_contextual_global).all().item<bool>(),"[rpb-v24] finite native required");
    return out;
  }
  torch::Tensor decode(const torch::Tensor &native,const torch::Tensor &ids) {
    const auto &c=config_.level;
    TORCH_CHECK(native.defined() && native.dim()==2 && native.size(1)==separated_change_native_width(config_) &&
        native.device()==c.device && native.scalar_type()==c.dtype && torch::isfinite(native).all().item<bool>(),
        "[rpb-v24] configured model-device native required");
    const auto B=native.size(0),C=c.channel_count,H=c.history_length,F=c.input_width;
    const auto base=level->decode(native.narrow(1,0,c.export_width),ids);
    const auto canonical=torch::matmul(native.narrow(1,c.export_width,config_.change_width),
        change_decoder.transpose(0,1)).reshape({B,C,H,F});
    const auto semantic=channel_indices(ids,c,B,c.device);
    return base+canonical.gather(1,semantic.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F}));
  }
  ForwardOutput forward(const Input &input,const torch::Tensor &hidden) {
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
  void validate_fixed_prior()const{level->validate_fixed_prior();}
  void validate_state()const {
    level->validate_state();const auto named=named_parameters();
    TORCH_CHECK(named.size()==parameters_.size(),"[rpb-v24] parameter count mismatch");
    for(const auto &p:parameters_){const auto *v=named.find(p.name);
      TORCH_CHECK(v && v->sizes().vec()==p.shape && v->requires_grad()==p.trainable &&
          v->device()==config_.level.device && v->scalar_type()==config_.level.dtype && torch::isfinite(*v).all().item<bool>(),
          "[rpb-v24] parameter contract mismatch: ",p.name);}
    // Catch replacement of the public holder without register_module replacement.
    for(const auto &p:level->named_parameters()) {
      const auto *registered=named.find("level."+p.key());
      TORCH_CHECK(registered && registered->unsafeGetTensorImpl()==p.value().unsafeGetTensorImpl(),
          "[rpb-v24] level holder/registered child mismatch");
    }
  }
  ContentOrderedRelationModel level{nullptr};torch::Tensor change_projection,change_decoder;
private:
  struct Contract{std::string name;std::vector<int64_t> shape;bool trainable;};
  SeparatedChangeConfig config_;std::vector<Contract> parameters_;
};
TORCH_MODULE(SeparatedChangeModel);
inline torch::Tensor separated_change_native(const EncodeOutput &out,const SeparatedChangeConfig &c){
  const auto &v=out.z_contextual_global;
  TORCH_CHECK(v.dim()==2 && v.size(1)==separated_change_native_width(c) &&
      v.device()==c.level.device && v.scalar_type()==c.level.dtype,"[rpb-v24] configured native export required");return v;
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v12
