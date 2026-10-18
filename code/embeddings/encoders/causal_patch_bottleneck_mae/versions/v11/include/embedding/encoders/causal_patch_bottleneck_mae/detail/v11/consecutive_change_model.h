// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v11/content_ordered_relation_model.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v11/consecutive_change_bank.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v11 {

struct ConsecutiveChangeModelImpl : torch::nn::Module {
  explicit ConsecutiveChangeModelImpl(const Config &c) : config_(c) {
    validate_consecutive_change_config(c);
    level=register_module("level",ContentOrderedRelationModel(c));
    // Literal zero consumes no initialization draws; v22 initialization is exact.
    change_weight=register_parameter("change_weight",torch::zeros(
        {configurable_spectral_shape_width(c),consecutive_change_bank_width(c)},
        torch::TensorOptions().device(c.device).dtype(c.dtype)));
    for (const auto &p:named_parameters()) parameters_.push_back(
        {p.key(),p.value().sizes().vec(),p.value().requires_grad()});
    validate_state();
  }
  const Config &config() const { return config_; }
  EncodeOutput encode(const Input &input) {
    const auto bank=consecutive_change_bank(input,config_);
    auto out=level->encode(input);
    const auto L=configurable_spectral_shape_width(config_);
    const auto native=out.z_contextual_global;
    const auto learned=native.narrow(1,0,L)+torch::matmul(bank.values,change_weight.transpose(0,1));
    const auto complete=torch::cat({learned,native.narrow(1,L,configurable_spectral_width(config_))},1);
    out.z_contextual_global=torch::where(out.sample_valid_mask.unsqueeze(-1),complete,torch::zeros_like(complete));
    TORCH_CHECK(torch::isfinite(out.z_contextual_global).all().item<bool>(),"[rpb-v23] finite native export required");
    return out;
  }
  torch::Tensor decode(const torch::Tensor &native,const torch::Tensor &ids) {
    return level->decode(native,ids);
  }
  ForwardOutput forward(const Input &input,const torch::Tensor &hidden) {
    validate_input(input,configurable_spectral_input_config(config_),true);
    const auto masks=configurable_spectral_mask_from_hidden(input.observed.to(config_.device),hidden.to(config_.device),config_);
    const Input visible{input.data,masks.visible,input.channel_ids,input.endpoints,input.sampling_interval};
    ForwardOutput out;out.encoding=encode(visible);
    out.reconstruction=decode(compact_reconstruction_export(out.encoding,config_),input.channel_ids);
    const auto loss=hierarchical_huber(out.reconstruction,input.data.to(config_.device).detach(),
        masks.target,masks.eligible_channels,config_.huber_delta);
    out.loss=loss.loss;out.target_counts=loss.target_counts;out.eligible_channels=loss.eligible_channels;
    out.eligible_examples=loss.eligible_examples;out.eligible_channel_count=loss.eligible_channel_count;
    out.eligible_example_count=loss.eligible_example_count;out.target_cell_count=loss.target_cell_count;
    return out;
  }
  void load(torch::serialize::InputArchive &a) override { torch::nn::Module::load(a);validate_state(); }
  void validate_fixed_prior() const { level->validate_fixed_prior(); }
  void validate_state() const {
    level->validate_state();const auto actual=named_parameters();
    TORCH_CHECK(actual.size()==parameters_.size(),"[rpb-v23] parameter count mismatch");
    for (const auto &p:parameters_) {
      const auto *t=actual.find(p.name);
      TORCH_CHECK(t && t->sizes().vec()==p.shape && t->device()==config_.device &&
          t->scalar_type()==config_.dtype && t->requires_grad()==p.trainable &&
          torch::isfinite(*t).all().item<bool>(),"[rpb-v23] model state mismatch: ",p.name);
    }
  }
  ContentOrderedRelationModel level{nullptr};
  torch::Tensor change_weight;
private:
  struct Contract { std::string name;std::vector<int64_t> shape;bool trainable; };
  Config config_;std::vector<Contract> parameters_;
};
TORCH_MODULE(ConsecutiveChangeModel);

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v11
