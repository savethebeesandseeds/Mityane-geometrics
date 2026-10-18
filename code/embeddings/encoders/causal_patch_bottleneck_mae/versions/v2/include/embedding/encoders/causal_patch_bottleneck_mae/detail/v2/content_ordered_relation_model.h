// SPDX-License-Identifier: MIT
// CPB-v2 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v2/configurable_spectral_model.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v2/content_ordered_relation_bank.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v2 {

struct ContentOrderedRelationProjectionImpl : torch::nn::Module {
  explicit ContentOrderedRelationProjectionImpl(const Config &c) {
    weight = register_parameter("weight", torch::zeros(
        {configurable_spectral_shape_width(c), content_ordered_relation_bank_width(c)},
        torch::kFloat32));
  }
  torch::Tensor forward(const torch::Tensor &bank) {
    TORCH_CHECK(bank.dim() == 2 && bank.size(1) == weight.size(1) &&
                    bank.scalar_type() == weight.scalar_type() && bank.device() == weight.device(),
                "[rpb-v22] model-device ordered relation bank required");
    // C1 has no cross-channel relation parameters or comparisons.
    if (!weight.size(1)) return torch::zeros({bank.size(0), weight.size(0)}, bank.options());
    return torch::matmul(bank, weight.transpose(0, 1));
  }
  torch::Tensor weight;
};
TORCH_MODULE(ContentOrderedRelationProjection);

// Historical v20 remains an unmodified child. The last, literal-zero module
// consumes no initialization RNG and adds only to its learned coordinates.
struct ContentOrderedRelationModelImpl : torch::nn::Module {
  explicit ContentOrderedRelationModelImpl(const Config &c) : config_(c) {
    validate_content_ordered_relation_config(c);
    base = register_module("base", ConfigurableSpectralModel(c));
    relation_projection = register_module("relation_projection", ContentOrderedRelationProjection(c));
    relation_projection->to(c.device, c.dtype);
    for (const auto &entry : named_parameters())
      parameters_.push_back({entry.key(), entry.value().sizes().vec(),
                             entry.value().scalar_type(), entry.value().device()});
    for (const auto &entry : named_buffers())
      buffers_.push_back({entry.key(), entry.value().sizes().vec(),
                          entry.value().scalar_type(), entry.value().device()});
    validate_state();
  }

  const Config &config() const { return config_; }
  EncodeOutput encode(const Input &input, TemporalExecutionTrace *trace = nullptr) {
    const auto bank = content_ordered_relation_bank(input, config_);
    if(trace){trace->content_values=bank.values;trace->content_counts=bank.common_counts;trace->content_eligible=bank.eligible;}
    auto out = base->encode(input,trace);
    const auto L = configurable_spectral_shape_width(config_);
    const auto fixed = configurable_spectral_width(config_);
    const auto native = out.z_contextual_global;
    const auto learned = native.narrow(1, 0, L) + relation_projection(bank.values);
    const auto complete = torch::cat({learned, native.narrow(1, L, fixed)}, 1);
    out.z_contextual_global = torch::where(out.sample_valid_mask.unsqueeze(-1), complete,
                                          torch::zeros_like(complete));
    TORCH_CHECK(out.z_contextual_global.size(1) == config_.export_width &&
                    torch::isfinite(out.z_contextual_global).all().item<bool>(),
                "[rpb-v22] finite configured native embedding required");
    return out;
  }
  torch::Tensor decode(const torch::Tensor &native, const torch::Tensor &ids) {
    return base->decode(native, ids);
  }
  ForwardOutput forward(const Input &input, const torch::Tensor &hidden) {
    validate_input(input, configurable_spectral_input_config(config_), true);
    const auto masks = configurable_spectral_mask_from_hidden(
        input.observed.to(config_.device), hidden.to(config_.device), config_);
    const Input visible{input.data, masks.visible, input.channel_ids, input.endpoints,
                        input.sampling_interval};
    ForwardOutput out;
    out.encoding = encode(visible);
    out.reconstruction = decode(compact_reconstruction_export(out.encoding, config_), input.channel_ids);
    const auto losses = hierarchical_huber(out.reconstruction, input.data.to(config_.device).detach(),
        masks.target, masks.eligible_channels, config_.huber_delta);
    out.loss = losses.loss; out.target_counts = losses.target_counts;
    out.eligible_channels = losses.eligible_channels; out.eligible_examples = losses.eligible_examples;
    out.eligible_channel_count = losses.eligible_channel_count;
    out.eligible_example_count = losses.eligible_example_count;
    out.target_cell_count = losses.target_cell_count;
    return out;
  }
  void load(torch::serialize::InputArchive &archive) override {
    torch::nn::Module::load(archive);
    // Recursive LibTorch loading invokes the historical child's prior checks.
    validate_state();
  }
  void validate_fixed_prior() const { base->validate_fixed_prior(); }
  void validate_state() const {
    base->validate_state();
    check_contract(named_parameters(), parameters_);
    check_contract(named_buffers(), buffers_);
    TORCH_CHECK(relation_projection->weight.requires_grad(),
                "[rpb-v22] relation projection must remain trainable");
  }

  ConfigurableSpectralModel base{nullptr};
  ContentOrderedRelationProjection relation_projection{nullptr};
private:
  struct Contract { std::string name; std::vector<int64_t> shape; torch::Dtype dtype; torch::Device device; };
  static void check_contract(const torch::OrderedDict<std::string, torch::Tensor> &state,
                             const std::vector<Contract> &contract) {
    TORCH_CHECK(state.size() == contract.size(), "[rpb-v22] model state count mismatch");
    for (const auto &expected : contract) {
      const auto *t = state.find(expected.name);
      TORCH_CHECK(t && t->defined() && t->sizes().vec() == expected.shape &&
                      t->scalar_type() == expected.dtype && t->device() == expected.device,
                  "[rpb-v22] model state geometry/dtype/device mismatch: ", expected.name);
      TORCH_CHECK(!t->is_floating_point() || torch::isfinite(*t).all().item<bool>(),
                  "[rpb-v22] nonfinite model state: ", expected.name);
    }
  }
  Config config_;
  std::vector<Contract> parameters_, buffers_;
};
TORCH_MODULE(ContentOrderedRelationModel);

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v2
