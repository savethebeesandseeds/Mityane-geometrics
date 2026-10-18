// SPDX-License-Identifier: MIT
// CPB-v2 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v2/configurable_spectral_relation.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v2/model.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v2/preprocessing.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v2 {

inline torch::Tensor configurable_spectral_pad_observed(const torch::Tensor &observed,
                                                       const Config &c) {
  const auto extra = configurable_spectral_backbone_config(c).history_length - c.history_length;
  if (!extra) return observed;
  return torch::cat({observed, torch::zeros(
      {observed.size(0), c.channel_count, extra, c.input_width}, observed.options())}, 2);
}

inline Input configurable_spectral_padded_input(const Input &input, const Config &c) {
  const auto extra = configurable_spectral_backbone_config(c).history_length - c.history_length;
  if (!extra) return input;
  return {torch::cat({input.data, torch::zeros(
      {input.data.size(0), c.channel_count, extra, c.input_width}, input.data.options())}, 2),
      configurable_spectral_pad_observed(input.observed.to(input.data.device()), c),
      input.channel_ids, input.endpoints, input.sampling_interval};
}

inline MaskPlan configurable_spectral_crop_masks(const MaskPlan &padded, const Config &c) {
  return {padded.hidden.narrow(2, 0, c.history_length),
          padded.visible.narrow(2, 0, c.history_length),
          padded.target.narrow(2, 0, c.history_length), padded.eligible_channels};
}

inline MaskPlan configurable_spectral_mask_from_hidden(const torch::Tensor &observed,
    const torch::Tensor &hidden, const Config &c) {
  validate_configurable_spectral_config(c);
  TORCH_CHECK(observed.defined() && hidden.defined() && observed.dim() == 4 &&
                  observed.size(0) > 0 && observed.size(1) == c.channel_count &&
                  observed.size(2) == c.history_length && observed.size(3) == c.input_width &&
                  observed.scalar_type() == torch::kBool && hidden.scalar_type() == torch::kBool &&
                  hidden.sizes() == observed.sizes(),
              "[rpb-v20] training masks require logical BCHF bool tensors");
  auto padded_hidden = hidden.to(observed.device());
  const auto extra = configurable_spectral_backbone_config(c).history_length - c.history_length;
  if (extra) {
    // Replicate the partial patch's choice into its unavailable private cells
    // so the preserved complete-patch validator still checks the true patch.
    padded_hidden = torch::cat({padded_hidden,
        padded_hidden.narrow(2, c.history_length - 1, 1)
            .expand({hidden.size(0), c.channel_count, extra, c.input_width})}, 2);
  }
  return configurable_spectral_crop_masks(mask_from_hidden(
      configurable_spectral_pad_observed(observed, c), padded_hidden,
      configurable_spectral_backbone_config(c)), c);
}

inline MaskPlan configurable_spectral_make_training_mask(const torch::Tensor &observed,
    const Config &c, uint64_t seed) {
  validate_configurable_spectral_config(c);
  TORCH_CHECK(observed.defined() && observed.dim() == 4 && observed.size(0) > 0 &&
                  observed.size(1) == c.channel_count && observed.size(2) == c.history_length &&
                  observed.size(3) == c.input_width && observed.scalar_type() == torch::kBool,
              "[rpb-v20] observed mask must have the logical BCHF shape");
  return configurable_spectral_crop_masks(make_training_mask(
      configurable_spectral_pad_observed(observed, c),
      configurable_spectral_backbone_config(c), seed), c);
}

struct ConfigurableNativeShapeProjectionImpl : torch::nn::Module {
  explicit ConfigurableNativeShapeProjectionImpl(const Config &c) {
    weight = register_parameter("weight", torch::zeros(
        {configurable_spectral_shape_width(c), c.export_width}, torch::kFloat32));
    torch::NoGradGuard guard;
    auto w = weight.accessor<float, 2>();
    for (int64_t row = 0; row < weight.size(0); ++row) w[row][row] = 1.f;
  }
  torch::Tensor forward(const torch::Tensor &native) {
    TORCH_CHECK(native.dim() == 2 && native.size(1) == weight.size(1) &&
                    native.scalar_type() == weight.scalar_type() && native.device() == weight.device(),
                "[rpb-v20] configured model-device backbone native required");
    return torch::matmul(native, weight.transpose(0, 1));
  }
  torch::Tensor weight;
};
TORCH_MODULE(ConfigurableNativeShapeProjection);

inline torch::Tensor literal_configurable_spectral_prior(const Config &c,
                                                        const torch::TensorOptions &options) {
  const auto pairs = configurable_spectral_pair_count(c), feature_pairs = c.input_width * c.input_width;
  auto prior = torch::zeros({pairs, 4, feature_pairs * 4}, torch::kFloat32);
  auto w = prior.accessor<float, 3>();
  for (int64_t pair = 0; pair < pairs; ++pair)
    for (int64_t k = 0; k < 4; ++k)
      for (int64_t feature_pair = 0; feature_pair < feature_pairs; ++feature_pair)
        w[pair][k][feature_pair * 4 + k] = 1.f / static_cast<float>(feature_pairs);
  return prior.to(options);
}

struct ConfigurableGroupedSpectralProjectionImpl : torch::nn::Module {
  explicit ConfigurableGroupedSpectralProjectionImpl(const Config &c) {
    weight = register_parameter("weight", literal_configurable_spectral_prior(
        c, torch::TensorOptions().dtype(torch::kFloat32)));
  }
  torch::Tensor forward(const torch::Tensor &bank) {
    const auto pairs = weight.size(0), pair_width = weight.size(2);
    TORCH_CHECK(bank.dim() == 2 && bank.size(1) == pairs * pair_width &&
                    bank.scalar_type() == weight.scalar_type() && bank.device() == weight.device(),
                "[rpb-v20] configured model-device spectral bank required");
    if (!pairs) return torch::zeros({bank.size(0), 0}, bank.options());
    const auto grouped = bank.reshape({bank.size(0), pairs, pair_width}).transpose(0, 1);
    return torch::bmm(grouped, weight.transpose(1, 2)).transpose(0, 1)
        .reshape({bank.size(0), pairs * 4});
  }
  torch::Tensor weight;
};
TORCH_MODULE(ConfigurableGroupedSpectralProjection);

// Separate design tag and configurable checkpoint contract. Old v19 headers
// remain unchanged. At C3/H32/F3/P8/D32 the same module names, registration
// order, initialization and arithmetic permit exact raw-module parity testing.
struct ConfigurableSpectralModelImpl : torch::nn::Module {
  explicit ConfigurableSpectralModelImpl(const Config &c) : config_(c) {
    validate_configurable_spectral_config(c);
    backbone = register_module("backbone", Model(configurable_spectral_backbone_config(c)));
    native_shape_projection = register_module("native_shape_projection", ConfigurableNativeShapeProjection(c));
    // Preserve the historical parameter name for exact old-shape weight maps.
    grouped_odd_relation_projection = register_module(
        "grouped_odd_relation_projection", ConfigurableGroupedSpectralProjection(c));
    native_shape_projection->to(c.device, c.dtype);
    grouped_odd_relation_projection->to(c.device, c.dtype);
    freeze_and_validate_prior();
    parameter_contract_ = capture_contract(named_parameters());
    buffer_contract_ = capture_contract(named_buffers());
  }

  const Config &config() const { return config_; }

  EncodeOutput encode(const Input &input, TemporalExecutionTrace *trace = nullptr) {
    validate_input(input, configurable_spectral_input_config(config_), true);
    const auto bank = configurable_spectral_relation_bank(input, config_);
    if(trace){trace->spectral_values=bank.values;trace->spectral_counts=bank.common_counts;trace->spectral_eligible=bank.eligible;}
    auto out = backbone->encode(configurable_spectral_padded_input(input, config_),trace);
    const auto shape = native_shape_projection(out.z_contextual_global);
    const auto spectral = configurable_unit_spectral_relations(
        grouped_odd_relation_projection(bank.values), config_);
    const auto native = torch::cat({shape, spectral}, 1);
    out.z_contextual_global = torch::where(out.sample_valid_mask.unsqueeze(-1), native,
                                          torch::zeros_like(native));
    TORCH_CHECK(out.z_contextual_global.size(1) == config_.export_width &&
                    torch::isfinite(out.z_contextual_global).all().item<bool>(),
                "[rpb-v20] finite configured native embedding required");
    return out;
  }

  torch::Tensor decode(const torch::Tensor &native, const torch::Tensor &ids) {
    return backbone->decode(native, ids).narrow(2, 0, config_.history_length);
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
    out.loss = losses.loss;
    out.target_counts = losses.target_counts;
    out.eligible_channels = losses.eligible_channels;
    out.eligible_examples = losses.eligible_examples;
    out.eligible_channel_count = losses.eligible_channel_count;
    out.eligible_example_count = losses.eligible_example_count;
    out.target_cell_count = losses.target_cell_count;
    return out;
  }

  void load(torch::serialize::InputArchive &archive) override {
    torch::nn::Module::load(archive);
    freeze_and_validate_prior();
    validate_state();
  }

  // LibTorch's recursive load can replace a tensor with one of another shape.
  // Check against this instance's original architecture before accepting it.
  void validate_state() const {
    check_contract(named_parameters(), parameter_contract_);
    check_contract(named_buffers(), buffer_contract_);
    validate_fixed_prior();
  }

  void validate_fixed_prior() const {
    const auto &w = grouped_odd_relation_projection->weight;
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32 &&
                    w.sizes() == torch::IntArrayRef({configurable_spectral_pair_count(config_),
                                                   4, config_.input_width * config_.input_width * 4}) &&
                    !w.requires_grad() && !w.grad().defined() &&
                    torch::equal(w, literal_configurable_spectral_prior(config_, w.options())),
                "[rpb-v20] exact frozen feature-pair mean prior without gradients required");
  }

  Model backbone{nullptr};
  ConfigurableNativeShapeProjection native_shape_projection{nullptr};
  ConfigurableGroupedSpectralProjection grouped_odd_relation_projection{nullptr};

private:
  struct TensorContract {
    std::string name;
    std::vector<int64_t> shape;
    torch::Dtype dtype;
    torch::Device device;
  };
  using TensorMap = torch::OrderedDict<std::string, torch::Tensor>;
  static std::vector<TensorContract> capture_contract(const TensorMap &state) {
    std::vector<TensorContract> out;
    out.reserve(state.size());
    for (const auto &entry : state)
      out.push_back({entry.key(), entry.value().sizes().vec(),
                     entry.value().scalar_type(), entry.value().device()});
    return out;
  }
  static void check_contract(const TensorMap &state, const std::vector<TensorContract> &contract) {
    TORCH_CHECK(state.size() == contract.size(), "[rpb-v20] model state count mismatch");
    for (const auto &expected : contract) {
      const auto *tensor = state.find(expected.name);
      TORCH_CHECK(tensor && tensor->defined() && tensor->sizes().vec() == expected.shape &&
                      tensor->scalar_type() == expected.dtype && tensor->device() == expected.device,
                  "[rpb-v20] model tensor geometry/dtype/device mismatch: ", expected.name);
      TORCH_CHECK(!tensor->is_floating_point() || torch::isfinite(*tensor).all().item<bool>(),
                  "[rpb-v20] nonfinite model tensor: ", expected.name);
    }
  }
  void freeze_and_validate_prior() {
    auto &w = grouped_odd_relation_projection->weight;
    TORCH_CHECK(torch::equal(w, literal_configurable_spectral_prior(config_, w.options())),
                "[rpb-v20] checkpoint must preserve its literal fixed spectral prior");
    w.set_requires_grad(false);
    w.mutable_grad() = torch::Tensor();
    validate_fixed_prior();
  }
  Config config_;
  // Plain metadata, never serialized or registered; no new parameters or RNG draws.
  std::vector<TensorContract> parameter_contract_, buffer_contract_;
};
TORCH_MODULE(ConfigurableSpectralModel);

inline FrozenScaler fit_configurable_spectral_scaler(const Input &training, const Config &c) {
  validate_configurable_spectral_config(c);
  return fit_scaler(training, configurable_spectral_input_config(c));
}

inline Input transform_configurable_spectral_input(const FrozenScaler &scaler,
                                                   const Input &input, const Config &c) {
  validate_configurable_spectral_config(c);
  return scaler.transform(input, configurable_spectral_input_config(c));
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v2
