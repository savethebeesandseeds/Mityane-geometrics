// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/temporal_relation_bank.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kPartitionedTemporalRelationModelTag = "RPB-v16";
inline constexpr const char *kPartitionedTemporalRelationArchitectureId =
    "early-native20-shape-plus12-grouped-time-odd-relations-v1";
inline constexpr const char *kPartitionedTemporalRelationLayout =
    "shape20;odd12;semantic-pairs-01-02-12;four-learned-coordinates-per-pair";
inline constexpr int64_t kPartitionedTemporalRelationParameterCount = 226877;

struct NativeShapeProjectionImpl : torch::nn::Module {
  NativeShapeProjectionImpl() {
    weight = register_parameter("weight", torch::zeros({20, 32}, torch::kFloat32));
    torch::NoGradGuard guard;
    auto w = weight.accessor<float, 2>();
    for (int64_t row = 0; row < 20; ++row) w[row][row] = 1.f;
  }
  torch::Tensor forward(const torch::Tensor &native32) {
    TORCH_CHECK(native32.dim() == 2 && native32.size(1) == 32 &&
                    native32.scalar_type() == weight.scalar_type() && native32.device() == weight.device(),
                "[rpb-partitioned-relations] exact model-device native32 required");
    return torch::matmul(native32, weight.transpose(0, 1));
  }
  torch::Tensor weight;
};
TORCH_MODULE(NativeShapeProjection);

// No bias, symmetric moments, support fractions or raw coordinates enter this
// module. Pair grouping remains structural for every possible trained weight.
// Joint time reversal of values and their masks changes A to -A, so this
// linear relation block changes sign without an assumption about its weights.
struct GroupedOddRelationProjectionImpl : torch::nn::Module {
  GroupedOddRelationProjectionImpl() {
    weight = register_parameter("weight", torch::zeros({3, 4, 36}, torch::kFloat32));
    torch::NoGradGuard guard;
    auto w = weight.accessor<float, 3>();
    for (int64_t pair = 0; pair < 3; ++pair)
      for (int64_t k = 0; k < 4; ++k)
        for (int64_t feature_pair = 0; feature_pair < 9; ++feature_pair)
          w[pair][k][feature_pair * 4 + k] = 1.f / 9.f;
  }
  torch::Tensor forward(const torch::Tensor &antisymmetric108) {
    TORCH_CHECK(antisymmetric108.dim() == 2 && antisymmetric108.size(1) == 108 &&
                    antisymmetric108.scalar_type() == weight.scalar_type() &&
                    antisymmetric108.device() == weight.device(),
                "[rpb-partitioned-relations] exact model-device odd108 required");
    const auto B = antisymmetric108.size(0);
    const auto pairs = antisymmetric108.reshape({B, 3, 36}).transpose(0, 1);
    return torch::bmm(pairs, weight.transpose(1, 2)).transpose(0, 1).reshape({B, 12});
  }
  torch::Tensor weight;
};
TORCH_MODULE(GroupedOddRelationProjection);

// New isolated model contract; the original Model/Config and v14 bank remain
// unchanged. Both new modules are registered after the entire old backbone,
// and their literal initializers consume no CPU or CUDA RNG draw.
struct PartitionedTemporalRelationModelImpl : torch::nn::Module {
  explicit PartitionedTemporalRelationModelImpl(const Config &c) : config_(c) {
    validate_temporal_relation_config(c);
    TORCH_CHECK(c.device.is_cuda(), "[rpb-partitioned-relations] model requires actual CUDA");
    backbone = register_module("backbone", Model(c));
    native_shape_projection = register_module("native_shape_projection", NativeShapeProjection());
    grouped_odd_relation_projection = register_module("grouped_odd_relation_projection", GroupedOddRelationProjection());
    native_shape_projection->to(c.device, c.dtype);
    grouped_odd_relation_projection->to(c.device, c.dtype);
  }
  const Config &config() const { return config_; }
  EncodeOutput encode(const Input &input) {
    const auto bank = temporal_relation_bank(input, config_);
    auto out = backbone->encode(input);
    const auto shape = native_shape_projection(out.z_contextual_global);
    const auto odd = grouped_odd_relation_projection(bank.values.narrow(1, 0, 108));
    const auto native32 = torch::cat({shape, odd}, 1);
    out.z_contextual_global = torch::where(out.sample_valid_mask.unsqueeze(-1),
        native32, torch::zeros_like(native32));
    TORCH_CHECK(out.z_contextual_global.size(1) == 32 && torch::isfinite(out.z_contextual_global).all().item<bool>(),
                "[rpb-partitioned-relations] finite partitioned native32 required");
    return out;
  }
  torch::Tensor decode(const torch::Tensor &native32, const torch::Tensor &ids) {
    return backbone->decode(native32, ids);
  }
  ForwardOutput forward(const Input &input, const torch::Tensor &hidden) {
    validate_input(input, config_, true);
    const auto masks = mask_from_hidden(input.observed.to(config_.device), hidden.to(config_.device), config_);
    const Input visible{input.data, masks.visible, input.channel_ids, input.endpoints, input.sampling_interval};
    ForwardOutput out;
    out.encoding = encode(visible);
    out.reconstruction = decode(compact_reconstruction_export(out.encoding, config_), input.channel_ids);
    const auto losses = hierarchical_huber(out.reconstruction, input.data.to(config_.device).detach(),
                                          masks.target, masks.eligible_channels, config_.huber_delta);
    out.loss = losses.loss; out.target_counts = losses.target_counts;
    out.eligible_channels = losses.eligible_channels; out.eligible_examples = losses.eligible_examples;
    out.eligible_channel_count = losses.eligible_channel_count;
    out.eligible_example_count = losses.eligible_example_count; out.target_cell_count = losses.target_cell_count;
    return out;
  }
  Model backbone{nullptr};
  NativeShapeProjection native_shape_projection{nullptr};
  GroupedOddRelationProjection grouped_odd_relation_projection{nullptr};
private:
  Config config_;
};
TORCH_MODULE(PartitionedTemporalRelationModel);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
