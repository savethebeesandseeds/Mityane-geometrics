// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/fixed_prior_temporal_relation.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kUnitTemporalRelationModelTag = "RPB-v18";
inline constexpr const char *kUnitTemporalRelationArchitectureId =
    "early-native20-shape-plus12-fixed-unit-grouped-time-odd-relations-v1";
inline constexpr const char *kUnitTemporalRelationLayout =
    "shape20;fixed-unit-odd12;semantic-pairs-01-02-12;four-coordinates-per-pair;norm-floor1e-12";
inline constexpr const char *kUnitTemporalRelationTrainingRecipeId =
    "original-waveform-huber1;fixed-grouped-odd-prior432;pair-unit-L2-floor1e-12;shape-backbone-decoder-trainable-v1";
inline constexpr int64_t kUnitTemporalRelationParameterCount = 226877;
inline constexpr int64_t kUnitTemporalRelationTrainableParameterCount = 226445;
inline constexpr int64_t kUnitTemporalRelationFrozenParameterCount = 432;
inline constexpr double kUnitTemporalRelationNormFloor = 1e-12;

// All three semantic pairs use the same even denominator. For vectors above
// the floor this removes their magnitude, preserving their oriented direction.
// Clamp energy BEFORE sqrt: zero support has a finite derivative and stays
// exactly zero. Sub-floor vectors are divided by the floor, not quantized.
inline torch::Tensor unit_grouped_odd_relations(const torch::Tensor &odd12) {
  TORCH_CHECK(odd12.defined() && odd12.is_cuda() && odd12.scalar_type() == torch::kFloat32 &&
                  odd12.dim() == 2 && odd12.size(0) > 0 && odd12.size(1) == 12 &&
                  torch::isfinite(odd12).all().item<bool>(),
              "[rpb-unit-relations] finite actual-CUDA F32 odd12 required");
  const auto groups = odd12.reshape({odd12.size(0), 3, 4});
  const auto norm = groups.square().sum(-1, true).clamp_min(
      kUnitTemporalRelationNormFloor * kUnitTemporalRelationNormFloor).sqrt();
  const auto out = (groups / norm).reshape({odd12.size(0), 12});
  TORCH_CHECK(torch::isfinite(out).all().item<bool>(), "[rpb-unit-relations] finite normalized odd12 required");
  return out;
}

// No added modules, parameters, initialization draws or Config fields. The
// exact v17 inherited load validates and refreezes the same literal432 prior.
// Neither the original shape20 nor any diagnostic export is normalized.
struct UnitTemporalRelationModelImpl : FixedPriorTemporalRelationModelImpl {
  explicit UnitTemporalRelationModelImpl(const Config &c) : FixedPriorTemporalRelationModelImpl(c) {}

  EncodeOutput encode(const Input &input) {
    auto out = FixedPriorTemporalRelationModelImpl::encode(input);
    const auto native = out.z_contextual_global;
    const auto relations = unit_grouped_odd_relations(native.narrow(1, 20, 12));
    out.z_contextual_global = torch::cat({native.narrow(1, 0, 20), relations}, 1);
    return out;
  }

  // The base forward is nonvirtual and binds its own encode. This explicit
  // route ensures training and public inference use the same normalized32.
  ForwardOutput forward(const Input &input, const torch::Tensor &hidden) {
    const auto &c = config();
    validate_input(input, c, true);
    const auto masks = mask_from_hidden(input.observed.to(c.device), hidden.to(c.device), c);
    const Input visible{input.data, masks.visible, input.channel_ids, input.endpoints, input.sampling_interval};
    ForwardOutput out;
    out.encoding = encode(visible);
    out.reconstruction = decode(compact_reconstruction_export(out.encoding, c), input.channel_ids);
    const auto losses = hierarchical_huber(out.reconstruction, input.data.to(c.device).detach(),
        masks.target, masks.eligible_channels, c.huber_delta);
    out.loss = losses.loss; out.target_counts = losses.target_counts;
    out.eligible_channels = losses.eligible_channels; out.eligible_examples = losses.eligible_examples;
    out.eligible_channel_count = losses.eligible_channel_count;
    out.eligible_example_count = losses.eligible_example_count; out.target_cell_count = losses.target_cell_count;
    return out;
  }
};
TORCH_MODULE(UnitTemporalRelationModel);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
