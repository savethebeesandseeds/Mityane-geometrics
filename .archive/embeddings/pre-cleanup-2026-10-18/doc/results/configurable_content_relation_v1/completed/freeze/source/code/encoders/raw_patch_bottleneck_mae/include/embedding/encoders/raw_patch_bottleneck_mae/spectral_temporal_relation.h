// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/unit_temporal_relation.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/spectral_relation_bank.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kSpectralTemporalRelationModelTag = "RPB-v19";
inline constexpr const char *kSpectralTemporalRelationArchitectureId =
    "early-native20-shape-plus12-fixed-two-band-complex-relations-v1";
inline constexpr const char *kSpectralTemporalRelationLayout =
    "shape20;spectral12;semantic-pairs-01-02-12;ReLow-ImLow-ReHigh-ImHigh;"
    "q1-3-low-q4-8-high;joint-pair-unit-L2-floor1e-12";
inline constexpr const char *kSpectralTemporalRelationTrainingRecipeId =
    "original-waveform-huber1;fixed-feature-mean432;fixed-q1-8-masked-harmonic-relations;"
    "joint-pair-unit-L2-floor1e-12;shape-backbone-decoder-trainable-v1";
inline constexpr int64_t kSpectralTemporalRelationParameterCount = 226877;
inline constexpr int64_t kSpectralTemporalRelationTrainableParameterCount = 226445;
inline constexpr int64_t kSpectralTemporalRelationFrozenParameterCount = 432;

// Exact v18 constructor/RNG/parameter names/load/frozen432 are inherited. The
// historical grouped_odd_relation_projection name is an implementation name:
// here its same literal1/9 map averages four MIXED even/odd spectral metrics.
// No old model or archive is relabelled; the new artifact contract binds this
// distinct layout. There are no extra modules, parameters or random draws.
struct SpectralTemporalRelationModelImpl : UnitTemporalRelationModelImpl {
  explicit SpectralTemporalRelationModelImpl(const Config &c) : UnitTemporalRelationModelImpl(c) {}
  EncodeOutput encode(const Input &input) {
    const auto bank = spectral_relation_bank(input, config());
    auto out = backbone->encode(input);
    const auto shape = native_shape_projection(out.z_contextual_global);
    const auto spectral = unit_spectral_relations(grouped_odd_relation_projection(bank.values));
    const auto native32 = torch::cat({shape,spectral},1);
    out.z_contextual_global = torch::where(out.sample_valid_mask.unsqueeze(-1), native32,
                                          torch::zeros_like(native32));
    TORCH_CHECK(out.z_contextual_global.size(1) == 32 && torch::isfinite(out.z_contextual_global).all().item<bool>(),
                "[rpb-spectral-relations] finite spectral native32 required");
    return out;
  }
  // Base forward is nonvirtual; this explicit route uses THIS encode with the
  // actual Q-visible mask and sends only the concatenated32 to the decoder.
  ForwardOutput forward(const Input &input, const torch::Tensor &hidden) {
    const auto &c = config();
    validate_input(input,c,true);
    const auto masks = mask_from_hidden(input.observed.to(c.device),hidden.to(c.device),c);
    const Input visible{input.data,masks.visible,input.channel_ids,input.endpoints,input.sampling_interval};
    ForwardOutput out;out.encoding = encode(visible);
    out.reconstruction = decode(compact_reconstruction_export(out.encoding,c),input.channel_ids);
    const auto losses = hierarchical_huber(out.reconstruction,input.data.to(c.device).detach(),
        masks.target,masks.eligible_channels,c.huber_delta);
    out.loss=losses.loss;out.target_counts=losses.target_counts;
    out.eligible_channels=losses.eligible_channels;out.eligible_examples=losses.eligible_examples;
    out.eligible_channel_count=losses.eligible_channel_count;
    out.eligible_example_count=losses.eligible_example_count;out.target_cell_count=losses.target_cell_count;
    return out;
  }
};
TORCH_MODULE(SpectralTemporalRelationModel);
} // namespace embedding::encoders::raw_patch_bottleneck_mae
