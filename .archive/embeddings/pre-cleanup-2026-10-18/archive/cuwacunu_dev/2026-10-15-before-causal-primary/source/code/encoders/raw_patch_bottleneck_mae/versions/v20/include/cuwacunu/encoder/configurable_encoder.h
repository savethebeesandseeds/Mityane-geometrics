// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_model.h"

namespace cuwacunu::encoder::configurable {

namespace implementation = embedding::encoders::raw_patch_bottleneck_mae;
namespace imported = implementation;
using Config = implementation::Config;
using Encoder = implementation::ConfigurableSpectralModel;
using Input = implementation::Input;
using FrozenScaler = implementation::FrozenScaler;
using EncodeOutput = implementation::EncodeOutput;
using ForwardOutput = implementation::ForwardOutput;
using MaskPlan = implementation::MaskPlan;
inline constexpr const char *model_tag = implementation::kConfigurableSpectralModelTag;
inline constexpr const char *architecture_id = implementation::kConfigurableSpectralArchitectureId;
inline constexpr const char *native_layout = implementation::kConfigurableSpectralLayout;

inline Config default_config(torch::Device device = torch::Device(torch::kCUDA, 0)) {
  return implementation::default_configurable_spectral_config(device);
}
inline void validate_config(const Config &c) {
  implementation::validate_configurable_spectral_config(c);
}
inline Config input_config(const Config &c) {
  implementation::validate_configurable_spectral_config(c);
  return implementation::configurable_spectral_input_config(c);
}
inline Config backbone_config(const Config &c) {
  implementation::validate_configurable_spectral_config(c);
  return implementation::configurable_spectral_backbone_config(c);
}
inline int64_t pair_count(const Config &c) {
  return implementation::configurable_spectral_pair_count(c);
}
inline int64_t spectral_width(const Config &c) {
  return implementation::configurable_spectral_width(c);
}
inline int64_t shape_width(const Config &c) {
  return implementation::configurable_spectral_shape_width(c);
}
inline torch::Tensor native_embedding(const EncodeOutput &encoded, const Config &c) {
  return implementation::compact_reconstruction_export(encoded, c);
}
inline FrozenScaler fit_train_scaler(const Input &training, const Config &c) {
  return implementation::fit_configurable_spectral_scaler(training, c);
}
inline Input transform(const FrozenScaler &scaler, const Input &input, const Config &c) {
  return implementation::transform_configurable_spectral_input(scaler, input, c);
}
inline MaskPlan make_training_mask(const torch::Tensor &observed, const Config &c, uint64_t seed) {
  return implementation::configurable_spectral_make_training_mask(observed, c, seed);
}
inline MaskPlan mask_from_hidden(const torch::Tensor &observed, const torch::Tensor &hidden,
                                const Config &c) {
  return implementation::configurable_spectral_mask_from_hidden(observed, hidden, c);
}

} // namespace cuwacunu::encoder::configurable
