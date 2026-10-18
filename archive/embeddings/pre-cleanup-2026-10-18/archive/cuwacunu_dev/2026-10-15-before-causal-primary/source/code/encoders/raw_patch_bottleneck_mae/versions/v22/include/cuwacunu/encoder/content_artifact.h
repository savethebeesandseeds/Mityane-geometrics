// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once

#include "cuwacunu/encoder/configurable_encoder.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_settings.h"

// Explicit optional API. The selected configurable/v20 facade stays unchanged.
namespace cuwacunu::encoder::content {

namespace implementation = embedding::encoders::raw_patch_bottleneck_mae;
using Config = configurable::Config;
using Input = configurable::Input;
using FrozenScaler = configurable::FrozenScaler;
using EncodeOutput = configurable::EncodeOutput;
using ForwardOutput = configurable::ForwardOutput;
using MaskPlan = configurable::MaskPlan;
using Encoder = implementation::ContentOrderedRelationModel;
using LoadedEncoder = implementation::LoadedContentOrderedRelationEncoder;
inline constexpr const char *model_tag = implementation::kContentOrderedRelationModelTag;
inline constexpr const char *architecture_id = implementation::kContentOrderedRelationArchitectureId;
inline constexpr const char *native_layout = implementation::kContentOrderedRelationLayout;

inline Config default_config(torch::Device device = torch::Device(torch::kCUDA, 0)) {
  return implementation::default_content_ordered_relation_config(device);
}
inline void validate_config(const Config &c) {
  implementation::validate_content_ordered_relation_config(c);
}
inline Config read_settings(const std::filesystem::path &path) {
  return implementation::read_content_ordered_relation_settings(path);
}

// The geometry, preprocessing, masks and native export use the existing base.
using configurable::input_config;
using configurable::backbone_config;
using configurable::pair_count;
using configurable::spectral_width;
using configurable::shape_width;
using configurable::native_embedding;
using configurable::fit_train_scaler;
using configurable::transform;
using configurable::make_training_mask;
using configurable::mask_from_hidden;

inline void save_checkpoint(const std::string &path, const Encoder &model,
                            const FrozenScaler &scaler, int64_t updates) {
  implementation::save_content_ordered_relation_checkpoint(path, model, scaler, updates);
}
inline LoadedEncoder load_checkpoint(const std::string &path,
    torch::Device device = torch::Device(torch::kCUDA, 0), const Config *expected = nullptr) {
  return implementation::load_content_ordered_relation_checkpoint(path, device, expected);
}

} // namespace cuwacunu::encoder::content
