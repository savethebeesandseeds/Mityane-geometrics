// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "cuwacunu/encoder/content_artifact.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/consecutive_change_settings.h"

// Explicit candidate facade. The primary RPB-v22 selection remains unchanged.
namespace cuwacunu::encoder::changes {
namespace implementation=embedding::encoders::raw_patch_bottleneck_mae;
using Config=content::Config;using Input=content::Input;using FrozenScaler=content::FrozenScaler;
using EncodeOutput=content::EncodeOutput;using Encoder=implementation::ConsecutiveChangeModel;
using LoadedEncoder=implementation::LoadedConsecutiveChangeEncoder;
using content::input_config;using content::transform;using content::native_embedding;
using content::make_training_mask;using content::mask_from_hidden;
inline void validate_config(const Config &c) { implementation::validate_consecutive_change_config(c); }
inline Config read_settings(const std::filesystem::path &path) { return implementation::read_consecutive_change_settings(path); }
inline void save_checkpoint(const std::string &path,const Encoder &model,const FrozenScaler &scaler,int64_t updates) {
  implementation::save_consecutive_change_checkpoint(path,model,scaler,updates);
}
} // namespace cuwacunu::encoder::changes
