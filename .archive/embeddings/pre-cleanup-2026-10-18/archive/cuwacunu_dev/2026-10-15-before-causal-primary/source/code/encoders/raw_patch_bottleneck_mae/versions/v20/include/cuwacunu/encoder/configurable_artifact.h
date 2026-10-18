// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once
#include "cuwacunu/encoder/configurable_encoder.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_settings.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_checkpoint.h"

namespace cuwacunu::encoder::configurable {
using LoadedEncoder = implementation::LoadedConfigurableSpectralEncoder;
inline Config read_settings(const std::filesystem::path &path) {
  return implementation::read_configurable_spectral_settings(path);
}
inline void save_checkpoint(const std::string &path, const Encoder &model,
                            const FrozenScaler &scaler, int64_t updates) {
  implementation::save_configurable_spectral_checkpoint(path, model, scaler, updates);
}
inline LoadedEncoder load_checkpoint(const std::string &path,
    torch::Device device = torch::Device(torch::kCUDA, 0), const Config *expected = nullptr) {
  return implementation::load_configurable_spectral_checkpoint(path, device, expected);
}
} // namespace cuwacunu::encoder::configurable
