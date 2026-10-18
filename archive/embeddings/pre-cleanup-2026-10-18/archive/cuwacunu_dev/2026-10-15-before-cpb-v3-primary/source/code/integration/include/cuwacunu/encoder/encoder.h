// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "cuwacunu/config/document.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h"
#include <filesystem>
#include <string_view>

namespace cuwacunu::encoder::primary {

namespace version = embedding::encoders::causal_patch_bottleneck_mae::v1;
using Config = version::Config;
using RawHistory = version::RawHistory;
using FrozenScaler = version::FrozenScaler;
using Encoding = version::Encoding;
using Encoder = version::Encoder;
using LoadedEncoder = version::Loaded;
inline constexpr const char *model_tag = "CPB-v1";
inline constexpr const char *instance = "CPB-v1.market-01";
inline constexpr const char *checkpoint_sha256 =
    "54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895";
inline constexpr int64_t native_width = 36;

// The application selects one evaluated instance. Geometry changes and new
// training belong to a separately named research version, not a serving toggle.
inline void validate_config(const Config &config) {
  const auto &c = config.level;
  TORCH_CHECK(c.channel_count == 3 && c.history_length == 128 &&
      c.input_width == 9 && c.patch_length == 8 && c.export_width == 32 &&
      config.change_width == 4 && c.encoder_width == 64 && c.num_layers == 3 &&
      c.num_heads == 4 && c.feedforward_width == 256 && c.decoder_hidden_width == 128 &&
      c.channel_ids == std::vector<int64_t>({0, 1, 2}) &&
      c.channel_mixer_layers == 1 && c.channel_mixer_placement == 1 &&
      c.global_bottleneck_mode == 2 && c.global_pool_input_source == 0 &&
      c.temporal_difference_input == 0 && c.dropout == 0. &&
      c.mask_ratio == .25 && c.huber_delta == 1. && c.scale_floor == 1e-6 &&
      c.layer_norm_epsilon == 1e-5 && c.sampling_interval == 60000. &&
      c.dtype == torch::kFloat32 && c.device.is_cuda() && c.device.index() == 0,
      "[primary encoder] CPB-v1.market-01 requires its exact saved CUDA:0 geometry/settings");
}

inline Config read_settings(const std::filesystem::path &path = "config/encoder.conf") {
  auto result = version::read_settings(path);
  validate_config(result);
  return result;
}

struct RuntimeSelection {
  std::filesystem::path checkpoint;
  std::string checkpoint_hash;
};

inline RuntimeSelection read_runtime(
    const std::filesystem::path &path = "config/encoder_runtime.conf") {
  const auto document = cuwacunu::config::Document::read(path);
  document.reject_unknown_blocks({"ENCODER_RUNTIME"});
  const auto &block = document.block("ENCODER_RUNTIME");
  block.reject_unknown_keys({"MODEL_TAG", "INSTANCE", "CHECKPOINT", "CHECKPOINT_SHA256"});
  TORCH_CHECK(block.string("MODEL_TAG") == model_tag &&
      block.string("INSTANCE") == instance &&
      block.string("CHECKPOINT_SHA256") == checkpoint_sha256,
      "[primary encoder] runtime must select the exact CPB-v1.market-01 artifact");
  return {document.resolve_path(block.string("CHECKPOINT")), block.string("CHECKPOINT_SHA256")};
}

inline LoadedEncoder load_checkpoint(const std::filesystem::path &path,
    const Config &config) {
  validate_config(config);
  TORCH_CHECK(std::filesystem::is_regular_file(path) &&
      version::boundary::sha256_file(path) == checkpoint_sha256,
      "[primary encoder] selected checkpoint missing or SHA256 differs; no fallback is available");
  auto loaded = version::load_checkpoint(path.string(), config.level.device, &config);
  loaded.model->validate_frozen();
  TORCH_CHECK(version::boundary::sha256_file(path) == checkpoint_sha256,
      "[primary encoder] selected checkpoint changed while loading");
  return loaded;
}

inline LoadedEncoder load_selected(
    const std::filesystem::path &settings = "config/encoder.conf",
    const std::filesystem::path &runtime = "config/encoder_runtime.conf") {
  const auto selection = read_runtime(runtime);
  return load_checkpoint(selection.checkpoint, read_settings(settings));
}

// Explicit time is required even when the supplied history ends at that time.
inline Encoding encode_at(LoadedEncoder &loaded, const RawHistory &history,
    const torch::Tensor &cutoffs) {
  validate_config(loaded.config);
  loaded.model->validate_frozen();
  auto result = loaded.model->encode_at(history, cutoffs, loaded.scaler);
  loaded.model->validate_frozen();
  return result;
}

} // namespace cuwacunu::encoder::primary
