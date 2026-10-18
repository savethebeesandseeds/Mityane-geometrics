// SPDX-License-Identifier: MIT
#pragma once

#include "cuwacunu/config/document.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_config.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

// Portable model settings only: no market sources, loader or application runtime.
inline Config configurable_spectral_settings(const cuwacunu::config::Document &document) {
  document.reject_unknown_blocks({"ENCODER"});
  const auto &b = document.block("ENCODER");
  b.reject_unknown_keys({"MODEL_TAG", "CHANNEL_COUNT", "HISTORY_LENGTH", "INPUT_WIDTH",
      "PATCH_LENGTH", "EMBEDDING_SIZE", "ENCODER_WIDTH", "NUM_LAYERS", "NUM_HEADS",
      "FEEDFORWARD_WIDTH", "DECODER_HIDDEN_WIDTH", "LAYER_NORM_EPSILON", "MASK_RATIO",
      "HUBER_DELTA", "SCALE_FLOOR", "SAMPLING_INTERVAL", "CHANNEL_IDS", "DEVICE"});
  TORCH_CHECK(b.string_or("MODEL_TAG", kConfigurableSpectralModelTag) == kConfigurableSpectralModelTag,
              "[rpb-v20 settings] MODEL_TAG must be RPB-v20");
  const auto device_text = b.string_or("DEVICE", "cuda:0");
  TORCH_CHECK(device_text == "cuda" || device_text.rfind("cuda:", 0) == 0,
              "[rpb-v20 settings] DEVICE must name a CUDA device");
  auto c = default_configurable_spectral_config(torch::Device(device_text));
  c.channel_count = b.integer_or("CHANNEL_COUNT", c.channel_count);
  c.history_length = b.integer_or("HISTORY_LENGTH", c.history_length);
  c.input_width = b.integer_or("INPUT_WIDTH", c.input_width);
  c.patch_length = b.integer_or("PATCH_LENGTH", c.patch_length);
  c.export_width = b.integer_or("EMBEDDING_SIZE", c.export_width);
  c.encoder_width = b.integer_or("ENCODER_WIDTH", c.encoder_width);
  c.num_layers = b.integer_or("NUM_LAYERS", c.num_layers);
  c.num_heads = b.integer_or("NUM_HEADS", c.num_heads);
  c.feedforward_width = b.integer_or("FEEDFORWARD_WIDTH", c.feedforward_width);
  c.decoder_hidden_width = b.integer_or("DECODER_HIDDEN_WIDTH", c.decoder_hidden_width);
  c.layer_norm_epsilon = b.number_or("LAYER_NORM_EPSILON", c.layer_norm_epsilon);
  c.mask_ratio = b.number_or("MASK_RATIO", c.mask_ratio);
  c.huber_delta = b.number_or("HUBER_DELTA", c.huber_delta);
  c.scale_floor = b.number_or("SCALE_FLOOR", c.scale_floor);
  c.sampling_interval = b.number_or("SAMPLING_INTERVAL", c.sampling_interval);
  c.channel_ids = b.integers_or("CHANNEL_IDS", {});
  validate_configurable_spectral_config(c);
  c.channel_ids = resolved_channel_ids(c);
  return c;
}

inline Config read_configurable_spectral_settings(const std::filesystem::path &path) {
  return configurable_spectral_settings(cuwacunu::config::Document::read(path));
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
