// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/spectral_temporal_relation.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/preprocessing.h"
#include <string>
#include <utility>

namespace cuwacunu::encoder {

// The embedding/ headers are an unchanged import. Only this encoder is part
// of the supported public API; inherited historical classes are implementation.
namespace imported = embedding::encoders::raw_patch_bottleneck_mae;
using Config = imported::Config;
using Input = imported::Input;
using Encoder = imported::SpectralTemporalRelationModel;
using FrozenScaler = imported::FrozenScaler;
using EncodeOutput = imported::EncodeOutput;
using ForwardOutput = imported::ForwardOutput;
inline constexpr const char *model_tag = imported::kSpectralTemporalRelationModelTag;
inline constexpr const char *architecture_id = imported::kSpectralTemporalRelationArchitectureId;
inline constexpr const char *native_layout = imported::kSpectralTemporalRelationLayout;

inline Config fixed_config(torch::Device device = torch::Device(torch::kCUDA, 0),
                           std::vector<int64_t> channel_ids = {0, 1, 2},
                           double sampling_interval = 1.0) {
  Config c;
  c.device = device;
  c.channel_ids = std::move(channel_ids);
  c.sampling_interval = sampling_interval;
  c.channel_mixer_layers = 1;
  c.channel_mixer_placement = 1;
  c.global_bottleneck_mode = 2;
  imported::validate_temporal_relation_config(c);
  return c;
}

inline torch::Tensor native_embedding(const EncodeOutput &encoded, const Config &c) {
  return imported::compact_reconstruction_export(encoded, c);
}

// Fit on the TRAIN population only. This preserved helper takes a complete
// input tensor; it is not yet a streaming statistics accumulator.
inline FrozenScaler fit_train_scaler(const Input &training, const Config &c) {
  return imported::fit_scaler(training, c);
}

struct LoadedEncoder {
  Encoder model{nullptr};
  FrozenScaler scaler;
};

namespace detail {
inline std::string archive_text(torch::serialize::InputArchive &archive, const char *key) {
  torch::Tensor value;
  archive.read(key, value, true);
  const auto bytes = value.to(torch::kCPU).contiguous();
  TORCH_CHECK(bytes.scalar_type() == torch::kUInt8 && bytes.dim() == 1,
              "[cuwacunu encoder] malformed checkpoint text: ", key);
  if (!bytes.numel()) return {};
  return std::string(reinterpret_cast<const char *>(bytes.const_data_ptr<uint8_t>()),
                     static_cast<size_t>(bytes.numel()));
}
inline int64_t archive_integer(torch::serialize::InputArchive &archive, const char *key) {
  torch::Tensor value;
  archive.read(key, value, true);
  TORCH_CHECK(value.scalar_type() == torch::kInt64 && value.dim() == 0,
              "[cuwacunu encoder] malformed checkpoint integer: ", key);
  return value.item<int64_t>();
}

inline LoadedEncoder load_confirmation_archive(torch::serialize::InputArchive &archive,
                                                const Config &c) {
  imported::validate_temporal_relation_config(c);
  TORCH_CHECK(c.device.is_cuda(), "[cuwacunu encoder] confirmation inference requires CUDA");
  // This research format recorded settings as a fixed recipe, not a portable
  // configurable model schema. Reject reinterpretation with different metadata.
  const auto reference = fixed_config(c.device);
  TORCH_CHECK(imported::resolved_channel_ids(c) == imported::resolved_channel_ids(reference) &&
                  c.sampling_interval == reference.sampling_interval &&
                  c.layer_norm_epsilon == reference.layer_norm_epsilon &&
                  c.mask_ratio == reference.mask_ratio && c.huber_delta == reference.huber_delta &&
                  c.scale_floor == reference.scale_floor,
              "[cuwacunu encoder] confirmation requires its original IDs/interval/numerical settings");
  TORCH_CHECK(archive_text(archive, "artifact_kind") == "rpb_spectral_confirmation_checkpoint_v1" &&
                  archive_text(archive, "candidate") == "v19" &&
                  archive_text(archive, "protocol") == "spectral-confirmation-v1" &&
                  archive_text(archive, "architecture_id") == architecture_id &&
                  archive_text(archive, "native_layout") == native_layout &&
                  archive_text(archive, "training_recipe_id") == imported::kSpectralTemporalRelationTrainingRecipeId &&
                  archive_text(archive, "frozen_parameter_name") == "grouped_odd_relation_projection.weight",
              "[cuwacunu encoder] confirmation checkpoint architecture/layout/recipe mismatch");
  TORCH_CHECK(archive_integer(archive, "parameter_count") == imported::kSpectralTemporalRelationParameterCount &&
                  archive_integer(archive, "trainable_parameter_count") == imported::kSpectralTemporalRelationTrainableParameterCount &&
                  archive_integer(archive, "frozen_parameter_count") == imported::kSpectralTemporalRelationFrozenParameterCount,
              "[cuwacunu encoder] confirmation checkpoint capacity mismatch");
  LoadedEncoder loaded{Encoder(c), {}};
  torch::serialize::InputArchive model, scaler;
  archive.read("model", model);
  loaded.model->load(model);
  loaded.model->validate_fixed_prior();
  loaded.model->eval();
  archive.read("scaler", scaler);
  loaded.scaler = FrozenScaler::load(scaler);
  loaded.scaler.validate(c);
  return loaded;
}
} // namespace detail

// Inference import only: optimizer/counters/RNG remain research artifacts.
// Check the caller's SHA-256 provenance manifest separately before loading.
inline LoadedEncoder load_confirmation(const std::string &path,
                                        const Config &c = fixed_config()) {
  torch::serialize::InputArchive archive;
  archive.load_from(path, c.device);
  return detail::load_confirmation_archive(archive, c);
}

} // namespace cuwacunu::encoder
