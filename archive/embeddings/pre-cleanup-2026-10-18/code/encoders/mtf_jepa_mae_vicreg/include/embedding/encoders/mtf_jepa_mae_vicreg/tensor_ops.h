// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/mtf_jepa_mae_vicreg/types.h"
#include "embedding/shared/tensor_ops.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {
namespace detail {

// Baseline adapters keep metadata and configuration out of shared primitives.
inline mtf_input_t canonicalize_input(
    const torch::Tensor &x, const torch::Tensor &feature_mask,
    const mtf_jepa_mae_vicreg_config_t &config) {
  return ::embedding::detail::canonicalize_input(x, feature_mask,
                                                input_shape(config));
}

inline torch::Tensor domain_mask(const mtf_token_metadata_t &metadata,
                                 const torch::Tensor &token_mask,
                                 int64_t domain) {
  auto domain_ids = metadata.domain_id.to(token_mask.device());
  return token_mask.logical_and(domain_ids.eq(domain).unsqueeze(0));
}

inline torch::Tensor metadata_features(
    const mtf_token_metadata_t &metadata,
    const mtf_jepa_mae_vicreg_config_t &config) {
  TORCH_CHECK(metadata.start_index.defined(),
              "[embedding] metadata is undefined");
  const auto opts =
      torch::TensorOptions().dtype(config.dtype).device(config.device);
  const auto denom_t =
      static_cast<double>(std::max<int64_t>(1, config.history_length));
  const auto denom_scale =
      static_cast<double>(std::max<int64_t>(1, config.time_scales.size() - 1));
  const auto denom_channel =
      static_cast<double>(std::max<int64_t>(1, config.channel_count - 1));
  const auto start = metadata.start_index.to(opts) / denom_t;
  const auto center =
      (metadata.start_index.to(opts) + 0.5 * metadata.width.to(opts)) / denom_t;
  const auto width = metadata.width.to(opts) / denom_t;
  const auto scale = metadata.scale_id.to(opts) / denom_scale;
  const auto channel = metadata.channel_id.to(opts) / denom_channel;
  const auto domain = metadata.domain_id.to(opts);
  return torch::stack({start, center, width, scale, channel, domain},
                      /*dim=*/1);
}

inline torch::Tensor channel_mask(const mtf_token_metadata_t &metadata,
                                  const torch::Tensor &token_mask,
                                  int64_t channel) {
  return ::embedding::detail::channel_mask(metadata.channel_id, token_mask,
                                          channel);
}

inline torch::Tensor channel_valid_mask(
    const mtf_token_metadata_t &metadata, const torch::Tensor &token_mask,
    const mtf_jepa_mae_vicreg_config_t &config) {
  return ::embedding::detail::channel_valid_mask(
      metadata.channel_id, token_mask, config.channel_count);
}

inline torch::Tensor pooled_by_channel(
    const torch::Tensor &embeddings, const torch::Tensor &mask,
    const mtf_token_metadata_t &metadata,
    const mtf_jepa_mae_vicreg_config_t &config) {
  return ::embedding::detail::pooled_by_channel(
      embeddings, mask, metadata.channel_id, config.channel_count);
}

} // namespace detail
} // namespace embedding::encoders::mtf_jepa_mae_vicreg
