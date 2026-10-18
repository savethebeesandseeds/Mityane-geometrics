// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/mtf_jepa_mae_vicreg/tensor_ops.h"
#include "embedding/shared/objectives.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {

namespace detail {

inline double resolved_vicreg_view_time_dropout_prob(
    const mtf_jepa_mae_vicreg_config_t &config) {
  return std::min(0.10,
                  std::max(0.0, config.mask_ratio_time *
                                    config.vicreg_view_time_dropout_scale));
}

inline mtf_input_t apply_vicreg_weak_view_augmentation(
    const torch::Tensor &x, const torch::Tensor &feature_mask,
    const mtf_jepa_mae_vicreg_config_t &config) {
  auto input = canonicalize_input(x, feature_mask, config);
  auto mask = input.feature_mask.clone();
  const double feature_drop =
      std::min(0.20, std::max(0.0, config.mask_ratio_channel));
  if (feature_drop > 0.0) {
    const auto keep = torch::rand(input.data.sizes(), input.data.options())
                          .lt(1.0 - feature_drop);
    mask = mask.logical_and(keep);
  }
  const double time_drop = resolved_vicreg_view_time_dropout_prob(config);
  // Keep the random draw schedule stable when an augmentation strength is zero.
  if (config.mask_ratio_time > 0.0) {
    auto keep_shape = input.data.sizes().vec();
    keep_shape.back() = 1;
    const auto keep = torch::rand(keep_shape, input.data.options())
                          .lt(1.0 - time_drop)
                          .expand_as(input.data);
    mask = mask.logical_and(keep);
  }
  auto data = torch::where(mask, input.data, torch::zeros_like(input.data));
  data = data + torch::where(mask,
                             torch::randn_like(data) *
                                 config.vicreg_view_gaussian_jitter_std,
                             torch::zeros_like(data));
  data = torch::where(mask, data, torch::zeros_like(data));
  return {data, mask};
}

} // namespace detail

} // namespace embedding::encoders::mtf_jepa_mae_vicreg
