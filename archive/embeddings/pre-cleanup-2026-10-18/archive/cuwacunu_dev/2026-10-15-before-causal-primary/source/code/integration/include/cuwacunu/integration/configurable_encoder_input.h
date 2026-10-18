// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once

#include "cuwacunu/data/kline_feature_registry.h"
#include "cuwacunu/data/sample.h"
#include "cuwacunu/encoder/configurable_encoder.h"

namespace cuwacunu::integration {

// All nine kline coordinates retain their registry order. The application
// boundary accepts raw CPU windows; the TRAIN-fitted encoder scaler performs
// centering/scaling and the one transfer to the configured CUDA device.
inline encoder::configurable::Input configurable_encoder_input(
    const data::sample_t &sample, const torch::Tensor &channel_ids,
    const encoder::configurable::Config &config) {
  encoder::configurable::validate_config(config);
  const auto &values = sample.features;
  const auto &mask = sample.mask;
  const auto &keys = sample.past_keys;
  TORCH_CHECK(!sample.normalized,
              "[cuwacunu integration] use raw loader values before the TRAIN-fitted encoder scaler");
  TORCH_CHECK(config.input_width == data::kKlineFeatureWidth,
              "[cuwacunu integration] kline input requires all nine registry features");
  TORCH_CHECK(values.defined() && values.device().is_cpu() &&
                  (values.scalar_type() == torch::kFloat32 ||
                   values.scalar_type() == torch::kFloat64) &&
                  values.dim() == 4 && values.size(0) > 0 &&
                  values.size(1) == config.channel_count &&
                  values.size(2) == config.history_length &&
                  values.size(3) == config.input_width,
              "[cuwacunu integration] expected raw CPU past values [B,C,H,9] matching encoder settings");
  TORCH_CHECK(mask.defined() && mask.device().is_cpu() &&
                  mask.scalar_type() == torch::kBool &&
                  mask.sizes() == values.sizes().slice(0, 3),
              "[cuwacunu integration] expected boolean row validity [B,C,H]");
  TORCH_CHECK(keys.defined() && keys.device().is_cpu() &&
                  keys.sizes() == mask.sizes() &&
                  (keys.is_floating_point() || keys.scalar_type() == torch::kInt64),
              "[cuwacunu integration] expected timestamp grid [B,C,H]");
  const auto common_keys = keys.select(1, 0);
  TORCH_CHECK(torch::equal(keys, common_keys.unsqueeze(1).expand_as(keys)),
              "[cuwacunu integration] native channel timestamps differ; declare alignment before encoding");
  const auto times = common_keys.to(torch::kFloat64);
  const auto differences = times.slice(1, 1) - times.slice(1, 0, config.history_length - 1);
  TORCH_CHECK(torch::isfinite(times).all().item<bool>() &&
                  differences.gt(0).all().item<bool>() &&
                  torch::allclose(differences, torch::full_like(differences, config.sampling_interval),
                                  1e-9, config.sampling_interval * 1e-9),
              "[cuwacunu integration] encoder requires a complete uniformly sampled timestamp grid");

  // A missing candle masks its whole row. A nonfinite source value masks only
  // that coordinate; finite siblings in the candle remain usable. Hidden data
  // is always zero before the scaler, so NaNs cannot enter encoder arithmetic.
  auto observed = mask.unsqueeze(-1).expand_as(values).logical_and(torch::isfinite(values));
  encoder::configurable::Input input{
      torch::where(observed, values, torch::zeros_like(values)), observed,
      channel_ids, times.select(1, config.history_length - 1).clone(),
      config.sampling_interval};
  encoder::configurable::implementation::validate_input(
      input, encoder::configurable::input_config(config));
  // future_features, future_mask and future_keys never participate in Input.
  return input;
}

} // namespace cuwacunu::integration
