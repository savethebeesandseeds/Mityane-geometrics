// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once
#include "cuwacunu/data/sample.h"
#include "cuwacunu/encoder/encoder.h"
#include <algorithm>
#include <array>

namespace cuwacunu::integration {

// The caller declares feature selection; there is no default market recipe.
// A future generalized encoder can consume all features through this boundary.
inline encoder::Input encoder_input(const data::sample_t &sample,
                                    const std::array<int64_t, 3> &feature_columns,
                                    const torch::Tensor &channel_ids,
                                    const encoder::Config &config) {
  encoder::imported::validate_temporal_relation_config(config);
  const auto &values = sample.features;
  const auto &mask = sample.mask;
  const auto &keys = sample.past_keys;
  TORCH_CHECK(!sample.normalized,
              "[cuwacunu integration] use raw loader values before the TRAIN-fitted encoder scaler");
  TORCH_CHECK(values.defined() && values.device().is_cpu() && values.is_floating_point() &&
                  values.dim() == 4 && values.size(0) > 0 &&
                  values.size(1) == config.channel_count && values.size(2) == config.history_length,
              "[cuwacunu integration] expected a CPU past batch [B,3,32,F]");
  TORCH_CHECK(mask.defined() && mask.device().is_cpu() && mask.scalar_type() == torch::kBool &&
                  mask.sizes() == values.sizes().slice(0, 3),
              "[cuwacunu integration] expected boolean row validity [B,3,32]");
  TORCH_CHECK(keys.defined() && keys.device().is_cpu() && keys.sizes() == mask.sizes() &&
                  (keys.is_floating_point() || keys.scalar_type() == torch::kInt64),
              "[cuwacunu integration] expected timestamp grid [B,3,32]");
  for (size_t i = 0; i < feature_columns.size(); ++i) {
    TORCH_CHECK(feature_columns[i] >= 0 && feature_columns[i] < values.size(3),
                "[cuwacunu integration] selected feature is outside the loader schema");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(feature_columns[i] != feature_columns[j],
                  "[cuwacunu integration] selected features must be distinct");
  }
  auto times = keys.to(torch::kFloat64);
  auto common = times.select(1, 0);
  TORCH_CHECK(torch::isfinite(times).all().item<bool>() &&
                  torch::equal(times, common.unsqueeze(1).expand_as(times)),
              "[cuwacunu integration] native channel timestamps differ; declare alignment before encoding");
  auto differences = common.slice(1, 1) - common.slice(1, 0, config.history_length - 1);
  TORCH_CHECK(differences.gt(0).all().item<bool>() &&
                  torch::allclose(differences, torch::full_like(differences, config.sampling_interval),
                                  1e-9, config.sampling_interval * 1e-9),
              "[cuwacunu integration] encoder requires a complete uniformly sampled timestamp grid");
  auto indices = torch::tensor(std::vector<int64_t>(feature_columns.begin(), feature_columns.end()),
                               torch::kInt64);
  auto selected = values.index_select(3, indices);
  auto observed = mask.unsqueeze(-1).expand_as(selected).clone();
  encoder::Input input{torch::where(observed, selected, torch::zeros_like(selected)),
                        observed, channel_ids, common.select(1, config.history_length - 1).clone(),
                        config.sampling_interval};
  encoder::imported::validate_input(input, config);
  // future_features, future_mask and future_keys deliberately never enter Input.
  return input;
}
} // namespace cuwacunu::integration
