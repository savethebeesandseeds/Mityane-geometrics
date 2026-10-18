// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "cuwacunu/data/kline_feature_registry.h"
#include "cuwacunu/data/sample.h"
#include "cuwacunu/encoder/encoder.h"
#include <set>

namespace cuwacunu::integration {

// Shape/schema checks do not read observations. In particular, the adapter
// neither inspects finite values nor sanitizes masks across a future suffix.
// The version-owned encode_at boundary crops before any value processing.
inline encoder::primary::RawHistory encoder_input(const data::sample_t &sample,
    const torch::Tensor &channel_ids, const encoder::primary::Config &config,
    const torch::Tensor &feature_observed = {}) {
  namespace primary = encoder::primary;
  primary::validate_config(config);
  const auto &c = config.level;
  const auto &values = sample.features;
  const auto &mask = sample.mask;
  const auto &keys = sample.past_keys;
  TORCH_CHECK(!sample.normalized,
      "[causal input] raw records required; the selected checkpoint owns its frozen scaler");
  TORCH_CHECK(values.defined() && values.device().is_cpu() &&
      values.scalar_type() == torch::kFloat64 && values.dim() == 4 &&
      values.size(0) > 0 && values.size(1) == c.channel_count && values.size(2) > 0 &&
      values.size(3) == data::kKlineFeatureWidth,
      "[causal input] genuine raw CPU F64 B,C,N,9 required; F32 upcasts lose source precision");
  const auto B = values.size(0), N = values.size(2);
  TORCH_CHECK(mask.defined() && mask.device().is_cpu() && mask.scalar_type() == torch::kBool &&
      (mask.sizes() == values.sizes() || mask.sizes() == values.sizes().slice(0, 3)),
      "[causal input] CPU bool row B,C,N or per-feature B,C,N,9 mask required");
  auto observed = mask.dim() == 3 ? mask.unsqueeze(-1).expand_as(values) : mask;
  if (feature_observed.defined()) {
    TORCH_CHECK(feature_observed.device().is_cpu() && feature_observed.scalar_type() == torch::kBool &&
        feature_observed.sizes() == values.sizes(), "[causal input] matching CPU bool per-feature mask required");
    observed = observed.logical_and(feature_observed);
  }
  TORCH_CHECK(channel_ids.defined() && channel_ids.device().is_cpu() &&
      channel_ids.scalar_type() == torch::kInt64 &&
      ((channel_ids.dim() == 1 && channel_ids.size(0) == c.channel_count) ||
       (channel_ids.dim() == 2 && channel_ids.sizes() == torch::IntArrayRef({B, c.channel_count}))),
      "[causal input] explicit CPU int64 semantic channel IDs C or B,C required");
  const auto ids = (channel_ids.dim() == 1 ? channel_ids.unsqueeze(0).expand({B, c.channel_count})
                                         : channel_ids).contiguous();
  const auto access = ids.accessor<int64_t, 2>();
  for (int64_t b = 0; b < B; ++b) {
    std::set<int64_t> seen;
    for (int64_t ch = 0; ch < c.channel_count; ++ch)
      TORCH_CHECK(access[b][ch] >= 0 && access[b][ch] < c.channel_count &&
          seen.insert(access[b][ch]).second, "[causal input] unknown or duplicated semantic channel ID");
  }
  TORCH_CHECK(keys.defined() && keys.device().is_cpu() &&
      (keys.scalar_type() == torch::kInt64 || keys.scalar_type() == torch::kFloat64) &&
      ((keys.dim() == 2 && keys.sizes() == torch::IntArrayRef({B, N})) ||
       (keys.dim() == 3 && keys.sizes() == torch::IntArrayRef({B, c.channel_count, N}))),
      "[causal input] actual CPU int64/F64 timestamps B,N or B,C,N required");
  if (keys.scalar_type() == torch::kInt64)
    TORCH_CHECK(keys.ge(-9007199254740992LL).all().item<bool>() &&
        keys.le(9007199254740992LL).all().item<bool>(), "[causal input] integer timestamps must be exactly representable as F64");
  const auto common = keys.dim() == 3 ? keys.select(1, 0) : keys;
  if (keys.dim() == 3)
    TORCH_CHECK(torch::equal(keys, common.unsqueeze(1).expand_as(keys)),
        "[causal input] channel timestamps differ; explicit alignment is required");
  const auto times = common.to(torch::kFloat64);
  TORCH_CHECK(torch::isfinite(times).all().item<bool>(), "[causal input] finite declared timestamp schema required");
  if (N > 1) {
    const auto differences = times.narrow(1, 1, N-1) - times.narrow(1, 0, N-1);
    TORCH_CHECK(differences.gt(0).all().item<bool>() &&
        torch::allclose(differences, torch::full_like(differences, c.sampling_interval),
                        1e-9, c.sampling_interval*1e-9),
        "[causal input] ordered uniform configured close-time grid required");
  }
  // These tensors remain raw; future_features/mask/keys never enter this view.
  return {values, observed, channel_ids, times, c.sampling_interval};
}

inline encoder::primary::Encoding encoder_input(encoder::primary::LoadedEncoder &loaded,
    const data::sample_t &sample, const torch::Tensor &channel_ids,
    const torch::Tensor &cutoffs, const torch::Tensor &feature_observed = {}) {
  return encoder::primary::encode_at(loaded,
      encoder_input(sample, channel_ids, loaded.config, feature_observed), cutoffs);
}

} // namespace cuwacunu::integration
