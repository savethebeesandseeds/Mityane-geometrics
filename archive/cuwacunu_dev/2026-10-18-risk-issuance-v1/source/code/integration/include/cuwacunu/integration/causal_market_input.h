// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "cuwacunu/data/memory_mapped_dataset.h"
#include "cuwacunu/integration/causal_encoder_input.h"
#include <array>

namespace cuwacunu::integration {

// Inference opens existing raw caches read-only. A CSV name locates its fresh
// published cache; this entry point never converts or rebuilds source files.
inline data::sample_t load_market_history(
    const std::array<std::filesystem::path, 3> &sources, int64_t cutoff,
    const encoder::primary::Config &config) {
  encoder::primary::validate_config(config);
  TORCH_CHECK(cutoff >= 0 && cutoff <= 9007199254740992LL,
      "[causal market input] an explicit exactly representable close-time cutoff is required");
  std::vector<torch::Tensor> values, masks, keys;
  std::set<std::filesystem::path> seen;
  for (const auto &source : sources) {
    auto cache = source;
    if (source.extension() == ".csv") {
      cache = data::raw_binary_for_csv(source.string());
      TORCH_CHECK(std::filesystem::is_regular_file(source) &&
          data::cache_file_is_strictly_newer(cache.string(), source.string()),
          "[causal market input] CSV source needs an existing fresh raw cache: ", source.string());
    }
    TORCH_CHECK(std::filesystem::is_regular_file(cache) &&
        seen.insert(std::filesystem::weakly_canonical(cache)).second,
        "[causal market input] three distinct existing raw CACHE88 sources are required");
    data::MemoryMappedDataset<data::kline_t> dataset(
        cache.string(), static_cast<size_t>(config.level.history_length), 0, torch::kFloat64);
    TORCH_CHECK(!dataset.is_normalized(), "[causal market input] normalized caches are forbidden");
    const auto sample = dataset.get_edge_sample_at_anchor_key(
        cutoff, static_cast<size_t>(config.level.history_length), 0);
    TORCH_CHECK(sample.past_keys.defined() &&
        sample.past_keys[-1].item<int64_t>() == cutoff,
        "[causal market input] cutoff must equal an actual candle close, not a nearest earlier row");
    values.push_back(sample.features);
    masks.push_back(sample.mask);
    keys.push_back(sample.past_keys);
  }
  data::sample_t result;
  result.features = torch::stack(values).unsqueeze(0);
  result.mask = torch::stack(masks).unsqueeze(0);
  result.past_keys = torch::stack(keys).unsqueeze(0);
  return result;
}

} // namespace cuwacunu::integration
