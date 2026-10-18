// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once

#include "cuwacunu/data/memory_mapped_dataset.h"
#include "cuwacunu/encoder/primary_encoder.h"
#include "cuwacunu/integration/configurable_encoder_input.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuwacunu::training::market_data {

using Dataset = data::MemoryMappedEdgeDataset<data::kline_t>;
inline constexpr int64_t sampling_interval_ms = 60000;

// Complete one-minute windows, with the endpoint included. Selections depend
// only on the declared role/geometry/count, never observation values or labels.
inline std::vector<int64_t> uniform_anchors(int64_t begin, int64_t end,
    int64_t history, int64_t n, bool nonoverlap) {
  TORCH_CHECK(begin >= 0 && end >= begin && (end-begin)%sampling_interval_ms == 0 &&
                  history > 0 && n >= 2,
              "[market data] positive history/count and ordered minute-aligned role required");
  TORCH_CHECK(history-1 <= (std::numeric_limits<int64_t>::max()-begin)/sampling_interval_ms &&
                  (!nonoverlap || history <= std::numeric_limits<int64_t>::max()/sampling_interval_ms),
              "[market data] history/stride timestamp overflow");
  const auto stride = nonoverlap ? history*sampling_interval_ms : sampling_interval_ms;
  const auto first = begin+(history-1)*sampling_interval_ms;
  TORCH_CHECK(first <= end,"[market data] role cannot contain one complete history");
  const auto available = (end-first)/stride+1;
  TORCH_CHECK(n <= available,"[market data] insufficient complete role windows");
  std::vector<int64_t> result;
  result.reserve(static_cast<size_t>(n));
  for (int64_t j=0; j<n; ++j) {
    // The output is bounded by end. A wide intermediate prevents overflow in
    // j*(available-1) while retaining the original exact integer-floor policy.
    const auto position = static_cast<int64_t>(static_cast<__int128>(j)*(available-1)/(n-1));
    result.push_back(first+position*stride);
  }
  return result;
}

// Count truncates at the remaining rows. Offset is strict, and offset+count is
// never formed before bounds checking (including a caller's SIZE_MAX count).
inline std::vector<int64_t> slice(const std::vector<int64_t> &rows,
    size_t offset, size_t count) {
  TORCH_CHECK(offset <= rows.size(),"[market data] slice offset exceeds row count");
  const auto take = std::min(count,rows.size()-offset);
  return {rows.begin()+offset,rows.begin()+offset+take};
}

// Random TRAIN ordering and repeated anchors are legal; chronological panel
// spacing is a selection policy. Every returned past timestamp must lie inside
// this role. A short-window-legal anchor can still be illegal for a longer H.
inline encoder::primary::Input load_batch(Dataset &dataset,
    const std::vector<int64_t> &anchors, int64_t role_begin, int64_t role_end,
    const encoder::primary::Config &config) {
  encoder::primary::validate_config(config);
  TORCH_CHECK(!anchors.empty() && role_begin >= 0 && role_end >= role_begin &&
                  (role_end-role_begin)%sampling_interval_ms == 0 &&
                  config.sampling_interval == static_cast<double>(sampling_interval_ms),
              "[market data] nonempty batch and millisecond one-minute role/config required");
  TORCH_CHECK(config.history_length-1 <= std::numeric_limits<int64_t>::max()/sampling_interval_ms,
              "[market data] history timestamp overflow");
  TORCH_CHECK(dataset.max_future_length_ == 0,
              "[market data] history-information loading requires future horizon zero");
  const auto lookback = (config.history_length-1)*sampling_interval_ms;
  std::vector<data::sample_t> samples;
  samples.reserve(anchors.size());
  for (const auto anchor:anchors) {
    TORCH_CHECK(anchor >= role_begin && anchor <= role_end && anchor-role_begin >= lookback,
                "[market data] input history crosses its chronological role boundary");
    samples.push_back(dataset.get_by_anchor_key(anchor));
  }
  const auto batch = data::sample_t::collate_fn(samples);
  TORCH_CHECK(batch.past_keys.defined() && batch.past_keys.scalar_type() == torch::kInt64 &&
                  batch.past_keys.sizes() == torch::IntArrayRef({static_cast<int64_t>(anchors.size()),
                      config.channel_count,config.history_length}) &&
                  batch.past_keys.ge(role_begin).all().item<bool>() &&
                  batch.past_keys.le(role_end).all().item<bool>(),
              "[market data] actual mmap timestamps/geometry cross the role boundary");
  const auto endpoint = batch.past_keys.select(2,config.history_length-1);
  TORCH_CHECK(torch::equal(endpoint,torch::tensor(anchors,torch::kInt64).unsqueeze(1).expand_as(endpoint)),
              "[market data] actual mmap endpoints differ from selected anchors");
  const auto ids = torch::tensor(encoder::primary::implementation::resolved_channel_ids(config),torch::kInt64);
  return integration::configurable_encoder_input(batch,ids,config);
}

} // namespace cuwacunu::training::market_data
