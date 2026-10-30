// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Packs each channel's own bars into HISTORY_LENGTH slots.
// HISTORY_LENGTH is a packing width, the same number of slots on every channel,
// not a shared duration. Each channel contributes that resolution's own bars,
// time-ordered and right-aligned. A shorter channel leaves the left slots empty.
// An empty slot is not a bar: its timestamp is NaN and its mask is false.
// Index i is the i-th packed bar of that channel. It is not the same timestamp
// as index i of another channel. Nothing is interpolated and nothing is held.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"
#include "dataloader.h"
#include "kline_feature_registry.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace embedding::encoders::causal_patch_bottleneck_mae::family {

struct NumericSeriesHistory {
  RawHistory history;
  torch::Tensor cutoffs;
};

namespace numeric_series_detail {
inline constexpr int64_t kFeatures = mityane::data::kKlineFeatureWidth;

struct Bar {
  int64_t key{0};
  std::array<double, kFeatures> value{};
};

inline int64_t channel_step(int64_t channel_id) {
  const auto *entry = timeframe_channel::find(channel_id);
  TORCH_CHECK(entry != nullptr && entry->step_ms > 0,
      "[cpb-family numeric-series] each channel id must be a kline interval with a fixed millisecond step");
  return entry->step_ms;
}

// Latest-key prepass only. Packing does not use this compacted list.
inline std::vector<Bar> valid_bars(const mityane::data::edge_sample_t &sample, int64_t channel,
                                   int64_t step) {
  TORCH_CHECK(sample.features.defined() && sample.mask.defined() && sample.past_keys.defined(),
      "[cpb-family numeric-series] features, mask and past keys are required");
  TORCH_CHECK(sample.features.dim() == 3 && sample.mask.dim() == 2 && sample.past_keys.dim() == 2,
      "[cpb-family numeric-series] one unbatched edge sample is [C,H,F], [C,H] and [C,H]");
  TORCH_CHECK(sample.features.size(0) == sample.mask.size(0) &&
                  sample.features.size(0) == sample.past_keys.size(0) &&
                  sample.features.size(1) == sample.mask.size(1) &&
                  sample.features.size(1) == sample.past_keys.size(1),
      "[cpb-family numeric-series] feature, mask and past-key lengths must match");
  TORCH_CHECK(sample.features.size(2) == kFeatures,
      "[cpb-family numeric-series] the kline window has nine features");
  TORCH_CHECK(sample.features.is_cpu() && sample.mask.is_cpu() && sample.past_keys.is_cpu(),
      "[cpb-family numeric-series] CPU edge sample required");
  TORCH_CHECK(sample.features.scalar_type() == torch::kFloat64 ||
                  sample.features.scalar_type() == torch::kFloat32,
      "[cpb-family numeric-series] Float64 or Float32 features required");
  TORCH_CHECK(sample.mask.scalar_type() == torch::kBool && sample.past_keys.scalar_type() == torch::kInt64,
      "[cpb-family numeric-series] bool mask and Int64 past keys required");
  const auto features = sample.features.to(torch::kFloat64).contiguous();
  const auto mask = sample.mask.contiguous();
  const auto keys = sample.past_keys.contiguous();
  const auto values = features.accessor<double, 3>();
  const auto valid = mask.accessor<bool, 2>();
  const auto stamps = keys.accessor<int64_t, 2>();
  std::vector<Bar> bars;
  for (int64_t row = 0; row < features.size(1); ++row) {
    if (!valid[channel][row]) continue;
    Bar bar;
    bar.key = stamps[channel][row];
    TORCH_CHECK(bars.empty() || bar.key > bars.back().key,
        "[cpb-family numeric-series] valid past keys must increase inside a channel");
    TORCH_CHECK(bar.key >= 0 && (bars.empty() || (bar.key - bars.front().key) % step == 0),
        "[cpb-family numeric-series] a valid past key must lie on that channel's interval");
    for (int64_t feature = 0; feature < kFeatures; ++feature) {
      bar.value[static_cast<std::size_t>(feature)] = values[channel][row][feature];
      TORCH_CHECK(std::isfinite(bar.value[static_cast<std::size_t>(feature)]),
          "[cpb-family numeric-series] a valid kline feature must be finite");
    }
    bars.push_back(bar);
  }
  return bars;
}

inline mityane::data::kline_cache_t record_from(const Bar &bar) {
  mityane::data::kline_cache_t record{};
  record.open_time = bar.key > 0 ? bar.key - 1 : 0;
  record.close_time = bar.key;
  record.open_price = bar.value[0];
  record.high_price = bar.value[1];
  record.low_price = bar.value[2];
  record.close_price = bar.value[3];
  record.volume = bar.value[4];
  record.quote_asset_volume = bar.value[5];
  record.number_of_trades = bar.value[6];
  record.taker_buy_base_volume = bar.value[7];
  record.taker_buy_quote_volume = bar.value[8];
  return record;
}

inline std::array<double, kFeatures> log_return_view(const Bar &current, const Bar &previous) {
  const auto now = record_from(current);
  const auto before = record_from(previous);
  const auto normalized = mityane::data::kline_cache_t::normalize_log_returns(now, &before);
  const auto features = normalized.tensor_features();
  TORCH_CHECK(static_cast<int64_t>(features.size()) == kFeatures,
      "[cpb-family numeric-series] log-return view keeps nine features");
  std::array<double, kFeatures> out{};
  for (int64_t feature = 0; feature < kFeatures; ++feature) {
    out[static_cast<std::size_t>(feature)] = features[static_cast<std::size_t>(feature)];
    TORCH_CHECK(std::isfinite(out[static_cast<std::size_t>(feature)]),
        "[cpb-family numeric-series] log-return feature must be finite");
  }
  return out;
}

struct PackedSlot {
  int64_t key{0};
  bool observed{false};
  std::array<double, kFeatures> value{};
};

// Walk every edge-sample row. Do not compact. A missing bar is mask false with
// its own past key. A false mask whose key is a later on-grid close keeps that
// finite time, a false mask, and value 0. A false mask that is not a later
// on-grid close is a leading pad and stays a NaN time. A NaN time is a leading
// prefix only. The log return uses the immediately previous row. When that row
// is absent or its mask is false, this row is masked and its raw close is kept
// only as the reference for the next row. The return does not reach across a
// hole. Open, high, and low use that same previous close. Volume, quote volume,
// trades, and both taker volumes are log1p of the current row, and they are
// masked with the row. A zero, negative, or non-finite price is not a missing
// bar; kline_cache_t::normalize_log_returns keeps the 1e-12 clamp.
inline std::vector<PackedSlot> packed_slots(const mityane::data::edge_sample_t &sample, int64_t channel,
                                           int64_t step, int64_t cutoff, bool log_returns) {
  const auto features = sample.features.to(torch::kFloat64).contiguous();
  const auto mask = sample.mask.contiguous();
  const auto keys = sample.past_keys.contiguous();
  TORCH_CHECK(features.dim() == 3 && mask.dim() == 2 && keys.dim() == 2 && features.size(0) > channel &&
                  mask.sizes() == keys.sizes() && mask.size(0) == features.size(0) &&
                  mask.size(1) == features.size(1) && features.size(2) == kFeatures,
      "[cpb-family numeric-series] channel rows are [C,H,F], [C,H] and [C,H]");
  const auto values = features.accessor<double, 3>();
  const auto valid = mask.accessor<bool, 2>();
  const auto stamps = keys.accessor<int64_t, 2>();
  std::vector<PackedSlot> slots;
  bool grid = false;
  int64_t anchor = 0;
  int64_t last_key = 0;
  bool have_reference = false;
  Bar reference{};
  for (int64_t row = 0; row < features.size(1); ++row) {
    const auto key = stamps[channel][row];
    const bool row_valid = valid[channel][row];
    const bool later_on_grid = grid && key >= 0 && key > last_key && (key - anchor) % step == 0;
    if (!later_on_grid) {
      if (!row_valid && !grid) continue;
      TORCH_CHECK(!grid && row_valid && key >= 0,
          "[cpb-family numeric-series] an interior row must be a later on-grid close; a NaN time is a leading prefix only");
    }
    if (key > cutoff) break;
    if (!grid) anchor = key;
    grid = true;
    last_key = key;
    PackedSlot slot;
    slot.key = key;
    if (!row_valid) {
      have_reference = false;
      slots.push_back(slot);
      continue;
    }
    Bar current;
    current.key = key;
    for (int64_t feature = 0; feature < kFeatures; ++feature) {
      current.value[static_cast<std::size_t>(feature)] = values[channel][row][feature];
      if (!log_returns) {
        TORCH_CHECK(std::isfinite(current.value[static_cast<std::size_t>(feature)]),
            "[cpb-family numeric-series] a valid raw kline feature must be finite");
      }
    }
    if (!log_returns) {
      slot.observed = true;
      slot.value = current.value;
    } else if (!have_reference) {
      have_reference = true;
      reference = current;
    } else {
      slot.observed = true;
      slot.value = log_return_view(current, reference);
      reference = current;
    }
    slots.push_back(slot);
  }
  return slots;
}
} // namespace numeric_series_detail

// samples[b] is one instrument. channel_ids[c] is the torch ordinal of
// samples[b]'s channel c, in that order, and it must equal the configuration.
// An omitted bar_counts uses the settings list on config. An explicit empty
// list is refused. settings() stores HISTORY_LENGTH on every channel when
// CHANNEL_BAR_COUNTS itself is absent. Each channel keeps the bars whose key
// is at or before the cutoff, takes the last count of the resulting slots, and
// right-aligns them. Left slots stay masked and their times are NaN.
// An omitted cutoff is the latest valid key in that sample. A log-return view
// walks every row in order and does not write a cache. It uses the immediately
// previous row of the same channel. A missing row keeps its own finite close
// time, a false mask, and value 0. The next row is masked, and that row's raw
// close is the reference for the following return. The return does not reach
// an older close. A false mask that is not a later on-grid close is a leading
// NaN pad. The first bar stays in its slot and stays masked. An edge sample
// that is already normalized cannot take the view again.
[[nodiscard]] inline NumericSeriesHistory history_from_edge_samples(
    const Config &config, const std::vector<mityane::data::edge_sample_t> &samples,
    const std::vector<int64_t> &channel_ids, bool log_returns,
    const std::optional<std::vector<int64_t>> &bar_counts = std::nullopt,
    const std::vector<int64_t> &cutoffs_ms = {}) {
  validate_config(config);
  const auto &level = config.level;
  TORCH_CHECK(level.input_width == numeric_series_detail::kFeatures,
      "[cpb-family numeric-series] INPUT_WIDTH must be the nine kline features");
  TORCH_CHECK(!samples.empty() && channel_ids.size() == static_cast<std::size_t>(level.channel_count) &&
                  channel_ids == implementation::resolved_channel_ids(level),
      "[cpb-family numeric-series] channel ids must match the configuration, one sample per instrument");
  const auto batch = static_cast<int64_t>(samples.size());
  const auto channels = level.channel_count;
  const auto history = level.history_length;
  const auto features = level.input_width;
  const auto counts = resolve_bar_counts(config, bar_counts);
  TORCH_CHECK(static_cast<int64_t>(counts.size()) == channels,
      "[cpb-family numeric-series] CHANNEL_BAR_COUNTS must list one count per channel");
  for (const auto count : counts) {
    TORCH_CHECK(count >= 1 && count <= history,
        "[cpb-family numeric-series] each bar count must be from 1 through HISTORY_LENGTH");
  }
  TORCH_CHECK(cutoffs_ms.empty() || static_cast<int64_t>(cutoffs_ms.size()) == batch,
      "[cpb-family numeric-series] an explicit cutoff is one decision time per instrument");
  std::vector<int64_t> steps;
  steps.reserve(channel_ids.size());
  for (const auto channel_id : channel_ids) steps.push_back(numeric_series_detail::channel_step(channel_id));

  auto values = torch::zeros({batch, channels, history, features}, torch::kFloat64);
  auto observed = torch::zeros({batch, channels, history, features}, torch::kBool);
  auto timestamps = torch::full({batch, channels, history}, std::numeric_limits<double>::quiet_NaN(), torch::kFloat64);
  auto cutoffs = torch::empty({batch}, torch::kFloat64);
  auto value_at = values.accessor<double, 4>();
  auto observed_at = observed.accessor<bool, 4>();
  auto time_at = timestamps.accessor<double, 3>();
  auto cutoff_at = cutoffs.accessor<double, 1>();

  for (int64_t row = 0; row < batch; ++row) {
    const auto &sample = samples[static_cast<std::size_t>(row)];
    TORCH_CHECK(sample.features.defined() && sample.features.size(0) == channels,
        "[cpb-family numeric-series] each sample must have one row per configured channel");
    TORCH_CHECK(!(sample.normalized && log_returns),
        "[cpb-family numeric-series] a normalized edge sample is already a view; a second log-return view is refused");
    int64_t latest = 0;
    bool any = false;
    for (int64_t channel = 0; channel < channels; ++channel) {
      for (const auto &bar : numeric_series_detail::valid_bars(sample, channel, steps[static_cast<std::size_t>(channel)])) {
        if (!any || bar.key > latest) latest = bar.key;
        any = true;
      }
    }
    const auto cutoff = cutoffs_ms.empty() ? latest : cutoffs_ms[static_cast<std::size_t>(row)];
    TORCH_CHECK(cutoffs_ms.empty() ? any : true,
        "[cpb-family numeric-series] an omitted cutoff needs one valid bar");
    TORCH_CHECK(cutoff >= 0 && cutoff <= 9007199254740991LL,
        "[cpb-family numeric-series] a cutoff must be an exact nonnegative millisecond count");
    cutoff_at[row] = static_cast<double>(cutoff);
    for (int64_t channel = 0; channel < channels; ++channel) {
      auto slots = numeric_series_detail::packed_slots(
          sample, channel, steps[static_cast<std::size_t>(channel)], cutoff, log_returns);
      const auto keep = static_cast<std::size_t>(counts[static_cast<std::size_t>(channel)]);
      if (slots.size() > keep) slots.erase(slots.begin(), slots.end() - static_cast<std::ptrdiff_t>(keep));
      const auto pad = history - static_cast<int64_t>(slots.size());
      for (std::size_t i = 0; i < slots.size(); ++i) {
        const auto tick = pad + static_cast<int64_t>(i);
        const auto &slot = slots[i];
        TORCH_CHECK(slot.key >= 0 && slot.key <= 9007199254740991LL,
            "[cpb-family numeric-series] a packed bar key must be an exact millisecond count");
        time_at[row][channel][tick] = static_cast<double>(slot.key);
        if (!slot.observed) continue;
        for (int64_t coord = 0; coord < features; ++coord) {
          value_at[row][channel][tick][coord] = slot.value[static_cast<std::size_t>(coord)];
          observed_at[row][channel][tick][coord] = true;
        }
      }
    }
  }
  return {RawHistory{values, observed, torch::tensor(channel_ids, torch::kInt64), timestamps,
                     level.sampling_interval},
          cutoffs};
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::family
