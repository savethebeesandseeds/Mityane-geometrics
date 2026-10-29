// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Places a numeric-series edge window onto this encoder's shared anchor grid.
// The loader keeps one length and one past-key row per resolution. This
// encoder keeps one history length and one sampling interval. A value at an
// anchor tick is the latest valid bar of that channel whose key is at or
// before the tick. Coarser bars are held. Nothing is interpolated.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"
#include "dataloader.h"
#include "kline_feature_registry.h"

#include <array>
#include <cmath>
#include <cstdint>
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

inline int64_t integer_milliseconds(double sampling_interval) {
  TORCH_CHECK(std::isfinite(sampling_interval) && sampling_interval >= 1. &&
                  sampling_interval <= 9007199254740991. &&
                  std::floor(sampling_interval) == sampling_interval,
      "[cpb-family numeric-series] SAMPLING_INTERVAL must be an exact positive millisecond count");
  return static_cast<int64_t>(sampling_interval);
}

inline int64_t channel_step(int64_t channel_id) {
  const auto *entry = timeframe_channel::find(channel_id);
  TORCH_CHECK(entry != nullptr && entry->step_ms > 0,
      "[cpb-family numeric-series] each channel id must be a kline interval with a fixed millisecond step");
  return entry->step_ms;
}

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

// Latest valid bar at or before the tick. Returns bars.size() when none exists.
inline std::size_t latest_at_or_before(const std::vector<Bar> &bars, int64_t tick) {
  std::size_t chosen = bars.size();
  for (std::size_t i = 0; i < bars.size(); ++i) {
    if (bars[i].key <= tick) chosen = i;
    else break;
  }
  return chosen;
}
} // namespace numeric_series_detail

// samples[b] is one instrument. channel_ids[c] is the torch ordinal of
// samples[b]'s channel c, in that order, and it must equal the configuration.
// The anchor channel is the one whose interval equals SAMPLING_INTERVAL.
// Its last HISTORY_LENGTH valid keys are the shared clock. A log-return view
// uses the previous valid bar of the same channel and does not write a cache.
// An edge sample that is already normalized cannot take that view again.
[[nodiscard]] inline NumericSeriesHistory history_from_edge_samples(
    const Config &config, const std::vector<mityane::data::edge_sample_t> &samples,
    const std::vector<int64_t> &channel_ids, bool log_returns) {
  validate_config(config);
  const auto &level = config.level;
  TORCH_CHECK(level.input_width == numeric_series_detail::kFeatures,
      "[cpb-family numeric-series] INPUT_WIDTH must be the nine kline features");
  TORCH_CHECK(!samples.empty() && channel_ids.size() == static_cast<std::size_t>(level.channel_count) &&
                  channel_ids == implementation::resolved_channel_ids(level),
      "[cpb-family numeric-series] channel ids must match the configuration, one sample per instrument");
  const auto anchor_step = numeric_series_detail::integer_milliseconds(level.sampling_interval);
  std::vector<int64_t> steps;
  steps.reserve(channel_ids.size());
  int64_t anchor_channel = -1;
  for (std::size_t channel = 0; channel < channel_ids.size(); ++channel) {
    const auto step = numeric_series_detail::channel_step(channel_ids[channel]);
    TORCH_CHECK(step % anchor_step == 0,
        "[cpb-family numeric-series] every channel interval must be a whole number of anchor steps");
    if (step == anchor_step) anchor_channel = static_cast<int64_t>(channel);
    steps.push_back(step);
  }
  TORCH_CHECK(anchor_channel >= 0,
      "[cpb-family numeric-series] one channel interval must equal SAMPLING_INTERVAL");

  const auto batch = static_cast<int64_t>(samples.size());
  const auto channels = level.channel_count;
  const auto history = level.history_length;
  const auto features = level.input_width;
  auto values = torch::zeros({batch, channels, history, features}, torch::kFloat64);
  auto observed = torch::zeros({batch, channels, history, features}, torch::kBool);
  auto timestamps = torch::empty({batch, history}, torch::kFloat64);
  auto cutoffs = torch::empty({batch}, torch::kFloat64);
  auto value_at = values.accessor<double, 4>();
  auto observed_at = observed.accessor<bool, 4>();
  auto time_at = timestamps.accessor<double, 2>();
  auto cutoff_at = cutoffs.accessor<double, 1>();

  for (int64_t row = 0; row < batch; ++row) {
    const auto &sample = samples[static_cast<std::size_t>(row)];
    TORCH_CHECK(sample.features.defined() && sample.features.size(0) == channels,
        "[cpb-family numeric-series] each sample must have one row per configured channel");
    TORCH_CHECK(!(sample.normalized && log_returns),
        "[cpb-family numeric-series] a normalized edge sample is already a view; a second log-return view is refused");
    std::vector<std::vector<numeric_series_detail::Bar>> bars(static_cast<std::size_t>(channels));
    for (int64_t channel = 0; channel < channels; ++channel)
      bars[static_cast<std::size_t>(channel)] =
          numeric_series_detail::valid_bars(sample, channel, steps[static_cast<std::size_t>(channel)]);
    const auto &anchor_bars = bars[static_cast<std::size_t>(anchor_channel)];
    TORCH_CHECK(static_cast<int64_t>(anchor_bars.size()) >= history,
        "[cpb-family numeric-series] the anchor channel needs HISTORY_LENGTH valid bars");
    const auto begin = anchor_bars.size() - static_cast<std::size_t>(history);
    for (int64_t tick = 0; tick < history; ++tick) {
      const auto key = anchor_bars[begin + static_cast<std::size_t>(tick)].key;
      const auto expected = static_cast<__int128>(anchor_bars[begin].key) +
                            static_cast<__int128>(tick) * anchor_step;
      TORCH_CHECK(expected == key && expected <= 9007199254740991LL,
          "[cpb-family numeric-series] the anchor channel's last HISTORY_LENGTH keys must be uniform");
      time_at[row][tick] = static_cast<double>(key);
    }
    cutoff_at[row] = time_at[row][history - 1];
    for (int64_t channel = 0; channel < channels; ++channel) {
      const auto &channel_bars = bars[static_cast<std::size_t>(channel)];
      for (int64_t tick = 0; tick < history; ++tick) {
        const auto key = anchor_bars[begin + static_cast<std::size_t>(tick)].key;
        const auto chosen = numeric_series_detail::latest_at_or_before(channel_bars, key);
        if (chosen == channel_bars.size()) continue;
        std::array<double, numeric_series_detail::kFeatures> feature{};
        if (log_returns) {
          if (chosen == 0) continue;
          feature = numeric_series_detail::log_return_view(channel_bars[chosen], channel_bars[chosen - 1]);
        } else {
          feature = channel_bars[chosen].value;
        }
        for (int64_t coord = 0; coord < features; ++coord) {
          value_at[row][channel][tick][coord] = feature[static_cast<std::size_t>(coord)];
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
