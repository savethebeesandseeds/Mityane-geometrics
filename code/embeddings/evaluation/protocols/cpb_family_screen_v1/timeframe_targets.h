// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Targets for one instrument whose channels are time resolutions.
// This is not past_window_targets. Volatility, trend and activity come from
// the anchor channel only: the channel whose step equals sampling_interval,
// and each observed slot of that channel is one of its own bars.
// cross_resolution abstains. A packed slot index is not a shared clock.
// Do not feed these names to the published three-instrument decision rule.
#pragma once

#include "embedding/encoders/timeframe_channel.h"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::evaluation::family_screen {
namespace timeframe_channel = embedding::encoders::timeframe_channel;

enum class FeatureView { RawPrices, LogReturns };

struct TargetSpec {
  const char *name;
  const char *unit;
};

// Column order is part of the witness contract.
inline constexpr std::array<TargetSpec, 4> kTimeframeTargets{{
    {"anchor_volatility", "logreturn"},
    {"anchor_trend", "logreturn/tick"},
    {"anchor_activity", "log1p(base_volume)"},
    {"cross_resolution", "correlation"},
}};

struct TimeframeTargets {
  torch::Tensor values; // CPU float64 [N,4]. Invalid rows store zero.
  torch::Tensor valid;  // CPU bool [N,4].
  int64_t anchor_channel{0};
  int64_t anchor_id{0};
};

namespace timeframe_detail {
inline constexpr int64_t kClose = 3;
inline constexpr int64_t kVolume = 4;
inline constexpr int64_t kWidth = 9;
inline constexpr int64_t kMinimumCount = 8;

inline int64_t integer_milliseconds(double sampling_interval) {
  TORCH_CHECK(std::isfinite(sampling_interval) && sampling_interval >= 1. &&
                  sampling_interval <= 9007199254740991. &&
                  std::floor(sampling_interval) == sampling_interval,
      "[family screen timeframe] sampling interval must be an exact positive millisecond count");
  return static_cast<int64_t>(sampling_interval);
}

inline torch::Tensor shared_ids(const torch::Tensor &ids, int64_t channels) {
  TORCH_CHECK(ids.defined() && ids.device().is_cpu() && ids.scalar_type() == torch::kInt64,
      "[family screen timeframe] CPU int64 channel ids required");
  if (ids.dim() == 1) {
    TORCH_CHECK(ids.size(0) == channels, "[family screen timeframe] channel ids must cover every channel");
    return ids.contiguous();
  }
  TORCH_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) == channels,
      "[family screen timeframe] channel ids must be [C] or [N,C]");
  const auto first = ids.select(0, 0).contiguous();
  for (int64_t row = 1; row < ids.size(0); ++row) {
    TORCH_CHECK(torch::equal(ids.select(0, row), first),
        "[family screen timeframe] every row must use the same timeframe ids");
  }
  return first;
}

struct Moment {
  double mean{0};
  double deviation{0};
  bool ok{false};
};

inline Moment moment(const std::vector<double> &samples) {
  Moment out;
  if (static_cast<int64_t>(samples.size()) < kMinimumCount) return out;
  double sum = 0;
  for (const double sample : samples) sum += sample;
  out.mean = sum / static_cast<double>(samples.size());
  double square = 0;
  for (const double sample : samples) {
    const double delta = sample - out.mean;
    square += delta * delta;
  }
  out.deviation = std::sqrt(square / static_cast<double>(samples.size()));
  out.ok = std::isfinite(out.mean) && std::isfinite(out.deviation);
  return out;
}
} // namespace timeframe_detail

// values and observed are CPU [N,C,H,9]. H is at least 9. channel_ids is
// int64 [C] or [N,C] and names cuwacunu_torch kline intervals. Exactly one
// channel has a fixed step equal to sampling_interval; every other step is a
// positive multiple of it. A calendar month is refused. The step check does
// not place those channels on one clock.
//
// RawPrices reads positive closes and nonnegative volume, then takes adjacent
// anchor log returns and log1p volume. LogReturns reads the stored anchor
// close as the per-bar return and the stored volume as log1p. A negative
// stored return is legal in that view and is not logged again. Each observed
// anchor slot is one bar of the anchor channel.
//
// cross_resolution stays invalid. Slot i of one channel is not the same
// instant as slot i of another, so those slots are not paired.
[[nodiscard]] inline TimeframeTargets timeframe_window_targets(
    const torch::Tensor &values, const torch::Tensor &observed, const torch::Tensor &channel_ids,
    double sampling_interval, FeatureView view) {
  using namespace timeframe_detail;
  TORCH_CHECK(values.defined() && values.device().is_cpu() && values.is_floating_point() && values.dim() == 4 &&
                  values.size(0) > 0 && values.size(1) > 0 && values.size(2) >= 9 && values.size(3) == kWidth,
      "[family screen timeframe] CPU [N,C,H>=9,9] values required");
  TORCH_CHECK(observed.defined() && observed.device().is_cpu() && observed.scalar_type() == torch::kBool &&
                  observed.sizes() == values.sizes(),
      "[family screen timeframe] matching CPU bool mask required");
  TORCH_CHECK(torch::isfinite(values.masked_select(observed)).all().item<bool>(),
      "[family screen timeframe] observed values must be finite");
  const auto rows = values.size(0), channels = values.size(1), history = values.size(2);
  const auto ids = shared_ids(channel_ids, channels);
  std::vector<int64_t> id_list(static_cast<std::size_t>(channels));
  const auto id_pointer = ids.const_data_ptr<int64_t>();
  for (int64_t channel = 0; channel < channels; ++channel) id_list[static_cast<std::size_t>(channel)] = id_pointer[channel];
  std::string channel_error;
  TORCH_CHECK(timeframe_channel::timeframe_ids_ok(id_list, &channel_error),
      "[family screen timeframe] ", channel_error);
  const auto anchor_step = integer_milliseconds(sampling_interval);
  int64_t anchor = -1;
  for (int64_t channel = 0; channel < channels; ++channel) {
    const auto *entry = timeframe_channel::find(id_list[static_cast<std::size_t>(channel)]);
    TORCH_CHECK(entry != nullptr && entry->step_ms > 0,
        "[family screen timeframe] each channel needs a fixed millisecond step");
    TORCH_CHECK(entry->step_ms % anchor_step == 0,
        "[family screen timeframe] every channel step must be a whole number of anchor steps");
    if (entry->step_ms == anchor_step) {
      TORCH_CHECK(anchor < 0, "[family screen timeframe] only one channel may use the anchor step");
      anchor = channel;
    }
  }
  TORCH_CHECK(anchor >= 0, "[family screen timeframe] one channel step must equal the sampling interval");

  const auto stored = torch::where(observed, values.to(torch::kFloat64), torch::zeros_like(values, torch::kFloat64)).contiguous();
  const auto mask = observed.contiguous();
  const auto at = stored.accessor<double, 4>();
  const auto seen = mask.accessor<bool, 4>();
  auto output = torch::zeros({rows, 4}, torch::kFloat64);
  auto eligible = torch::zeros({rows, 4}, torch::kBool);
  auto out = output.accessor<double, 2>();
  auto ok = eligible.accessor<bool, 2>();

  for (int64_t row = 0; row < rows; ++row) {
    std::vector<char> anchor_ok(static_cast<std::size_t>(history), 0);
    std::vector<double> anchor_return(static_cast<std::size_t>(history), 0);
    if (view == FeatureView::LogReturns) {
      for (int64_t tick = 0; tick < history; ++tick) {
        if (!seen[row][anchor][tick][kClose]) continue;
        anchor_ok[static_cast<std::size_t>(tick)] = 1;
        anchor_return[static_cast<std::size_t>(tick)] = at[row][anchor][tick][kClose];
      }
    } else {
      for (int64_t tick = 1; tick < history; ++tick) {
        const bool left = seen[row][anchor][tick - 1][kClose] && at[row][anchor][tick - 1][kClose] > 0;
        const bool right = seen[row][anchor][tick][kClose] && at[row][anchor][tick][kClose] > 0;
        if (!left || !right) continue;
        anchor_ok[static_cast<std::size_t>(tick)] = 1;
        anchor_return[static_cast<std::size_t>(tick)] =
            std::log(at[row][anchor][tick][kClose]) - std::log(at[row][anchor][tick - 1][kClose]);
      }
    }
    std::vector<double> returns;
    returns.reserve(static_cast<std::size_t>(history));
    for (int64_t tick = 0; tick < history; ++tick) {
      if (anchor_ok[static_cast<std::size_t>(tick)]) returns.push_back(anchor_return[static_cast<std::size_t>(tick)]);
    }
    const auto movement = moment(returns);
    if (movement.ok) {
      out[row][0] = movement.deviation;
      out[row][1] = movement.mean;
      ok[row][0] = true;
      ok[row][1] = true;
    }

    std::vector<double> activity;
    for (int64_t tick = 0; tick < history; ++tick) {
      if (!seen[row][anchor][tick][kVolume]) continue;
      const double volume = at[row][anchor][tick][kVolume];
      if (!(volume >= 0)) continue;
      activity.push_back(view == FeatureView::LogReturns ? volume : std::log1p(volume));
    }
    const auto activity_moment = moment(activity);
    if (activity_moment.ok) {
      out[row][2] = activity_moment.mean;
      ok[row][2] = true;
    }
  }
  TORCH_CHECK(torch::isfinite(output).all().item<bool>(), "[family screen timeframe] target arithmetic overflow");
  return {output, eligible, anchor, id_list[static_cast<std::size_t>(anchor)]};
}

} // namespace embedding::evaluation::family_screen
