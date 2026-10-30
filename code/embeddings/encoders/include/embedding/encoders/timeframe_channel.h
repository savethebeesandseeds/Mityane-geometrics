// SPDX-License-Identifier: MIT
// Channel axis for the causal and raw patch-bottleneck encoders.
// cuwacunu_torch interval_type_e is the standard. Ordinals and keys match
// src/include/ujcamei/source/registry/types/enums.h in declaration order.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::encoders::timeframe_channel {

// One window is one instrument. The channel axis holds time resolutions of
// that instrument: minutes, hours, weeks, and the other kline intervals below.
// Different instruments are different batch rows, never different channels.
// A channel id is the torch ordinal of one kline interval. Utility ordinals
// 0, 1 and 2 (constant, sine, triangular) are not channels.
//
// The set {0,1,2} is the withdrawn instrument binding stored by historical
// checkpoints (three instruments on one clock). It is not this axis. Recorded
// scores on that binding stay scores of that binding.
//
// HISTORY_LENGTH is a packing width, not a shared duration. Each channel's own
// bars are right-aligned into that many slots. Index i is that channel's i-th
// packed bar. Timestamps are [B,C,N]: NaN on a pad, finite and strictly
// increasing on that channel's bars. sampling_interval is metadata. It is not
// a grid every channel is forced onto.
inline constexpr const char *kContract =
    "One window is one instrument. Channels are cuwacunu_torch kline intervals.";

struct Entry {
  int64_t id;
  const char *key;
  int64_t step_ms; // 0 means a calendar month, which has no fixed millisecond step.
};

// step_ms is the duration of the key. The torch enum does not store it.
inline constexpr Entry kKline[] = {
    {3, "1s", 1000},       {4, "1m", 60000},      {5, "3m", 180000},
    {6, "5m", 300000},     {7, "15m", 900000},    {8, "30m", 1800000},
    {9, "1h", 3600000},    {10, "2h", 7200000},   {11, "4h", 14400000},
    {12, "6h", 21600000},  {13, "8h", 28800000},  {14, "12h", 43200000},
    {15, "1d", 86400000},  {16, "3d", 259200000}, {17, "1w", 604800000},
    {18, "1M", 0},
};

inline constexpr std::size_t kKlineCount = sizeof(kKline) / sizeof(kKline[0]);

inline const Entry *find(int64_t id) {
  for (const auto &entry : kKline)
    if (entry.id == id) return &entry;
  return nullptr;
}

inline bool is_kline_timeframe(int64_t id) { return find(id) != nullptr; }

// True only for a permutation of {0,1,2}. Any other id list is not this binding.
inline bool is_withdrawn_instrument_binding(const int64_t *ids, std::size_t count) {
  if (count != 3 || ids == nullptr) return false;
  bool seen[3] = {};
  for (std::size_t i = 0; i < 3; ++i) {
    if (ids[i] < 0 || ids[i] > 2 || seen[ids[i]]) return false;
    seen[ids[i]] = true;
  }
  return true;
}

inline bool timeframe_ids_ok(const int64_t *ids, std::size_t count, std::string *error) {
  auto fail = [&](const std::string &message) {
    if (error != nullptr) *error = std::string(kContract) + " " + message;
    return false;
  };
  if (ids == nullptr || count == 0)
    return fail("CHANNEL_IDS must name at least one kline interval.");
  for (std::size_t i = 0; i < count; ++i) {
    if (!is_kline_timeframe(ids[i])) {
      if (is_withdrawn_instrument_binding(ids, count))
        return fail("Ids {0,1,2} are the withdrawn instrument binding, not time resolutions.");
      return fail("Channel id " + std::to_string(ids[i]) +
                  " is not a cuwacunu_torch kline interval ordinal.");
    }
    for (std::size_t j = 0; j < i; ++j)
      if (ids[j] == ids[i])
        return fail("Duplicate timeframe id " + std::to_string(ids[i]) + ".");
  }
  return true;
}

inline bool timeframe_ids_ok(const std::vector<int64_t> &ids, std::string *error) {
  return timeframe_ids_ok(ids.data(), ids.size(), error);
}

// New settings require timeframe ids. A recorded checkpoint may still carry
// the withdrawn instrument binding so that checkpoint can be opened.
inline bool admit_configured_ids(const int64_t *ids, std::size_t count, std::string *error) {
  if (is_withdrawn_instrument_binding(ids, count)) return true;
  return timeframe_ids_ok(ids, count, error);
}

inline bool admit_configured_ids(const std::vector<int64_t> &ids, std::string *error) {
  return admit_configured_ids(ids.data(), ids.size(), error);
}

} // namespace embedding::encoders::timeframe_channel
