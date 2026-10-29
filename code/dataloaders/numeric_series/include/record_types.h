// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include "support.h"
namespace mityane::data {
using i64 = std::int64_t;
using ms_t = std::int64_t;
struct normalization_warning_summary_t {
  std::size_t log_return_value_nonpositive{0};
  std::size_t log_return_value_nonfinite{0};
  std::size_t log_return_reference_nonpositive{0};
  std::size_t log_return_reference_nonfinite{0};
  std::size_t log1p_below_floor{0};
  std::size_t log1p_nonfinite{0};

  [[nodiscard]] bool any() const {
    return log_return_value_nonpositive > 0 || log_return_value_nonfinite > 0 ||
           log_return_reference_nonpositive > 0 ||
           log_return_reference_nonfinite > 0 || log1p_below_floor > 0 ||
           log1p_nonfinite > 0;
  }
};

void begin_normalization_warning_scope(std::string_view filename,
                                       std::string_view policy_name);
normalization_warning_summary_t end_normalization_warning_scope();

/* Ensure binary conversion compatibility and memory efficiency */
#pragma pack(push, 1)
struct trade_t {
  using key_type_t = ms_t;
  /* Methods */
  static constexpr std::size_t key_offset() { return offsetof(trade_t, time); }
  key_type_t key_value();
  static trade_t null_instance(key_type_t key_value = INT64_MIN);
  static trade_t from_binary(const char *data);
  static trade_t from_csv(const std::string &line, char delimiter = ',',
                          size_t line_number = 0);
  std::vector<double> tensor_features() const;
  void to_csv(std::ostream &os, char delimiter = ',') const;
  bool is_valid() const;
  /* Values */
  i64 id;
  double price;
  double qty;
  double quoteQty;
  ms_t time;
  bool isBuyerMaker;
  bool isBestMatch;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct kline_t {
  using key_type_t = ms_t;
  /* Methods */
  static constexpr std::size_t key_offset() {
    return offsetof(kline_t, close_time);
  }
  key_type_t key_value();
  static kline_t from_binary(const char *data);
  static kline_t null_instance(key_type_t key_value = INT64_MIN);
  static kline_t from_csv(const std::string &line, char delimiter = ',',
                          size_t line_number = 0);
  std::vector<double> tensor_features() const;
  void to_csv(std::ostream &os, char delimiter = ',') const;
  bool is_valid() const;
  /* Values */
  ms_t open_time;
  double open_price;
  double high_price;
  double low_price;
  double close_price;
  double volume;
  ms_t close_time;
  double quote_asset_volume;
  int32_t number_of_trades;
  double taker_buy_base_volume;
  double taker_buy_quote_volume;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct basic_t {
  using key_type_t = double;
  /* Methods */
  static constexpr std::size_t key_offset() { return offsetof(basic_t, time); }
  key_type_t key_value();
  static basic_t from_binary(const char *data);
  static basic_t
  null_instance(key_type_t key_value = std::numeric_limits<double>::min());
  static basic_t from_csv(const std::string &line, char delimiter = ',',
                          size_t line_number = 0);
  std::vector<double> tensor_features() const;
  void to_csv(std::ostream &os, char delimiter = ',') const;
  bool is_valid() const;
  /* Values */
  double time;
  double value;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct trade_cache_t {
  using key_type_t = ms_t;
  using raw_type_t = trade_t;
  /* Methods */
  static constexpr std::size_t key_offset() {
    return offsetof(trade_cache_t, time);
  }
  key_type_t key_value();
  static trade_cache_t from_binary(const char *data);
  static trade_cache_t from_raw(const raw_type_t &raw);
  static trade_cache_t null_instance(key_type_t key_value = INT64_MIN);
  static trade_cache_t
  normalize_log_returns(const trade_cache_t &current,
                        const trade_cache_t *previous_valid);
  std::vector<double> tensor_features() const;
  bool is_valid() const;
  /* Values */
  i64 id;
  double price;
  double qty;
  double quoteQty;
  ms_t time;
  bool isBuyerMaker;
  bool isBestMatch;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct kline_cache_t {
  using key_type_t = ms_t;
  using raw_type_t = kline_t;
  /* Methods */
  static constexpr std::size_t key_offset() {
    return offsetof(kline_cache_t, close_time);
  }
  key_type_t key_value();
  static kline_cache_t from_binary(const char *data);
  static kline_cache_t from_raw(const raw_type_t &raw);
  static kline_cache_t null_instance(key_type_t key_value = INT64_MIN);
  static kline_cache_t
  normalize_log_returns(const kline_cache_t &current,
                        const kline_cache_t *previous_valid);
  std::vector<double> tensor_features() const;
  bool is_valid() const;
  /* Values */
  ms_t open_time;
  double open_price;
  double high_price;
  double low_price;
  double close_price;
  double volume;
  ms_t close_time;
  double quote_asset_volume;
  double number_of_trades;
  double taker_buy_base_volume;
  double taker_buy_quote_volume;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct basic_cache_t {
  using key_type_t = double;
  using raw_type_t = basic_t;
  /* Methods */
  static constexpr std::size_t key_offset() {
    return offsetof(basic_cache_t, time);
  }
  key_type_t key_value();
  static basic_cache_t from_binary(const char *data);
  static basic_cache_t from_raw(const raw_type_t &raw);
  static basic_cache_t
  null_instance(key_type_t key_value = std::numeric_limits<double>::min());
  static basic_cache_t
  normalize_log_returns(const basic_cache_t &current,
                        const basic_cache_t *previous_valid);
  std::vector<double> tensor_features() const;
  bool is_valid() const;
  /* Values */
  double time;
  double value;
};
#pragma pack(pop)

/* Compile-time guarantees for binary copy usage */
static_assert(std::is_trivially_copyable_v<trade_t>,
              "trade_t must be trivially copyable");
static_assert(std::is_trivially_copyable_v<kline_t>,
              "kline_t must be trivially copyable");
static_assert(std::is_trivially_copyable_v<basic_t>,
              "basic_t must be trivially copyable");
static_assert(std::is_trivially_copyable_v<trade_cache_t>,
              "trade_cache_t must be trivially copyable");
static_assert(std::is_trivially_copyable_v<kline_cache_t>,
              "kline_cache_t must be trivially copyable");
static_assert(std::is_trivially_copyable_v<basic_cache_t>,
              "basic_cache_t must be trivially copyable");


} // namespace mityane::data
