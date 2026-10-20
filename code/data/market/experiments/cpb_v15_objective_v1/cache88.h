// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// CACHE88 field order follows the preserved MIT cuwacunu kline cache contract.
// This owned converter does not import its writer or normalize observations.
#pragma once
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace mityane::data::cpb_v15 {
#pragma pack(push, 1)
struct Cache88 {
  int64_t open_time;
  double open_price, high_price, low_price, close_price, volume;
  int64_t close_time;
  double quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume;
};
#pragma pack(pop)
static_assert(sizeof(Cache88) == 88 && offsetof(Cache88, close_time) == 48);
static_assert(std::endian::native == std::endian::little && sizeof(double) == 8
  && std::numeric_limits<double>::is_iec559 && std::is_trivially_copyable_v<Cache88>);
inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
inline std::array<std::string_view, 12> columns(std::string_view line) {
  require(!line.empty() && line.size() <= 4096, "nonempty bounded CSV row required");
  if (line.back() == '\r') line.remove_suffix(1);
  require(!line.empty() && line.find_first_of("\"\r\n\t ") == std::string_view::npos
    && line.find('\0') == std::string_view::npos, "unquoted numeric CSV row required");
  std::array<std::string_view, 12> result;
  for (size_t index = 0; index < result.size(); ++index) {
    const auto end = line.find(',');
    require((index + 1 == result.size()) == (end == std::string_view::npos), "exactly twelve CSV columns required");
    result[index] = line.substr(0, end);
    require(!result[index].empty(), "empty CSV field");
    if (end != std::string_view::npos) line.remove_prefix(end + 1);
  }
  return result;
}
template<class Integer> inline Integer integer(std::string_view token) {
  Integer value{};
  const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value, 10);
  require(parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size(), "exact bounded decimal integer required");
  return value;
}
inline double decimal(std::string_view token) {
  // std::stod is the preserved importer's binary64 conversion. The executable
  // fixes LC_NUMERIC=C; exact token consumption rejects headers and suffixes.
  size_t consumed = 0;
  double value{};
  try { value = std::stod(std::string(token), &consumed); }
  catch (...) { throw std::runtime_error("finite binary64 decimal required"); }
  require(consumed == token.size() && std::isfinite(value), "finite whole binary64 decimal required");
  // The source format is decimal rather than C hexadecimal/locale notation.
  require(token.find_first_not_of("+-0123456789.eE") == std::string_view::npos, "decimal field spelling required");
  return value;
}
inline Cache88 parse_row(std::string_view line, int64_t expected_open) {
  const auto fields = columns(line);
  Cache88 result{};
  result.open_time = integer<int64_t>(fields[0]);
  result.open_price = decimal(fields[1]); result.high_price = decimal(fields[2]);
  result.low_price = decimal(fields[3]); result.close_price = decimal(fields[4]);
  result.volume = decimal(fields[5]); result.close_time = integer<int64_t>(fields[6]);
  result.quote_asset_volume = decimal(fields[7]);
  const auto trades = integer<int32_t>(fields[8]);
  result.number_of_trades = static_cast<double>(trades);
  result.taker_buy_base_volume = decimal(fields[9]); result.taker_buy_quote_volume = decimal(fields[10]);
  // Binance labels column 12 "ignore". Validate its finite decimal syntax,
  // retain the CSV verbatim, and omit it from the exact legacy CACHE88 layout.
  (void)decimal(fields[11]);
  require(result.open_time == expected_open && expected_open >= 0
    && expected_open <= std::numeric_limits<int64_t>::max() - 59999
    && result.close_time == expected_open + 59999, "exact declared millisecond candle chronology required");
  require(result.open_price > 0 && result.high_price > 0 && result.low_price > 0 && result.close_price > 0,
    "positive finite OHLC prices required");
  require(trades >= 0 && result.volume >= 0 && result.quote_asset_volume >= 0
    && result.taker_buy_base_volume >= 0 && result.taker_buy_quote_volume >= 0, "nonnegative activity fields required");
  return result;
}
struct Conversion { int64_t rows, first_open_ms, first_close_ms, last_close_ms, cache_bytes; };
template<class Writer> Conversion convert_csv(std::istream& input, int64_t first_open, int64_t expected_rows, Writer writer) {
  require(first_open >= 0 && first_open % 60000 == 0 && expected_rows > 0
    && expected_rows <= 100000 && first_open <= std::numeric_limits<int64_t>::max() - expected_rows * 60000,
    "bounded explicit conversion geometry required");
  std::string line;
  int64_t count = 0;
  while (std::getline(input, line)) {
    require(count < expected_rows, "more CSV rows than declared");
    try { const auto record = parse_row(line, first_open + count * 60000); writer(record); }
    catch (const std::exception& error) { throw std::runtime_error("CSV row " + std::to_string(count + 1) + ": " + error.what()); }
    ++count;
  }
  require(input.eof() && !input.bad() && count == expected_rows, "complete exact declared CSV row count required");
  return {count, first_open, first_open + 59999, first_open + count * 60000 - 1, count * 88};
}
} // namespace mityane::data::cpb_v15
