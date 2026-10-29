// Settings reader and the published raw-cache door.
// SPDX-License-Identifier: MIT
#include "dataloader.h"
#include "sha256.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace data = mityane::data;

int failures = 0;

void check(bool condition, const std::string &message) {
  if (condition)
    return;
  std::cerr << "[settings] " << message << "\n";
  ++failures;
}

void rejects(const std::function<void()> &call, const std::string &message) {
  try {
    call();
    check(false, message);
  } catch (const std::exception &) {
    check(true, message);
  }
}

std::filesystem::path temp_dir() {
  auto dir = std::filesystem::temp_directory_path() /
             ("mityane_mm_raw_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

void write_cache(const std::filesystem::path &path, const std::vector<double> &closes,
                 std::int64_t first_close, std::int64_t step) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
  for (std::size_t i = 0; i < closes.size(); ++i) {
    data::kline_cache_t row{};
    row.close_time = first_close + static_cast<std::int64_t>(i) * step;
    row.open_time = row.close_time - (step - 1);
    row.open_price = closes[i];
    row.high_price = closes[i];
    row.low_price = closes[i];
    row.close_price = closes[i];
    row.volume = 1;
    row.quote_asset_volume = 1;
    row.number_of_trades = 1;
    row.taker_buy_base_volume = 1;
    row.taker_buy_quote_volume = 1;
    out.write(reinterpret_cast<const char *>(&row), sizeof row);
  }
}

void test_sha256_vector() {
  check(data::sha256_bytes_hex("abc") ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA256 of abc");
}

void test_settings_reader() {
  const char *text = R"(
    // defaults may follow the sets they configure
    KLINE_SOURCE_SET {
      INSTRUMENT = BTCUSDT;
      MARKET_TYPE = spot;
      VENUE = binance;
      BASE_ASSET = BTC;
      QUOTE_ASSET = USDT;
      SOURCE_KIND = real;
      INTERVALS = 1d, 1w;
    }
    CHANNEL_SET {
      ACTIVE = true;
      RECORD_TYPE = kline;
      INPUT_LENGTH = 30;
      FUTURE_LENGTH = 1;
      CHANNEL_WEIGHT = 1.0;
      NORMALIZATION_POLICY = log_returns;
      INTERVALS = 1d, 1w;
    }
    SOURCE_DEFAULTS {
      SOURCE_ROOT = /data/raw/;
      KLINE_INTERVALS = 1m;
    }
  )";
  const auto request = data::read_loader_settings(text);
  check(request.source_forms.size() == 2, "one source row per listed interval");
  check(request.source_forms[0].source ==
            "/data/raw/BTCUSDT/1d/BTCUSDT-1d-all-years.csv",
        "1d path uses the instrument and interval");
  check(request.source_forms[1].interval == data::interval_type_e::interval_1w,
        "1w interval");
  check(request.source_forms[0].record_type == "kline", "kline record type");
  check(request.channel_forms.size() == 2, "one channel row per listed interval");
  check(request.channel_forms[0].active == "true" &&
            request.channel_forms[0].input_length == "30" &&
            request.channel_forms[0].future_length == "1" &&
            request.channel_forms[0].channel_weight == "1.0" &&
            request.channel_forms[0].normalization_policy == "log_returns",
        "channel fields keep the torch text");
  check(request.channel_forms[1].interval == data::interval_type_e::interval_1w,
        "channel interval 1w");

  rejects([] { (void)data::read_loader_settings("CSV_POLICY { CSV_BOOTSTRAP_DELTAS = 8; }"); },
          "unsupported block is rejected");
  rejects([] {
    (void)data::read_loader_settings(
        "KLINE_SOURCE_SET { INSTRUMENT = BTCUSDT; MARKET_TYPE = spot; VENUE = binance; "
        "BASE_ASSET = BTC; QUOTE_ASSET = USDT; SOURCE_KIND = real; }");
  }, "missing root and intervals is rejected");
}

void test_raw_cache_door() {
  static_assert(sizeof(data::kline_cache_t) == 88, "kline cache record is 88 bytes");
  const auto dir = temp_dir();
  const auto path = dir / "BTCUSDT-1m.cache.bin";
  const std::vector<double> closes{10, 20, 30, 40, 50, 60};
  constexpr std::int64_t first = 1700000000000;
  constexpr std::int64_t step = 60000;
  write_cache(path, closes, first, step);
  const auto hash = data::sha256_file_hex(path.string());

  data::raw_cache_channel_t channel;
  channel.path = path.string();
  channel.sha256 = hash;
  channel.input_length = 3;
  channel.future_length = 1;
  auto edge = data::open_raw_caches<data::kline_t>({channel});
  const auto anchor = first + 3 * step;
  const auto sample = edge.get_by_anchor_key(anchor);
  check(sample.features.dim() == 3 && sample.features.size(0) == 1 &&
            sample.features.size(1) == 3 && sample.features.size(2) == 9,
        "window shape");
  check(sample.past_keys.dim() == 2 && sample.past_keys.size(0) == 1 &&
            sample.past_keys.size(1) == 3,
        "past key shape");
  check(sample.features[0][0][3].item<float>() == 20.f &&
            sample.features[0][1][3].item<float>() == 30.f &&
            sample.features[0][2][3].item<float>() == 40.f,
        "close feature is the raw cache value");
  check(sample.past_keys[0][0].item<std::int64_t>() == first + step &&
            sample.past_keys[0][2].item<std::int64_t>() == anchor,
        "past keys are the channel close times");
  check(sample.normalized == false, "raw cache stays raw");

  channel.sha256.back() = channel.sha256.back() == '0' ? '1' : '0';
  rejects([&] { data::open_raw_caches<data::kline_t>({channel}); }, "wrong SHA256 is rejected");

  const auto normalized = dir / "BTCUSDT-1m.cache.norm.log_returns.bin";
  std::filesystem::copy_file(path, normalized);
  channel.path = normalized.string();
  channel.sha256 = data::sha256_file_hex(normalized.string());
  rejects([&] { data::open_raw_caches<data::kline_t>({channel}); },
          "normalized filename is rejected");
  std::filesystem::remove_all(dir);
}

} // namespace

int main() {
  try {
    test_sha256_vector();
    test_settings_reader();
    test_raw_cache_door();
  } catch (const std::exception &ex) {
    std::cerr << "[settings] unexpected: " << ex.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << "[settings] " << failures << " checks failed\n";
    return 1;
  }
  std::cout << "Settings and raw cache passed\n";
  return 0;
}
