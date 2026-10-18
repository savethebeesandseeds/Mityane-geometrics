// SPDX-License-Identifier: MIT
#include "cuwacunu/data/dataloader.h"
#include "cuwacunu/data/record_types.h"
#include <filesystem>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace {
namespace data = cuwacunu::data;
void check(bool condition, const char *message) { TORCH_CHECK(condition, message); }
template <typename Function> void rejects(Function action, const char *message) {
  bool failed = false;
  try { action(); } catch (const std::exception &) { failed = true; }
  check(failed, message);
}
struct Fixture {
  std::filesystem::path path;
  explicit Fixture(const std::filesystem::path &parent) {
    std::filesystem::create_directories(parent);
    auto pattern = (parent / "cuwacunu-data-safety-XXXXXX").string();
    std::vector<char> bytes(pattern.begin(), pattern.end()); bytes.push_back('\0');
    const auto directory = ::mkdtemp(bytes.data());
    check(directory != nullptr, "temporary fixture directory"); path = directory;
  }
  ~Fixture() { std::error_code error; std::filesystem::remove_all(path, error); }
};
data::kline_t row(int64_t key, double price) {
  data::kline_t value{};
  value.open_time = key - 1; value.close_time = key;
  value.open_price = value.high_price = value.low_price = value.close_price = price;
  value.volume = 10; value.quote_asset_volume = 10 * price; value.number_of_trades = 3;
  value.taker_buy_base_volume = 4; value.taker_buy_quote_volume = 4 * price;
  return value;
}
void write_csv(const std::filesystem::path &path, int count, double price, bool bad = false) {
  std::ofstream csv(path);
  for (int i = 0; i < count; ++i) { row(1000 + i, price + i).to_csv(csv); csv << '\n'; }
  if (bad) csv << "malformed late row\n";
  check(csv.good(), "fixture CSV write");
}
std::vector<char> bytes(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  check(file.good(), "read fixture cache");
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void cache_publication(const std::filesystem::path &directory) {
  const auto csv = directory / "existing.csv";
  write_csv(csv, 80, 100);
  const auto raw = data::sanitize_csv_into_binary_file<data::kline_t>(csv.string(), "none", true, 2);
  const auto previous = bytes(raw);
  data::MemoryMappedDataset<data::kline_t> mapped(raw, 2, 0);
  const auto old_sample = mapped.get(0);
  write_csv(csv, 90, 900, true);
  rejects([&] { data::sanitize_csv_into_binary_file<data::kline_t>(csv.string(), "none", true, 2); },
          "malformed late CSV must fail cache preparation");
  check(bytes(raw) == previous, "failed refresh preserves complete previous cache");
  check(torch::equal(mapped.get(0).features, old_sample.features), "failed refresh preserves live mapped input");
  const auto first_csv = directory / "first.csv";
  write_csv(first_csv, 90, 100, true);
  const auto first_raw = data::raw_binary_for_csv(first_csv.string());
  for (int attempt = 0; attempt < 2; ++attempt) {
    rejects([&] { data::sanitize_csv_into_binary_file<data::kline_t>(first_csv.string(), "none", false, 2); },
            "failed first cache build cannot become a successful retry");
    check(!std::filesystem::exists(first_raw), "failed first build publishes no partial cache");
  }
  write_csv(csv, 120, 500);
  data::sanitize_csv_into_binary_file<data::kline_t>(csv.string(), "none", true, 2);
  data::MemoryMappedDataset<data::kline_t> refreshed(raw, 2, 0);
  check(mapped.raw_records() == 80 && refreshed.raw_records() == 120, "mapped readers retain their original cache snapshot");
  check(torch::equal(mapped.get(0).features, old_sample.features), "successful refresh preserves old mapping and values");
  check(refreshed.get(0).features.index({0, 0}).item<float>() == 500, "new reader sees complete refreshed cache");
  for (const auto &entry : std::filesystem::directory_iterator(directory))
    check(entry.path().filename().string().find(".tmp.") == std::string::npos,
          "cache staging files removed after success and failure");

  const auto normalized = data::sanitize_csv_into_binary_file<data::kline_t>(csv.string(), "log_returns", true, 2);
  data::MemoryMappedDataset<data::kline_t> old_normalized(normalized, 2, 0);
  const auto normalized_sample = old_normalized.get(0);
  write_csv(csv, 100, 800);
  data::sanitize_csv_into_binary_file<data::kline_t>(csv.string(), "log_returns", true, 2);
  check(torch::equal(old_normalized.get(0).features, normalized_sample.features),
        "normalized refresh also preserves an existing mapped snapshot");
}
void normalization_and_bounds(const std::filesystem::path &directory) {
  const auto a = directory / "raw-a.csv", b = directory / "raw-b.csv";
  write_csv(a, 80, 100); write_csv(b, 80, 200);
  data::MemoryMappedEdgeDataset<data::kline_t> raw;
  raw.add_dataset(a.string(), 4, 0, "none", true);
  raw.add_dataset(b.string(), 4, 0, "raw", true);
  check(!raw.get(0).normalized, "all raw aliases remain explicitly raw");
  data::MemoryMappedEdgeDataset<data::kline_t> mixed;
  mixed.add_dataset(a.string(), 4, 0, "none", false);
  rejects([&] { mixed.add_dataset(b.string(), 4, 0, "log_returns", false); },
          "mixed raw and normalized channels rejected");
  check(mixed.get(0).features.size(0) == 1 && !mixed.get(0).normalized,
        "rejected source leaves existing dataset usable");
  const auto raw_path = data::raw_binary_for_csv(a.string());
  data::MemoryMappedDataset<data::kline_t> single(raw_path, 1, 0);
  auto sample = single.get(0); auto normalized = sample; normalized.normalized = true;
  rejects([&] { data::sample_t::collate_fn({sample, normalized}); },
          "mixed raw/normalized samples must not be mislabeled raw");
  rejects([&] { data::sample_t::collate_fn_past({sample, normalized}); },
          "past-only collation rejects mixed normalization");
  rejects([&] { data::sample_t::collate_fn_future({sample, normalized}); },
          "future-only collation rejects mixed normalization");
  check(data::sample_t::collate_fn_past({normalized, normalized}).normalized &&
        data::sample_t::collate_fn_future({normalized, normalized}).normalized,
        "partial collators preserve known normalization state");
  auto stats_batch = data::sample_t::collate_fn(std::vector<data::sample_t>(9, sample));
  stats_batch.feature_mean = torch::arange(9, torch::kFloat32);
  stats_batch.feature_std = torch::ones({9}, torch::kFloat32);
  for (const auto &unbatched : data::sample_t::decollate_fn(stats_batch))
    check(torch::equal(unbatched.feature_mean, stats_batch.feature_mean) &&
          torch::equal(unbatched.feature_std, stats_batch.feature_std),
          "shared [F] statistics stay shared when batch size equals feature width");
  check(!single.can_get_edge_sample_at_anchor_key(999, 1, 0), "before-first anchor availability rejected");
  rejects([&] { single.get_edge_sample_at_anchor_key(999, 1, 0); },
          "before-first anchor must not expose a future candle as past");
  check(single.get_edge_sample_at_anchor_key(1000, 1, 0).past_keys.item<int64_t>() == 1000,
        "first causal anchor remains usable");
  const auto maximum = std::numeric_limits<std::size_t>::max();
  rejects([&] { data::MemoryMappedDataset<data::kline_t> invalid(raw_path, maximum, 1); },
          "oversized constructor horizons rejected before overflow");
  rejects([&] { raw.add_dataset((directory / "must-not-be-read.csv").string(), maximum, 1); },
          "oversized source horizons rejected before materialization");
  rejects([&] { single.get_edge_sample_at_anchor_key(1001, 1, maximum); },
          "overflowing future horizon rejected using safe bounds");
  rejects([&] { data::detail::parse_required_size("-1", "input_length", "test"); },
          "negative source horizon rejected");
  rejects([&] { data::detail::parse_required_size("1bad", "input_length", "test"); },
          "malformed source horizon rejected");
}
void writer_serialization(const std::filesystem::path &directory) {
  const auto csv = directory / "concurrent.csv";
  write_csv(csv, 80, 100);
  const auto raw = data::raw_binary_for_csv(csv.string());
  const auto lock_path = raw + ".lock";
  const auto fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  check(fd >= 0, "external cache lock open");
  check(::flock(fd, LOCK_EX) == 0, "external cache writer lock");
  std::promise<void> started;
  auto ready = started.get_future();
  auto writer = std::async(std::launch::async, [&] {
    started.set_value();
    return data::sanitize_csv_into_binary_file<data::kline_t>(csv.string(), "none", true, 2);
  });
  ready.get();
  const bool blocked = writer.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout;
  const bool unpublished = !std::filesystem::exists(raw);
  ::flock(fd, LOCK_UN); ::close(fd);
  writer.get();
  check(blocked && unpublished, "cache writer waits for another writer before publication");
  std::vector<std::future<std::string>> writers;
  for (int i = 0; i < 4; ++i)
    writers.push_back(std::async(std::launch::async, [&, i] {
      return data::sanitize_csv_into_binary_file<data::kline_t>(
          csv.string(), i % 2 ? "none" : "log_returns", true, 2);
    }));
  for (auto &future : writers) future.get();
  const auto normalized = data::normalized_bin_for_csv(csv.string(), "log_returns");
  data::MemoryMappedDataset<data::kline_t> raw_data(raw, 1, 0), normalized_data(normalized, 1, 0);
  check(raw_data.raw_records() == 80 && normalized_data.raw_records() == 80,
        "concurrent raw and normalized writers publish complete snapshots");
}
void strict_csv() {
  const std::string valid = "1000,100,101,99,100,10,1001,1000,3,4,400,0";
  check(data::kline_t::from_csv(valid).open_price == 100, "standard Binance row parses");
  for (const auto &invalid : {std::string("1000junk,100,101,99,100,10,1001,1000,3,4,400,0"),
                              std::string("1000,100junk,101,99,100,10,1001,1000,3,4,400,0"),
                              std::string("1000,100,101,99,100,10,1001,1000,3junk,4,400,0")})
    rejects([&] { data::kline_t::from_csv(invalid); }, "CSV numerical token suffix rejected");
  check(data::kline_t::from_csv("1000,100 ,101,99,100,10,1001,1000,3,4,400,0\r").open_price == 100,
        "ordinary whitespace and CRLF remain accepted");
}
} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 1 || argc == 2, "usage: test_data_safety [fixture-parent-directory]");
    torch::set_num_threads(1);
    Fixture fixture(argc == 2 ? std::filesystem::path(argv[1]) : std::filesystem::temp_directory_path());
    cache_publication(fixture.path);
    normalization_and_bounds(fixture.path);
    writer_serialization(fixture.path);
    strict_csv();
    std::cout << "Data safety passed: cache failure atomicity, live raw/normalized mappings, mixed normalization rejection, causal anchors, horizon bounds, strict CSV tokens\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
