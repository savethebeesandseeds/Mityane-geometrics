// Copyright (c) 2026 Waajacu. See LICENSE at repository root.
// Published January 2024 spot klines -> observation windows.
// One instrument per row. Channels are the publisher's 1m, 1h, and 1d files.
// The numeric-series loader reads each file. history_from_edge_samples places
// those bars on the shared 1-minute clock. This program does not aggregate
// minute bars into coarser bars.
#include "embedding/data/observation_windows.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/family/numeric_series_history.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace ow = embedding::data::observation_windows;
namespace cp = embedding::encoders::causal_patch_bottleneck_mae::family;
namespace ns = mityane::data;

constexpr int64_t kInterval = 60000;
constexpr int64_t kHistory = 128;
constexpr int64_t kTrainBegin = 1704067259999LL;
constexpr int64_t kTrainEnd = 1705881599999LL;
constexpr int64_t kValidationBegin = 1705881659999LL;
constexpr int64_t kValidationEnd = 1706486399999LL;
constexpr int64_t kRowScale = 1000000000000000LL;
constexpr int64_t kTrainAnchors = 1024;
constexpr int64_t kValidationAnchors = 32;

const std::array<const char *, 3> kSymbols{"BTCUSDT", "ETHUSDT", "SOLUSDT"};
const std::array<const char *, 3> kIntervals{"1m", "1h", "1d"};
const std::array<int64_t, 3> kDesired{129, 6, 3};
int64_t active_history = kHistory;
std::array<int64_t, 3> active_desired = kDesired;

void check(bool value, const std::string &message) {
  TORCH_CHECK(value, "[published timeframe windows] ", message);
}

std::vector<int64_t> anchors(int64_t begin, int64_t end, int64_t count) {
  check(end >= begin && (end - begin) % kInterval == 0 && count >= 2, "selection geometry");
  const auto first = begin + (active_history - 1) * kInterval;
  check(first <= end, "role contains one full anchor window");
  const auto available = (end - first) / kInterval + 1;
  check(count <= available, "enough anchor times");
  std::vector<int64_t> rows;
  rows.reserve(static_cast<std::size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const auto step = static_cast<int64_t>(static_cast<__int128>(i) * (available - 1) / (count - 1));
    rows.push_back(first + step * kInterval);
  }
  check(std::set<int64_t>(rows.begin(), rows.end()).size() == rows.size(), "unique anchor times");
  check(rows.front() >= begin && rows.back() <= end, "anchors stay inside the role");
  check(rows.front() - (active_history - 1) * kInterval >= begin, "retained minute grid stays inside the role");
  return rows;
}

struct Channel {
  ns::MemoryMappedDataset<ns::kline_t> series;
  explicit Channel(const fs::path &csv)
      : series(ns::sanitize_csv_into_binary_file<ns::kline_t>(csv.string(), "none"), 1, 0, torch::kFloat64) {}
};

ns::edge_sample_t masked_channel(int64_t length) {
  ns::edge_sample_t sample;
  sample.features = torch::zeros({length, 9}, torch::kFloat64);
  sample.mask = torch::zeros({length}, torch::kBool);
  sample.past_keys = torch::zeros({length}, torch::kInt64);
  sample.future_features = torch::empty({0, 9}, torch::kFloat64);
  sample.future_mask = torch::empty({0}, torch::kBool);
  sample.future_keys = torch::empty({0}, torch::kInt64);
  sample.normalized = false;
  return sample;
}

ns::edge_sample_t load_channel(Channel &channel, int64_t cutoff, int64_t desired, int64_t length) {
  int64_t count = desired;
  while (count > 0 && !channel.series.can_get_edge_sample_at_anchor_key(cutoff, static_cast<std::size_t>(count), 0))
    --count;
  if (count == 0) return masked_channel(length);
  auto sample = channel.series.get_edge_sample_at_anchor_key(cutoff, static_cast<std::size_t>(count), 0);
  check(!sample.normalized && sample.features.scalar_type() == torch::kFloat64 && sample.past_keys.scalar_type() == torch::kInt64,
      "raw Float64 channel with Int64 keys");
  check(sample.features.dim() == 2 && sample.features.size(0) == count && sample.features.size(1) == 9, "channel feature shape");
  if (count == length) return sample;
  auto padded = masked_channel(length);
  const auto start = length - count;
  padded.features.narrow(0, start, count).copy_(sample.features);
  padded.mask.narrow(0, start, count).copy_(sample.mask);
  padded.past_keys.narrow(0, start, count).copy_(sample.past_keys);
  return padded;
}

ow::Dataset build_role(const cp::Config &config, std::vector<std::array<Channel, 3>> &instruments,
    const std::vector<int64_t> &times, const std::string &source, const std::string &split, const std::string &role) {
  const auto ids = cp::implementation::resolved_channel_ids(config.level);
  std::vector<torch::Tensor> values, observed, timestamps, cutoffs, row_ids;
  for (const auto cutoff : times) {
    std::vector<ns::edge_sample_t> samples;
    samples.reserve(3);
    for (int64_t instrument = 0; instrument < 3; ++instrument) {
      auto minute = load_channel(instruments[static_cast<std::size_t>(instrument)][0], cutoff, active_desired[0], active_desired[0]);
      const auto length = minute.features.size(0);
      check(length == active_history || length == active_history + 1, "minute window is history or history+1 bars");
      std::vector<ns::edge_sample_t> channels;
      channels.push_back(std::move(minute));
      for (int64_t channel = 1; channel < 3; ++channel)
        channels.push_back(load_channel(instruments[static_cast<std::size_t>(instrument)][static_cast<std::size_t>(channel)],
            cutoff, active_desired[static_cast<std::size_t>(channel)], length));
      ns::edge_sample_t sample;
      sample.features = torch::stack({channels[0].features, channels[1].features, channels[2].features});
      sample.mask = torch::stack({channels[0].mask, channels[1].mask, channels[2].mask});
      sample.past_keys = torch::stack({channels[0].past_keys, channels[1].past_keys, channels[2].past_keys});
      sample.normalized = false;
      samples.push_back(std::move(sample));
    }
    auto history = cp::history_from_edge_samples(config, samples, ids, true);
    check(history.history.data.size(0) == 3 && history.cutoffs.size(0) == 3, "one row per instrument");
    values.push_back(history.history.data);
    observed.push_back(history.history.observed);
    timestamps.push_back(history.history.timestamps);
    cutoffs.push_back(history.cutoffs);
    auto ids_for_anchor = torch::empty({3}, torch::kInt64);
    for (int64_t instrument = 0; instrument < 3; ++instrument) {
      check(history.cutoffs[instrument].item<double>() == static_cast<double>(cutoff), "cutoff is the anchor close");
      const auto closes = history.history.observed[instrument][0].select(1, 3);
      check(closes.sum().item<int64_t>() >= active_history - 1, "anchor minute log returns cover the window");
      ids_for_anchor[instrument] = instrument * kRowScale + cutoff;
    }
    row_ids.push_back(ids_for_anchor);
  }
  ow::Dataset data;
  data.values = torch::cat(values);
  data.observed = torch::cat(observed);
  data.timestamps = torch::cat(timestamps);
  data.cutoffs = torch::cat(cutoffs);
  data.row_ids = torch::cat(row_ids);
  data.channel_ids = torch::tensor(ids, torch::kInt64);
  data.sampling_interval = static_cast<double>(kInterval);
  data.source_identity = source;
  data.split_identity = split;
  data.role = role;
  ow::validate(data);
  return data;
}

fs::path csv_path(const fs::path &root, const char *symbol, const char *interval) {
  return root / symbol / interval / (std::string(symbol) + "-" + interval + "-2024-01.csv");
}
} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 5 || argc == 7, "usage: timeframe_published_windows --root PUBLISHED_ROOT --output NEWLEAF [--settings FILE]");
    check(std::string(argv[1]) == "--root" && std::string(argv[3]) == "--output",
        "usage: timeframe_published_windows --root PUBLISHED_ROOT --output NEWLEAF [--settings FILE]");
    fs::path settings_path = "code/data/market/experiments/timeframe_published_bars_v1/width64.conf";
    if (argc == 7) {
      check(std::string(argv[5]) == "--settings", "optional pair is --settings FILE");
      settings_path = argv[6];
    }
    const auto root = fs::canonical(argv[2]);
    const fs::path output = fs::absolute(argv[4]).lexically_normal();
    check(!fs::exists(output) && fs::is_directory(output.parent_path()), "new output leaf in an existing parent");
    const auto settings = fs::canonical(settings_path);
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    const auto config = cp::read_settings(settings);
    active_history = config.level.history_length;
    check(active_history >= 17 && config.level.sampling_interval == static_cast<double>(kInterval),
        "settings history and anchor grid");
    if (active_history == kHistory) active_desired = kDesired;
    else active_desired = {active_history + 1, active_history / 60 + 2, 3};
    const auto validation_span = (kValidationEnd - (kValidationBegin + (active_history - 1) * kInterval)) / kInterval;
    int64_t validation_anchors = kValidationAnchors;
    if (active_history != kHistory) {
      validation_anchors = 8;
      for (const int64_t count : {int64_t{32}, int64_t{16}}) {
        if (validation_span / (count - 1) >= active_history) { validation_anchors = count; break; }
      }
      check(validation_span / (validation_anchors - 1) >= active_history, "validation windows do not overlap");
    }
    std::vector<std::array<Channel, 3>> instruments;
    instruments.reserve(3);
    for (const auto symbol : kSymbols) {
      std::array<Channel, 3> channels{
          Channel(csv_path(root, symbol, kIntervals[0])),
          Channel(csv_path(root, symbol, kIntervals[1])),
          Channel(csv_path(root, symbol, kIntervals[2]))};
      instruments.push_back(std::move(channels));
    }
    const auto train_times = anchors(kTrainBegin, kTrainEnd, kTrainAnchors);
    const auto validation_times = anchors(kValidationBegin, kValidationEnd, validation_anchors);
    check(train_times.back() < validation_times.front(), "train anchors end before validation");
    const std::string source =
        "timeframe-published-bars-v1;population=january-2024-mkt-hist-2;one-instrument-per-row;"
        "channel-ids=4,9,15;keys=1m,1h,1d;anchor-ms=60000;history=" + std::to_string(active_history) + ";"
        "source=binance-spot-monthly-klines;view=numeric-series-log-returns";
    auto train = build_role(config, instruments, train_times, source,
        "role=TRAIN;begin=1704067259999;end=1705881599999;anchors=1024;days=2024-01-01/2024-01-21", "TRAIN");
    auto validation = build_role(config, instruments, validation_times, source,
        "role=VALIDATION;begin=1705881659999;end=1706486399999;anchors=" + std::to_string(validation_anchors) +
        ";days=2024-01-22/2024-01-28", "VALIDATION");
    check(fs::create_directory(output), "exclusive output leaf");
    ow::write(output / "calibration-TRAIN.pt", train);
    ow::write(output / "probe-VALIDATION.pt", validation);
    std::ostringstream json;
    json << "{\"kind\":\"timeframe_published_bars_window_v1\",\"status\":\"passed\""
         << ",\"rows\":{\"train\":" << train.rows() << ",\"validation\":" << validation.rows() << "}"
         << ",\"history\":" << active_history
         << ",\"anchors\":{\"train\":" << kTrainAnchors << ",\"validation\":" << validation_anchors << "}"
         << ",\"instruments\":[\"BTCUSDT\",\"ETHUSDT\",\"SOLUSDT\"],\"channel_ids\":[4,9,15]"
         << ",\"promotes_reference\":false,\"confirmation_bound\":false}";
    std::ofstream report(output / "preparation.json");
    check(bool(report), "open preparation report");
    report << json.str() << '\n';
    std::cout << json.str() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
