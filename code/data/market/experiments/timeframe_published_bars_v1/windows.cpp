// Copyright (c) 2026 Waajacu. See LICENSE at repository root.
// Published January 2024 spot klines -> observation windows.
// One instrument per row. Each channel is that resolution's own exchange file.
// history_from_edge_samples right-aligns each channel's own bars into the
// packing width. This program does not aggregate and does not hold a bar
// across another channel's slots. The first decision leaves one finest bar
// before the packed suffix so the log-return view can fill every requested
// finest slot. Validation decisions stay far enough apart that every packed
// channel, including its log-return predecessor, is disjoint.
#include "embedding/data/observation_windows.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/family/numeric_series_history.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace ow = embedding::data::observation_windows;
namespace cp = embedding::encoders::causal_patch_bottleneck_mae::family;
namespace ns = mityane::data;
namespace tf = embedding::encoders::timeframe_channel;

// 2024-01-01 00:00:00 UTC. Role edges are closes of whatever finest step divides
// a day, so a 1-minute run keeps the previous 1-minute constants.
constexpr int64_t kEpochMs = 1704067200000LL;
constexpr int64_t kDayMs = 86400000LL;
constexpr int64_t kRowScale = 1000000000000000LL;
constexpr int64_t kTrainAnchors = 1024;
constexpr int64_t kValidationAnchors = 32;
constexpr int64_t kAnchorMinimumBars = 8;

const std::vector<const char *> kSymbols{"BTCUSDT", "ETHUSDT", "SOLUSDT"};

void check(bool value, const std::string &message) {
  TORCH_CHECK(value, "[published timeframe windows] ", message);
}

std::vector<int64_t> anchors(int64_t begin, int64_t end, int64_t count, int64_t step, int64_t bar_count) {
  check(end >= begin && step > 0 && (end - begin) % step == 0 && count >= 2 && bar_count >= 1, "selection geometry");
  const auto first = begin + bar_count * step;
  check(first <= end, "role contains one packed window of the finest channel");
  const auto available = (end - first) / step + 1;
  check(count <= available, "enough decision times");
  std::vector<int64_t> rows;
  rows.reserve(static_cast<std::size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const auto stride = static_cast<int64_t>(static_cast<__int128>(i) * (available - 1) / (count - 1));
    rows.push_back(first + stride * step);
  }
  check(std::set<int64_t>(rows.begin(), rows.end()).size() == rows.size(), "unique decision times");
  check(rows.front() >= begin && rows.back() <= end, "decision times stay inside the role");
  check(rows.front() - (bar_count - 1) * step >= begin, "packed finest bars stay inside the role");
  check(rows.front() - bar_count * step >= begin, "finest log-return predecessor stays inside the role");
  return rows;
}

struct Series {
  ns::MemoryMappedDataset<ns::kline_t> bars;
  explicit Series(const fs::path &csv)
      : bars(ns::sanitize_csv_into_binary_file<ns::kline_t>(csv.string(), "none"), 1, 0, torch::kFloat64) {}
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

ns::edge_sample_t load_channel(Series &series, int64_t cutoff, int64_t desired, int64_t width) {
  int64_t count = desired;
  while (count > 0 && !series.bars.can_get_edge_sample_at_anchor_key(cutoff, static_cast<std::size_t>(count), 0)) --count;
  if (count == 0) return masked_channel(width);
  auto sample = series.bars.get_edge_sample_at_anchor_key(cutoff, static_cast<std::size_t>(count), 0);
  check(!sample.normalized && sample.features.scalar_type() == torch::kFloat64 && sample.past_keys.scalar_type() == torch::kInt64,
      "raw Float64 channel with Int64 keys");
  check(sample.features.dim() == 2 && sample.features.size(0) == count && sample.features.size(1) == 9, "channel feature shape");
  if (count == width) return sample;
  auto padded = masked_channel(width);
  const auto start = width - count;
  padded.features.narrow(0, start, count).copy_(sample.features);
  padded.mask.narrow(0, start, count).copy_(sample.mask);
  padded.past_keys.narrow(0, start, count).copy_(sample.past_keys);
  return padded;
}

ow::Dataset build_role(const cp::Config &config, const std::vector<int64_t> &counts, int64_t finest,
    std::vector<std::vector<Series>> &instruments, const std::vector<int64_t> &times, const std::string &source,
    const std::string &split, const std::string &role) {
  const auto ids = cp::implementation::resolved_channel_ids(config.level);
  const auto channels = static_cast<int64_t>(ids.size());
  int64_t width = 0;
  for (const auto count : counts) width = std::max(width, count + 1);
  std::vector<torch::Tensor> values, observed, timestamps, cutoffs, row_ids;
  for (const auto cutoff : times) {
    std::vector<ns::edge_sample_t> samples;
    samples.reserve(kSymbols.size());
    for (std::size_t instrument = 0; instrument < kSymbols.size(); ++instrument) {
      std::vector<torch::Tensor> features, masks, keys;
      for (int64_t channel = 0; channel < channels; ++channel) {
        auto loaded = load_channel(instruments[instrument][static_cast<std::size_t>(channel)], cutoff,
            counts[static_cast<std::size_t>(channel)] + 1, width);
        features.push_back(loaded.features);
        masks.push_back(loaded.mask);
        keys.push_back(loaded.past_keys);
      }
      ns::edge_sample_t sample;
      sample.features = torch::stack(features);
      sample.mask = torch::stack(masks);
      sample.past_keys = torch::stack(keys);
      sample.normalized = false;
      samples.push_back(std::move(sample));
    }
    auto history = cp::history_from_edge_samples(config, samples, ids, true, counts,
        std::vector<int64_t>(samples.size(), cutoff));
    check(history.history.data.size(0) == static_cast<int64_t>(kSymbols.size()) &&
              history.cutoffs.size(0) == static_cast<int64_t>(kSymbols.size()),
        "one row per instrument");
    values.push_back(history.history.data);
    observed.push_back(history.history.observed);
    timestamps.push_back(history.history.timestamps);
    cutoffs.push_back(history.cutoffs);
    auto ids_for_cutoff = torch::empty({static_cast<int64_t>(kSymbols.size())}, torch::kInt64);
    const auto slots = config.level.history_length;
    for (int64_t instrument = 0; instrument < static_cast<int64_t>(kSymbols.size()); ++instrument) {
      check(history.cutoffs[instrument].item<double>() == static_cast<double>(cutoff), "cutoff is the decision close");
      const auto closes = history.history.observed[instrument][finest].select(1, 3);
      check(closes.sum().item<int64_t>() == counts[static_cast<std::size_t>(finest)],
          "finest channel keeps one observed bar per packed slot");
      check(history.history.timestamps[instrument][finest][slots - 1].item<double>() == static_cast<double>(cutoff),
          "finest channel's last packed bar is the decision close");
      ids_for_cutoff[instrument] = instrument * kRowScale + cutoff;
    }
    row_ids.push_back(ids_for_cutoff);
  }
  ow::Dataset data;
  data.values = torch::cat(values);
  data.observed = torch::cat(observed);
  data.timestamps = torch::cat(timestamps);
  data.cutoffs = torch::cat(cutoffs);
  data.row_ids = torch::cat(row_ids);
  data.channel_ids = torch::tensor(ids, torch::kInt64);
  data.sampling_interval = config.level.sampling_interval;
  data.source_identity = source;
  data.split_identity = split;
  data.role = role;
  ow::validate(data);
  return data;
}

fs::path csv_path(const fs::path &root, const char *symbol, const char *interval) {
  return root / symbol / interval / (std::string(symbol) + "-" + interval + "-2024-01.csv");
}

std::string join_ids(const std::vector<int64_t> &ids) {
  std::ostringstream out;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) out << ',';
    out << ids[i];
  }
  return out.str();
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
    const auto settings_file = fs::canonical(settings_path);
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    const auto document = cp::implementation::configuration::Document::read(settings_file);
    const auto config = cp::settings(document);
    const auto counts = cp::channel_bar_counts(document.block("ENCODER"), config);
    const auto ids = cp::implementation::resolved_channel_ids(config.level);
    check(config.level.history_length >= 17 && !ids.empty() && counts.size() == ids.size(), "settings packing width");
    int64_t finest_step = std::numeric_limits<int64_t>::max();
    int64_t finest = -1;
    std::vector<std::string> keys;
    for (std::size_t channel = 0; channel < ids.size(); ++channel) {
      const auto *entry = tf::find(ids[channel]);
      check(entry != nullptr && entry->step_ms > 0, "each channel needs a fixed millisecond step");
      keys.emplace_back(entry->key);
      if (entry->step_ms < finest_step) {
        finest_step = entry->step_ms;
        finest = static_cast<int64_t>(channel);
      }
    }
    check(finest >= 0, "one finest channel");
    check(kEpochMs % finest_step == 0 && kDayMs % finest_step == 0,
        "finest step divides the January role lattice");
    check(config.level.sampling_interval == static_cast<double>(finest_step),
        "SAMPLING_INTERVAL must equal the finest channel step");
    const auto train_begin = kEpochMs + finest_step - 1;
    const auto train_end = kEpochMs + 21 * kDayMs - 1;
    const auto validation_begin = kEpochMs + 21 * kDayMs + finest_step - 1;
    const auto validation_end = kEpochMs + 28 * kDayMs - 1;
    const auto finest_count = counts[static_cast<std::size_t>(finest)];
    check(finest_count >= kAnchorMinimumBars, "anchor channel needs at least 8 observed bars");
    int64_t lookback_ms = 0;
    for (std::size_t channel = 0; channel < counts.size(); ++channel) {
      const auto span = counts[channel] * tf::find(ids[channel])->step_ms;
      if (span > lookback_ms) lookback_ms = span;
    }
    check(lookback_ms % finest_step == 0, "packed lookback is a whole number of finest bars");
    const auto lookback_bars = lookback_ms / finest_step;
    const auto validation_first = validation_begin + finest_count * finest_step;
    check(validation_first <= validation_end, "validation role contains one packed finest window");
    const auto validation_span = (validation_end - validation_first) / finest_step;
    int64_t validation_anchors = 0;
    for (int64_t candidate = kValidationAnchors; candidate >= 2; --candidate) {
      if (validation_span / (candidate - 1) >= lookback_bars) {
        validation_anchors = candidate;
        break;
      }
    }
    check(validation_anchors >= 2,
        "validation windows do not overlap: packed lookback is " + std::to_string(lookback_bars) +
            " finest bars and the role holds " + std::to_string(validation_span) + " steps");
    std::vector<std::vector<Series>> instruments;
    instruments.reserve(kSymbols.size());
    for (const auto symbol : kSymbols) {
      std::vector<Series> channels;
      channels.reserve(ids.size());
      for (const auto &key : keys) channels.emplace_back(csv_path(root, symbol, key.c_str()));
      instruments.push_back(std::move(channels));
    }
    const auto train_times = anchors(train_begin, train_end, kTrainAnchors, finest_step, finest_count);
    const auto validation_times = anchors(validation_begin, validation_end, validation_anchors, finest_step, finest_count);
    check(train_times.back() < validation_times.front(), "train decision times end before validation");
    std::ostringstream key_list;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (i) key_list << ',';
      key_list << keys[i];
    }
    const std::string source =
        "timeframe-published-bars-v1;population=january-2024-mkt-hist-2;one-instrument-per-row;"
        "layout=per-channel-right-aligned;channel-ids=" + join_ids(ids) + ";keys=" + key_list.str() +
        ";packing-slots=" + std::to_string(config.level.history_length) +
        ";bar-counts=" + join_ids(counts) + ";finest-step-ms=" + std::to_string(finest_step) +
        ";source=binance-spot-monthly-klines;view=numeric-series-log-returns";
    auto train = build_role(config, counts, finest, instruments, train_times, source,
        "role=TRAIN;begin=" + std::to_string(train_begin) + ";end=" + std::to_string(train_end) +
        ";anchors=1024;days=2024-01-01/2024-01-21", "TRAIN");
    auto validation = build_role(config, counts, finest, instruments, validation_times, source,
        "role=VALIDATION;begin=" + std::to_string(validation_begin) + ";end=" + std::to_string(validation_end) +
        ";anchors=" + std::to_string(validation_anchors) + ";days=2024-01-22/2024-01-28", "VALIDATION");
    check(fs::create_directory(output), "exclusive output leaf");
    ow::write(output / "calibration-TRAIN.pt", train);
    ow::write(output / "probe-VALIDATION.pt", validation);
    std::ostringstream json;
    json << "{\"kind\":\"timeframe_published_bars_window_v1\",\"status\":\"passed\""
         << ",\"layout\":\"per-channel-right-aligned\""
         << ",\"rows\":{\"train\":" << train.rows() << ",\"validation\":" << validation.rows() << "}"
         << ",\"history\":" << config.level.history_length
         << ",\"packed_lookback_finest_bars\":" << lookback_bars
         << ",\"anchors\":{\"train\":" << kTrainAnchors << ",\"validation\":" << validation_anchors << "}"
         << ",\"instruments\":[\"BTCUSDT\",\"ETHUSDT\",\"SOLUSDT\"],\"channel_ids\":[" << join_ids(ids) << "]"
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
