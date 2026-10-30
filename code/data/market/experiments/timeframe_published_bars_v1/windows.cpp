// Copyright (c) 2026 Waajacu. See LICENSE at repository root.
// Published spot klines -> observation windows.
//
// One instrument per row. One sample has one anchor time. The channels are
// independent series. They share that anchor and they do not share a span.
// Each channel contributes its own last count bars of its own interval,
// ending at that channel's latest bar at or before the anchor.
//   32 five-minute bars are about 160 minutes of the five-minute series.
//   24 hourly bars are 24 bars of the hourly series.
//   24 daily bars are 24 bars of the daily series.
// None of those counts is a length of another channel. The shared row is the
// longest count. A shorter channel is left-padded with empty slots. Those
// slots are not five-minute bars, and the five-minute channel is not asked
// to cover the daily series.
// This is the same cut cuwacunu_torch uses in get_edge_sample_at_anchor_key
// and get_by_anchor_key, and the same cut cuwacunu_dev uses in
// batch_on_anchor: the input field is that channel's own
// [i-(count-1) ... i], the field ends at the closest bar at or before the
// anchor, and a shorter channel is padded on the left so the last row is
// the anchor side.
// history_from_edge_samples right-aligns each channel. This program does not
// aggregate and does not hold a bar across another channel's slots.
// Each channel is loaded with one extra bar of its own series. That bar is
// the log-return predecessor. The view masks it, and the requested count
// drops it. Every requested slot is a return against the immediately previous
// bar of the same channel. The first decision leaves the finest channel's
// predecessor inside the role. A coarser channel may need its own earlier
// bars from before the role. Those bars stay on that channel.
// Validation decisions live on the finest channel. They are spaced by that
// channel's own packed count, so its packed bars do not overlap. A coarser
// channel's duration is not that gap. Nearby anchors may share a coarser
// channel's bars.
// width128.conf and width64.conf are the one-clock record of the 29 October
// measurement. This builder refuses those filenames and any settings file
// that does not name CHANNEL_BAR_COUNTS.
#include "embedding/data/observation_windows.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/family/numeric_series_history.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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
  check(rows.front() - (bar_count - 1) * step >= begin, "finest channel's own packed bars stay inside the role");
  check(rows.front() - bar_count * step >= begin, "finest channel's own log-return predecessor stays inside the role");
  return rows;
}

struct Series {
  ns::MemoryMappedDataset<ns::kline_t> bars;
  Series(const fs::path &csv, bool force_rebuild)
      : bars(ns::sanitize_csv_into_binary_file<ns::kline_t>(csv.string(), "none", force_rebuild), 1, 0, torch::kFloat64) {}
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

ns::edge_sample_t load_channel(Series &series, int64_t cutoff, int64_t desired, int64_t width, const std::string &key) {
  // desired is this channel's own packed count plus its own log-return
  // predecessor. A short file is not trimmed into a shorter channel, and it
  // is not filled with another channel's bars.
  check(series.bars.can_get_edge_sample_at_anchor_key(cutoff, static_cast<std::size_t>(desired), 0),
      "the " + key + " channel needs " + std::to_string(desired) +
          " of its own bars at or before anchor " + std::to_string(cutoff) +
          ". That is this channel's packed count plus its log-return predecessor. "
          "It is not a length of another channel");
  const int64_t count = desired;
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
    const std::vector<std::string> &keys, std::vector<std::vector<Series>> &instruments,
    const std::vector<int64_t> &times, const std::string &source, const std::string &split, const std::string &role) {
  const auto ids = cp::implementation::resolved_channel_ids(config.level);
  const auto channels = static_cast<int64_t>(ids.size());
  int64_t width = 0;
  for (const auto count : counts) width = std::max(width, count + 1);
  std::vector<torch::Tensor> values, observed, timestamps, cutoffs, row_ids;
  for (const auto cutoff : times) {
    std::vector<ns::edge_sample_t> samples;
    samples.reserve(kSymbols.size());
    for (std::size_t instrument = 0; instrument < kSymbols.size(); ++instrument) {
      std::vector<torch::Tensor> features, masks, past;
      for (int64_t channel = 0; channel < channels; ++channel) {
        auto loaded = load_channel(instruments[instrument][static_cast<std::size_t>(channel)], cutoff,
            counts[static_cast<std::size_t>(channel)] + 1, width, keys[static_cast<std::size_t>(channel)]);
        features.push_back(loaded.features);
        masks.push_back(loaded.mask);
        past.push_back(loaded.past_keys);
      }
      ns::edge_sample_t sample;
      sample.features = torch::stack(features);
      sample.mask = torch::stack(masks);
      sample.past_keys = torch::stack(past);
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
      for (int64_t channel = 0; channel < channels; ++channel) {
        const auto own = history.history.observed[instrument][channel].select(1, 3);
        check(own.sum().item<int64_t>() == counts[static_cast<std::size_t>(channel)],
            "the " + keys[static_cast<std::size_t>(channel)] +
                " channel keeps one observed bar per one of its own packed slots");
      }
      // Only the anchor channel ends on the decision close. A coarser channel
      // ends on its own latest bar at or before that close.
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

fs::path month_csv(const fs::path &root, const char *symbol, const char *interval, const char *stamp) {
  return root / symbol / interval / (std::string(symbol) + "-" + interval + "-" + stamp + ".csv");
}

// January is the role month. December 2023, when that tree sits beside it, is
// earlier bars of the same channel. The publisher files stay untouched. The
// stitched copy is only this channel's own read span, oldest month first.
// A channel with no December file is read from January alone.
struct ChannelFile {
  fs::path csv;
  bool stitched{false};
};

ChannelFile channel_file(const fs::path &january_root, const char *symbol, const char *interval) {
  const auto january = month_csv(january_root, symbol, interval, "2024-01");
  check(fs::is_regular_file(january), std::string(symbol) + " " + interval + " January file");
  const auto december = month_csv(january_root.parent_path() / "timeframe-december-2023", symbol, interval, "2023-12");
  if (!fs::is_regular_file(december)) return {january, false};
  const auto span_dir = january_root.parent_path() / "timeframe-channel-span";
  std::error_code error;
  fs::create_directories(span_dir, error);
  check(!error, "channel span directory");
  const auto stitched = span_dir / (std::string(symbol) + "-" + interval + "-2023-12-2024-01.csv");
  std::ifstream older(december, std::ios::binary);
  std::ifstream newer(january, std::ios::binary);
  check(bool(older) && bool(newer), "read this channel's own months");
  const std::string older_bytes((std::istreambuf_iterator<char>(older)), {});
  const std::string newer_bytes((std::istreambuf_iterator<char>(newer)), {});
  check(!older_bytes.empty() && !newer_bytes.empty(), "this channel's month file is empty");
  std::ofstream out(stitched, std::ios::binary | std::ios::trunc);
  check(bool(out), "write this channel's own span");
  out << older_bytes;
  if (older_bytes.back() != '\n') out << '\n';
  out << newer_bytes;
  check(bool(out), "finish this channel's own span");
  return {stitched, true};
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
    check(argc == 7, "usage: timeframe_published_windows --root PUBLISHED_ROOT --output NEWLEAF --settings FILE");
    check(std::string(argv[1]) == "--root" && std::string(argv[3]) == "--output" && std::string(argv[5]) == "--settings",
        "usage: timeframe_published_windows --root PUBLISHED_ROOT --output NEWLEAF --settings FILE");
    const fs::path settings_path = argv[6];
    const auto settings_name = settings_path.filename().string();
    check(settings_name != "width128.conf" && settings_name != "width64.conf",
        "width128.conf and width64.conf are the one-clock record of the 29 October measurement; "
        "seed 949102 held 129 minute bars, 6 hour bars, and 3 day bars on the minute sample. "
        "This builder will not read them as packed geometry and will not rebuild that dataset");
    const fs::path output = fs::absolute(argv[4]).lexically_normal();
    check(!fs::exists(output) && fs::is_directory(output.parent_path()), "new output leaf in an existing parent");
    const auto root = fs::canonical(argv[2]);
    const auto settings_file = fs::canonical(settings_path);
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    const auto document = cp::implementation::configuration::Document::read(settings_file);
    check(document.block("ENCODER").contains("CHANNEL_BAR_COUNTS"),
        "packed geometry requires CHANNEL_BAR_COUNTS; an absent list is the shared-clock record "
        "and this builder will not read it as HISTORY_LENGTH bars of every channel");
    const auto config = cp::settings(document);
    const auto counts = config.channel_bar_counts;
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
    // The gap is the anchor channel's own packed count, in that channel's
    // own steps. A coarser count is not converted into these steps.
    const auto anchor_gap_bars = finest_count;
    const auto validation_first = validation_begin + finest_count * finest_step;
    check(validation_first <= validation_end, "validation role contains one window of the anchor channel");
    const auto validation_span = (validation_end - validation_first) / finest_step;
    int64_t validation_anchors = 0;
    for (int64_t candidate = kValidationAnchors; candidate >= 2; --candidate) {
      if (validation_span / (candidate - 1) >= anchor_gap_bars) {
        validation_anchors = candidate;
        break;
      }
    }
    check(validation_anchors >= 2,
        "validation decisions on the " + keys[static_cast<std::size_t>(finest)] +
            " channel overlap: that channel's own packed count is " + std::to_string(anchor_gap_bars) +
            " bars and the role holds " + std::to_string(validation_span) +
            " steps of the same channel. A coarser channel is not part of this gap");
    std::vector<std::vector<Series>> instruments;
    instruments.reserve(kSymbols.size());
    for (const auto symbol : kSymbols) {
      std::vector<Series> channels;
      channels.reserve(ids.size());
      for (const auto &key : keys) {
        const auto file = channel_file(root, symbol, key.c_str());
        channels.emplace_back(file.csv, file.stitched);
      }
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
        "layout=independent-channels-shared-anchor;channel-ids=" + join_ids(ids) + ";keys=" + key_list.str() +
        ";packing-slots=" + std::to_string(config.level.history_length) +
        ";bar-counts=" + join_ids(counts) + ";finest-step-ms=" + std::to_string(finest_step) +
        ";source=binance-spot-monthly-klines;view=numeric-series-log-returns";
    auto train = build_role(config, counts, finest, keys, instruments, train_times, source,
        "role=TRAIN;begin=" + std::to_string(train_begin) + ";end=" + std::to_string(train_end) +
        ";anchors=1024;days=2024-01-01/2024-01-21", "TRAIN");
    auto validation = build_role(config, counts, finest, keys, instruments, validation_times, source,
        "role=VALIDATION;begin=" + std::to_string(validation_begin) + ";end=" + std::to_string(validation_end) +
        ";anchors=" + std::to_string(validation_anchors) + ";days=2024-01-22/2024-01-28", "VALIDATION");
    check(fs::create_directory(output), "exclusive output leaf");
    ow::write(output / "calibration-TRAIN.pt", train);
    ow::write(output / "probe-VALIDATION.pt", validation);
    std::ostringstream json;
    json << "{\"kind\":\"timeframe_published_bars_window_v1\",\"status\":\"passed\""
         << ",\"layout\":\"independent-channels-shared-anchor\""
         << ",\"rows\":{\"train\":" << train.rows() << ",\"validation\":" << validation.rows() << "}"
         << ",\"history\":" << config.level.history_length
         << ",\"anchor_channel\":\"" << keys[static_cast<std::size_t>(finest)] << "\""
         << ",\"anchor_channel_bars\":" << finest_count
         << ",\"validation_gap_anchor_bars\":" << anchor_gap_bars
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
