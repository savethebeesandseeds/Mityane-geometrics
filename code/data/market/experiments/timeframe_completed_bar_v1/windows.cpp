// Copyright (c) 2026 Waajacu. See LICENSE at repository root.
// January 1m CACHE88 -> one instrument per row, channels 1m / 1h / 1d.
// Does not modify the historical three-instrument window builder.
// This builder invents 1h and 1d bars from one 1-minute cache. The source
// that replaces it is the imported loader in code/dataloaders/numeric_series/.
// Do not train on this output.
#include "embedding/data/observation_windows.h"
#include "cuwacunu/artifacts/v1/sidecar.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <type_traits>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace ow = embedding::data::observation_windows;
namespace artifacts = cuwacunu::artifacts::v1;
constexpr int64_t kInterval = 60000, kHistory = 128, kTrainBegin = 1704067259999LL,
    kTrainEnd = 1705881599999LL, kValidationBegin = 1705881659999LL,
    kValidationEnd = 1706486399999LL, kCacheRows = 44640, kCacheBytes = 3928320;
constexpr int64_t kHour = 60, kDay = 1440, kRowScale = 1000000000000000LL;
constexpr double kPriceFloor = 1e-12, kActivityFloor = -1.0 + 1e-12;
const std::array<int64_t, 3> kChannelIds{4, 9, 15};
const std::array<int64_t, 3> kSpans{1, kHour, kDay};
const std::array<const char *, 3> kAssets{"BTCUSDT", "ETHUSDT", "SOLUSDT"};
const std::array<std::string, 3> kHashes{
    "67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58",
    "0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa",
    "364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7"};

void check(bool value, const std::string &message) { TORCH_CHECK(value, "[timeframe windows] ", message); }

#pragma pack(push, 1)
struct Record {
  int64_t open_time;
  double open_price, high_price, low_price, close_price, volume;
  int64_t close_time;
  double quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume;
};
#pragma pack(pop)
static_assert(sizeof(Record) == 88 && offsetof(Record, close_time) == 48 &&
              std::endian::native == std::endian::little && std::is_trivially_copyable_v<Record>);

struct Bar {
  bool ok{false};
  double value[9]{};
};

struct Cache {
  std::vector<Bar> minutes;
  std::array<std::vector<Bar>, 3> bars;
  explicit Cache(const fs::path &path, const std::string &pin) {
    std::ifstream in(path, std::ios::binary);
    check(bool(in), "open cache");
    std::string bytes(kCacheBytes, '\0');
    in.read(bytes.data(), kCacheBytes);
    check(in.gcount() == kCacheBytes && !in.bad(), "read complete cache");
    check(artifacts::sha256_bytes(bytes) == pin, "loaded cache buffer differs from the pin");
    minutes.resize(static_cast<size_t>(kCacheRows));
    for (int64_t i = 0; i < kCacheRows; ++i) {
      Record record;
      std::memcpy(&record, bytes.data() + 88 * i, sizeof(record));
      check(record.close_time == kTrainBegin + i * kInterval, "exact original minute lattice required");
      const double values[9]{record.open_price, record.high_price, record.low_price, record.close_price,
          record.volume, record.quote_asset_volume, record.number_of_trades, record.taker_buy_base_volume,
          record.taker_buy_quote_volume};
      bool finite = record.open_time != INT64_MIN;
      for (const double value : values) finite = finite && std::isfinite(value);
      finite = finite && values[0] > 0 && values[1] > 0 && values[2] > 0 && values[3] > 0 && values[1] >= values[2];
      for (int f = 4; f < 9; ++f) finite = finite && values[f] >= 0;
      if (!finite) continue;
      auto &minute = minutes[static_cast<size_t>(i)];
      minute.ok = true;
      std::copy(std::begin(values), std::end(values), std::begin(minute.value));
    }
    for (size_t channel = 0; channel < 3; ++channel) bars[channel] = aggregate(kSpans[channel]);
  }
  std::vector<Bar> aggregate(int64_t span) const {
    check(kCacheRows % span == 0, "cache length must contain whole timeframe bars");
    std::vector<Bar> out(static_cast<size_t>(kCacheRows / span));
    for (int64_t bar = 0; bar < kCacheRows / span; ++bar) {
      bool ok = true;
      double high = -std::numeric_limits<double>::infinity(), low = std::numeric_limits<double>::infinity();
      double open = 0, close = 0, sum[5]{};
      for (int64_t step = 0; step < span; ++step) {
        const auto &minute = minutes[static_cast<size_t>(bar * span + step)];
        if (!minute.ok) { ok = false; break; }
        if (step == 0) open = minute.value[0];
        high = std::max(high, minute.value[1]);
        low = std::min(low, minute.value[2]);
        close = minute.value[3];
        for (int f = 0; f < 5; ++f) sum[f] += minute.value[4 + f];
      }
      if (!ok || !(open > 0 && high >= low && low > 0 && close > 0)) continue;
      auto &item = out[static_cast<size_t>(bar)];
      item.ok = true;
      item.value[0] = open;
      item.value[1] = high;
      item.value[2] = low;
      item.value[3] = close;
      for (int f = 0; f < 5; ++f) item.value[4 + f] = sum[f];
    }
    return out;
  }
};

int64_t completed_bar(int64_t minute, int64_t span) {
  if (minute < span - 1) return -1;
  return (minute + 1) / span - 1;
}

bool store_raw(double out[9], const Bar &bar) {
  if (!bar.ok) return false;
  for (int f = 0; f < 9; ++f) {
    if (!std::isfinite(bar.value[f]) || !std::isfinite(static_cast<float>(bar.value[f]))) return false;
    out[f] = bar.value[f];
  }
  return true;
}

bool store_log(double out[9], const Bar &current, const Bar &previous) {
  if (!current.ok || !previous.ok || !(previous.value[3] > 0)) return false;
  const double base = std::max(previous.value[3], kPriceFloor);
  for (int f = 0; f < 4; ++f) {
    if (!(current.value[f] > 0)) return false;
    out[f] = std::log(std::max(current.value[f], kPriceFloor) / base);
  }
  for (int f = 4; f < 9; ++f) {
    if (current.value[f] < kActivityFloor) return false;
    out[f] = std::log1p(std::max(current.value[f], kActivityFloor));
  }
  for (int f = 0; f < 9; ++f)
    if (!std::isfinite(out[f]) || !std::isfinite(static_cast<float>(out[f]))) return false;
  return true;
}

std::vector<int64_t> anchors(int64_t begin, int64_t end, int64_t selection_history, int64_t count, bool nonoverlap) {
  check(begin >= 0 && end >= begin && (end - begin) % kInterval == 0 && selection_history > 0 && count >= 2,
      "selection geometry");
  const auto stride = nonoverlap ? selection_history * kInterval : kInterval;
  const auto first = begin + (selection_history - 1) * kInterval;
  check(first <= end, "complete selected role support");
  const auto available = (end - first) / stride + 1;
  check(count <= available, "enough selected windows");
  std::vector<int64_t> rows;
  for (int64_t i = 0; i < count; ++i)
    rows.push_back(first + static_cast<int64_t>(static_cast<__int128>(i) * (available - 1) / (count - 1)) * stride);
  check(std::set<int64_t>(rows.begin(), rows.end()).size() == rows.size(), "unique anchor times");
  return rows;
}

int64_t make_row_id(int64_t instrument, int64_t close) {
  check(instrument >= 0 && instrument < 3 && close > 0 && close < kRowScale, "row id fields fit");
  return instrument * kRowScale + close;
}

struct Series {
  ow::Dataset data;
  int64_t observed_cells{0};
  std::array<int64_t, 3> channel_cells{};
  std::array<int64_t, 3> flat_steps{};
  std::array<int64_t, 3> comparable_steps{};
};

struct Pair {
  Series log_view, raw_view;
  int64_t movement_rows{0}, relation_rows{0}, rows{0};
};

const Bar *lookup(const Cache &cache, int64_t minute, int64_t channel, int64_t *index) {
  const auto found = completed_bar(minute, kSpans[static_cast<size_t>(channel)]);
  if (index) *index = found;
  if (found < 0 || found >= static_cast<int64_t>(cache.bars[static_cast<size_t>(channel)].size())) return nullptr;
  return &cache.bars[static_cast<size_t>(channel)][static_cast<size_t>(found)];
}

Series blank(int64_t rows, const std::string &source, const std::string &split, const std::string &role) {
  Series out;
  out.data.values = torch::zeros({rows, 3, kHistory, 9}, torch::kFloat64);
  out.data.observed = torch::zeros(out.data.values.sizes(), torch::kBool);
  out.data.timestamps = torch::zeros({rows, kHistory}, torch::kFloat64);
  out.data.channel_ids = torch::tensor(std::vector<int64_t>(kChannelIds.begin(), kChannelIds.end()), torch::kInt64);
  out.data.cutoffs = torch::empty({rows}, torch::kFloat64);
  out.data.row_ids = torch::empty({rows}, torch::kInt64);
  out.data.sampling_interval = double(kInterval);
  out.data.source_identity = source;
  out.data.split_identity = split;
  out.data.role = role;
  check(source.size() <= 1024 && split.size() <= 1024, "identity limit");
  return out;
}

void eligibility(const Series &raw, int64_t *movement, int64_t *relation) {
  const auto values = raw.data.values.accessor<double, 4>();
  const auto observed = raw.data.observed.accessor<bool, 4>();
  int64_t movement_rows = 0, relation_rows = 0;
  for (int64_t row = 0; row < raw.data.rows(); ++row) {
    std::array<std::array<double, 127>, 3> returns{};
    std::array<std::array<char, 127>, 3> valid{};
    std::array<int, 3> counts{};
    for (int64_t c = 0; c < 3; ++c) {
      for (int64_t t = 1; t < kHistory; ++t) {
        const bool left = observed[row][c][t - 1][3] && values[row][c][t - 1][3] > 0;
        const bool right = observed[row][c][t][3] && values[row][c][t][3] > 0;
        if (!left || !right) continue;
        returns[static_cast<size_t>(c)][static_cast<size_t>(t - 1)] = std::log(values[row][c][t][3] / values[row][c][t - 1][3]);
        valid[static_cast<size_t>(c)][static_cast<size_t>(t - 1)] = 1;
        ++counts[static_cast<size_t>(c)];
      }
    }
    if (counts[0] >= 8 && counts[1] >= 8 && counts[2] >= 8) ++movement_rows;
    int pairs = 0;
    for (int c = 0; c < 3; ++c) for (int d = c + 1; d < 3; ++d) {
      double xs[127], ys[127];
      int n = 0;
      for (int t = 0; t < 127; ++t) {
        if (!valid[static_cast<size_t>(c)][static_cast<size_t>(t)] || !valid[static_cast<size_t>(d)][static_cast<size_t>(t)]) continue;
        xs[n] = returns[static_cast<size_t>(c)][static_cast<size_t>(t)];
        ys[n] = returns[static_cast<size_t>(d)][static_cast<size_t>(t)];
        ++n;
      }
      if (n < 8) continue;
      double sx = 0, sy = 0;
      for (int i = 0; i < n; ++i) { sx += xs[i]; sy += ys[i]; }
      sx /= double(n); sy /= double(n);
      double vx = 0, vy = 0;
      for (int i = 0; i < n; ++i) { vx += (xs[i] - sx) * (xs[i] - sx); vy += (ys[i] - sy) * (ys[i] - sy); }
      if (vx / double(n) >= 1e-12 && vy / double(n) >= 1e-12) ++pairs;
    }
    if (pairs >= 2) ++relation_rows;
  }
  *movement = movement_rows;
  *relation = relation_rows;
}

Pair build(const std::vector<Cache> &caches, const std::vector<int64_t> &times, const std::string &base,
    const std::string &split, const std::string &role, int64_t begin, int64_t end) {
  const auto anchor_count = static_cast<int64_t>(times.size());
  const auto rows = anchor_count * 3;
  Pair out;
  out.log_view = blank(rows, base + ";encoder-view=log-return-previous-same-timeframe-close-log1p-v1", split, role);
  out.raw_view = blank(rows, base + ";target-view=raw-completed-bar-v1", split, role);
  out.rows = rows;
  auto lx = out.log_view.data.values.accessor<double, 4>();
  auto lm = out.log_view.data.observed.accessor<bool, 4>();
  auto rx = out.raw_view.data.values.accessor<double, 4>();
  auto rm = out.raw_view.data.observed.accessor<bool, 4>();
  auto lts = out.log_view.data.timestamps.accessor<double, 2>();
  auto rts = out.raw_view.data.timestamps.accessor<double, 2>();
  for (int64_t a = 0; a < anchor_count; ++a) {
    const auto cutoff = times[static_cast<size_t>(a)];
    check(cutoff <= end && cutoff - (kHistory - 1) * kInterval >= begin, "retained minute grid stays inside the role");
    const auto minute0 = (cutoff - kTrainBegin) / kInterval - (kHistory - 1);
    check(minute0 >= 0, "window starts inside the January cache");
    for (int64_t instrument = 0; instrument < 3; ++instrument) {
      const auto row = a * 3 + instrument;
      const auto id = make_row_id(instrument, cutoff);
      out.log_view.data.cutoffs[row] = double(cutoff);
      out.raw_view.data.cutoffs[row] = double(cutoff);
      out.log_view.data.row_ids[row] = id;
      out.raw_view.data.row_ids[row] = id;
      const auto &cache = caches[static_cast<size_t>(instrument)];
      for (int64_t t = 0; t < kHistory; ++t) {
        const auto minute = minute0 + t;
        const auto close = kTrainBegin + minute * kInterval;
        lts[row][t] = double(close);
        rts[row][t] = double(close);
        for (int64_t c = 0; c < 3; ++c) {
          int64_t index = -1;
          const auto *bar = lookup(cache, minute, c, &index);
          double raw[9]{};
          if (bar && store_raw(raw, *bar)) {
            for (int f = 0; f < 9; ++f) rx[row][c][t][f] = raw[f];
            for (int f = 0; f < 9; ++f) rm[row][c][t][f] = true;
            out.raw_view.observed_cells += 9;
            out.raw_view.channel_cells[static_cast<size_t>(c)] += 9;
          }
          const Bar *previous = nullptr;
          if (index >= 1) previous = &cache.bars[static_cast<size_t>(c)][static_cast<size_t>(index - 1)];
          double viewed[9]{};
          if (bar && previous && store_log(viewed, *bar, *previous)) {
            for (int f = 0; f < 9; ++f) lx[row][c][t][f] = viewed[f];
            for (int f = 0; f < 9; ++f) lm[row][c][t][f] = true;
            out.log_view.observed_cells += 9;
            out.log_view.channel_cells[static_cast<size_t>(c)] += 9;
          }
          if (t > 0 && lm[row][c][t][3] && lm[row][c][t - 1][3]) {
            ++out.log_view.comparable_steps[static_cast<size_t>(c)];
            if (lx[row][c][t][3] == lx[row][c][t - 1][3]) ++out.log_view.flat_steps[static_cast<size_t>(c)];
          }
        }
      }
    }
  }
  eligibility(out.raw_view, &out.movement_rows, &out.relation_rows);
  return out;
}

bool same_bits(const torch::Tensor &left, const torch::Tensor &right) {
  if (left.device() != right.device() || left.scalar_type() != right.scalar_type() || left.sizes() != right.sizes())
    return false;
  const auto a = left.contiguous(), b = right.contiguous();
  return std::memcmp(a.const_data_ptr(), b.const_data_ptr(), a.numel() * a.element_size()) == 0;
}

void publish(const fs::path &path, const Series &series) {
  ow::write(path, series.data);
  const auto restored = ow::read(path);
  check(restored.source_identity == series.data.source_identity && restored.role == series.data.role, "reload identity");
  for (const auto &pair : std::vector<std::pair<torch::Tensor, torch::Tensor>>{
           {series.data.values, restored.values}, {series.data.observed, restored.observed},
           {series.data.timestamps, restored.timestamps}, {series.data.channel_ids, restored.channel_ids},
           {series.data.cutoffs, restored.cutoffs}, {series.data.row_ids, restored.row_ids}})
    check(same_bits(pair.first, pair.second), "exact archive reload");
}

struct Args {
  fs::path output;
  std::array<fs::path, 3> caches{};
  explicit Args(int argc, char **argv) {
    std::vector<fs::path> paths;
    std::vector<std::string> hashes;
    check(argc > 1 && (argc - 1) % 2 == 0, "--cache PATH --cache-sha256 SHA three times, ordered BTC ETH SOL, --output NEWLEAF");
    for (int i = 1; i < argc; i += 2) {
      const std::string key = argv[i], value = argv[i + 1];
      if (key == "--output") { check(output.empty(), "duplicate output"); output = value; }
      else if (key == "--cache") paths.emplace_back(value);
      else if (key == "--cache-sha256") hashes.push_back(value);
      else check(false, "unknown option " + key);
    }
    check(paths.size() == 3 && hashes.size() == 3 && !output.empty(), "three caches and one new output");
    output = fs::absolute(output).lexically_normal();
    check(!fs::exists(output) && fs::is_directory(output.parent_path()), "new output leaf in an existing parent");
    for (size_t c = 0; c < 3; ++c) {
      check(hashes[c] == kHashes[c], "cache pin/order must match the January BTC ETH SOL files");
      caches[c] = fs::canonical(paths[c]);
      check(fs::is_regular_file(caches[c]) && fs::file_size(caches[c]) == kCacheBytes, "complete cache required");
      check(artifacts::sha256_file(caches[c]) == hashes[c], "cache SHA256 mismatch before read");
    }
  }
};

std::string json_series(const Series &series) {
  std::ostringstream out;
  out << "{\"observed_cells\":" << series.observed_cells << ",\"channel_cells\":[";
  for (size_t c = 0; c < 3; ++c) { if (c) out << ','; out << series.channel_cells[c]; }
  out << "],\"flat_close_steps\":[";
  for (size_t c = 0; c < 3; ++c) { if (c) out << ','; out << series.flat_steps[c]; }
  out << "],\"comparable_close_steps\":[";
  for (size_t c = 0; c < 3; ++c) { if (c) out << ','; out << series.comparable_steps[c]; }
  out << "]}";
  return out.str();
}
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "timeframe_windows --cache PATH --cache-sha256 SHA (three each, BTC ETH SOL) --output NEWLEAF\n"
                << "One instrument per row. Channels are torch ordinals 4, 9, 15 (1m, 1h, 1d).\n"
                << "Anchor grid is 1 minute. Coarse channels hold the last completed bar.\n";
      return 0;
    }
    const Args args(argc, argv);
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    std::vector<Cache> caches;
    for (size_t c = 0; c < 3; ++c) caches.emplace_back(args.caches[c], kHashes[c]);
    const int64_t valid_minutes = static_cast<int64_t>(std::count_if(caches[0].minutes.begin(), caches[0].minutes.end(),
        [](const Bar &bar) { return bar.ok; }));
    check(valid_minutes == kCacheRows, "January BTC minute cache is complete and positive");
    check(caches[0].bars[1][0].ok && caches[0].bars[1][0].value[3] == caches[0].minutes[59].value[3],
        "first completed hour closes on minute 59");
    check(caches[0].bars[2][0].ok && caches[0].bars[2][0].value[0] == caches[0].minutes[0].value[0],
        "first completed day opens on minute 0");
    const auto &hour1 = caches[0].bars[1][1];
    const auto &hour0 = caches[0].bars[1][0];
    check(hour1.ok && hour0.ok, "first two BTC hours aggregate");
    const double hour_log = std::log(hour1.value[3] / hour0.value[3]);
    double probe[9]{};
    check(store_log(probe, hour1, hour0) && probe[3] == hour_log, "hour log-return uses the previous hour close");

    const std::string base = "timeframe-completed-bar-v1;population=january-2024-mkt-hist-2;one-instrument-per-row;"
        "channel-ids=4,9,15;keys=1m,1h,1d;anchor-ms=60000;history=128;rule=last-completed-causal-bar;"
        "row-id=instrument*1000000000000000+close;caches=BTCUSDT:" + kHashes[0] + ",ETHUSDT:" + kHashes[1] +
        ",SOLUSDT:" + kHashes[2];
    const auto train_anchors = anchors(kTrainBegin, kTrainEnd, 128, 1024, false);
    const auto probe_train = anchors(kTrainBegin, kTrainEnd, 256, 112, true);
    const auto probe_validation = anchors(kValidationBegin, kValidationEnd, 256, 32, true);
    auto split_of = [](const std::string &role, int64_t begin, int64_t end, int64_t selection, int64_t count, bool nonoverlap) {
      return "timeframe-completed-bar-v1;role=" + role + ";begin=" + std::to_string(begin) + ";end=" + std::to_string(end) +
          ";selection-history=" + std::to_string(selection) + ";retained=128;" + (nonoverlap ? "nonoverlap" : "overlap") +
          ";anchors=" + std::to_string(count) + ";instruments=3;order=anchor-then-BTC-ETH-SOL";
    };
    const auto calibration = build(caches, train_anchors, base, split_of("TRAIN", kTrainBegin, kTrainEnd, 128, 1024, false),
        "TRAIN", kTrainBegin, kTrainEnd);
    const auto train_probe = build(caches, probe_train, base, split_of("TRAIN", kTrainBegin, kTrainEnd, 256, 112, true),
        "TRAIN", kTrainBegin, kTrainEnd);
    const auto validation_probe = build(caches, probe_validation, base,
        split_of("VALIDATION", kValidationBegin, kValidationEnd, 256, 32, true), "VALIDATION", kValidationBegin, kValidationEnd);
    check(fs::create_directory(args.output), "exclusive output leaf");
    const fs::path log_dir = args.output / "log", raw_dir = args.output / "raw";
    publish(log_dir / "calibration-TRAIN.pt", calibration.log_view);
    publish(log_dir / "probe-TRAIN.pt", train_probe.log_view);
    publish(log_dir / "probe-VALIDATION.pt", validation_probe.log_view);
    publish(raw_dir / "probe-TRAIN.pt", train_probe.raw_view);
    publish(raw_dir / "probe-VALIDATION.pt", validation_probe.raw_view);
    std::ostringstream json;
    json << std::setprecision(17) << "{\"kind\":\"timeframe_completed_bar_window_v1\",\"status\":\"passed\""
         << ",\"population\":\"january-2024-mkt-hist-2\",\"instruments\":[\"BTCUSDT\",\"ETHUSDT\",\"SOLUSDT\"]"
         << ",\"channel_ids\":[4,9,15],\"channel_keys\":[\"1m\",\"1h\",\"1d\"],\"anchor_ms\":60000,\"history\":128"
         << ",\"rule\":\"last completed 1m, 1h or 1d bar at or before the anchor minute\""
         << ",\"encoder_view\":\"log(price/previous same-timeframe close), log1p(volume fields)\""
         << ",\"target_view\":\"raw completed-bar prices; targets are computed from this archive\""
         << ",\"row_id\":\"instrument index * 1000000000000000 + close time\",\"promotes_reference\":false"
         << ",\"btc_first_hour_log_close\":" << hour_log
         << ",\"valid_btc_minutes\":" << valid_minutes
         << ",\"calibration\":{\"rows\":" << calibration.rows << ",\"anchors\":1024,\"log\":" << json_series(calibration.log_view)
         << ",\"movement_rows\":" << calibration.movement_rows << ",\"relation_rows\":" << calibration.relation_rows << "}"
         << ",\"probe_train\":{\"rows\":" << train_probe.rows << ",\"anchors\":112,\"log\":" << json_series(train_probe.log_view)
         << ",\"raw\":" << json_series(train_probe.raw_view) << ",\"movement_rows\":" << train_probe.movement_rows
         << ",\"relation_rows\":" << train_probe.relation_rows << "}"
         << ",\"probe_validation\":{\"rows\":" << validation_probe.rows << ",\"anchors\":32,\"log\":"
         << json_series(validation_probe.log_view) << ",\"raw\":" << json_series(validation_probe.raw_view)
         << ",\"movement_rows\":" << validation_probe.movement_rows << ",\"relation_rows\":" << validation_probe.relation_rows
         << "}}";
    const auto text = args.output / "preparation.json";
    std::ofstream report(text);
    check(bool(report), "open preparation report");
    report << json.str() << '\n';
    report.close();
    check(bool(report), "write preparation report");
    std::cout << json.str() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
