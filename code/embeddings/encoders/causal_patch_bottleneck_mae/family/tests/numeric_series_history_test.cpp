// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// One synthetic instrument. Channels are 1m and 15m. No market population and no training run.
#include "embedding/encoders/causal_patch_bottleneck_mae/family/numeric_series_history.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include "sha256.h"

#include <torch/cuda.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

namespace cp = embedding::encoders::causal_patch_bottleneck_mae::family;
namespace tr = cp::training;
namespace data = mityane::data;
namespace fs = std::filesystem;

namespace {

void check(bool condition, const char *message) {
  TORCH_CHECK(condition, "[cpb numeric-series] ", message);
}

template <class F>
void refuses(F call, const char *message) {
  bool caught = false;
  try {
    call();
  } catch (const std::exception &) {
    caught = true;
  }
  check(caught, message);
}

fs::path temp_dir() {
  auto dir = fs::temp_directory_path() /
             ("mityane_cpb_numeric_series_" + std::to_string(static_cast<long long>(::getpid())));
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void write_cache(const fs::path &path, const std::vector<double> &closes, int64_t first, int64_t step) {
  std::ofstream out(path, std::ios::binary);
  TORCH_CHECK(out, "[cpb numeric-series] cannot write cache");
  for (std::size_t i = 0; i < closes.size(); ++i) {
    data::kline_cache_t row{};
    row.close_time = first + static_cast<int64_t>(i) * step;
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
  TORCH_CHECK(out, "[cpb numeric-series] cache write failed");
}

std::vector<double> view_of(double current, double previous) {
  data::kline_cache_t now{};
  data::kline_cache_t before{};
  now.open_time = 1;
  now.close_time = 2;
  before = now;
  for (double *price : {&now.open_price, &now.high_price, &now.low_price, &now.close_price})
    *price = current;
  for (double *price : {&before.open_price, &before.high_price, &before.low_price, &before.close_price})
    *price = previous;
  now.volume = now.quote_asset_volume = now.number_of_trades = now.taker_buy_base_volume =
      now.taker_buy_quote_volume = 1;
  before.volume = before.quote_asset_volume = before.number_of_trades = before.taker_buy_base_volume =
      before.taker_buy_quote_volume = 1;
  return data::kline_cache_t::normalize_log_returns(now, &before).tensor_features();
}

cp::Config read_fixture(const fs::path &path) {
  std::ofstream out(path);
  out << "ENCODER {\n"
         "  MODEL_TAG = CPB-family;\n"
         "  CHANNEL_COUNT = 2;\n"
         "  CHANNEL_IDS = 4, 7;\n"
         "  HISTORY_LENGTH = 17;\n"
         "  INPUT_WIDTH = 9;\n"
         "  PATCH_LENGTH = 4;\n"
         "  LEVEL_EMBEDDING_SIZE = 8;\n"
         "  EMBEDDING_SIZE = 12;\n"
         "  ENCODER_WIDTH = 16;\n"
         "  NUM_LAYERS = 1;\n"
         "  NUM_HEADS = 2;\n"
         "  FEEDFORWARD_WIDTH = 32;\n"
         "  DECODER_HIDDEN_WIDTH = 16;\n"
         "  SAMPLING_INTERVAL = 60000;\n"
         "  DEVICE = cuda:0;\n"
         "}\n";
  TORCH_CHECK(out, "[cpb numeric-series] cannot write settings");
  out.close();
  return cp::read_settings(path);
}

} // namespace

int main() {
  TORCH_CHECK(torch::cuda::is_available(), "[cpb numeric-series] CUDA is required for the encoder forward");
  const auto dir = temp_dir();
  constexpr int64_t minute = 60000;
  constexpr int64_t quarter = 900000;
  constexpr int64_t cutoff = 1800000000000;
  constexpr int64_t history = 17;
  const auto config = read_fixture(dir / "encoder.conf");

  std::vector<double> minute_closes;
  const int64_t minute_first = cutoff - 17 * minute;
  for (int i = 0; i < 18; ++i) minute_closes.push_back(1000. + i);
  std::vector<double> quarter_closes{2000., 2010., 2030.};
  const auto minute_path = dir / "ONE-1m.cache.bin";
  const auto quarter_path = dir / "ONE-15m.cache.bin";
  write_cache(minute_path, minute_closes, minute_first, minute);
  write_cache(quarter_path, quarter_closes, cutoff - 2 * quarter, quarter);

  data::raw_cache_channel_t minute_channel;
  minute_channel.path = minute_path.string();
  minute_channel.sha256 = data::sha256_file_hex(minute_path.string());
  minute_channel.input_length = 18;
  minute_channel.future_length = 0;
  data::raw_cache_channel_t quarter_channel = minute_channel;
  quarter_channel.path = quarter_path.string();
  quarter_channel.sha256 = data::sha256_file_hex(quarter_path.string());
  quarter_channel.input_length = 3;
  auto edge = data::open_raw_caches<data::kline_t>({minute_channel, quarter_channel}, torch::kFloat64);
  const auto sample = edge.get_by_anchor_key(cutoff);
  check(sample.normalized == false, "raw caches stay raw");
  check(sample.features.size(0) == 2 && sample.features.size(2) == 9, "two kline channels");

  const std::vector<int64_t> ids{4, 7};
  const auto batch = cp::history_from_edge_samples(config, {sample, sample}, ids, true);
  check(batch.history.data.sizes() == torch::IntArrayRef({2, 2, history, 9}), "batched encoder history");
  check(batch.history.observed.sizes() == batch.history.data.sizes(), "matching observation mask");
  check(batch.cutoffs.sizes() == torch::IntArrayRef({2}), "one cutoff per instrument");
  const auto values = batch.history.data.accessor<double, 4>();
  const auto observed = batch.history.observed.accessor<bool, 4>();
  const auto times = batch.history.timestamps.accessor<double, 2>();
  for (int64_t tick = 0; tick < history; ++tick) {
    const auto key = static_cast<double>(cutoff - (16 - tick) * minute);
    check(times[0][tick] == key && times[1][tick] == key, "shared anchor clock");
    const auto minute_view = view_of(1000. + (tick + 1), 1000. + tick);
    for (int64_t feature = 0; feature < 9; ++feature) {
      check(observed[0][0][tick][feature], "minute bar is observed");
      check(values[0][0][tick][feature] == minute_view[static_cast<std::size_t>(feature)],
            "minute log-return view");
    }
  }
  check(batch.cutoffs[0].item<double>() == static_cast<double>(cutoff), "cutoff is the last anchor");
  for (int64_t feature = 0; feature < 9; ++feature)
    check(!observed[0][1][0][feature], "15m bar without a previous bar stays masked");
  const auto held = view_of(2010., 2000.);
  const auto last = view_of(2030., 2010.);
  for (int64_t tick = 1; tick <= 15; ++tick) {
    for (int64_t feature = 0; feature < 9; ++feature) {
      check(observed[0][1][tick][feature], "held 15m bar is observed");
      check(values[0][1][tick][feature] == held[static_cast<std::size_t>(feature)],
            "15m bar is held between its own closes");
    }
  }
  for (int64_t feature = 0; feature < 9; ++feature)
    check(observed[0][1][16][feature] &&
              values[0][1][16][feature] == last[static_cast<std::size_t>(feature)],
          "the anchor tick uses the 15m bar that just closed");

  const auto raw = cp::history_from_edge_samples(config, {sample}, ids, false);
  check(raw.history.data[0][0][16][3].item<double>() == 1017., "raw close is the cache close");
  auto normalized = sample;
  normalized.normalized = true;
  refuses([&] { (void)cp::history_from_edge_samples(config, {normalized}, ids, true); },
          "a second log-return view is refused");
  refuses([&] { (void)cp::history_from_edge_samples(config, {sample}, {7, 4}, true); },
          "channel order must match the configuration");

  const auto retained = cp::boundary::retain(batch.history, batch.cutoffs, config);
  check(retained.raw.data.sizes() == torch::IntArrayRef({2, 2, history, 9}), "encoder retains the shared window");
  tr::PreprocessingIdentity identity{"synthetic-numeric-series-no-market-population", "TRAIN-only-fixture",
                                     "numeric-series-log-return-view-v1"};
  tr::PreprocessingFitter fitter(config, identity);
  fitter.add(batch.history, batch.cutoffs);
  const auto fitted = fitter.freeze();
  auto instance = cp::fresh(config, fitted.scaler, fitted.bank, 221119);
  const auto encoding = instance.model->encode_at(batch.history, batch.cutoffs, fitted.scaler);
  check(encoding.native.is_cuda() && encoding.native.size(0) == 2 &&
            torch::isfinite(encoding.native).all().item<bool>() && encoding.valid.all().item<bool>(),
        "finite CUDA forward on the numeric-series window");

  fs::remove_all(dir);
  std::cout << "numeric-series history passed\n";
  return 0;
}
