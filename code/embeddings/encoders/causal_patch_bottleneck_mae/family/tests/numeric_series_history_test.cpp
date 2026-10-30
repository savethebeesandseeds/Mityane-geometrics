// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// One synthetic instrument. Channels are 1m and 15m. No market population and no training run.
#include "embedding/encoders/causal_patch_bottleneck_mae/family/numeric_series_history.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include "sha256.h"

#include <torch/cuda.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
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

template <class F>
void refuses_with(F call, const char *fragment, const char *message) {
  bool caught = false;
  try {
    call();
  } catch (const std::exception &error) {
    caught = std::string(error.what()).find(fragment) != std::string::npos;
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
         "  PATCH_LENGTH = 1;\n"
         "  LEVEL_EMBEDDING_SIZE = 8;\n"
         "  EMBEDDING_SIZE = 12;\n"
         "  ENCODER_WIDTH = 16;\n"
         "  NUM_LAYERS = 1;\n"
         "  NUM_HEADS = 2;\n"
         "  FEEDFORWARD_WIDTH = 32;\n"
         "  DECODER_HIDDEN_WIDTH = 16;\n"
         "  SAMPLING_INTERVAL = 60000;\n"
         "  CHANNEL_BAR_COUNTS = 17, 3;\n"
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
  const auto settings = dir / "encoder.conf";
  const auto config = read_fixture(settings);
  const auto counts = cp::channel_bar_counts(
      cp::implementation::configuration::Document::read(settings).block("ENCODER"), config);
  check(counts.size() == 2 && counts[0] == 17 && counts[1] == 3, "CHANNEL_BAR_COUNTS is one count per channel");

  std::vector<double> minute_closes;
  const int64_t minute_first = cutoff - 17 * minute;
  for (int i = 0; i < 18; ++i) minute_closes.push_back(1000. + i);
  std::vector<double> quarter_closes{1990., 2000., 2010., 2030.};
  const auto minute_path = dir / "ONE-1m.cache.bin";
  const auto quarter_path = dir / "ONE-15m.cache.bin";
  write_cache(minute_path, minute_closes, minute_first, minute);
  write_cache(quarter_path, quarter_closes, cutoff - 3 * quarter, quarter);

  data::raw_cache_channel_t minute_channel;
  minute_channel.path = minute_path.string();
  minute_channel.sha256 = data::sha256_file_hex(minute_path.string());
  minute_channel.input_length = 18;
  minute_channel.future_length = 0;
  data::raw_cache_channel_t quarter_channel = minute_channel;
  quarter_channel.path = quarter_path.string();
  quarter_channel.sha256 = data::sha256_file_hex(quarter_path.string());
  quarter_channel.input_length = 4;
  auto edge = data::open_raw_caches<data::kline_t>({minute_channel, quarter_channel}, torch::kFloat64);
  const auto sample = edge.get_by_anchor_key(cutoff);
  check(sample.normalized == false, "raw caches stay raw");
  check(sample.features.size(0) == 2 && sample.features.size(2) == 9, "two kline channels");

  const std::vector<int64_t> ids{4, 7};
  const auto batch = cp::history_from_edge_samples(config, {sample, sample}, ids, true);
  check(batch.history.data.sizes() == torch::IntArrayRef({2, 2, history, 9}), "batched encoder history");
  check(batch.history.observed.sizes() == batch.history.data.sizes(), "matching observation mask");
  check(batch.history.timestamps.sizes() == torch::IntArrayRef({2, 2, history}), "per-channel timestamps");
  check(batch.cutoffs.sizes() == torch::IntArrayRef({2}), "one cutoff per instrument");
  const auto values = batch.history.data.accessor<double, 4>();
  const auto observed = batch.history.observed.accessor<bool, 4>();
  const auto times = batch.history.timestamps.accessor<double, 3>();
  for (int64_t tick = 0; tick < history; ++tick) {
    const auto key = static_cast<double>(cutoff - (16 - tick) * minute);
    check(times[0][0][tick] == key && times[1][0][tick] == key, "minute slot is that minute bar");
    const auto minute_view = view_of(1000. + (tick + 1), 1000. + tick);
    for (int64_t feature = 0; feature < 9; ++feature) {
      check(observed[0][0][tick][feature], "minute bar is observed");
      check(values[0][0][tick][feature] == minute_view[static_cast<std::size_t>(feature)],
            "minute log-return view");
    }
  }
  check(batch.cutoffs[0].item<double>() == static_cast<double>(cutoff), "cutoff is the latest valid bar");
  const auto first_kept = view_of(2000., 1990.);
  const auto earlier = view_of(2010., 2000.);
  const auto last = view_of(2030., 2010.);
  for (int64_t tick = 0; tick < 14; ++tick) {
    check(!std::isfinite(times[0][1][tick]), "the settings count of three 15m bars leaves one whole empty stretch on the left");
    for (int64_t feature = 0; feature < 9; ++feature)
      check(!observed[0][1][tick][feature] && values[0][1][tick][feature] == 0.,
            "the omitted list leaves the other 15m slots empty");
  }
  check(times[0][1][14] == static_cast<double>(cutoff - 2 * quarter), "15m slot 14 is the first observed return");
  check(times[0][1][15] == static_cast<double>(cutoff - quarter), "15m slot 15 is the middle observed return");
  check(times[0][1][16] == static_cast<double>(cutoff), "15m slot 16 is the bar that closes at the cutoff");
  for (int64_t feature = 0; feature < 9; ++feature) {
    check(observed[0][1][14][feature] && values[0][1][14][feature] == first_kept[static_cast<std::size_t>(feature)],
          "the first kept 15m bar returns against the trimmed predecessor");
    check(observed[0][1][15][feature] && values[0][1][15][feature] == earlier[static_cast<std::size_t>(feature)],
          "the middle 15m bar returns against the immediately previous cache row");
    check(observed[0][1][16][feature] && values[0][1][16][feature] == last[static_cast<std::size_t>(feature)],
          "the omitted settings list keeps the latest 15m return");
  }

  const std::vector<int64_t> three_quarters{17, 3};
  const auto counted = cp::history_from_edge_samples(config, {sample}, ids, true, three_quarters);
  check(counted.history.observed[0][1].sum().item<int64_t>() == 27, "an explicit count of three keeps three observed 15m bars");
  check(!std::isfinite(counted.history.timestamps[0][1][13].item<double>()), "the trimmed predecessor is a pad");
  check(counted.history.timestamps[0][1][14].item<double>() == static_cast<double>(cutoff - 2 * quarter),
        "the first kept 15m bar is the first observed return");
  check(counted.history.timestamps[0][1][16].item<double>() == static_cast<double>(cutoff),
        "the kept 15m bar is the latest one");
  check(counted.history.data[0][1][14][3].item<double>() == first_kept[3], "the first kept close returns against the predecessor");
  check(counted.history.data[0][1][16][3].item<double>() == last[3], "the kept 15m close is the latest view");
  refuses_with([&] { (void)cp::history_from_edge_samples(config, {sample}, ids, true, std::vector<int64_t>{17, 1}); },
               "at least 3 groups", "one 15m bar is fewer than three groups");
  refuses_with([&] { (void)cp::boundary::retain(batch.history, batch.cutoffs, config, std::vector<int64_t>{17, 1}); },
               "at least 3 groups", "retain refuses a channel count of one group");

  const auto raw = cp::history_from_edge_samples(config, {sample}, ids, false);
  check(raw.history.data[0][0][16][3].item<double>() == 1017., "raw close is the cache close");
  auto normalized = sample;
  normalized.normalized = true;
  refuses([&] { (void)cp::history_from_edge_samples(config, {normalized}, ids, true); },
          "a second log-return view is refused");
  refuses([&] { (void)cp::history_from_edge_samples(config, {sample}, {7, 4}, true); },
          "channel order must match the configuration");
  {
    std::ofstream empty(dir / "empty.conf");
    empty << "ENCODER {\n"
             "  MODEL_TAG = CPB-family;\n"
             "  CHANNEL_COUNT = 2;\n"
             "  CHANNEL_IDS = 4, 7;\n"
             "  HISTORY_LENGTH = 17;\n"
             "  INPUT_WIDTH = 9;\n"
             "  PATCH_LENGTH = 1;\n"
             "  LEVEL_EMBEDDING_SIZE = 8;\n"
             "  EMBEDDING_SIZE = 12;\n"
             "  ENCODER_WIDTH = 16;\n"
             "  NUM_LAYERS = 1;\n"
             "  NUM_HEADS = 2;\n"
             "  FEEDFORWARD_WIDTH = 32;\n"
             "  DECODER_HIDDEN_WIDTH = 16;\n"
             "  SAMPLING_INTERVAL = 60000;\n"
             "  CHANNEL_BAR_COUNTS = ;\n"
             "  DEVICE = cuda:0;\n"
             "}\n";
  }
  refuses([&] { (void)cp::read_settings(dir / "empty.conf"); }, "an empty CHANNEL_BAR_COUNTS list is refused");
  refuses([&] { (void)cp::history_from_edge_samples(config, {sample}, ids, true, std::vector<int64_t>{}); },
          "an explicit empty bar-count list is refused");
  refuses([&] { (void)cp::boundary::retain(batch.history, batch.cutoffs, config, std::vector<int64_t>{}); },
          "retain refuses an explicit empty bar-count list");

  {
    std::ofstream absent(dir / "absent.conf");
    absent << "ENCODER {\n"
              "  MODEL_TAG = CPB-family;\n"
              "  CHANNEL_COUNT = 2;\n"
              "  CHANNEL_IDS = 4, 7;\n"
              "  HISTORY_LENGTH = 17;\n"
              "  INPUT_WIDTH = 9;\n"
              "  PATCH_LENGTH = 1;\n"
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
  }
  const auto absent_config = cp::read_settings(dir / "absent.conf");
  check(absent_config.channel_bar_counts == std::vector<int64_t>{17, 17},
        "an absent CHANNEL_BAR_COUNTS key stores HISTORY_LENGTH on every channel");
  const auto wide = cp::history_from_edge_samples(absent_config, {sample}, ids, true);
  const auto wide_times = wide.history.timestamps.accessor<double, 3>();
  const auto wide_values = wide.history.data.accessor<double, 4>();
  const auto wide_observed = wide.history.observed.accessor<bool, 4>();
  check(!std::isfinite(wide_times[0][1][12]), "absent-key history pads before the predecessor and the three 15m returns");
  check(wide_times[0][1][13] == static_cast<double>(cutoff - 3 * quarter), "absent-key history keeps the masked predecessor at its own finite time");
  check(wide_times[0][1][14] == static_cast<double>(cutoff - 2 * quarter), "absent-key history keeps the first observed 15m return");
  check(wide_times[0][1][15] == static_cast<double>(cutoff - quarter), "absent-key history keeps the middle 15m bar");
  check(wide_times[0][1][16] == static_cast<double>(cutoff), "absent-key history keeps the last 15m bar");
  for (int64_t feature = 0; feature < 9; ++feature) {
    check(!wide_observed[0][1][13][feature] && wide_values[0][1][13][feature] == 0.,
          "the absent-key default masks the predecessor and does not store its close");
    check(wide_observed[0][1][14][feature] &&
              wide_values[0][1][14][feature] == first_kept[static_cast<std::size_t>(feature)],
          "the absent-key default returns the next bar against that predecessor");
    check(wide_observed[0][1][15][feature] &&
              wide_values[0][1][15][feature] == earlier[static_cast<std::size_t>(feature)],
          "the absent-key default returns the middle bar against its previous cache row");
    check(wide_observed[0][1][16][feature] &&
              wide_values[0][1][16][feature] == last[static_cast<std::size_t>(feature)],
          "the absent-key default keeps three observed 15m returns plus the masked predecessor");
  }

  data::edge_sample_t gap;
  constexpr int64_t gap_rows = 4;
  gap.features = torch::zeros({2, gap_rows, 9}, torch::kFloat64);
  gap.mask = torch::zeros({2, gap_rows}, torch::kBool);
  gap.past_keys = torch::zeros({2, gap_rows}, torch::kInt64);
  gap.normalized = false;
  auto gap_features = gap.features.accessor<double, 3>();
  auto gap_mask = gap.mask.accessor<bool, 2>();
  auto gap_keys = gap.past_keys.accessor<int64_t, 2>();
  for (int64_t row = 0; row < gap_rows; ++row) {
    gap_mask[0][row] = true;
    gap_keys[0][row] = cutoff - (3 - row) * minute;
    for (int64_t feature = 0; feature < 9; ++feature) gap_features[0][row][feature] = 1000. + static_cast<double>(row);
    for (int64_t feature = 4; feature < 9; ++feature) gap_features[0][row][feature] = 1.;
  }
  const int64_t quarter_keys[4] = {cutoff - 3 * quarter, cutoff - 2 * quarter, cutoff - quarter, cutoff};
  const double gap_quarter_closes[4] = {2000., 50., 2010., 2030.};
  const bool quarter_valid[4] = {true, false, true, true};
  for (int64_t row = 0; row < gap_rows; ++row) {
    gap_mask[1][row] = quarter_valid[row];
    gap_keys[1][row] = quarter_keys[row];
    for (int64_t feature = 0; feature < 9; ++feature) gap_features[1][row][feature] = gap_quarter_closes[row];
    if (!quarter_valid[row]) continue;
    for (int64_t feature = 4; feature < 9; ++feature) gap_features[1][row][feature] = 1.;
  }
  const auto gapped = cp::history_from_edge_samples(config, {gap}, ids, true, std::vector<int64_t>{17, 4});
  const auto gap_out_times = gapped.history.timestamps.accessor<double, 3>();
  const auto gap_out_values = gapped.history.data.accessor<double, 4>();
  const auto gap_out_observed = gapped.history.observed.accessor<bool, 4>();
  const double close_after = 2030.;
  const double close_of_the_masked_post_hole_bar = 2010.;
  const auto followed = view_of(close_after, close_of_the_masked_post_hole_bar);
  check(!std::isfinite(gap_out_times[0][1][12]), "four on-grid rows right-align at ticks 13 through 16");
  check(gap_out_times[0][1][13] == static_cast<double>(cutoff - 3 * quarter), "row 0 keeps its own finite close time");
  check(gap_out_times[0][1][14] == static_cast<double>(cutoff - 2 * quarter), "the missing bar keeps its own finite close time");
  check(gap_out_times[0][1][15] == static_cast<double>(cutoff - quarter), "the bar after the hole does not move into the hole's index");
  check(gap_out_times[0][1][16] == static_cast<double>(cutoff), "the bar after the masked post-hole bar closes at the cutoff");
  for (int64_t feature = 0; feature < 9; ++feature) {
    check(!gap_out_observed[0][1][13][feature] && gap_out_values[0][1][13][feature] == 0.,
          "row 0 is masked and its value is 0");
    check(!gap_out_observed[0][1][14][feature] && gap_out_values[0][1][14][feature] == 0.,
          "the missing row is masked and its value is 0");
    check(!gap_out_observed[0][1][15][feature] && gap_out_values[0][1][15][feature] == 0.,
          "the bar after the hole is masked and its value is 0");
    check(gap_out_observed[0][1][16][feature] &&
              gap_out_values[0][1][16][feature] == followed[static_cast<std::size_t>(feature)],
          "the following bar returns against the masked post-hole bar");
  }
  check(gap_out_values[0][1][16][3] == std::log(close_after / close_of_the_masked_post_hole_bar), "log(close_after / close_of_the_masked_post_hole_bar)");

  const auto retained = cp::boundary::retain(batch.history, batch.cutoffs, config);
  check(retained.raw.data.sizes() == torch::IntArrayRef({2, 2, history, 9}), "encoder retains the packed window");
  check(retained.times.sizes() == torch::IntArrayRef({2, 2, history}), "retained times stay per channel");
  check(!std::isfinite(retained.times[0][1][0].item<double>()), "retain keeps the 15m pad");
  check(retained.times[0][0][16].item<double>() == static_cast<double>(cutoff), "retain keeps the last minute bar");
  auto shared = batch.history;
  shared.timestamps = batch.history.timestamps.select(1, 0).contiguous();
  refuses([&] { (void)cp::boundary::retain(shared, batch.cutoffs, config); }, "a shared timestamp row is refused");
  const auto trimmed = cp::boundary::retain(batch.history, batch.cutoffs, config, three_quarters);
  check(trimmed.raw.observed[0][1].sum().item<int64_t>() == 27, "retain honors the settings count of three observed 15m bars");
  check(trimmed.times[0][1][16].item<double>() == static_cast<double>(cutoff), "retain keeps the latest counted bar");
  const auto trimmed_wide = cp::boundary::retain(wide.history, wide.cutoffs, absent_config, three_quarters);
  check(trimmed_wide.raw.observed[0][1].sum().item<int64_t>() == 27,
        "an explicit count of three drops the masked predecessor and keeps three observed bars");
  check(!std::isfinite(trimmed_wide.times[0][1][13].item<double>()), "the override drops the masked predecessor");
  check(trimmed_wide.times[0][1][14].item<double>() == static_cast<double>(cutoff - 2 * quarter),
        "the first kept 15m bar is the first observed return");
  check(trimmed_wide.times[0][1][16].item<double>() == static_cast<double>(cutoff),
        "the override keeps the latest 15m bar");
  const auto patch_counts = cp::patches_containing_a_real_bar(batch.history.timestamps, config);
  check(patch_counts[0][0].item<int64_t>() == 17 && patch_counts[0][1].item<int64_t>() == 3,
        "history 17 and patch length 1 give seventeen minute groups and three 15m groups");
  const auto masks = tr::make_mask(batch.history, batch.cutoffs, config, 5, 0);
  const auto eligible = masks.eligible_channels.accessor<bool, 2>();
  check(eligible[0][0] && eligible[1][0] && eligible[0][1] && eligible[1][1],
        "exactly three filled groups stay in the loss, and a larger count stays too");
  check(masks.target[0][1].sum().item<int64_t>() == 9 && masks.target[1][1].sum().item<int64_t>() == 9,
        "exactly three filled groups hide one and leave two visible");
  check(masks.visible[0][1].sum().item<int64_t>() == 18 && masks.visible[1][1].sum().item<int64_t>() == 18,
        "exactly three filled groups leave two groups visible");
  check(masks.target[0][0].sum().item<int64_t>() == 36 && masks.target[1][0].sum().item<int64_t>() == 36,
        "a larger filled-group count still masks");
  tr::PreprocessingIdentity identity{"synthetic-numeric-series-no-market-population", "TRAIN-only-fixture",
                                     "numeric-series-log-return-view-v1"};
  tr::PreprocessingFitter fitter(config, identity);
  fitter.add(batch.history, batch.cutoffs);
  fitter.add(wide.history, wide.cutoffs);
  const auto fitted = fitter.freeze();
  auto instance = cp::fresh(config, fitted.scaler, fitted.bank, 221119);
  check(instance.model->config().channel_bar_counts == three_quarters,
        "fresh construction keeps the settings bar-count list");
  const auto encoding = instance.model->encode_at(batch.history, batch.cutoffs, fitted.scaler);
  check(encoding.native.is_cuda() && encoding.native.size(0) == 2 &&
            torch::isfinite(encoding.native).all().item<bool>() && encoding.valid.all().item<bool>(),
        "finite CUDA forward on the numeric-series window");
  const auto wide_encoding = instance.model->encode_at(wide.history, wide.cutoffs, fitted.scaler);
  check(wide_encoding.native.size(0) == 1 && torch::isfinite(wide_encoding.native).all().item<bool>(),
        "encode_at applies the settings bar counts to a longer history");
  const auto checkpoint = dir / "roundtrip.pt";
  cp::save_checkpoint(checkpoint.string(), instance.model, fitted.scaler, 0);
  const auto loaded = cp::load_checkpoint(checkpoint.string(), config.level.device, &config);
  check(loaded.config.channel_bar_counts == three_quarters, "checkpoint round-trip keeps the settings list");
  check(loaded.model->config().channel_bar_counts == three_quarters, "the loaded encoder uses the stored list");
  tr::OptimizerConfig recipe;
  auto optimizer = tr::make_optimizer(instance.model, recipe);
  tr::TrainingProgress progress{0, 0, 11, 12, 13};
  instance.model->train();
  const auto step = tr::train_step(instance.model, fitted.scaler, *optimizer, progress, batch.history, batch.cutoffs, recipe);
  check(step.updated && progress.completed_updates == 1 && progress.attempted_batches == 1,
        "train_step honors the settings list and trains both channels");

  fs::remove_all(dir);
  std::cout << "numeric-series history passed\n";
  return 0;
}
