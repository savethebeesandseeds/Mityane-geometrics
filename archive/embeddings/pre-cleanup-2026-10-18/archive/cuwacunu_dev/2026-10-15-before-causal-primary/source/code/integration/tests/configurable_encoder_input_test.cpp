// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#include "cuwacunu/integration/configurable_encoder_input.h"
#include "cuwacunu/encoder/configurable_artifact.h"
#include "cuwacunu/data/dataloader.h"
#include "cuwacunu/data/record_types.h"
#include <torch/cuda.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

namespace {
namespace data = cuwacunu::data;
namespace enc = cuwacunu::encoder::configurable;
namespace integration = cuwacunu::integration;

void check(bool condition, const char *message) { TORCH_CHECK(condition, message); }
template <typename Function> void rejects(Function action, const char *message) {
  bool failed = false;
  try { action(); } catch (const std::exception &) { failed = true; }
  check(failed, message);
}

struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    const auto pattern = (std::filesystem::temp_directory_path() / "cuwacunu-configurable-input-XXXXXX").string();
    std::vector<char> bytes(pattern.begin(), pattern.end());
    bytes.push_back('\0');
    const auto directory = ::mkdtemp(bytes.data());
    check(directory != nullptr, "temporary directory allocation");
    path = directory;
  }
  ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};

void write_channel(const std::filesystem::path &path, int channel) {
  std::ofstream csv(path);
  check(csv.good(), "fixture CSV open");
  for (int t = 0; t < 320; ++t) {
    if (channel == 1 && t == 47) continue;
    const auto key = int64_t(1700000000) + t;
    const double price = 100 + 4 * std::sin(t * .24 + channel * .3) + std::cos(t * .61);
    const double volume = 20 + 3 * std::cos(t * .19 - channel * .2);
    data::kline_t row{};
    row.open_time = key - 1; row.close_time = key;
    row.open_price = price - .2; row.close_price = price;
    row.high_price = price + 1; row.low_price = price - 1;
    row.volume = volume; row.quote_asset_volume = volume * price;
    row.number_of_trades = 20 + t % 7;
    row.taker_buy_base_volume = volume * (.35 + .01 * (t % 5));
    row.taker_buy_quote_volume = row.taker_buy_base_volume * price;
    row.to_csv(csv, ',');
    csv << '\n';
  }
  check(csv.good(), "fixture CSV write");
}

void run_case(const std::vector<std::filesystem::path> &files, int64_t channels,
              int64_t width) {
  auto config = enc::read_settings("config/encoder_v20.conf");
  config.channel_count = channels;
  config.history_length = 129;
  config.export_width = width;
  config.channel_ids.clear();
  for (int64_t c = 0; c < channels; ++c) config.channel_ids.push_back(10 + 10 * c);
  enc::validate_config(config);
  const auto ids = torch::tensor(config.channel_ids, torch::kInt64);
  data::MemoryMappedEdgeDataset<data::kline_t> dataset;
  for (int64_t c = 0; c < channels; ++c)
    dataset.add_dataset(files[c].string(), config.history_length, 2, "none", false);
  auto loader = data::edge_dataloader_t<data::kline_t>(dataset, {.batch_size = 2, .workers = 0});
  auto iterator = loader.begin();
  check(iterator != loader.end(), "nonempty mmap loader");
  auto batch = data::sample_t::collate_fn(*iterator);
  check(batch.features.sizes() == torch::IntArrayRef({2, channels, config.history_length, 9}),
        "loader history comes from encoder configuration and retains nine features");
  auto input = integration::configurable_encoder_input(batch, ids, config);
  check(input.data.sizes() == batch.features.sizes(), "adapter keeps every kline coordinate");
  check(torch::equal(input.observed, batch.mask.unsqueeze(-1).expand_as(batch.features)),
        "finite source row support expands to all nine features");
  check(!input.observed.all().item<bool>(), "absent candle remains masked");
  for (int64_t f = 0; f < data::kKlineFeatureWidth; ++f) {
    const auto expected = torch::where(batch.mask, batch.features.select(3, f),
                                       torch::zeros_like(batch.features.select(3, f)));
    check(torch::equal(input.data.select(3, f), expected),
          "price and activity feature values pass without selection or reduction");
  }

  auto nonfinite = batch;
  nonfinite.features = batch.features.clone();
  nonfinite.mask = batch.mask.clone();
  nonfinite.features.index_put_({0, 0, 10, 8}, std::numeric_limits<double>::quiet_NaN());
  nonfinite.mask.index_put_({0, 1, 11}, false);
  nonfinite.features.index_put_({0, 1, 11, 4}, std::numeric_limits<double>::infinity());
  auto sanitized = integration::configurable_encoder_input(nonfinite, ids, config);
  check(!sanitized.observed.index({0, 0, 10, 8}).item<bool>() &&
            sanitized.data.index({0, 0, 10, 8}).item<double>() == 0,
        "a nonfinite source coordinate is masked and zeroed");
  check(sanitized.observed.index({0, 0, 10, 7}).item<bool>(),
        "finite siblings of a nonfinite coordinate stay observed");
  check(!sanitized.observed.index({0, 1, 11}).any().item<bool>() &&
            torch::isfinite(sanitized.data).all().item<bool>(),
        "hidden nonfinite rows cannot contaminate arithmetic");

  auto changed_future = batch;
  changed_future.future_features = torch::full_like(batch.future_features, 999999.);
  changed_future.future_keys = torch::full_like(batch.future_keys, -999999);
  changed_future.future_mask = torch::logical_not(batch.future_mask);
  auto unchanged = integration::configurable_encoder_input(changed_future, ids, config);
  check(torch::equal(input.data, unchanged.data) && torch::equal(input.observed, unchanged.observed) &&
            torch::equal(input.endpoints, unchanged.endpoints), "future never enters encoder input");
  auto shifted = batch;
  shifted.past_keys = batch.past_keys.clone();
  shifted.past_keys.select(1, 1).add_(1);
  rejects([&] { integration::configurable_encoder_input(shifted, ids, config); },
          "different channel timestamp grids rejected");
  auto irregular = batch;
  irregular.past_keys = batch.past_keys.clone();
  irregular.past_keys.select(2, 20).add_(1);
  rejects([&] { integration::configurable_encoder_input(irregular, ids, config); },
          "nonuniform common grid rejected");
  auto preprocessed = batch; preprocessed.normalized = true;
  rejects([&] { integration::configurable_encoder_input(preprocessed, ids, config); },
          "double normalization rejected");
  auto wrong_width = config; wrong_width.input_width = 3;
  rejects([&] { integration::configurable_encoder_input(batch, ids, wrong_width); },
          "silent reduction of kline features rejected");
  rejects([&] { integration::configurable_encoder_input(batch, torch::zeros_like(ids), config); },
          "duplicate semantic channel IDs rejected");
  auto wrong_history = config; wrong_history.history_length = 128;
  rejects([&] { integration::configurable_encoder_input(batch, ids, wrong_history); },
          "loader and encoder history mismatch rejected");

  auto scaler = enc::fit_train_scaler(sanitized, config);
  auto normalized = enc::transform(scaler, sanitized, config);
  check(normalized.data.is_cuda() && normalized.data.size(2) == 129,
        "scaler transfers logical nondivisible history to CUDA without padding public input");
  enc::Encoder model(config);
  model->eval();
  torch::NoGradGuard no_grad;
  const auto native = enc::native_embedding(model->encode(normalized), config);
  check(native.is_cuda() && native.sizes() == torch::IntArrayRef({2, width}) &&
            torch::isfinite(native).all().item<bool>(), "mmap all-feature batch to configured CUDA native embedding");
}
} // namespace

int main() {
  try {
    check(torch::cuda::is_available(), "CUDA required for configurable kline integration witness");
    torch::set_num_threads(1);
    TemporaryDirectory fixture;
    std::vector<std::filesystem::path> files;
    for (int c = 0; c < 3; ++c) {
      files.push_back(fixture.path / ("channel-" + std::to_string(c) + ".csv"));
      write_channel(files.back(), c);
    }
    run_case(files, 2, 32);
    run_case(files, 3, 64);
    std::cout << "Configurable integration passed on CUDA: all nine kline features, C2/C3, H129, D32/D64, row/per-value masks, raw-only scaling, past-only input, aligned regular grids\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
