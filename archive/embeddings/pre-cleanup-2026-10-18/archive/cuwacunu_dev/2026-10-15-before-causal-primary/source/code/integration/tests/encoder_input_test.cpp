// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#include "cuwacunu/integration/encoder_input.h"
#include "cuwacunu/data/dataloader.h"
#include "cuwacunu/data/record_types.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace {
namespace data = cuwacunu::data;
namespace enc = cuwacunu::encoder;
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
    const auto pattern = (std::filesystem::temp_directory_path() / "cuwacunu-integration-XXXXXX").string();
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
  for (int t = 0; t < 160; ++t) {
    // Exercise one absent candle without altering the aligned timestamp grid.
    if (channel == 2 && t == 12) continue;
    const auto key = int64_t(1700000000) + t;
    const double price = 100 + 4 * std::sin(t * .24 + channel * .3) + std::cos(t * .61);
    const double volume = 20 + 3 * std::cos(t * .19 - channel * .2);
    data::kline_t row{};
    row.open_time = key - 1; row.close_time = key;
    row.open_price = price - .2; row.close_price = price;
    row.high_price = price + 1; row.low_price = price - 1;
    row.volume = volume; row.quote_asset_volume = volume * price;
    row.number_of_trades = 20; row.taker_buy_base_volume = volume * .4;
    row.taker_buy_quote_volume = row.quote_asset_volume * .4;
    row.to_csv(csv, ',');
    csv << '\n';
  }
  check(csv.good(), "fixture CSV write");
}
} // namespace

int main() {
  try {
    check(torch::cuda::is_available(), "CUDA required for encoder integration witness");
    torch::set_num_threads(1);
    TemporaryDirectory fixture;
    std::vector<std::filesystem::path> files;
    for (int c = 0; c < 3; ++c) {
      files.push_back(fixture.path / ("channel-" + std::to_string(c) + ".csv"));
      write_channel(files.back(), c);
    }
    data::MemoryMappedEdgeDataset<data::kline_t> dataset;
    for (const auto &path : files) dataset.add_dataset(path.string(), 32, 2, "none", false);
    auto loader = data::edge_dataloader_t<data::kline_t>(dataset, {.batch_size = 2, .workers = 0});
    auto iterator = loader.begin();
    check(iterator != loader.end(), "nonempty mmap loader");
    auto batch = data::sample_t::collate_fn(*iterator);
    check(batch.features.sizes() == torch::IntArrayRef({2, 3, 32, 9}), "kline batch retains nine features");
    const auto config = enc::fixed_config();
    const auto ids = torch::tensor({0, 1, 2}, torch::kInt64);
    // Explicit fixture mapping: open, close, volume. This declares no market recipe.
    const std::array<int64_t, 3> columns{0, 3, 4};
    auto input = integration::encoder_input(batch, columns, ids, config);
    check(input.data.sizes() == torch::IntArrayRef({2, 3, 32, 3}), "explicit selected feature shape");
    check(torch::equal(input.observed.select(3, 0), batch.mask), "row mask expands across features");
    check(!input.observed.all().item<bool>(), "absent candle remains masked");

    auto changed_future = batch;
    changed_future.future_features = torch::full_like(batch.future_features, 999999.);
    changed_future.future_keys = torch::full_like(batch.future_keys, -999999.);
    auto unchanged = integration::encoder_input(changed_future, columns, ids, config);
    check(torch::equal(input.data, unchanged.data) && torch::equal(input.observed, unchanged.observed) &&
              torch::equal(input.endpoints, unchanged.endpoints), "future never enters encoder input");
    auto shifted_grid = batch;
    shifted_grid.past_keys = batch.past_keys.clone();
    shifted_grid.past_keys.select(1, 1).add_(1);
    rejects([&] { integration::encoder_input(shifted_grid, columns, ids, config); }, "unaligned channels rejected");
    auto preprocessed = batch; preprocessed.normalized = true;
    rejects([&] { integration::encoder_input(preprocessed, columns, ids, config); }, "double normalization rejected");
    rejects([&] { integration::encoder_input(batch, {0, 0, 4}, ids, config); }, "duplicate feature selection rejected");

    auto scaler = enc::fit_train_scaler(input, config);
    auto normalized = scaler.transform(input, config);
    enc::Encoder model(config);
    model->eval();
    torch::NoGradGuard no_grad;
    auto native = enc::native_embedding(model->encode(normalized), config);
    check(native.is_cuda() && native.sizes() == torch::IntArrayRef({2, 32}) &&
              torch::isfinite(native).all().item<bool>(), "loader-to-CUDA native32");

    // Loader history is independently configurable; this larger window is not
    // accepted by the unchanged v19 encoder and is not a capacity benchmark.
    data::MemoryMappedEdgeDataset<data::kline_t> longer;
    for (const auto &path : files) longer.add_dataset(path.string(), 128, 0, "none", false);
    const auto large = data::sample_t::collate_fn({longer.get(0), longer.get(1)});
    check(large.features.sizes() == torch::IntArrayRef({2, 3, 128, 9}), "loader supports a separately requested longer history");
    rejects([&] { integration::encoder_input(large, columns, ids, config); }, "unchanged encoder rejects larger history");
    std::cout << "Integration passed: mmap nine-feature windows, masks, explicit selection, past-only input, aligned grids, CUDA native32; longer loader history independently checked\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
