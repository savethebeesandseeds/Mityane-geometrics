// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Artificial primary-selection witness; no market evaluation or training.
#include "cuwacunu/encoder/primary_encoder.h"
#include "cuwacunu/encoder/configurable_artifact.h"
#include "cuwacunu/integration/configurable_encoder_input.h"
#include "cuwacunu/data/dataloader.h"
#include "cuwacunu/data/record_types.h"
#include <torch/cuda.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string_view>
#include <unistd.h>

namespace {
namespace enc = cuwacunu::encoder::primary;
namespace v20 = cuwacunu::encoder::configurable;
namespace data = cuwacunu::data;
namespace integration = cuwacunu::integration;
namespace fs = std::filesystem;
static_assert(std::string_view(enc::model_tag) == "RPB-v22");
void check(bool condition, const char *message) {
  TORCH_CHECK(condition, "[primary encoder] ", message);
}
template<class F> void rejects(F action, const char *message) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  check(rejected, message);
}
bool exact(const torch::Tensor &a, const torch::Tensor &b) {
  const auto x = a.detach().to(torch::kCPU).contiguous();
  const auto y = b.detach().to(torch::kCPU).contiguous();
  return x.scalar_type() == y.scalar_type() && x.sizes() == y.sizes() &&
      (!x.numel() || !std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.numel()*x.element_size()));
}
struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    const auto pattern = (fs::temp_directory_path()/"cuwacunu-primary-input-XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end()); name.push_back('\0');
    const auto created = ::mkdtemp(name.data()); check(created != nullptr, "private loader/checkpoint fixture");
    path = created;
  }
  ~TemporaryDirectory() { std::error_code error; fs::remove_all(path,error); }
};
void write_channel(const fs::path &path, int channel) {
  std::ofstream csv(path); check(csv.good(), "fixture CSV open");
  for (int t = 0; t < 320; ++t) {
    if (channel == 1 && t == 47) continue;
    const auto key = int64_t(1700000000) + t;
    const double price = 100 + 4*std::sin(t*.24 + channel*.3) + std::cos(t*.61);
    const double volume = 20 + 3*std::cos(t*.19 - channel*.2);
    data::kline_t row{};
    row.open_time = key-1; row.close_time = key;
    row.open_price = price-.2; row.close_price = price;
    row.high_price = price+1; row.low_price = price-1;
    row.volume = volume; row.quote_asset_volume = volume*price;
    row.number_of_trades = 20+t%7;
    row.taker_buy_base_volume = volume*(.35+.01*(t%5));
    row.taker_buy_quote_volume = row.taker_buy_base_volume*price;
    row.to_csv(csv, ','); csv << '\n';
  }
  check(csv.good(), "fixture CSV write");
}
} // namespace

int main(int argc, char **argv) try {
  check(torch::cuda::is_available(), "actual CUDA required"); torch::set_num_threads(1);
  check(argc == 2, "explicit primary settings path required");
  const auto c = enc::read_settings(argv[1]);
  check(c.channel_count == 3 && c.history_length == 128 && c.input_width == 9 &&
            c.export_width == 32 && c.sampling_interval == 1.,
        "selected primary fixture geometry is C3/H128/F9/D32 on the one-second grid");
  rejects([&] { v20::read_settings(argv[1]); }, "v20 settings reader refuses primary v22 tag");
  rejects([&] { enc::read_settings("config/encoder_v20.conf"); }, "primary settings reader refuses old v20 tag");
  TemporaryDirectory fixture;
  data::MemoryMappedEdgeDataset<data::kline_t> dataset;
  for (int channel = 0; channel < c.channel_count; ++channel) {
    const auto file = fixture.path/("channel-"+std::to_string(channel)+".csv");
    write_channel(file,channel);
    dataset.add_dataset(file.string(),c.history_length,2,"none",false);
  }
  auto loader = data::edge_dataloader_t<data::kline_t>(dataset,{.batch_size=2,.workers=0});
  auto iterator = loader.begin(); check(iterator != loader.end(), "real mmap loader is nonempty");
  auto batch = data::sample_t::collate_fn(*iterator);
  check(!batch.normalized && batch.features.sizes() == torch::IntArrayRef({2,3,c.history_length,9}),
        "selected history loads every raw kline feature");
  const auto ids = torch::tensor(c.channel_ids,torch::kInt64);
  const auto input = integration::configurable_encoder_input(batch,ids,c);
  check(exact(input.observed,batch.mask.unsqueeze(-1).expand_as(batch.features)) &&
            !input.observed.all().item<bool>(), "loader candle gap remains an observed mask");
  for (int64_t f = 0; f < data::kKlineFeatureWidth; ++f)
    check(exact(input.data.select(3,f),torch::where(batch.mask,batch.features.select(3,f),
                                                 torch::zeros_like(batch.features.select(3,f)))),
          "raw registry coordinates reach primary without selection or reduction");
  auto nonfinite = batch; nonfinite.features = batch.features.clone();
  nonfinite.features[0][0][10][8] = std::numeric_limits<double>::quiet_NaN();
  const auto sanitized = integration::configurable_encoder_input(nonfinite,ids,c);
  check(!sanitized.observed[0][0][10][8].item<bool>() && sanitized.observed[0][0][10][7].item<bool>() &&
            sanitized.data[0][0][10][8].item<double>() == 0.,
        "per-value missingness masks only the nonfinite coordinate");
  auto future = nonfinite;
  future.future_features = torch::full_like(batch.future_features,std::numeric_limits<double>::infinity());
  future.future_mask = torch::zeros_like(batch.future_mask);
  future.future_keys = torch::full_like(batch.future_keys,-999.);
  const auto unchanged = integration::configurable_encoder_input(future,ids,c);
  check(exact(sanitized.data,unchanged.data) && exact(sanitized.observed,unchanged.observed) &&
            exact(sanitized.endpoints,unchanged.endpoints), "future fields cannot enter primary encoder input");
  const auto scaler = enc::fit_train_scaler(sanitized,c);
  const auto scaled = enc::transform(scaler,sanitized,c);
  check(scaled.data.is_cuda() && scaled.observed.is_cuda(), "one configured transfer to CUDA");
  torch::manual_seed(220014); enc::Encoder model(c); model->eval(); torch::NoGradGuard no_grad;
  check(model->relation_projection && model->relation_projection->weight.sizes() == torch::IntArrayRef({20,1944}),
        "primary constructs the content-only v22 relation projection");
  const auto native = enc::native_embedding(model->encode(scaled),c);
  check(native.is_cuda() && native.sizes() == torch::IntArrayRef({2,c.export_width}) &&
            torch::isfinite(native).all().item<bool>(), "mmap primary loader serves finite native CUDA values");
  check(exact(native,model->base->encode(scaled).z_contextual_global), "zero-start primary preserves its v20 child bytes");
  auto poisoned = scaled;
  poisoned.data = torch::where(scaled.observed,scaled.data,
      torch::full_like(scaled.data,std::numeric_limits<float>::quiet_NaN()));
  check(exact(native,enc::native_embedding(model->encode(poisoned),c)), "masked storage cannot alter native primary values");
  auto missing = poisoned; missing.observed = torch::zeros_like(scaled.observed);
  const auto empty = model->encode(missing);
  check(empty.z_contextual_global.eq(0).all().item<bool>() && !empty.sample_valid_mask.any().item<bool>(),
        "allmissing primary output remains finite invalid zero");
  const auto file = fixture.path/"primary-v22.pt";
  enc::save_checkpoint(file.string(),model,scaler,0);
  rejects([&] { enc::save_checkpoint(file.string(),model,scaler,0); }, "primary checkpoint destination is immutable");
  auto loaded = enc::load_checkpoint(file.string(),c.device,&c);
  const auto restored = enc::transform(loaded.scaler,sanitized,loaded.config);
  check(loaded.training_updates == 0 && loaded.scaler.identity() == scaler.identity() &&
            exact(native,enc::native_embedding(loaded.model->encode(restored),loaded.config)) &&
            exact(model->relation_projection->weight,loaded.model->relation_projection->weight),
        "selected geometry/scaler/native/v22 weights round trip through the primary typed loader");
  rejects([&] { v20::load_checkpoint(file.string(),c.device,&c); }, "historical typed loader refuses primary v22 artifact");
  const auto legacy = fixture.path/"explicit-v20.pt";
  v20::save_checkpoint(legacy.string(),model->base,scaler,0);
  rejects([&] { enc::load_checkpoint(legacy.string(),c.device,&c); }, "primary typed loader refuses historical v20 artifact");
  torch::cuda::synchronize();
  std::cout << "PASS primary RPB-v22: selected file settings, real mmap raw9 loader, row/per-value masks, future exclusion, "
      "native CUDA encode and immutable typed reload; artificial initialized fixture, no training/head/quality execution\n";
  return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
