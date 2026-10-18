// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Selected-artifact integration proof; no encoder training or quality head fits.
#include "cuwacunu/integration/causal_market_input.h"
#include <torch/cuda.h>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string_view>
#include <unistd.h>

namespace {
namespace primary = cuwacunu::encoder::primary;
namespace integration = cuwacunu::integration;
namespace data = cuwacunu::data;
namespace fs = std::filesystem;
int64_t checks = 0, cuda_calls = 0, suffix_cases = 0;
constexpr int64_t first_close = 1704067259999LL, rows = 160, interval = 60000;
void check(bool success, const char *message) {
  ++checks; TORCH_CHECK(success, "[causal integration test] ", message);
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
      (!x.nbytes() || std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.nbytes()) == 0);
}
template<class T> concept AllowsImplicitCutoff = requires(T &loaded,
    const data::sample_t &sample, const torch::Tensor &ids) {
  integration::encoder_input(loaded, sample, ids);
};
static_assert(!AllowsImplicitCutoff<primary::LoadedEncoder>);
static_assert(std::string_view(primary::model_tag) == "CPB-v3");
static_assert(std::string_view(primary::version::kInitializationKind) ==
    "fresh-seeded-constructor-no-import-v1");
struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    auto pattern = (fs::temp_directory_path()/"cuwacunu-causal-input-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end()); buffer.push_back('\0');
    const auto created = ::mkdtemp(buffer.data());
    check(created != nullptr, "private fixture directory"); path = created;
  }
  ~TemporaryDirectory() { std::error_code error; fs::remove_all(path, error); }
};
data::kline_cache_t record(int64_t channel, int64_t t) {
  const std::array<double, 3> prices{42000., 2200., 100.};
  const double price = prices[channel]*(1.+.00003*std::sin(t*.17+channel*.4)) + .0000001234567;
  data::kline_cache_t out{};
  out.close_time = first_close+t*interval;
  out.open_time = (channel == 1 && t == 47) ? INT64_MIN : out.close_time-interval+1;
  out.open_price = price-.03; out.high_price = price+.2;
  out.low_price = price-.2; out.close_price = price;
  out.volume = 3.+channel*10.+std::cos(t*.19)*.5;
  out.quote_asset_volume = out.volume*price;
  out.number_of_trades = 30.+t%11;
  out.taker_buy_base_volume = out.volume*.43;
  out.taker_buy_quote_volume = out.taker_buy_base_volume*price;
  return out;
}
std::array<fs::path, 3> write_caches(const fs::path &directory) {
  std::array<fs::path, 3> paths;
  for (int64_t channel = 0; channel < 3; ++channel) {
    paths[channel] = directory/("channel-"+std::to_string(channel)+".cache.bin");
    std::ofstream file(paths[channel], std::ios::binary);
    check(file.good(), "CACHE88 fixture open");
    for (int64_t t = 0; t < rows; ++t) {
      const auto value = record(channel, t);
      file.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
    check(file.good(), "CACHE88 fixture complete");
  }
  return paths;
}
data::sample_t long_history(const std::array<fs::path, 3> &paths) {
  std::vector<torch::Tensor> values, masks, keys;
  for (int64_t channel = 0; channel < 3; ++channel) {
    data::MemoryMappedDataset<data::kline_t> dataset(paths[channel].string(), rows, 0, torch::kFloat64);
    const auto sample = dataset.get_edge_sample_at_anchor_key(first_close+(rows-1)*interval, rows, 0);
    check(sample.features.scalar_type() == torch::kFloat64, "raw cache loads directly to F64");
    const auto access = sample.features.accessor<double, 2>();
    for (int64_t t = 0; t < rows; ++t) {
      const auto expected = record(channel, t).tensor_features();
      for (int64_t f = 0; f < 9; ++f)
        check(std::bit_cast<uint64_t>(access[t][f]) == std::bit_cast<uint64_t>(expected[f]),
            "generic mmap preserves original CACHE88 double bits");
    }
    values.push_back(sample.features); masks.push_back(sample.mask); keys.push_back(sample.past_keys);
  }
  data::sample_t sample;
  sample.features = torch::stack(values).unsqueeze(0);
  sample.mask = torch::stack(masks).unsqueeze(0);
  sample.past_keys = torch::stack(keys).unsqueeze(0);
  sample.future_features = torch::full({1, 3, 5, 9}, 17., torch::kFloat64);
  sample.future_mask = torch::ones({1, 3, 5}, torch::kBool);
  sample.future_keys = torch::zeros({1, 3, 5}, torch::kInt64);
  return sample;
}
primary::Encoding encode(primary::LoadedEncoder &loaded, const data::sample_t &sample,
    const torch::Tensor &ids, int64_t index) {
  ++cuda_calls;
  return integration::encoder_input(loaded, sample, ids,
      torch::tensor({static_cast<double>(first_close+index*interval)}, torch::kFloat64));
}
void same_encoding(const primary::Encoding &a, const primary::Encoding &b, const char *message) {
  check(exact(a.native,b.native) && exact(a.valid,b.valid) && exact(a.cutoffs,b.cutoffs), message);
}
} // namespace

int main(int argc, char **argv) try {
  check(argc == 1 || argc == 3, "usage: test_causal_encoder_input [SETTINGS RUNTIME]");
  check(torch::cuda::is_available(), "actual CUDA required"); torch::set_num_threads(1);
  const fs::path settings = argc == 3 ? argv[1] : "config/encoder.conf";
  const fs::path runtime = argc == 3 ? argv[2] : "config/encoder_runtime.conf";
  auto loaded = primary::load_selected(settings, runtime);
  const auto selection = primary::read_runtime(runtime);
  check(primary::version::boundary::sha256_file(selection.checkpoint) == primary::checkpoint_sha256 &&
      loaded.source_updates == 0 && loaded.model->source_updates() == 0 &&
      loaded.causal_updates == 2048 && loaded.model->causal_updates() == 2048 &&
      loaded.initialization_seed == 943001 && loaded.model->initialization_seed() == 943001 &&
      loaded.scaler.identity() == primary::input_scaler_identity &&
      loaded.model->normalizer().identity() == primary::bank_normalizer_identity &&
      !loaded.initial_state_identity.empty() &&
      loaded.initial_state_identity == loaded.model->initial_state_identity(),
      "selected exact fresh typed artifact: zero source updates, 2048 causal updates, seed and TRAIN statistics");
  const auto initial_state = loaded.model->state_identity();
  const auto initial_scaler = loaded.scaler.identity();
  const auto initial_bank = loaded.model->normalizer().identity();
  auto wrong_provenance = loaded;
  wrong_provenance.source_updates = 2048;
  rejects([&]{primary::validate_selection(wrong_provenance);},"imported source-update metadata rejected");
  wrong_provenance = loaded; wrong_provenance.causal_updates = 0;
  rejects([&]{primary::validate_selection(wrong_provenance);},"untrained initialization metadata rejected");
  wrong_provenance = loaded; ++wrong_provenance.initialization_seed;
  rejects([&]{primary::validate_selection(wrong_provenance);},"wrong fresh seed metadata rejected");
  wrong_provenance = loaded; wrong_provenance.initial_state_identity = "wrong-state";
  rejects([&]{primary::validate_selection(wrong_provenance);},"wrong fresh initial-state binding rejected");
  TemporaryDirectory fixture;
  const auto paths = write_caches(fixture.path);
  const auto sample = long_history(paths);
  const auto ids = torch::tensor(loaded.config.level.channel_ids, torch::kInt64);
  const auto raw = integration::encoder_input(sample, ids, loaded.config);
  check(raw.data.data_ptr() == sample.features.data_ptr() && exact(raw.data, sample.features),
      "adapter neither casts nor sanitizes raw observations");
  check(exact(raw.observed,sample.mask.unsqueeze(-1).expand_as(sample.features)) &&
      !raw.observed[0][1][47].any().item<bool>(), "missing row remains masked");
  check(raw.data[0][0][0][0].item<double>() !=
      static_cast<double>(static_cast<float>(raw.data[0][0][0][0].item<double>())),
      "fixture distinguishes original F64 from lossy F32 upcast");
  check(exact(raw.timestamps,sample.past_keys.select(1,0).to(torch::kFloat64)),
      "actual common close-time grid reaches version boundary");
  const auto complete = encode(loaded, sample, ids, rows-1);
  check(complete.native.is_cuda() && complete.valid.is_cuda() && complete.valid.all().item<bool>() &&
      complete.native.sizes() == torch::IntArrayRef({1,36}) && torch::isfinite(complete.native).all().item<bool>(),
      "selected artifact produces finite valid native36 on CUDA");
  const auto cutoff = first_close+(rows-1)*interval;
  const auto recent = integration::load_market_history(paths, cutoff, loaded.config);
  // A shorter supplied history has the same absolute endpoint; use that actual time.
  ++cuda_calls;
  const auto recent_encoding = integration::encoder_input(loaded,recent,ids,
      torch::tensor({static_cast<double>(cutoff)},torch::kFloat64));
  same_encoding(complete,recent_encoding,"cache inference path equals causal rolling last128");
  ++cuda_calls;
  const auto direct = loaded.model->encode_at(raw,
      torch::tensor({static_cast<double>(cutoff)},torch::kFloat64),loaded.scaler);
  same_encoding(complete,direct,"stable primary facade equals direct evaluated CPB API");

  for (const int64_t t : {31,63,95,127}) {
    const auto baseline = encode(loaded,sample,ids,t);
    for (int mode = 0; mode < 3; ++mode) {
      auto changed = sample;
      changed.features = sample.features.clone(); changed.mask = sample.mask.clone();
      if (mode != 1) changed.features.narrow(2,t+1,rows-t-1).fill_(std::numeric_limits<double>::quiet_NaN());
      if (mode != 0) changed.mask.narrow(2,t+1,rows-t-1).logical_not_();
      const auto view = integration::encoder_input(changed,ids,loaded.config);
      if (mode != 1) check(std::isnan(view.data[0][0][t+1][0].item<double>()),
          "future NaN is left untouched for cutoff-owned removal");
      same_encoding(baseline,encode(loaded,changed,ids,t),"future values or row masks cannot affect earlier embedding");
      ++suffix_cases;
    }
  }
  auto feature_sample = sample;
  feature_sample.mask = raw.observed.clone();
  feature_sample.features = sample.features.clone();
  feature_sample.mask[0][0][12][2] = false;
  feature_sample.features[0][0][12][2] = std::numeric_limits<double>::infinity();
  const auto feature_encoded = encode(loaded,feature_sample,ids,63);
  auto suffix_feature = feature_sample;
  suffix_feature.features = feature_sample.features.clone(); suffix_feature.mask = feature_sample.mask.clone();
  suffix_feature.features.narrow(2,64,rows-64).fill_(std::numeric_limits<double>::infinity());
  suffix_feature.mask.narrow(2,64,rows-64).logical_not_();
  same_encoding(feature_encoded,encode(loaded,suffix_feature,ids,63),"future per-feature masks are cutoff invariant");
  ++suffix_cases;
  const auto combined = integration::encoder_input(sample,ids,loaded.config,feature_sample.mask);
  check(!combined.observed[0][0][12][2].item<bool>() && combined.observed[0][0][12][1].item<bool>() &&
      !combined.observed[0][1][47].any().item<bool>(), "row and per-feature masks combine without value-derived support");

  auto heterogeneous = sample;
  heterogeneous.features = sample.features.repeat({2,1,1,1});
  heterogeneous.mask = sample.mask.repeat({2,1,1});
  heterogeneous.past_keys = sample.past_keys.repeat({2,1,1});
  const auto batch_ids = ids.unsqueeze(0).repeat({2,1});
  const auto different_cutoffs = torch::tensor({double(first_close+31*interval),double(first_close+95*interval)},torch::kFloat64);
  ++cuda_calls;
  const auto different = integration::encoder_input(loaded,heterogeneous,batch_ids,different_cutoffs);
  heterogeneous.features = heterogeneous.features.clone(); heterogeneous.mask = heterogeneous.mask.clone();
  heterogeneous.features[0].narrow(1,32,rows-32).fill_(std::numeric_limits<double>::quiet_NaN());
  heterogeneous.features[1].narrow(1,96,rows-96).fill_(std::numeric_limits<double>::infinity());
  heterogeneous.mask[0].narrow(1,32,rows-32).logical_not_();
  heterogeneous.mask[1].narrow(1,96,rows-96).logical_not_();
  ++cuda_calls;
  same_encoding(different,integration::encoder_input(loaded,heterogeneous,batch_ids,different_cutoffs),
      "each batch member retains only its own explicit cutoff");
  ++suffix_cases;

  auto future = sample;
  future.future_features = torch::full_like(sample.future_features,std::numeric_limits<double>::infinity());
  future.future_mask = torch::zeros_like(sample.future_mask);
  future.future_keys = torch::full_like(sample.future_keys,-1);
  same_encoding(complete,encode(loaded,future,ids,rows-1),"future sample fields are ignored");
  auto visible_poison = sample; visible_poison.features = sample.features.clone();
  visible_poison.features[0][0][12][2] = std::numeric_limits<double>::quiet_NaN();
  rejects([&]{encode(loaded,visible_poison,ids,63);},"retained observed NaN is rejected, not silently hidden");
  auto hidden_poison = feature_sample;
  hidden_poison.features = feature_sample.features.clone();
  hidden_poison.features[0][0][12][2] = std::numeric_limits<double>::quiet_NaN();
  same_encoding(feature_encoded,encode(loaded,hidden_poison,ids,63),"masked stored Inf/NaN does not enter encoding");
  auto positive = sample; positive.features = sample.features.clone();
  positive.features.narrow(2,0,64).select(3,3).mul_(1.1);
  check(!exact(encode(loaded,sample,ids,63).native,encode(loaded,positive,ids,63).native),
      "positive control: retained observed values affect the embedding");

  auto f32 = sample; f32.features = sample.features.to(torch::kFloat32);
  rejects([&]{integration::encoder_input(f32,ids,loaded.config);},"lossy F32 sample rejected");
  auto normalized = sample; normalized.normalized = true;
  rejects([&]{integration::encoder_input(normalized,ids,loaded.config);},"pre-normalized sample rejected");
  auto misaligned = sample; misaligned.past_keys = sample.past_keys.clone();
  misaligned.past_keys.select(1,1).add_(interval);
  rejects([&]{integration::encoder_input(misaligned,ids,loaded.config);},"channel timestamp misalignment rejected");
  auto irregular = sample; irregular.past_keys = sample.past_keys.clone();
  irregular.past_keys.select(2,12).add_(1);
  rejects([&]{integration::encoder_input(irregular,ids,loaded.config);},"nonuniform actual time grid rejected");
  rejects([&]{integration::encoder_input(sample,torch::zeros_like(ids),loaded.config);},"duplicate channel IDs rejected");
  auto bad_geometry = loaded.config; bad_geometry.level.history_length = 256;
  rejects([&]{integration::encoder_input(sample,ids,bad_geometry);},"unselected geometry rejected");
  rejects([&]{integration::encoder_input(loaded,sample,ids,torch::tensor({double(cutoff+1)},torch::kFloat64));},
      "cutoff between actual closes rejected");
  rejects([&]{integration::encoder_input(loaded,sample,ids,torch::tensor({double(cutoff+interval)},torch::kFloat64));},
      "cutoff beyond supplied history rejected");
  rejects([&]{integration::load_market_history(paths,cutoff-1,loaded.config);},"nearest earlier cache close is not accepted as requested endpoint");
  auto empty = sample; empty.mask = torch::zeros_like(sample.mask);
  const auto invalid = encode(loaded,empty,ids,rows-1);
  check(invalid.native.eq(0).all().item<bool>() && !invalid.valid.any().item<bool>(),"allmissing row yields finite invalid zero");

  const auto bad_settings = fixture.path/"legacy.conf";
  { std::ofstream file(bad_settings); file << "ENCODER { MODEL_TAG = RPB-v22; }\n"; }
  rejects([&]{primary::read_settings(bad_settings);},"legacy model settings rejected");
  const auto wrong_checkpoint = fixture.path/"wrong.pt";
  { std::ofstream file(wrong_checkpoint,std::ios::binary); file << "not the selected checkpoint"; }
  rejects([&]{primary::load_checkpoint(wrong_checkpoint,loaded.config);},"wrong or historical checkpoint bytes rejected before use");
  const auto relative_runtime = fixture.path/"relative.conf";
  { std::ofstream file(relative_runtime); file << "ENCODER_RUNTIME { MODEL_TAG = CPB-v3; INSTANCE = CPB-v3.market-01; CHECKPOINT = wrong.pt; CHECKPOINT_SHA256 = "
      << primary::checkpoint_sha256 << "; }\n"; }
  check(primary::read_runtime(relative_runtime).checkpoint == wrong_checkpoint,"runtime checkpoint path resolves relative to runtime file");
  rejects([&]{primary::load_selected(settings,relative_runtime);},"runtime refuses wrong checkpoint with no fallback");
  const auto legacy_runtime = fixture.path/"legacy-runtime.conf";
  { std::ofstream file(legacy_runtime); file << "ENCODER_RUNTIME { MODEL_TAG = RPB-v22; INSTANCE = RPB-v22.market-02; CHECKPOINT = wrong.pt; CHECKPOINT_SHA256 = "
      << primary::checkpoint_sha256 << "; }\n"; }
  rejects([&]{primary::read_runtime(legacy_runtime);},"legacy runtime model/instance rejected");
  for (const auto historical_model : {"CPB-v1", "CPB-v2"}) {
    const auto historical_runtime = fixture.path/(std::string(historical_model)+"-runtime.conf");
    { std::ofstream file(historical_runtime); file << "ENCODER_RUNTIME { MODEL_TAG = " << historical_model
        << "; INSTANCE = " << historical_model << ".market-01; CHECKPOINT = wrong.pt; CHECKPOINT_SHA256 = "
        << primary::checkpoint_sha256 << "; }\n"; }
    rejects([&]{primary::read_runtime(historical_runtime);},"historical causal family runtime selection rejected");
  }
  primary::validate_selection(loaded);
  check(loaded.model->state_identity() == initial_state && loaded.model->causal_updates() == 2048 &&
      loaded.model->source_updates() == 0 && loaded.scaler.identity() == initial_scaler &&
      loaded.model->normalizer().identity() == initial_bank,
      "selected weights/normalizer/scaler remain unchanged and no training occurs");
  torch::cuda::synchronize();
  std::cout << "PASS CPB-v3.market-01 primary integration: " << checks << " checks, "
      << suffix_cases << " future value/mask cases, " << cuda_calls
      << " CUDA encode attempts; raw CACHE88 F64 parity, explicit close-time cutoffs, native36, "
      "typed fresh checkpoint/config/TRAIN statistics and frozen state; new encoder updates/head fits: 0\n";
  return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
