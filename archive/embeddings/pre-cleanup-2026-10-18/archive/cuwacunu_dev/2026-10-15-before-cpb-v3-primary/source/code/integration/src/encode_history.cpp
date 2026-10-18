// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include "cuwacunu/integration/causal_market_input.h"
#include <torch/cuda.h>
#include <array>
#include <charconv>
#include <iomanip>
#include <iostream>

namespace {
namespace primary = cuwacunu::encoder::primary;
namespace integration = cuwacunu::integration;
namespace fs = std::filesystem;
struct Arguments {
  fs::path settings{"config/encoder.conf"}, runtime{"config/encoder_runtime.conf"};
  std::array<fs::path, 3> sources;
  int64_t cutoff{0};
};
constexpr const char *usage =
    "Usage: encode_history --cutoff CLOSE_TIME_MS --source BTC_RAW_CACHE_OR_CSV "
    "--source ETH_RAW_CACHE_OR_CSV --source SOL_RAW_CACHE_OR_CSV "
    "[--settings config/encoder.conf] [--runtime config/encoder_runtime.conf]\n"
    "Sources must follow semantic IDs 0,1,2. CSV inputs require existing fresh raw caches.";
Arguments parse(int argc, char **argv) {
  Arguments result;
  size_t source_count = 0;
  bool cutoff_seen = false, settings_seen = false, runtime_seen = false;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    TORCH_CHECK(i+1 < argc, usage);
    const std::string value = argv[++i];
    if (flag == "--source") {
      TORCH_CHECK(source_count < result.sources.size(), usage);
      result.sources[source_count++] = value;
    } else if (flag == "--cutoff") {
      TORCH_CHECK(!cutoff_seen, "[encode_history] duplicate cutoff");
      const auto parsed = std::from_chars(value.data(), value.data()+value.size(), result.cutoff);
      TORCH_CHECK(parsed.ec == std::errc{} && parsed.ptr == value.data()+value.size(),
          "[encode_history] cutoff must be an integer candle close time in milliseconds");
      cutoff_seen = true;
    } else if (flag == "--settings") {
      TORCH_CHECK(!settings_seen, "[encode_history] duplicate settings");
      result.settings = value; settings_seen = true;
    } else if (flag == "--runtime") {
      TORCH_CHECK(!runtime_seen, "[encode_history] duplicate runtime");
      result.runtime = value; runtime_seen = true;
    } else TORCH_CHECK(false, usage);
  }
  TORCH_CHECK(cutoff_seen && source_count == result.sources.size(), usage);
  return result;
}
} // namespace

int main(int argc, char **argv) try {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << usage << '\n'; return 0;
  }
  const auto arguments = parse(argc, argv);
  TORCH_CHECK(torch::cuda::is_available(), "[encode_history] selected encoder requires the managed CUDA runtime");
  torch::set_num_threads(1);
  auto loaded = primary::load_selected(arguments.settings, arguments.runtime);
  const auto sample = integration::load_market_history(arguments.sources, arguments.cutoff, loaded.config);
  const auto ids = torch::tensor(loaded.config.level.channel_ids, torch::kInt64);
  const auto cutoffs = torch::tensor({static_cast<double>(arguments.cutoff)}, torch::kFloat64);
  const auto encoded = integration::encoder_input(loaded, sample, ids, cutoffs);
  const auto native = encoded.native.to(torch::kCPU).contiguous();
  TORCH_CHECK(native.sizes() == torch::IntArrayRef({1, primary::native_width}) &&
      torch::isfinite(native).all().item<bool>(), "[encode_history] finite native36 output required");
  std::cout << std::setprecision(9) << "{\"instance\":\"" << primary::instance
      << "\",\"native_width\":" << primary::native_width << ",\"cutoffs\":["
      << static_cast<int64_t>(encoded.cutoffs[0].item<double>()) << "],\"valid\":["
      << (encoded.valid[0].item<bool>() ? "true" : "false") << "],\"native\":[[";
  const auto coordinates = native.accessor<float, 2>();
  for (int64_t d = 0; d < primary::native_width; ++d) {
    if (d) std::cout << ',';
    std::cout << coordinates[0][d];
  }
  std::cout << "]]}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n'; return 1;
}
