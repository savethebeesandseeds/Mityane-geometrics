// SPDX-License-Identifier: MIT
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/preprocessing.h"
#include "shared_test_support.h"

namespace rpb_test {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using test::check;
using test::close;
using test::finite;

inline rpb::Config config() {
  rpb::Config c;
  c.channel_count = 2; c.input_width = 2; c.encoder_width = 16;
  c.export_width = 8; c.num_layers = 1; c.num_heads = 2;
  c.feedforward_width = 32; c.decoder_hidden_width = 24;
  c.channel_ids = {101, 202};
  return c;
}

inline rpb::Input input(const rpb::Config &c, int64_t B = 3) {
  auto t = torch::arange(c.history_length, torch::kFloat64).view({1, 1, c.history_length});
  auto b = torch::arange(B, torch::kFloat64).view({B, 1, 1});
  auto channel = torch::arange(c.channel_count, torch::kFloat64).view({1, c.channel_count, 1});
  auto phase = t * (0.17 + b * 0.03) + channel * 0.5;
  std::vector<torch::Tensor> features;
  for (int64_t f = 0; f < c.input_width; ++f)
    features.push_back(torch::sin(phase * (1.0 + f * 0.3)) * (1.0 + b * 0.2 + channel) + t * 0.01 + channel * 5.0);
  auto x = torch::stack(features, -1);
  return {x, torch::ones_like(x, torch::kBool),
          torch::tensor(rpb::resolved_channel_ids(c), torch::kInt64),
          torch::full({B}, (c.history_length - 1) * c.sampling_interval, torch::kFloat64),
          c.sampling_interval};
}

template <typename Function> void rejects(Function function, const std::string &label) {
  bool rejected = false;
  try { function(); } catch (const std::exception &) { rejected = true; }
  check(rejected, label + " was accepted");
}

} // namespace rpb_test
