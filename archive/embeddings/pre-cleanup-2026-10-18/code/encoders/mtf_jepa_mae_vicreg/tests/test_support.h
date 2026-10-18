#pragma once

#include <torch/torch.h>
#include "shared_test_support.h"

#include <stdexcept>
#include <string>

namespace test {

template <typename Config> Config small_config() {
  Config config{};
  config.channel_count = 2;
  config.history_length = 16;
  config.input_width = 3;
  config.d_model = 12;
  config.latent_dim = 8;
  config.projector_dim = 12;
  config.predictor_hidden_dim = 16;
  config.num_encoder_layers = 1;
  config.num_predictor_layers = 1;
  config.num_decoder_layers = 1;
  config.num_heads = 2;
  config.dropout = 0.0;
  config.time_scales = {4, 8};
  config.scale_strides = {4, 4};
  config.frequency_num_bins = 4;
  config.mask_ratio_time = 0.4;
  config.mask_ratio_frequency = 0.3;
  config.mask_ratio_channel = 0.0;
  config.min_context_ratio = 0.25;
  return config;
}

// Varied deterministic signals give VICReg a non-degenerate batch.
inline torch::Tensor input() {
  const auto time = torch::arange(16, torch::kFloat32).view({1, 1, 16});
  const auto batch = torch::arange(4, torch::kFloat32).view({4, 1, 1});
  const auto channel = torch::arange(2, torch::kFloat32).view({1, 2, 1});
  const auto phase = time * (0.25 + batch * 0.04) + channel * 0.4;
  return torch::stack({torch::sin(phase) * (1.0 + batch * 0.2),
                       torch::cos(phase * 1.7),
                       (time / 15.0 + batch * 0.1 + channel * 0.2)},
                      -1);
}

inline bool teacher(const std::string &name) {
  return name.rfind("target_tokenizer.", 0) == 0 ||
         name.rfind("target_encoder.", 0) == 0;
}

} // namespace test
