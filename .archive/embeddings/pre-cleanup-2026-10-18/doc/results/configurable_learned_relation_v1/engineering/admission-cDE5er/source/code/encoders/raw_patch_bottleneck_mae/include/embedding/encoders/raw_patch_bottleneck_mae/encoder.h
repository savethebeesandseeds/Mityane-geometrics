// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/config.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct PreNormBlockImpl : torch::nn::Module {
  explicit PreNormBlockImpl(const Config &c) {
    norm_attention = register_module("norm_attention", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.encoder_width}).eps(c.layer_norm_epsilon)));
    norm_feedforward = register_module("norm_feedforward", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.encoder_width}).eps(c.layer_norm_epsilon)));
    attention = register_module("attention", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(c.encoder_width, c.num_heads).dropout(c.dropout)));
    first = register_module("first", torch::nn::Linear(c.encoder_width, c.feedforward_width));
    second = register_module("second", torch::nn::Linear(c.feedforward_width, c.encoder_width));
    dropout = register_module("dropout", torch::nn::Dropout(c.dropout));
  }

  torch::Tensor forward(torch::Tensor x, const torch::Tensor &valid) {
    TORCH_CHECK(x.dim() == 3 && valid.scalar_type() == torch::kBool &&
                    valid.sizes() == x.sizes().slice(0, 2) && valid.any(1).all().item<bool>(),
                "[rpb-mae] attention rows must have at least one valid token");
    x = torch::where(valid.unsqueeze(-1), x, torch::zeros_like(x));
    auto normalized = norm_attention(x).transpose(0, 1);
    auto attended = std::get<0>(attention->forward(normalized, normalized, normalized, valid.logical_not(), false)).transpose(0, 1);
    x = torch::where(valid.unsqueeze(-1), x + dropout(attended), torch::zeros_like(x));
    auto residual = second(torch::gelu(first(norm_feedforward(x))));
    return torch::where(valid.unsqueeze(-1), x + dropout(residual), torch::zeros_like(x));
  }

  torch::nn::LayerNorm norm_attention{nullptr}, norm_feedforward{nullptr};
  torch::nn::MultiheadAttention attention{nullptr};
  torch::nn::Linear first{nullptr}, second{nullptr};
  torch::nn::Dropout dropout{nullptr};
};
TORCH_MODULE(PreNormBlock);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
