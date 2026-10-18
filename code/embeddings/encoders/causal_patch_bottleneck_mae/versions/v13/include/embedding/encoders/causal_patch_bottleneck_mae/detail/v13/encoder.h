// SPDX-License-Identifier: MIT
// CPB-v13 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v13/config.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v13 {

struct PreNormBlockImpl : torch::nn::Module {
  explicit PreNormBlockImpl(const Config &c) {
    norm_attention = register_module("norm_attention", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.encoder_width}).eps(c.layer_norm_epsilon)));
    norm_feedforward = register_module("norm_feedforward", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.encoder_width}).eps(c.layer_norm_epsilon)));
    attention = register_module("attention", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(c.encoder_width, c.num_heads).dropout(c.dropout)));
    first = register_module("first", torch::nn::Linear(c.encoder_width, c.feedforward_width));
    second = register_module("second", torch::nn::Linear(c.feedforward_width, c.encoder_width));
    dropout = register_module("dropout", torch::nn::Dropout(c.dropout));
  }

  torch::Tensor forward(torch::Tensor x, const torch::Tensor &valid,
                        const torch::Tensor &original_positions = {}) {
    TORCH_CHECK(x.dim() == 3 && valid.scalar_type() == torch::kBool &&
                    valid.sizes() == x.sizes().slice(0, 2) && valid.any(1).all().item<bool>(),
                "[rpb-mae] attention rows must have at least one valid token");
    x = torch::where(valid.unsqueeze(-1), x, torch::zeros_like(x));
    auto normalized = norm_attention(x).transpose(0, 1);
    torch::Tensor causal;
    if (original_positions.defined()) {
      TORCH_CHECK(original_positions.scalar_type() == torch::kInt64 &&
          original_positions.device() == valid.device() && original_positions.sizes() == valid.sizes(),
          "[cpb-v13] temporal original patch positions required");
      const auto T = x.size(1);
      const auto prefix = torch::arange(T, original_positions.options()).unsqueeze(0)
          .lt(valid.sum(1).unsqueeze(1));
      TORCH_CHECK(torch::equal(valid, prefix) &&
          torch::where(valid, original_positions.ge(0), original_positions.eq(-1)).all().item<bool>(),
          "[cpb-v13] temporal tokens must be a valid prefix followed by padding");
      TORCH_CHECK(original_positions.slice(1, 1).gt(original_positions.slice(1, 0, T - 1))
          .logical_or(valid.slice(1, 1).logical_not()).all().item<bool>(),
          "[cpb-v13] temporal original patch positions must strictly increase");
      // This LibTorch C++ path accepts a 2D additive mask, not a 3D bool mask.
      // The enforced packing order makes j>i exactly later original time for
      // valid keys, including gaps. Tail-padded queries still see earlier valid
      // keys (each row has one), avoiding all-masked softmax before zeroing.
      causal = torch::full({T, T}, -std::numeric_limits<double>::infinity(),
          normalized.options()).triu(1);
    }
    auto attended = std::get<0>(attention->forward(normalized, normalized, normalized,
        valid.logical_not(), false, causal)).transpose(0, 1);
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

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v13
