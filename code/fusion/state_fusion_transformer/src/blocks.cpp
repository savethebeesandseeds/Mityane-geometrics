#include "state_fusion/blocks.h"

#include <tuple>

namespace state_fusion {

MABImpl::MABImpl(int64_t width, int64_t heads, int64_t feedforward, double dropout)
    : attention_(register_module(
          "attention",
          torch::nn::MultiheadAttention(
              torch::nn::MultiheadAttentionOptions(width, heads).dropout(dropout)))),
      norm_attention_(register_module(
          "norm_attention", torch::nn::LayerNorm(torch::nn::LayerNormOptions({width})))),
      norm_feedforward_(register_module(
          "norm_feedforward", torch::nn::LayerNorm(torch::nn::LayerNormOptions({width})))),
      feedforward_in_(register_module("feedforward_in", torch::nn::Linear(width, feedforward))),
      feedforward_out_(register_module("feedforward_out", torch::nn::Linear(feedforward, width))),
      dropout_(register_module("dropout", torch::nn::Dropout(dropout))) {}

torch::Tensor MABImpl::forward(const torch::Tensor& query, const torch::Tensor& key,
                               const torch::Tensor& key_padding) {
  const auto q = query.contiguous();
  const auto k = key.contiguous();
  // A true padding entry drops that key. add_zero_attn stays off, so a hole is
  // not a learned or zero attention slot.
  torch::Tensor attended;
  if (key_padding.defined()) {
    attended = std::get<0>(attention_->forward(q, k, k, key_padding.contiguous(), false));
  } else {
    attended = std::get<0>(attention_->forward(q, k, k, {}, false));
  }
  const auto mixed = norm_attention_(q + dropout_(attended));
  const auto transformed = feedforward_out_(dropout_(torch::relu(feedforward_in_(mixed))));
  return norm_feedforward_(mixed + transformed);
}

ISABImpl::ISABImpl(int64_t width, int64_t heads, int64_t feedforward, int64_t inducing_points,
                   double dropout)
    : induce_(register_module("induce", MAB(width, heads, feedforward, dropout))),
      process_(register_module("process", MAB(width, heads, feedforward, dropout))) {
  auto inducing = torch::empty({inducing_points, width});
  torch::nn::init::xavier_uniform_(inducing);
  inducing_ = register_parameter("inducing", inducing);
}

torch::Tensor ISABImpl::forward(const torch::Tensor& set, const torch::Tensor& set_padding) {
  const auto batch = set.size(1);
  const auto inducing = inducing_.unsqueeze(1).repeat({1, batch, 1});
  const auto hidden = induce_->forward(inducing, set, set_padding);
  return process_->forward(set, hidden);
}

PMAImpl::PMAImpl(int64_t width, int64_t heads, int64_t feedforward, int64_t seeds, double dropout)
    : pool_(register_module("pool", MAB(width, heads, feedforward, dropout))),
      feedforward_in_(register_module("feedforward_in", torch::nn::Linear(width, feedforward))),
      feedforward_out_(register_module("feedforward_out", torch::nn::Linear(feedforward, width))) {
  auto seed = torch::empty({seeds, width});
  torch::nn::init::xavier_uniform_(seed);
  seeds_ = register_parameter("seeds", seed);
}

torch::Tensor PMAImpl::forward(const torch::Tensor& set, const torch::Tensor& set_padding) {
  const auto prepared = feedforward_out_(torch::relu(feedforward_in_(set)));
  const auto seeds = seeds_.unsqueeze(1).repeat({1, set.size(1), 1});
  return pool_->forward(seeds, prepared, set_padding);
}

}  // namespace state_fusion
