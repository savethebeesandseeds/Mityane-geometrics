#pragma once

#include <torch/torch.h>

namespace state_fusion {

// Sequence-first tensors [length, batch, width], matching LibTorch
// MultiheadAttention. A true entry in a key-padding mask ignores that key.
// An undefined mask attends to every key.

class MABImpl : public torch::nn::Module {
 public:
  MABImpl(int64_t width, int64_t heads, int64_t feedforward, double dropout);
  torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key,
                        const torch::Tensor& key_padding = {});

 private:
  torch::nn::MultiheadAttention attention_{nullptr};
  torch::nn::LayerNorm norm_attention_{nullptr};
  torch::nn::LayerNorm norm_feedforward_{nullptr};
  torch::nn::Linear feedforward_in_{nullptr};
  torch::nn::Linear feedforward_out_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
};
TORCH_MODULE(MAB);

class ISABImpl : public torch::nn::Module {
 public:
  ISABImpl(int64_t width, int64_t heads, int64_t feedforward, int64_t inducing_points,
           double dropout);
  torch::Tensor forward(const torch::Tensor& set, const torch::Tensor& set_padding);

 private:
  MAB induce_{nullptr};
  MAB process_{nullptr};
  torch::Tensor inducing_;
};
TORCH_MODULE(ISAB);

class PMAImpl : public torch::nn::Module {
 public:
  PMAImpl(int64_t width, int64_t heads, int64_t feedforward, int64_t seeds, double dropout);
  torch::Tensor forward(const torch::Tensor& set, const torch::Tensor& set_padding);

 private:
  MAB pool_{nullptr};
  torch::nn::Linear feedforward_in_{nullptr};
  torch::nn::Linear feedforward_out_{nullptr};
  torch::Tensor seeds_;
};
TORCH_MODULE(PMA);

}  // namespace state_fusion
