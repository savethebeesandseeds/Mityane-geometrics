#pragma once

#include <torch/torch.h>

#include <vector>

namespace state_fusion {

struct Config;
struct SetBatch;

// Projects one record into model width. A missing field contributes a learned
// absent vector. The stored number in a hole is not read.
class TokenEncoderImpl : public torch::nn::Module {
 public:
  explicit TokenEncoderImpl(const Config& config);
  torch::Tensor forward(const SetBatch& batch);

 private:
  int64_t width_;
  std::vector<int64_t> cardinalities_;
  torch::nn::Linear scalars_{nullptr};
  torch::nn::Linear semantic_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::vector<torch::nn::Embedding> categories_;
  torch::Tensor categorical_absent_;
  torch::Tensor semantic_absent_;
};
TORCH_MODULE(TokenEncoder);

}  // namespace state_fusion
