// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../LICENSE.
#pragma once
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace cuwacunu::networks::state_representation::v1 {
inline constexpr const char *model_tag = "affine_state_representation_v1";
struct Config {
  int64_t embedding_width{36}, state_width{36};
  bool identity_initialization{true};
  torch::Device device{torch::kCUDA, 0};
  torch::ScalarType dtype{torch::kFloat32};
};
inline void validate(const Config &c) {
  TORCH_CHECK(c.embedding_width > 0 && c.state_width > 0,
              "[state v1] positive embedding/state widths required");
  TORCH_CHECK(c.dtype == torch::kFloat32 || c.dtype == torch::kFloat64,
              "[state v1] float32/float64 only");
  TORCH_CHECK(!c.device.is_cuda()||c.device.index()>=0,"[state v1] explicit CUDA device ordinal required");
  const auto maximum=std::min(static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
    static_cast<uint64_t>(std::numeric_limits<size_t>::max())/(c.dtype==torch::kFloat64?8:4));
  const auto d=static_cast<uint64_t>(c.state_width),embedding=static_cast<uint64_t>(c.embedding_width);
  TORCH_CHECK(d<=maximum&&embedding<=(maximum-d)/d,
              "[state v1] affine parameter element/byte-count overflow");
  TORCH_CHECK(!c.identity_initialization || c.embedding_width == c.state_width,
              "[state v1] identity initialization requires equal widths");
}
// Maps native e to G(e), with one frozen TRAIN transform for online and teacher.
// Arbitrary leading axes are allowed; the final axis is always native D.
struct AffineStateImpl : torch::nn::Module {
  Config config;
  torch::nn::Linear affine{nullptr};
  torch::Tensor embedding_mean, embedding_scale;
  explicit AffineStateImpl(const Config &c, torch::Tensor mean={}, torch::Tensor scale={}) : config(c) {
    validate(c);
    auto options=torch::TensorOptions().dtype(c.dtype).device(c.device);
    if (!mean.defined()) mean=torch::zeros({c.embedding_width},options);
    if (!scale.defined()) scale=torch::ones({c.embedding_width},options);
    TORCH_CHECK(mean.dim()==1 && mean.size(0)==c.embedding_width && scale.sizes()==mean.sizes() &&
      torch::isfinite(mean).all().item<bool>() && torch::isfinite(scale).all().item<bool>() &&
      scale.gt(0).all().item<bool>(), "[state v1] finite mean and positive frozen scale required");
    embedding_mean=register_buffer("embedding_mean",mean.detach().to(options).clone());
    embedding_scale=register_buffer("embedding_scale",scale.detach().to(options).clone());
    affine=register_module("affine",torch::nn::Linear(c.embedding_width,c.state_width));
    affine->to(c.device,c.dtype);
    if(c.identity_initialization) {
      torch::NoGradGuard guard;
      affine->weight.copy_(torch::eye(c.state_width,options)); affine->bias.zero_();
    }
  }
  torch::Tensor forward(const torch::Tensor &e) {
    TORCH_CHECK(e.dim()>=2 && e.size(-1)==config.embedding_width && e.device()==config.device &&
      e.scalar_type()==config.dtype, "[state v1] input axes/device/dtype mismatch");
    return affine->forward((e-embedding_mean)/embedding_scale);
  }
  // z = e W^T + b, including the frozen preprocessing transform.
  std::pair<torch::Tensor,torch::Tensor> native_affine() const {
    auto w=affine->weight/embedding_scale.unsqueeze(0);
    return {w,affine->bias-torch::matmul(w,embedding_mean)};
  }
  void validate_state() const {
    validate(config);
    TORCH_CHECK(affine->weight.sizes()==torch::IntArrayRef({config.state_width,config.embedding_width})&&
      affine->bias.sizes()==torch::IntArrayRef({config.state_width}),"[state v1] affine tensor shape drift");
    for(const auto &p:parameters()) TORCH_CHECK(p.device()==config.device && p.scalar_type()==config.dtype &&
      torch::isfinite(p).all().item<bool>(), "[state v1] invalid parameters");
    TORCH_CHECK(embedding_mean.sizes()==torch::IntArrayRef({config.embedding_width}) &&
      embedding_scale.sizes()==embedding_mean.sizes() &&
      embedding_mean.device()==config.device && embedding_scale.device()==config.device &&
      embedding_mean.scalar_type()==config.dtype && embedding_scale.scalar_type()==config.dtype &&
      !embedding_mean.requires_grad() && !embedding_scale.requires_grad() &&
      torch::isfinite(embedding_mean).all().item<bool>() &&
      torch::isfinite(embedding_scale).all().item<bool>() && embedding_scale.gt(0).all().item<bool>(),
      "[state v1] invalid frozen transform");
  }
};
TORCH_MODULE(AffineState);
inline AffineState frozen_copy(const AffineState &source) {
  AffineState copy(source->config,source->embedding_mean,source->embedding_scale);
  torch::NoGradGuard guard;
  const auto parameters=source->named_parameters();
  for(auto &p:copy->named_parameters()) p.value().copy_(parameters[p.key()]);
  copy->eval(); for(auto &p:copy->parameters()) p.set_requires_grad(false);
  return copy;
}
} // namespace cuwacunu::networks::state_representation::v1
