// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../LICENSE.
#pragma once
#include <torch/torch.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

namespace cuwacunu::networks::latent_dynamics::v1 {
inline constexpr const char *model_tag="horizon_affine_latent_dynamics_v1";
struct Config {
  int64_t state_width{36}, history_length{4};
  std::vector<int64_t> horizons{1,16,128};
  bool residual{false};
  torch::Device device{torch::kCUDA,0};
  torch::ScalarType dtype{torch::kFloat32};
};
inline void validate(const Config &c) {
  TORCH_CHECK(c.state_width>0 && c.history_length>0 && !c.horizons.empty(),
              "[dynamics v1] positive dimensions and explicit horizons required");
  TORCH_CHECK(c.dtype==torch::kFloat32 || c.dtype==torch::kFloat64,"[dynamics v1] float32/64 only");
  TORCH_CHECK(!c.device.is_cuda()||c.device.index()>=0,"[dynamics v1] explicit CUDA device ordinal required");
  const auto limit=std::numeric_limits<int64_t>::max();
  TORCH_CHECK(c.horizons.size()<=static_cast<uint64_t>(limit)&&c.history_length<=limit/c.state_width&&
    static_cast<int64_t>(c.horizons.size())<=limit/c.state_width,
    "[dynamics v1] flattened dimension overflow");
  const auto columns=static_cast<uint64_t>(c.history_length*c.state_width);
  const auto rows=static_cast<uint64_t>(static_cast<int64_t>(c.horizons.size())*c.state_width);
  const auto maximum=std::min(static_cast<uint64_t>(limit),
    static_cast<uint64_t>(std::numeric_limits<size_t>::max())/(c.dtype==torch::kFloat64?8:4));
  TORCH_CHECK(rows<=maximum&&columns<=(maximum-rows)/rows,
              "[dynamics v1] affine parameter element/byte-count overflow");
  std::set<int64_t> seen;
  for(auto h:c.horizons) TORCH_CHECK(h>0 && seen.insert(h).second,"[dynamics v1] positive distinct horizons required");
}
// Independent affine coefficients per declared horizon; never a rollout.
struct AffineDynamicsImpl : torch::nn::Module {
  Config config;
  torch::nn::Linear affine{nullptr};
  explicit AffineDynamicsImpl(const Config &c):config(c) {
    validate(c);
    affine=register_module("affine",torch::nn::Linear(c.history_length*c.state_width,
      static_cast<int64_t>(c.horizons.size())*c.state_width));
    affine->to(c.device,c.dtype);
    if(c.residual) { torch::NoGradGuard guard; affine->weight.zero_(); affine->bias.zero_(); }
  }
  torch::Tensor forward(const torch::Tensor &history) {
    TORCH_CHECK(history.dim()==3 && history.size(1)==config.history_length &&
      history.size(2)==config.state_width && history.device()==config.device &&
      history.scalar_type()==config.dtype,"[dynamics v1] expected [B,K,d] on declared device/dtype");
    auto forecast=affine->forward(history.flatten(1)).reshape({history.size(0),
      static_cast<int64_t>(config.horizons.size()),config.state_width});
    if(config.residual) forecast=forecast+history.select(1,config.history_length-1).unsqueeze(1);
    return forecast;
  }
  // Includes the fixed last-state residual, if declared.
  torch::Tensor absolute_weight() const {
    auto weight=affine->weight.clone();
    if(config.residual) {
      auto identity=torch::eye(config.state_width,weight.options()).repeat({static_cast<int64_t>(config.horizons.size()),1});
      weight.slice(1,(config.history_length-1)*config.state_width,config.history_length*config.state_width).add_(identity);
    }
    return weight;
  }
  void validate_state() const {
    validate(config);
    const auto q=static_cast<int64_t>(config.horizons.size());
    TORCH_CHECK(affine->weight.sizes()==torch::IntArrayRef({q*config.state_width,config.history_length*config.state_width})&&
      affine->bias.sizes()==torch::IntArrayRef({q*config.state_width}),"[dynamics v1] affine tensor shape drift");
    for(const auto &p:parameters()) TORCH_CHECK(p.device()==config.device && p.scalar_type()==config.dtype &&
      torch::isfinite(p).all().item<bool>(),"[dynamics v1] invalid parameters");
  }
};
TORCH_MODULE(AffineDynamics);
} // namespace cuwacunu::networks::latent_dynamics::v1
