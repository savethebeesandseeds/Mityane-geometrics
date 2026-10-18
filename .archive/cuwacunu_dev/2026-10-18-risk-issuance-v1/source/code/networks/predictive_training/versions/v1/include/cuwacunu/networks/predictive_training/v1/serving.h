// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/networks/state_representation/v1/state.h"
#include "cuwacunu/networks/latent_dynamics/v1/dynamics.h"

namespace cuwacunu::networks::predictive_training::v1 {
// Each output feature is affine in flattened native embedding history.
// Weight convention is [output,K*D], identical to torch::nn::Linear.
struct FoldedAffine {
  torch::Tensor weight,bias;
  int64_t history_length{0},embedding_width{0};
  torch::Tensor forward(const torch::Tensor &history) const {
    TORCH_CHECK(history.dim()==3&&history.size(1)==history_length&&history.size(2)==embedding_width&&
      history.device()==weight.device()&&history.scalar_type()==weight.scalar_type(),
      "[predictive serving v1] expected native [B,K,D] on admitted device/dtype");
    return torch::matmul(history.flatten(1),weight.transpose(0,1))+bias;
  }
};
// No inverse of G is required: this check remains valid for singular or
// rectangular G and includes frozen embedding preprocessing and residual A.
inline FoldedAffine fold_state_history(const cuwacunu::networks::state_representation::v1::AffineState &state,
    int64_t history_length) {
  TORCH_CHECK(history_length>0,"[predictive serving v1] positive K required");
  const auto &c=state->config;cuwacunu::networks::state_representation::v1::validate(c);
  const auto limit=std::numeric_limits<int64_t>::max();
  TORCH_CHECK(history_length<=limit/c.state_width&&history_length<=limit/c.embedding_width,
    "[predictive serving v1] state-history dimension overflow");
  const auto rows=static_cast<uint64_t>(history_length*c.state_width),columns=static_cast<uint64_t>(history_length*c.embedding_width);
  const auto maximum=std::min(static_cast<uint64_t>(limit),
    static_cast<uint64_t>(std::numeric_limits<size_t>::max())/(c.dtype==torch::kFloat64?8:4));
  TORCH_CHECK(rows<=maximum/columns,"[predictive serving v1] folded state-history element/byte-count overflow");
  torch::NoGradGuard guard;const auto mapping=state->native_affine();
  auto weight=torch::kron(torch::eye(history_length,mapping.first.options()),mapping.first.contiguous());
  return {weight.detach().clone(),mapping.second.repeat({history_length}).detach().clone(),
    history_length,state->config.embedding_width};
}
inline FoldedAffine fold_state_forecast(const cuwacunu::networks::state_representation::v1::AffineState &state,
    const cuwacunu::networks::latent_dynamics::v1::AffineDynamics &dynamics) {
  TORCH_CHECK(state->config.state_width==dynamics->config.state_width&&
    state->config.device==dynamics->config.device&&state->config.dtype==dynamics->config.dtype,
    "[predictive serving v1] component mismatch");
  torch::NoGradGuard guard;const auto history=fold_state_history(state,dynamics->config.history_length);
  auto a=dynamics->absolute_weight();
  const auto maximum=static_cast<uint64_t>(std::numeric_limits<size_t>::max())/(a.scalar_type()==torch::kFloat64?8:4);
  TORCH_CHECK(static_cast<uint64_t>(a.size(0))<=maximum/static_cast<uint64_t>(history.weight.size(1)),
    "[predictive serving v1] folded forecast byte-count overflow");
  return {torch::matmul(a,history.weight).detach().clone(),
    (torch::matmul(a,history.bias)+dynamics->affine->bias).detach().clone(),history.history_length,history.embedding_width};
}
inline FoldedAffine fold_linear_readout(const FoldedAffine &features,const torch::Tensor &readout_weight,
    const torch::Tensor &readout_bias) {
  TORCH_CHECK(readout_weight.dim()==2&&readout_weight.size(1)==features.weight.size(0)&&
    readout_bias.dim()==1&&readout_bias.size(0)==readout_weight.size(0)&&
    readout_weight.device()==features.weight.device()&&readout_bias.device()==features.weight.device()&&
    readout_weight.scalar_type()==features.weight.scalar_type()&&readout_bias.scalar_type()==features.weight.scalar_type(),
    "[predictive serving v1] linear readout axes/device/dtype mismatch");
  torch::NoGradGuard guard;
  const auto maximum=static_cast<uint64_t>(std::numeric_limits<size_t>::max())/(features.weight.scalar_type()==torch::kFloat64?8:4);
  TORCH_CHECK(static_cast<uint64_t>(readout_weight.size(0))<=maximum/static_cast<uint64_t>(features.weight.size(1)),
    "[predictive serving v1] folded readout byte-count overflow");
  return {torch::matmul(readout_weight,features.weight).detach().clone(),
    (torch::matmul(readout_weight,features.bias)+readout_bias).detach().clone(),features.history_length,features.embedding_width};
}
inline FoldedAffine concatenate(const FoldedAffine &a,const FoldedAffine &b) {
  TORCH_CHECK(a.history_length==b.history_length&&a.embedding_width==b.embedding_width&&
    a.weight.device()==b.weight.device()&&a.weight.scalar_type()==b.weight.scalar_type(),
    "[predictive serving v1] concatenated features need identical native input support");
  return {torch::cat({a.weight,b.weight},0),torch::cat({a.bias,b.bias},0),a.history_length,a.embedding_width};
}
} // namespace cuwacunu::networks::predictive_training::v1
