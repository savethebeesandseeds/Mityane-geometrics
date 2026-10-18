// Copyright (c) 2026 Waajacu. All rights reserved.
// See ../../../../../../../../../LICENSE for the copyright and permission notice.
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/preprocessing.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

// TRAIN-only accumulation. The caller owns the split boundary and supplies raw
// batches; this class never opens a dataset or observes a held-out split. Pass
// configurable_spectral_input_config(logical) for configurable v20/v22 inputs,
// so validation keeps logical H without imposing backbone patch divisibility.
// Persistent memory is three CPU [C,F] arrays; temporary memory is one batch.
class StreamingScaler {
public:
  explicit StreamingScaler(const Config &input_config) : config_(input_config) {
    validate_config(config_);
    const auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
    mean_ = torch::zeros({config_.channel_count,config_.input_width},options);
    m2_ = torch::zeros_like(mean_);
    count_ = torch::zeros(mean_.sizes(),options.dtype(torch::kInt64));
  }

  const Config &config() const { return config_; }
  torch::Tensor count() const { return count_.clone(); }

  void add_train_batch(const Input &input) {
    torch::NoGradGuard no_grad;
    validate_input(input,config_);
    const auto raw = input.data.detach().to(torch::kCPU,torch::kFloat64);
    const auto observed = input.observed.to(torch::kCPU);
    const auto indices = channel_indices(input.channel_ids,config_,input.data.size(0),torch::kCPU);
    const auto safe = torch::where(observed,raw,torch::zeros_like(raw));
    const auto flat = safe.reshape({-1,config_.input_width});
    std::vector<torch::Tensor> counts, means, moments;
    counts.reserve(config_.channel_count); means.reserve(config_.channel_count);
    moments.reserve(config_.channel_count);
    for (int64_t id = 0; id < config_.channel_count; ++id) {
      const auto support = observed.logical_and(indices.eq(id).unsqueeze(-1).unsqueeze(-1));
      const auto count = support.sum(std::vector<int64_t>{0,1,2});
      const auto denominator = count.clamp_min(1).to(torch::kFloat64);
      // Subtract one supported value before summation. This avoids summing a
      // large common offset and keeps the batch moments in float64 precision.
      const auto first = support.reshape({-1,config_.input_width}).to(torch::kInt64).argmax(0);
      const auto first_value = flat.gather(0,first.unsqueeze(0)).squeeze(0);
      const auto anchor = torch::where(count.gt(0),first_value,torch::zeros_like(first_value));
      const auto shifted = torch::where(support,safe-anchor,torch::zeros_like(safe));
      const auto mean = anchor+shifted.sum(std::vector<int64_t>{0,1,2})/denominator;
      const auto centered = torch::where(support,safe-mean,torch::zeros_like(safe));
      counts.push_back(count); means.push_back(mean);
      moments.push_back(centered.square().sum(std::vector<int64_t>{0,1,2}));
    }
    combine(torch::stack(counts),torch::stack(means),torch::stack(moments));
  }

  // Chan's parallel combination rule. Streams must describe the same input
  // geometry and semantic coordinate order; model capacity is irrelevant here.
  // The streams must cover disjoint TRAIN observations unless duplication is
  // intentional. As with batch addition, no historical arrays are retained.
  void merge(const StreamingScaler &other) {
    torch::NoGradGuard no_grad;
    TORCH_CHECK(config_.channel_count == other.config_.channel_count &&
                    config_.input_width == other.config_.input_width &&
                    config_.history_length == other.config_.history_length &&
                    resolved_channel_ids(config_) == resolved_channel_ids(other.config_) &&
                    config_.scale_floor == other.config_.scale_floor &&
                    config_.sampling_interval == other.config_.sampling_interval,
                "[rpb streaming scaler] incompatible input/semantic/floor schema");
    combine(other.count_,other.mean_,other.m2_);
  }

  FrozenScaler finish() const {
    torch::NoGradGuard no_grad;
    TORCH_CHECK(count_.gt(0).all().item<bool>(),
                "[rpb streaming scaler] every semantic coordinate needs TRAIN observations");
    const auto deviation = (m2_/count_.to(torch::kFloat64)).sqrt();
    FrozenScaler out{mean_.clone(),deviation.clamp_min(config_.scale_floor),count_.clone(),
        torch::tensor(resolved_channel_ids(config_),torch::kInt64),config_.scale_floor,
        deviation.lt(config_.scale_floor)};
    out.validate(config_);
    return out;
  }

private:
  void combine(const torch::Tensor &count, const torch::Tensor &mean, const torch::Tensor &m2) {
    // Calculate and validate a complete new state before publication. Rejected
    // batches/merges cannot leave partially updated semantic coordinates.
    TORCH_CHECK(count.scalar_type() == torch::kInt64 && count.sizes() == count_.sizes() &&
                    mean.scalar_type() == torch::kFloat64 && mean.sizes() == mean_.sizes() &&
                    m2.scalar_type() == torch::kFloat64 && m2.sizes() == m2_.sizes() &&
                    count.device().is_cpu() && mean.device().is_cpu() && m2.device().is_cpu() &&
                    count.ge(0).all().item<bool>() && torch::isfinite(mean).all().item<bool>() &&
                    torch::isfinite(m2).all().item<bool>() && m2.ge(0).all().item<bool>(),
                "[rpb streaming scaler] invalid/nonfinite batch moments");
    TORCH_CHECK(count_.le(std::numeric_limits<int64_t>::max()-count).all().item<bool>(),
                "[rpb streaming scaler] int64 observation count overflow");
    const auto total = count_+count;
    const auto ratio = count.to(torch::kFloat64)/total.clamp_min(1).to(torch::kFloat64);
    // An empty coordinate must not form 0 * infinity when a finite but large
    // first batch mean is installed. No support contributes zero moments.
    const auto both = count_.gt(0).logical_and(count.gt(0));
    const auto delta = torch::where(both,mean-mean_,torch::zeros_like(mean_));
    const auto merged_mean = torch::where(count_.eq(0),mean,mean_+delta*ratio);
    const auto merged_m2 = m2_+m2+delta.square()*(count_.to(torch::kFloat64)*ratio);
    TORCH_CHECK(torch::isfinite(merged_mean).all().item<bool>() &&
                    torch::isfinite(merged_m2).all().item<bool>() && merged_m2.ge(0).all().item<bool>(),
                "[rpb streaming scaler] float64 merged moment overflow");
    count_ = total; mean_ = merged_mean; m2_ = merged_m2;
  }

  Config config_;
  torch::Tensor count_, mean_, m2_;
};

} // namespace embedding::encoders::raw_patch_bottleneck_mae
