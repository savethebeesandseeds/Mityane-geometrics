#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace praesidium_humanitatis {

struct Config {
  int64_t hidden_size = 16;
  int64_t attention_heads = 4;
  double dropout = 0.1;
  int64_t static_continuous = 1;
  int64_t past_continuous = 2;
  int64_t future_continuous = 1;
  std::vector<int64_t> static_categorical_cardinalities;
  std::vector<int64_t> past_categorical_cardinalities;
  std::vector<int64_t> future_categorical_cardinalities;
  std::vector<double> quantiles{0.1, 0.5, 0.9};
  // Optional monotone output parameterization: an unconstrained central
  // quantile and nonnegative softplus gaps. This prevents crossings, but does
  // not by itself calibrate coverage. False preserves the original TFT head.
  bool ordered_quantiles = false;
  // Legacy defaults preserve existing checkpoints and parameter topology.
  double layer_norm_epsilon = 1e-5;
  // Empty means independent future embeddings. Otherwise each future variable
  // maps to a same-type past variable, or -1 for an independent embedding.
  // Categorical mappings require equal cardinalities. A shared module is owned
  // and registered only by the past group, so its optimizer update occurs once.
  std::vector<int64_t> future_continuous_past_indices;
  std::vector<int64_t> future_categorical_past_indices;
  // Width of one pooled vector, selected as a single variable. Zero omits it.
  // A positive width is projected by Linear(width, hidden), not split into scalars.
  int64_t static_vector = 0;
  int64_t past_vector = 0;
  int64_t future_vector = 0;

  void validate() const;
};

// Static tensors: [batch, variables]; temporal tensors: [batch, time, variables].
// Both tensors must be defined, including a zero-width tensor for an absent type.
// Continuous inputs are float32 or float64, matching the model's dtype. Category
// IDs are int64. Every tensor must be on the model's CPU or CUDA device.
struct FeatureBatch {
  FeatureBatch() = default;
  FeatureBatch(torch::Tensor continuous_values, torch::Tensor categorical_values,
               torch::Tensor vector_values = {})
      : continuous(std::move(continuous_values)),
        categorical(std::move(categorical_values)),
        vector(std::move(vector_values)) {}

  torch::Tensor continuous;
  torch::Tensor categorical;
  // Defined only when that group's vector width is positive.
  // Static shape is [batch, width]. Temporal shape is [batch, time, width].
  torch::Tensor vector;
};

struct Batch {
  Batch() = default;
  Batch(FeatureBatch static_values, FeatureBatch past_values, FeatureBatch future_values,
        torch::Tensor past_mask = {}, torch::Tensor future_mask = {})
      : statics(std::move(static_values)),
        past(std::move(past_values)),
        future(std::move(future_values)),
        past_observed(std::move(past_mask)),
        future_observed(std::move(future_mask)) {}

  FeatureBatch statics;
  FeatureBatch past;
  // Only values known at forecast issuance belong here; never future targets or
  // future observed-only covariates. Their semantic availability is caller-owned.
  FeatureBatch future;
  // Bool [batch, time]. True means the step was observed. Undefined means every
  // step was observed. A false step holds the LSTM state, is not an attention
  // key, and its values cannot move the forecast. An observed zero still advances.
  torch::Tensor past_observed;
  torch::Tensor future_observed;
};

struct Output {
  torch::Tensor predictions;       // [batch, horizon, quantiles], single target
  torch::Tensor static_weights;    // [batch, static variables]
  torch::Tensor past_weights;      // [batch, history, past variables]
  torch::Tensor future_weights;    // [batch, horizon, future variables]
  // Decoder queries only. Key order is history followed by future; row j can
  // attend to keys <= history + j, including its own known-future representation.
  // A query whose every key is a hole stores zeros instead of a softmax row.
  torch::Tensor attention_weights; // [batch, heads, horizon, history + horizon]
};

namespace detail {
struct TFTNetworkImpl;
}

struct TFTImpl : torch::nn::Module {
  explicit TFTImpl(Config configuration);
  // Optional intermediate tensors for numerical inspection. The supplied map
  // is cleared and filled during this call. Tensors retain their autograd graph;
  // callers own its lifetime and may retain_grad() before backward if needed.
  Output forward(const Batch& batch,
                 std::map<std::string, torch::Tensor>* diagnostics = nullptr);

  // Immutable because changing dimensions after registration invalidates modules.
  const Config config;

 private:
  std::shared_ptr<detail::TFTNetworkImpl> network_;
};
TORCH_MODULE(TFT);

// Pinball loss summed over quantiles, then averaged over batch and horizon.
// predictions: [batch, horizon, quantiles]; targets: [batch, horizon].
// Predictions and targets must be finite. An undefined observed mask trains every
// step. A bool [batch, horizon] mask averages pinball only over observed targets.
torch::Tensor quantile_loss(const torch::Tensor& predictions,
                            const torch::Tensor& targets,
                            const std::vector<double>& quantiles,
                            const torch::Tensor& observed = {});

}  // namespace praesidium_humanitatis
