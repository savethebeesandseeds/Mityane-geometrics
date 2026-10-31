#pragma once

#include "state_fusion/blocks.h"
#include "state_fusion/embeddings.h"

#include <torch/torch.h>

#include <string>
#include <vector>

namespace state_fusion {

struct Config {
  int64_t scalar_features = 8;
  int64_t semantic_width = 16;
  std::vector<int64_t> categorical_cardinalities{16, 8, 8, 4, 6};
  int64_t d_model = 64;
  int64_t heads = 4;
  int64_t inducing_points = 16;
  int64_t isab_layers = 2;
  int64_t pma_seeds = 1;
  int64_t feedforward = 128;
  double dropout = 0.0;
};

// One batch of unordered records. Set length is dimension 1.
// Observed flags are true when the value is present. A false flag is a hole,
// including when the stored value is zero or non-finite.
// Categorical columns, in order, are place, kind, record type, side, and
// currency. Cardinalities are configuration. The defaults are synthetic bounds.
struct SetBatch {
  torch::Tensor scalars;               // [batch, set, features] float32
  torch::Tensor scalar_observed;       // [batch, set, features] bool
  torch::Tensor categoricals;          // [batch, set, fields] int64
  torch::Tensor categorical_observed;  // [batch, set, fields] bool
  torch::Tensor semantic;              // [batch, set, semantic_width] float32
  torch::Tensor semantic_observed;     // [batch, set] bool
  torch::Tensor record_observed;       // [batch, set] bool
};

class StateFusionModelImpl : public torch::nn::Module {
 public:
  explicit StateFusionModelImpl(Config config);

  // [batch, seeds, d_model]. Permutation-invariant in the set dimension.
  torch::Tensor encode(const SetBatch& batch);

  // Answer for each query row, [batch, queries, d_model].
  // state is a cached market state from encode. query_batch uses the same
  // fields as a record, with the set dimension as the query dimension. Each
  // query is assembled by the shared token encoder, then one attention block
  // reads the pooled state. The set is not encoded again. A hole uses the
  // same absent path as a record hole. Mixture density sampling of this
  // answer is deferred.
  torch::Tensor query(const torch::Tensor& state, const SetBatch& query_batch);

  // Linear probe on query(), [batch, queries, scalar_features].
  // A training run that owns its task head can ignore this and read query().
  torch::Tensor query_scalars(const torch::Tensor& state, const SetBatch& query_batch);

  // Scalar reconstruction for every record, [batch, set, features].
  torch::Tensor reconstruct(const SetBatch& batch);

  // Mean squared error on hidden cells. hidden is true where a formerly
  // observed scalar was withheld. Those cells must already be holes in batch.
  torch::Tensor reconstruction_loss(const SetBatch& visible, const torch::Tensor& clean_scalars,
                                    const torch::Tensor& hidden);

  const Config& config() const { return config_; }

 private:
  Config config_;
  TokenEncoder tokens_{nullptr};
  std::vector<ISAB> layers_;
  PMA pool_{nullptr};
  MAB decode_{nullptr};
  torch::nn::Linear scalar_head_{nullptr};
  MAB query_{nullptr};
  torch::nn::Linear query_scalar_head_{nullptr};
};
TORCH_MODULE(StateFusionModel);

// LibTorch archive of the config and the module weights. The caller creates
// the parent directory. This package does not ship a trained checkpoint.
void save_checkpoint(const StateFusionModel& model, const std::string& path);

// Restores a checkpoint written by save_checkpoint. device must be CUDA.
StateFusionModel load_checkpoint(const std::string& path, torch::Device device);

}  // namespace state_fusion
