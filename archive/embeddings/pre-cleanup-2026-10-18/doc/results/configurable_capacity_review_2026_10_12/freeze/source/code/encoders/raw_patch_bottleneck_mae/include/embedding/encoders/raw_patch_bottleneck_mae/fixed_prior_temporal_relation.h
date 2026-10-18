// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/partitioned_temporal_relation.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kFixedPriorTemporalRelationModelTag = "RPB-v17";
inline constexpr const char *kFixedPriorTemporalRelationArchitectureId =
    "early-native20-shape-plus12-fixed-grouped-time-odd-relations-v1";
inline constexpr const char *kFixedPriorTemporalRelationTrainingRecipeId =
    "original-waveform-huber1;fixed-grouped-odd-prior432;shape-backbone-decoder-trainable-v1";
inline constexpr int64_t kFixedPriorTemporalRelationParameterCount = 226877;
inline constexpr int64_t kFixedPriorTemporalRelationTrainableParameterCount = 226445;
inline constexpr int64_t kFixedPriorTemporalRelationFrozenParameterCount = 432;

inline torch::Tensor literal_fixed_odd_relation_prior(const torch::TensorOptions &options) {
  auto prior = torch::zeros({3, 4, 36}, torch::kFloat32);
  auto w = prior.accessor<float, 3>();
  for (int64_t pair = 0; pair < 3; ++pair)
    for (int64_t k = 0; k < 4; ++k)
      for (int64_t feature_pair = 0; feature_pair < 9; ++feature_pair)
        w[pair][k][feature_pair * 4 + k] = 1.f / 9.f;
  return prior.to(options);
}

// The v16 architecture, parameter names, ordering, initializer and all forward
// arithmetic are inherited unchanged. Only the grouped odd weight is frozen.
// The original waveform decoder still trains through the sole partitioned32.
struct FixedPriorTemporalRelationModelImpl : PartitionedTemporalRelationModelImpl {
  explicit FixedPriorTemporalRelationModelImpl(const Config &c) : PartitionedTemporalRelationModelImpl(c) {
    freeze_and_validate_prior();
  }
  void load(torch::serialize::InputArchive &archive) override {
    PartitionedTemporalRelationModelImpl::load(archive);
    freeze_and_validate_prior();
  }
  void validate_fixed_prior() const {
    const auto &w = grouped_odd_relation_projection->weight;
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32 &&
                    w.sizes() == torch::IntArrayRef({3, 4, 36}) && !w.requires_grad() &&
                    !w.grad().defined() && torch::equal(w, literal_fixed_odd_relation_prior(w.options())),
                "[rpb-fixed-relations] exact frozen432 literal prior without gradients required");
  }
private:
  void freeze_and_validate_prior() {
    auto &w = grouped_odd_relation_projection->weight;
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32 &&
                    w.sizes() == torch::IntArrayRef({3, 4, 36}) &&
                    torch::equal(w, literal_fixed_odd_relation_prior(w.options())),
                "[rpb-fixed-relations] checkpoint must retain the exact original432 prior");
    w.set_requires_grad(false);
    w.mutable_grad() = torch::Tensor();
    validate_fixed_prior();
  }
};
TORCH_MODULE(FixedPriorTemporalRelationModel);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
