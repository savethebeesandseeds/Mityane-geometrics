#include "state_fusion/embeddings.h"

#include "state_fusion/model.h"

#include <stdexcept>
#include <string>

namespace state_fusion {

TokenEncoderImpl::TokenEncoderImpl(const Config& config)
    : width_(config.d_model),
      cardinalities_(config.categorical_cardinalities),
      scalars_(register_module(
          "scalars", torch::nn::Linear(config.scalar_features * 2, config.d_model))),
      semantic_(register_module(
          "semantic", torch::nn::Linear(config.semantic_width, config.d_model))),
      norm_(register_module(
          "norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.d_model})))) {
  if (config.scalar_features < 1 || config.semantic_width < 1 || cardinalities_.empty()) {
    throw std::invalid_argument("token encoder requires scalars, a semantic width, and categories");
  }
  for (size_t field = 0; field < cardinalities_.size(); ++field) {
    if (cardinalities_[field] < 1) {
      throw std::invalid_argument("categorical cardinality must be positive");
    }
    categories_.push_back(register_module(
        "category_" + std::to_string(field),
        torch::nn::Embedding(cardinalities_[field], width_)));
  }
  categorical_absent_ = register_parameter(
      "categorical_absent",
      torch::empty({static_cast<int64_t>(cardinalities_.size()), width_}));
  semantic_absent_ = register_parameter("semantic_absent", torch::empty({width_}));
  // Absent tokens are learned vectors. A zero vector would look like a missing
  // projection, so the initial draw is a small normal and not all zeros.
  torch::nn::init::normal_(categorical_absent_, 0.0, 0.02);
  torch::nn::init::normal_(semantic_absent_, 0.0, 0.02);
}

torch::Tensor TokenEncoderImpl::forward(const SetBatch& batch) {
  // torch::where does not read the discarded branch. A non-finite hole, and
  // every field of an unobserved record, stays out of the token. The record
  // is then dropped by key padding rather than replaced with a zero token.
  const auto record = batch.record_observed;
  const auto scalar_present = batch.scalar_observed & record.unsqueeze(-1);
  const auto clean = torch::where(scalar_present, batch.scalars, torch::zeros_like(batch.scalars));
  const auto flags = scalar_present.to(batch.scalars.dtype());
  auto token = scalars_(torch::cat({clean, flags}, -1));

  for (size_t field = 0; field < categories_.size(); ++field) {
    const auto index = static_cast<int64_t>(field);
    const auto ids = batch.categoricals.select(-1, index);
    const auto usable = batch.categorical_observed.select(-1, index) & record;
    const auto safe = torch::where(usable, ids, torch::zeros_like(ids));
    const auto looked = categories_[field](safe);
    const auto absent = categorical_absent_.select(0, index).view({1, 1, width_});
    token = token + torch::where(usable.unsqueeze(-1), looked, absent.expand_as(looked));
  }

  const auto semantic_present = (batch.semantic_observed & record).unsqueeze(-1);
  const auto semantic_clean =
      torch::where(semantic_present, batch.semantic, torch::zeros_like(batch.semantic));
  const auto projected = semantic_(semantic_clean);
  const auto absent = semantic_absent_.view({1, 1, width_});
  token = token + torch::where(semantic_present, projected, absent.expand_as(projected));
  return norm_(token);
}

}  // namespace state_fusion
