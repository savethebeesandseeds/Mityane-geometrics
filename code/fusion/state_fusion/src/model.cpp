#include "state_fusion/model.h"

#include <stdexcept>
#include <string>

namespace state_fusion {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

void validate_config(const Config& config) {
  require(config.d_model >= 64 && config.d_model <= 128, "d_model must be from 64 through 128");
  require(config.inducing_points >= 16 && config.inducing_points <= 64,
          "inducing_points must be from 16 through 64");
  require(config.pma_seeds == 1 || config.pma_seeds == 2, "pma_seeds must be 1 or 2");
  require(config.heads > 0 && config.d_model % config.heads == 0, "heads must divide d_model");
  require(config.isab_layers >= 1, "isab_layers must be at least 1");
  require(config.feedforward >= 1, "feedforward must be positive");
  require(config.scalar_features >= 1, "scalar_features must be positive");
  require(config.semantic_width >= 1, "semantic_width must be positive");
  require(!config.categorical_cardinalities.empty(), "categorical fields are required");
  require(config.dropout >= 0.0 && config.dropout < 1.0, "dropout must be in [0, 1)");
  for (const auto cardinality : config.categorical_cardinalities) {
    require(cardinality >= 1, "categorical cardinality must be positive");
  }
}

bool same_size(const torch::Tensor& tensor, std::initializer_list<int64_t> sizes) {
  if (tensor.dim() != static_cast<int64_t>(sizes.size())) {
    return false;
  }
  int64_t axis = 0;
  for (const auto size : sizes) {
    if (tensor.size(axis++) != size) {
      return false;
    }
  }
  return true;
}

void check_batch(const Config& config, const SetBatch& batch) {
  require(batch.scalars.defined() && batch.scalars.device().is_cuda(),
          "state fusion requires CUDA tensors");
  const auto device = batch.scalars.device();
  const auto fields = static_cast<int64_t>(config.categorical_cardinalities.size());
  require(batch.scalars.scalar_type() == torch::kFloat, "scalars must be float32");
  require(batch.scalars.dim() == 3, "scalars must be [batch, set, features]");
  const auto rows = batch.scalars.size(0);
  const auto records = batch.scalars.size(1);
  require(rows >= 1 && records >= 1, "a set batch needs at least one set and one record");
  require(batch.scalars.size(2) == config.scalar_features, "scalar feature count does not match the model");

  const auto expect_cuda = [&](const torch::Tensor& tensor, const char* name) {
    require(tensor.defined() && tensor.device() == device, std::string(name) + " must be on the same CUDA device");
  };
  expect_cuda(batch.scalar_observed, "scalar_observed");
  expect_cuda(batch.categoricals, "categoricals");
  expect_cuda(batch.categorical_observed, "categorical_observed");
  expect_cuda(batch.semantic, "semantic");
  expect_cuda(batch.semantic_observed, "semantic_observed");
  expect_cuda(batch.record_observed, "record_observed");

  require(batch.scalar_observed.dtype() == torch::kBool &&
              same_size(batch.scalar_observed, {rows, records, config.scalar_features}),
          "scalar_observed must be bool [batch, set, features]");
  require(batch.categoricals.dtype() == torch::kLong &&
              same_size(batch.categoricals, {rows, records, fields}),
          "categoricals must be int64 [batch, set, fields]");
  require(batch.categorical_observed.dtype() == torch::kBool &&
              same_size(batch.categorical_observed, {rows, records, fields}),
          "categorical_observed must be bool [batch, set, fields]");
  require(batch.semantic.dtype() == torch::kFloat &&
              same_size(batch.semantic, {rows, records, config.semantic_width}),
          "semantic must be float32 [batch, set, semantic_width]");
  require(batch.semantic_observed.dtype() == torch::kBool &&
              same_size(batch.semantic_observed, {rows, records}),
          "semantic_observed must be bool [batch, set]");
  require(batch.record_observed.dtype() == torch::kBool && same_size(batch.record_observed, {rows, records}),
          "record_observed must be bool [batch, set]");

  // An all-masked key row makes attention softmax undefined.
  require(batch.record_observed.any(/*dim=*/1).all().item<bool>(),
          "each set needs at least one observed record");

  const auto active_scalar = batch.record_observed.unsqueeze(-1) & batch.scalar_observed;
  const auto illegal_scalar = active_scalar & torch::logical_not(torch::isfinite(batch.scalars));
  require(!illegal_scalar.any().item<bool>(), "an observed scalar must be finite");

  const auto illegal_semantic =
      batch.record_observed & batch.semantic_observed &
      torch::logical_not(torch::isfinite(batch.semantic)).any(-1);
  require(!illegal_semantic.any().item<bool>(), "an observed semantic vector must be finite");

  auto ids_ok = torch::ones({rows, records}, batch.record_observed.options());
  for (int64_t field = 0; field < fields; ++field) {
    const auto ids = batch.categoricals.select(2, field);
    const auto present = batch.categorical_observed.select(2, field) & batch.record_observed;
    const auto card = config.categorical_cardinalities[static_cast<size_t>(field)];
    const auto in_range = (ids >= 0) & (ids < card);
    ids_ok = ids_ok & (present.logical_not() | in_range);
  }
  require(ids_ok.all().item<bool>(), "an observed categorical id is outside its cardinality");
}

}  // namespace

StateFusionModelImpl::StateFusionModelImpl(Config config) : config_(std::move(config)) {
  validate_config(config_);
  tokens_ = register_module("tokens", TokenEncoder(config_));
  for (int64_t layer = 0; layer < config_.isab_layers; ++layer) {
    layers_.push_back(register_module(
        "isab_" + std::to_string(layer),
        ISAB(config_.d_model, config_.heads, config_.feedforward, config_.inducing_points,
             config_.dropout)));
  }
  pool_ = register_module(
      "pool", PMA(config_.d_model, config_.heads, config_.feedforward, config_.pma_seeds, config_.dropout));
  decode_ = register_module(
      "decode", MAB(config_.d_model, config_.heads, config_.feedforward, config_.dropout));
  scalar_head_ = register_module(
      "scalar_head", torch::nn::Linear(config_.d_model, config_.scalar_features));
}

torch::Tensor StateFusionModelImpl::encode(const SetBatch& batch) {
  for (const auto& parameter : parameters()) {
    if (!parameter.is_cuda()) {
      throw std::invalid_argument("state fusion requires CUDA parameters");
    }
  }
  check_batch(config_, batch);
  auto sequence = tokens_->forward(batch).permute({1, 0, 2}).contiguous();
  const auto padding = batch.record_observed.logical_not().contiguous();
  for (auto& layer : layers_) {
    sequence = layer->forward(sequence, padding);
  }
  const auto pooled = pool_->forward(sequence, padding);
  return pooled.permute({1, 0, 2}).contiguous();
}

torch::Tensor StateFusionModelImpl::reconstruct(const SetBatch& batch) {
  const auto state = encode(batch);
  const auto queries = tokens_->forward(batch).permute({1, 0, 2}).contiguous();
  const auto keys = state.permute({1, 0, 2}).contiguous();
  const auto seed_padding = torch::zeros(
      {state.size(0), state.size(1)},
      torch::TensorOptions().dtype(torch::kBool).device(state.device()));
  const auto decoded = decode_->forward(queries, keys, seed_padding);
  return scalar_head_(decoded).permute({1, 0, 2}).contiguous();
}

torch::Tensor StateFusionModelImpl::reconstruction_loss(const SetBatch& visible,
                                                       const torch::Tensor& clean_scalars,
                                                       const torch::Tensor& hidden) {
  require(clean_scalars.defined() && hidden.defined(), "reconstruction needs a target and a mask");
  require(clean_scalars.device() == visible.scalars.device() && hidden.device() == visible.scalars.device(),
          "reconstruction tensors must share the visible batch device");
  require(clean_scalars.scalar_type() == torch::kFloat && same_size(clean_scalars, {
      visible.scalars.size(0), visible.scalars.size(1), config_.scalar_features}),
          "clean_scalars must be float32 [batch, set, features]");
  require(hidden.dtype() == torch::kBool && hidden.sizes() == clean_scalars.sizes(),
          "hidden must be bool [batch, set, features]");
  const auto still_visible = hidden & visible.scalar_observed;
  require(!still_visible.any().item<bool>(), "hidden cells must be holes in the visible batch");

  const auto prediction = reconstruct(visible);
  const auto count = hidden.sum();
  require(count.item<int64_t>() >= 1, "reconstruction needs at least one hidden cell");
  const auto error = (prediction - clean_scalars).square();
  const auto selected = torch::where(hidden, error, torch::zeros_like(error));
  return selected.sum() / count.to(selected.dtype());
}

}  // namespace state_fusion
