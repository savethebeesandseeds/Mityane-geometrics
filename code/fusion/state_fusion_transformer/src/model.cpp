#include "state_fusion/model.h"

#include <ATen/CPUGeneratorImpl.h>
#include <torch/serialize.h>

#include <filesystem>
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

// Same field contract as a record, without the key-row rule. An empty query
// is a row of holes and is a valid question. A fully masked key row is not.
void check_query(const Config& config, const SetBatch& batch) {
  require(batch.scalars.defined() && batch.scalars.device().is_cuda(),
          "state fusion requires CUDA tensors");
  const auto device = batch.scalars.device();
  const auto fields = static_cast<int64_t>(config.categorical_cardinalities.size());
  require(batch.scalars.scalar_type() == torch::kFloat, "scalars must be float32");
  require(batch.scalars.dim() == 3, "scalars must be [batch, set, features]");
  const auto rows = batch.scalars.size(0);
  const auto queries = batch.scalars.size(1);
  require(rows >= 1 && queries >= 1, "a query batch needs at least one batch row and one query");
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
              same_size(batch.scalar_observed, {rows, queries, config.scalar_features}),
          "scalar_observed must be bool [batch, set, features]");
  require(batch.categoricals.dtype() == torch::kLong &&
              same_size(batch.categoricals, {rows, queries, fields}),
          "categoricals must be int64 [batch, set, fields]");
  require(batch.categorical_observed.dtype() == torch::kBool &&
              same_size(batch.categorical_observed, {rows, queries, fields}),
          "categorical_observed must be bool [batch, set, fields]");
  require(batch.semantic.dtype() == torch::kFloat &&
              same_size(batch.semantic, {rows, queries, config.semantic_width}),
          "semantic must be float32 [batch, set, semantic_width]");
  require(batch.semantic_observed.dtype() == torch::kBool &&
              same_size(batch.semantic_observed, {rows, queries}),
          "semantic_observed must be bool [batch, set]");
  require(batch.record_observed.dtype() == torch::kBool && same_size(batch.record_observed, {rows, queries}),
          "record_observed must be bool [batch, set]");

  const auto active_scalar = batch.record_observed.unsqueeze(-1) & batch.scalar_observed;
  const auto illegal_scalar = active_scalar & torch::logical_not(torch::isfinite(batch.scalars));
  require(!illegal_scalar.any().item<bool>(), "an observed scalar must be finite");

  const auto illegal_semantic =
      batch.record_observed & batch.semantic_observed &
      torch::logical_not(torch::isfinite(batch.semantic)).any(-1);
  require(!illegal_semantic.any().item<bool>(), "an observed semantic vector must be finite");

  auto ids_ok = torch::ones({rows, queries}, batch.record_observed.options());
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
  // Registered after the encoder and the reconstruction head so their
  // initialization order is unchanged. This is one attention block over the
  // cached state, not a second induced-set stack.
  query_ = register_module(
      "query", MAB(config_.d_model, config_.heads, config_.feedforward, config_.dropout));
  // The probe head is drawn here, then the CPU generator is restored.
  // Encoder, reconstruction, and query-block initialization stay on the
  // previous stream, and a later model constructed in the same process does
  // too. LibTorch exposes the process generator as const; the object is the
  // mutable default, and set_state is the call that puts the stream back.
  const auto& cpu_generator = at::detail::getDefaultCPUGenerator();
  const auto cpu_state = cpu_generator.get_state();
  query_scalar_head_ = register_module(
      "query_scalar_head", torch::nn::Linear(config_.d_model, config_.scalar_features));
  const_cast<at::Generator&>(cpu_generator).set_state(cpu_state);
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

torch::Tensor StateFusionModelImpl::query(const torch::Tensor& state, const SetBatch& query_batch) {
  for (const auto& parameter : parameters()) {
    if (!parameter.is_cuda()) {
      throw std::invalid_argument("state fusion requires CUDA parameters");
    }
  }
  require(state.defined() && state.is_cuda() && state.scalar_type() == torch::kFloat,
          "state fusion requires a CUDA float32 market state");
  require(state.dim() == 3 && state.size(0) >= 1 && state.size(1) == config_.pma_seeds &&
              state.size(2) == config_.d_model,
          "market state must be [batch, seeds, d_model]");
  require(torch::isfinite(state).all().item<bool>(), "market state must be finite");
  check_query(config_, query_batch);
  require(query_batch.scalars.size(0) == state.size(0),
          "query batch must match the market-state batch");
  require(query_batch.scalars.device() == state.device(),
          "query batch must be on the market-state device");

  const auto queries = tokens_->forward(query_batch).permute({1, 0, 2}).contiguous();
  const auto keys = state.permute({1, 0, 2}).contiguous();
  return query_->forward(queries, keys).permute({1, 0, 2}).contiguous();
}

torch::Tensor StateFusionModelImpl::query_scalars(const torch::Tensor& state, const SetBatch& query_batch) {
  return query_scalar_head_(query(state, query_batch));
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

namespace {

constexpr int64_t kCheckpointVersion = 1;

int64_t read_int(torch::serialize::InputArchive& archive, const char* name) {
  c10::IValue value;
  archive.read(name, value);
  require(value.isInt(), std::string("checkpoint field ") + name + " must be an integer");
  return value.toInt();
}

double read_double(torch::serialize::InputArchive& archive, const char* name) {
  c10::IValue value;
  archive.read(name, value);
  require(value.isDouble(), std::string("checkpoint field ") + name + " must be a double");
  return value.toDouble();
}

std::string read_string(torch::serialize::InputArchive& archive, const char* name) {
  c10::IValue value;
  archive.read(name, value);
  require(value.isString(), std::string("checkpoint field ") + name + " must be a string");
  return value.toStringRef();
}

}  // namespace

void save_checkpoint(const StateFusionModel& model, const std::string& path) {
  require(!path.empty(), "checkpoint path must be non-empty");
  const auto& config = model->config();
  torch::serialize::OutputArchive archive;
  archive.write("format_version", c10::IValue(kCheckpointVersion));
  archive.write("family", c10::IValue("state_fusion"));
  archive.write("scalar_features", c10::IValue(config.scalar_features));
  archive.write("semantic_width", c10::IValue(config.semantic_width));
  archive.write("d_model", c10::IValue(config.d_model));
  archive.write("heads", c10::IValue(config.heads));
  archive.write("inducing_points", c10::IValue(config.inducing_points));
  archive.write("isab_layers", c10::IValue(config.isab_layers));
  archive.write("pma_seeds", c10::IValue(config.pma_seeds));
  archive.write("feedforward", c10::IValue(config.feedforward));
  archive.write("dropout", c10::IValue(config.dropout));
  archive.write("categorical_cardinalities",
                torch::tensor(config.categorical_cardinalities, torch::kLong), true);
  torch::serialize::OutputArchive weights(archive.compilation_unit());
  model->save(weights);
  archive.write("model", weights);
  archive.save_to(path);
}

StateFusionModel load_checkpoint(const std::string& path, torch::Device device) {
  require(device.is_cuda(), "state fusion requires CUDA parameters");
  require(!path.empty() && std::filesystem::is_regular_file(path),
          "checkpoint path must name an existing file");
  torch::NoGradGuard guard;
  torch::serialize::InputArchive archive;
  archive.load_from(path, torch::kCPU);
  require(read_int(archive, "format_version") == kCheckpointVersion, "unsupported checkpoint format");
  require(read_string(archive, "family") == "state_fusion", "checkpoint family must be state_fusion");

  Config config;
  config.scalar_features = read_int(archive, "scalar_features");
  config.semantic_width = read_int(archive, "semantic_width");
  config.d_model = read_int(archive, "d_model");
  config.heads = read_int(archive, "heads");
  config.inducing_points = read_int(archive, "inducing_points");
  config.isab_layers = read_int(archive, "isab_layers");
  config.pma_seeds = read_int(archive, "pma_seeds");
  config.feedforward = read_int(archive, "feedforward");
  config.dropout = read_double(archive, "dropout");
  torch::Tensor cards;
  archive.read("categorical_cardinalities", cards, true);
  require(cards.defined() && cards.dim() == 1 && cards.scalar_type() == torch::kLong,
          "checkpoint cardinalities must be an int64 vector");
  cards = cards.cpu().contiguous();
  config.categorical_cardinalities.assign(cards.data_ptr<int64_t>(),
                                          cards.data_ptr<int64_t>() + cards.numel());

  StateFusionModel model(config);
  torch::serialize::InputArchive weights;
  archive.read("model", weights);
  model->load(weights);
  model->to(device);
  return model;
}

}  // namespace state_fusion
