#include "praesidium-humanitatis/checkpoint.h"

#include <torch/cuda.h>
#include <torch/serialize.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using torch::indexing::Slice;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

void require_close(const torch::Tensor& actual, const torch::Tensor& expected,
                   const std::string& message, double atol = 0.0, double rtol = 0.0) {
  require(actual.sizes() == expected.sizes(), message + " (shape)");
  require(torch::allclose(actual, expected, rtol, atol), message);
}

void require_finite(const torch::Tensor& tensor, const std::string& message) {
  require(tensor.defined() && torch::isfinite(tensor).all().item<bool>(), message);
}

praesidium_humanitatis::Config make_config() {
  praesidium_humanitatis::Config config;
  config.hidden_size = 8;
  config.attention_heads = 2;
  config.dropout = 0.0;
  config.static_continuous = 1;
  config.past_continuous = 2;
  config.future_continuous = 1;
  config.static_categorical_cardinalities = {3};
  config.past_categorical_cardinalities = {4};
  config.future_categorical_cardinalities = {5};
  config.quantiles = {0.1, 0.5, 0.9};
  config.static_vector = 3;
  config.past_vector = 4;
  config.future_vector = 5;
  return config;
}

torch::Tensor categories(const std::vector<int64_t>& cardinalities,
                         const std::vector<int64_t>& leading, const torch::Device& device) {
  const auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
  std::vector<torch::Tensor> columns;
  for (const auto cardinality : cardinalities) {
    columns.push_back(torch::randint(cardinality, leading, options));
  }
  if (!columns.empty()) return torch::stack(columns, -1);
  auto shape = leading;
  shape.push_back(0);
  return torch::empty(shape, options);
}

void copy_group(praesidium_humanitatis::FeatureBatch& destination,
                const praesidium_humanitatis::FeatureBatch& source) {
  destination.continuous = source.continuous.defined() ? source.continuous.clone() : torch::Tensor();
  destination.categorical = source.categorical.defined() ? source.categorical.clone() : torch::Tensor();
  destination.vector = source.vector.defined() ? source.vector.clone() : torch::Tensor();
}

praesidium_humanitatis::Batch clone_batch(const praesidium_humanitatis::Batch& batch) {
  praesidium_humanitatis::Batch copy;
  copy_group(copy.statics, batch.statics);
  copy_group(copy.past, batch.past);
  copy_group(copy.future, batch.future);
  if (batch.past_observed.defined()) copy.past_observed = batch.past_observed.clone();
  if (batch.future_observed.defined()) copy.future_observed = batch.future_observed.clone();
  return copy;
}

praesidium_humanitatis::Batch make_batch(const praesidium_humanitatis::Config& config,
                                         const torch::Device& device, int64_t batch_size = 2,
                                         int64_t history = 4, int64_t horizon = 3) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  praesidium_humanitatis::Batch batch;
  batch.statics.continuous = torch::randn({batch_size, config.static_continuous}, options);
  batch.statics.categorical =
      categories(config.static_categorical_cardinalities, {batch_size}, device);
  batch.past.continuous = torch::randn({batch_size, history, config.past_continuous}, options);
  batch.past.categorical =
      categories(config.past_categorical_cardinalities, {batch_size, history}, device);
  batch.future.continuous = torch::randn({batch_size, horizon, config.future_continuous}, options);
  batch.future.categorical =
      categories(config.future_categorical_cardinalities, {batch_size, horizon}, device);
  if (config.static_vector > 0) {
    batch.statics.vector = torch::randn({batch_size, config.static_vector}, options);
  }
  if (config.past_vector > 0) {
    batch.past.vector = torch::randn({batch_size, history, config.past_vector}, options);
  }
  if (config.future_vector > 0) {
    batch.future.vector = torch::randn({batch_size, horizon, config.future_vector}, options);
  }
  return batch;
}

torch::Tensor ones_mask(int64_t batch, int64_t time, const torch::Device& device) {
  return torch::ones({batch, time}, torch::TensorOptions().dtype(torch::kBool).device(device));
}

praesidium_humanitatis::TFT make_model(const praesidium_humanitatis::Config& config,
                                       const torch::Device& device) {
  praesidium_humanitatis::TFT model(config);
  model->to(device);
  model->eval();
  return model;
}

void mark_observed(praesidium_humanitatis::Batch& batch) {
  batch.past_observed = ones_mask(batch.past.continuous.size(0), batch.past.continuous.size(1),
                                  batch.past.continuous.device());
  batch.future_observed = ones_mask(batch.future.continuous.size(0), batch.future.continuous.size(1),
                                    batch.future.continuous.device());
}

void test_vector_changes_forecast(const torch::Device& device) {
  const auto config = make_config();
  auto model = make_model(config, device);
  auto batch = make_batch(config, device);
  torch::NoGradGuard no_grad;
  const auto baseline = model->forward(batch);
  require(baseline.past_weights.size(-1) == config.past_continuous + 1 + 1,
          "past weights gain one column for the pooled vector");
  require(baseline.static_weights.size(-1) == config.static_continuous + 1 + 1,
          "static weights gain one column for the pooled vector");
  require(baseline.future_weights.size(-1) == config.future_continuous + 1 + 1,
          "future weights gain one column for the pooled vector");
  auto changed = clone_batch(batch);
  changed.past.vector.mul_(4).add_(1);
  require(!torch::allclose(model->forward(changed).predictions, baseline.predictions, 0, 0),
          "a pooled past vector must be able to move the forecast");
  bool found = false;
  for (const auto& parameter : model->named_parameters()) {
    if (parameter.key() == "network.past_embedding.vector.weight") {
      found = true;
      require(parameter.value().size(0) == config.hidden_size &&
                  parameter.value().size(1) == config.past_vector,
              "vector projection is Linear(width, hidden)");
    }
  }
  require(found, "past vector projection must be registered");
}

void test_all_true_matches_bare(const torch::Device& device) {
  const auto config = make_config();
  auto model = make_model(config, device);
  const auto bare = make_batch(config, device);
  auto masked = clone_batch(bare);
  mark_observed(masked);
  torch::NoGradGuard no_grad;
  const auto expected = model->forward(bare);
  const auto actual = model->forward(masked);
  require_close(actual.predictions, expected.predictions, "all-true mask must match an absent mask");
  require_close(actual.static_weights, expected.static_weights, "all-true mask must preserve static weights");
  require_close(actual.past_weights, expected.past_weights, "all-true mask must preserve past weights");
  require_close(actual.future_weights, expected.future_weights, "all-true mask must preserve future weights");
  require_close(actual.attention_weights, expected.attention_weights,
                "all-true mask must preserve attention");
}

void require_unchanged_forecast(praesidium_humanitatis::TFT& model,
                                const praesidium_humanitatis::Batch& original,
                                praesidium_humanitatis::Batch& dirty, int64_t time,
                                praesidium_humanitatis::FeatureBatch& group, const std::string& name) {
  group.continuous.select(1, time).copy_(torch::randn_like(group.continuous.select(1, time)));
  group.vector.select(1, time).copy_(torch::randn_like(group.vector.select(1, time)));
  group.categorical.select(1, time).zero_();
  torch::NoGradGuard no_grad;
  require_close(model->forward(dirty).predictions, model->forward(original).predictions,
                name + " content must not move the forecast");
}

void test_hole_content_is_ignored(const torch::Device& device) {
  const auto config = make_config();
  auto model = make_model(config, device);
  auto past_hole = make_batch(config, device);
  mark_observed(past_hole);
  past_hole.past_observed.select(1, 1).fill_(false);
  auto dirty_past = clone_batch(past_hole);
  require_unchanged_forecast(model, past_hole, dirty_past, 1, dirty_past.past, "past hole");
  auto moved_past = clone_batch(past_hole);
  moved_past.past.continuous.select(1, 0).add_(3);
  {
    torch::NoGradGuard no_grad;
    require(!torch::allclose(model->forward(moved_past).predictions,
                             model->forward(past_hole).predictions, 1e-6, 1e-6),
            "an observed past step must still be able to move the forecast");
  }

  auto future_hole = make_batch(config, device);
  mark_observed(future_hole);
  future_hole.future_observed.select(1, 1).fill_(false);
  auto dirty_future = clone_batch(future_hole);
  require_unchanged_forecast(model, future_hole, dirty_future, 1, dirty_future.future, "future hole");
  auto moved_future = clone_batch(future_hole);
  moved_future.future.vector.select(1, 0).add_(3);
  {
    torch::NoGradGuard no_grad;
    require(!torch::allclose(model->forward(moved_future).predictions,
                             model->forward(future_hole).predictions, 1e-6, 1e-6),
            "an observed future step must still be able to move the forecast");
  }
}

void test_observed_zero_differs_from_hole(const torch::Device& device) {
  torch::manual_seed(11);
  const auto config = make_config();
  auto model = make_model(config, device);
  auto batch = make_batch(config, device);
  batch.past.continuous.select(1, 2).zero_();
  batch.past.vector.select(1, 2).zero_();
  batch.past.categorical.select(1, 2).zero_();
  auto stepped = clone_batch(batch);
  mark_observed(stepped);
  auto held = clone_batch(batch);
  mark_observed(held);
  held.past_observed.select(1, 2).fill_(false);
  torch::NoGradGuard no_grad;
  const auto stepped_forecast = model->forward(stepped).predictions;
  const auto held_forecast = model->forward(held).predictions;
  require(!torch::allclose(stepped_forecast, held_forecast, 1e-5, 1e-5),
          "an observed zero must advance time, while a hole holds it");
}

void test_encoder_holds_state(const torch::Device& device) {
  const auto config = make_config();
  auto model = make_model(config, device);
  const auto base = make_batch(config, device);
  auto longer = clone_batch(base);
  mark_observed(longer);
  longer.past_observed.select(1, 2).fill_(false);
  auto shifted = clone_batch(base);
  shifted.past.continuous.select(1, 2).copy_(base.past.continuous.select(1, 3));
  shifted.past.vector.select(1, 2).copy_(base.past.vector.select(1, 3));
  shifted.past.categorical.select(1, 2).copy_(base.past.categorical.select(1, 3));
  mark_observed(shifted);
  shifted.past_observed.select(1, 3).fill_(false);
  torch::NoGradGuard no_grad;
  std::map<std::string, torch::Tensor> diagnostics;
  model->forward(longer, &diagnostics);
  const auto longer_encoder = diagnostics.at("encoder").clone();
  model->forward(shifted, &diagnostics);
  const auto shifted_encoder = diagnostics.at("encoder").clone();
  require(longer_encoder.select(1, 2).abs().max().item<double>() == 0.0, "a past hole emits zeros");
  require(shifted_encoder.select(1, 3).abs().max().item<double>() == 0.0, "a trailing past hole emits zeros");
  require_close(longer_encoder.select(1, 0), shifted_encoder.select(1, 0),
                "held prefixes must share the first state");
  require_close(longer_encoder.select(1, 1), shifted_encoder.select(1, 1),
                "held prefixes must share the second state");
  require_close(longer_encoder.select(1, 3), shifted_encoder.select(1, 2),
                "the next observed step must continue from the state before the hole");

  auto bare = clone_batch(base);
  model->forward(bare, &diagnostics);
  const auto bare_encoder = diagnostics.at("encoder").clone();
  auto last_hole = clone_batch(base);
  mark_observed(last_hole);
  last_hole.past_observed.select(1, 3).fill_(false);
  model->forward(last_hole, &diagnostics);
  const double tolerance = device.is_cuda() ? 1e-4 : 1e-5;
  require_close(diagnostics.at("encoder").narrow(1, 0, 3), bare_encoder.narrow(1, 0, 3),
                "observed steps before a hole must match the unmasked LSTM", tolerance, tolerance);
}

void test_decoder_holds_state(const torch::Device& device) {
  const auto config = make_config();
  auto model = make_model(config, device);
  const auto base = make_batch(config, device);
  auto longer = clone_batch(base);
  mark_observed(longer);
  longer.future_observed.select(1, 1).fill_(false);
  auto shifted = clone_batch(base);
  shifted.future.continuous.select(1, 1).copy_(base.future.continuous.select(1, 2));
  shifted.future.vector.select(1, 1).copy_(base.future.vector.select(1, 2));
  shifted.future.categorical.select(1, 1).copy_(base.future.categorical.select(1, 2));
  mark_observed(shifted);
  shifted.future_observed.select(1, 2).fill_(false);
  torch::NoGradGuard no_grad;
  std::map<std::string, torch::Tensor> diagnostics;
  model->forward(longer, &diagnostics);
  const auto longer_decoder = diagnostics.at("decoder").clone();
  model->forward(shifted, &diagnostics);
  const auto shifted_decoder = diagnostics.at("decoder").clone();
  require(longer_decoder.select(1, 1).abs().max().item<double>() == 0.0, "a future hole emits zeros");
  require_close(longer_decoder.select(1, 0), shifted_decoder.select(1, 0),
                "decoder prefixes must match before a hole");
  require_close(longer_decoder.select(1, 2), shifted_decoder.select(1, 1),
                "the next known future step must continue from the state before the hole");
}

void test_masked_loss(const torch::Device& device) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const std::vector<double> quantiles{0.1, 0.5, 0.9};
  auto predictions = torch::randn({2, 4, 3}, options);
  auto targets = torch::randn({2, 4}, options);
  auto observed = torch::tensor({1, 1, 0, 1, 0, 1, 1, 1},
                                torch::TensorOptions().dtype(torch::kBool).device(device))
                      .reshape({2, 4});
  const auto loss = praesidium_humanitatis::quantile_loss(predictions, targets, quantiles, observed);
  const auto rows = observed.reshape({-1});
  auto kept_predictions = predictions.reshape({-1, 3}).index({rows}).reshape({1, -1, 3});
  auto kept_targets = targets.reshape({-1}).index({rows}).reshape({1, -1});
  require_close(loss, praesidium_humanitatis::quantile_loss(kept_predictions, kept_targets, quantiles),
                "masked pinball must average only observed targets");
  auto wild = targets.clone();
  wild.index_put_({0, 2}, 1.0e9);
  wild.index_put_({1, 0}, std::numeric_limits<float>::quiet_NaN());
  require_close(praesidium_humanitatis::quantile_loss(predictions, wild, quantiles, observed), loss,
                "a missing target must not enter the loss");
  auto all_true = torch::ones_like(observed);
  require_close(praesidium_humanitatis::quantile_loss(predictions, targets, quantiles, all_true),
                praesidium_humanitatis::quantile_loss(predictions, targets, quantiles),
                "an all-true target mask must match an absent mask");
  require_throws([&] {
    praesidium_humanitatis::quantile_loss(predictions, targets, quantiles, torch::zeros_like(observed));
  }, "a fully missing target window must be rejected");
  require_throws([&] {
    praesidium_humanitatis::quantile_loss(
        predictions, torch::full_like(targets, std::numeric_limits<float>::quiet_NaN()), quantiles, observed);
  }, "a non-finite observed target must be rejected");
}

void test_hidden_queries_stay_finite(const torch::Device& device) {
  const auto config = make_config();
  auto model = make_model(config, device);
  auto hidden = make_batch(config, device);
  mark_observed(hidden);
  hidden.past_observed.fill_(false);
  hidden.future_observed.fill_(false);
  torch::NoGradGuard no_grad;
  const auto output = model->forward(hidden);
  require_finite(output.predictions, "a fully hidden window must stay finite");
  require(output.attention_weights.abs().max().item<double>() == 0.0,
          "a query with no allowed key must have zero attention");

  auto past_hidden = make_batch(config, device);
  mark_observed(past_hidden);
  past_hidden.past_observed.fill_(false);
  const auto attended = model->forward(past_hidden);
  require_finite(attended.predictions, "future queries over a hidden history must stay finite");
  const auto history = past_hidden.past.continuous.size(1);
  const auto first = attended.attention_weights.select(2, 0);
  auto expected = torch::zeros_like(first);
  expected.select(-1, history).fill_(1);
  require_close(first, expected, "the first future query can attend only to its own observed key");
}

void test_backward_through_hole(const torch::Device& device) {
  const auto config = make_config();
  auto model = make_model(config, device);
  model->train();
  auto batch = make_batch(config, device);
  mark_observed(batch);
  batch.past_observed.select(1, 1).fill_(false);
  batch.past.continuous.requires_grad_(true);
  batch.past.vector.requires_grad_(true);
  const auto targets = torch::randn({batch.past.continuous.size(0), batch.future.continuous.size(1)},
                                    batch.past.continuous.options());
  const auto loss = praesidium_humanitatis::quantile_loss(model->forward(batch).predictions, targets,
                                                          config.quantiles, batch.future_observed);
  require_finite(loss, "loss through a hole must be finite");
  loss.backward();
  require(batch.past.continuous.grad().defined() && batch.past.vector.grad().defined(),
          "observed inputs must receive gradients");
  require(batch.past.continuous.grad().select(1, 1).abs().max().item<double>() == 0.0 &&
              batch.past.vector.grad().select(1, 1).abs().max().item<double>() == 0.0,
          "a hole must not receive an input gradient");
  require(batch.past.continuous.grad().select(1, 0).abs().sum().item<double>() > 0.0 ||
              batch.past.vector.grad().select(1, 0).abs().sum().item<double>() > 0.0,
          "an observed step must receive an input gradient");
  for (const auto& parameter : model->named_parameters()) {
    require_finite(parameter.value().grad(), "gradient for " + parameter.key());
  }
}

void test_vector_only_group(const torch::Device& device) {
  praesidium_humanitatis::Config config;
  config.hidden_size = 8;
  config.attention_heads = 2;
  config.dropout = 0.0;
  config.static_continuous = 0;
  config.past_continuous = 0;
  config.future_continuous = 0;
  config.static_vector = 4;
  config.past_vector = 4;
  config.future_vector = 4;
  auto model = make_model(config, device);
  auto batch = make_batch(config, device, 2, 3, 2);
  torch::NoGradGuard no_grad;
  const auto output = model->forward(batch);
  require_finite(output.predictions, "vector-only inputs must forecast");
  require(output.static_weights.size(-1) == 1 && output.past_weights.size(-1) == 1 &&
              output.future_weights.size(-1) == 1,
          "a vector is one selected variable");
  auto changed = clone_batch(batch);
  changed.statics.vector.add_(2);
  require(!torch::allclose(model->forward(changed).predictions, output.predictions, 0, 0),
          "a static pooled vector must be able to move the forecast");
}

void test_checkpoint_widths(const torch::Device& device) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("praesidium-mask-checkpoint-" + std::to_string(stamp));
  std::filesystem::create_directory(directory);
  const auto path = (directory / "training.pt").string();
  try {
    const auto config = make_config();
    auto model = make_model(config, device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.01));
    praesidium_humanitatis::TrainingProgress progress;
    progress.data_id = "mask-width-roundtrip";
    praesidium_humanitatis::save_checkpoint(path, model, optimizer, progress);
    torch::serialize::InputArchive archive;
    archive.load_from(path, torch::kCPU);
    c10::IValue version;
    archive.read("format_version", version);
    require(version.isInt() && version.toInt() == 4, "vector widths require checkpoint format 4");
    auto restored = praesidium_humanitatis::load_checkpoint(path, device);
    require(restored.model->config.static_vector == config.static_vector &&
                restored.model->config.past_vector == config.past_vector &&
                restored.model->config.future_vector == config.future_vector,
            "checkpoint must restore vector widths");
    auto batch = make_batch(config, device);
    mark_observed(batch);
    batch.past_observed.select(1, 1).fill_(false);
    torch::NoGradGuard no_grad;
    require_close(restored.model->forward(batch).predictions, model->forward(batch).predictions,
                  "a restored vector model must keep the holed forecast");
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    throw;
  }
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
}

void test_width_zero_adds_no_parameter(const torch::Device& device) {
  auto config = make_config();
  config.static_vector = 0;
  config.past_vector = 0;
  config.future_vector = 0;
  auto model = make_model(config, device);
  for (const auto& parameter : model->named_parameters()) {
    require(parameter.key().find(".vector.") == std::string::npos,
            "width zero must not register a vector projection");
  }
  auto batch = make_batch(config, device);
  batch.past.vector = torch::randn({2, 4, 3}, batch.past.continuous.options());
  require_throws([&] { model->forward(batch); }, "a defined vector must be rejected when width is zero");
}

void test_rejections(const torch::Device& device) {
  require_throws([&] {
    auto config = make_config();
    config.past_vector = -1;
    praesidium_humanitatis::TFT model(config);
  }, "a negative vector width must be rejected");
  require_throws([&] {
    praesidium_humanitatis::Config config;
    config.hidden_size = 8;
    config.attention_heads = 2;
    config.static_continuous = 0;
    config.past_continuous = 0;
    config.future_continuous = 0;
    praesidium_humanitatis::TFT model(config);
  }, "a group with no variables must be rejected");
  const auto config = make_config();
  auto model = make_model(config, device);
  auto missing = make_batch(config, device);
  missing.past.vector = torch::Tensor();
  require_throws([&] { model->forward(missing); }, "a missing vector must be rejected");
  auto nonfinite = make_batch(config, device);
  nonfinite.future.vector.index_put_({0, 0, 0}, std::numeric_limits<float>::quiet_NaN());
  require_throws([&] { model->forward(nonfinite); }, "a non-finite vector must be rejected");
  auto bad_mask = make_batch(config, device);
  bad_mask.past_observed = torch::ones({bad_mask.past.continuous.size(0), bad_mask.past.continuous.size(1)},
                                       bad_mask.past.continuous.options());
  require_throws([&] { model->forward(bad_mask); }, "a floating observed mask must be rejected");
}

void test_double_hole(const torch::Device& device) {
  const auto config = make_config();
  auto model = make_model(config, device);
  model->to(torch::kFloat64);
  auto batch = make_batch(config, device);
  for (auto* group : {&batch.statics, &batch.past, &batch.future}) {
    group->continuous = group->continuous.to(torch::kFloat64);
    group->vector = group->vector.to(torch::kFloat64);
  }
  mark_observed(batch);
  batch.future_observed.select(1, 0).fill_(false);
  auto dirty = clone_batch(batch);
  dirty.future.continuous.select(1, 0).add_(8);
  dirty.future.vector.select(1, 0).mul_(5);
  torch::NoGradGuard no_grad;
  require_close(model->forward(dirty).predictions, model->forward(batch).predictions,
                "a float64 future hole must ignore its content");
}

}  // namespace

int main(int argc, char** argv) {
  torch::Device device(torch::kCPU);
  if (argc == 3 && std::string(argv[1]) == "--device") {
    const std::string requested(argv[2]);
    if (requested == "cuda") {
      if (!torch::cuda::is_available()) {
        std::cout << "SKIP: CUDA requested, but no CUDA device is available\n";
        return 77;
      }
      device = torch::Device(torch::kCUDA, 0);
    } else if (requested != "cpu") {
      std::cerr << "Usage: tft_mask_test [--device cpu|cuda]\n";
      return 2;
    }
  } else if (argc != 1) {
    std::cerr << "Usage: tft_mask_test [--device cpu|cuda]\n";
    return 2;
  }
  torch::set_num_threads(1);
  const std::vector<std::pair<std::string, std::function<void()>>> tests{
      {"pooled vector changes the forecast", [&] { test_vector_changes_forecast(device); }},
      {"all-true mask matches an absent mask", [&] { test_all_true_matches_bare(device); }},
      {"hole content does not move the forecast", [&] { test_hole_content_is_ignored(device); }},
      {"observed zero differs from a hole", [&] { test_observed_zero_differs_from_hole(device); }},
      {"encoder holds state across a hole", [&] { test_encoder_holds_state(device); }},
      {"decoder holds state across a hole", [&] { test_decoder_holds_state(device); }},
      {"masked pinball ignores a missing target", [&] { test_masked_loss(device); }},
      {"a fully hidden query stays finite", [&] { test_hidden_queries_stay_finite(device); }},
      {"backward through a hole stays finite", [&] { test_backward_through_hole(device); }},
      {"vector-only input group", [&] { test_vector_only_group(device); }},
      {"checkpoint format 4 restores vector widths", [&] { test_checkpoint_widths(device); }},
      {"width zero adds no vector parameter", [&] { test_width_zero_adds_no_parameter(device); }},
      {"invalid widths, vectors and masks", [&] { test_rejections(device); }},
      {"float64 hole content is ignored", [&] { test_double_hole(device); }},
  };
  int failures = 0;
  for (const auto& test : tests) {
    torch::manual_seed(2026);
    try {
      test.second();
      std::cout << "PASS: " << test.first << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
    }
  }
  std::cout << tests.size() - failures << '/' << tests.size() << " tests passed on " << device << '\n';
  return failures == 0 ? 0 : 1;
}
