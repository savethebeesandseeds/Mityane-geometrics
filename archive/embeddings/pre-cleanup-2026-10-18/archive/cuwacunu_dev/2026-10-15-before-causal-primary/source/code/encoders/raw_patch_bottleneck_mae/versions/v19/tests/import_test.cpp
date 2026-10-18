// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#include "cuwacunu/encoder/encoder.h"
#include <torch/cuda.h>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
namespace encoder = cuwacunu::encoder;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;

void check(bool condition, const char *message) {
  TORCH_CHECK(condition, "[rpb-v19 import test] ", message);
}
template <typename Callable> void rejects(Callable action, const char *message) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  check(rejected, message);
}
encoder::Input artificial_input(int64_t batch) {
  auto time = torch::arange(32, torch::kFloat64).reshape({1, 1, 32, 1});
  auto channels = torch::arange(3, torch::kFloat64).reshape({1, 3, 1, 1});
  auto features = torch::arange(3, torch::kFloat64).reshape({1, 1, 1, 3});
  auto examples = torch::arange(batch, torch::kFloat64).reshape({batch, 1, 1, 1});
  auto values = (time * .31 + channels * .42 + examples * .13).sin() * (1. + features * .1) +
                (time * 1.1 - channels * .25).cos() * .6 + features * .2;
  auto mask = torch::ones_like(values, torch::kBool);
  mask[0][2].narrow(0, 9, 3).fill_(false);
  return {values, mask, torch::tensor({0, 1, 2}, torch::kInt64),
          torch::full({batch}, 31., torch::kFloat64), 1.};
}
void put_text(torch::serialize::OutputArchive &archive, const char *key, const std::string &text) {
  archive.write(key, torch::tensor(std::vector<uint8_t>(text.begin(), text.end()), torch::kUInt8), true);
}
void put_integer(torch::serialize::OutputArchive &archive, const char *key, int64_t value) {
  archive.write(key, torch::tensor(value, torch::kInt64), true);
}
std::string fixture_checkpoint(encoder::Encoder &model, const encoder::FrozenScaler &scaler,
                               const std::string &layout = encoder::native_layout) {
  torch::serialize::OutputArchive checkpoint, weights, fitted;
  put_text(checkpoint, "artifact_kind", "rpb_spectral_confirmation_checkpoint_v1");
  put_text(checkpoint, "candidate", "v19");
  put_text(checkpoint, "protocol", "spectral-confirmation-v1");
  put_text(checkpoint, "architecture_id", encoder::architecture_id);
  put_text(checkpoint, "native_layout", layout);
  put_text(checkpoint, "training_recipe_id", rpb::kSpectralTemporalRelationTrainingRecipeId);
  put_text(checkpoint, "frozen_parameter_name", "grouped_odd_relation_projection.weight");
  put_integer(checkpoint, "parameter_count", 226877);
  put_integer(checkpoint, "trainable_parameter_count", 226445);
  put_integer(checkpoint, "frozen_parameter_count", 432);
  model->save(weights); scaler.save(fitted);
  checkpoint.write("model", weights); checkpoint.write("scaler", fitted);
  std::stringstream bytes;
  checkpoint.save_to(bytes);
  return bytes.str();
}
encoder::LoadedEncoder read_fixture(const std::string &bytes, const encoder::Config &config) {
  std::stringstream stream(bytes);
  torch::serialize::InputArchive archive;
  archive.load_from(stream, config.device);
  return encoder::detail::load_confirmation_archive(archive, config);
}
void focused_import() {
  const auto config = encoder::fixed_config();
  torch::manual_seed(19101);
  encoder::Encoder model(config);
  model->eval();
  auto raw = artificial_input(3);
  const auto scaler = encoder::fit_train_scaler(raw, config);
  const auto normalized = scaler.transform(raw, config);
  check(normalized.data.is_cuda() && normalized.observed.is_cuda(), "scaled minibatch must be CUDA");
  int64_t total = 0, frozen = 0;
  for (const auto &p : model->parameters()) {
    check(p.is_cuda(), "all model parameters must be CUDA");
    total += p.numel();
    if (!p.requires_grad()) frozen += p.numel();
  }
  check(total == 226877 && frozen == 432, "imported capacity and frozen prior must match");
  torch::Tensor original;
  {
    torch::NoGradGuard no_grad;
    original = encoder::native_embedding(model->encode(normalized), config).clone();
    check(original.sizes() == torch::IntArrayRef({3, 32}), "one native32 per window");
    check(torch::isfinite(original).all().item<bool>(), "finite native output");
    auto poisoned = normalized;
    poisoned.data = torch::where(normalized.observed, normalized.data,
        torch::full_like(normalized.data, std::numeric_limits<float>::quiet_NaN()));
    check(torch::equal(original, encoder::native_embedding(model->encode(poisoned), config)),
          "masked NaNs cannot change the embedding");
    auto absent = poisoned;
    absent.observed = torch::zeros_like(normalized.observed);
    const auto empty = model->encode(absent);
    check(!empty.sample_valid_mask.any().item<bool>() &&
              encoder::native_embedding(empty, config).eq(0).all().item<bool>(),
          "fully absent windows are invalid exact zero");
    const auto single = scaler.transform(artificial_input(1), config);
    check(encoder::native_embedding(model->encode(single), config).sizes() == torch::IntArrayRef({1, 32}),
          "batch dimension remains variable");
    check(model->decode(original, normalized.channel_ids).sizes() == torch::IntArrayRef({3, 3, 32, 3}),
          "native32 reconstructs the original window shape");
  }
  auto wrong = config;
  wrong.history_length = 64;
  rejects([&] { encoder::Encoder invalid(wrong); }, "larger history must not silently become v19");
  wrong = config;
  wrong.input_width = 9;
  rejects([&] { encoder::Encoder invalid(wrong); }, "nine-feature input needs a separately adapted design");
  auto restored = read_fixture(fixture_checkpoint(model, scaler), config);
  {
    torch::NoGradGuard no_grad;
    const auto replay = encoder::native_embedding(restored.model->encode(restored.scaler.transform(raw, config)), config);
    check(torch::equal(replay, original), "checkpoint/scaler roundtrip must preserve native output exactly");
  }
  rejects([&] { read_fixture(fixture_checkpoint(model, scaler, "wrong-layout"), config); },
          "wrong native layout must be rejected");

  // One CUDA optimizer update verifies import trainability and fixed-prior
  // protection. It is a disposable correctness fixture, not a quality run.
  const auto prior = model->grouped_odd_relation_projection->weight.detach().clone();
  const auto shape = model->native_shape_projection->weight.detach().clone();
  model->train();
  torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
  const auto query = rpb::make_training_mask(normalized.observed, config, 19103);
  optimizer.zero_grad();
  const auto forward = model->forward(normalized, query.hidden);
  check(forward.loss.is_cuda() && torch::isfinite(forward.loss).item<bool>() && forward.eligible_example_count > 0,
        "masked reconstruction has a finite eligible CUDA loss");
  forward.loss.backward();
  check(!model->grouped_odd_relation_projection->weight.grad().defined(), "frozen prior must receive no gradient");
  check(model->native_shape_projection->weight.grad().is_cuda() &&
              torch::isfinite(model->native_shape_projection->weight.grad()).all().item<bool>() &&
              model->native_shape_projection->weight.grad().abs().sum().item<double>() > 0,
        "learned shape must have finite nonzero CUDA gradient");
  torch::nn::utils::clip_grad_norm_(model->parameters(), 1., 2., true);
  optimizer.step();
  model->validate_fixed_prior();
  check(torch::equal(prior, model->grouped_odd_relation_projection->weight), "optimizer must preserve the fixed prior");
  check(!torch::equal(shape, model->native_shape_projection->weight), "learned shape must update");
  check(optimizer.state().find(model->grouped_odd_relation_projection->weight.unsafeGetTensorImpl()) == optimizer.state().end(),
        "frozen prior must have no optimizer state");
  std::cout << "RPB-v19 import CUDA checks passed: variable batches, masks, fixed geometry, checkpoint/scaler roundtrip, one disposable update\n";
}

void saved_native_witness(const char *checkpoint_path, const char *input_path, const char *features_path) {
  const auto config = encoder::fixed_config();
  auto loaded = encoder::load_confirmation(checkpoint_path, config);
  torch::serialize::InputArchive observations, expected;
  observations.load_from(input_path, torch::kCPU);
  expected.load_from(features_path, torch::kCPU);
  torch::Tensor data, mask, features, valid;
  observations.read("observations", data, true);
  observations.read("feature_mask", mask, true);
  expected.read("features", features, true);
  expected.read("valid", valid, true);
  const encoder::Input raw{data, mask, torch::tensor({0, 1, 2}, torch::kInt64),
                            torch::full({data.size(0)}, 31., torch::kFloat64), 1.};
  torch::NoGradGuard no_grad;
  const auto encoded = loaded.model->encode(loaded.scaler.transform(raw, config));
  const auto actual = encoder::native_embedding(encoded, config).to(torch::kCPU, torch::kFloat64).contiguous();
  check(features.scalar_type() == torch::kFloat64 && torch::equal(actual, features),
        "imported reference checkpoint must exactly reproduce saved native32");
  check(valid.scalar_type() == torch::kBool && torch::equal(encoded.sample_valid_mask.to(torch::kCPU), valid),
        "imported validity must match saved feature validity");
  std::cout << "RPB-v19.alt-01 saved-native CUDA parity passed: " << data.size(0)
            << " windows; zero optimizer updates; no source encoder execution\n";
}
} // namespace

int main(int argc, char **argv) try {
  check(torch::cuda::is_available(), "actual CUDA is required");
  torch::set_num_threads(1);
  if (argc == 1) focused_import();
  else if (argc == 4) saved_native_witness(argv[1], argv[2], argv[3]);
  else TORCH_CHECK(false, "usage: rpb_v19_import_test [confirmation-checkpoint controlled-input saved-features]");
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
