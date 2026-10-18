// SPDX-License-Identifier: MIT
#include "embedding/shared/data.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace embedding {
namespace {

void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}

void validate_shape(const input_shape_t &shape) {
  require(shape.channel_count > 0 && shape.history_length > 0 &&
              shape.input_width > 0,
          "input channel_count, history_length and input_width must be positive");
}

} // namespace

namespace archive {

void validate_batch(const Batch &batch, const input_shape_t &shape) {
  validate_shape(shape);
  require(batch.data.defined() && batch.data.scalar_type() == torch::kFloat32 &&
              batch.data.dim() == 4 && batch.data.size(0) > 0,
          "data must be nonempty float32 [B,C,H,F]");
  require(batch.data.size(1) == shape.channel_count &&
              batch.data.size(2) == shape.history_length &&
              batch.data.size(3) == shape.input_width,
          "data C/H/F dimensions do not match model configuration");
  require(batch.feature_mask.defined() &&
              batch.feature_mask.scalar_type() == torch::kBool &&
              batch.feature_mask.sizes() == batch.data.sizes(),
          "feature_mask must be bool with the same [B,C,H,F] shape as data");
  require(torch::isfinite(batch.data)
              .logical_or(batch.feature_mask.logical_not())
              .all()
              .item<bool>(),
          "observed data contains NaN or infinity");
}

void save_archive(const std::string &path,
                  torch::serialize::OutputArchive &value) {
  require(!path.empty(), "output path is empty");
  const std::filesystem::path destination(path);
  if (!destination.parent_path().empty())
    std::filesystem::create_directories(destination.parent_path());
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto temporary = path + ".tmp-" + std::to_string(nonce);
  require(!std::filesystem::exists(temporary),
          "temporary output already exists: " + temporary);
  value.save_to(temporary);
  std::filesystem::rename(temporary, destination);
}

void distinct_paths(const std::string &input, const std::string &output) {
  if (!input.empty())
    require(std::filesystem::weakly_canonical(input) !=
                std::filesystem::weakly_canonical(output),
            "input and output paths must differ");
}

torch::Tensor text_tensor(const std::string &text) {
  return torch::tensor(std::vector<uint8_t>(text.begin(), text.end()),
                       torch::kUInt8);
}

std::string tensor_text(const torch::Tensor &tensor) {
  auto bytes = tensor.to(torch::kCPU).contiguous();
  require(bytes.scalar_type() == torch::kUInt8 && bytes.dim() == 1,
          "invalid checkpoint config");
  return std::string(reinterpret_cast<const char *>(bytes.data_ptr<uint8_t>()),
                     bytes.numel());
}

} // namespace archive

Batch synthetic_batch(const input_shape_t &shape, int64_t samples,
                      int64_t step) {
  using torch::indexing::Slice;
  validate_shape(shape);
  require(shape.input_width == 3,
          "synthetic data requires input_width=3; supply --input for other widths");
  require(samples > 0 && step >= 0,
          "samples must be positive and step nonnegative");
  constexpr double pi = 3.14159265358979323846;
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto t = torch::arange(shape.history_length, options);
  auto normalized =
      t / static_cast<double>(std::max<int64_t>(1, shape.history_length - 1));
  auto data = torch::zeros(
      {samples, shape.channel_count, shape.history_length, shape.input_width},
      options);
  for (int64_t b = 0; b < samples; ++b) {
    const int64_t family = (step + b) % 4;
    for (int64_t c = 0; c < shape.channel_count; ++c) {
      const double phase = 0.13 * static_cast<double>(step + b + c);
      const double gain = 1.0 + 0.10 * static_cast<double>(c);
      auto wave = gain * torch::sin(2.0 * pi * normalized * 4.0 + phase);
      auto trend = normalized * (0.25 + 0.05 * static_cast<double>(c)) +
                   static_cast<double>(b) * 0.01;
      if (family == 1)
        wave = torch::where(t.lt(shape.history_length / 2),
                            torch::sin(2.0 * pi * normalized * 3.0 + phase),
                            torch::sin(2.0 * pi * normalized * 7.0 + phase));
      else if (family == 2)
        wave = gain * torch::sin(2.0 * pi * normalized * 5.0 + phase) +
               0.20 * torch::cos(2.0 * pi * normalized * (2.0 + c) + phase);
      else if (family == 3)
        trend = trend + 0.15 * torch::sin(2.0 * pi * normalized);
      auto noise = torch::randn({shape.history_length}, options) *
                   (family == 3 ? 0.06 : 0.03);
      data.index_put_({b, c, Slice(), Slice()},
                     torch::stack({wave, trend, noise}, 1));
    }
  }
  auto mask = torch::ones(data.sizes(), options.dtype(torch::kBool));
  for (int64_t b = 0; b < samples; ++b) {
    const double missing =
        (step + b) % 4 == 3 ? 0.30 : (step % 5 == 0 ? 0.08 : 0.0);
    if (missing > 0)
      mask[b].copy_(torch::rand(data[b].sizes(), options).gt(missing));
  }
  return {torch::where(mask, data, torch::zeros_like(data)), mask};
}

void save_batch(const std::string &path, const Batch &batch) {
  torch::serialize::OutputArchive value;
  value.write("data", batch.data.detach().to(torch::kCPU), true);
  value.write("feature_mask", batch.feature_mask.detach().to(torch::kCPU), true);
  archive::save_archive(path, value);
}

Batch load_batch(const std::string &path, const input_shape_t &shape) {
  torch::serialize::InputArchive value;
  value.load_from(path, torch::kCPU);
  Batch batch;
  value.read("data", batch.data, true);
  value.read("feature_mask", batch.feature_mask, true);
  archive::validate_batch(batch, shape);
  batch.data =
      torch::where(batch.feature_mask, batch.data, torch::zeros_like(batch.data));
  return batch;
}

} // namespace embedding
