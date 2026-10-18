// SPDX-License-Identifier: MIT
#pragma once

#include <utility>
#include "embedding/encoders/mtf_jepa_mae_vicreg/tensor_ops.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {

namespace detail {

inline std::vector<std::pair<int64_t, int64_t>>
window_plan(int64_t history_length, int64_t scale, int64_t stride) {
  const int64_t width = std::max<int64_t>(1, std::min(scale, history_length));
  std::vector<std::pair<int64_t, int64_t>> out;
  if (history_length <= width) {
    out.push_back({0, width});
    return out;
  }
  for (int64_t start = 0; start + width <= history_length; start += stride) {
    out.push_back({start, width});
    if (start + width == history_length) {
      break;
    }
  }
  const int64_t final_start = history_length - width;
  if (out.empty() || out.back().first != final_start) {
    out.push_back({final_start, width});
  }
  return out;
}

inline torch::Tensor masked_patch_descriptor(const torch::Tensor &patch,
                                             const torch::Tensor &valid,
                                             double eps = 1e-6) {
  TORCH_CHECK(patch.dim() == 3, "[embedding] patch must be [B,W,F]");
  TORCH_CHECK(valid.sizes() == patch.sizes(),
              "[embedding] patch validity shape mismatch");
  const auto mask = valid.to(torch::kBool);
  // Center before squaring to preserve small variation around large offsets.
  // Float64 accumulation also avoids squaring finite float32 values in float32.
  const auto observed =
      torch::where(mask, patch, torch::zeros_like(patch)).to(torch::kFloat64);
  const auto valid_f = mask.to(torch::kFloat64);
  const auto denom = valid_f.sum(/*dim=*/1).clamp_min(1.0); // [B,F]
  const auto mean = observed.sum(/*dim=*/1) / denom;
  const auto centered = torch::where(
      mask, observed - mean.unsqueeze(1), torch::zeros_like(observed));
  const auto var = centered.pow(2).sum(/*dim=*/1) / denom;
  const auto std = torch::sqrt(var + eps);
  return torch::cat({mean, std}, /*dim=*/1).to(patch.dtype());
}

} // namespace detail

class MultiScalePatchTokenizerImpl : public torch::nn::Module {
public:
  explicit MultiScalePatchTokenizerImpl(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
    time_projection_ = register_module(
        "time_projection",
        torch::nn::Linear(2 * config_.input_width, config_.d_model));
    scale_embedding_ = register_module(
        "scale_embedding",
        torch::nn::Embedding(static_cast<int64_t>(config_.time_scales.size()),
                             config_.d_model));
    channel_embedding_ = register_module(
        "channel_embedding",
        torch::nn::Embedding(config_.channel_count, config_.d_model));
    domain_embedding_ = register_module(
        "domain_embedding", torch::nn::Embedding(2, config_.d_model));
    position_projection_ = register_module(
        "position_projection", torch::nn::Linear(4, config_.d_model));
    this->to(config_.device, config_.dtype);
  }

  [[nodiscard]] mtf_token_batch_t
  forward(const torch::Tensor &x,
          const torch::Tensor &feature_mask = torch::Tensor()) {
    using torch::indexing::Slice;
    const auto input = detail::canonicalize_input(x, feature_mask, config_);
    const auto &data = input.data;
    const auto &mask = input.feature_mask;
    const int64_t B = data.size(0);
    const int64_t Hx = data.size(2);
    const auto strides = detail::resolved_scale_strides(config_);

    std::vector<torch::Tensor> token_parts;
    std::vector<torch::Tensor> target_parts;
    std::vector<torch::Tensor> time_raw_parts;
    std::vector<torch::Tensor> frequency_raw_parts;
    std::vector<torch::Tensor> time_mask_parts;
    std::vector<torch::Tensor> frequency_mask_parts;
    std::vector<torch::Tensor> valid_parts;
    std::vector<int64_t> starts;
    std::vector<int64_t> widths;
    std::vector<int64_t> scale_ids;
    std::vector<int64_t> channel_ids;
    std::vector<int64_t> domain_ids;

    for (int64_t c = 0; c < config_.channel_count; ++c) {
      for (std::size_t scale_i = 0; scale_i < config_.time_scales.size();
           ++scale_i) {
        const auto windows = detail::window_plan(
            Hx, config_.time_scales[scale_i], strides[scale_i]);
        for (const auto &[start, width] : windows) {
          auto patch =
              data.index({Slice(), c, Slice(start, start + width), Slice()});
          auto valid =
              mask.index({Slice(), c, Slice(start, start + width), Slice()});
          const auto descriptor = detail::masked_patch_descriptor(patch, valid);
          const auto feature_valid = valid.any(/*dim=*/1);
          auto base = torch::gelu(time_projection_->forward(descriptor));
          auto pos =
              torch::tensor({static_cast<double>(start) /
                                 static_cast<double>(std::max<int64_t>(1, Hx)),
                             static_cast<double>(start) +
                                 0.5 * static_cast<double>(width),
                             static_cast<double>(width) /
                                 static_cast<double>(std::max<int64_t>(1, Hx)),
                             0.0},
                            torch::TensorOptions()
                                .dtype(config_.dtype)
                                .device(config_.device))
                  .view({1, 4});
          pos.index_put_({0, 1},
                         pos.index({0, 1}) /
                             static_cast<double>(std::max<int64_t>(1, Hx)));
          auto scale = torch::full({1}, static_cast<int64_t>(scale_i),
                                   torch::TensorOptions()
                                       .dtype(torch::kInt64)
                                       .device(config_.device));
          auto channel = torch::full({1}, c,
                                     torch::TensorOptions()
                                         .dtype(torch::kInt64)
                                         .device(config_.device));
          auto domain = torch::zeros({1}, torch::TensorOptions()
                                              .dtype(torch::kInt64)
                                              .device(config_.device));
          auto token = base + scale_embedding_->forward(scale) +
                       channel_embedding_->forward(channel) +
                       domain_embedding_->forward(domain) +
                       position_projection_->forward(pos);
          token_parts.push_back(token.unsqueeze(1));
          target_parts.push_back(base.unsqueeze(1));
          time_raw_parts.push_back(descriptor.unsqueeze(1));
          frequency_raw_parts.push_back(torch::zeros(
              {B, 1, config_.frequency_num_bins * config_.input_width},
              data.options()));
          time_mask_parts.push_back(
              torch::cat({feature_valid, feature_valid}, /*dim=*/1)
                  .unsqueeze(1));
          frequency_mask_parts.push_back(torch::zeros(
              {B, 1, config_.frequency_num_bins * config_.input_width},
              torch::TensorOptions()
                  .dtype(torch::kBool)
                  .device(config_.device)));
          valid_parts.push_back(
              valid.any(/*dim=*/1).any(/*dim=*/1).unsqueeze(1));
          starts.push_back(start);
          widths.push_back(width);
          scale_ids.push_back(static_cast<int64_t>(scale_i));
          channel_ids.push_back(c);
          domain_ids.push_back(0);
        }
      }
    }

    mtf_token_batch_t out{};
    out.tokens = torch::cat(token_parts, /*dim=*/1);
    out.reconstruction_targets = torch::cat(target_parts, /*dim=*/1);
    out.time_reconstruction_targets = torch::cat(time_raw_parts, /*dim=*/1);
    out.frequency_reconstruction_targets =
        torch::cat(frequency_raw_parts, /*dim=*/1);
    out.time_reconstruction_mask = torch::cat(time_mask_parts, /*dim=*/1);
    out.frequency_reconstruction_mask =
        torch::cat(frequency_mask_parts, /*dim=*/1);
    out.token_mask = torch::cat(valid_parts, /*dim=*/1);
    auto meta_opts =
        torch::TensorOptions().dtype(torch::kInt64).device(config_.device);
    out.metadata.start_index = torch::tensor(starts, meta_opts);
    out.metadata.width = torch::tensor(widths, meta_opts);
    out.metadata.scale_id = torch::tensor(scale_ids, meta_opts);
    out.metadata.channel_id = torch::tensor(channel_ids, meta_opts);
    out.metadata.domain_id = torch::tensor(domain_ids, meta_opts);
    TORCH_CHECK(out.tokens.size(0) == B,
                "[embedding] tokenizer batch mismatch");
    return out;
  }

private:
  mtf_jepa_mae_vicreg_config_t config_{};
  torch::nn::Linear time_projection_{nullptr};
  torch::nn::Embedding scale_embedding_{nullptr};
  torch::nn::Embedding channel_embedding_{nullptr};
  torch::nn::Embedding domain_embedding_{nullptr};
  torch::nn::Linear position_projection_{nullptr};
};

TORCH_MODULE(MultiScalePatchTokenizer);

class FrequencyTokenizerImpl : public torch::nn::Module {
public:
  explicit FrequencyTokenizerImpl(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
    frequency_projection_ = register_module(
        "frequency_projection",
        torch::nn::Linear(config_.frequency_num_bins * config_.input_width,
                          config_.d_model));
    scale_embedding_ = register_module(
        "scale_embedding",
        torch::nn::Embedding(static_cast<int64_t>(config_.time_scales.size()),
                             config_.d_model));
    channel_embedding_ = register_module(
        "channel_embedding",
        torch::nn::Embedding(config_.channel_count, config_.d_model));
    domain_embedding_ = register_module(
        "domain_embedding", torch::nn::Embedding(2, config_.d_model));
    position_projection_ = register_module(
        "position_projection", torch::nn::Linear(4, config_.d_model));
    this->to(config_.device, config_.dtype);
  }

  [[nodiscard]] mtf_token_batch_t
  forward(const torch::Tensor &x,
          const torch::Tensor &feature_mask = torch::Tensor()) {
    using torch::indexing::Slice;
    const auto input = detail::canonicalize_input(x, feature_mask, config_);
    const auto &data = input.data;
    const auto &mask = input.feature_mask;
    const int64_t B = data.size(0);
    const int64_t Hx = data.size(2);
    const auto strides = detail::resolved_scale_strides(config_);

    std::vector<torch::Tensor> token_parts;
    std::vector<torch::Tensor> target_parts;
    std::vector<torch::Tensor> time_raw_parts;
    std::vector<torch::Tensor> frequency_raw_parts;
    std::vector<torch::Tensor> time_mask_parts;
    std::vector<torch::Tensor> frequency_mask_parts;
    std::vector<torch::Tensor> valid_parts;
    std::vector<int64_t> starts;
    std::vector<int64_t> widths;
    std::vector<int64_t> scale_ids;
    std::vector<int64_t> channel_ids;
    std::vector<int64_t> domain_ids;

    for (int64_t c = 0; c < config_.channel_count; ++c) {
      for (std::size_t scale_i = 0; scale_i < config_.time_scales.size();
           ++scale_i) {
        const auto windows = detail::window_plan(
            Hx, config_.time_scales[scale_i], strides[scale_i]);
        for (const auto &[start, width] : windows) {
          auto patch =
              data.index({Slice(), c, Slice(start, start + width), Slice()});
          auto valid =
              mask.index({Slice(), c, Slice(start, start + width), Slice()});
          patch = torch::where(valid, patch, torch::zeros_like(patch));
          const auto feature_valid = valid.any(/*dim=*/1);
          auto descriptor = frequency_descriptor(patch, valid);
          auto base = torch::gelu(frequency_projection_->forward(descriptor));
          auto pos =
              torch::tensor({static_cast<double>(start) /
                                 static_cast<double>(std::max<int64_t>(1, Hx)),
                             static_cast<double>(start) +
                                 0.5 * static_cast<double>(width),
                             static_cast<double>(width) /
                                 static_cast<double>(std::max<int64_t>(1, Hx)),
                             1.0},
                            torch::TensorOptions()
                                .dtype(config_.dtype)
                                .device(config_.device))
                  .view({1, 4});
          pos.index_put_({0, 1},
                         pos.index({0, 1}) /
                             static_cast<double>(std::max<int64_t>(1, Hx)));
          auto scale = torch::full({1}, static_cast<int64_t>(scale_i),
                                   torch::TensorOptions()
                                       .dtype(torch::kInt64)
                                       .device(config_.device));
          auto channel = torch::full({1}, c,
                                     torch::TensorOptions()
                                         .dtype(torch::kInt64)
                                         .device(config_.device));
          auto domain = torch::ones({1}, torch::TensorOptions()
                                             .dtype(torch::kInt64)
                                             .device(config_.device));
          auto token = base + scale_embedding_->forward(scale) +
                       channel_embedding_->forward(channel) +
                       domain_embedding_->forward(domain) +
                       position_projection_->forward(pos);
          token_parts.push_back(token.unsqueeze(1));
          target_parts.push_back(base.unsqueeze(1));
          time_raw_parts.push_back(
              torch::zeros({B, 1, 2 * config_.input_width}, data.options()));
          frequency_raw_parts.push_back(descriptor.unsqueeze(1));
          time_mask_parts.push_back(torch::zeros(
              {B, 1, 2 * config_.input_width}, torch::TensorOptions()
                                                   .dtype(torch::kBool)
                                                   .device(config_.device)));
          frequency_mask_parts.push_back(
              feature_valid.unsqueeze(-1)
                  .expand({B, config_.input_width, config_.frequency_num_bins})
                  .reshape({B, 1,
                            config_.frequency_num_bins * config_.input_width}));
          valid_parts.push_back(
              valid.any(/*dim=*/1).any(/*dim=*/1).unsqueeze(1));
          starts.push_back(start);
          widths.push_back(width);
          scale_ids.push_back(static_cast<int64_t>(scale_i));
          channel_ids.push_back(c);
          domain_ids.push_back(1);
        }
      }
    }

    mtf_token_batch_t out{};
    out.tokens = torch::cat(token_parts, /*dim=*/1);
    out.reconstruction_targets = torch::cat(target_parts, /*dim=*/1);
    out.time_reconstruction_targets = torch::cat(time_raw_parts, /*dim=*/1);
    out.frequency_reconstruction_targets =
        torch::cat(frequency_raw_parts, /*dim=*/1);
    out.time_reconstruction_mask = torch::cat(time_mask_parts, /*dim=*/1);
    out.frequency_reconstruction_mask =
        torch::cat(frequency_mask_parts, /*dim=*/1);
    out.token_mask = torch::cat(valid_parts, /*dim=*/1);
    auto meta_opts =
        torch::TensorOptions().dtype(torch::kInt64).device(config_.device);
    out.metadata.start_index = torch::tensor(starts, meta_opts);
    out.metadata.width = torch::tensor(widths, meta_opts);
    out.metadata.scale_id = torch::tensor(scale_ids, meta_opts);
    out.metadata.channel_id = torch::tensor(channel_ids, meta_opts);
    out.metadata.domain_id = torch::tensor(domain_ids, meta_opts);
    TORCH_CHECK(out.tokens.size(0) == B,
                "[embedding] frequency tokenizer batch mismatch");
    return out;
  }

private:
  [[nodiscard]] torch::Tensor frequency_descriptor(const torch::Tensor &patch,
                                                   const torch::Tensor &valid) {
    const int64_t width = patch.size(1);
    const int64_t actual_bins =
        std::min<int64_t>(config_.frequency_num_bins, width / 2 + 1);
    // Keep large baselines, centering and squared DFT components in float64.
    // The public descriptor remains in the configured input dtype.
    auto patch_accumulated = patch.to(torch::kFloat64);
    auto valid_f = valid.to(torch::kFloat64);
    auto valid_count = valid_f.sum(/*dim=*/1).clamp_min(1.0); // [B,F]
    auto observed = torch::where(valid, patch_accumulated,
                                 torch::zeros_like(patch_accumulated));
    auto mean = observed.sum(/*dim=*/1) / valid_count;
    auto centered = torch::where(valid, observed - mean.unsqueeze(1),
                                 torch::zeros_like(observed));
    torch::Tensor hann;
    if (width <= 1) {
      hann = torch::ones({width}, patch_accumulated.options());
    } else {
      constexpr double pi = 3.14159265358979323846;
      auto n = torch::arange(width, patch_accumulated.options());
      hann =
          0.5 - 0.5 * torch::cos(2.0 * pi * n / static_cast<double>(width - 1));
    }
    centered = centered * hann.view({1, width, 1});
    auto denom =
        (valid_f * hann.view({1, width, 1})).sum(/*dim=*/1).clamp_min(1.0);

    auto t = torch::arange(width, patch_accumulated.options());
    auto k = torch::arange(actual_bins, patch_accumulated.options()).unsqueeze(1);
    constexpr double pi = 3.14159265358979323846;
    auto angle = 2.0 * pi * k * t.unsqueeze(0) /
                 static_cast<double>(std::max<int64_t>(1, width));
    auto cos_basis = torch::cos(angle).transpose(0, 1); // [W,K]
    auto sin_basis = torch::sin(angle).transpose(0, 1); // [W,K]
    auto by_feature = centered.transpose(1, 2);         // [B,F,W]
    auto real = torch::matmul(by_feature, cos_basis);
    auto imag = torch::matmul(by_feature, sin_basis);
    auto mag =
        torch::sqrt(real.pow(2) + imag.pow(2) + 1e-8) / denom.unsqueeze(-1);
    if (actual_bins > 0) {
      mag.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0},
                     mean.abs());
    }
    if (config_.frequency_log_magnitude) {
      mag = torch::log1p(mag);
    }
    auto padded = torch::zeros(
        {patch.size(0), config_.input_width, config_.frequency_num_bins},
        patch_accumulated.options());
    if (actual_bins > 0) {
      padded.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(0, actual_bins)},
                        mag);
    }
    return padded.reshape(
        {patch.size(0), config_.frequency_num_bins * config_.input_width})
        .to(patch.dtype());
  }

  mtf_jepa_mae_vicreg_config_t config_{};
  torch::nn::Linear frequency_projection_{nullptr};
  torch::nn::Embedding scale_embedding_{nullptr};
  torch::nn::Embedding channel_embedding_{nullptr};
  torch::nn::Embedding domain_embedding_{nullptr};
  torch::nn::Linear position_projection_{nullptr};
};

TORCH_MODULE(FrequencyTokenizer);

inline mtf_token_batch_t concat_token_batches(const mtf_token_batch_t &a,
                                              const mtf_token_batch_t &b) {
  mtf_token_batch_t out{};
  out.tokens = torch::cat({a.tokens, b.tokens}, /*dim=*/1);
  out.reconstruction_targets =
      torch::cat({a.reconstruction_targets, b.reconstruction_targets},
                 /*dim=*/1);
  out.time_reconstruction_targets =
      torch::cat({a.time_reconstruction_targets, b.time_reconstruction_targets},
                 /*dim=*/1);
  out.frequency_reconstruction_targets = torch::cat(
      {a.frequency_reconstruction_targets, b.frequency_reconstruction_targets},
      /*dim=*/1);
  out.time_reconstruction_mask =
      torch::cat({a.time_reconstruction_mask, b.time_reconstruction_mask},
                 /*dim=*/1);
  out.frequency_reconstruction_mask = torch::cat(
      {a.frequency_reconstruction_mask, b.frequency_reconstruction_mask},
      /*dim=*/1);
  out.token_mask = torch::cat({a.token_mask, b.token_mask}, /*dim=*/1);
  out.metadata.start_index =
      torch::cat({a.metadata.start_index, b.metadata.start_index}, 0);
  out.metadata.width = torch::cat({a.metadata.width, b.metadata.width}, 0);
  out.metadata.scale_id =
      torch::cat({a.metadata.scale_id, b.metadata.scale_id}, 0);
  out.metadata.channel_id =
      torch::cat({a.metadata.channel_id, b.metadata.channel_id}, 0);
  out.metadata.domain_id =
      torch::cat({a.metadata.domain_id, b.metadata.domain_id}, 0);
  return out;
}

class TimeFrequencyViewBuilderImpl : public torch::nn::Module {
public:
  explicit TimeFrequencyViewBuilderImpl(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
    time_tokenizer_ =
        register_module("time_tokenizer", MultiScalePatchTokenizer(config_));
    frequency_tokenizer_ =
        register_module("frequency_tokenizer", FrequencyTokenizer(config_));
  }

  [[nodiscard]] mtf_token_batch_t
  forward(const torch::Tensor &x,
          const torch::Tensor &feature_mask = torch::Tensor()) {
    auto time_tokens = time_tokenizer_->forward(x, feature_mask);
    if (!config_.use_frequency_tokens) {
      return time_tokens;
    }
    auto frequency_tokens = frequency_tokenizer_->forward(x, feature_mask);
    return concat_token_batches(time_tokens, frequency_tokens);
  }

private:
  mtf_jepa_mae_vicreg_config_t config_{};
  MultiScalePatchTokenizer time_tokenizer_{nullptr};
  FrequencyTokenizer frequency_tokenizer_{nullptr};
};

TORCH_MODULE(TimeFrequencyViewBuilder);

} // namespace embedding::encoders::mtf_jepa_mae_vicreg
