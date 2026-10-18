// SPDX-License-Identifier: MIT
#pragma once

#include <utility>
#include "embedding/encoders/mtf_jepa_mae_vicreg/tensor_ops.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {

namespace detail {

inline torch::Tensor multi_head_context_attention(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
    const torch::Tensor &context_mask, int64_t num_heads) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
              "[embedding] attention tensors must be [B,N,D]");
  TORCH_CHECK(q.size(0) == k.size(0) && k.sizes() == v.sizes() &&
                  q.size(2) == k.size(2),
              "[embedding] attention tensor shape mismatch");
  TORCH_CHECK(context_mask.size(0) == k.size(0) &&
                  context_mask.size(1) == k.size(1),
              "[embedding] attention mask shape mismatch");
  const int64_t B = q.size(0);
  const int64_t Q = q.size(1);
  const int64_t K = k.size(1);
  const int64_t D = q.size(2);
  TORCH_CHECK(num_heads > 0 && D % num_heads == 0,
              "[embedding] latent_dim must divide num_heads");
  const int64_t head_dim = D / num_heads;
  auto qh =
      q.contiguous().view({B, Q, num_heads, head_dim}).permute({0, 2, 1, 3});
  auto kh =
      k.contiguous().view({B, K, num_heads, head_dim}).permute({0, 2, 1, 3});
  auto vh =
      v.contiguous().view({B, K, num_heads, head_dim}).permute({0, 2, 1, 3});
  auto scores = torch::matmul(qh, kh.transpose(-2, -1)) /
                std::sqrt(static_cast<double>(head_dim));
  const auto mask =
      context_mask.to(torch::kBool).view({B, 1, 1, K}).to(scores.device());
  const auto mask_f = mask.to(scores.dtype());
  scores = scores.masked_fill(mask.logical_not(), -1.0e9);
  auto weights = torch::softmax(scores, /*dim=*/-1) * mask_f;
  weights = weights / weights.sum(/*dim=*/-1, /*keepdim=*/true).clamp_min(1e-6);
  auto attended = torch::matmul(weights, vh);
  const auto has_context = context_mask.to(torch::kBool)
                               .any(/*dim=*/1)
                               .to(scores.dtype())
                               .view({B, 1, 1, 1})
                               .to(scores.device());
  attended = attended * has_context;
  return attended.permute({0, 2, 1, 3}).contiguous().view({B, Q, D});
}

} // namespace detail

class SharedTokenEncoderImpl : public torch::nn::Module {
public:
  explicit SharedTokenEncoderImpl(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
    for (int64_t i = 0; i < config_.num_encoder_layers; ++i) {
      token_layers_.push_back(
          register_module("token_layer_" + std::to_string(i),
                          torch::nn::Linear(config_.d_model, config_.d_model)));
      mix_layers_.push_back(
          register_module("mix_layer_" + std::to_string(i),
                          torch::nn::Linear(config_.d_model, config_.d_model)));
      norms_.push_back(
          register_module("norm_" + std::to_string(i),
                          torch::nn::LayerNorm(
                              torch::nn::LayerNormOptions({config_.d_model}))));
    }
    out_norm_ = register_module(
        "out_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({config_.d_model})));
    latent_projection_ =
        register_module("latent_projection",
                        torch::nn::Linear(config_.d_model, config_.latent_dim));
    dropout_ = register_module("dropout", torch::nn::Dropout(config_.dropout));
    this->to(config_.device, config_.dtype);
  }

  [[nodiscard]] torch::Tensor forward(const torch::Tensor &tokens,
                                      const torch::Tensor &token_mask) {
    auto h = tokens.to(
        torch::TensorOptions().dtype(config_.dtype).device(config_.device));
    const auto mask = token_mask.to(
        torch::TensorOptions().dtype(torch::kBool).device(config_.device));
    h = h.masked_fill(mask.logical_not().unsqueeze(-1), 0.0);
    for (std::size_t i = 0; i < token_layers_.size(); ++i) {
      const auto pooled = ::embedding::detail::masked_mean(h, mask);
      auto mixed = mix_layers_[i]->forward(pooled).unsqueeze(1).expand_as(h);
      auto update = token_layers_[i]->forward(norms_[i]->forward(h + mixed));
      update = torch::gelu(update);
      update = dropout_->forward(update);
      h = (h + update).masked_fill(mask.logical_not().unsqueeze(-1), 0.0);
    }
    return latent_projection_->forward(out_norm_->forward(h))
        .masked_fill(mask.logical_not().unsqueeze(-1), 0.0);
  }

private:
  mtf_jepa_mae_vicreg_config_t config_{};
  std::vector<torch::nn::Linear> token_layers_{};
  std::vector<torch::nn::Linear> mix_layers_{};
  std::vector<torch::nn::LayerNorm> norms_{};
  torch::nn::LayerNorm out_norm_{nullptr};
  torch::nn::Linear latent_projection_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(SharedTokenEncoder);

class LatentPredictorImpl : public torch::nn::Module {
public:
  explicit LatentPredictorImpl(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
    metadata_projection_ = register_module(
        "metadata_projection", torch::nn::Linear(6, config_.latent_dim));
    q_projection_ =
        register_module("q_projection", torch::nn::Linear(config_.latent_dim,
                                                          config_.latent_dim));
    k_projection_ =
        register_module("k_projection", torch::nn::Linear(config_.latent_dim,
                                                          config_.latent_dim));
    v_projection_ =
        register_module("v_projection", torch::nn::Linear(config_.latent_dim,
                                                          config_.latent_dim));
    layers_.push_back(register_module(
        "predictor_in",
        torch::nn::Linear(config_.latent_dim, config_.predictor_hidden_dim)));
    for (int64_t i = 1; i < config_.num_predictor_layers; ++i) {
      layers_.push_back(
          register_module("predictor_hidden_" + std::to_string(i),
                          torch::nn::Linear(config_.predictor_hidden_dim,
                                            config_.predictor_hidden_dim)));
    }
    out_ = register_module(
        "predictor_out",
        torch::nn::Linear(config_.predictor_hidden_dim, config_.latent_dim));
    dropout_ = register_module("dropout", torch::nn::Dropout(config_.dropout));
    this->to(config_.device, config_.dtype);
  }

  [[nodiscard]] torch::Tensor forward(const torch::Tensor &context_latents,
                                      const torch::Tensor &context_mask,
                                      const mtf_token_metadata_t &metadata) {
    const auto meta = metadata_projection_->forward(
        detail::metadata_features(metadata, config_));
    auto q = q_projection_->forward(meta).unsqueeze(0).expand(
        {context_latents.size(0), meta.size(0), config_.latent_dim});
    auto k = k_projection_->forward(context_latents);
    auto v = v_projection_->forward(context_latents);
    const auto mask = context_mask.to(
        torch::TensorOptions().dtype(torch::kBool).device(config_.device));
    auto attended =
        detail::multi_head_context_attention(q, k, v, mask, config_.num_heads);
    auto h = attended + q;
    for (auto &layer : layers_) {
      h = dropout_->forward(torch::gelu(layer->forward(h)));
    }
    return out_->forward(h);
  }

private:
  mtf_jepa_mae_vicreg_config_t config_{};
  torch::nn::Linear metadata_projection_{nullptr};
  torch::nn::Linear q_projection_{nullptr};
  torch::nn::Linear k_projection_{nullptr};
  torch::nn::Linear v_projection_{nullptr};
  std::vector<torch::nn::Linear> layers_{};
  torch::nn::Linear out_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(LatentPredictor);

class MaeDecoderImpl : public torch::nn::Module {
public:
  explicit MaeDecoderImpl(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
    metadata_projection_ = register_module(
        "metadata_projection", torch::nn::Linear(6, config_.latent_dim));
    q_projection_ =
        register_module("q_projection", torch::nn::Linear(config_.latent_dim,
                                                          config_.latent_dim));
    k_projection_ =
        register_module("k_projection", torch::nn::Linear(config_.latent_dim,
                                                          config_.latent_dim));
    v_projection_ =
        register_module("v_projection", torch::nn::Linear(config_.latent_dim,
                                                          config_.latent_dim));
    layers_.push_back(register_module(
        "decoder_in",
        torch::nn::Linear(config_.latent_dim, config_.predictor_hidden_dim)));
    for (int64_t i = 1; i < config_.num_decoder_layers; ++i) {
      layers_.push_back(
          register_module("decoder_hidden_" + std::to_string(i),
                          torch::nn::Linear(config_.predictor_hidden_dim,
                                            config_.predictor_hidden_dim)));
    }
    projected_out_ = register_module(
        "decoder_projected_out",
        torch::nn::Linear(config_.predictor_hidden_dim, config_.d_model));
    time_out_ = register_module("decoder_time_out",
                                torch::nn::Linear(config_.predictor_hidden_dim,
                                                  2 * config_.input_width));
    frequency_out_ = register_module(
        "decoder_frequency_out",
        torch::nn::Linear(config_.predictor_hidden_dim,
                          config_.frequency_num_bins * config_.input_width));
    dropout_ = register_module("dropout", torch::nn::Dropout(config_.dropout));
    this->to(config_.device, config_.dtype);
  }

  [[nodiscard]] mae_decoder_output_t
  forward(const torch::Tensor &context_latents,
          const torch::Tensor &context_mask,
          const mtf_token_metadata_t &metadata) {
    const auto meta = metadata_projection_->forward(
        detail::metadata_features(metadata, config_));
    auto q = q_projection_->forward(meta).unsqueeze(0).expand(
        {context_latents.size(0), meta.size(0), config_.latent_dim});
    auto k = k_projection_->forward(context_latents);
    auto v = v_projection_->forward(context_latents);
    const auto mask = context_mask.to(
        torch::TensorOptions().dtype(torch::kBool).device(config_.device));
    auto h =
        detail::multi_head_context_attention(q, k, v, mask, config_.num_heads) +
        q;
    for (auto &layer : layers_) {
      h = dropout_->forward(torch::gelu(layer->forward(h)));
    }
    mae_decoder_output_t out{};
    out.projected = projected_out_->forward(h);
    out.time = time_out_->forward(h);
    out.frequency = frequency_out_->forward(h);
    return out;
  }

private:
  mtf_jepa_mae_vicreg_config_t config_{};
  torch::nn::Linear metadata_projection_{nullptr};
  torch::nn::Linear q_projection_{nullptr};
  torch::nn::Linear k_projection_{nullptr};
  torch::nn::Linear v_projection_{nullptr};
  std::vector<torch::nn::Linear> layers_{};
  torch::nn::Linear projected_out_{nullptr};
  torch::nn::Linear time_out_{nullptr};
  torch::nn::Linear frequency_out_{nullptr};
  torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(MaeDecoder);

class VICRegStabilityHeadImpl : public torch::nn::Module {
public:
  explicit VICRegStabilityHeadImpl(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
    projector_layers_.push_back(register_module(
        "projector_in",
        torch::nn::Linear(config_.latent_dim, config_.predictor_hidden_dim)));
    projector_layers_.push_back(register_module(
        "projector_hidden", torch::nn::Linear(config_.predictor_hidden_dim,
                                              config_.predictor_hidden_dim)));
    projector_out_ = register_module(
        "projector_out",
        torch::nn::Linear(config_.predictor_hidden_dim, config_.projector_dim));
    this->to(config_.device, config_.dtype);
  }

  [[nodiscard]] torch::Tensor forward(const torch::Tensor &pooled_latents) {
    TORCH_CHECK(pooled_latents.defined() && pooled_latents.dim() == 2,
                "[embedding] stability head input must be [B,D]");
    TORCH_CHECK(pooled_latents.size(1) == config_.latent_dim,
                "[embedding] stability head dim mismatch");
    auto h = pooled_latents.to(
        torch::TensorOptions().dtype(config_.dtype).device(config_.device));
    for (auto &layer : projector_layers_) {
      h = torch::gelu(layer->forward(h));
    }
    return projector_out_->forward(h).unsqueeze(1);
  }

private:
  mtf_jepa_mae_vicreg_config_t config_{};
  std::vector<torch::nn::Linear> projector_layers_{};
  torch::nn::Linear projector_out_{nullptr};
};

TORCH_MODULE(VICRegStabilityHead);

} // namespace embedding::encoders::mtf_jepa_mae_vicreg
