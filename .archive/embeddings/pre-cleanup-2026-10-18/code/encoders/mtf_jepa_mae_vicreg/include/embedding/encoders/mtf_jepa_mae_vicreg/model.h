// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/mtf_jepa_mae_vicreg/tokenization.h"
#include "embedding/encoders/mtf_jepa_mae_vicreg/masking.h"
#include "embedding/encoders/mtf_jepa_mae_vicreg/encoder.h"
#include "embedding/encoders/mtf_jepa_mae_vicreg/objectives.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {

class MtfJepaMaeVicregImpl : public torch::nn::Module {
public:
  explicit MtfJepaMaeVicregImpl(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)), masker_(config_) {
    detail::validate_config(config_);
    tokenizer_ =
        register_module("tokenizer", TimeFrequencyViewBuilder(config_));
    target_tokenizer_ =
        register_module("target_tokenizer", TimeFrequencyViewBuilder(config_));
    encoder_ = register_module("encoder", SharedTokenEncoder(config_));
    target_encoder_ =
        register_module("target_encoder", SharedTokenEncoder(config_));
    predictor_ = register_module("predictor", LatentPredictor(config_));
    mae_decoder_ = register_module("mae_decoder", MaeDecoder(config_));
    vicreg_stability_head_ =
        register_module("vicreg_stability_head", VICRegStabilityHead(config_));
    this->to(config_.device, config_.dtype);
    update_target_network(0.0);
    for (auto &param : target_tokenizer_->parameters()) {
      param.set_requires_grad(false);
    }
    for (auto &param : target_encoder_->parameters()) {
      param.set_requires_grad(false);
    }
  }

  [[nodiscard]] const mtf_jepa_mae_vicreg_config_t &config() const {
    return config_;
  }

  [[nodiscard]] torch::Tensor
  project_vicreg(const torch::Tensor &pooled_latents) {
    TORCH_CHECK(pooled_latents.defined() &&
                    (pooled_latents.dim() == 2 || pooled_latents.dim() == 3),
                "[embedding] VICReg projection input must be [B,D] "
                "or [B,C,D]");
    if (pooled_latents.dim() == 2) {
      return vicreg_stability_head_->forward(pooled_latents).squeeze(1);
    }
    const int64_t batch_size = pooled_latents.size(0);
    const int64_t channel_count = pooled_latents.size(1);
    auto flattened = pooled_latents.reshape(
        {batch_size * channel_count, pooled_latents.size(2)});
    return vicreg_stability_head_->forward(flattened).squeeze(1).view(
        {batch_size, channel_count, config_.projector_dim});
  }

  [[nodiscard]] mtf_token_batch_t
  tokenize(const torch::Tensor &x,
           const torch::Tensor &feature_mask = torch::Tensor()) {
    return tokenizer_->forward(x, feature_mask);
  }

  [[nodiscard]] jepa_context_target_mask_t
  create_masks(const mtf_token_batch_t &batch) const {
    return masker_.create_masks(batch);
  }

  [[nodiscard]] mtf_jepa_mae_vicreg_encode_output_t
  encode(const torch::Tensor &x,
         const torch::Tensor &feature_mask = torch::Tensor()) {
    auto batch = tokenizer_->forward(x, feature_mask);
    auto embeddings = encoder_->forward(batch.tokens, batch.token_mask);
    mtf_jepa_mae_vicreg_encode_output_t out{};
    out.embeddings = embeddings;
    out.pooled_embedding = ::embedding::detail::masked_mean(embeddings, batch.token_mask);
    out.pooled_by_channel = detail::pooled_by_channel(
        embeddings, batch.token_mask, batch.metadata, config_);
    out.pooled_time = ::embedding::detail::masked_mean(
        embeddings, detail::domain_mask(batch.metadata, batch.token_mask, 0));
    out.pooled_frequency = ::embedding::detail::masked_mean(
        embeddings, detail::domain_mask(batch.metadata, batch.token_mask, 1));
    out.token_mask = batch.token_mask;
    out.sample_valid_mask = batch.token_mask.any(/*dim=*/1);
    out.channel_valid_mask =
        detail::channel_valid_mask(batch.metadata, batch.token_mask, config_);
    out.metadata = batch.metadata;
    TORCH_CHECK(torch::isfinite(embeddings).all().template item<bool>() &&
                    torch::isfinite(out.pooled_embedding).all().template item<bool>() &&
                    torch::isfinite(out.pooled_by_channel).all().template item<bool>(),
                "[embedding] non-finite encoding; standardize input using training-set statistics");
    return out;
  }

  [[nodiscard]] torch::Tensor
  target_encode(const torch::Tensor &x,
                const torch::Tensor &feature_mask = torch::Tensor()) {
    auto batch = tokenizer_->forward(x, feature_mask);
    auto fallback_latents = encoder_->forward(batch.tokens, batch.token_mask);
    auto target_latents =
        compute_target_latents(x, feature_mask, batch, fallback_latents);
    if (config_.stop_gradient_target) {
      target_latents = target_latents.detach();
    }
    return target_latents;
  }

  [[nodiscard]] mtf_jepa_mae_vicreg_output_t
  forward(const torch::Tensor &x,
          const torch::Tensor &feature_mask = torch::Tensor()) {
    auto batch = tokenizer_->forward(x, feature_mask);
    auto masks = masker_.create_masks(batch);

    auto full_latents = encoder_->forward(batch.tokens, batch.token_mask);
    auto context_tokens = batch.tokens.masked_fill(
        masks.context_mask.logical_not().unsqueeze(-1), 0.0);
    auto context_latents =
        encoder_->forward(context_tokens, masks.context_mask);

    auto target_latents =
        compute_target_latents(x, feature_mask, batch, full_latents);
    if (config_.stop_gradient_target) {
      target_latents = target_latents.detach();
    }

    const auto zero = torch::zeros({}, batch.tokens.options());
    const auto check_or_sanitize_loss = [&](const torch::Tensor &value) {
      return config_.strict_finite_loss ? value : ::embedding::detail::finite_or_zero(value);
    };
    auto pred_latents = predictor_->forward(context_latents, masks.context_mask,
                                            batch.metadata);
    auto loss_jepa = config_.use_jepa_loss
                         ? ::embedding::detail::masked_mse(pred_latents, target_latents,
                                              masks.target_mask)
                         : zero;
    loss_jepa = check_or_sanitize_loss(loss_jepa);

    auto decoded = mae_decoder_->forward(context_latents, masks.context_mask,
                                         batch.metadata);
    auto loss_mae = zero;
    auto loss_mae_time = zero;
    auto loss_mae_frequency = zero;
    if (config_.use_mae_decoder) {
      if (config_.use_raw_reconstruction_targets) {
        const auto time_target_mask = masks.target_mask.logical_and(
            batch.metadata.domain_id.to(masks.target_mask.device())
                .eq(0)
                .unsqueeze(0));
        const auto frequency_target_mask = masks.target_mask.logical_and(
            batch.metadata.domain_id.to(masks.target_mask.device())
                .eq(1)
                .unsqueeze(0));
        loss_mae_time = ::embedding::detail::masked_weighted_mse(
            decoded.time, batch.time_reconstruction_targets.detach(),
            time_target_mask, batch.time_reconstruction_mask);
        loss_mae_frequency = ::embedding::detail::masked_weighted_mse(
            decoded.frequency, batch.frequency_reconstruction_targets.detach(),
            frequency_target_mask, batch.frequency_reconstruction_mask);
        loss_mae = loss_mae_time + loss_mae_frequency;
      } else {
        loss_mae = ::embedding::detail::masked_mse(decoded.projected,
                                      batch.reconstruction_targets.detach(),
                                      masks.target_mask);
      }
    }
    loss_mae_time = check_or_sanitize_loss(loss_mae_time);
    loss_mae_frequency = check_or_sanitize_loss(loss_mae_frequency);
    loss_mae = check_or_sanitize_loss(loss_mae);

    tf_alignment_result_t tf_alignment{};
    tf_alignment.loss = zero;
    if (config_.use_tf_align_loss && config_.use_frequency_tokens) {
      tf_alignment =
          compute_pairwise_time_frequency_alignment(full_latents, batch);
    }
    auto loss_tf_align = tf_alignment.loss;
    loss_tf_align = check_or_sanitize_loss(loss_tf_align);

    torch::Tensor loss_vicreg = zero;
    torch::Tensor loss_vicreg_global = zero;
    torch::Tensor loss_vicreg_channel = zero;
    if (config_.use_vicreg_loss) {
      const auto result = compute_vicreg_branch(x, feature_mask);
      loss_vicreg = result.loss;
      loss_vicreg_global = result.global_loss;
      loss_vicreg_channel = result.channel_loss;
    }
    loss_vicreg = check_or_sanitize_loss(loss_vicreg);

    auto total_loss = config_.lambda_jepa * loss_jepa +
                      config_.lambda_mae * loss_mae +
                      config_.lambda_tf_align * loss_tf_align +
                      config_.lambda_vicreg * loss_vicreg;
    if (config_.strict_finite_loss) {
      const auto all_losses_finite =
          torch::stack({loss_jepa, loss_mae_time, loss_mae_frequency, loss_mae,
                        loss_tf_align, loss_vicreg, total_loss})
              .isfinite()
              .all()
              .template item<bool>();
      TORCH_CHECK(all_losses_finite,
                  "[embedding] non-finite loss");
    } else {
      total_loss = ::embedding::detail::finite_or_zero(total_loss);
    }

    mtf_jepa_mae_vicreg_output_t out{};
    out.embeddings = full_latents;
    out.pooled_embedding = ::embedding::detail::masked_mean(full_latents, batch.token_mask);
    out.pooled_by_channel = detail::pooled_by_channel(
        full_latents, batch.token_mask, batch.metadata, config_);
    out.pooled_time = ::embedding::detail::masked_mean(
        full_latents, detail::domain_mask(batch.metadata, batch.token_mask, 0));
    out.pooled_frequency = ::embedding::detail::masked_mean(
        full_latents, detail::domain_mask(batch.metadata, batch.token_mask, 1));
    out.loss = total_loss;
    out.loss_jepa = ::embedding::detail::finite_or_zero(loss_jepa);
    out.loss_mae = ::embedding::detail::finite_or_zero(loss_mae);
    out.loss_mae_time = ::embedding::detail::finite_or_zero(loss_mae_time);
    out.loss_mae_frequency = ::embedding::detail::finite_or_zero(loss_mae_frequency);
    out.loss_tf_align = ::embedding::detail::finite_or_zero(loss_tf_align);
    out.loss_vicreg = ::embedding::detail::finite_or_zero(loss_vicreg);
    out.loss_vicreg_global = ::embedding::detail::finite_or_zero(loss_vicreg_global);
    out.loss_vicreg_channel = ::embedding::detail::finite_or_zero(loss_vicreg_channel);
    out.jepa_target_mask = masks.target_mask.detach();
    out.jepa_context_mask = masks.context_mask.detach();
    out.sample_valid_mask = batch.token_mask.any(/*dim=*/1);
    out.channel_valid_mask =
        detail::channel_valid_mask(batch.metadata, batch.token_mask, config_);

    return out;
  }

  bool update_target_encoder(double tau = -1.0) {
    return update_target_network(tau);
  }

  bool update_target_network(double tau = -1.0) {
    if (!config_.use_target_ema && tau < 0.0) {
      return false;
    }
    if (tau < 0.0) {
      tau = config_.target_ema_tau;
    }
    TORCH_CHECK(tau >= 0.0 && tau <= 1.0,
                "[embedding] EMA tau must be in [0,1]");
    torch::NoGradGuard no_grad;
    update_ema_parameters(tokenizer_->parameters(),
                          target_tokenizer_->parameters(), tau,
                          "target tokenizer");
    update_ema_parameters(encoder_->parameters(), target_encoder_->parameters(),
                          tau, "target encoder");
    return true;
  }


private:
  static void update_ema_parameters(std::vector<torch::Tensor> online_params,
                                    std::vector<torch::Tensor> target_params,
                                    double tau, const char *label) {
    TORCH_CHECK(online_params.size() == target_params.size(),
                "[embedding] ", label, " parameter mismatch");
    for (std::size_t i = 0; i < online_params.size(); ++i) {
      target_params[i].mul_(tau);
      target_params[i].add_(online_params[i].detach(), 1.0 - tau);
    }
  }

  static void assert_matching_token_layout(const mtf_token_batch_t &online,
                                           const mtf_token_batch_t &target) {
    TORCH_CHECK(online.tokens.sizes() == target.tokens.sizes(),
                "[embedding] target tokenizer token shape mismatch");
    TORCH_CHECK(online.token_mask.sizes() == target.token_mask.sizes(),
                "[embedding] target tokenizer mask shape mismatch");
    const auto layout_exact =
        torch::stack({
            torch::eq(online.token_mask, target.token_mask).all(),
            torch::eq(online.metadata.start_index,
                      target.metadata.start_index)
                .all(),
            torch::eq(online.metadata.width, target.metadata.width).all(),
            torch::eq(online.metadata.scale_id, target.metadata.scale_id).all(),
            torch::eq(online.metadata.channel_id,
                      target.metadata.channel_id)
                .all(),
            torch::eq(online.metadata.domain_id, target.metadata.domain_id)
                .all()})
            .all()
            .template item<bool>();
    TORCH_CHECK(layout_exact,
                "[embedding] target tokenizer layout mismatch");
  }

  [[nodiscard]] torch::Tensor
  compute_target_latents(const torch::Tensor &x,
                         const torch::Tensor &feature_mask,
                         const mtf_token_batch_t &online_batch,
                         const torch::Tensor &fallback_latents) {
    if (!config_.use_target_ema) {
      return fallback_latents;
    }
    torch::NoGradGuard no_grad;
    const bool tokenizer_was_training = target_tokenizer_->is_training();
    const bool encoder_was_training = target_encoder_->is_training();
    target_tokenizer_->eval();
    target_encoder_->eval();
    const auto target_batch = target_tokenizer_->forward(x, feature_mask);
    assert_matching_token_layout(online_batch, target_batch);
    auto target_latents = target_encoder_->forward(target_batch.tokens.detach(),
                                                   target_batch.token_mask);
    target_tokenizer_->train(tokenizer_was_training);
    target_encoder_->train(encoder_was_training);
    return target_latents;
  }

  [[nodiscard]] tf_alignment_result_t
  compute_pairwise_time_frequency_alignment(const torch::Tensor &latents,
                                            const mtf_token_batch_t &batch) {
    auto domain_cpu = batch.metadata.domain_id.to(torch::kCPU).contiguous();
    auto channel_cpu = batch.metadata.channel_id.to(torch::kCPU).contiguous();
    auto scale_cpu = batch.metadata.scale_id.to(torch::kCPU).contiguous();
    auto start_cpu = batch.metadata.start_index.to(torch::kCPU).contiguous();
    auto width_cpu = batch.metadata.width.to(torch::kCPU).contiguous();
    auto domain = domain_cpu.accessor<int64_t, 1>();
    auto channel = channel_cpu.accessor<int64_t, 1>();
    auto scale = scale_cpu.accessor<int64_t, 1>();
    auto start = start_cpu.accessor<int64_t, 1>();
    auto width = width_cpu.accessor<int64_t, 1>();

    std::vector<int64_t> time_indices;
    std::vector<int64_t> frequency_indices;
    const int64_t N = batch.metadata.domain_id.size(0);
    for (int64_t i = 0; i < N; ++i) {
      if (domain[i] != 0) {
        continue;
      }
      for (int64_t j = 0; j < N; ++j) {
        if (domain[j] == 1 && channel[i] == channel[j] &&
            scale[i] == scale[j] && start[i] == start[j] &&
            width[i] == width[j]) {
          time_indices.push_back(i);
          frequency_indices.push_back(j);
          break;
        }
      }
    }
    tf_alignment_result_t out{};
    out.loss = torch::zeros({}, latents.options());
    out.pair_count = static_cast<int64_t>(time_indices.size());
    if (time_indices.empty()) {
      return out;
    }
    const auto idx_options =
        torch::TensorOptions().dtype(torch::kInt64).device(config_.device);
    auto time_idx = torch::tensor(time_indices, idx_options);
    auto frequency_idx = torch::tensor(frequency_indices, idx_options);
    auto time_latents = latents.index_select(/*dim=*/1, time_idx);
    auto frequency_latents = latents.index_select(/*dim=*/1, frequency_idx);
    auto time_mask = batch.token_mask.index_select(/*dim=*/1, time_idx);
    auto frequency_mask =
        batch.token_mask.index_select(/*dim=*/1, frequency_idx);
    auto pair_mask = time_mask.logical_and(frequency_mask);
    if (!pair_mask.any().template item<bool>()) {
      return out;
    }
    out.pair_valid_count = pair_mask.sum().item<int64_t>();
    auto pair_loss =
        1.0 - torch::cosine_similarity(time_latents, frequency_latents,
                                       /*dim=*/-1, /*eps=*/1e-8);
    const auto weights = pair_mask.to(pair_loss.dtype());
    out.loss = (pair_loss * weights).sum() / weights.sum().clamp_min(1.0);
    return out;
  }

  [[nodiscard]] mtf_input_t weak_augment(const torch::Tensor &x,
                                         const torch::Tensor &feature_mask) {
    return detail::apply_vicreg_weak_view_augmentation(x, feature_mask,
                                                       config_);
  }

  [[nodiscard]] vicreg_branch_loss_result_t
  compute_vicreg_branch(const torch::Tensor &x,
                        const torch::Tensor &feature_mask) {
    const auto view_a = weak_augment(x, feature_mask);
    const auto view_b = weak_augment(x, feature_mask);
    auto encoded_a = encode(view_a.data, view_a.feature_mask);
    auto encoded_b = encode(view_b.data, view_b.feature_mask);
    auto zero = torch::zeros(
        {}, torch::TensorOptions().dtype(config_.dtype).device(config_.device));
    vicreg_branch_loss_result_t out{};
    out.loss = zero;
    out.global_loss = zero;
    out.channel_loss = zero;
    torch::Tensor channel_joint_mask{};
    if (config_.use_channel_vicreg) {
      channel_joint_mask =
          detail::channel_valid_mask(encoded_a.metadata, encoded_a.token_mask,
                                     config_)
              .logical_and(detail::channel_valid_mask(
                  encoded_b.metadata, encoded_b.token_mask, config_))
              .to(torch::TensorOptions()
                      .dtype(torch::kBool)
                      .device(config_.device));
    }

    ::embedding::vicreg_stability_loss_options_t opts{};
    opts.invariance_weight = config_.vicreg_sim_weight;
    opts.variance_weight = config_.vicreg_var_weight;
    opts.covariance_weight = config_.vicreg_cov_weight;
    opts.variance_floor = config_.vicreg_variance_floor;
    opts.eps = config_.vicreg_variance_epsilon;
    if (config_.use_global_vicreg) {
      auto projected_a =
          project_vicreg(encoded_a.pooled_embedding).unsqueeze(1);
      auto projected_b =
          project_vicreg(encoded_b.pooled_embedding).unsqueeze(1);
      auto mask = encoded_a.token_mask.any(/*dim=*/1)
                      .logical_and(encoded_b.token_mask.any(/*dim=*/1))
                      .unsqueeze(1)
                      .to(torch::TensorOptions()
                              .dtype(torch::kBool)
                              .device(config_.device));
      const auto global_result = ::embedding::compute_vicreg_stability_loss(
          projected_a, mask, projected_b, mask, opts);

      out.global_loss = global_result.loss;
      out.loss = out.loss + config_.lambda_global_vicreg * global_result.loss;
    }
    if (config_.use_channel_vicreg) {
      auto projected_a = project_vicreg(encoded_a.pooled_by_channel);
      auto projected_b = project_vicreg(encoded_b.pooled_by_channel);
      const auto channel_result = ::embedding::compute_vicreg_stability_loss(
          projected_a, channel_joint_mask, projected_b, channel_joint_mask, opts);
      out.channel_loss = channel_result.loss;
      out.loss = out.loss + config_.lambda_channel_vicreg * channel_result.loss;
    }
    return out;
  }

  mtf_jepa_mae_vicreg_config_t config_{};
  TimeFrequencyViewBuilder tokenizer_{nullptr};
  TimeFrequencyViewBuilder target_tokenizer_{nullptr};
  SharedTokenEncoder encoder_{nullptr};
  SharedTokenEncoder target_encoder_{nullptr};
  LatentPredictor predictor_{nullptr};
  MaeDecoder mae_decoder_{nullptr};
  VICRegStabilityHead vicreg_stability_head_{nullptr};
  JEPAContextTargetMasker masker_;
};

TORCH_MODULE(MtfJepaMaeVicreg);

using Model = MtfJepaMaeVicreg;

} // namespace embedding::encoders::mtf_jepa_mae_vicreg
