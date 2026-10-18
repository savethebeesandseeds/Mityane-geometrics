// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "embedding/shared/types.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {

struct mtf_jepa_mae_vicreg_config_t {
  int64_t channel_count{1};
  int64_t history_length{64};
  int64_t input_width{9};

  int64_t d_model{32};
  int64_t latent_dim{32};
  int64_t projector_dim{64};
  int64_t predictor_hidden_dim{64};
  int64_t num_encoder_layers{2};
  int64_t num_predictor_layers{2};
  int64_t num_decoder_layers{1};
  int64_t num_heads{4};
  double dropout{0.10};

  std::vector<int64_t> time_scales{8, 16, 32, 64};
  std::vector<int64_t> scale_strides{4, 8, 16, 32};
  bool use_frequency_tokens{true};
  int64_t frequency_num_bins{16};
  bool frequency_log_magnitude{true};

  double mask_ratio_time{0.50};
  double mask_ratio_frequency{0.30};
  double mask_ratio_channel{0.10};
  double min_context_ratio{0.25};

  double lambda_jepa{1.0};
  double lambda_mae{0.25};
  double lambda_tf_align{0.10};
  double lambda_vicreg{0.05};

  double vicreg_sim_weight{25.0};
  double vicreg_var_weight{25.0};
  double vicreg_cov_weight{1.0};
  double vicreg_variance_floor{1.0};
  double vicreg_variance_epsilon{1e-4};

  double target_ema_tau{0.996};
  bool use_target_ema{true};
  bool stop_gradient_target{true};

  bool use_mae_decoder{true};
  bool use_jepa_loss{true};
  bool use_tf_align_loss{true};
  bool use_vicreg_loss{true};
  bool use_global_vicreg{true};
  bool use_channel_vicreg{false};
  double lambda_global_vicreg{1.0};
  double lambda_channel_vicreg{1.0};
  bool use_raw_reconstruction_targets{true};
  bool strict_finite_loss{true};
  bool strict_jepa_support{false};
  bool couple_time_frequency_masks{true};
  bool mask_same_window_across_domains{true};
  bool mask_same_channel_block{false};
  double max_context_target_time_overlap{0.50};

  double vicreg_view_gaussian_jitter_std{0.005};
  double vicreg_view_time_dropout_scale{0.10};

  torch::Dtype dtype{torch::kFloat32};
  torch::Device device{torch::kCPU};
};

using Config = mtf_jepa_mae_vicreg_config_t;

inline ::embedding::input_shape_t input_shape(const Config &config) {
  return {config.channel_count, config.history_length, config.input_width,
          config.dtype, config.device};
}

namespace detail {

inline void validate_probability(double value, const char *name) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::runtime_error(std::string("[embedding] invalid ") +
                             name);
  }
}

inline void validate_nonnegative_scalar(double value, const char *name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::runtime_error(std::string("[embedding] invalid ") + name);
  }
}

inline void validate_positive_scalar(double value, const char *name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(std::string("[embedding] invalid ") + name);
  }
}


inline std::vector<int64_t>
resolved_scale_strides(const mtf_jepa_mae_vicreg_config_t &config) {
  if (config.scale_strides.empty()) {
    std::vector<int64_t> out;
    out.reserve(config.time_scales.size());
    for (const auto scale : config.time_scales) {
      out.push_back(std::max<int64_t>(1, scale / 2));
    }
    return out;
  }
  return config.scale_strides;
}

inline void
validate_architecture_config(const mtf_jepa_mae_vicreg_config_t &config) {
  if (config.channel_count <= 0 || config.history_length <= 0 ||
      config.input_width <= 0 || config.d_model <= 0 ||
      config.latent_dim <= 0 || config.projector_dim <= 0 ||
      config.predictor_hidden_dim <= 0 || config.num_encoder_layers < 0 ||
      config.num_predictor_layers < 1 || config.num_decoder_layers < 1 ||
      config.num_heads <= 0 || config.frequency_num_bins <= 0) {
    throw std::runtime_error(
        "[embedding] invalid architecture dimensions");
  }
  if (config.latent_dim % config.num_heads != 0) {
    throw std::runtime_error(
        "[embedding] latent_dim must be divisible by num_heads");
  }
  if (config.time_scales.empty()) {
    throw std::runtime_error("[embedding] time_scales is empty");
  }
  const auto strides = resolved_scale_strides(config);
  if (strides.size() != config.time_scales.size()) {
    throw std::runtime_error(
        "[embedding] scale_strides/time_scales size mismatch");
  }
  for (std::size_t i = 0; i < config.time_scales.size(); ++i) {
    if (config.time_scales[i] <= 0 || strides[i] <= 0) {
      throw std::runtime_error(
          "[embedding] time scale and stride must be positive");
    }
  }

}

inline void
validate_training_config(const mtf_jepa_mae_vicreg_config_t &config) {
  if (!std::isfinite(config.dropout) || config.dropout < 0.0 ||
      config.dropout >= 1.0) {
    throw std::runtime_error("[embedding] invalid dropout");
  }
  validate_probability(config.mask_ratio_time, "mask_ratio_time");
  validate_probability(config.mask_ratio_frequency, "mask_ratio_frequency");
  validate_probability(config.mask_ratio_channel, "mask_ratio_channel");
  validate_probability(config.min_context_ratio, "min_context_ratio");
  validate_probability(config.max_context_target_time_overlap,
                       "max_context_target_time_overlap");

  validate_nonnegative_scalar(config.vicreg_view_gaussian_jitter_std,
                              "vicreg_view_gaussian_jitter_std");
  validate_probability(config.vicreg_view_time_dropout_scale,
                       "vicreg_view_time_dropout_scale");
  validate_nonnegative_scalar(config.lambda_jepa, "lambda_jepa");
  validate_nonnegative_scalar(config.lambda_mae, "lambda_mae");
  validate_nonnegative_scalar(config.lambda_tf_align, "lambda_tf_align");
  validate_nonnegative_scalar(config.lambda_vicreg, "lambda_vicreg");
  validate_nonnegative_scalar(config.lambda_global_vicreg, "lambda_global_vicreg");
  validate_nonnegative_scalar(config.lambda_channel_vicreg, "lambda_channel_vicreg");
  validate_nonnegative_scalar(config.vicreg_sim_weight, "vicreg_sim_weight");
  validate_nonnegative_scalar(config.vicreg_var_weight, "vicreg_var_weight");
  validate_nonnegative_scalar(config.vicreg_cov_weight, "vicreg_cov_weight");
  validate_positive_scalar(config.vicreg_variance_floor, "vicreg_variance_floor");
  validate_positive_scalar(config.vicreg_variance_epsilon, "vicreg_variance_epsilon");
  validate_probability(config.target_ema_tau, "target_ema_tau");
}

inline void validate_config(const mtf_jepa_mae_vicreg_config_t &config) {
  validate_architecture_config(config);
  validate_training_config(config);
}

} // namespace detail

inline void validate_config(const Config &config) { detail::validate_config(config); }

} // namespace embedding::encoders::mtf_jepa_mae_vicreg
