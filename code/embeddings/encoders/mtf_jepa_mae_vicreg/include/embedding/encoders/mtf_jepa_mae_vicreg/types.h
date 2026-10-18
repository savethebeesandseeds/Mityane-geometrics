// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/mtf_jepa_mae_vicreg/config.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {

using mtf_input_t = ::embedding::input_t;

struct mtf_token_metadata_t {
  torch::Tensor start_index{}; // [N], int64
  torch::Tensor width{};       // [N], int64
  torch::Tensor scale_id{};    // [N], int64
  torch::Tensor channel_id{};  // [N], int64
  torch::Tensor domain_id{};   // [N], int64, 0=time, 1=frequency
};

struct mtf_token_batch_t {
  torch::Tensor tokens{};                           // [B,N,d_model]
  torch::Tensor reconstruction_targets{};           // [B,N,d_model], fallback
  torch::Tensor time_reconstruction_targets{};      // [B,N,2*Dx]
  torch::Tensor frequency_reconstruction_targets{}; // [B,N,K*Dx]
  torch::Tensor time_reconstruction_mask{};         // [B,N,2*Dx], bool
  torch::Tensor frequency_reconstruction_mask{};    // [B,N,K*Dx], bool
  torch::Tensor token_mask{};                       // [B,N], bool
  mtf_token_metadata_t metadata{};
};

struct jepa_context_target_mask_t {
  torch::Tensor context_mask{}; // [B,N], bool
  torch::Tensor target_mask{};  // [B,N], bool
  torch::Tensor valid_mask{};   // [B,N], bool
  // Diagnostic flags do not change the selected tokens. Legacy masks can relax
  // temporal exclusions; strict masks only reduce targets to retain context.
  torch::Tensor support_relaxed{}; // [B], bool; a legacy soft ban was removed
  torch::Tensor targets_reduced{}; // [B], bool; sampled targets were dropped
  torch::Tensor target_resampled{}; // [B], bool; strict mode tried alternatives
};

struct jepa_mask_audit_t {
  int64_t sample_count{0};
  int64_t valid_tokens{0};
  int64_t context_tokens{0};
  int64_t target_tokens{0};
  int64_t overlapping_target_tokens{0};
  double target_overlap_fraction{0.0};
  // Union lengths in sample/channel/time coordinates, not token-pair counts.
  int64_t target_support_points{0};
  int64_t shared_support_points{0};
  double shared_support_ratio{0.0};
  int64_t support_relaxed_samples{0};
  int64_t target_reduced_samples{0};
  int64_t target_resampled_samples{0};
  int64_t no_target_samples{0};
  int64_t valid_no_target_samples{0};
  // Metadata describes full temporal windows, so this overestimates observed
  // raw support for sparse feature masks and does not measure channel leakage.
  bool conservative_support{true};
};

struct mtf_jepa_mae_vicreg_encode_output_t {
  torch::Tensor embeddings{};         // [B,N,latent_dim]
  torch::Tensor pooled_embedding{};   // [B,latent_dim]
  torch::Tensor pooled_by_channel{};  // [B,C,latent_dim]
  torch::Tensor pooled_time{};        // [B,latent_dim]
  torch::Tensor pooled_frequency{};   // [B,latent_dim]
  torch::Tensor token_mask{};         // [B,N]
  torch::Tensor sample_valid_mask{};  // [B]
  torch::Tensor channel_valid_mask{}; // [B,C]
  mtf_token_metadata_t metadata{};
};

struct mtf_jepa_mae_vicreg_output_t {
  torch::Tensor embeddings{};
  torch::Tensor pooled_embedding{};
  torch::Tensor pooled_by_channel{};
  torch::Tensor pooled_time{};
  torch::Tensor pooled_frequency{};
  torch::Tensor loss{};
  torch::Tensor loss_jepa{};
  torch::Tensor loss_mae{};
  torch::Tensor loss_mae_time{};
  torch::Tensor loss_mae_frequency{};
  torch::Tensor loss_tf_align{};
  torch::Tensor loss_vicreg{};
  torch::Tensor loss_vicreg_global{};
  torch::Tensor loss_vicreg_channel{};
  torch::Tensor jepa_target_mask{};
  torch::Tensor jepa_context_mask{};
  torch::Tensor sample_valid_mask{};
  torch::Tensor channel_valid_mask{};
};

struct mae_decoder_output_t {
  torch::Tensor projected{};
  torch::Tensor time{};
  torch::Tensor frequency{};
};

struct tf_alignment_result_t {
  torch::Tensor loss{};
  int64_t pair_count{0};
  int64_t pair_valid_count{0};
};

struct vicreg_branch_loss_result_t {
  torch::Tensor loss{};
  torch::Tensor global_loss{};
  torch::Tensor channel_loss{};
};

using EncodeOutput = mtf_jepa_mae_vicreg_encode_output_t;
using TrainingOutput = mtf_jepa_mae_vicreg_output_t;

} // namespace embedding::encoders::mtf_jepa_mae_vicreg
