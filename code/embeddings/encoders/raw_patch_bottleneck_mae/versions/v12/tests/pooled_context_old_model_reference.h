// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/encoder.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/masking.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/objectives.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/tokenization.h"
#include <limits>

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct PrePooledModelImpl : torch::nn::Module {
  explicit PrePooledModelImpl(const Config &config) : config_(config) {
    validate_config(config_);
    const auto W = config_.encoder_width, K = config_.history_length / config_.patch_length;
    const auto PF = config_.patch_length * config_.input_width;
    patch_projection = register_module("patch_projection", torch::nn::Linear(PF * 2, W));
    positions = register_module("positions", torch::nn::Embedding(K, W));
    channels = register_module("channels", torch::nn::Embedding(config_.channel_count, W));
    for (int64_t i = 0; i < config_.num_layers; ++i) {
      auto block = PreNormBlock(config_);
      register_module("block_" + std::to_string(i), block);
      blocks.push_back(block);
    }
    final_norm = register_module("final_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({W}).eps(config_.layer_norm_epsilon)));
    pool_score = register_module("pool_score", torch::nn::Linear(W, 1));
    pool_positions = register_module("pool_positions", torch::nn::Embedding(K, 1));
    export_projection = register_module("export_projection", torch::nn::Linear(W, config_.export_width));
    decoder_positions = register_module("decoder_positions", torch::nn::Embedding(K, 16));
    decoder_channels = register_module("decoder_channels", torch::nn::Embedding(config_.channel_count, 16));
    decoder_first = register_module("decoder_first", torch::nn::Linear(config_.export_width + 32, config_.decoder_hidden_width));
    decoder_second = register_module("decoder_second", torch::nn::Linear(config_.decoder_hidden_width, PF));
    // Register last so enabling the mixer preserves every common initial weight.
    for (int64_t i = 0; i < config_.channel_mixer_layers; ++i) {
      auto block = PreNormBlock(config_);
      register_module("channel_mixer_block_" + std::to_string(i), block);
      channel_mixer_blocks.push_back(block);
    }
    // Register after every old module, including the optional mixer, to keep
    // common initialized weights and mode-0/1 random streams exactly unchanged.
    if (config_.global_bottleneck_mode == 2) {
      global_pool_first = register_module("global_pool_first",
          torch::nn::Linear(config_.channel_count * (config_.export_width + 1), W));
      global_pool_second = register_module("global_pool_second",
          torch::nn::Linear(W, config_.export_width));
    }
    if (config_.global_bottleneck_mode == 3) {
      global_patch_pool_first = register_module("global_patch_pool_first",
          torch::nn::Linear(config_.channel_count * K * (W + 1), W));
      global_patch_pool_second = register_module("global_patch_pool_second",
          torch::nn::Linear(W, config_.export_width));
    }
    to(config_.device, config_.dtype);
  }

  const Config &config() const { return config_; }

  EncodeOutput encode(const Input &input) {
    const auto patches = tokenize_visible(input, config_);
    const auto B = input.data.size(0), C = config_.channel_count, D = config_.export_width;
    auto local = torch::zeros({B * C, D}, torch::TensorOptions().dtype(config_.dtype).device(config_.device));
    torch::Tensor contextual;
    if (config_.channel_mixer_layers > 0) contextual = torch::zeros_like(local);
    torch::Tensor patch_global, contextual_patch_global;
    if (config_.global_bottleneck_mode == 3) {
      patch_global = torch::zeros({B, D}, local.options());
      if (config_.channel_mixer_layers > 0) contextual_patch_global = torch::zeros_like(patch_global);
    }
    if (patches.row_indices.numel() != 0) {
      auto projected = patch_projection(torch::cat({patches.values, patches.visibility.to(config_.dtype)}, -1));
      // Public original indices keep padding=-1; only private embedding lookup uses 0.
      auto lookup_positions = patches.positions.clamp_min(0);
      auto h = projected + positions(lookup_positions) + channels(patches.channel_indices).unsqueeze(1);
      h = torch::where(patches.valid.unsqueeze(-1), h, torch::zeros_like(h));
      // Retain the original ordinary temporal path for independent local
      // diagnostics. The early contextual path uses the same registered blocks,
      // with ordinary autograd on both passes and no new parameters or RNG.
      torch::Tensor early_contextual;
      if (config_.channel_mixer_placement == 1) {
        early_contextual = mix_aligned_channels(h, patches, B);
        for (auto &block : blocks) early_contextual = block(early_contextual, patches.valid);
        early_contextual = torch::where(patches.valid.unsqueeze(-1), final_norm(early_contextual),
                                         torch::zeros_like(early_contextual));
      }
      for (auto &block : blocks) h = block(h, patches.valid);
      h = torch::where(patches.valid.unsqueeze(-1), final_norm(h), torch::zeros_like(h));
      if (config_.global_bottleneck_mode == 3)
        patch_global = learned_patch_global(h, patches, B, input.channel_ids);
      auto scores = pool_score(torch::tanh(h)).squeeze(-1) + pool_positions(lookup_positions).squeeze(-1);
      scores = scores.masked_fill(patches.valid.logical_not(), -std::numeric_limits<float>::infinity());
      auto weights = torch::softmax(scores, 1);
      auto pooled = (h * weights.unsqueeze(-1)).sum(1);
      auto z = export_projection(pooled);
      local = local.index_copy(0, patches.row_indices, z);
      if (config_.channel_mixer_layers > 0) {
        auto mixed = config_.channel_mixer_placement == 1 ? early_contextual :
                                                           mix_aligned_channels(h, patches, B);
        if (config_.global_bottleneck_mode == 3)
          contextual_patch_global = learned_patch_global(mixed, patches, B, input.channel_ids);
        auto contextual_scores = pool_score(torch::tanh(mixed)).squeeze(-1) +
            pool_positions(lookup_positions).squeeze(-1);
        contextual_scores = contextual_scores.masked_fill(
            patches.valid.logical_not(), -std::numeric_limits<float>::infinity());
        auto contextual_weights = torch::softmax(contextual_scores, 1);
        auto contextual_pooled = (mixed * contextual_weights.unsqueeze(-1)).sum(1);
        contextual = contextual.index_copy(0, patches.row_indices,
                                           export_projection(contextual_pooled));
      }
    }
    EncodeOutput out;
    out.z_local = local.reshape({B, C, D});
    out.channel_valid_mask = patches.patch_counts.gt(0);
    out.sample_valid_mask = out.channel_valid_mask.any(1);
    const auto denominator = out.channel_valid_mask.sum(1).clamp_min(1).to(config_.dtype).unsqueeze(-1);
    out.z_global = out.z_local.sum(1) / denominator;
    if (config_.channel_mixer_layers > 0) {
      out.z_contextual = contextual.reshape({B, C, D});
      out.z_contextual_global = out.z_contextual.sum(1) / denominator;
    }
    out.visible_observation_counts = patches.observation_counts;
    out.visible_patch_counts = patches.patch_counts;
    out.channel_ids = (input.channel_ids.dim() == 1 ? input.channel_ids.unsqueeze(0).expand({B, C}) : input.channel_ids).to(config_.device);
    if (config_.global_bottleneck_mode == 2) {
      out.z_global = learned_global(out.z_local, out.channel_valid_mask, out.channel_ids);
      if (config_.channel_mixer_layers > 0)
        out.z_contextual_global = learned_global(out.z_contextual, out.channel_valid_mask,
                                                 out.channel_ids);
    }
    if (config_.global_bottleneck_mode == 3) {
      out.z_global = patch_global;
      if (config_.channel_mixer_layers > 0) out.z_contextual_global = contextual_patch_global;
    }
    TORCH_CHECK(torch::isfinite(out.z_local).all().item<bool>() && torch::isfinite(out.z_global).all().item<bool>(),
                "[rpb-mae] nonfinite encoding");
    if (config_.channel_mixer_layers > 0)
      TORCH_CHECK(torch::isfinite(out.z_contextual).all().item<bool>() &&
                      torch::isfinite(out.z_contextual_global).all().item<bool>(),
                  "[rpb-mae] nonfinite contextual encoding");
    return out;
  }

  // This is also the explicit intervention surface: decoder receives exact exports + metadata.
  torch::Tensor decode(const torch::Tensor &compact_export, const torch::Tensor &ids) {
    TORCH_CHECK(compact_export.defined() &&
                    (compact_export.dim() == 2 || compact_export.dim() == 3),
                "[rpb-mae] decoder requires the configured compact export");
    const auto B = compact_export.size(0), C = config_.channel_count;
    const auto K = config_.history_length / config_.patch_length;
    torch::Tensor per_channel;
    if (config_.global_bottleneck_mode > 0) {
      TORCH_CHECK(compact_export.dim() == 2 && compact_export.size(1) == config_.export_width,
                  "[rpb-mae] global decoder requires the exact served BD global export");
      per_channel = compact_export.unsqueeze(1).expand({B, C, config_.export_width});
    } else {
      TORCH_CHECK(compact_export.dim() == 3 && compact_export.size(1) == C &&
                      compact_export.size(2) == config_.export_width,
                  "[rpb-mae] per-channel decoder requires the exact served BCD export");
      per_channel = compact_export;
    }
    auto semantic = channel_indices(ids, config_, B, config_.device);
    auto patch = torch::arange(K, semantic.options());
    auto metadata_position = decoder_positions(patch).view({1, 1, K, 16}).expand({B, C, K, 16});
    auto metadata_channel = decoder_channels(semantic).unsqueeze(2).expand({B, C, K, 16});
    auto input = torch::cat({per_channel.unsqueeze(2).expand({B, C, K, config_.export_width}), metadata_position, metadata_channel}, -1);
    return decoder_second(torch::gelu(decoder_first(input)))
        .reshape({B, C, config_.history_length, config_.input_width});
  }

  ForwardOutput forward(const Input &input, const torch::Tensor &hidden) {
    validate_input(input, config_, true);
    const auto observed = input.observed.to(config_.device);
    const auto masks = mask_from_hidden(observed, hidden.to(config_.device), config_);
    Input visible{input.data, masks.visible, input.channel_ids, input.endpoints, input.sampling_interval};
    ForwardOutput out;
    out.encoding = encode(visible);
    out.reconstruction = decode(compact_reconstruction_export(out.encoding, config_), input.channel_ids);
    auto losses = hierarchical_huber(out.reconstruction, input.data.to(config_.device).detach(),
                                    masks.target, masks.eligible_channels, config_.huber_delta);
    out.loss = losses.loss; out.target_counts = losses.target_counts;
    out.eligible_channels = losses.eligible_channels; out.eligible_examples = losses.eligible_examples;
    out.eligible_channel_count = losses.eligible_channel_count;
    out.eligible_example_count = losses.eligible_example_count; out.target_cell_count = losses.target_cell_count;
    return out;
  }

private:
  torch::Tensor learned_patch_global(const torch::Tensor &h, const VisiblePatches &patches,
                                     int64_t B, const torch::Tensor &ids) {
    const auto C = config_.channel_count, K = config_.history_length / config_.patch_length;
    const auto W = config_.encoder_width;
    // Packed rank is not time: scatter only valid states to original patch slots.
    const auto packed = torch::nonzero(patches.valid.reshape({-1})).reshape({-1});
    const auto lookup = (patches.row_indices.unsqueeze(1) * K +
                         patches.positions.clamp_min(0)).reshape({-1});
    const auto indices = lookup.index_select(0, packed);
    const auto grid = torch::zeros({B * C * K, W}, h.options()).index_copy(
        0, indices, h.reshape({-1, W}).index_select(0, packed)).reshape({B, C, K, W});
    const auto valid = torch::zeros({B * C * K}, patches.valid.options())
                           .index_fill(0, indices, true).reshape({B, C, K});
    const auto order = channel_indices(ids, config_, B, config_.device).argsort(int64_t{1});
    const auto canonical = grid.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, C, K, W}));
    const auto support = valid.gather(1, order.unsqueeze(-1).expand({B, C, K}));
    const auto observed = torch::where(support.unsqueeze(-1), canonical, torch::zeros_like(canonical));
    const auto features = torch::cat({observed.flatten(1), support.flatten(1).to(config_.dtype)}, 1);
    const auto pooled = global_patch_pool_second(torch::gelu(global_patch_pool_first(features)));
    return torch::where(support.flatten(1).any(1).unsqueeze(-1), pooled, torch::zeros_like(pooled));
  }

  torch::Tensor learned_global(const torch::Tensor &vectors, const torch::Tensor &valid,
                               const torch::Tensor &ids) {
    const auto B = vectors.size(0), C = config_.channel_count, D = config_.export_width;
    const auto semantic = channel_indices(ids, config_, B, config_.device);
    const auto order = semantic.argsort(int64_t{1});
    const auto canonical = vectors.gather(1, order.unsqueeze(-1).expand({B, C, D}));
    const auto support = valid.gather(1, order);
    // Config semantic order is canonical; physical storage positions carry no
    // identity. Absent channels contribute zeros and explicit observed bits.
    const auto observed_vectors = torch::where(support.unsqueeze(-1), canonical,
                                               torch::zeros_like(canonical));
    const auto features = torch::cat({observed_vectors.flatten(1),
                                     support.to(config_.dtype)}, 1);
    const auto pooled = global_pool_second(torch::gelu(global_pool_first(features)));
    return torch::where(valid.any(1).unsqueeze(-1), pooled, torch::zeros_like(pooled));
  }

  torch::Tensor mix_aligned_channels(const torch::Tensor &h, const VisiblePatches &patches,
                                    int64_t batch) {
    const auto C = config_.channel_count, K = config_.history_length / config_.patch_length;
    const auto W = config_.encoder_width;
    // Packed rank differs by channel: restore original patch coordinates first.
    const auto packed = torch::nonzero(patches.valid.reshape({-1})).reshape({-1});
    const auto grid_lookup = (patches.row_indices.unsqueeze(1) * K +
                              patches.positions.clamp_min(0)).reshape({-1});
    const auto grid_indices = grid_lookup.index_select(0, packed);
    auto grid = torch::zeros({batch * C * K, W}, h.options()).index_copy(
        0, grid_indices, h.reshape({-1, W}).index_select(0, packed));
    auto grid_valid = torch::zeros({batch * C * K}, patches.valid.options())
                          .index_fill(0, grid_indices, true);
    auto groups = grid.reshape({batch, C, K, W}).permute({0, 2, 1, 3})
                      .reshape({batch * K, C, W});
    auto group_valid = grid_valid.reshape({batch, C, K}).permute({0, 2, 1})
                           .reshape({batch * K, C});
    const auto active_groups = torch::nonzero(group_valid.any(1)).reshape({-1});
    auto active = groups.index_select(0, active_groups);
    auto active_valid = group_valid.index_select(0, active_groups);
    for (auto &block : channel_mixer_blocks) active = block(active, active_valid);
    auto mixed_grid = torch::zeros_like(groups).index_copy(0, active_groups, active)
                          .reshape({batch, K, C, W}).permute({0, 2, 1, 3})
                          .reshape({batch * C * K, W});
    auto mixed_packed = mixed_grid.index_select(0, grid_lookup).reshape_as(h);
    return torch::where(patches.valid.unsqueeze(-1), mixed_packed, torch::zeros_like(h));
  }

  Config config_;
  torch::nn::Linear patch_projection{nullptr}, pool_score{nullptr}, export_projection{nullptr};
  torch::nn::Linear decoder_first{nullptr}, decoder_second{nullptr};
  torch::nn::Linear global_pool_first{nullptr}, global_pool_second{nullptr};
  torch::nn::Linear global_patch_pool_first{nullptr}, global_patch_pool_second{nullptr};
  torch::nn::Embedding positions{nullptr}, channels{nullptr}, pool_positions{nullptr};
  torch::nn::Embedding decoder_positions{nullptr}, decoder_channels{nullptr};
  torch::nn::LayerNorm final_norm{nullptr};
  std::vector<PreNormBlock> blocks;
  std::vector<PreNormBlock> channel_mixer_blocks;
};
TORCH_MODULE(PrePooledModel);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
