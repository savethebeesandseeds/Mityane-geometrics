// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/model.h"
#include <array>
#include <cmath>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kTemporalRelationModelTag = "RPB-v14";
inline constexpr const char *kTemporalRelationArchitectureId =
    "early-raw-global-plus-common-support-temporal-relations-v1";
inline constexpr const char *kTemporalRelationBankLayout =
    "A108-S108-support108;semantic-pairs-01-02-12;feature-i-major-feature-j-major;spacing-1-2-3-4";
inline constexpr int64_t kTemporalRelationBankWidth = 324;
inline constexpr int64_t kTemporalRelationParameterCount = 236173;
inline constexpr double kTemporalRelationDenominatorFloor = 1e-12;

inline void validate_temporal_relation_config(const Config &c) {
  validate_config(c);
  TORCH_CHECK(c.channel_count == 3 && c.history_length == 32 && c.input_width == 3 &&
                  c.patch_length == 8 && c.encoder_width == 64 && c.export_width == 32 &&
                  c.num_layers == 3 && c.num_heads == 4 && c.feedforward_width == 256 &&
                  c.decoder_hidden_width == 128 && c.channel_mixer_layers == 1 &&
                  c.channel_mixer_placement == 1 && c.global_bottleneck_mode == 2 &&
                  c.global_pool_input_source == 0 && c.temporal_difference_input == 0 &&
                  c.dropout == 0,
              "[rpb-relations] requires the unchanged closed early-v10 backbone");
}

struct TemporalRelationBankOutput {
  torch::Tensor values;        // [B,324], float32, A108 then S108 then support108.
  torch::Tensor common_counts; // [B,108], int64; identical support for all moments.
};

// Original time coordinates are used throughout. A k-spaced difference uses
// observed endpoints, not interpolation or a neighbouring packed token. Each
// product requires both complete observed three-time feature triplets.
inline TemporalRelationBankOutput temporal_relation_bank(const Input &input, const Config &c) {
  validate_temporal_relation_config(c);
  validate_input(input, c, true);
  const auto B = input.data.size(0), H = c.history_length, F = c.input_width;
  auto data = input.data.to(c.device);
  auto mask = input.observed.to(data.device());
  const auto order = channel_indices(input.channel_ids, c, B, data.device()).argsort(int64_t{1});
  data = data.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, 3, H, F}));
  mask = mask.gather(1, order.unsqueeze(-1).unsqueeze(-1).expand({B, 3, H, F}));
  // Sanitize before subtraction/multiplication. Double CUDA arithmetic avoids
  // overflow in products of otherwise finite float32 observations. No model
  // or optimizer is executed on CPU by this helper.
  const auto safe = torch::where(mask, data, torch::zeros_like(data)).to(torch::kFloat64);
  constexpr std::array<std::array<int64_t, 2>, 3> pairs{{{0, 1}, {0, 2}, {1, 2}}};
  std::vector<torch::Tensor> antisymmetric, symmetric, support, counts;
  for (const auto &pair : pairs) {
    std::vector<torch::Tensor> pa, ps, pq, pn;
    for (int64_t k = 1; k <= 4; ++k) {
      const auto T = H - 2 * k;
      const auto centre = safe.narrow(2, k, T);
      const auto left = centre - safe.narrow(2, 0, T);
      const auto right = safe.narrow(2, 2 * k, T) - centre;
      const auto legal = mask.narrow(2, 0, T).logical_and(mask.narrow(2, k, T))
                            .logical_and(mask.narrow(2, 2 * k, T));
      const auto common = legal.select(1, pair[0]).unsqueeze(-1)
                              .logical_and(legal.select(1, pair[1]).unsqueeze(-2));
      const auto li = torch::where(common, left.select(1, pair[0]).unsqueeze(-1),
                                   torch::zeros_like(left.select(1, pair[0]).unsqueeze(-1)));
      const auto ri = torch::where(common, right.select(1, pair[0]).unsqueeze(-1),
                                   torch::zeros_like(right.select(1, pair[0]).unsqueeze(-1)));
      const auto lj = torch::where(common, left.select(1, pair[1]).unsqueeze(-2),
                                   torch::zeros_like(left.select(1, pair[1]).unsqueeze(-2)));
      const auto rj = torch::where(common, right.select(1, pair[1]).unsqueeze(-2),
                                   torch::zeros_like(right.select(1, pair[1]).unsqueeze(-2)));
      const auto n = common.sum(1);
      const auto divisor = n.clamp_min(1).to(torch::kFloat64);
      const auto ei = (li.square() + ri.square()).sum(1) / divisor;
      const auto ej = (lj.square() + rj.square()).sum(1) / divisor;
      // Clamp before sqrt: the inactive zero-energy branch must also have a
      // finite input derivative, rather than an infinity multiplied by zero.
      const auto denominator = (ei * ej).clamp_min(
          kTemporalRelationDenominatorFloor * kTemporalRelationDenominatorFloor).sqrt();
      const auto active = n.gt(0).logical_and(ei.gt(0)).logical_and(ej.gt(0));
      const auto a = (li * rj - ri * lj).sum(1) / divisor / denominator;
      const auto s = (li * lj + ri * rj).sum(1) / divisor / denominator;
      pa.push_back(torch::where(active, a, torch::zeros_like(a)).flatten(1));
      ps.push_back(torch::where(active, s, torch::zeros_like(s)).flatten(1));
      pq.push_back((n.to(torch::kFloat64) / double(T)).flatten(1));
      pn.push_back(n.flatten(1));
    }
    // Feature-i / feature-j / spacing order, not spacing-major order.
    antisymmetric.push_back(torch::stack(pa, -1).flatten(1));
    symmetric.push_back(torch::stack(ps, -1).flatten(1));
    support.push_back(torch::stack(pq, -1).flatten(1));
    counts.push_back(torch::stack(pn, -1).flatten(1));
  }
  TemporalRelationBankOutput out;
  out.values = torch::cat({torch::cat(antisymmetric, 1), torch::cat(symmetric, 1),
                           torch::cat(support, 1)}, 1).to(c.dtype);
  out.common_counts = torch::cat(counts, 1);
  TORCH_CHECK(out.values.sizes() == torch::IntArrayRef({B, kTemporalRelationBankWidth}) &&
                  out.common_counts.sizes() == torch::IntArrayRef({B, 108}) &&
                  torch::isfinite(out.values).all().item<bool>(),
              "[rpb-relations] nonfinite or malformed temporal bank");
  return out;
}

struct TemporalRelationProjectionImpl : torch::nn::Module {
  TemporalRelationProjectionImpl() {
    // Literal deterministic initialization consumes neither CPU nor CUDA RNG.
    // All pairs/features/spacings participate, with no task label or lag choice.
    weight = register_parameter("weight", torch::zeros({32, kTemporalRelationBankWidth}, torch::kFloat32));
    torch::NoGradGuard guard;
    auto w = weight.accessor<float, 2>();
    for (int64_t pair = 0; pair < 3; ++pair)
      for (int64_t k = 0; k < 4; ++k)
        for (int64_t fi = 0; fi < 3; ++fi)
          for (int64_t fj = 0; fj < 3; ++fj) {
            const auto column = (pair * 9 + fi * 3 + fj) * 4 + k;
            w[pair * 4 + k][column] = 1.f / 9.f;
            w[12 + pair * 4 + k][108 + column] = 1.f / 9.f;
          }
    const auto scale = static_cast<float>(1.0 / std::sqrt(108.0));
    for (int64_t row = 0; row < 8; ++row)
      for (int64_t column = 0; column < 108; ++column) {
        auto bits = static_cast<unsigned>((row + 1) & column);
        unsigned parity = 0;
        while (bits) { parity ^= bits & 1U; bits >>= 1U; }
        w[24 + row][216 + column] = parity ? -scale : scale;
      }
  }
  torch::Tensor forward(const torch::Tensor &bank) {
    TORCH_CHECK(bank.dim() == 2 && bank.size(1) == kTemporalRelationBankWidth &&
                    bank.scalar_type() == weight.scalar_type() && bank.device() == weight.device(),
                "[rpb-relations] exact model-device float32 bank required");
    return torch::matmul(bank, weight.transpose(0, 1));
  }
  torch::Tensor weight;
};
TORCH_MODULE(TemporalRelationProjection);

// Isolated wrapper: old Model, Config, serialization and historical factories
// remain untouched. Checkpoints for this type need an explicitly new contract.
struct TemporalRelationModelImpl : torch::nn::Module {
  explicit TemporalRelationModelImpl(const Config &c) : config_(c) {
    validate_temporal_relation_config(c);
    TORCH_CHECK(c.device.is_cuda(), "[rpb-relations] model requires actual CUDA");
    backbone = register_module("backbone", Model(c));
    temporal_relation_projection = register_module("temporal_relation_projection", TemporalRelationProjection());
    temporal_relation_projection->to(c.device, c.dtype);
  }
  const Config &config() const { return config_; }
  EncodeOutput encode(const Input &input) {
    const auto bank = temporal_relation_bank(input, config_);
    auto out = backbone->encode(input);
    const auto residual = temporal_relation_projection(bank.values);
    out.z_contextual_global = torch::where(out.sample_valid_mask.unsqueeze(-1),
        out.z_contextual_global + residual, torch::zeros_like(out.z_contextual_global));
    TORCH_CHECK(torch::isfinite(out.z_contextual_global).all().item<bool>(),
                "[rpb-relations] nonfinite native32");
    return out;
  }
  torch::Tensor decode(const torch::Tensor &native32, const torch::Tensor &ids) {
    return backbone->decode(native32, ids);
  }
  ForwardOutput forward(const Input &input, const torch::Tensor &hidden) {
    validate_input(input, config_, true);
    const auto masks = mask_from_hidden(input.observed.to(config_.device), hidden.to(config_.device), config_);
    Input visible{input.data, masks.visible, input.channel_ids, input.endpoints, input.sampling_interval};
    ForwardOutput out;
    out.encoding = encode(visible);
    out.reconstruction = decode(compact_reconstruction_export(out.encoding, config_), input.channel_ids);
    const auto losses = hierarchical_huber(out.reconstruction, input.data.to(config_.device).detach(),
                                          masks.target, masks.eligible_channels, config_.huber_delta);
    out.loss = losses.loss; out.target_counts = losses.target_counts;
    out.eligible_channels = losses.eligible_channels; out.eligible_examples = losses.eligible_examples;
    out.eligible_channel_count = losses.eligible_channel_count;
    out.eligible_example_count = losses.eligible_example_count; out.target_cell_count = losses.target_cell_count;
    return out;
  }
  Model backbone{nullptr};
  TemporalRelationProjection temporal_relation_projection{nullptr};
private:
  Config config_;
};
TORCH_MODULE(TemporalRelationModel);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
