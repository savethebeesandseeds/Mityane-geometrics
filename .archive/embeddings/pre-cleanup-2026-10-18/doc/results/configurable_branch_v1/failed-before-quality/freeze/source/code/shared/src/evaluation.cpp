// SPDX-License-Identifier: MIT
#include "embedding/shared/evaluation.h"
#include <ATen/ops/linalg_eigvalsh.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace embedding {
namespace {
constexpr int64_t classes = 4;

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[embedding evaluate] " + message);
}
} // namespace

SyntheticEvaluationData make_evaluation_data(const input_shape_t &config, int64_t samples,
                                             uint64_t signal_seed, uint64_t mask_seed,
                                             uint64_t label_seed) {
  require(config.channel_count > 0 && config.input_width > 0, "evaluation requires positive channel_count and input_width");
  require(samples >= classes && config.history_length >= 4, "evaluation requires at least four samples and history_length>=4");
  std::mt19937_64 signal_rng(signal_seed), mask_rng(mask_seed), label_rng(label_seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<int64_t> labels(static_cast<size_t>(samples));
  for (int64_t b = 0; b < samples; ++b) labels[b] = b % classes;
  std::shuffle(labels.begin(), labels.end(), label_rng);
  auto clean = torch::zeros({samples, config.channel_count, config.history_length, config.input_width}, torch::kFloat32);
  auto mask = torch::ones(clean.sizes(), torch::kBool);
  auto x = clean.accessor<float, 4>();
  auto observed = mask.accessor<bool, 4>();
  constexpr double pi = 3.14159265358979323846;
  for (int64_t b = 0; b < samples; ++b) {
    const double phase = 2 * pi * uniform(signal_rng);
    const double gain = 0.8 + 0.4 * uniform(signal_rng);
    const double offset = 0.2 * normal(signal_rng);
    const double missing = 0.35 * uniform(mask_rng);
    for (int64_t c = 0; c < config.channel_count; ++c)
      for (int64_t h = 0; h < config.history_length; ++h) {
        const double t = static_cast<double>(h) / (config.history_length - 1);
        for (int64_t f = 0; f < config.input_width; ++f) {
          const double local_phase = phase + 0.17 * c + 0.23 * f;
          double wave = 0;
          if (labels[b] == 0) wave = std::sin(2 * pi * 1.5 * t + local_phase);
          else if (labels[b] == 1) wave = std::sin(2 * pi * 3.5 * t + local_phase);
          else if (labels[b] == 2) wave = std::sin(2 * pi * (1.5 * t + 2.0 * t * t) + local_phase);
          else wave = std::sin(2 * pi * 2.5 * t + local_phase) * (t < 0.5 ? 0.4 : 1.6);
          x[b][c][h][f] = static_cast<float>(offset + gain * (1 + 0.05 * c + 0.03 * f) * wave + 0.03 * normal(signal_rng));
          observed[b][c][h][f] = uniform(mask_rng) >= missing;
        }
      }
    // Guarantee trainable histories independently of the assigned label.
    if (!mask[b].any().item<bool>()) observed[b][0][0][0] = true;
  }
  SyntheticEvaluationData result;
  result.clean = {clean, torch::ones_like(mask)};
  result.observed = {torch::where(mask, clean, torch::zeros_like(clean)), mask};
  result.labels = torch::tensor(labels, torch::kInt64);
  return result;
}

RepresentationDiagnostics representation_diagnostics(const torch::Tensor &values,
                                                       const torch::Tensor &valid) {
  require(values.defined() && valid.defined() && (values.dim() == 2 || values.dim() == 3), "diagnostics expect [B,D] or [B,C,D]");
  require(values.size(-1) > 0, "diagnostics require a positive embedding dimension");
  require(values.dim() == valid.dim() + 1 && values.numel() / values.size(-1) == valid.numel(), "diagnostic validity shape mismatch");
  for (int64_t i = 0; i < valid.dim(); ++i) require(values.size(i) == valid.size(i), "diagnostic validity shape mismatch");
  const auto cpu = values.detach().to(torch::kCPU).to(torch::kFloat64).reshape({-1, values.size(-1)});
  const auto rows = valid.to(torch::kCPU).to(torch::kBool).reshape({-1}).nonzero().reshape({-1});
  RepresentationDiagnostics out;
  out.valid_rows = rows.numel(); out.dimensions = values.size(-1);
  out.valid_fraction = valid.numel() ? static_cast<double>(rows.numel()) / valid.numel() : 0;
  out.per_dimension_std = torch::zeros({out.dimensions}, torch::kFloat64);
  if (!out.valid_rows) return out;
  const auto z = cpu.index_select(0, rows);
  require(torch::isfinite(z).all().item<bool>(), "valid embeddings contain non-finite values");
  const auto centered = z - z.mean(0);
  out.per_dimension_std = centered.pow(2).mean(0).sqrt();
  out.std_mean = out.per_dimension_std.mean().item<double>();
  out.std_min = out.per_dimension_std.min().item<double>(); out.std_max = out.per_dimension_std.max().item<double>();
  const auto norms = z.pow(2).sum(1).sqrt();
  out.norm_mean = norms.mean().item<double>();
  out.norm_min = norms.min().item<double>(); out.norm_max = norms.max().item<double>();
  if (out.valid_rows >= 2) {
    const auto covariance = centered.transpose(0, 1).matmul(centered) / (out.valid_rows - 1);
    const auto spectrum = at::linalg_eigvalsh(covariance).clamp_min(0);
    const double total = spectrum.sum().item<double>();
    if (total > 1e-20) {
      const auto p = spectrum / total;
      out.covariance_effective_rank = torch::exp(-(p * p.clamp_min(1e-30).log()).sum()).item<double>();
    }
  }
  return out;
}

} // namespace embedding
