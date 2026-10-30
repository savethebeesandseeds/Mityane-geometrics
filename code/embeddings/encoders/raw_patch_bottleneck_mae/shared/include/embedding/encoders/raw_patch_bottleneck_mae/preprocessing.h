// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/types.h"
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct FrozenScaler {
  torch::Tensor mean, scale, count, channel_ids; // CPU [C,F] double/double/int64; IDs [C]
  double scale_floor{1e-6};
  torch::Tensor floor_applied; // CPU bool [C,F]; empirical std was below the configured floor

  void validate(const Config &c) const {
    validate_config(c);
    TORCH_CHECK(mean.defined() && scale.defined() && count.defined() &&
                    mean.sizes() == torch::IntArrayRef({c.channel_count, c.input_width}) &&
                    scale.sizes() == mean.sizes() && count.sizes() == mean.sizes() &&
                    mean.scalar_type() == torch::kFloat64 && scale.scalar_type() == torch::kFloat64 &&
                    count.scalar_type() == torch::kInt64 && mean.device().is_cpu() &&
                    scale.device().is_cpu() && count.device().is_cpu() &&
                    torch::isfinite(mean).all().item<bool>() &&
                    torch::isfinite(scale).all().item<bool>() && scale.gt(0).all().item<bool>() &&
                    count.gt(0).all().item<bool>() && std::isfinite(scale_floor) && scale_floor > 0,
                "[rpb-mae] malformed or unsupported frozen scaler coordinates");
    const auto expected = torch::tensor(resolved_channel_ids(c), torch::kInt64);
    TORCH_CHECK(channel_ids.defined() && channel_ids.scalar_type() == torch::kInt64 &&
                    channel_ids.device().is_cpu() && torch::equal(channel_ids, expected) &&
                    scale_floor == c.scale_floor && scale.ge(scale_floor).all().item<bool>() &&
                    floor_applied.defined() && floor_applied.scalar_type() == torch::kBool &&
                    floor_applied.device().is_cpu() && floor_applied.sizes() == mean.sizes() &&
                    !floor_applied.logical_and(scale.ne(scale_floor)).any().item<bool>(),
                "[rpb-mae] scaler schema/floor mismatch");
  }

  Input transform(const Input &input, const Config &c) const {
    validate(c);
    validate_input(input, c);
    auto indices = channel_indices(input.channel_ids, c, input.data.size(0), torch::kCPU);
    auto mu = mean.index_select(0, indices.reshape({-1}))
                  .reshape({input.data.size(0), c.channel_count, 1, c.input_width});
    auto sigma = scale.index_select(0, indices.reshape({-1}))
                     .reshape({input.data.size(0), c.channel_count, 1, c.input_width});
    const auto observed = input.observed.to(torch::kCPU);
    const auto raw = input.data.to(torch::kCPU, torch::kFloat64);
    const auto safe = torch::where(observed, raw, torch::zeros_like(raw));
    auto scaled = torch::where(observed, (safe - mu) / sigma, torch::zeros_like(raw));
    scaled = scaled.to(c.device, torch::kFloat32);
    TORCH_CHECK(torch::isfinite(scaled).all().item<bool>(),
                "[rpb-mae] nonfinite/overflow after centering and scaling");
    return {scaled, observed.to(c.device), input.channel_ids.to(c.device),
            input.endpoints.defined() ? input.endpoints.to(c.device) : torch::Tensor(),
            input.sampling_interval};
  }

  void save(torch::serialize::OutputArchive &archive) const {
    archive.write("mean", mean, true); archive.write("scale", scale, true);
    archive.write("count", count, true); archive.write("channel_ids", channel_ids, true);
    archive.write("floor_applied", floor_applied, true);
    archive.write("scale_floor", torch::tensor(scale_floor, torch::kFloat64), true);
  }

  static FrozenScaler load(torch::serialize::InputArchive &archive) {
    FrozenScaler out; torch::Tensor floor;
    archive.read("mean", out.mean, true); archive.read("scale", out.scale, true);
    archive.read("count", out.count, true); archive.read("channel_ids", out.channel_ids, true);
    archive.read("floor_applied", out.floor_applied, true);
    archive.read("scale_floor", floor, true);
    TORCH_CHECK(floor.numel() == 1, "[rpb-mae] malformed scaler floor");
    out.mean = out.mean.to(torch::kCPU); out.scale = out.scale.to(torch::kCPU);
    out.count = out.count.to(torch::kCPU); out.channel_ids = out.channel_ids.to(torch::kCPU);
    out.floor_applied = out.floor_applied.to(torch::kCPU);
    out.scale_floor = floor.item<double>();
    return out;
  }

  // Stable content fingerprint, not a cryptographic integrity check.
  std::string identity() const {
    uint64_t hash = 14695981039346656037ULL;
    const auto add = [&](const unsigned char *bytes, size_t n) {
      for (size_t i = 0; i < n; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
    };
    for (const auto &value : {mean, scale, count, channel_ids, floor_applied}) {
      TORCH_CHECK(value.defined(), "[rpb-mae] identity requires fitted scaler");
      const auto cpu = value.to(torch::kCPU).contiguous();
      const auto type = static_cast<int64_t>(cpu.scalar_type());
      add(reinterpret_cast<const unsigned char *>(&type), sizeof(type));
      for (const auto dim : cpu.sizes()) add(reinterpret_cast<const unsigned char *>(&dim), sizeof(dim));
      add(reinterpret_cast<const unsigned char *>(cpu.data_ptr()), cpu.numel() * cpu.element_size());
    }
    add(reinterpret_cast<const unsigned char *>(&scale_floor), sizeof(scale_floor));
    std::ostringstream out; out << "rpb-scaler-v1-fnv1a64-" << std::hex << std::setw(16)
                                << std::setfill('0') << hash;
    return out.str();
  }
};

inline FrozenScaler fit_scaler(const Input &input, const Config &c) {
  torch::NoGradGuard no_grad;
  validate_input(input, c);
  auto raw = input.data.to(torch::kCPU, torch::kFloat64);
  auto observed = input.observed.to(torch::kCPU);
  auto indices = channel_indices(input.channel_ids, c, input.data.size(0), torch::kCPU);
  auto safe = torch::where(observed, raw, torch::zeros_like(raw));
  std::vector<torch::Tensor> means, scales, counts, floors;
  for (int64_t id = 0; id < c.channel_count; ++id) {
    auto support = observed.logical_and(indices.eq(id).unsqueeze(-1).unsqueeze(-1));
    auto count = support.sum(std::vector<int64_t>{0, 1, 2});
    TORCH_CHECK(count.gt(0).all().item<bool>(), "[rpb-mae] scaler coordinate has no fitted observations");
    auto values = torch::where(support, safe, torch::zeros_like(safe));
    auto mean = values.sum(std::vector<int64_t>{0, 1, 2}) / count.to(torch::kFloat64);
    auto centered = torch::where(support, safe - mean, torch::zeros_like(safe));
    auto variance = centered.square().sum(std::vector<int64_t>{0, 1, 2}) / count.to(torch::kFloat64);
    auto deviation = variance.sqrt();
    means.push_back(mean); scales.push_back(deviation.clamp_min(c.scale_floor));
    floors.push_back(deviation.lt(c.scale_floor));
    counts.push_back(count);
  }
  FrozenScaler out{torch::stack(means), torch::stack(scales), torch::stack(counts),
                   torch::tensor(resolved_channel_ids(c), torch::kInt64), c.scale_floor, torch::stack(floors)};
  out.validate(c);
  return out;
}

// Crops each channel's own bars at or before the endpoint, then right-aligns the
// last history_length of them. A short channel is left-padded. The endpoint need
// not land on a sample, and a channel's step need not equal sampling_interval.
// Timestamps are [B,C,T]. A finite time is a present slot. A NaN is a leading pad.
inline Input prepare_endpoint_history(const torch::Tensor &values,
                                      const torch::Tensor &observed,
                                      const torch::Tensor &ids,
                                      const torch::Tensor &timestamps,
                                      const torch::Tensor &endpoints,
                                      const Config &c) {
  validate_config(c);
  TORCH_CHECK(values.dim() == 4 && values.size(0) > 0 && values.size(1) == c.channel_count &&
                  values.size(3) == c.input_width && observed.sizes() == values.sizes() &&
                  observed.scalar_type() == torch::kBool,
              "[rpb-mae] raw endpoint preparation requires BCTF values/bool support");
  const auto B = values.size(0), C = values.size(1), T = values.size(2), H = c.history_length;
  const auto F = values.size(3);
  TORCH_CHECK(timestamps.is_floating_point() && timestamps.dim() == 3 && timestamps.size(0) == B &&
                  timestamps.size(1) == C && timestamps.size(2) == T && endpoints.is_floating_point() &&
                  endpoints.dim() == 1 && endpoints.size(0) == B && torch::isfinite(endpoints).all().item<bool>(),
              "[rpb-mae] preparation requires per-channel times [B,C,T] and finite endpoints [B]");
  auto times = timestamps.to(torch::kCPU, torch::kFloat64).contiguous();
  auto ends = endpoints.to(torch::kCPU, torch::kFloat64).contiguous();
  auto stored = values.to(torch::kCPU, torch::kFloat64);
  auto support = observed.to(torch::kCPU);
  const auto time_at = times.accessor<double, 3>();
  auto out_data = torch::zeros({B, C, H, F}, stored.options());
  auto out_mask = torch::zeros({B, C, H, F}, torch::kBool);
  for (int64_t b = 0; b < B; ++b) {
    const double endpoint = ends[b].item<double>();
    for (int64_t channel = 0; channel < C; ++channel) {
      std::vector<int64_t> kept;
      bool seen = false;
      double previous = 0;
      for (int64_t slot = 0; slot < T; ++slot) {
        const double stamp = time_at[b][channel][slot];
        if (!std::isfinite(stamp)) {
          TORCH_CHECK(std::isnan(stamp), "[rpb-mae] a timestamp must be finite or NaN");
          TORCH_CHECK(!support[b][channel][slot].any().item<bool>(),
                      "[rpb-mae] an observed feature requires a finite time");
          TORCH_CHECK(!seen, "[rpb-mae] a NaN time after a finite time is a hole");
          continue;
        }
        TORCH_CHECK(!seen || stamp > previous, "[rpb-mae] present times must strictly increase");
        seen = true;
        previous = stamp;
        if (stamp <= endpoint) kept.push_back(slot);
      }
      if (kept.size() > static_cast<std::size_t>(H))
        kept.erase(kept.begin(), kept.end() - static_cast<std::ptrdiff_t>(H));
      if (kept.empty()) continue;
      const auto pad = H - static_cast<int64_t>(kept.size());
      const auto index = torch::tensor(kept, torch::kLong);
      const auto taken_values = stored[b][channel].index_select(0, index);
      const auto taken_mask = support[b][channel].index_select(0, index);
      const auto safe = torch::where(taken_mask, taken_values, torch::zeros_like(taken_values));
      out_data[b][channel].narrow(0, pad, static_cast<int64_t>(kept.size())).copy_(safe);
      out_mask[b][channel].narrow(0, pad, static_cast<int64_t>(kept.size())).copy_(taken_mask);
    }
  }
  Input out{out_data, out_mask, ids, ends, c.sampling_interval};
  validate_input(out, c);
  return out;
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
