// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/types.h"
#include <iomanip>
#include <sstream>

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

// Crops raw observations before model/scaler use; future samples cannot influence the result.
// In this uniform schema the endpoint denotes the last included timestamp, not a
// cutoff between samples. Reject unaligned cutoffs instead of misreporting support.
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
  const auto B = values.size(0), T = values.size(2);
  TORCH_CHECK(timestamps.is_floating_point() &&
                  ((timestamps.dim() == 1 && timestamps.size(0) == T) ||
                   (timestamps.dim() == 2 && timestamps.size(0) == B && timestamps.size(1) == T)) &&
                  endpoints.is_floating_point() && endpoints.dim() == 1 && endpoints.size(0) == B &&
                  torch::isfinite(timestamps).all().item<bool>() &&
                  torch::isfinite(endpoints).all().item<bool>(),
              "[rpb-mae] preparation requires finite times [T]/[B,T] and endpoints [B]");
  auto times = (timestamps.dim() == 1 ? timestamps.unsqueeze(0).expand({B, T}) : timestamps)
                   .to(torch::kCPU, torch::kFloat64);
  auto ends = endpoints.to(torch::kCPU, torch::kFloat64);
  const double sampling_tolerance = c.sampling_interval * 1e-9;
  std::vector<torch::Tensor> histories, supports;
  for (int64_t b = 0; b < B; ++b) {
    auto row = times.select(0, b);
    auto differences = row.slice(0, 1) - row.slice(0, 0, T - 1);
    TORCH_CHECK(differences.gt(0).all().item<bool>() &&
                    torch::allclose(differences, torch::full_like(differences, c.sampling_interval), 1e-9, sampling_tolerance),
                "[rpb-mae] timestamp support must be ordered and uniformly sampled");
    const auto available = row.le(ends[b]).sum().item<int64_t>();
    TORCH_CHECK(available >= c.history_length, "[rpb-mae] endpoint has insufficient history");
    TORCH_CHECK(std::abs(row[available - 1].item<double>() - ends[b].item<double>()) <=
                    2.0 * sampling_tolerance,
                "[rpb-mae] endpoint must coincide with the last included uniform timestamp");
    const auto begin = available - c.history_length;
    histories.push_back(values.select(0, b).slice(1, begin, available));
    supports.push_back(observed.select(0, b).slice(1, begin, available));
  }
  Input out{torch::stack(histories), torch::stack(supports), ids, ends, c.sampling_interval};
  validate_input(out, c);
  return out;
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
