// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/types.h"
#include <array>
#include <cmath>
#include <set>

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr int64_t kTemporalDynamicsDimensions = 18;
inline constexpr int64_t kTemporalDynamicsMinimumTimes = 4;
inline constexpr double kTemporalDynamicsEnergyFloor = 1e-12;
inline constexpr double kTemporalDynamicsScaleFloor = 1e-6;
inline constexpr const char *kTemporalDynamicsRecipe =
    "all-semantic-pairs-k124-common-four-endpoint-centered-correlation-even-odd-v1";

struct TemporalDynamicsTargets {
  torch::Tensor values; // detached F64 [B,18]; invalid dimensions exactly zero
  torch::Tensor valid; // bool [B,18]
  torch::Tensor supported_feature_pairs; // int64 [B,18], 0..9
};

inline void validate_temporal_dynamics_targets(const TemporalDynamicsTargets &targets) {
  TORCH_CHECK(targets.values.defined() && targets.values.dim() == 2 && targets.values.size(0) > 0 &&
                  targets.values.size(1) == kTemporalDynamicsDimensions &&
                  targets.values.scalar_type() == torch::kFloat64 &&
                  targets.valid.defined() && targets.valid.scalar_type() == torch::kBool &&
                  targets.valid.sizes() == targets.values.sizes() &&
                  targets.valid.device() == targets.values.device() &&
                  targets.supported_feature_pairs.defined() &&
                  targets.supported_feature_pairs.scalar_type() == torch::kInt64 &&
                  targets.supported_feature_pairs.sizes() == targets.values.sizes() &&
                  targets.supported_feature_pairs.device() == targets.values.device(),
              "[rpb-dynamics] exact values/valid/feature-pair target geometry required");
  TORCH_CHECK(targets.supported_feature_pairs.ge(0).all().item<bool>() &&
                  targets.supported_feature_pairs.le(9).all().item<bool>() &&
                  torch::equal(targets.valid, targets.supported_feature_pairs.gt(0)) &&
                  torch::isfinite(targets.values.masked_select(targets.valid)).all().item<bool>() &&
                  targets.values.masked_select(targets.valid.logical_not()).eq(0).all().item<bool>(),
              "[rpb-dynamics] target support and finite legal values required");
}

// Original observed TRAIN only. Neither labels nor a reconstruction/context
// mask is an argument. The producer must index these detached targets using
// the unchanged sampled rows after fitting the map once on the original TRAIN.
inline TemporalDynamicsTargets temporal_dynamics_targets(const Input &original, const Config &config) {
  validate_input(original, config, false);
  TORCH_CHECK(config.channel_count == 3 && config.history_length == 32 && config.input_width == 3 &&
                  config.export_width == 32,
              "[rpb-dynamics] temporal objective requires C3/H32/F3/native32");
  torch::NoGradGuard no_grad;
  const auto mask = original.observed.to(original.data.device());
  // Mask before arithmetic, including centering: hidden NaN is never read.
  const auto safe = torch::where(mask, original.data.detach(), torch::zeros_like(original.data)).to(torch::kFloat64);
  const auto B = safe.size(0), H = safe.size(2);
  const auto order = channel_indices(original.channel_ids, config, B, safe.device()).argsort(int64_t{1});
  const auto gather = order.unsqueeze(-1).unsqueeze(-1).expand({B, 3, H, 3});
  const auto x = safe.gather(1, gather), observed = mask.gather(1, gather);
  std::vector<torch::Tensor> values, validity, feature_counts;
  const std::array<std::array<int64_t, 2>, 3> pairs{{{{0,1}}, {{0,2}}, {{1,2}}}};
  for (const auto &pair : pairs) for (const int64_t k : {int64_t{1}, int64_t{2}, int64_t{4}}) {
    auto even_sum = torch::zeros({B}, x.options()), odd_sum = torch::zeros_like(even_sum);
    auto supported = torch::zeros({B}, observed.options().dtype(torch::kInt64));
    for (int64_t f = 0; f < 3; ++f) for (int64_t g = 0; g < 3; ++g) {
      const auto a0 = x.select(1, pair[0]).select(2, f).narrow(1, 0, H-k);
      const auto a1 = x.select(1, pair[0]).select(2, f).narrow(1, k, H-k);
      const auto b0 = x.select(1, pair[1]).select(2, g).narrow(1, 0, H-k);
      const auto b1 = x.select(1, pair[1]).select(2, g).narrow(1, k, H-k);
      const auto am = observed.select(1, pair[0]).select(2, f), bm = observed.select(1, pair[1]).select(2, g);
      const auto common = am.narrow(1, 0, H-k).logical_and(am.narrow(1, k, H-k))
          .logical_and(bm.narrow(1, 0, H-k)).logical_and(bm.narrow(1, k, H-k));
      const auto count = common.sum(1), denominator = count.clamp_min(1).to(torch::kFloat64);
      auto centered = [&](const torch::Tensor &vector) {
        const auto legal = torch::where(common, vector, torch::zeros_like(vector));
        const auto mean = legal.sum(1) / denominator;
        return torch::where(common, legal - mean.unsqueeze(1), torch::zeros_like(legal));
      };
      const auto ca0 = centered(a0), ca1 = centered(a1), cb0 = centered(b0), cb1 = centered(b1);
      const auto va0 = ca0.square().sum(1)/denominator, va1 = ca1.square().sum(1)/denominator;
      const auto vb0 = cb0.square().sum(1)/denominator, vb1 = cb1.square().sum(1)/denominator;
      const auto cp = (ca0*cb1).sum(1)/denominator, cm = (ca1*cb0).sum(1)/denominator;
      TORCH_CHECK(torch::isfinite(va0).all().item<bool>() && torch::isfinite(va1).all().item<bool>() &&
                      torch::isfinite(vb0).all().item<bool>() && torch::isfinite(vb1).all().item<bool>() &&
                      torch::isfinite(cp).all().item<bool>() && torch::isfinite(cm).all().item<bool>(),
                  "[rpb-dynamics] legal centered-moment overflow");
      const auto eligible = count.ge(kTemporalDynamicsMinimumTimes)
          .logical_and(va0.gt(kTemporalDynamicsEnergyFloor)).logical_and(va1.gt(kTemporalDynamicsEnergyFloor))
          .logical_and(vb0.gt(kTemporalDynamicsEnergyFloor)).logical_and(vb1.gt(kTemporalDynamicsEnergyFloor));
      const auto plus_scale = torch::where(eligible, va0.sqrt()*vb1.sqrt(), torch::ones_like(va0));
      const auto minus_scale = torch::where(eligible, va1.sqrt()*vb0.sqrt(), torch::ones_like(va1));
      const auto plus = (cp/plus_scale).clamp(-1., 1.), minus = (cm/minus_scale).clamp(-1., 1.);
      even_sum = even_sum + torch::where(eligible, .5*(plus+minus), torch::zeros_like(plus));
      odd_sum = odd_sum + torch::where(eligible, .5*(plus-minus), torch::zeros_like(plus));
      supported = supported + eligible.to(torch::kInt64);
    }
    const auto valid = supported.gt(0), divisor = supported.clamp_min(1).to(torch::kFloat64);
    values.push_back(torch::where(valid, even_sum/divisor, torch::zeros_like(even_sum)));
    values.push_back(torch::where(valid, odd_sum/divisor, torch::zeros_like(odd_sum)));
    validity.push_back(valid); validity.push_back(valid);
    feature_counts.push_back(supported); feature_counts.push_back(supported);
  }
  TemporalDynamicsTargets result{torch::stack(values, 1).detach(), torch::stack(validity, 1), torch::stack(feature_counts, 1)};
  validate_temporal_dynamics_targets(result);
  return result;
}

struct TemporalDynamicsScaler {
  torch::Tensor mean, scale, count; // frozen CPU F64/F64/int64 [18]

  void validate() const {
    TORCH_CHECK(mean.defined() && scale.defined() && count.defined() &&
                    mean.device().is_cpu() && scale.device().is_cpu() && count.device().is_cpu() &&
                    mean.scalar_type() == torch::kFloat64 && scale.scalar_type() == torch::kFloat64 &&
                    count.scalar_type() == torch::kInt64 && mean.sizes() == torch::IntArrayRef({18}) &&
                    scale.sizes() == mean.sizes() && count.sizes() == mean.sizes() &&
                    torch::isfinite(mean).all().item<bool>() && torch::isfinite(scale).all().item<bool>() &&
                    scale.ge(kTemporalDynamicsScaleFloor).all().item<bool>() && count.ge(0).all().item<bool>() &&
                    torch::where(count.eq(0), mean.eq(0).logical_and(scale.eq(1)), torch::ones_like(count, torch::kBool)).all().item<bool>(),
                "[rpb-dynamics] exact finite frozen TRAIN map with zero-count defaults required");
  }

  TemporalDynamicsTargets standardize(const TemporalDynamicsTargets &targets) const {
    validate(); validate_temporal_dynamics_targets(targets);
    const auto fitted = count.gt(0).to(targets.values.device());
    TORCH_CHECK(!targets.valid.logical_and(fitted.logical_not().unsqueeze(0)).any().item<bool>(),
                "[rpb-dynamics] a supported target dimension was absent from frozen TRAIN");
    const auto safe = torch::where(targets.valid, targets.values.detach(), torch::zeros_like(targets.values));
    const auto normalized = (safe - mean.to(safe.device()).unsqueeze(0))/scale.to(safe.device()).unsqueeze(0);
    return {torch::where(targets.valid, normalized, torch::zeros_like(normalized)).detach(),
            targets.valid, targets.supported_feature_pairs};
  }

  void save(torch::serialize::OutputArchive &archive) const {
    validate();
    archive.write("format_version", torch::tensor(int64_t{1}, torch::kInt64), true);
    archive.write("scale_floor", torch::tensor(kTemporalDynamicsScaleFloor, torch::kFloat64), true);
    archive.write("mean", mean.detach().contiguous(), true);
    archive.write("scale", scale.detach().contiguous(), true);
    archive.write("count", count.detach().contiguous(), true);
  }

  void load(torch::serialize::InputArchive &archive) {
    const auto keys = archive.keys();
    TORCH_CHECK(std::set<std::string>(keys.begin(), keys.end()) ==
                    std::set<std::string>({"format_version", "scale_floor", "mean", "scale", "count"}),
                "[rpb-dynamics] closed frozen TRAIN map keys required");
    torch::Tensor version, floor; TemporalDynamicsScaler loaded;
    archive.read("format_version", version, true); archive.read("scale_floor", floor, true);
    archive.read("mean", loaded.mean, true); archive.read("scale", loaded.scale, true); archive.read("count", loaded.count, true);
    TORCH_CHECK(version.device().is_cpu() && version.scalar_type() == torch::kInt64 && version.dim() == 0 &&
                    version.item<int64_t>() == 1 && floor.device().is_cpu() && floor.scalar_type() == torch::kFloat64 &&
                    floor.dim() == 0 && floor.item<double>() == kTemporalDynamicsScaleFloor,
                "[rpb-dynamics] frozen TRAIN map version/floor mismatch");
    loaded.validate(); *this = std::move(loaded);
  }
};

inline TemporalDynamicsScaler fit_temporal_dynamics_scaler(const TemporalDynamicsTargets &training) {
  validate_temporal_dynamics_targets(training); torch::NoGradGuard no_grad;
  const auto values = torch::where(training.valid, training.values.detach(), torch::zeros_like(training.values));
  const auto count = training.valid.sum(0), denominator = count.clamp_min(1).to(torch::kFloat64);
  const auto mean = values.sum(0)/denominator;
  const auto centered = torch::where(training.valid, values - mean.unsqueeze(0), torch::zeros_like(values));
  const auto scale = (centered.square().sum(0)/denominator).sqrt().clamp_min(kTemporalDynamicsScaleFloor);
  TemporalDynamicsScaler result{torch::where(count.gt(0), mean, torch::zeros_like(mean)).to(torch::kCPU).contiguous(),
      torch::where(count.gt(0), scale, torch::ones_like(scale)).to(torch::kCPU).contiguous(), count.to(torch::kCPU).contiguous()};
  result.validate(); return result;
}

// Training-only. No Linear constructor/reset and no generator: constructing
// this module after the backbone never advances common CPU/CUDA RNG streams.
struct TemporalDynamicsDecoderImpl : torch::nn::Module {
  explicit TemporalDynamicsDecoderImpl(torch::Device device = torch::kCPU) {
    const auto rows = torch::arange(1, 19, torch::kInt64).unsqueeze(1);
    const auto columns = torch::arange(1, 33, torch::kInt64).unsqueeze(0);
    const auto numerator = torch::remainder(rows*columns, int64_t{31})*2 - 31;
    weight = register_parameter("weight", numerator.to(torch::kFloat32)/(31.*std::sqrt(32.)), true);
    to(device, torch::kFloat32);
  }
  torch::Tensor forward(const torch::Tensor &native32) {
    TORCH_CHECK(native32.defined() && native32.dim() == 2 && native32.size(0) > 0 && native32.size(1) == 32 &&
                    native32.scalar_type() == torch::kFloat32 && native32.device() == weight.device() &&
                    torch::isfinite(native32).all().item<bool>(),
                "[rpb-dynamics] the exact served finite native32 is the sole objective decoder input");
    return torch::matmul(native32, weight.transpose(0, 1));
  }
  torch::Tensor weight;
};
TORCH_MODULE(TemporalDynamicsDecoder);

struct TemporalDynamicsLoss {
  torch::Tensor loss, valid_dimensions, eligible_examples;
  int64_t total_valid_dimensions{0}, eligible_example_count{0};
};

inline TemporalDynamicsLoss temporal_dynamics_huber(const torch::Tensor &prediction,
    const torch::Tensor &standardized_target, const torch::Tensor &valid, double delta = 1.0) {
  TORCH_CHECK(delta == 1.0 && prediction.defined() && prediction.dim() == 2 && prediction.size(0) > 0 && prediction.size(1) == 18 &&
                  prediction.scalar_type() == torch::kFloat32 && standardized_target.defined() &&
                  standardized_target.scalar_type() == torch::kFloat32 && standardized_target.sizes() == prediction.sizes() &&
                  standardized_target.device() == prediction.device() && valid.defined() && valid.scalar_type() == torch::kBool &&
                  valid.sizes() == prediction.sizes() && valid.device() == prediction.device() &&
                  torch::isfinite(prediction.masked_select(valid)).all().item<bool>() &&
                  torch::isfinite(standardized_target.masked_select(valid)).all().item<bool>(),
              "[rpb-dynamics] exact finite original target/support objective geometry required");
  const auto counts = valid.sum(1), examples = counts.gt(0);
  TemporalDynamicsLoss output; output.valid_dimensions = counts; output.eligible_examples = examples;
  output.total_valid_dimensions = counts.sum().item<int64_t>(); output.eligible_example_count = examples.sum().item<int64_t>();
  const auto safe_prediction = torch::where(valid, prediction, torch::zeros_like(prediction));
  if (output.eligible_example_count == 0) { output.loss = safe_prediction.sum()*0.; return output; }
  const auto safe_target = torch::where(valid, standardized_target.detach(), torch::zeros_like(standardized_target));
  const auto error = safe_prediction - safe_target, absolute = error.abs();
  const auto cells = torch::where(absolute.le(1.), .5*error.clamp(-1., 1.).square(), absolute-.5);
  const auto per_example = cells.sum(1)/counts.clamp_min(1).to(prediction.dtype());
  output.loss = per_example.masked_select(examples).mean();
  TORCH_CHECK(torch::isfinite(output.loss).item<bool>(), "[rpb-dynamics] nonfinite normalized objective loss");
  return output;
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
