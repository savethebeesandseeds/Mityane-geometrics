// SPDX-License-Identifier: MIT
#include "observed_information.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace embedding::evaluation::two_component {
namespace {
constexpr int history = 32;
constexpr int dimensions = 5;
constexpr double tau = 6.283185307179586476925286766559;
using Vector = std::array<double, dimensions>;
using Matrix = std::array<Vector, dimensions>;
struct Wave {
  std::array<double, history> sine{}, cosine{};
};
struct Series {
  int channel = 0;
  int feature = 0;
  std::array<bool, history> legal{};
  std::array<double, history> centered{};
  double energy = 0;
  double mean = 0;
};
struct Fit {
  Vector coefficients{}; // constant, slow sine/cosine, fast sine/cosine.
  double residual = 0;
};
struct Candidate {
  bool valid = false;
  double slow = 0, fast = 0, score = 0;
  int64_t evaluations = 0;
  std::vector<Fit> fits;
};

Wave wave(double frequency) {
  Wave out;
  for (int t = 0; t < history; ++t) {
    out.sine[t] = std::sin(tau * frequency * t);
    out.cosine[t] = std::cos(tau * frequency * t);
  }
  return out;
}

// Diagonal scaling keeps the pivot rule independent of the number of legal
// cells and basis column magnitudes. Never add a ridge or silently lower rank.
bool solve(const Matrix &gram, const Vector &rhs, Vector &coefficients) {
  Matrix lower{};
  Vector diagonal{}, scaled_rhs{}, temporary{}, solution{};
  for (int i = 0; i < dimensions; ++i) {
    if (!(gram[i][i] > 0) || !std::isfinite(gram[i][i])) return false;
    diagonal[i] = std::sqrt(gram[i][i]);
    scaled_rhs[i] = rhs[i] / diagonal[i];
    if (!std::isfinite(scaled_rhs[i])) return false;
  }
  for (int i = 0; i < dimensions; ++i) {
    for (int j = 0; j <= i; ++j) {
      double value = gram[i][j] / diagonal[i] / diagonal[j];
      for (int k = 0; k < j; ++k) value -= lower[i][k] * lower[j][k];
      if (!std::isfinite(value)) return false;
      if (i == j) {
        if (value < kInformationScaledPivotFloor) return false;
        lower[i][j] = std::sqrt(value);
      } else {
        lower[i][j] = value / lower[j][j];
      }
    }
  }
  for (int i = 0; i < dimensions; ++i) {
    double value = scaled_rhs[i];
    for (int j = 0; j < i; ++j) value -= lower[i][j] * temporary[j];
    temporary[i] = value / lower[i][i];
  }
  for (int i = dimensions - 1; i >= 0; --i) {
    double value = temporary[i];
    for (int j = i + 1; j < dimensions; ++j) value -= lower[j][i] * solution[j];
    solution[i] = value / lower[i][i];
    coefficients[i] = solution[i] / diagonal[i];
    if (!std::isfinite(coefficients[i])) return false;
  }
  return true;
}

bool fit(const Series &series, const std::array<Vector, history> &basis, Fit &out) {
  Matrix gram{};
  Vector rhs{};
  for (int t = 0; t < history; ++t) {
    if (!series.legal[t]) continue;
    for (int i = 0; i < dimensions; ++i) {
      rhs[i] += basis[t][i] * series.centered[t];
      for (int j = 0; j <= i; ++j) gram[i][j] += basis[t][i] * basis[t][j];
    }
  }
  for (int i = 0; i < dimensions; ++i)
    for (int j = 0; j < i; ++j) gram[j][i] = gram[i][j];
  if (!solve(gram, rhs, out.coefficients)) return false;
  // Direct residual evaluation avoids cancellation in y'y - beta'X'y near an
  // exact noiseless fit. Per-series normalization gives positive gains no vote.
  long double squared_error = 0;
  for (int t = 0; t < history; ++t) {
    if (!series.legal[t]) continue;
    double prediction = 0;
    for (int i = 0; i < dimensions; ++i) prediction += basis[t][i] * out.coefficients[i];
    const long double error = series.centered[t] - prediction;
    squared_error += error * error;
  }
  out.residual = static_cast<double>(squared_error / series.energy);
  const double slow_energy = out.coefficients[1] * out.coefficients[1] +
                             out.coefficients[2] * out.coefficients[2];
  const double fast_energy = out.coefficients[3] * out.coefficients[3] +
                             out.coefficients[4] * out.coefficients[4];
  const double total = slow_energy + fast_energy;
  return std::isfinite(out.residual) && std::isfinite(total) && total > kInformationEnergyFloor &&
      slow_energy / total >= kInformationMinimumComponentEnergyFraction &&
      fast_energy / total >= kInformationMinimumComponentEnergyFraction;
}

// Features have independently fitted coefficients, but the declared positive
// whole-mixture gains preserve each component's phase within a channel. This
// observed coherence rule also excludes cross-feature sampling aliases.
bool phases(const std::vector<Series> &series, const std::vector<Fit> &fits,
            std::array<std::array<double, 2>, 2> &slow_direction) {
  for (int channel = 0; channel < 2; ++channel) {
    for (int component = 0; component < 2; ++component) {
      const int first = 1 + 2 * component;
      double cosine = 0, sine = 0;
      int count = 0;
      for (size_t i = 0; i < series.size(); ++i) {
        if (series[i].channel != channel) continue;
        const auto &coefficients = fits[i].coefficients;
        const double amplitude = std::hypot(coefficients[first], coefficients[first + 1]);
        if (!(amplitude > 0) || !std::isfinite(amplitude)) return false;
        cosine += coefficients[first] / amplitude;
        sine += coefficients[first + 1] / amplitude;
        ++count;
      }
      const double length = std::hypot(cosine, sine);
      if (count < kInformationMinimumFeaturesPerChannel ||
          !std::isfinite(length) || length / count < kInformationMinimumPhaseCoherence) return false;
      if (component == 0) slow_direction[channel] = {cosine / length, sine / length};
    }
  }
  return true;
}

Candidate evaluate(double slow, double fast, const Wave &slow_wave, const Wave &fast_wave,
                   const std::vector<Series> &series) {
  Candidate out;
  out.slow = slow;
  out.fast = fast;
  std::array<Vector, history> basis{};
  for (int t = 0; t < history; ++t)
    basis[t] = {1, slow_wave.sine[t], slow_wave.cosine[t], fast_wave.sine[t], fast_wave.cosine[t]};
  out.fits.resize(series.size());
  for (size_t i = 0; i < series.size(); ++i) {
    if (!fit(series[i], basis, out.fits[i])) return out;
    out.score += out.fits[i].residual;
  }
  std::array<std::array<double, 2>, 2> direction{};
  if (!phases(series, out.fits, direction)) return out;
  out.score /= series.size();
  out.valid = std::isfinite(out.score);
  return out;
}

bool better(const Candidate &candidate, const Candidate &best) {
  if (!candidate.valid) return false;
  if (!best.valid) return true;
  if (candidate.score != best.score) return candidate.score < best.score;
  return candidate.slow < best.slow || (candidate.slow == best.slow && candidate.fast < best.fast);
}

Candidate search(const std::vector<Series> &series,
                 const std::vector<Wave> &slow_waves, const std::vector<Wave> &fast_waves) {
  const double slow_step = (kInformationSlowFrequencyMax - kInformationSlowFrequencyMin) /
                           kInformationSlowGridIntervals;
  const double fast_step = (kInformationFastFrequencyMax - kInformationFastFrequencyMin) /
                           kInformationFastGridIntervals;
  Candidate best;
  int64_t evaluations = 0;
  for (int i = 0; i <= kInformationSlowGridIntervals; ++i) {
    const double slow = kInformationSlowFrequencyMin + i * slow_step;
    for (int j = 0; j <= kInformationFastGridIntervals; ++j) {
      const double fast = kInformationFastFrequencyMin + j * fast_step;
      const auto candidate = evaluate(slow, fast, slow_waves[i], fast_waves[j], series);
      ++evaluations;
      if (better(candidate, best)) best = candidate;
    }
  }
  if (!best.valid) { best.evaluations = evaluations; return best; }
  double ds = slow_step, df = fast_step;
  for (int round = 0; round < kInformationRefinementRounds; ++round) {
    ds *= .5;
    df *= .5;
    // All nine points are anchored to this round's winner, not a moving centre.
    const auto centre = best;
    for (int i = -1; i <= 1; ++i) {
      const double slow = std::clamp(centre.slow + i * ds,
          kInformationSlowFrequencyMin, kInformationSlowFrequencyMax);
      const auto slow_wave = wave(slow);
      for (int j = -1; j <= 1; ++j) {
        const double fast = std::clamp(centre.fast + j * df,
            kInformationFastFrequencyMin, kInformationFastFrequencyMax);
        const auto candidate = evaluate(slow, fast, slow_wave, wave(fast), series);
        ++evaluations;
        if (better(candidate, best)) best = candidate;
      }
    }
  }
  best.evaluations = evaluations;
  return best;
}

InformationDecision decide(const std::vector<Series> &series, const Candidate &best, InformationTask task) {
  InformationDecision out;
  out.frequency_pairs_evaluated = best.evaluations;
  if (!best.valid) { out.reason = "no_stable_coherent_two_band_fit"; return out; }
  out.slow_frequency = best.slow;
  out.fast_frequency = best.fast;
  out.normalized_residual = best.score;
  for (size_t i = 0; i < series.size(); ++i) {
    const auto row = series[i].channel * 3 + series[i].feature;
    out.eligible[row] = true;
    out.coefficients[row] = best.fits[i].coefficients;
    out.coefficients[row][0] += series[i].mean;
  }
  if (best.score > kInformationMaximumNormalizedResidual) {
    out.reason = "normalized_residual_above_limit";
    return out;
  }
  std::vector<double> ratios;
  double mean_ratio = 0;
  for (const auto &fit : best.fits) {
    const auto &c = fit.coefficients;
    const double ratio = std::log(std::hypot(c[1], c[2]) / std::hypot(c[3], c[4]));
    ratios.push_back(ratio);
    mean_ratio += ratio;
  }
  mean_ratio /= ratios.size();
  if (task == InformationTask::ComponentBalance) {
    for (double ratio : ratios) {
      if (!std::isfinite(ratio) || std::abs(ratio - mean_ratio) > kInformationMaximumLogRatioSpread) {
        out.reason = "inconsistent_component_balance";
        return out;
      }
    }
  }
  if (task == InformationTask::SlowLagSign) {
    std::array<std::array<double, 2>, 2> direction{};
    if (!phases(series, best.fits, direction)) { out.reason = "incoherent_component_phases"; return out; }
    const auto &a = direction[0];
    const auto &b = direction[1];
    // sin(wt+phi) = cos(phi)*sin(wt)+sin(phi)*cos(wt). The oriented
    // cross term is positive when channel1 uses t + positiveDelay.
    out.margin = std::atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
    if (std::abs(out.margin) < kInformationMinimumSlowPhaseMargin ||
        std::abs(out.margin) > kInformationMaximumSlowPhaseMargin) {
      out.reason = "slow_phase_margin_outside_support";
      return out;
    }
  } else {
    out.margin = mean_ratio;
    if (std::abs(out.margin) < kInformationMinimumLogRatioMargin ||
        std::abs(out.margin) > kInformationMaximumLogRatioMargin) {
      out.reason = "component_ratio_margin_outside_support";
      return out;
    }
  }
  if (!std::isfinite(out.margin)) { out.reason = "nonfinite_decision_margin"; return out; }
  out.predicted_label = out.margin > 0 ? 1 : 0;
  out.supported = true;
  out.reason = "supported";
  return out;
}
} // namespace

std::vector<InformationDecision> infer_information(
    const torch::Tensor &observed, const torch::Tensor &mask, InformationTask task) {
  TORCH_CHECK(task == InformationTask::SlowLagSign || task == InformationTask::ComponentBalance,
              "[two-band-information] closed task required");
  TORCH_CHECK(observed.defined() && mask.defined() && observed.device().is_cpu() && mask.device().is_cpu() &&
              (observed.scalar_type() == torch::kFloat64 || observed.scalar_type() == torch::kFloat32) &&
              mask.scalar_type() == torch::kBool && observed.dim() == 4 && observed.size(0) > 0 &&
              observed.size(1) == 3 && observed.size(2) == history && observed.size(3) == 3 &&
              mask.sizes() == observed.sizes(), "[two-band-information] CPU F64/F32 C3H32F3 and Bool mask required");
  const auto values = observed.detach().to(torch::kFloat64).contiguous();
  const auto legal = mask.contiguous();
  const auto x = values.accessor<double, 4>();
  const auto m = legal.accessor<bool, 4>();
  for (int64_t b = 0; b < values.size(0); ++b)
    for (int c = 0; c < 3; ++c)
      for (int t = 0; t < history; ++t)
        for (int f = 0; f < 3; ++f)
          TORCH_CHECK(!m[b][c][t][f] || std::isfinite(x[b][c][t][f]),
                      "[two-band-information] every legal observation must be finite");
  std::vector<Wave> slow_waves, fast_waves;
  for (int i = 0; i <= kInformationSlowGridIntervals; ++i)
    slow_waves.push_back(wave(kInformationSlowFrequencyMin +
        (kInformationSlowFrequencyMax - kInformationSlowFrequencyMin) * i / kInformationSlowGridIntervals));
  for (int i = 0; i <= kInformationFastGridIntervals; ++i)
    fast_waves.push_back(wave(kInformationFastFrequencyMin +
        (kInformationFastFrequencyMax - kInformationFastFrequencyMin) * i / kInformationFastGridIntervals));
  std::vector<InformationDecision> decisions;
  decisions.reserve(values.size(0));
  for (int64_t b = 0; b < values.size(0); ++b) {
    std::vector<Series> series;
    bool enough = true;
    for (int c = 0; c < 2; ++c) {
      std::array<bool, history> union_ticks{};
      int features = 0;
      for (int f = 0; f < 3; ++f) {
        Series item;
        item.channel = c;
        item.feature = f;
        int count = 0;
        long double sum = 0;
        for (int t = 0; t < history; ++t) {
          item.legal[t] = m[b][c][t][f];
          if (item.legal[t]) { ++count; sum += x[b][c][t][f]; }
        }
        if (count < kInformationMinimumFeatureTicks) continue;
        const long double mean = sum / count;
        item.mean = static_cast<double>(mean);
        long double energy = 0;
        for (int t = 0; t < history; ++t) {
          if (!item.legal[t]) continue;
          item.centered[t] = static_cast<double>(static_cast<long double>(x[b][c][t][f]) - mean);
          energy += static_cast<long double>(item.centered[t]) * item.centered[t];
        }
        item.energy = static_cast<double>(energy);
        if (!(item.energy > kInformationEnergyFloor) || !std::isfinite(item.energy)) continue;
        for (int t = 0; t < history; ++t) union_ticks[t] = union_ticks[t] || item.legal[t];
        ++features;
        series.push_back(item);
      }
      int first = history, last = -1, ticks = 0;
      for (int t = 0; t < history; ++t) {
        if (union_ticks[t]) { first = std::min(first, t); last = t; ++ticks; }
      }
      enough = enough && features >= kInformationMinimumFeaturesPerChannel &&
          ticks >= kInformationMinimumChannelUnionTicks && last - first >= kInformationMinimumChannelSpan;
    }
    if (!enough) {
      InformationDecision out;
      out.reason = "insufficient_observed_time_or_feature_support";
      decisions.push_back(out);
    } else {
      decisions.push_back(decide(series, search(series, slow_waves, fast_waves), task));
    }
  }
  return decisions;
}
} // namespace embedding::evaluation::two_component
