// SPDX-License-Identifier: MIT
#include "observed_information.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace info = embedding::evaluation::two_component;
namespace {
constexpr double tau = 6.283185307179586476925286766559;
struct Input { torch::Tensor observed, mask; };
void check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const char *message) {
  check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}
template<class F> void rejects(F call, const char *message) {
  bool rejected = false;
  try { call(); } catch (const std::exception &) { rejected = true; }
  check(rejected, message);
}

// Independent scalar signal construction; the information implementation and
// generator are not called to construct this oracle. No latent enters infer.
Input signal(double slow_period, double fast_period, double slow_phase, double fast_phase,
             double slow_shift, double fast_shift, const std::array<double, 3> &ratios) {
  Input out{torch::zeros({1, 3, 32, 3}, torch::kFloat64),
            torch::ones({1, 3, 32, 3}, torch::kBool)};
  auto x = out.observed.accessor<double, 4>();
  for (int c = 0; c < 3; ++c) {
    for (int t = 0; t < 32; ++t) {
      for (int f = 0; f < 3; ++f) {
        const double ratio = ratios[f];
        const double slow = ratio / std::sqrt(1 + ratio * ratio);
        const double fast = 1 / std::sqrt(1 + ratio * ratio);
        const double gain = (c == 0 ? .57 : c == 1 ? 1.31 : .89) * (1 + .1 * f);
        const double offset = c == 0 ? -.61 + .11 * f : c == 1 ? .63 - .1 * f : .21;
        const double ds = c == 1 ? slow_shift : 0;
        const double df = c == 1 ? fast_shift : 0;
        const double ps = c == 2 ? slow_phase + 1.7 : slow_phase;
        const double pf = c == 2 ? fast_phase - .9 : fast_phase;
        x[0][c][t][f] = offset + gain * (slow * std::sin(tau * (t + ds) / slow_period + ps) +
                                      fast * std::sin(tau * (t + df) / fast_period + pf));
      }
    }
  }
  return out;
}
Input signal(double ps, double pf, double phase_s, double phase_f, double ds, double df, double ratio) {
  return signal(ps, pf, phase_s, phase_f, ds, df, {ratio, ratio, ratio});
}
info::InformationDecision infer(const Input &input, info::InformationTask task) {
  const auto out = info::infer_information(input.observed, input.mask, task);
  check(out.size() == 1, "one input row has one decision");
  return out.front();
}
void same(const info::InformationDecision &a, const info::InformationDecision &b, const char *message) {
  check(a.supported == b.supported && a.predicted_label == b.predicted_label &&
      a.slow_frequency == b.slow_frequency && a.fast_frequency == b.fast_frequency &&
      a.normalized_residual == b.normalized_residual && a.margin == b.margin &&
      a.coefficients == b.coefficients && a.eligible == b.eligible &&
      a.frequency_pairs_evaluated == b.frequency_pairs_evaluated && a.reason == b.reason, message);
}
void abstains(const Input &input, info::InformationTask task, const char *message) {
  const auto result = infer(input, task);
  check(!result.supported && result.predicted_label == 0 && !result.reason.empty(), message);
}

void selected_fit_witness(const Input &input, const info::InformationDecision &decision) {
  check(decision.frequency_pairs_evaluated == info::kInformationMaximumFrequencyPairs,
      "selected fit has exact bounded candidate evaluation count");
  const auto x = input.observed.accessor<double, 4>();
  const auto m = input.mask.accessor<bool, 4>();
  double residual = 0;
  int series = 0;
  for (int c = 0; c < 2; ++c) {
    for (int f = 0; f < 3; ++f) {
      const auto row = c * 3 + f;
      if (!decision.eligible[row]) {
        for (double coefficient : decision.coefficients[row])
          check(coefficient == 0, "ineligible selected-fit coefficients are zero");
        continue;
      }
      ++series;
      int count = 0;
      double mean = 0;
      for (int t = 0; t < 32; ++t)
        if (m[0][c][t][f]) { ++count; mean += x[0][c][t][f]; }
      mean /= count;
      std::array<double, 5> stationarity{};
      double squared_error = 0, centered_energy = 0;
      for (int t = 0; t < 32; ++t) {
        if (!m[0][c][t][f]) continue;
        const std::array<double, 5> basis{1, std::sin(tau * decision.slow_frequency * t),
            std::cos(tau * decision.slow_frequency * t), std::sin(tau * decision.fast_frequency * t),
            std::cos(tau * decision.fast_frequency * t)};
        double prediction = 0;
        for (int j = 0; j < 5; ++j) prediction += basis[j] * decision.coefficients[row][j];
        const double error = x[0][c][t][f] - prediction;
        squared_error += error * error;
        centered_energy += (x[0][c][t][f] - mean) * (x[0][c][t][f] - mean);
        for (int j = 0; j < 5; ++j) stationarity[j] += basis[j] * error;
      }
      for (double value : stationarity) near(value, 0, 1e-8, "selected original-unit normal-equation stationarity");
      residual += squared_error / centered_energy;
    }
  }
  near(residual / series, decision.normalized_residual, 1e-12, "independent selected fit residual without search replay");
}

void timing_cases() {
  int cases = 0;
  for (double ps : {12., 16.25, 20.}) {
    for (double pf : {4.5, 5.75, 7.}) {
      for (double phase : {0., .71, 5.93}) {
        for (int label : {0, 1}) {
          const double delay = .73;
          const double signed_delay = label ? delay : -delay;
          const auto input = signal(ps, pf, phase, 1.37 - phase, signed_delay,
              cases % 2 ? .41 : -.41, .93);
          const auto result = infer(input, info::InformationTask::SlowLagSign);
          check(result.supported && result.predicted_label == label, "independent noiseless slow-band phase sign");
          near(result.slow_frequency, 1 / ps, 1e-5, "bounded search slow frequency");
          near(result.fast_frequency, 1 / pf, 1e-5, "bounded search fast frequency");
          near(result.margin, tau * signed_delay / ps, .003, "sine/cosine oriented phase formula");
          check(result.normalized_residual < 1e-5, "noiseless two-band residual");
          ++cases;
        }
      }
    }
  }
  check(cases == 54, "closed timing fixture population");
}

void balance_cases() {
  int cases = 0;
  for (double ps : {12.5, 17.7, 19.8}) {
    for (double pf : {4.6, 5.9, 6.9}) {
      for (double ratio : {1.8, 2., 2.2}) {
        for (int label : {0, 1}) {
          const double actual_ratio = label ? ratio : 1 / ratio;
          const auto input = signal(ps, pf, .29 * cases, 2.17 - .17 * cases,
              cases % 2 ? .62 : -.62, cases % 3 ? .52 : -.52, actual_ratio);
          const auto result = infer(input, info::InformationTask::ComponentBalance);
          check(result.supported && result.predicted_label == label, "independent normalized component-balance classification");
          near(result.margin, std::log(actual_ratio), .003, "gain-free component ratio formula");
          check(result.normalized_residual < 1e-5, "balance noiseless residual");
          ++cases;
        }
      }
    }
  }
  check(cases == 54, "closed balance fixture population");
}

void masks_affine_and_scope() {
  auto input = signal(18.1, 5.3, .63, 5.2, .81, -.44, 1.07);
  auto mask = input.mask.accessor<bool, 4>();
  for (int c = 0; c < 3; ++c) {
    for (int t = 0; t < 32; ++t) {
      for (int f = 0; f < 3; ++f)
        mask[0][c][t][f] = (t + 2 * f + c) % 5 != 0 && !(t >= 8 + c && t < 11 + c);
    }
  }
  input.observed.masked_fill_(~input.mask, 0);
  const auto result = infer(input, info::InformationTask::SlowLagSign);
  check(result.supported && result.predicted_label == 1, "gappy per-feature legal fit and channel union");
  selected_fit_witness(input, result);
  same(result, infer(input, info::InformationTask::SlowLagSign), "deterministic bounded search");
  auto poison = input;
  poison.observed = input.observed.clone();
  poison.observed.masked_fill_(~input.mask, std::numeric_limits<double>::quiet_NaN());
  same(result, infer(poison, info::InformationTask::SlowLagSign), "hidden NaNs never enter centering, Gram or residual");
  auto nuisance = input;
  nuisance.observed = input.observed.clone();
  auto x = nuisance.observed.accessor<double, 4>();
  for (int c = 0; c < 2; ++c)
    for (int t = 0; t < 32; ++t)
      for (int f = 0; f < 3; ++f)
        if (mask[0][c][t][f]) x[0][c][t][f] = (1.1 + .1 * c + .03 * f) * x[0][c][t][f] - .4 + .08 * f;
  const auto transformed = infer(nuisance, info::InformationTask::SlowLagSign);
  check(transformed.supported && transformed.predicted_label == result.predicted_label, "independent positive affine nuisances");
  near(transformed.margin, result.margin, 1e-10, "offset/positive gain leave phase decision unchanged");
  auto unrelated = input;
  unrelated.observed = input.observed.clone();
  unrelated.observed.select(1, 2).fill_(1e100);
  same(result, infer(unrelated, info::InformationTask::SlowLagSign), "channel2 cannot affect the information rule");
  auto strided = input;
  strided.observed = input.observed.transpose(1, 3).contiguous().transpose(1, 3);
  strided.mask = input.mask.transpose(1, 3).contiguous().transpose(1, 3);
  check(!strided.observed.is_contiguous(), "actual noncontiguous fixture");
  same(result, infer(strided, info::InformationTask::SlowLagSign), "noncontiguous storage does not change logical observations");
  auto f32 = input;
  f32.observed = input.observed.to(torch::kFloat32);
  const auto reduced = infer(f32, info::InformationTask::SlowLagSign);
  check(reduced.supported && reduced.predicted_label == 1, "F32 legal values accepted");
  near(reduced.margin, result.margin, 1e-4, "F32 phase reference");

  // Different features expose complementary original ticks. The rule fits
  // their separate coefficients; it never fills a hidden cell or averages
  // features whose fitted constant offsets differ.
  auto complementary = signal(16.8, 5.45, 1.3, .49, -.68, .37, .88);
  auto cm = complementary.mask.accessor<bool, 4>();
  for (int c = 0; c < 3; ++c)
    for (int t = 0; t < 32; ++t)
      for (int f = 0; f < 3; ++f) cm[0][c][t][f] = (t + c) % 3 == f;
  complementary.observed.masked_fill_(~complementary.mask, 0);
  const auto crossed = infer(complementary, info::InformationTask::SlowLagSign);
  check(crossed.supported && crossed.predicted_label == 0, "complementary-feature frequency fit with observed phase coherence");

  auto noisy = signal(17.3, 6.1, 2.2, .71, -.77, -.49, 2.05);
  auto nx = noisy.observed.accessor<double, 4>();
  for (int c = 0; c < 3; ++c)
    for (int t = 0; t < 32; ++t)
      for (int f = 0; f < 3; ++f) nx[0][c][t][f] += .003 * std::sin(2.71 * t + .6 * f + .4 * c);
  const auto noise_result = infer(noisy, info::InformationTask::ComponentBalance);
  check(noise_result.supported && noise_result.predicted_label == 1 &&
      noise_result.normalized_residual < info::kInformationMaximumNormalizedResidual,
      "bounded small deterministic noise, no noise-level search");

  const auto negative = signal(18.1, 5.3, .63, 5.2, -.81, -.44, 1.07);
  const auto batch_values = torch::cat({input.observed, negative.observed}, 0);
  const auto batch_mask = torch::cat({input.mask, negative.mask}, 0);
  const auto batch = info::infer_information(batch_values, batch_mask, info::InformationTask::SlowLagSign);
  check(batch.size() == 2 && batch[0].predicted_label == 1 && batch[1].predicted_label == 0,
      "row order retained without pair IDs or labels");
  same(batch[0], result, "no cross-row frequency or coefficient fitting");
}

void abstention_and_validation() {
  auto input = signal(17.2, 5.8, .91, 2.3, .66, -.48, 1.03);
  auto empty = input;
  empty.mask = torch::zeros_like(input.mask);
  empty.observed = torch::full_like(input.observed, std::numeric_limits<double>::quiet_NaN());
  abstains(empty, info::InformationTask::SlowLagSign, "empty support retains undefined decision with zero prediction");
  auto one_feature = input;
  one_feature.mask = torch::zeros_like(input.mask);
  one_feature.mask.select(3, 0).fill_(true);
  abstains(one_feature, info::InformationTask::SlowLagSign, "one feature/channel below support contract");
  auto short_span = input;
  short_span.mask = input.mask.clone();
  short_span.mask.slice(2, 16, 32).fill_(false);
  abstains(short_span, info::InformationTask::SlowLagSign, "short time span cannot identify a declared two-band model");
  auto too_few = input;
  too_few.mask = input.mask.clone();
  too_few.mask.slice(2, 9, 32).fill_(false);
  abstains(too_few, info::InformationTask::ComponentBalance, "fewer than ten legal ticks per feature abstain");
  auto constant = input;
  constant.observed = torch::full_like(input.observed, .37);
  abstains(constant, info::InformationTask::SlowLagSign, "constant signal lacks two-component energy");
  const auto zero_shift = signal(17.2, 5.8, .91, 2.3, 0, -.48, 1.03);
  const auto zero_decision = infer(zero_shift, info::InformationTask::SlowLagSign);
  check(!zero_decision.supported && zero_decision.predicted_label == 0,
      "zero physical slow shift is not forced into a label");
  selected_fit_witness(zero_shift, zero_decision);
  abstains(signal(17.2, 5.8, .91, 2.3, 4, -.48, 1.03), info::InformationTask::SlowLagSign,
      "phase outside declared support abstains");
  abstains(signal(17.2, 5.8, .91, 2.3, .66, -.48, 1), info::InformationTask::ComponentBalance,
      "equal components have ambiguous balance");
  abstains(signal(17.2, 5.8, .91, 2.3, .66, -.48, 3), info::InformationTask::ComponentBalance,
      "ratio outside declared support abstains");
  abstains(signal(17.2, 5.8, .91, 2.3, .66, -.48, {1.8, 2.0, 2.7}),
      info::InformationTask::ComponentBalance, "inconsistent positive component ratios abstain");
  auto wrong_law = input;
  wrong_law.observed = input.observed.clone();
  auto wx = wrong_law.observed.accessor<double, 4>();
  for (int c = 0; c < 2; ++c)
    for (int t = 0; t < 32; ++t)
      for (int f = 0; f < 3; ++f) wx[0][c][t][f] = (t % 2 ? 1. : -1.) * (1 + .1 * f) + .02 * t;
  abstains(wrong_law, info::InformationTask::SlowLagSign, "non-two-band residual cannot create supported timing");
  auto nonfinite = input;
  nonfinite.observed = input.observed.clone();
  nonfinite.observed[0][0][0][0] = std::numeric_limits<double>::infinity();
  rejects([&]{infer(nonfinite, info::InformationTask::SlowLagSign);}, "legal infinity rejected");
  auto wrong_mask = input;
  wrong_mask.mask = input.mask.to(torch::kFloat64);
  rejects([&]{infer(wrong_mask, info::InformationTask::SlowLagSign);}, "mask dtype rejected");
  auto wrong_shape = input;
  wrong_shape.mask = input.mask.narrow(2, 0, 31);
  rejects([&]{infer(wrong_shape, info::InformationTask::SlowLagSign);}, "mask shape rejected");
  rejects([&]{info::infer_information(input.observed.to(torch::kInt64), input.mask, info::InformationTask::SlowLagSign);},
      "observations dtype rejected");
  rejects([&]{info::infer_information(input.observed.squeeze(0), input.mask.squeeze(0), info::InformationTask::SlowLagSign);},
      "missing batch dimension rejected");
  rejects([&]{info::infer_information(input.observed, input.mask, static_cast<info::InformationTask>(99));},
      "unknown information task rejected");
}
} // namespace

int main() try {
  torch::set_num_threads(1);
  check(info::kInformationMaximumFrequencyPairs == 515, "fixed bounded search budget");
  timing_cases();
  balance_cases();
  masks_affine_and_scope();
  abstention_and_validation();
  std::cout << "Two-component observed information fixtures passed: 54 timing and54 balance affine noiseless cases; "
               "legal masks, independent scalar phases/ratios and abstentions; no encoder/head fitting or quality evaluation\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
