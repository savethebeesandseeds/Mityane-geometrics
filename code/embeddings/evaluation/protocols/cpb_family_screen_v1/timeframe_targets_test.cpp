// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// CPU contract test. No checkpoint, population, or encoder update.
#include "timeframe_targets.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace screen = embedding::evaluation::family_screen;

namespace {
void expect(bool condition, const char *why) { TORCH_CHECK(condition, "[timeframe targets test] ", why); }

void near(double actual, double expected, const char *why) {
  expect(std::isfinite(actual) && std::abs(actual - expected) <= 1e-9 * (1.0 + std::abs(expected)), why);
}

double population_std(const std::vector<double> &samples) {
  double sum = 0;
  for (const double sample : samples) sum += sample;
  const double mean = sum / static_cast<double>(samples.size());
  double square = 0;
  for (const double sample : samples) square += (sample - mean) * (sample - mean);
  return std::sqrt(square / static_cast<double>(samples.size()));
}

struct Window {
  torch::Tensor values, observed, ids;
};

Window blank(int64_t rows, int64_t channels, int64_t history, std::vector<int64_t> ids) {
  Window window;
  window.values = torch::zeros({rows, channels, history, 9}, torch::kFloat64);
  window.observed = torch::ones({rows, channels, history, 9}, torch::kBool);
  window.ids = torch::tensor(std::move(ids), torch::kInt64);
  return window;
}

void raw_anchor_ignores_the_coarser_channel() {
  constexpr int64_t kHistory = 9;
  auto window = blank(1, 2, kHistory, {7, 4});
  auto values = window.values.accessor<double, 4>();
  const std::vector<double> returns{1, 1, -1, 1, -1, 1, -1, 1};
  values[0][1][0][3] = 1;
  for (int64_t tick = 1; tick < kHistory; ++tick)
    values[0][1][tick][3] = values[0][1][tick - 1][3] * std::exp(returns[static_cast<std::size_t>(tick - 1)]);
  for (int64_t tick = 0; tick < kHistory; ++tick) values[0][0][tick][3] = 100;
  for (int64_t tick = 0; tick < kHistory; ++tick) values[0][0][tick][4] = 50;
  const auto targets = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::RawPrices);
  expect(targets.anchor_channel == 1 && targets.anchor_id == 4, "anchor is the 1m channel, not storage order");
  const auto row = targets.values.accessor<double, 2>();
  const auto valid = targets.valid.accessor<bool, 2>();
  expect(valid[0][0] && valid[0][1] && valid[0][2], "anchor columns are measured");
  near(row[0][1], 0.25, "anchor trend is the mean anchor log return");
  near(row[0][0], population_std(returns), "anchor volatility is not diluted by the coarser channel");
  near(row[0][2], 0, "anchor activity is log1p of the anchor volume");
  expect(!valid[0][3] && row[0][3] == 0, "cross_resolution abstains, and the stored value is zero");
}

void log_return_view_is_not_logged_again() {
  auto window = blank(1, 2, 9, {4, 7});
  auto values = window.values.accessor<double, 4>();
  for (int64_t tick = 0; tick < 9; ++tick) {
    values[0][0][tick][3] = -0.2;
    values[0][0][tick][4] = 1;
    values[0][1][tick][3] = 40;
    values[0][1][tick][4] = 80;
  }
  const auto logged = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::LogReturns);
  const auto row = logged.values.accessor<double, 2>();
  const auto valid = logged.valid.accessor<bool, 2>();
  expect(logged.anchor_id == 4 && valid[0][0] && valid[0][1] && valid[0][2], "log-return anchor columns are measured");
  near(row[0][1], -0.2, "a stored negative return stays the trend");
  near(row[0][0], 0, "a constant stored return has zero volatility");
  near(row[0][2], 1, "stored log1p volume is not transformed again");
  const auto raw = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::RawPrices);
  const auto raw_row = raw.values.accessor<double, 2>();
  const auto raw_valid = raw.valid.accessor<bool, 2>();
  expect(!raw_valid[0][0] && !raw_valid[0][1], "a negative close yields no raw return");
  expect(raw_valid[0][2], "nonnegative raw volume is still activity");
  near(raw_row[0][2], std::log1p(1), "raw activity applies log1p once");
}

void cross_resolution_abstains() {
  constexpr int64_t kHistory = 120;
  auto window = blank(1, 3, kHistory, {4, 7, 9});
  auto values = window.values.accessor<double, 4>();
  for (int64_t tick = 0; tick < kHistory; ++tick) {
    values[0][0][tick][3] = 0.01 * static_cast<double>(tick + 1);
    values[0][1][tick][3] = 0.02 * static_cast<double>(tick + 1);
    values[0][2][tick][3] = tick < 60 ? 5.0 : -5.0;
  }
  const auto targets = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::LogReturns);
  const auto row = targets.values.accessor<double, 2>();
  const auto valid = targets.valid.accessor<bool, 2>();
  expect(valid[0][0] && valid[0][1] && valid[0][2], "anchor columns stay measured");
  expect(!valid[0][3] && row[0][3] == 0, "slot index is not a shared clock");
}

void raw_cross_resolution_abstains() {
  constexpr int64_t kHistory = 30;
  auto window = blank(1, 2, kHistory, {4, 7});
  auto values = window.values.accessor<double, 4>();
  for (int64_t tick = 0; tick < kHistory; ++tick) {
    values[0][0][tick][3] = 1.0 + static_cast<double>(tick);
    values[0][1][tick][3] = 100.0 + static_cast<double>(tick);
  }
  const auto targets = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::RawPrices);
  const auto row = targets.values.accessor<double, 2>();
  const auto valid = targets.valid.accessor<bool, 2>();
  expect(valid[0][0] && valid[0][1], "raw anchor columns stay measured");
  expect(!valid[0][3] && row[0][3] == 0, "raw slot index is not a shared clock");
}

void one_channel_abstains_only_on_cross_resolution() {
  auto window = blank(1, 1, 9, {4});
  auto values = window.values.accessor<double, 4>();
  for (int64_t tick = 0; tick < 9; ++tick) values[0][0][tick][3] = 0.01 * static_cast<double>(tick + 1);
  const auto targets = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::LogReturns);
  const auto valid = targets.valid.accessor<bool, 2>();
  const auto row = targets.values.accessor<double, 2>();
  expect(valid[0][0] && valid[0][1] && valid[0][2], "one channel still measures its anchor");
  near(row[0][2], 0, "observed zero volume is log1p zero");
  expect(!valid[0][3] && row[0][3] == 0, "one channel has no cross-resolution correlation");
}

void short_anchor_support_abstains() {
  auto window = blank(1, 1, 9, {4});
  auto values = window.values.accessor<double, 4>();
  auto observed = window.observed.accessor<bool, 4>();
  for (int64_t tick = 0; tick < 9; ++tick) values[0][0][tick][3] = 0.05;
  for (int64_t tick = 7; tick < 9; ++tick) observed[0][0][tick][3] = false;
  const auto targets = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::LogReturns);
  const auto valid = targets.valid.accessor<bool, 2>();
  const auto row = targets.values.accessor<double, 2>();
  expect(!valid[0][0] && !valid[0][1] && row[0][0] == 0 && row[0][1] == 0,
      "seven anchor returns abstain");
}

template <class F>
void refuses(F &&call, const char *why) {
  bool threw = false;
  try {
    call();
  } catch (const std::exception &) {
    threw = true;
  }
  expect(threw, why);
}

void refusals() {
  auto withdrawn = blank(1, 3, 9, {0, 1, 2});
  refuses([&] {
    (void)screen::timeframe_window_targets(withdrawn.values, withdrawn.observed, withdrawn.ids, 60000,
        screen::FeatureView::RawPrices);
  }, "withdrawn instrument ids are refused");
  auto month = blank(1, 2, 9, {4, 18});
  refuses([&] {
    (void)screen::timeframe_window_targets(month.values, month.observed, month.ids, 60000, screen::FeatureView::LogReturns);
  }, "a calendar month has no fixed step");
  auto no_anchor = blank(1, 2, 9, {7, 9});
  refuses([&] {
    (void)screen::timeframe_window_targets(no_anchor.values, no_anchor.observed, no_anchor.ids, 60000,
        screen::FeatureView::LogReturns);
  }, "a window with no anchor step is refused");
  auto short_history = blank(1, 1, 8, {4});
  refuses([&] {
    (void)screen::timeframe_window_targets(short_history.values, short_history.observed, short_history.ids, 60000,
        screen::FeatureView::LogReturns);
  }, "history shorter than 9 is refused");
  auto broken = blank(1, 1, 9, {4});
  broken.values.accessor<double, 4>()[0][0][0][3] = std::numeric_limits<double>::quiet_NaN();
  refuses([&] {
    (void)screen::timeframe_window_targets(broken.values, broken.observed, broken.ids, 60000, screen::FeatureView::LogReturns);
  }, "a nonfinite observed value is refused");
}
} // namespace

int main() {
  raw_anchor_ignores_the_coarser_channel();
  log_return_view_is_not_logged_again();
  cross_resolution_abstains();
  raw_cross_resolution_abstains();
  one_channel_abstains_only_on_cross_resolution();
  short_anchor_support_abstains();
  refusals();
  std::cout << "timeframe targets passed\n";
  return 0;
}
