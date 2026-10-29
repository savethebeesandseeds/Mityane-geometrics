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

double pearson(const std::vector<double> &left, const std::vector<double> &right) {
  expect(left.size() == right.size() && left.size() >= 2, "fixture correlation needs two pairs");
  const double count = static_cast<double>(left.size());
  double sum_left = 0, sum_right = 0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    sum_left += left[i];
    sum_right += right[i];
  }
  const double mean_left = sum_left / count, mean_right = sum_right / count;
  double var_left = 0, var_right = 0, product = 0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    const double dx = left[i] - mean_left, dy = right[i] - mean_right;
    var_left += dx * dx;
    var_right += dy * dy;
    product += dx * dy;
  }
  var_left /= count;
  var_right /= count;
  product /= count;
  expect(var_left > 0 && var_right > 0, "fixture correlation needs variation");
  return product / std::sqrt(var_left * var_right);
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
  expect(!valid[0][3] && row[0][3] == 0, "fewer than eight coarser bars abstains, and the stored value is zero");
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

void cross_resolution_uses_one_sample_per_held_bar() {
  constexpr int64_t kHistory = 120;
  constexpr int64_t kBars = 8;
  auto window = blank(2, 3, kHistory, {4, 7, 9});
  auto values = window.values.accessor<double, 4>();
  std::vector<double> anchor_at_change, coarse_at_change, every_anchor, every_coarse;
  for (int64_t bar = 0; bar < kBars; ++bar) {
    const double sample = 0.1 * static_cast<double>(bar + 1);
    for (int64_t offset = 0; offset < 15; ++offset) {
      const int64_t tick = bar * 15 + offset;
      values[0][1][tick][3] = sample;
      every_coarse.push_back(sample);
      const double anchor = offset == 0 ? sample : 0;
      values[0][0][tick][3] = anchor;
      every_anchor.push_back(anchor);
    }
    anchor_at_change.push_back(sample);
    coarse_at_change.push_back(sample);
  }
  for (int64_t tick = 0; tick < kHistory; ++tick) values[0][2][tick][3] = tick < 60 ? 5.0 : -5.0;
  for (int64_t tick = 0; tick < kHistory; ++tick) values[1][1][tick][3] = 0.4;
  const auto targets = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::LogReturns);
  const auto row = targets.values.accessor<double, 2>();
  const auto valid = targets.valid.accessor<bool, 2>();
  expect(valid[0][3], "eight 15m bars support cross_resolution");
  near(row[0][3], 1, "paired change points are perfectly correlated");
  const double naive = pearson(every_anchor, every_coarse);
  expect(std::abs(naive - 1) > 0.05, "scoring every held tick is a different statistic");
  expect(!valid[1][3] && row[1][3] == 0, "one held bar abstains on that row only");
  expect(std::abs(row[0][3] - 0.5) > 0.05, "the two-bar hour channel does not halve the correlation");
}

void raw_cross_resolution_pairs_the_change_tick() {
  constexpr int64_t kBars = 9;
  constexpr int64_t kHistory = kBars * 15;
  auto window = blank(1, 2, kHistory, {4, 7});
  auto values = window.values.accessor<double, 4>();
  std::vector<double> step(static_cast<std::size_t>(kHistory), 0);
  std::vector<double> change_anchor, change_coarse, summed, coarse_return;
  for (int64_t bar = 1; bar < kBars; ++bar) {
    const double sample = static_cast<double>(bar);
    step[static_cast<std::size_t>(bar * 15)] = sample;
    step[static_cast<std::size_t>(bar * 15 + 1)] = bar % 2 == 1 ? 10 : -10;
    change_anchor.push_back(sample);
    change_coarse.push_back(sample);
    summed.push_back(sample + (bar % 2 == 1 ? 10 : -10));
    coarse_return.push_back(sample);
  }
  values[0][0][0][3] = 1;
  for (int64_t tick = 1; tick < kHistory; ++tick)
    values[0][0][tick][3] = values[0][0][tick - 1][3] * std::exp(step[static_cast<std::size_t>(tick)]);
  for (int64_t offset = 0; offset < 15; ++offset) values[0][1][offset][3] = 1;
  for (int64_t bar = 1; bar < kBars; ++bar) {
    const double close = std::exp(static_cast<double>(bar * (bar + 1)) / 2.0);
    for (int64_t offset = 0; offset < 15; ++offset) values[0][1][bar * 15 + offset][3] = close;
  }
  const auto targets = screen::timeframe_window_targets(
      window.values, window.observed, window.ids, 60000, screen::FeatureView::RawPrices);
  const auto row = targets.values.accessor<double, 2>();
  const auto valid = targets.valid.accessor<bool, 2>();
  expect(valid[0][3], "nine raw bars produce eight paired returns");
  near(row[0][3], pearson(change_anchor, change_coarse), "raw pairing uses the anchor return at the change tick");
  expect(std::abs(pearson(summed, coarse_return) - row[0][3]) > 0.05,
      "summing the held anchor ticks is a different statistic");
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
    screen::timeframe_window_targets(withdrawn.values, withdrawn.observed, withdrawn.ids, 60000,
        screen::FeatureView::RawPrices);
  }, "withdrawn instrument ids are refused");
  auto month = blank(1, 2, 9, {4, 18});
  refuses([&] {
    screen::timeframe_window_targets(month.values, month.observed, month.ids, 60000, screen::FeatureView::LogReturns);
  }, "a calendar month has no fixed step");
  auto no_anchor = blank(1, 2, 9, {7, 9});
  refuses([&] {
    screen::timeframe_window_targets(no_anchor.values, no_anchor.observed, no_anchor.ids, 60000,
        screen::FeatureView::LogReturns);
  }, "a window with no anchor step is refused");
  auto short_history = blank(1, 1, 8, {4});
  refuses([&] {
    screen::timeframe_window_targets(short_history.values, short_history.observed, short_history.ids, 60000,
        screen::FeatureView::LogReturns);
  }, "history shorter than 9 is refused");
  auto broken = blank(1, 1, 9, {4});
  broken.values.accessor<double, 4>()[0][0][0][3] = std::numeric_limits<double>::quiet_NaN();
  refuses([&] {
    screen::timeframe_window_targets(broken.values, broken.observed, broken.ids, 60000, screen::FeatureView::LogReturns);
  }, "a nonfinite observed value is refused");
}
} // namespace

int main() {
  raw_anchor_ignores_the_coarser_channel();
  log_return_view_is_not_logged_again();
  cross_resolution_uses_one_sample_per_held_bar();
  raw_cross_resolution_pairs_the_change_tick();
  one_channel_abstains_only_on_cross_resolution();
  short_anchor_support_abstains();
  refusals();
  std::cout << "timeframe targets passed\n";
  return 0;
}
