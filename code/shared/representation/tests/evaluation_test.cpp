// SPDX-License-Identifier: MIT
#include "embedding/shared/evaluation.h"
#include "shared_test_support.h"

#include <cmath>
#include <iostream>

namespace {
void test_generator_independence() {
  const embedding::input_shape_t config{2, 16, 3};
  const auto a = embedding::make_evaluation_data(config, 16, 11, 12, 13);
  const auto repeated = embedding::make_evaluation_data(config, 16, 11, 12, 13);
  const auto new_labels = embedding::make_evaluation_data(config, 16, 11, 12, 99);
  const auto new_signals = embedding::make_evaluation_data(config, 16, 97, 12, 13);
  const auto new_masks = embedding::make_evaluation_data(config, 16, 11, 98, 13);
  test::close(a.clean.data, repeated.clean.data, "reproducible trajectories", 0, 0);
  test::check(torch::equal(a.labels, repeated.labels), "labels are not reproducible");
  test::check(torch::equal(a.observed.feature_mask, repeated.observed.feature_mask), "masks are not reproducible");
  test::check(torch::equal(a.observed.feature_mask, new_labels.observed.feature_mask), "missingness depends on regime labels");
  test::check(torch::equal(a.observed.feature_mask, new_signals.observed.feature_mask), "missingness depends on signal values");
  test::check(torch::equal(a.clean.data, new_masks.clean.data), "signal generator depends on missingness stream");
  test::check(!torch::equal(a.clean.data, new_signals.clean.data), "separate signal seeds reuse trajectories");
  test::check(!torch::equal(a.observed.feature_mask, new_masks.observed.feature_mask), "mask seed has no effect");
  for (int64_t label = 0; label < 4; ++label)
    test::check(a.labels.eq(label).sum().item<int64_t>() == 4, "regime classes are not balanced");
}

void test_diagnostics() {
  const auto z = torch::tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {999.0, 999.0}});
  const auto valid = torch::tensor({true, true, true, true, false}, torch::kBool);
  const auto stats = embedding::representation_diagnostics(z, valid);
  test::check(stats.valid_rows == 4 && stats.dimensions == 2 && stats.valid_fraction == 0.8,
              "diagnostics count invalid rows");
  test::check(std::abs(stats.covariance_effective_rank - 2.0) < 1e-8, "effective rank of isotropic data is incorrect");
  test::check(std::abs(stats.norm_mean - 1.0) < 1e-8, "invalid row polluted norms");
  test::close(stats.per_dimension_std, torch::full({2}, std::sqrt(0.5), torch::kFloat64), "per-coordinate standard deviation");
  const auto empty = embedding::representation_diagnostics(z, torch::zeros_like(valid));
  test::check(empty.valid_rows == 0 && empty.covariance_effective_rank == 0 && empty.norm_mean == 0,
              "empty validity diagnostics are incorrect");
  const auto channel_stats = embedding::representation_diagnostics(z.unsqueeze(1), valid.unsqueeze(1));
  test::check(channel_stats.valid_rows == stats.valid_rows, "channel diagnostics flatten incorrectly");
  test::close(channel_stats.per_dimension_std, stats.per_dimension_std, "channel invalid-row exclusion", 0, 0);
  const auto singleton = embedding::representation_diagnostics(z.slice(0, 0, 1), valid.slice(0, 0, 1));
  test::check(singleton.valid_rows == 1 && singleton.covariance_effective_rank == 0 && singleton.std_mean == 0,
              "singleton covariance/variance diagnostics are incorrect");
  bool rejected = false;
  try { embedding::representation_diagnostics(torch::empty({2, 0}), torch::ones({2}, torch::kBool)); }
  catch (const std::exception &) { rejected = true; }
  test::check(rejected, "zero-dimensional diagnostics were accepted");
}

} // namespace

int main() {
  try {
    torch::set_num_threads(1);
    test_generator_independence();
    test_diagnostics();
    std::cout << "PASS: independent evaluation streams, valid-row diagnostics\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
  }
}
