// SPDX-License-Identifier: MIT
#include "embedding/shared/data.h"
#include "shared_test_support.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>

namespace {

struct TemporaryFiles {
  std::filesystem::path directory;

  TemporaryFiles() {
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path() /
                ("embedding-shared-data-test-" + std::to_string(id));
    test::check(std::filesystem::create_directory(directory),
                "could not create a unique test directory");
  }

  ~TemporaryFiles() {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }

  std::string file(const std::string &name) const {
    return (directory / name).string();
  }
};

template <typename Function>
void rejects(Function function, const std::string &label) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception &) {
    rejected = true;
  }
  test::check(rejected, label + " was accepted");
}

embedding::input_shape_t shape() { return {2, 16, 3}; }

void test_synthetic_fixture() {
  const auto dimensions = shape();
  torch::manual_seed(71);
  const auto batch = embedding::synthetic_batch(dimensions, 5, 2);
  torch::manual_seed(71);
  const auto repeated = embedding::synthetic_batch(dimensions, 5, 2);
  test::check(batch.data.sizes() == torch::IntArrayRef({5, 2, 16, 3}) &&
                  batch.data.scalar_type() == torch::kFloat32 &&
                  batch.data.device().is_cpu(),
              "synthetic data must be CPU float32 [B,C,H,F]");
  test::check(batch.feature_mask.scalar_type() == torch::kBool &&
                  batch.feature_mask.sizes() == batch.data.sizes(),
              "synthetic observation mask contract changed");
  test::finite(batch.data, "synthetic data");
  test::check(batch.data.masked_select(batch.feature_mask.logical_not())
                  .eq(0).all().item<bool>(),
              "synthetic missing values were not zeroed");
  test::close(batch.data, repeated.data, "seeded synthetic data", 0, 0);
  test::check(torch::equal(batch.feature_mask, repeated.feature_mask),
              "seeded synthetic mask changed");
  auto shortest = dimensions;
  shortest.history_length = 1;
  embedding::archive::validate_batch(
      embedding::synthetic_batch(shortest, 1, 1), shortest);
}

void test_invalid_dimensions(const TemporaryFiles &files) {
  for (auto member : {&embedding::input_shape_t::channel_count,
                      &embedding::input_shape_t::history_length,
                      &embedding::input_shape_t::input_width}) {
    for (const int64_t invalid : {int64_t{0}, int64_t{-1}}) {
      auto dimensions = shape();
      dimensions.*member = invalid;
      rejects([&] { embedding::synthetic_batch(dimensions, 2); },
              "nonpositive synthetic input dimension");
    }
    auto dimensions = shape();
    dimensions.*member = 0;
    const auto data = torch::zeros(
        {2, dimensions.channel_count, dimensions.history_length,
         dimensions.input_width}, torch::kFloat32);
    const embedding::Batch empty{data, torch::ones_like(data, torch::kBool)};
    rejects([&] { embedding::archive::validate_batch(empty, dimensions); },
            "shape-matching zero-dimension batch");
    const auto path = files.file("zero-dimension.pt");
    embedding::save_batch(path, empty);
    rejects([&] { embedding::load_batch(path, dimensions); },
            "shape-matching zero-dimension archive");
  }
}

void test_archive_contract(const TemporaryFiles &files) {
  const auto dimensions = shape();
  torch::manual_seed(71);
  auto batch = embedding::synthetic_batch(dimensions, 5, 2);
  batch.feature_mask.index_put_({0, 1}, false);
  batch.feature_mask.index_put_({1}, false);
  batch.data.masked_fill_(batch.feature_mask.logical_not(), 0.0);
  const auto expected = batch.data.clone();
  batch.data.masked_fill_(batch.feature_mask.logical_not(),
                         std::numeric_limits<float>::quiet_NaN());
  const auto path = files.file("input.pt");
  embedding::save_batch(path, batch);

  torch::serialize::InputArchive archive;
  archive.load_from(path, torch::kCPU);
  torch::Tensor data, mask;
  archive.read("data", data, true);
  archive.read("feature_mask", mask, true);
  test::check(data.scalar_type() == torch::kFloat32 &&
                  mask.scalar_type() == torch::kBool,
              "input archive tensor types changed");
  test::check(torch::equal(mask, batch.feature_mask),
              "input archive feature mask changed");
  const auto loaded = embedding::load_batch(path, dimensions);
  test::close(loaded.data, expected, "input archive sanitized data", 0, 0);
  test::check(torch::equal(loaded.feature_mask, batch.feature_mask),
              "loaded observation mask changed");

  auto incompatible = dimensions;
  incompatible.history_length += 1;
  rejects([&] { embedding::load_batch(path, incompatible); },
          "input archive with incompatible shape");
  for (const auto &invalid : {
           embedding::Batch{expected.to(torch::kFloat64), batch.feature_mask},
           embedding::Batch{expected, batch.feature_mask.to(torch::kFloat32)},
           embedding::Batch{expected, batch.feature_mask.narrow(2, 0, 15)}}) {
    const auto bad_path = files.file("invalid.pt");
    embedding::save_batch(bad_path, invalid);
    rejects([&] { embedding::load_batch(bad_path, dimensions); },
            "invalid input archive tensor contract");
  }
  auto poisoned = expected.clone();
  auto observed = batch.feature_mask.clone();
  poisoned.index_put_({2, 0, 0, 0},
                     std::numeric_limits<float>::quiet_NaN());
  observed.index_put_({2, 0, 0, 0}, true);
  const auto bad_path = files.file("observed-nan.pt");
  embedding::save_batch(bad_path, {poisoned, observed});
  rejects([&] { embedding::load_batch(bad_path, dimensions); },
          "observed nonfinite input archive value");
}

} // namespace

int main() {
  try {
    torch::set_num_threads(1);
    TemporaryFiles files;
    test_synthetic_fixture();
    test_invalid_dimensions(files);
    test_archive_contract(files);
    std::cout << "PASS: shared synthetic fixture, dimensions, input archives and masks\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
