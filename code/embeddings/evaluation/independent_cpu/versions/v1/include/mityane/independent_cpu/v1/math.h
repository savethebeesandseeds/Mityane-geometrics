#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mityane::independent_cpu::v1 {
class ArithmeticError : public std::runtime_error {
 public: using std::runtime_error::runtime_error;
};
using Rows = std::vector<std::vector<double>>;
using Mask = std::vector<std::uint8_t>;

// Seed text never passes through a floating-point JSON value.
std::uint64_t parse_seed(std::string_view decimal);
std::uint64_t stream_seed(std::uint64_t seed, std::uint64_t stream);
std::uint64_t mixed(std::uint64_t value);
std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes);
std::string fnv1a64_hex(std::span<const std::uint8_t> bytes);
std::string source_manifest(std::span<const std::string> utf8_ids);

class MT19937_64 {
 public:
  explicit MT19937_64(std::uint64_t seed);
  std::uint64_t draw();
 private:
  std::array<std::uint64_t, 312> state_;
  std::size_t index_ = 312;
};
std::uint64_t uniform_index(MT19937_64& engine, std::uint64_t count);

// Finite-only arithmetic profile: IEEE binary32/binary64, round-to-nearest,
// gradual underflow. Build with -fno-fast-math -ffp-contract=off.
void require_numeric_profile();
double f32(double value);
double finite_sum(std::span<const double> values);
bool close(double actual, double reference, double atol = 2e-9, double rtol = 2e-9);
Rows normalize(const Rows& rows, const Mask& valid, std::span<const double> mean, std::span<const double> scale);
Rows logits(const Rows& rows, const Rows& weights, std::span<const double> intercept);
std::vector<std::int64_t> binary_argmax(const Rows& saved_logits);
struct Score {
  std::uint64_t total = 0, valid = 0, correct = 0;
  double coverage = 0;
  std::optional<double> accuracy;
};
Score score(std::span<const std::int64_t> prediction, std::span<const std::int64_t> labels, const Mask& valid);
struct Group { double numerator = 0; std::uint64_t denominator = 0; };
std::vector<Group> accuracy_groups(std::span<const std::int64_t> prediction, std::span<const std::int64_t> labels,
                                   const Mask& valid, std::span<const std::string> sources);
std::vector<Group> paired_groups(std::span<const std::int64_t> left, std::span<const std::int64_t> right,
                                 std::span<const std::int64_t> labels, const Mask& left_valid,
                                 const Mask& right_valid, std::span<const std::string> sources);
struct Interval {
  std::optional<double> estimate, lower, upper;
  std::size_t source_groups = 0, replicates = 1000;
};
Interval bootstrap_groups(std::span<const Group> groups, std::uint64_t seed, std::size_t replicates = 1000);
struct MapStatistics { std::vector<double> mean, scale; std::size_t fitted_rows; };
MapStatistics map_statistics(const Rows& rows, const Mask& training_valid, double floor = 1e-8);
struct DeletionView { std::vector<double> values; Mask mask; };
DeletionView deletion_view(std::span<const double> values, const Mask& observed, const Mask& erasure);
struct QueryMasks { Mask requested, visible, query, eligible; };
QueryMasks query_masks(const Mask& observed, std::size_t rows, std::size_t channels = 3,
                       std::size_t history = 32, std::size_t features = 3, std::size_t patch = 8);
struct Reductions {
  std::vector<std::uint64_t> counts;
  std::vector<double> channel_mae, channel_huber, example_mae, example_huber;
  Mask example_valid;
  std::optional<double> mae, huber;
};
Reductions reductions(std::span<const double> prediction, std::span<const double> target, const Mask& query,
                       std::size_t rows, std::size_t channels = 3, std::size_t history = 32,
                       std::size_t features = 3, std::size_t patch = 8);
}  // namespace mityane::independent_cpu::v1
