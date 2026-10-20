#include "mityane/independent_cpu/v1/math.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace mityane::independent_cpu::v1 {
namespace {
void require(bool value, const char* why) { if (!value) throw ArithmeticError(why); }
void finite(double value) { require(std::isfinite(value), "nonfinite independent arithmetic"); }
void mask(const Mask& values) { for (const auto v : values) require(v <= 1, "noncanonical mask"); }
std::size_t size_product(std::initializer_list<std::size_t> factors) {
  std::size_t out = 1;
  for (const auto n : factors) { require(n == 0 || out <= SIZE_MAX / n, "geometry overflow"); out *= n; }
  return out;
}
std::pair<std::uint64_t, std::uint64_t> wide_multiply(std::uint64_t a, std::uint64_t b) {
  constexpr std::uint64_t low_mask = 0xffffffffULL;
  const auto a0 = a & low_mask, a1 = a >> 32, b0 = b & low_mask, b1 = b >> 32;
  const auto first = a0 * b0, second = a1 * b0 + (first >> 32);
  const auto third = a0 * b1 + (second & low_mask);
  return {a1 * b1 + (second >> 32) + (third >> 32), (third << 32) | (first & low_mask)};
}
void binary_rows(std::span<const std::int64_t> prediction, std::span<const std::int64_t> labels, const Mask& valid) {
  require(prediction.size() == labels.size() && labels.size() == valid.size() && !labels.empty(), "score row population"); mask(valid);
  for (std::size_t i = 0; i < labels.size(); ++i) require((labels[i] == 0 || labels[i] == 1) && (prediction[i] == 0 || prediction[i] == 1), "nonbinary prediction/label");
}
}  // namespace
std::uint64_t parse_seed(std::string_view decimal) {
  require(!decimal.empty(), "empty uint64 seed"); std::uint64_t out = 0;
  for (const char c : decimal) {
    require(c >= '0' && c <= '9', "seed must contain decimal digits only"); const auto digit = static_cast<unsigned>(c - '0');
    require(out <= (UINT64_MAX - digit) / 10, "uint64 seed overflow"); out = out * 10 + digit;
  }
  return out;
}
std::uint64_t stream_seed(std::uint64_t seed, std::uint64_t stream) {
  auto value = seed + 0x9e3779b97f4a7c15ULL * stream;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
std::uint64_t mixed(std::uint64_t value) { return stream_seed(value, 1); }
std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes) {
  std::uint64_t out = 14695981039346656037ULL; for (auto b : bytes) out = (out ^ b) * 1099511628211ULL; return out;
}
std::string fnv1a64_hex(std::span<const std::uint8_t> bytes) {
  auto value = fnv1a64(bytes); std::string out(16, '0');
  for (std::size_t i = 16; i-- > 0;) { out[i] = "0123456789abcdef"[value & 15]; value >>= 4; } return out;
}
std::string source_manifest(std::span<const std::string> ids) {
  std::string out; for (const auto& id : ids) { out += std::to_string(id.size()); out += ':'; out += id; } return out;
}
MT19937_64::MT19937_64(std::uint64_t seed) {
  state_[0] = seed;
  for (std::size_t i = 1; i < state_.size(); ++i) state_[i] = 6364136223846793005ULL * (state_[i - 1] ^ (state_[i - 1] >> 62)) + i;
}
std::uint64_t MT19937_64::draw() {
  if (index_ == state_.size()) {
    for (std::size_t i = 0; i < state_.size(); ++i) {
      const auto value = (state_[i] & 0xffffffff80000000ULL) | (state_[(i + 1) % 312] & 0x7fffffffULL);
      state_[i] = state_[(i + 156) % 312] ^ (value >> 1) ^ ((value & 1) ? 0xb5026f5aa96619e9ULL : 0);
    }
    index_ = 0;
  }
  auto value = state_[index_++];
  value ^= (value >> 29) & 0x5555555555555555ULL; value ^= (value << 17) & 0x71d67fffeda60000ULL;
  value ^= (value << 37) & 0xfff7eee000000000ULL; return value ^ (value >> 43);
}
std::uint64_t uniform_index(MT19937_64& engine, std::uint64_t count) {
  require(count > 0, "positive uniform range required"); const auto threshold = (std::uint64_t(0) - count) % count;
  for (;;) { const auto [high, low] = wide_multiply(engine.draw(), count); if (low >= threshold) return high; }
}
void require_numeric_profile() {
  require(sizeof(float) == 4 && sizeof(double) == 8 && std::numeric_limits<float>::is_iec559 &&
          std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53 &&
          std::numeric_limits<float>::digits == 24 && std::fegetround() == FE_TONEAREST,
          "requires IEEE binary32/binary64 with round-to-nearest");
  volatile double denormal = std::numeric_limits<double>::denorm_min();
  volatile double one = 1.0; const double preserved = denormal * one;
  volatile float float_denormal = std::numeric_limits<float>::denorm_min();
  volatile float float_one = 1.0f; const float float_preserved = float_denormal * float_one;
  require(preserved != 0.0 && float_preserved != 0.0f, "binary32/binary64 gradual underflow required");
}
double f32(double value) {
  require_numeric_profile(); finite(value); const auto result = static_cast<float>(value); finite(result); return static_cast<double>(result);
}
double finite_sum(std::span<const double> values) {
  // Error-free TwoSum expansion. The final round includes the halfway correction
  // used by finite correctly rounded partials summation, not naive accumulation.
  require_numeric_profile(); std::vector<double> partials; partials.reserve(64);
  for (double value : values) {
    finite(value); std::size_t used = 0;
    for (double part : partials) {
      if (std::abs(value) < std::abs(part)) std::swap(value, part);
      const double high = value + part; finite(high);
      const double recovered = high - value, low = part - recovered;
      if (low != 0.0) partials[used++] = low;
      value = high;
    }
    partials.resize(used); if (value != 0.0) partials.push_back(value);
  }
  if (partials.empty()) return 0.0;
  auto n = partials.size(); double high = partials[--n], low = 0.0;
  while (n) {
    const double x = high, y = partials[--n]; high = x + y; low = y - (high - x);
    if (low != 0.0) break;
  }
  if (n && ((low < 0 && partials[n - 1] < 0) || (low > 0 && partials[n - 1] > 0))) {
    const double twice = low * 2, rounded = high + twice;
    if (rounded - high == twice) high = rounded;
  }
  finite(high); return high;
}
bool close(double actual, double reference, double atol, double rtol) {
  require(std::isfinite(atol) && std::isfinite(rtol) && atol >= 0 && rtol >= 0, "invalid comparison tolerance");
  return std::isfinite(actual) && std::isfinite(reference) && std::abs(actual - reference) <= atol + rtol * std::abs(reference);
}
Rows normalize(const Rows& rows, const Mask& valid, std::span<const double> mean, std::span<const double> scale) {
  require(rows.size() == valid.size() && !mean.empty() && mean.size() == scale.size(), "frozen map geometry"); mask(valid);
  for (std::size_t d = 0; d < mean.size(); ++d) { finite(mean[d]); require(std::isfinite(scale[d]) && scale[d] > 0, "frozen map positive scale"); }
  Rows out;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    require(rows[i].size() == mean.size(), "feature width"); std::vector<double> row(mean.size(), 0.0);
    for (std::size_t d = 0; d < mean.size(); ++d) { finite(rows[i][d]); if (valid[i]) { row[d] = (rows[i][d] - mean[d]) / scale[d]; finite(row[d]); } }
    out.push_back(std::move(row));
  }
  return out;
}
Rows logits(const Rows& rows, const Rows& weights, std::span<const double> intercept) {
  require(!weights.empty() && !intercept.empty(), "linear geometry"); for (const auto& row : weights) require(row.size() == intercept.size(), "linear weight geometry");
  Rows out; std::vector<double> products(weights.size());
  for (const auto& row : rows) {
    require(row.size() == weights.size(), "linear feature width"); std::vector<double> result(intercept.size());
    for (std::size_t c = 0; c < result.size(); ++c) {
      finite(intercept[c]); for (std::size_t d = 0; d < row.size(); ++d) { finite(row[d]); finite(weights[d][c]); products[d] = row[d] * weights[d][c]; }
      result[c] = finite_sum(products) + intercept[c]; finite(result[c]);
    }
    out.push_back(std::move(result));
  }
  return out;
}
std::vector<std::int64_t> binary_argmax(const Rows& values) {
  std::vector<std::int64_t> out;
  for (const auto& row : values) { require(row.size() == 2 && std::isfinite(row[0]) && std::isfinite(row[1]), "finite binary logits"); out.push_back(row[1] > row[0] ? 1 : 0); }
  return out;
}
Score score(std::span<const std::int64_t> prediction, std::span<const std::int64_t> labels, const Mask& valid) {
  binary_rows(prediction, labels, valid); Score out; out.total = labels.size();
  for (std::size_t i = 0; i < labels.size(); ++i) if (valid[i]) { ++out.valid; out.correct += prediction[i] == labels[i]; }
  out.coverage = double(out.valid) / double(out.total); if (out.valid) out.accuracy = double(out.correct) / double(out.valid); return out;
}
std::vector<Group> accuracy_groups(std::span<const std::int64_t> prediction, std::span<const std::int64_t> labels,
                                   const Mask& valid, std::span<const std::string> sources) {
  binary_rows(prediction, labels, valid); require(sources.size() == labels.size(), "source row geometry"); std::map<std::string, Group> groups;
  for (std::size_t i = 0; i < labels.size(); ++i) if (valid[i]) { auto& group = groups[sources[i]]; group.numerator += prediction[i] == labels[i]; ++group.denominator; }
  std::vector<Group> out; for (const auto& [key, group] : groups) { (void)key; out.push_back(group); } return out;
}
std::vector<Group> paired_groups(std::span<const std::int64_t> left, std::span<const std::int64_t> right,
                                 std::span<const std::int64_t> labels, const Mask& lv, const Mask& rv, std::span<const std::string> sources) {
  binary_rows(left, labels, lv); binary_rows(right, labels, rv); require(sources.size() == labels.size(), "paired source geometry"); std::map<std::string, Group> groups;
  for (std::size_t i = 0; i < labels.size(); ++i) if (lv[i] && rv[i]) { auto& group = groups[sources[i]]; group.numerator += int(left[i] == labels[i]) - int(right[i] == labels[i]); ++group.denominator; }
  std::vector<Group> out; for (const auto& [key, group] : groups) { (void)key; out.push_back(group); } return out;
}
Interval bootstrap_groups(std::span<const Group> groups, std::uint64_t seed, std::size_t replicates) {
  require(replicates == 1000, "frozen bootstrap budget is 1000"); Interval out; out.source_groups = groups.size(); out.replicates = replicates;
  if (groups.empty()) return out;
  std::vector<double> numerators; std::uint64_t total = 0;
  for (const auto& group : groups) { finite(group.numerator); require(group.denominator > 0 && total <= UINT64_MAX - group.denominator, "bootstrap denominator"); total += group.denominator; numerators.push_back(group.numerator); }
  out.estimate = finite_sum(numerators) / double(total); if (groups.size() == 1) return out;
  MT19937_64 engine(seed); std::vector<double> samples; samples.reserve(replicates);
  for (std::size_t r = 0; r < replicates; ++r) {
    numerators.clear(); total = 0;
    for (std::size_t i = 0; i < groups.size(); ++i) { const auto& group = groups[uniform_index(engine, groups.size())]; numerators.push_back(group.numerator); require(total <= UINT64_MAX - group.denominator, "bootstrap denominator overflow"); total += group.denominator; }
    const auto sample = finite_sum(numerators) / double(total); finite(sample); samples.push_back(sample);
  }
  std::sort(samples.begin(), samples.end()); out.lower = samples[24]; out.upper = samples[974]; return out;
}
MapStatistics map_statistics(const Rows& rows, const Mask& valid, double floor) {
  require(rows.size() == valid.size() && !rows.empty() && !rows[0].empty() && std::isfinite(floor) && floor > 0, "TRAIN map geometry/floor"); mask(valid);
  const auto width = rows[0].size(); MapStatistics out; out.fitted_rows = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) { require(rows[i].size() == width, "ragged TRAIN rows"); for (auto x : rows[i]) finite(x); out.fitted_rows += valid[i]; }
  require(out.fitted_rows >= 2, "TRAIN map requires two supported rows"); std::vector<double> values;
  for (std::size_t d = 0; d < width; ++d) {
    values.clear(); for (std::size_t i = 0; i < rows.size(); ++i) if (valid[i]) values.push_back(rows[i][d]);
    const auto mean = finite_sum(values) / double(out.fitted_rows); values.clear();
    for (std::size_t i = 0; i < rows.size(); ++i) if (valid[i]) { const auto delta = rows[i][d] - mean; values.push_back(delta * delta); }
    const auto scale = std::sqrt(finite_sum(values) / double(out.fitted_rows)); finite(mean); finite(scale);
    out.mean.push_back(mean); out.scale.push_back(std::max(floor, scale));
  }
  return out;
}
DeletionView deletion_view(std::span<const double> values, const Mask& observed, const Mask& erasure) {
  require(values.size() == observed.size() && values.size() == erasure.size(), "deletion geometry"); mask(observed); mask(erasure); DeletionView out;
  for (std::size_t i = 0; i < values.size(); ++i) { finite(values[i]); require(observed[i] || values[i] == 0, "hidden base payload"); const auto keep = observed[i] && !erasure[i]; out.mask.push_back(keep); out.values.push_back(keep ? values[i] : 0.0); }
  return out;
}
QueryMasks query_masks(const Mask& observed, std::size_t rows, std::size_t channels, std::size_t history, std::size_t features, std::size_t patch) {
  require(rows > 0 && channels > 0 && features > 0 && patch > 0 && history > 0 && history % patch == 0, "query dimensions"); mask(observed);
  const auto trials = history / patch, row_channels = size_product({rows, channels});
  require(observed.size() == size_product({rows, channels, history, features}), "query geometry");
  auto index = [=](auto b, auto c, auto h, auto f) { return ((b * channels + c) * history + h) * features + f; };
  QueryMasks out;
  for (std::size_t trial = 0; trial < trials; ++trial) {
    Mask flags(row_channels, 0);
    for (std::size_t b = 0; b < rows; ++b) for (std::size_t c = 0; c < channels; ++c) {
      std::size_t count = 0; bool requested = false;
      for (std::size_t p = 0; p < trials; ++p) {
        bool any = false; for (std::size_t h = p * patch; h < (p + 1) * patch; ++h) for (std::size_t f = 0; f < features; ++f) any = any || observed[index(b, c, h, f)];
        count += any; if (p == trial) requested = any;
      }
      flags[b * channels + c] = requested && count >= 3;
    }
    out.eligible.insert(out.eligible.end(), flags.begin(), flags.end());
    for (std::size_t b = 0; b < rows; ++b) for (std::size_t c = 0; c < channels; ++c)
      for (std::size_t h = 0; h < history; ++h) for (std::size_t f = 0; f < features; ++f) {
        const bool original = observed[index(b, c, h, f)], hidden = h / patch == trial && flags[b * channels + c];
        out.requested.push_back(original && h / patch == trial); out.visible.push_back(original && !hidden); out.query.push_back(original && hidden);
      }
  }
  return out;
}
Reductions reductions(std::span<const double> prediction, std::span<const double> target, const Mask& query,
                       std::size_t rows, std::size_t channels, std::size_t history, std::size_t features, std::size_t patch) {
  require(rows > 0 && channels > 0 && features > 0 && patch > 0 && history > 0 && history % patch == 0, "reduction dimensions"); mask(query);
  const auto trials = history / patch, row_channels = size_product({rows, channels});
  require(prediction.size() == size_product({trials, rows, channels, history, features}) && prediction.size() == target.size() && target.size() == query.size(), "query reduction geometry");
  Reductions out; out.counts.assign(row_channels, 0); out.channel_mae.assign(row_channels, 0); out.channel_huber.assign(row_channels, 0);
  for (std::size_t t = 0; t < trials; ++t) for (std::size_t b = 0; b < rows; ++b) for (std::size_t c = 0; c < channels; ++c)
    for (std::size_t h = 0; h < history; ++h) for (std::size_t f = 0; f < features; ++f) {
      const auto i = ((((t * rows + b) * channels + c) * history + h) * features + f), j = b * channels + c;
      finite(prediction[i]); finite(target[i]);
      if (query[i]) { const auto error = std::abs(prediction[i] - target[i]); finite(error); out.channel_mae[j] += error; out.channel_huber[j] += error <= 1 ? .5 * error * error : error - .5; ++out.counts[j]; }
      else require(prediction[i] == 0 && target[i] == 0, "nonzero unqueried payload");
    }
  for (std::size_t i = 0; i < row_channels; ++i) if (out.counts[i]) { out.channel_mae[i] /= double(out.counts[i]); out.channel_huber[i] /= double(out.counts[i]); finite(out.channel_mae[i]); finite(out.channel_huber[i]); }
  std::vector<double> selected_mae, selected_huber;
  for (std::size_t b = 0; b < rows; ++b) {
    std::vector<double> cm, ch;
    for (std::size_t c = 0; c < channels; ++c) if (out.counts[b * channels + c]) { cm.push_back(out.channel_mae[b * channels + c]); ch.push_back(out.channel_huber[b * channels + c]); }
    const auto mae = cm.empty() ? 0 : finite_sum(cm) / double(cm.size()), huber = ch.empty() ? 0 : finite_sum(ch) / double(ch.size());
    out.example_valid.push_back(!cm.empty()); out.example_mae.push_back(mae); out.example_huber.push_back(huber);
    if (!cm.empty()) { selected_mae.push_back(mae); selected_huber.push_back(huber); }
  }
  if (!selected_mae.empty()) { out.mae = finite_sum(selected_mae) / double(selected_mae.size()); out.huber = finite_sum(selected_huber) / double(selected_huber.size()); }
  return out;
}
}  // namespace mityane::independent_cpu::v1
