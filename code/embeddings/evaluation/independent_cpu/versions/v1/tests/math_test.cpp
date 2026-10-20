#include "mityane/independent_cpu/v1/math.h"

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace M = mityane::independent_cpu::v1;
namespace {
void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
void rejects(const std::function<void()>& action, const char* why) {
  try { action(); } catch (const M::ArithmeticError&) { return; } throw std::runtime_error(why);
}
double sum(std::initializer_list<double> values) { return M::finite_sum(std::vector<double>(values)); }
}
int main() {
  try {
    M::require_numeric_profile();
    check(M::parse_seed("18446744073709551615") == UINT64_MAX && M::parse_seed("9007199254740993") == 9007199254740993ULL, "exact wide decimal seeds");
    rejects([] { M::parse_seed("18446744073709551616"); }, "seed overflow accepted");
    rejects([] { M::parse_seed("-1"); }, "negative seed accepted"); rejects([] { M::parse_seed("1.0"); }, "float seed accepted");
    check(M::mixed(0) == 0xe220a8397b1dcdafULL && M::stream_seed(0, 0) == 0, "SplitMix boundary golden");
    const std::array<std::uint8_t, 5> hello{'h','e','l','l','o'};
    check(M::fnv1a64_hex(hello) == "a430d84680aabd0b" && M::fnv1a64({}) == 14695981039346656037ULL, "FNV byte golden");
    const std::vector<std::string> ids{"a", "\xc3\xa9", ""}; check(M::source_manifest(ids) == "1:a2:\xc3\xa9" "0:", "UTF-8 byte-length source manifest");
    M::MT19937_64 known(5489); check(known.draw() == 14514284786278117030ULL && known.draw() == 4620546740167642908ULL, "MT19937_64 initial golden");
    // Standard-library engine is test-only; production arithmetic does not call it.
    for (auto seed : {0ULL, 1ULL, 5489ULL, 18446744073709551615ULL}) {
      M::MT19937_64 actual(seed); std::mt19937_64 expected(seed);
      for (int i = 0; i < 4000; ++i) check(actual.draw() == expected(), "MT19937_64 twist/temper parity");
    }
    for (std::uint64_t count : std::array<std::uint64_t,6>{1, 2, 3, 64, 0x8000000000000001ULL, UINT64_MAX}) {
      M::MT19937_64 actual(17); std::mt19937_64 expected(17);
      for (int i = 0; i < 1000; ++i) {
        // Direct wide multiplication independently checks the limb product and
        // rejection path, including a range with frequent rejected draws.
        std::uint64_t answer;
        for (;;) { const __uint128_t product = __uint128_t(expected()) * count; if (std::uint64_t(product) >= (std::uint64_t(0) - count) % count) { answer = std::uint64_t(product >> 64); break; } }
        check(M::uniform_index(actual, count) == answer, "uint64 rejection mapping parity");
      }
    }
    M::MT19937_64 engine(0); rejects([&] { M::uniform_index(engine, 0); }, "zero uniform range accepted");
    check(M::f32(1.0 + std::ldexp(1.0, -24)) == 1.0 && M::f32(1.0 + 3 * std::ldexp(1.0, -24)) == 1.0 + std::ldexp(1.0, -22), "float32 ties-to-even");
    check(std::signbit(M::f32(-0.0)), "float32 signed zero");
    rejects([] { M::f32(std::numeric_limits<double>::max()); }, "float32 overflow accepted");
    check(sum({1e16, 1, -1e16}) == 1 && sum({1e16, 1, 1e-16}) == 10000000000000002.0, "compensated cancellation/halfway golden");
    check(sum({0x1p-1074, 0x1p-1074}) == 0x1p-1073 && sum({-0.0}) == 0, "subnormal/zero sum");
    rejects([] { sum({std::numeric_limits<double>::infinity()}); }, "nonfinite sum accepted");
    const auto original_rounding = std::fegetround(); std::fesetround(FE_UPWARD);
    rejects([] { M::require_numeric_profile(); }, "rounding profile accepted"); std::fesetround(original_rounding);
    const M::Rows rows{{2,6}, {4,10}, {100,200}}; const M::Mask valid{1,1,0};
    const auto stats = M::map_statistics(rows, valid);
    check(stats.mean == std::vector<double>({3,8}) && stats.scale == std::vector<double>({1,2}) && stats.fitted_rows == 2, "TRAIN-only population map");
    const auto normalized = M::normalize(rows, valid, stats.mean, stats.scale);
    check(normalized == M::Rows({{-1,-1}, {1,1}, {0,0}}), "saved normalization/abstention zeros");
    check(M::binary_argmax({{1,1}, {0,1}, {-0.0,0.0}}) == std::vector<std::int64_t>({0,1,0}), "exact saved-logit tie policy");
    const auto linear = M::logits({{1e16,1,-1e16}}, {{1,0}, {1,0}, {1,0}}, std::array<double,2>{0,1});
    check(linear == M::Rows({{1,1}}), "linear dot uses independent compensated arithmetic");
    const std::vector<std::int64_t> labels{0,1,0}, predictions{0,0,1}; const auto score = M::score(predictions, labels, valid);
    check(score.valid == 2 && score.correct == 1 && score.accuracy == .5 && score.coverage == 2.0/3, "saved score population");
    const auto abstention = M::score(predictions, labels, {0,0,0}); check(!abstention.accuracy && abstention.coverage == 0, "undefined accuracy retained");
    const auto empty = M::bootstrap_groups({}, 0); check(!empty.estimate && !empty.lower && empty.source_groups == 0, "empty bootstrap support");
    const auto one = M::bootstrap_groups(std::array<M::Group,1>{{{1,2}}}, 0); check(one.estimate == .5 && !one.lower, "one-group interval remains undefined");
    const std::array<M::Group,2> constant{{{1,2}, {2,4}}}; const auto interval = M::bootstrap_groups(constant, UINT64_MAX);
    check(interval.estimate == .5 && interval.lower == .5 && interval.upper == .5 && interval.replicates == 1000, "constant grouped bootstrap");
    const auto view = M::deletion_view(std::array<double,3>{2,0,6}, {1,0,1}, {0,1,1}); check(view.values == std::vector<double>({2,0,0}) && view.mask == M::Mask({1,0,0}), "deletion is mask intersection");
    const auto query = M::query_masks({1,1,1,1}, 1, 1, 4, 1, 1);
    check(query.eligible == M::Mask({1,1,1,1}) && query.query == query.requested, "three-context-patch eligibility");
    const auto unsupported = M::query_masks({1,1,0,0}, 1, 1, 4, 1, 1);
    check(unsupported.eligible == M::Mask({0,0,0,0}), "insufficient visible context abstains");
    std::vector<double> target(query.query.size(), 0), prediction(query.query.size(), 0);
    for (std::size_t i = 0; i < query.query.size(); ++i) if (query.query[i]) prediction[i] = 2;
    const auto reduced = M::reductions(prediction, target, query.query, 1, 1, 4, 1, 1);
    check(reduced.counts == std::vector<std::uint64_t>({4}) && reduced.mae == 2 && reduced.huber == 1.5, "whole-query reduction golden");
    prediction[1] = 1; rejects([&] { M::reductions(prediction, target, query.query, 1, 1, 4, 1, 1); }, "unqueried payload accepted");
    std::cout << "independent CPU v1 arithmetic checks passed\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
