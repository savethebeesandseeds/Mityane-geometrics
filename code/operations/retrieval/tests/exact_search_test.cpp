#include "mityane/operations/retrieval/exact_search.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace retrieval = mityane::operations::retrieval::v1;

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

retrieval::SpaceBinding space(retrieval::Metric metric, retrieval::Normalization normalization,
                              std::size_t dimensions) {
  return {"space-v1", "snapshot-v1", metric, normalization, dimensions};
}

retrieval::VectorRecord record(const std::string& id, std::vector<double> values) {
  return {id, std::move(values)};
}

retrieval::Query query(const retrieval::SpaceBinding& binding, const std::string& id,
                       std::vector<double> values) {
  retrieval::Query result;
  result.binding = binding;
  result.query_id = "query-" + id;
  result.source_id = id;
  result.values = std::move(values);
  return result;
}

void test_cosine_rank_and_tie() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::cosine, retrieval::Normalization::none, 2);
  retrieval::Corpus corpus{binding, {
      record("m", {1, 1}),
      record("a", {1, 1}),
      record("far", {-1, 0}),
  }};
  const retrieval::SearchResult result =
      retrieval::search(corpus, query(binding, "q", {1, 0}), 2, {});
  check(result.outcome == retrieval::Outcome::ranked, "cosine ranking should succeed");
  check(result.hits.size() == 2, "k limits the emitted neighbors");
  check(result.hits[0].source_id == "a" && result.hits[0].rank == 1, "tie breaks by source id");
  check(result.hits[1].source_id == "m" && result.hits[1].rank == 2, "second tie follows source id");
  check(result.eligible_count == 3, "eligible count keeps members beyond k");
  check(std::abs(result.hits[0].similarity - std::sqrt(0.5)) < 1e-12, "cosine of (1,0) and (1,1)");
}

void test_l2_order_and_pairwise_agreement() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::l2, retrieval::Normalization::none, 2);
  retrieval::Corpus corpus{binding, {record("near", {1, 0}), record("far", {4, 0})}};
  const retrieval::Query probe = query(binding, "q", {0, 0});
  const retrieval::SearchResult result = retrieval::search(corpus, probe, 5, {});
  check(result.outcome == retrieval::Outcome::ranked, "l2 ranking should succeed");
  check(result.hits.size() == 2, "k larger than the corpus returns every member");
  check(result.hits[0].source_id == "near" && result.hits[0].distance == 1, "nearest l2 member ranks first");
  check(result.hits[1].source_id == "far" && result.hits[1].distance == 4, "farther l2 member ranks second");
  const retrieval::PairwiseResult pair = retrieval::pairwise(binding, record("q", {0, 0}), record("near", {1, 0}));
  check(pair.outcome == retrieval::Outcome::ranked, "pairwise should succeed");
  check(pair.distance == result.hits[0].distance && pair.similarity == result.hits[0].similarity,
        "pairwise matches the corpus search");
}

void test_l2_normalization_collapses_scale() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::cosine, retrieval::Normalization::l2, 2);
  const retrieval::PairwiseResult pair =
      retrieval::pairwise(binding, record("left", {2, 0}), record("right", {4, 0}));
  check(pair.outcome == retrieval::Outcome::ranked, "normalized cosine should succeed");
  check(pair.similarity == 1 && pair.distance == 0, "positive scale copies are the same direction");
}

void test_exclusions_are_reported() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::l2, retrieval::Normalization::none, 1);
  retrieval::Corpus corpus{binding, {
      record("self", {1}),
      record("held", {2}),
      record("kept", {3}),
  }};
  retrieval::ExclusionPolicy policy;
  policy.source_ids = {"held"};
  const retrieval::SearchResult result = retrieval::search(corpus, query(binding, "self", {0}), 3, policy);
  check(result.outcome == retrieval::Outcome::ranked, "a partial exclusion still ranks");
  check(result.hits.size() == 1 && result.hits[0].source_id == "kept", "only the eligible member is ranked");
  check(result.skipped.size() == 2, "both skips are reported");
  check(result.skipped[0].reason == retrieval::SkipReason::self_match, "self-match is reported");
  check(result.skipped[1].reason == retrieval::SkipReason::excluded_source, "named exclusion is reported");
}

void test_empty_is_distinct_from_failure() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::l2, retrieval::Normalization::none, 1);
  retrieval::Corpus corpus{binding, {record("self", {1})}};
  const retrieval::SearchResult result = retrieval::search(corpus, query(binding, "self", {0}), 1, {});
  check(result.outcome == retrieval::Outcome::empty, "excluding the only member is empty");
  check(result.reason == retrieval::Reason::none && result.hits.empty(), "empty carries no failure reason");
  check(result.skipped.size() == 1, "the excluded member is still reported");
}

void test_refusals() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::cosine, retrieval::Normalization::none, 2);
  retrieval::Corpus corpus{binding, {record("a", {1, 0})}};
  retrieval::Query probe = query(binding, "q", {0, 0});
  const retrieval::SearchResult abstained = retrieval::search(corpus, probe, 1, {});
  check(abstained.outcome == retrieval::Outcome::abstained, "zero cosine query abstains");
  check(abstained.reason == retrieval::Reason::undefined_geometry && abstained.hits.empty(),
        "abstention is not a ranking");

  probe.values = {1, 0};
  retrieval::Corpus duplicated{binding, {record("a", {1, 0}), record("a", {0, 1})}};
  check(retrieval::search(duplicated, probe, 1, {}).reason == retrieval::Reason::duplicate_source_id,
        "duplicate source ids fail");

  retrieval::Corpus broken{binding, {record("a", {1, std::numeric_limits<double>::quiet_NaN()})}};
  check(retrieval::search(broken, probe, 1, {}).reason == retrieval::Reason::nonfinite_value,
        "non-finite corpus values fail");

  retrieval::Query mismatched = probe;
  mismatched.binding.snapshot_id = "other-snapshot";
  check(retrieval::search(corpus, mismatched, 1, {}).reason == retrieval::Reason::incompatible_space,
        "snapshot mismatch fails");
  check(retrieval::search(corpus, probe, 0, {}).reason == retrieval::Reason::invalid_request,
        "k of zero fails");

  retrieval::Corpus zero_member{binding, {record("z", {0, 0}), record("u", {1, 0})}};
  const retrieval::SearchResult skipped = retrieval::search(zero_member, probe, 2, {});
  check(skipped.outcome == retrieval::Outcome::ranked, "one undefined member does not fail the query");
  check(skipped.hits.size() == 1 && skipped.hits[0].source_id == "u", "defined member still ranks");
  check(skipped.skipped.size() == 1 &&
            skipped.skipped[0].reason == retrieval::SkipReason::undefined_geometry,
        "zero cosine member is skipped");
}

void test_hubness() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::l2, retrieval::Normalization::none, 1);
  retrieval::Corpus corpus{binding, {record("a", {0}), record("b", {1}), record("c", {2})}};
  const retrieval::DiagnosticReport report = retrieval::neighborhood_diagnostics(corpus, 1);
  check(report.outcome == retrieval::Outcome::ranked, "diagnostics should rank a line of points");
  check(report.rows.size() == 3, "every member receives a row");
  check(report.rows[0].source_id == "b" && report.rows[0].hub_count == 2, "center is both endpoints' neighbor");
  check(report.rows[1].source_id == "a" && report.rows[1].hub_count == 1, "tie at the center resolves to a");
  check(report.rows[2].source_id == "c" && report.rows[2].hub_count == 0, "loser of the tie is not a hub");
  check(report.rows[0].mean_neighbor_distance == 1, "center neighbor distance is one");

  retrieval::Corpus alone{binding, {record("a", {0})}};
  const retrieval::DiagnosticReport single = retrieval::neighborhood_diagnostics(alone, 1);
  check(single.outcome == retrieval::Outcome::empty, "one point has an empty neighborhood");
  check(single.rows.size() == 1 && single.rows[0].neighbor_count == 0, "the lone point is reported");
}

}  // namespace

int main() {
  try {
    test_cosine_rank_and_tie();
    test_l2_order_and_pairwise_agreement();
    test_l2_normalization_collapses_scale();
    test_exclusions_are_reported();
    test_empty_is_distinct_from_failure();
    test_refusals();
    test_hubness();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "exact-search reference: 7 checks passed\n";
  return 0;
}
