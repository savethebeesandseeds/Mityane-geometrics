#include "mityane/operations/retrieval/exact_search.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
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

void test_difference_and_shared_direction() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::l2, retrieval::Normalization::none, 2);
  const retrieval::ConstructorResult delta =
      retrieval::difference(binding, record("origin", {1, 0}), record("target", {4, 0}));
  check(delta.outcome == retrieval::ConstructorOutcome::constructed, "difference is constructed");
  check(delta.values.size() == 2 && delta.values[0] == 3 && delta.values[1] == 0,
        "difference is target minus origin");

  const retrieval::ConstructorResult shared = retrieval::shared_direction(
      binding, {record("a", {1, 1}), record("b", {2, 1})});
  check(shared.outcome == retrieval::ConstructorOutcome::constructed, "shared direction is constructed");
  check(shared.values.size() == 2 && shared.values[0] == 0 && shared.values[1] == 1,
        "shared residual is the orthogonal component");

  const retrieval::ConstructorResult line =
      retrieval::shared_direction(binding, {record("a", {1, 0}), record("b", {2, 0})});
  check(line.outcome == retrieval::ConstructorOutcome::failed &&
            line.reason == retrieval::Reason::collinear_inputs,
        "collinear inputs are refused");
  const retrieval::ConstructorResult same =
      retrieval::difference(binding, record("a", {1, 0}), record("b", {1, 0}));
  check(same.outcome == retrieval::ConstructorOutcome::failed &&
            same.reason == retrieval::Reason::collinear_inputs,
        "a zero difference is refused");
  const retrieval::ConstructorResult bad = retrieval::difference(
      binding, record("a", {std::numeric_limits<double>::quiet_NaN(), 0}), record("b", {1, 0}));
  check(bad.outcome == retrieval::ConstructorOutcome::failed &&
            bad.reason == retrieval::Reason::nonfinite_value,
        "a nonfinite constructor input is refused");

  const retrieval::SpaceBinding volume = space(retrieval::Metric::l2, retrieval::Normalization::none, 3);
  const retrieval::ConstructorResult triple = retrieval::shared_direction(volume, {
      record("a", {1, 0, 1}),
      record("b", {2, 0, 1}),
      record("c", {1, 1, 1}),
  });
  check(triple.outcome == retrieval::ConstructorOutcome::constructed, "a three-point residual is constructed");
  check(triple.values.size() == 3 && triple.values[0] == 0 && triple.values[1] == 0 && triple.values[2] == 1,
        "the residual removes both difference axes");

  const retrieval::SpaceBinding normalized = space(retrieval::Metric::cosine, retrieval::Normalization::l2, 2);
  const retrieval::ConstructorResult collapsed =
      retrieval::difference(normalized, record("a", {2, 0}), record("b", {4, 0}));
  check(collapsed.outcome == retrieval::ConstructorOutcome::failed &&
            collapsed.reason == retrieval::Reason::collinear_inputs,
        "l2 normalization collapses positive scale copies");
}

void test_cone_rank_and_diagnostics() {
  const retrieval::SpaceBinding binding = space(retrieval::Metric::l2, retrieval::Normalization::none, 2);
  const retrieval::ConeResult narrow_cone = retrieval::make_cone(binding, {1, 0}, 0);
  check(narrow_cone.outcome == retrieval::ConstructorOutcome::constructed, "a zero aperture is constructed");
  retrieval::Corpus corpus{binding, {
      record("same", {2, 0}),
      record("side", {0, 1}),
      record("back", {-1, 0}),
  }};
  const retrieval::SearchResult narrow = retrieval::search_cone(corpus, narrow_cone.cone, 3, {});
  check(narrow.outcome == retrieval::Outcome::ranked, "a narrow cone ranks the matching direction");
  check(narrow.hits.size() == 1 && narrow.hits[0].source_id == "same" && narrow.hits[0].distance == 1,
        "only the exact direction remains, at exact-search distance");
  check(narrow.skipped.size() == 2 &&
            narrow.skipped[0].reason == retrieval::SkipReason::outside_cone &&
            narrow.skipped[1].reason == retrieval::SkipReason::outside_cone,
        "other directions are reported as outside the cone");

  const retrieval::ConeResult wide_cone = retrieval::make_cone(binding, {1, 0}, std::numbers::pi_v<double>);
  const retrieval::SearchResult wide = retrieval::search_cone(corpus, wide_cone.cone, 3, {});
  check(wide.outcome == retrieval::Outcome::ranked && wide.hits.size() == 3, "a full aperture keeps every point");
  check(wide.hits[0].source_id == "same" && wide.hits[0].distance == 1, "exact search still ranks the nearer point");
  check(wide.hits[2].source_id == "back" && wide.hits[2].distance == 2, "the opposite point is farthest");

  const retrieval::ConeResult limited =
      retrieval::make_cone(binding, {1, 0}, std::numbers::pi_v<double>, 0, 1.5);
  const retrieval::SearchResult radial = retrieval::search_cone(corpus, limited.cone, 3, {});
  check(radial.hits.size() == 2 && radial.hits[0].source_id == "side" && radial.hits[1].source_id == "back",
        "a radial bound removes the longer point and keeps exact-search order");

  retrieval::Corpus with_origin{binding, {record("origin", {0, 0}), record("same", {1, 0})}};
  const retrieval::SearchResult angles = retrieval::search_cone(with_origin, wide_cone.cone, 2, {});
  check(angles.outcome == retrieval::Outcome::ranked && angles.hits.size() == 1 &&
            angles.hits[0].source_id == "same",
        "a zero vector has no cone angle");
  check(angles.skipped.size() == 1 && angles.skipped[0].source_id == "origin" &&
            angles.skipped[0].reason == retrieval::SkipReason::undefined_geometry,
        "the zero vector is an undefined-geometry skip");

  const retrieval::ConeResult negative = retrieval::make_cone(binding, {1, 0}, -0.1);
  check(negative.outcome == retrieval::ConstructorOutcome::failed &&
            negative.reason == retrieval::Reason::invalid_request,
        "a negative aperture is refused");
  const retrieval::ConeResult mismatched = retrieval::make_cone(binding, {1, 0, 0}, 0);
  check(mismatched.outcome == retrieval::ConstructorOutcome::failed &&
            mismatched.reason == retrieval::Reason::dimension_mismatch,
        "a cone axis of the wrong length is refused");

  const retrieval::ConeResult vacant = retrieval::make_cone(binding, {1, 0}, 0, 10, 11);
  const retrieval::SearchResult empty = retrieval::search_cone(corpus, vacant.cone, 2, {});
  check(empty.outcome == retrieval::Outcome::empty && empty.hits.empty() && empty.skipped.size() == 3,
        "an empty cone stays empty rather than failed");

  const retrieval::ConeReport report = retrieval::cone_diagnostics(corpus, wide_cone.cone, 1, {"same", "same"});
  check(report.outcome == retrieval::Outcome::ranked && report.inside_count == 3, "the cone diagnostic covers its members");
  check(report.purity.has_value() && *report.purity == 1.0 / 3.0, "purity counts each labeled member once");
  check(report.neighborhood.rows.size() == 3, "hubness stays leave-one-out inside the cone");
  const retrieval::ConeReport unlabeled = retrieval::cone_diagnostics(corpus, wide_cone.cone, 1, {});
  check(!unlabeled.purity.has_value(), "an empty label list leaves purity unset");
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
    test_difference_and_shared_direction();
    test_cone_rank_and_diagnostics();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "exact-search reference: 7 checks passed\n";
  std::cout << "geometric constructors: 2 checks passed\n";
  return 0;
}
