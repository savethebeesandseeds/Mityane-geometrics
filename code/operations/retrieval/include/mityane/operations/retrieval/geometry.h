#pragma once

// Deprecated retrieval strategy, retained as internal utilities. These
// constructors build a query vector or a cone from saved vectors. Ranking calls
// exact k-retrieval in exact_search.h. Positive sparse combination lives in
// combination.h. This header does not read artifacts,
// call a provider, or judge relevance.

#include "mityane/operations/retrieval/exact_search.h"

#include <optional>
#include <string>
#include <vector>

namespace mityane::operations::retrieval::v1 {

enum class ConstructorOutcome { constructed, abstained, failed };

struct ConstructorResult {
  ConstructorOutcome outcome = ConstructorOutcome::failed;
  Reason reason = Reason::invalid_request;
  std::vector<double> values;
};

struct Cone {
  SpaceBinding binding;
  std::vector<double> axis;
  double theta = 0;
  std::optional<double> radius_min;
  std::optional<double> radius_max;
};

struct ConeResult {
  ConstructorOutcome outcome = ConstructorOutcome::failed;
  Reason reason = Reason::invalid_request;
  Cone cone;
};

struct ConeReport {
  Outcome outcome = Outcome::failed;
  Reason reason = Reason::invalid_request;
  std::size_t inside_count = 0;
  std::optional<double> purity;
  DiagnosticReport neighborhood;
};

// target - origin after the binding normalization. A zero difference is
// collinear_inputs. At most eight records may be supplied to shared_direction.
ConstructorResult difference(const SpaceBinding& binding, const VectorRecord& origin,
                             const VectorRecord& target);

// Component of the first point that remains after removing the span of
// pairwise differences from that point. Two identical prepared points, or a
// zero residual, are collinear_inputs.
ConstructorResult shared_direction(const SpaceBinding& binding,
                                   const std::vector<VectorRecord>& points);

// Unit axis, angular aperture in radians on [0, pi], and optional radial
// bounds on the prepared L2 norm. theta of pi admits every defined direction.
// theta of 0 admits only an exact cosine of 1. The boundary at any other
// aperture is member cosine >= cos(theta), evaluated in binary64.
ConeResult make_cone(const SpaceBinding& binding, const std::vector<double>& axis, double theta,
                     std::optional<double> radius_min = std::nullopt,
                     std::optional<double> radius_max = std::nullopt);

// Exact k-retrieval against the cone axis. Members outside the aperture or the
// radial bounds are skipped as outside_cone. A zero prepared vector has no
// angle and is skipped as undefined_geometry.
SearchResult search_cone(const Corpus& corpus, const Cone& cone, std::size_t k,
                         const ExclusionPolicy& exclusions);

// Leave-one-out hubness and mean neighbor distance on the members inside the
// cone, plus the fraction of those members named in labeled_source_ids.
// The label list is a diagnostic partition. It is not a relevance rule.
// An empty label list leaves purity unset.
ConeReport cone_diagnostics(const Corpus& corpus, const Cone& cone, std::size_t k,
                            const std::vector<std::string>& labeled_source_ids);

}  // namespace mityane::operations::retrieval::v1
