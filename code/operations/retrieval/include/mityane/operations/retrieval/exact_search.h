#pragma once

// Engineering reference for OR01-RET 0.1.0. Caller-supplied arrays only.
// This library does not read artifacts, call a provider, or judge relevance.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace mityane::operations::retrieval::v1 {

enum class Metric { cosine, l2 };
enum class Normalization { none, l2 };
enum class Outcome { ranked, empty, abstained, failed };
enum class Reason {
  none,
  invalid_request,
  duplicate_source_id,
  incompatible_space,
  nonfinite_value,
  undefined_geometry,
  dimension_mismatch,
  collinear_inputs
};
enum class SkipReason { self_match, excluded_source, undefined_geometry, outside_cone };
enum class ConstructorOutcome { constructed, abstained, failed };

struct SpaceBinding {
  std::string space_id;
  std::string snapshot_id;
  Metric metric = Metric::cosine;
  Normalization normalization = Normalization::none;
  std::size_t dimensions = 0;
  bool operator==(const SpaceBinding&) const = default;
};

struct VectorRecord {
  std::string source_id;
  std::vector<double> values;
};

struct Query {
  SpaceBinding binding;
  std::string query_id;
  std::string source_id;
  std::vector<double> values;
};

struct ExclusionPolicy {
  bool exclude_self = true;
  std::vector<std::string> source_ids;
};

struct Hit {
  std::string source_id;
  std::size_t rank = 0;
  double distance = 0;
  double similarity = 0;
};

struct Skipped {
  std::string source_id;
  SkipReason reason = SkipReason::excluded_source;
};

struct SearchResult {
  Outcome outcome = Outcome::failed;
  Reason reason = Reason::invalid_request;
  std::vector<Hit> hits;
  std::vector<Skipped> skipped;
  std::size_t eligible_count = 0;
};

struct PairwiseResult {
  Outcome outcome = Outcome::failed;
  Reason reason = Reason::invalid_request;
  double distance = 0;
  double similarity = 0;
};

struct NeighborDiagnostic {
  std::string source_id;
  Outcome query_outcome = Outcome::failed;
  Reason query_reason = Reason::invalid_request;
  std::size_t hub_count = 0;
  std::optional<double> mean_neighbor_distance;
  std::size_t neighbor_count = 0;
};

struct DiagnosticReport {
  Outcome outcome = Outcome::failed;
  Reason reason = Reason::invalid_request;
  std::size_t k = 0;
  std::vector<NeighborDiagnostic> rows;
};

struct Corpus {
  SpaceBinding binding;
  std::vector<VectorRecord> records;
};

// Rank eligible corpus members. Cosine uses descending cosine. L2 uses
// ascending Euclidean distance. Exact binary64 ties break by ascending source ID.
// Similarity is cosine, or 1/(1+distance) for L2. Distance for cosine is
// 1 - cosine. k must be at least 1. Fewer eligible members returns all of them.
SearchResult search(const Corpus& corpus, const Query& query, std::size_t k,
                    const ExclusionPolicy& exclusions);

// One pair under the same geometry. snapshot_id is not consulted.
PairwiseResult pairwise(const SpaceBinding& binding, const VectorRecord& left,
                        const VectorRecord& right);

// Leave-one-out kNN diagnostics. Hub count is how often a source appears in
// other members' k neighbors. The caller's exclusion list is not applied;
// filter the snapshot first when those exclusions are required.
DiagnosticReport neighborhood_diagnostics(const Corpus& corpus, std::size_t k);

// Geometric constructors. Each one is a pure function of the supplied vectors,
// the SpaceRef normalization, and its parameters. A constructed vector is a
// query for search(). A cone stores a unit axis in that prepared space.
// These functions do not rank relevance and do not admit a corpus.

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

// Exact search against the cone axis. Members outside the aperture or the
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
